"""Advanced LightGBM with cross-fitted multi-smoothing target encodings.

The recipe is an independently structured implementation of the public Naji V3
approach. Attribution and source links are recorded in SOURCES.rst.
"""

from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

from .config import (
    ID_COLUMN,
    N_SPLITS,
    ORIGINAL_PATH,
    PROCESSED_DIR,
    REPORT_DIR,
    SEED,
    TARGET,
    ensure_output_directories,
)
from .data import encode_target, load_data, validate_data

SMOOTH_KEYS = (
    "income_exact_int",
    "income100_floor",
    "income1000_floor",
    "commute_integer",
)


def build_triple_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    original: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    """Create static features and the string columns encoded inside each fold."""
    train = train.copy()
    test = test.copy()
    original = original.copy()
    target = encode_target(train[TARGET]).to_numpy(dtype="int8")
    train[TARGET] = target
    original[TARGET] = encode_target(original[TARGET])

    train["is_train"] = 1
    test["is_train"] = 0
    test[TARGET] = np.nan
    combined = pd.concat([train, test], ignore_index=True)
    combined = combined.drop(columns=["Number_of_Cars_Owned"])

    categorical = combined.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = [
        column
        for column in combined.columns
        if column not in categorical + [ID_COLUMN, "is_train", TARGET]
    ]

    digit_block: dict[str, np.ndarray] = {}
    for column in numeric:
        values = combined[column].fillna(0).to_numpy(dtype="float64")
        for power in range(-4, 4):
            digit_block[f"{column}_digit_{power:+d}"] = (
                np.floor(values / (10.0**power)).astype("int64") % 10
            ).astype("int8")
    combined = pd.concat(
        [combined, pd.DataFrame(digit_block, index=combined.index)], axis=1
    )
    numeric = numeric + list(digit_block)

    original_mean = float(original[TARGET].mean())
    original_block: dict[str, np.ndarray] = {}
    for column in categorical + numeric:
        if column not in original.columns:
            continue
        means = original.groupby(column, dropna=False, observed=False)[TARGET].mean()
        original_block[f"{column}_OriginalMean"] = (
            combined[column]
            .map(means)
            .fillna(original_mean)
            .astype("float32")
            .to_numpy()
        )

    numeric_as_category = [f"{column}_cat" for column in numeric]
    category_block = {
        f"{column}_cat": combined[column].fillna("NaN").astype(str).to_numpy()
        for column in numeric
    }
    all_categories = categorical + numeric_as_category
    frequency_block: dict[str, np.ndarray] = {}
    for column in all_categories:
        values = (
            combined[column]
            if column in combined.columns
            else pd.Series(category_block[column], index=combined.index)
        )
        frequency = values.value_counts(normalize=True, dropna=False)
        frequency_block[f"{column}_GlobalFreq"] = (
            values.map(frequency).fillna(0.0).astype("float32").to_numpy()
        )

    income = combined["Annual_Income_USD"]
    flags = {
        "Income_At_30000": income.eq(30_000).astype("int8").to_numpy(),
        "Income_Millionaire_Cliff": income.ge(170_537).astype("int8").to_numpy(),
        "Income_Dead_Zone": income.between(38_000, 42_000).astype("int8").to_numpy(),
        "Concern_At_One": combined["Environmental_Concern_Level"]
        .eq(1)
        .astype("int8")
        .to_numpy(),
        "income_exact_int": np.floor(income).astype(str).to_numpy(),
        "income100_floor": np.floor(income / 100).astype(str).to_numpy(),
        "income1000_floor": np.floor(income / 1_000).astype(str).to_numpy(),
        "commute_integer": np.floor(combined["Daily_Commute_km"])
        .astype(str)
        .to_numpy(),
    }
    combined = pd.concat(
        [
            combined,
            pd.DataFrame(
                {**category_block, **original_block, **frequency_block, **flags},
                index=combined.index,
            ),
        ],
        axis=1,
    )
    all_categories += list(SMOOTH_KEYS)

    train_features = combined.loc[combined["is_train"].eq(1)].drop(columns="is_train")
    test_features = combined.loc[combined["is_train"].eq(0)].drop(
        columns=["is_train", TARGET]
    )

    numeric_for_correlation = [
        column
        for column in train_features.columns
        if column not in {ID_COLUMN, TARGET}
        and pd.api.types.is_numeric_dtype(train_features[column])
    ]
    correlation = train_features[numeric_for_correlation].corr().abs()
    upper = correlation.where(
        np.triu(np.ones(correlation.shape), k=1).astype(bool)
    )
    duplicate = [column for column in upper if upper[column].eq(1.0).any()]
    constant = [
        column
        for column in train_features.columns
        if column not in {ID_COLUMN, TARGET}
        and (
            train_features[column].nunique(dropna=False) == 1
            or test_features[column].nunique(dropna=False) == 1
        )
    ]
    to_drop = sorted((set(duplicate) | set(constant)) - {ID_COLUMN, TARGET})
    train_features = train_features.drop(columns=to_drop, errors="ignore")
    test_features = test_features.drop(columns=to_drop, errors="ignore")
    encoding_columns = [
        column
        for column in all_categories
        if column in train_features.columns and column not in to_drop
    ]
    return train_features, test_features, target, encoding_columns


def _replace_categories_with_encodings(
    frame: pd.DataFrame,
    encoding_columns: list[str],
    blocks: list[tuple[str, np.ndarray]],
) -> pd.DataFrame:
    encoded = {
        f"{column}_TargetMean_{tag}": values[:, index]
        for tag, values in blocks
        for index, column in enumerate(encoding_columns)
    }
    return pd.concat(
        [
            frame.drop(columns=encoding_columns),
            pd.DataFrame(encoded, index=frame.index),
        ],
        axis=1,
    )


def main(n_splits: int = N_SPLITS, cv_seed: int = SEED, profile: str = "standard") -> None:
    ensure_output_directories()
    train, test, sample_submission = load_data()
    validate_data(train, test, sample_submission)
    if not ORIGINAL_PATH.exists():
        raise FileNotFoundError(
            "Missing data/raw/original.csv; see data/DATA_SOURCES.rst for the public source"
        )
    original = pd.read_csv(ORIGINAL_PATH)
    expected_original_columns = [
        "Buyer_ID",
        *[column for column in train.columns if column not in {ID_COLUMN, TARGET}],
        TARGET,
    ]
    if set(original.columns) != set(expected_original_columns):
        raise ValueError("Unexpected original-data schema")

    features, test_features, target, encoding_columns = build_triple_frames(
        train, test, original
    )
    model_columns = [
        column for column in test_features.columns if column != ID_COLUMN
    ]
    x = features[model_columns]
    x_test = test_features[model_columns]
    if n_splits == N_SPLITS and cv_seed == SEED:
        experiment_name = "triple_lgbm"
    else:
        experiment_name = f"triple_lgbm_{n_splits}fold"
        if cv_seed != SEED:
            experiment_name += f"_seed{cv_seed}"
    if profile != "standard":
        experiment_name += f"_{profile}"
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
    oof = np.zeros(len(train), dtype="float64")
    test_prediction = np.zeros(len(test), dtype="float64")
    fold_rows: list[dict[str, object]] = []
    importances: list[pd.DataFrame] = []
    started = time.perf_counter()

    for fold, (fit_indices, valid_indices) in enumerate(
        splitter.split(x, target), start=1
    ):
        y_fit = target[fit_indices]
        blocks: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        for smoothing, tag in (("auto", "auto"), (10.0, "10"), (100.0, "100")):
            encoder = TargetEncoder(
                cv=StratifiedKFold(
                    n_splits=10 if profile == "shallow" else N_SPLITS,
                    shuffle=True,
                    random_state=cv_seed,
                ),
                smooth=smoothing,
                target_type="binary",
            )
            blocks.append(
                (
                    tag,
                    encoder.fit_transform(
                        x.iloc[fit_indices][encoding_columns], y_fit
                    ).astype("float32"),
                    encoder.transform(
                        x.iloc[valid_indices][encoding_columns]
                    ).astype("float32"),
                    encoder.transform(x_test[encoding_columns]).astype("float32"),
                )
            )

        x_fit = _replace_categories_with_encodings(
            x.iloc[fit_indices],
            encoding_columns,
            [(tag, fit) for tag, fit, _, _ in blocks],
        )
        x_valid = _replace_categories_with_encodings(
            x.iloc[valid_indices],
            encoding_columns,
            [(tag, valid) for tag, _, valid, _ in blocks],
        )
        x_test_fold = _replace_categories_with_encodings(
            x_test,
            encoding_columns,
            [(tag, transformed) for tag, _, _, transformed in blocks],
        )

        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=20_000,
            learning_rate=0.02,
            max_depth=4 if profile == "shallow" else 5,
            num_leaves=16 if profile == "shallow" else 32,
            min_child_samples=30 if profile == "shallow" else 10,
            subsample=0.812763,
            colsample_bytree=0.5 if profile == "shallow" else 0.30293,
            reg_alpha=0.07094,
            reg_lambda=5.0 if profile == "shallow" else 2.03303,
            max_bin=1024,
            # Preserve the original five-fold experiment exactly; repeated
            # ten-fold runs also vary the learner seed across folds.
            random_state=(SEED if n_splits == N_SPLITS and cv_seed == SEED else cv_seed + fold),
            feature_pre_filter=False,
            n_jobs=8,
            verbosity=-1,
        )
        fold_started = time.perf_counter()
        model.fit(
            x_fit,
            y_fit,
            eval_set=[(x_valid, target[valid_indices])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(500, verbose=False)],
        )
        valid_prediction = model.predict_proba(x_valid)[:, 1]
        oof[valid_indices] = valid_prediction
        test_prediction += model.predict_proba(x_test_fold)[:, 1] / n_splits
        fold_auc = float(roc_auc_score(target[valid_indices], valid_prediction))
        fold_rows.append(
            {
                "experiment": experiment_name,
                "fold": fold,
                "auc": fold_auc,
                "best_iteration": model.best_iteration_,
                "seconds": time.perf_counter() - fold_started,
            }
        )
        importances.append(
            pd.DataFrame(
                {
                    "feature": x_fit.columns,
                    "importance": model.feature_importances_,
                    "fold": fold,
                }
            )
        )
        print(
            f"[{experiment_name}] fold {fold}/{n_splits}: "
            f"AUC={fold_auc:.6f}, best_iteration={model.best_iteration_}",
            flush=True,
        )

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(REPORT_DIR / f"{experiment_name}_folds.csv", index=False)
    (
        pd.concat(importances, ignore_index=True)
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .to_csv(REPORT_DIR / f"{experiment_name}_feature_importance.csv", index=False)
    )
    np.save(PROCESSED_DIR / f"{experiment_name}_oof.npy", oof)
    np.save(PROCESSED_DIR / f"{experiment_name}_test.npy", test_prediction)

    result = {
        "experiment": experiment_name,
        "model_kind": "lgbm",
        "feature_variant": "digits_frequency_original_triple_te",
        "fold_target_encoding": True,
        "oof_auc": float(roc_auc_score(target, oof)),
        "fold_auc_mean": float(folds["auc"].mean()),
        "fold_auc_std": float(folds["auc"].std(ddof=1)),
        "feature_count": int(x_fit.shape[1]),
        "runtime_seconds": time.perf_counter() - started,
        "outer_folds": n_splits,
        "cv_seed": cv_seed,
        "profile": profile,
        "inner_folds": 10 if profile == "shallow" else N_SPLITS,
    }
    metrics_path = REPORT_DIR / "model_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else []
    by_name = {row["experiment"]: row for row in metrics}
    by_name[result["experiment"]] = result
    metrics_path.write_text(
        json.dumps(list(by_name.values()), indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=N_SPLITS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--profile", choices=["standard", "shallow"], default="standard")
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("folds must be at least 2")
    main(args.folds, args.seed, args.profile)
