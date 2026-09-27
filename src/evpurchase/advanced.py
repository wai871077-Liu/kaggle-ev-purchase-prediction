"""Original feature-cross LightGBM built on the leakage-safe triple pipeline."""

from __future__ import annotations

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
from .data import load_data, validate_data
from .triple import build_triple_frames


def _as_key(values: pd.Series) -> pd.Series:
    return values.fillna("NaN").astype(str)


def _build_extra_block(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create compact, interpretable numeric features and categorical crosses."""
    subsidy = frame["Subsidy_Available"].eq("Yes").astype("int8")
    home_charge = frame["Home_Charging_Possible"].eq("Yes").astype("int8")
    income = frame["Annual_Income_USD"].astype("float64")
    commute = frame["Daily_Commute_km"].astype("float64")
    concern = frame["Environmental_Concern_Level"].astype("float64")
    cars = frame["Number_of_Cars_Owned"].astype("float64")
    stations_home = frame["Charging_Stations_Near_Home"].astype("float64")
    stations_work = frame["Charging_Stations_Near_Work"].astype("float64")

    numeric = pd.DataFrame(
        {
            "Adv_LogIncome": np.log1p(income),
            "Adv_IncomePerCar": income / (cars + 1.0),
            "Adv_IncomePerCommute": income / (commute + 1.0),
            "Adv_StationsTotal": stations_home + stations_work,
            "Adv_StationsGap": stations_home - stations_work,
            "Adv_StationsPerCommute": (stations_home + stations_work) / (commute + 1.0),
            "Adv_ConcernSubsidy": concern * subsidy,
            "Adv_ConcernHomeCharge": concern * home_charge,
            "Adv_IncomeConcern": income * concern,
            "Adv_IncomeSubsidy": income * subsidy,
            "Adv_CommuteHomeCharge": commute * home_charge,
        },
        index=frame.index,
    ).astype("float32")

    key_values: dict[str, pd.Series] = {}
    for width in (10, 50, 100, 250, 500, 1_000, 2_500, 5_000):
        key_values[f"Adv_IncomeBin_{width}"] = _as_key(np.floor(income / width))
    for width in (0.5, 1.0, 2.0, 5.0, 10.0):
        key_values[f"Adv_CommuteBin_{str(width).replace('.', '_')}"] = _as_key(
            np.floor(commute / width)
        )
    for width in (2, 5, 10):
        key_values[f"Adv_AgeBin_{width}"] = _as_key(
            np.floor(frame["Age"].astype("float64") / width)
        )

    concern_key = _as_key(frame["Environmental_Concern_Level"])
    subsidy_key = _as_key(frame["Subsidy_Available"])
    home_key = _as_key(frame["Home_Charging_Possible"])
    city_key = _as_key(frame["City_Type"])
    range_key = _as_key(frame["Range_Anxiety_Level"])
    car_key = _as_key(frame["Current_Car_Type"])
    income_100 = key_values["Adv_IncomeBin_100"]
    income_500 = key_values["Adv_IncomeBin_500"]
    income_1000 = key_values["Adv_IncomeBin_1000"]
    commute_1 = key_values["Adv_CommuteBin_1_0"]
    commute_5 = key_values["Adv_CommuteBin_5_0"]
    age_5 = key_values["Adv_AgeBin_5"]

    key_values.update(
        {
            "Adv_Concern_x_Subsidy": concern_key + "|" + subsidy_key,
            "Adv_Concern_x_Home": concern_key + "|" + home_key,
            "Adv_Concern_x_City": concern_key + "|" + city_key,
            "Adv_Concern_x_Range": concern_key + "|" + range_key,
            "Adv_Subsidy_x_Home": subsidy_key + "|" + home_key,
            "Adv_Subsidy_x_City": subsidy_key + "|" + city_key,
            "Adv_Income100_x_Concern": income_100 + "|" + concern_key,
            "Adv_Income500_x_ConcernSubsidy": (
                income_500 + "|" + concern_key + "|" + subsidy_key
            ),
            "Adv_Income1000_x_City": income_1000 + "|" + city_key,
            "Adv_Commute1_x_Concern": commute_1 + "|" + concern_key,
            "Adv_Commute5_x_Range": commute_5 + "|" + range_key,
            "Adv_Age5_x_Car": age_5 + "|" + car_key,
        }
    )
    categorical = pd.DataFrame(key_values, index=frame.index)
    return pd.concat([numeric, categorical], axis=1), list(categorical.columns)


def build_advanced_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    original: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str], list[str]]:
    features, test_features, target, core_encoding_columns = build_triple_frames(
        train, test, original
    )
    combined_raw = pd.concat(
        [train.drop(columns=TARGET), test], ignore_index=True
    )
    extra, extra_encoding_columns = _build_extra_block(combined_raw)

    for column in extra.columns:
        features[column] = extra.iloc[: len(train)][column].to_numpy()
        test_features[column] = extra.iloc[len(train) :][column].to_numpy()

    for column in extra_encoding_columns:
        frequencies = extra[column].value_counts(normalize=True, dropna=False)
        feature_name = f"{column}_GlobalFreq"
        features[feature_name] = (
            features[column].map(frequencies).fillna(0).astype("float32")
        )
        test_features[feature_name] = (
            test_features[column].map(frequencies).fillna(0).astype("float32")
        )
    return (
        features,
        test_features,
        target,
        core_encoding_columns,
        extra_encoding_columns,
    )


def _replace_advanced_categories(
    frame: pd.DataFrame,
    core_columns: list[str],
    core_blocks: list[tuple[str, np.ndarray]],
    extra_columns: list[str],
    extra_values: np.ndarray,
) -> pd.DataFrame:
    encoded: dict[str, np.ndarray] = {}
    for tag, values in core_blocks:
        for index, column in enumerate(core_columns):
            encoded[f"{column}_TargetMean_{tag}"] = values[:, index]
    for index, column in enumerate(extra_columns):
        encoded[f"{column}_TargetMean_cross_auto"] = extra_values[:, index]
    return pd.concat(
        [
            frame.drop(columns=core_columns + extra_columns),
            pd.DataFrame(encoded, index=frame.index),
        ],
        axis=1,
    )


def main() -> None:
    ensure_output_directories()
    train, test, sample_submission = load_data()
    validate_data(train, test, sample_submission)
    if not ORIGINAL_PATH.exists():
        raise FileNotFoundError("Missing data/raw/original.csv")
    original = pd.read_csv(ORIGINAL_PATH)
    features, test_features, target, core_columns, extra_columns = (
        build_advanced_frames(train, test, original)
    )
    model_columns = [column for column in test_features.columns if column != ID_COLUMN]
    x = features[model_columns]
    x_test = test_features[model_columns]
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype="float64")
    test_prediction = np.zeros(len(test), dtype="float64")
    fold_rows: list[dict[str, object]] = []
    importances: list[pd.DataFrame] = []
    started = time.perf_counter()

    for fold, (fit_indices, valid_indices) in enumerate(
        splitter.split(x, target), start=1
    ):
        y_fit = target[fit_indices]
        core_blocks: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        for smoothing, tag in (("auto", "auto"), (10.0, "10"), (100.0, "100")):
            encoder = TargetEncoder(
                cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED),
                smooth=smoothing,
                target_type="binary",
            )
            core_blocks.append(
                (
                    tag,
                    encoder.fit_transform(x.iloc[fit_indices][core_columns], y_fit).astype("float32"),
                    encoder.transform(x.iloc[valid_indices][core_columns]).astype("float32"),
                    encoder.transform(x_test[core_columns]).astype("float32"),
                )
            )

        extra_encoder = TargetEncoder(
            cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + 17),
            smooth="auto",
            target_type="binary",
        )
        extra_fit = extra_encoder.fit_transform(
            x.iloc[fit_indices][extra_columns], y_fit
        ).astype("float32")
        extra_valid = extra_encoder.transform(
            x.iloc[valid_indices][extra_columns]
        ).astype("float32")
        extra_test = extra_encoder.transform(x_test[extra_columns]).astype("float32")

        x_fit = _replace_advanced_categories(
            x.iloc[fit_indices],
            core_columns,
            [(tag, fit) for tag, fit, _, _ in core_blocks],
            extra_columns,
            extra_fit,
        )
        x_valid = _replace_advanced_categories(
            x.iloc[valid_indices],
            core_columns,
            [(tag, valid) for tag, _, valid, _ in core_blocks],
            extra_columns,
            extra_valid,
        )
        x_test_fold = _replace_advanced_categories(
            x_test,
            core_columns,
            [(tag, transformed) for tag, _, _, transformed in core_blocks],
            extra_columns,
            extra_test,
        )

        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=20_000,
            learning_rate=0.018,
            max_depth=6,
            num_leaves=40,
            min_child_samples=20,
            subsample=0.88,
            colsample_bytree=0.42,
            reg_alpha=0.08,
            reg_lambda=2.2,
            max_bin=1024,
            random_state=SEED + fold,
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
        test_prediction += model.predict_proba(x_test_fold)[:, 1] / N_SPLITS
        fold_auc = float(roc_auc_score(target[valid_indices], valid_prediction))
        fold_rows.append(
            {
                "experiment": "advanced_lgbm",
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
            f"[advanced_lgbm] fold {fold}/{N_SPLITS}: "
            f"AUC={fold_auc:.6f}, best_iteration={model.best_iteration_}",
            flush=True,
        )

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(REPORT_DIR / "advanced_lgbm_folds.csv", index=False)
    (
        pd.concat(importances, ignore_index=True)
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .to_csv(REPORT_DIR / "advanced_lgbm_feature_importance.csv", index=False)
    )
    np.save(PROCESSED_DIR / "advanced_lgbm_oof.npy", oof)
    np.save(PROCESSED_DIR / "advanced_lgbm_test.npy", test_prediction)
    result = {
        "experiment": "advanced_lgbm",
        "model_kind": "lgbm",
        "feature_variant": "triple_plus_multiscale_crosses",
        "fold_target_encoding": True,
        "oof_auc": float(roc_auc_score(target, oof)),
        "fold_auc_mean": float(folds["auc"].mean()),
        "fold_auc_std": float(folds["auc"].std(ddof=1)),
        "feature_count": int(x_fit.shape[1]),
        "runtime_seconds": time.perf_counter() - started,
    }
    metrics_path = REPORT_DIR / "model_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else []
    by_name = {row["experiment"]: row for row in metrics}
    by_name[result["experiment"]] = result
    metrics_path.write_text(json.dumps(list(by_name.values()), indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
