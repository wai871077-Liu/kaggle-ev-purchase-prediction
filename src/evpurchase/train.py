from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import (
    N_SPLITS,
    PROCESSED_DIR,
    REPORT_DIR,
    SEED,
    TARGET,
    TARGET_ENCODING_COLUMNS,
    ensure_output_directories,
)
from .data import encode_target, load_data, validate_data
from .features import add_fold_target_encoding, build_static_matrices
from .models import make_model


@dataclass(frozen=True)
class Experiment:
    name: str
    kind: str
    feature_variant: str
    target_encoding: bool


EXPERIMENTS = {
    "baseline_lgbm": Experiment("baseline_lgbm", "lgbm", "base", False),
    "enhanced_lgbm": Experiment("enhanced_lgbm", "lgbm", "enhanced", False),
    "xgb_digits": Experiment("xgb_digits", "xgb", "digits", False),
    "xgb_digits_te": Experiment("xgb_digits_te", "xgb", "digits", True),
    "catboost_diverse": Experiment("catboost_diverse", "catboost", "base", False),
}


def _fit_model(model, kind: str, x_train, y_train, x_valid, y_valid) -> None:
    if kind == "lgbm":
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
    elif kind == "xgb":
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    elif kind == "catboost":
        model.fit(
            x_train,
            y_train,
            eval_set=(x_valid, y_valid),
            early_stopping_rounds=100,
            verbose=False,
        )
    else:
        raise ValueError(kind)


def run_experiment(experiment: Experiment) -> dict[str, object]:
    ensure_output_directories()
    train, test, sample_submission = load_data()
    validate_data(train, test, sample_submission)
    target = encode_target(train[TARGET])
    train_features, test_features = build_static_matrices(
        train, test, experiment.feature_variant
    )

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype="float64")
    test_prediction = np.zeros(len(test), dtype="float64")
    fold_rows: list[dict[str, object]] = []
    importances: list[pd.DataFrame] = []
    start = time.perf_counter()

    for fold, (train_indices, valid_indices) in enumerate(
        splitter.split(train_features, target), start=1
    ):
        x_train = train_features.iloc[train_indices].copy()
        x_valid = train_features.iloc[valid_indices].copy()
        x_test = test_features.copy()
        y_train = target.iloc[train_indices]
        y_valid = target.iloc[valid_indices]

        if experiment.target_encoding:
            x_train, x_valid, x_test = add_fold_target_encoding(
                x_train,
                x_valid,
                x_test,
                y_train,
                TARGET_ENCODING_COLUMNS,
            )

        model = make_model(experiment.kind, SEED + fold)
        fold_start = time.perf_counter()
        _fit_model(model, experiment.kind, x_train, y_train, x_valid, y_valid)
        validation_prediction = model.predict_proba(x_valid)[:, 1]
        oof[valid_indices] = validation_prediction
        test_prediction += model.predict_proba(x_test)[:, 1] / N_SPLITS
        fold_auc = roc_auc_score(y_valid, validation_prediction)
        elapsed = time.perf_counter() - fold_start

        best_iteration = getattr(model, "best_iteration_", None)
        if best_iteration is None:
            best_iteration = getattr(model, "best_iteration", None)
        fold_rows.append(
            {
                "experiment": experiment.name,
                "fold": fold,
                "auc": fold_auc,
                "best_iteration": best_iteration,
                "seconds": elapsed,
            }
        )
        if hasattr(model, "feature_importances_"):
            importances.append(
                pd.DataFrame(
                    {
                        "feature": x_train.columns,
                        "importance": model.feature_importances_,
                        "fold": fold,
                    }
                )
            )
        print(
            f"[{experiment.name}] fold {fold}/{N_SPLITS}: "
            f"AUC={fold_auc:.6f}, seconds={elapsed:.1f}",
            flush=True,
        )

    overall_auc = roc_auc_score(target, oof)
    fold_table = pd.DataFrame(fold_rows)
    fold_table.to_csv(REPORT_DIR / f"{experiment.name}_folds.csv", index=False)
    np.save(PROCESSED_DIR / f"{experiment.name}_oof.npy", oof)
    np.save(PROCESSED_DIR / f"{experiment.name}_test.npy", test_prediction)
    if importances:
        (
            pd.concat(importances, ignore_index=True)
            .groupby("feature", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)
            .to_csv(
                REPORT_DIR / f"{experiment.name}_feature_importance.csv", index=False
            )
        )

    result = {
        "experiment": experiment.name,
        "model_kind": experiment.kind,
        "feature_variant": experiment.feature_variant,
        "fold_target_encoding": experiment.target_encoding,
        "oof_auc": overall_auc,
        "fold_auc_mean": float(fold_table["auc"].mean()),
        "fold_auc_std": float(fold_table["auc"].std(ddof=1)),
        "feature_count": int(train_features.shape[1] + 2 * experiment.target_encoding),
        "runtime_seconds": time.perf_counter() - start,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible CV experiments")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(EXPERIMENTS),
        default=list(EXPERIMENTS),
    )
    args = parser.parse_args()

    metrics_path = REPORT_DIR / "model_metrics.json"
    if metrics_path.exists():
        existing = {row["experiment"]: row for row in json.loads(metrics_path.read_text())}
    else:
        existing = {}
    for name in args.models:
        existing[name] = run_experiment(EXPERIMENTS[name])
        metrics_path.write_text(
            json.dumps(list(existing.values()), indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
