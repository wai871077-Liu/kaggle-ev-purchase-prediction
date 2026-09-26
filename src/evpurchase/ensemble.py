from __future__ import annotations

import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import (
    FIGURE_DIR,
    N_SPLITS,
    PROCESSED_DIR,
    REPORT_DIR,
    SEED,
    SUBMISSION_DIR,
    TARGET,
    ensure_output_directories,
)
from .data import encode_target, load_data, validate_data

MODEL_NAMES = [
    "triple_lgbm",
    "baseline_lgbm",
    "enhanced_lgbm",
    "xgb_digits",
    "xgb_digits_te",
    "catboost_diverse",
]


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / (len(values) + 1.0)


def _targeted_weights(names: list[str]) -> list[np.ndarray]:
    """Small, auditable blend search instead of a dense OOF weight sweep."""
    count = len(names)
    candidates: list[np.ndarray] = []

    # Every single model and a conservative equal-weight reference.
    for index in range(count):
        weights = np.zeros(count)
        weights[index] = 1.0
        candidates.append(weights)
    candidates.append(np.repeat(1.0 / count, count))

    # Pairwise blends at coarse, interpretable increments.
    for left in range(count):
        for right in range(left + 1, count):
            for left_weight in np.arange(0.2, 0.81, 0.1):
                weights = np.zeros(count)
                weights[left] = left_weight
                weights[right] = 1.0 - left_weight
                candidates.append(weights)

    # A few predeclared three-model hypotheses centered on the two strongest
    # competition-specific models with a small diversity allocation.
    if {"enhanced_lgbm", "xgb_digits", "catboost_diverse"}.issubset(names):
        lookup = {name: index for index, name in enumerate(names)}
        for enhanced_weight, xgb_weight, cat_weight in [
            (0.50, 0.45, 0.05),
            (0.55, 0.40, 0.05),
            (0.60, 0.35, 0.05),
            (0.55, 0.35, 0.10),
            (0.60, 0.30, 0.10),
        ]:
            weights = np.zeros(count)
            weights[lookup["enhanced_lgbm"]] = enhanced_weight
            weights[lookup["xgb_digits"]] = xgb_weight
            weights[lookup["catboost_diverse"]] = cat_weight
            candidates.append(weights)

    # Deduplicate floating-point representations without changing order.
    unique: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for weights in candidates:
        key = tuple(np.round(weights, 8))
        if key not in seen:
            seen.add(key)
            unique.append(weights)
    return unique


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ensure_output_directories()
    train, test, sample_submission = load_data()
    validate_data(train, test, sample_submission)
    target = encode_target(train[TARGET]).to_numpy()

    available = [
        name
        for name in MODEL_NAMES
        if (PROCESSED_DIR / f"{name}_oof.npy").exists()
        and (PROCESSED_DIR / f"{name}_test.npy").exists()
    ]
    if len(available) < 2:
        raise RuntimeError("At least two completed model experiments are required")

    oof = {name: np.load(PROCESSED_DIR / f"{name}_oof.npy") for name in available}
    test_predictions = {
        name: np.load(PROCESSED_DIR / f"{name}_test.npy") for name in available
    }
    oof_frame = pd.DataFrame(oof)
    correlation = oof_frame.corr(method="spearman")
    correlation.to_csv(REPORT_DIR / "prediction_correlation.csv")

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_indices = [valid for _, valid in splitter.split(oof_frame, target)]
    base_scores = {name: roc_auc_score(target, values) for name, values in oof.items()}
    best_single = max(base_scores, key=base_scores.get)

    candidates: list[dict[str, object]] = []
    matrices = {
        "probability": np.column_stack([oof[name] for name in available]),
        "rank": np.column_stack([_rank_normalize(oof[name]) for name in available]),
    }
    test_matrices = {
        "probability": np.column_stack([test_predictions[name] for name in available]),
        "rank": np.column_stack(
            [_rank_normalize(test_predictions[name]) for name in available]
        ),
    }

    weights_to_test = _targeted_weights(available)
    for blend_type, matrix in matrices.items():
        for weights in weights_to_test:
            prediction = matrix @ weights
            score = roc_auc_score(target, prediction)
            candidates.append(
                {
                    "blend_type": blend_type,
                    "weights": weights,
                    "oof_auc": score,
                }
            )

    base_fold_scores = [
        roc_auc_score(target[indices], oof[best_single][indices])
        for indices in fold_indices
    ]
    finalists = sorted(candidates, key=lambda row: row["oof_auc"], reverse=True)[:12]
    for row in finalists:
        prediction = matrices[str(row["blend_type"])] @ np.asarray(row["weights"])
        fold_scores = [
            roc_auc_score(target[indices], prediction[indices]) for indices in fold_indices
        ]
        row["fold_mean"] = float(np.mean(fold_scores))
        row["fold_std"] = float(np.std(fold_scores, ddof=1))
        row["folds_improved"] = int(
            sum(a > b for a, b in zip(fold_scores, base_fold_scores))
        )

    # Prefer a finalist that improves at least three folds. Fall back to the
    # highest global OOF candidate if the diversity hypothesis does not hold.
    stable = [row for row in finalists if row["folds_improved"] >= 3]
    selected = max(stable or finalists, key=lambda row: row["oof_auc"])
    selected_weights = np.asarray(selected["weights"])
    selected_type = str(selected["blend_type"])
    final_test_prediction = test_matrices[selected_type] @ selected_weights

    submission = sample_submission.copy()
    submission[TARGET] = final_test_prediction
    submission_path = SUBMISSION_DIR / "submission_best.csv"
    submission.to_csv(submission_path, index=False)

    if not submission["id"].equals(test["id"]):
        raise ValueError("Submission ID order differs from test data")
    if not submission[TARGET].between(0, 1).all():
        raise ValueError("Submission probabilities are outside [0, 1]")

    metrics = {
        "models": available,
        "base_oof_auc": base_scores,
        "best_single": best_single,
        "blend_type": selected_type,
        "weights": {
            name: float(weight) for name, weight in zip(available, selected_weights)
        },
        "blend_oof_auc": float(selected["oof_auc"]),
        "fold_mean": float(selected["fold_mean"]),
        "fold_std": float(selected["fold_std"]),
        "folds_improved_vs_best_single": int(selected["folds_improved"]),
        "submission_rows": len(submission),
        "submission_sha256": _sha256(submission_path),
    }
    (REPORT_DIR / "ensemble_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    manifest = pd.DataFrame(
        [
            {
                "file": submission_path.name,
                "local_oof_auc": metrics["blend_oof_auc"],
                "blend_type": selected_type,
                "weights": json.dumps(metrics["weights"], sort_keys=True),
                "sha256": metrics["submission_sha256"],
                "kaggle_uploaded": False,
                "public_score": "",
                "submission_id": "",
            }
        ]
    )
    manifest.to_csv(SUBMISSION_DIR / "submission_manifest.csv", index=False)

    sns.set_theme(style="whitegrid")
    score_table = pd.DataFrame(
        [{"model": name, "OOF AUC": score} for name, score in base_scores.items()]
        + [{"model": "selected_blend", "OOF AUC": metrics["blend_oof_auc"]}]
    )
    score_table = score_table[
        score_table["OOF AUC"] >= score_table["OOF AUC"].max() - 0.01
    ].sort_values("OOF AUC")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colors = ["#3267A8"] * (len(score_table) - 1) + ["#D6A13A"]
    bars = ax.barh(score_table["model"], score_table["OOF AUC"], color=colors)
    ax.bar_label(bars, labels=[f"{value:.5f}" for value in score_table["OOF AUC"]])
    ax.set_title("Leakage-aware out-of-fold model comparison")
    ax.set_xlabel("ROC-AUC")
    ax.set_ylabel("")
    ax.set_xlim(score_table["OOF AUC"].min() - 0.0015, score_table["OOF AUC"].max() + 0.0015)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".4f",
        cmap=sns.light_palette("#3267A8", as_cmap=True),
        vmin=max(0.9, float(correlation.min().min()) - 0.01),
        vmax=1.0,
        square=True,
        ax=ax,
    )
    ax.set_title("Spearman correlation of OOF predictions")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "prediction_correlation.png", bbox_inches="tight")
    plt.close(fig)

    importance_name = (
        "triple_lgbm" if (REPORT_DIR / "triple_lgbm_feature_importance.csv").exists()
        else "enhanced_lgbm"
    )
    importance_path = REPORT_DIR / f"{importance_name}_feature_importance.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path).head(15).sort_values("importance")
        fig, ax = plt.subplots(figsize=(8.0, 5.7))
        ax.barh(importance["feature"], importance["importance"], color="#3267A8")
        ax.set_title(f"{importance_name} feature importance (mean across folds)")
        ax.set_xlabel("Split importance")
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "feature_importance.png", bbox_inches="tight")
        plt.close(fig)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
