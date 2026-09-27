from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .config import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    FIGURE_DIR,
    NUMERIC_COLUMNS,
    REPORT_DIR,
    SEED,
    TARGET,
    ensure_output_directories,
)
from .data import encode_target, load_data, validate_data

BLUE = "#3267A8"
GOLD = "#D6A13A"
ORANGE = "#D97732"
CHARCOAL = "#293241"


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#667085",
            "grid.color": "#E5E7EB",
            "font.size": 10,
        }
    )


def _univariate_auc(train: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in NUMERIC_COLUMNS:
        ranks = train[column].rank(method="average", pct=True)
        auc = roc_auc_score(target, ranks)
        rows.append({"feature": column, "univariate_auc": max(auc, 1 - auc)})

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for column in CATEGORICAL_COLUMNS:
        predictions = np.zeros(len(train), dtype="float64")
        for fit_indices, valid_indices in splitter.split(train, target):
            fit = pd.DataFrame(
                {
                    "value": train.iloc[fit_indices][column].astype(str),
                    "target": target.iloc[fit_indices].to_numpy(),
                }
            )
            global_mean = float(fit["target"].mean())
            stats = fit.groupby("value")["target"].agg(["mean", "count"])
            smoothed = (stats["mean"] * stats["count"] + global_mean * 20) / (
                stats["count"] + 20
            )
            predictions[valid_indices] = (
                train.iloc[valid_indices][column]
                .astype(str)
                .map(smoothed)
                .fillna(global_mean)
            )
        auc = roc_auc_score(target, predictions)
        rows.append({"feature": column, "univariate_auc": max(auc, 1 - auc)})
    return pd.DataFrame(rows).sort_values("univariate_auc", ascending=False)


def _adversarial_auc(train: pd.DataFrame, test: pd.DataFrame) -> float:
    sample_size = min(100_000, len(test))
    train_sample = train[FEATURE_COLUMNS].sample(sample_size, random_state=SEED)
    test_sample = test[FEATURE_COLUMNS].sample(sample_size, random_state=SEED)
    combined = pd.concat([train_sample, test_sample], ignore_index=True)
    labels = np.r_[np.zeros(sample_size), np.ones(sample_size)]
    for column in CATEGORICAL_COLUMNS:
        combined[column] = pd.factorize(combined[column].astype(str), sort=True)[0]
    model = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.08, max_leaf_nodes=31, random_state=SEED
    )
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    probabilities = cross_val_predict(
        model, combined, labels, cv=splitter, method="predict_proba", n_jobs=1
    )[:, 1]
    return float(roc_auc_score(labels, probabilities))


def main() -> None:
    ensure_output_directories()
    _style()
    train, test, sample_submission = load_data()
    summary = validate_data(train, test, sample_submission)
    target = encode_target(train[TARGET])
    univariate = _univariate_auc(train, target)
    adversarial_auc = _adversarial_auc(train, test)

    payload = summary.to_dict() | {
        "adversarial_validation_auc": adversarial_auc,
        "source_note": "Kaggle files retrieved from a public mirror; see data/DATA_SOURCES.rst",
    }
    (REPORT_DIR / "data_quality.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    univariate.to_csv(REPORT_DIR / "univariate_auc.csv", index=False)

    counts = train[TARGET].value_counts().reindex(["No", "Yes"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(counts.index, counts.values, color=[BLUE, GOLD], width=0.62)
    ax.set_title("Target distribution in the training set")
    ax.set_ylabel("Rows")
    ax.set_xlabel("Will buy an EV")
    ax.bar_label(
        bars,
        labels=[f"{value:,}\n({value / len(train):.1%})" for value in counts],
        padding=4,
    )
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "target_distribution.png", bbox_inches="tight")
    plt.close(fig)

    plot_data = univariate.sort_values("univariate_auc")
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.barh(plot_data["feature"], plot_data["univariate_auc"], color=BLUE)
    ax.axvline(0.5, color=CHARCOAL, linestyle="--", linewidth=1)
    ax.set_title("Leakage-safe univariate discrimination")
    ax.set_xlabel("ROC-AUC (direction normalized to ≥ 0.5)")
    ax.set_ylabel("")
    ax.set_xlim(0.48, max(0.87, plot_data["univariate_auc"].max() + 0.02))
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "univariate_auc.png", bbox_inches="tight")
    plt.close(fig)

    heatmap = (
        train.assign(target=target)
        .pivot_table(
            index="Environmental_Concern_Level",
            columns="Subsidy_Available",
            values="target",
            aggfunc="mean",
        )
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(7.1, 5.2))
    sns.heatmap(
        heatmap,
        annot=True,
        fmt=".1%",
        cmap=sns.light_palette(BLUE, as_cmap=True),
        vmin=0,
        vmax=max(0.5, float(heatmap.max().max())),
        cbar_kws={"label": "Observed EV purchase rate"},
        ax=ax,
    )
    ax.set_title("Purchase rate by environmental concern and subsidy")
    ax.set_xlabel("Subsidy available")
    ax.set_ylabel("Environmental concern level")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "concern_subsidy_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
