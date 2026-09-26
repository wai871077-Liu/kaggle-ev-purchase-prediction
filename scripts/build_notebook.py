"""Build the portfolio notebook from small, reviewable cells."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "01_ev_purchase_project.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    markdown(
        """
# Predicting Electric Vehicle Purchases

**Kaggle Playground Series S6E9 · reproducible tabular classification**

This notebook is the reader-friendly companion to a modular training pipeline.
It audits the data, explains the feature strategy, compares leakage-safe
out-of-fold experiments and documents the final submission.
"""
    ),
    markdown(
        """
## TL;DR

- **Best validation:** 0.94607 five-fold OOF ROC-AUC.
- **Final model:** 80/20 rank blend of triple-encoding LightGBM and XGBoost.
- **Stability:** the blend beats the best single model on four of five folds.
- **Quality checks:** no missing values, duplicate IDs or detectable global
  train/test shift.
- **Scientific boundary:** findings describe a synthetic competition dataset;
  they are predictive associations, not causal claims.
"""
    ),
    code(
        """
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
if not (ROOT / "reports").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda value: f"{value:,.6f}")
print(f"Project root detected: {ROOT.name}")
"""
    ),
    markdown(
        """
## 1. Context and evaluation design

The target is the probability that a customer purchases an electric vehicle.
The competition metric is ROC-AUC. All models use the same stratified five-fold
split with seed 42. Target encoding is learned only from the training
portion of each fold, so validation labels never enter their own features.

Model and blend selection use out-of-fold predictions rather than the public
leaderboard. This separates local evidence from Kaggle feedback and reduces the
risk of tuning to the public test subset.
"""
    ),
    markdown("## 2. Data integrity and distribution shift"),
    code(
        """
quality = json.loads((ROOT / "reports/data_quality.json").read_text())
pd.Series(quality, name="value").to_frame()
"""
    ),
    markdown(
        """
Adversarial validation asks whether a model can distinguish training rows from
test rows. An AUC of 0.50035 is effectively random, providing no evidence of a
broad distribution shift. This does not prove that every subgroup is identical,
but it supports using ordinary stratified validation for this project.
"""
    ),
    code(
        """
display(Image(filename=str(ROOT / "reports/figures/target_distribution.png"), width=700))
"""
    ),
    markdown("## 3. Signal exploration"),
    code(
        """
univariate = pd.read_csv(ROOT / "reports/univariate_auc.csv")
univariate.head(10)
"""
    ),
    code(
        """
display(Image(filename=str(ROOT / "reports/figures/univariate_auc.png"), width=760))
display(Image(filename=str(ROOT / "reports/figures/concern_subsidy_heatmap.png"), width=720))
"""
    ),
    markdown(
        """
Environmental concern is the strongest standalone signal. Subsidy availability
and income are also important, while the heatmap shows a clear nonlinear
concern–subsidy interaction. These patterns favor boosted trees over a purely
additive linear model.
"""
    ),
    markdown("## 4. Experiments and fold-safe features"),
    code(
        """
metrics = pd.DataFrame(json.loads((ROOT / "reports/model_metrics.json").read_text()))
metrics[["experiment", "model_kind", "feature_variant", "fold_target_encoding",
         "oof_auc", "fold_auc_mean", "fold_auc_std", "runtime_seconds"]].sort_values(
    "oof_auc", ascending=False
).reset_index(drop=True)
"""
    ),
    markdown(
        """
The enhanced LightGBM adds global frequency features. XGBoost receives
digit-oriented numeric features to capture structure in the generated values.
The strongest LightGBM adds source-data means and cross-fitted target encodings
at three smoothing levels. A simpler leave-one-out target-encoding experiment
is kept as a documented negative result rather than hidden.
"""
    ),
    code(
        """
ensemble = json.loads((ROOT / "reports/ensemble_metrics.json").read_text())
summary = pd.Series({
    "blend_type": ensemble["blend_type"],
    "blend_oof_auc": ensemble["blend_oof_auc"],
    "fold_mean": ensemble["fold_mean"],
    "fold_std": ensemble["fold_std"],
    "folds_improved_vs_best_single": ensemble["folds_improved_vs_best_single"],
    "weights": {k: v for k, v in ensemble["weights"].items() if v > 0},
}, name="selected blend")
summary.to_frame()
"""
    ),
    code(
        """
display(Image(filename=str(ROOT / "reports/figures/model_comparison.png"), width=760))
display(Image(filename=str(ROOT / "reports/figures/prediction_correlation.png"), width=680))
"""
    ),
    markdown(
        """
The final 80/20 rank average combines two strong but non-identical views of the data.
The targeted blend search tested a small set of auditable weights instead of a
dense optimization over the same OOF labels. The chosen blend improved four of
five held-out folds as well as the pooled score.
"""
    ),
    markdown("## 5. Interpretation and submission checks"),
    code(
        """
display(Image(filename=str(ROOT / "reports/figures/feature_importance.png"), width=760))

manifest = pd.read_csv(ROOT / "submissions/submission_manifest.csv")
manifest.T
"""
    ),
    code(
        """
submission = pd.read_csv(ROOT / "submissions/submission_best.csv")
pd.Series({
    "rows": len(submission),
    "unique_ids": submission["id"].nunique(),
    "min_probability": submission["Will_Buy_EV"].min(),
    "max_probability": submission["Will_Buy_EV"].max(),
    "missing_values": int(submission.isna().sum().sum()),
}, name="submission audit").to_frame()
"""
    ),
    markdown(
        """
## 6. Takeaways

1. Domain-like signal in concern, subsidy and income is strong, but nonlinear
   interactions materially improve discrimination.
2. Nested cross-fitted target encodings and source-data statistics give the
   largest improvement; a simpler leave-one-out version was rejected.
3. A simple two-model rank blend is more reliable here than a larger ensemble:
   it produces the best pooled AUC and improves four of five folds.
4. The repository keeps raw data, temporary predictions and actual leaderboard
   evidence distinct. The manifest is updated only after Kaggle confirms the
   upload.

**Limitation:** this is a synthetic competition dataset. Feature importance and
observed purchase rates should not be presented as causal or population-level
facts without external validation.
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
