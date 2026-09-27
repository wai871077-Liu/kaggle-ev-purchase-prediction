"""Reconstruct the selected OOF score and summarize subgroup performance.

These are descriptive diagnostics on the model-selection data, not new test
results. Rank blends are ranking scores and must not be read as calibrated
purchase probabilities.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evpurchase.ensemble import _rank_normalize


def main() -> None:
    metrics = json.loads((ROOT / "reports/ensemble_metrics.json").read_text())
    train = pd.read_csv(ROOT / "data/raw/train.csv")
    y = train["Will_Buy_EV"].map({"No": 0, "Yes": 1}).to_numpy()
    prediction = np.zeros(len(train))
    for model, weight in metrics["weights"].items():
        if weight <= 0:
            continue
        values = np.load(ROOT / f"data/processed/{model}_oof.npy")
        if metrics["blend_type"] == "rank":
            values = _rank_normalize(values)
        prediction += weight * values
    auc = float(roc_auc_score(y, prediction))
    # Summation order can move nearly tied ranks at around 1e-10 AUC.
    if not np.isclose(auc, metrics["blend_oof_auc"], atol=1e-9, rtol=0):
        raise ValueError("Saved predictions do not reproduce the selected OOF score")
    groups = train[["City_Type", "Gender", "Subsidy_Available"]].copy()
    groups["Income_quintile"] = pd.qcut(
        train["Annual_Income_USD"], q=5, duplicates="drop"
    ).astype(str)
    rows = []
    for column in groups:
        for group, indices in groups.groupby(column, observed=True).groups.items():
            mask = np.asarray(indices)
            group_y = y[mask]
            rows.append({
                "dimension": column,
                "group": str(group),
                "rows": len(mask),
                "positive_rows": int(group_y.sum()),
                "positive_rate": float(group_y.mean()),
                "oof_auc": float(roc_auc_score(group_y, prediction[mask]))
                if len(np.unique(group_y)) == 2 else None,
            })
    pd.DataFrame(rows).to_csv(ROOT / "reports/subgroup_metrics.csv", index=False)
    result = {
        "reconstructed_oof_auc": auc,
        "rows": len(train),
        "mean_ranking_score": float(prediction.mean()),
        "positive_rate": float(y.mean()),
        "calibrated_probability": False,
        "interpretation": "Descriptive OOF subgroup AUC, not independent validation or a causal/fairness conclusion. Group case mix differs.",
    }
    (ROOT / "reports/prediction_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
