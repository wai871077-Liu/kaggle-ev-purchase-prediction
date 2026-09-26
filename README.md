# Predicting Electric Vehicle Purchases

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20S6E9-20BEFF.svg)](https://www.kaggle.com/competitions/playground-series-s6e9)

A leakage-aware, reproducible tabular machine-learning project for the Kaggle
Playground Series Season 6, Episode 9. The task is to predict the probability
that a customer will purchase an electric vehicle.

> **Current status:** local pipeline complete and submission validated. Kaggle
> upload and public leaderboard score will be recorded after the web submission
> is accepted.

## Result at a glance

| Model | 5-fold OOF ROC-AUC |
|---|---:|
| CatBoost | 0.94195 |
| LightGBM baseline | 0.94216 |
| LightGBM with frequency features | 0.94322 |
| XGBoost with digit features | 0.94401 |
| XGBoost with leave-one-out target encoding (rejected) | 0.90200 |
| Triple-encoding LightGBM | 0.94606 |
| **Selected rank blend** | **0.94607** |

The final blend combines 80% of the triple-encoding LightGBM prediction ranks
with 20% of the XGBoost ranks. It improves over the best individual model on
four of five held-out folds. The ready-to-submit file is generated locally at
`submissions/submission_best.csv`; only its auditable
[`submission_manifest.csv`](submissions/submission_manifest.csv) is published
while the competition is active.

![Model comparison](reports/figures/model_comparison.png)

## What makes this project rigorous

- A fixed stratified 5-fold split is used for every experiment.
- The advanced target encodings are cross-fitted inside each training fold;
  validation statistics come only from the corresponding outer training fold.
- A simpler leave-one-out encoding was retained as a transparent negative
  experiment because it hurt validation performance.
- Feature selection and blending use out-of-fold predictions, never the Kaggle
  public score.
- Train/test shift is checked with adversarial validation; AUC = **0.50035**,
  which shows no detectable global distribution shift.
- The final CSV is validated for row count, ID order, probability range and
  SHA-256 checksum.
- Raw competition data and generated prediction arrays are intentionally not
  committed.
- The live-competition submission CSV is also excluded to prevent direct reuse;
  the code, metrics, weights and checksum remain fully reviewable.

## Main findings

Environmental concern is the strongest standalone predictor (univariate AUC
**0.8435**), followed by subsidy availability (**0.7175**) and annual income
(**0.6704**). Their interaction is especially informative: observed purchase
rate rises to about **69.3%** for the highest concern level when a subsidy is
available.

![Concern and subsidy interaction](reports/figures/concern_subsidy_heatmap.png)

## Repository layout

```text
.
├── notebooks/                 # Executed, reader-friendly analysis
├── reports/                   # Metrics, figures and final report
├── scripts/                   # Submission validation and notebook builder
├── src/evpurchase/            # Reusable data, feature, model and blend code
├── submissions/               # Final prediction file and audit manifest
├── tests/                     # Fast pipeline tests
├── CITATIONS.md               # Data and library attribution
└── requirements.txt
```

## Reproduce the analysis

Python 3.12 is required. Download the three competition CSV files into
`data/raw/` as described in [`data/README.md`](data/README.md), then run:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m evpurchase.eda
python -m evpurchase.train --models baseline_lgbm enhanced_lgbm xgb_digits xgb_digits_te catboost_diverse
python -m evpurchase.triple
python -m evpurchase.ensemble
python scripts/validate_submission.py submissions/submission_best.csv
jupyter nbconvert --execute --to notebook --inplace notebooks/01_ev_purchase_project.ipynb
```

The random seed, paths and cross-validation settings are centralized in
[`src/evpurchase/config.py`](src/evpurchase/config.py).

## Portfolio notes

This is a competition project, not a causal study. The observed relationships
describe the synthetic competition data and should not be interpreted as real
consumer behavior without external validation. The complete methodology,
limitations and experiment log are in
[`reports/final_report.md`](reports/final_report.md).

## License

Code is released under the [MIT License](LICENSE). Competition data remains
subject to Kaggle's rules and is excluded from this repository.
