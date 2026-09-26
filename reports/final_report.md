# Final Model Report

## Executive summary

This project predicts electric-vehicle purchase probability for Kaggle
Playground Series Season 6, Episode 9. A reproducible five-fold evaluation was
used to compare tree-boosting models and feature sets. The selected rank blend
of triple-encoding LightGBM and XGBoost reached **0.946072 OOF ROC-AUC**,
slightly above the best single model's **0.946057**, and improved four of the
five validation folds.

The final artifact contains 286,571 predictions and passed structural and range
checks. Its SHA-256 checksum is
`94a2862a98d7ee83424bbbbcd1d2951fc6a19b8ff0bdb58f1014f096f2ed5e08`.

## Data audit

| Check | Result |
|---|---:|
| Training rows | 668,665 |
| Test rows | 286,571 |
| Predictors | 13 plus ID |
| Positive class rate | 17.46% |
| Missing cells | 0 |
| Duplicate IDs | 0 |
| Sample/test ID order | Exact match |
| Adversarial validation AUC | 0.50035 |

The adversarial validation result is essentially random, so no broad
train-versus-test shift was detected. Raw files were obtained from a public
mirror after the Kaggle data API rejected the active browser session; file
provenance and Git object hashes are recorded in `data/README.md`.

## Signal exploration

The strongest individual features were Environmental Concern Level (AUC
0.8435), Subsidy Available (0.7175) and Annual Income (0.6704). The concern and
subsidy heatmap also revealed a strong interaction: high concern becomes far
more predictive when a subsidy is available. This motivated nonlinear boosted
trees and interaction-friendly engineered features.

## Experiments

| Experiment | Feature strategy | OOF AUC | Fold mean ± SD |
|---|---|---:|---:|
| `baseline_lgbm` | Cleaned raw variables | 0.942159 | 0.942169 ± 0.000829 |
| `catboost_diverse` | Native categorical handling | 0.941949 | 0.941956 ± 0.000853 |
| `enhanced_lgbm` | Global frequency features | 0.943221 | 0.943231 ± 0.000798 |
| `xgb_digits` | Encoded categories and numeric digit features | 0.944006 | 0.944012 ± 0.000705 |
| `xgb_digits_te` | Digit features + leave-one-out target encoding | 0.901997 | 0.916376 ± 0.005240 |
| `triple_lgbm` | Digits + frequencies + source means + triple cross-fitted TE | 0.946057 | 0.946062 ± 0.000690 |
| **Selected blend** | Rank average: 80% triple LGBM, 20% XGB | **0.946072** | **0.946078 ± 0.000692** |

The blend search was deliberately small and interpretable: single models,
coarse pairwise weights, and five predeclared three-model hypotheses. The final
choice was required to be competitive globally and stable by fold. It beat the
best single model in four of five folds.

## Leakage controls

The selected target-encoded variables were computed independently inside every
outer training fold. Training-row encodings were cross-fitted through a nested
five-fold split at three smoothing levels; validation and test rows were mapped
only from the corresponding outer training portion. A simpler leave-one-out
variant performed poorly and was rejected. The Kaggle leaderboard was not used
to select features, models or blend weights.

## Interpretation

Model importance places target-smoothed environmental concern, income,
subsidy, source-data means and value frequencies near the top. These explain
predictive behavior within this synthetic dataset; they do not establish
causal effects or real-world policy conclusions.

## Submission receipt

| Field | Value |
|---|---|
| File | `submissions/submission_best.csv` |
| Rows | 286,571 |
| Target type | Continuous probability in [0, 1] |
| Kaggle upload | Pending web submission |
| Public score | Pending |

The machine-readable receipt is `submissions/submission_manifest.csv`. It is
updated only after Kaggle confirms an upload, keeping local validation separate
from actual leaderboard evidence.

## Limitations and next steps

- Competition data appears synthetic, so conclusions may not generalize.
- One fixed five-fold partition gives a strong comparison but not the full
  uncertainty of repeated cross-validation.
- The public leaderboard is a noisy selection signal; future iterations should
  remain driven by out-of-fold evidence.
- For a stronger research-style extension, probability calibration and grouped
  error analysis could be added without changing the core pipeline.
