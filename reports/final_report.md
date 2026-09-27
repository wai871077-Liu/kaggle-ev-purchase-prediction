# Final Model Report

## Executive summary

This project predicts electric-vehicle purchase probability for Kaggle
Playground Series Season 6, Episode 9. A reproducible five-fold evaluation was
used to compare tree-boosting models and feature sets, followed by repeated
10-fold fits and a shallower 20-fold fit of the strongest pipeline. The
selected repeated-CV rank blend reached **0.946314 OOF ROC-AUC**, above the
strongest single model's **0.946186**, and improved all five audit groups.

The final artifact contains 286,571 predictions and passed structural and range
checks. Its SHA-256 checksum is
`616a1d0dedfb6c72b5dbd8cdda4926960cb9cbce0f6f3f4b1cd91d190545f82b`.

The independently built first version scored **0.94627** on the Kaggle public
leaderboard. The final artifact, selected using local OOF evidence, scored
**0.94641** on 2026-09-27. Its SHA-256 matched the uploaded description. The
same-day Top-10% cutoff was approximately **0.94650**, so the original model
improved but did not reach the target.

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
| `advanced_lgbm` | Triple features + multiscale bins and compact cross keys | 0.945971 | 0.945978 ± 0.000676 |
| `triple_lgbm_10fold` | Strongest pipeline with 90% fit fraction per submodel | 0.946184 | 0.946192 ± 0.000812 |
| `triple_lgbm_10fold_seed137` | Independent 10-fold split of the strongest pipeline | 0.946155 | 0.946164 ± 0.000703 |
| `triple_lgbm_20fold_shallow` | Shallower trees with 20 outer folds and 10 inner folds | 0.946186 | 0.946197 ± 0.001258 |
| **Selected blend** | Rank average across two 10-fold fits, shallow 20-fold fit, 5-fold fit, cross-feature LGBM and XGB | **0.946314** | **0.946317 ± 0.000664** |

The blend search was deliberately small and interpretable: single models,
coarse pairwise weights, and predeclared three- and four-model hypotheses. The final
choice was required to be competitive globally and stable by group. The final
weights are 25% shallow 20-fold, 18.75% seed-137 10-fold, 33.75% seed-42
10-fold, 5.625% 5-fold, 11.25% cross-feature LightGBM and 5.625% XGBoost. The
blend beat the strongest single model in all five audit groups.

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
| SHA-256 | `616a1d0dedfb6c72b5dbd8cdda4926960cb9cbce0f6f3f4b1cd91d190545f82b` |
| Public score | 0.94641; Complete; SHA-matched; verified 2026-09-27 |

The same-day leaderboard export contained 3,123 teams. A score of 0.94641
occupied the 502–525 score band because 24 teams were tied at that rounded
score, corresponding to roughly the top 16–17%. The live page later showed
3,127 teams. This is not a Top-10% result, and the final private ranking can
differ because the public leaderboard uses approximately 20% of the test set.

The machine-readable record is `submissions/submission_manifest.csv`.

## Limitations and next steps

- Competition data appears synthetic, so conclusions may not generalize.
- Repeated 10-fold and shallow 20-fold predictions add diversity, but more
  independent seeds would still be needed to quantify very small AUC changes.
- Features, early stopping and blend weights were selected using validation
  labels. The selected OOF score is therefore not an untouched test estimate.
  The five audit groups reuse the OOF rows and are a consistency diagnostic,
  not independent replication or evidence of statistical significance.
- The public leaderboard is a noisy selection signal; future iterations should
  remain driven by out-of-fold evidence.
- For a stronger research-style extension, probability calibration and grouped
  error analysis could be added without changing the core pipeline.
