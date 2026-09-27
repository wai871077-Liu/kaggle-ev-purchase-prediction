# Model card

## Intended use

Rank synthetic Kaggle Playground S6E9 test rows by electric-vehicle purchase
likelihood and demonstrate a reproducible tabular ML workflow. This model is
an educational competition artifact, not a deployed consumer-scoring product.

## Inputs and outputs

Inputs include age, income, commute, charging access, environmental concern,
subsidy availability and categorical consumer attributes. The label is
`Will_Buy_EV`. Identifiers are used only to align rows, not as predictors.

The selected ensemble outputs scores in [0, 1]. When rank blending is selected,
these are not calibrated probabilities. A score of 0.8 does not imply an 80%
purchase probability. The observed positive-class prevalence is about 17.46%.

## Training and evaluation

The project compares LightGBM, XGBoost and CatBoost with stratified
cross-validation. Target encodings are cross-fitted inside each outer training
fold; validation-row labels do not enter their own feature encodings. Public
source-data means and label-free frequency statistics are documented in
`CITATIONS.md` and `data/README.md`.

The authoritative selected configuration is `reports/ensemble_metrics.json`.
Individual results are in `reports/model_metrics.json`; exact submitted-file
receipts are in `reports/submission_receipts.json`.

Model selection, early stopping and blend selection reuse validation labels.
Consequently, selected OOF performance may be optimistic. Fixed audit groups
are diagnostics, not independent replications. A contemporaneous leaderboard
cutoff audit is in `reports/leaderboard_audit.json`.

## Limitations

- The data is synthetic; associations are not causal evidence about consumers.
- Digit and frequency features may depend on this specific data generator.
- Frequency statistics use train and test feature values jointly, without
  test labels. This transductive competition setting differs from predicting
  a single future customer in production.
- Subgroup AUCs have different case mixes and class prevalences. They do not
  establish fairness, and small differences are not necessarily significant.
- Public leaderboard scoring uses only part of the test set. Final private
  rankings may differ, and a current Top-10% threshold can change.
- Software versions and seeds are pinned; bit-for-bit equality across operating
  systems or numerical libraries is not guaranteed. The published checksum
  identifies the delivered file, not every possible retraining run.

## Learning and extension

See `中文解题思路.md` for an explanation of feature engineering, leakage control,
negative experiments, rank blending and honest interpretation. Before any real
deployment, obtain external data, perform untouched temporal validation,
calibrate scores, assess subgroup harms and define an appropriate decision rule.
