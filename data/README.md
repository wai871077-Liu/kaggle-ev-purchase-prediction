# Data provenance and handling

The canonical source is the Kaggle competition **Playground Series S6E9 —
Predicting Electric Vehicle Purchases**:
https://www.kaggle.com/competitions/playground-series-s6e9

The Kaggle web download endpoint did not accept the active browser session in
the build environment. For this run, the three identically named competition
files were retrieved from the public mirror below and verified against the
mirror's Git object hashes before any modeling:

https://github.com/dbwf923/Predicting-Electric-Vehicle-Purchases

| File | Rows | Columns | Git blob SHA-1 |
|---|---:|---:|---|
| `train.csv` | 668,665 | 15 | `95c0ebe6e0fd2ab351675485c3cbe680ab7d7e14` |
| `test.csv` | 286,571 | 14 | `c8befb9b9b15e47afcf7a2c934db5b71f029e119` |
| `sample_submission.csv` | 286,571 | 2 | `5646f411c5e94a32dd09d98ec362c956ec5f0ed6` |

Additional checks performed locally:

- expected schemas and row counts;
- unique train/test IDs;
- exact equality of test and sample-submission ID order;
- target values restricted to `Yes` and `No`;
- zero missing values in train and test.

The raw files are excluded from Git. To reproduce from the canonical source,
join the competition, accept its rules, and run:

```bash
kaggle competitions download -c playground-series-s6e9 -p data/raw
```

The advanced experiment also uses the 10,000-row public source dataset from
Kaggle, **EV Adoption Behavior and Range Anxiety**:
https://www.kaggle.com/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety

Download its CSV as `data/raw/original.csv`. The file used for this run had
SHA-256 `c271a380df51b18177d0a039d54525ca7c3d71500701dc7496714a091df16fae`.
It is used only for target-mean lookup features; competition validation labels
remain fully isolated by the outer folds.
