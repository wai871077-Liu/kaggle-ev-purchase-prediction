from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .config import (
    CATEGORICAL_COLUMNS,
    EXPECTED_TEST_ROWS,
    EXPECTED_TRAIN_ROWS,
    FEATURE_COLUMNS,
    ID_COLUMN,
    SAMPLE_SUBMISSION_PATH,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
)


@dataclass(frozen=True)
class DataQualitySummary:
    train_rows: int
    test_rows: int
    train_columns: int
    test_columns: int
    positive_rate: float
    train_missing_cells: int
    test_missing_cells: int
    duplicate_train_ids: int
    duplicate_test_ids: int
    sample_ids_match_test: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    return train, test, sample_submission


def encode_target(target: pd.Series) -> pd.Series:
    mapped = target.map({"No": 0, "Yes": 1})
    if mapped.isna().any():
        unknown = sorted(target[mapped.isna()].astype(str).unique())
        raise ValueError(f"Unexpected target labels: {unknown}")
    return mapped.astype("int8")


def validate_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> DataQualitySummary:
    expected_train_columns = [ID_COLUMN, *FEATURE_COLUMNS, TARGET]
    expected_test_columns = [ID_COLUMN, *FEATURE_COLUMNS]
    if list(train.columns) != expected_train_columns:
        raise ValueError(f"Unexpected train schema: {list(train.columns)}")
    if list(test.columns) != expected_test_columns:
        raise ValueError(f"Unexpected test schema: {list(test.columns)}")
    if list(sample_submission.columns) != [ID_COLUMN, TARGET]:
        raise ValueError("Unexpected sample-submission schema")
    if len(train) != EXPECTED_TRAIN_ROWS or len(test) != EXPECTED_TEST_ROWS:
        raise ValueError("Unexpected competition row count")
    if set(train[TARGET].dropna().unique()) != {"Yes", "No"}:
        raise ValueError("Target must contain exactly Yes and No")

    for column in CATEGORICAL_COLUMNS:
        if train[column].dtype.kind not in {"O", "U", "S"}:
            raise ValueError(f"Expected categorical text column: {column}")

    y = encode_target(train[TARGET])
    summary = DataQualitySummary(
        train_rows=len(train),
        test_rows=len(test),
        train_columns=train.shape[1],
        test_columns=test.shape[1],
        positive_rate=float(y.mean()),
        train_missing_cells=int(train.isna().sum().sum()),
        test_missing_cells=int(test.isna().sum().sum()),
        duplicate_train_ids=int(train[ID_COLUMN].duplicated().sum()),
        duplicate_test_ids=int(test[ID_COLUMN].duplicated().sum()),
        sample_ids_match_test=bool(sample_submission[ID_COLUMN].equals(test[ID_COLUMN])),
    )
    if any(
        [
            summary.train_missing_cells,
            summary.test_missing_cells,
            summary.duplicate_train_ids,
            summary.duplicate_test_ids,
        ]
    ):
        raise ValueError(f"Data quality checks failed: {summary}")
    if not summary.sample_ids_match_test:
        raise ValueError("Sample submission IDs do not exactly match test IDs")
    return summary

