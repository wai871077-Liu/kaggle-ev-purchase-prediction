from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS


def add_domain_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[FEATURE_COLUMNS].copy()
    subsidy = result["Subsidy_Available"].eq("Yes").astype("int8")
    home_charge = result["Home_Charging_Possible"].eq("Yes").astype("int8")

    result["Subsidy_Yes"] = subsidy
    result["Home_Charging_Yes"] = home_charge
    result["Income_x_Subsidy"] = result["Annual_Income_USD"] * subsidy
    result["Concern_x_Subsidy"] = result["Environmental_Concern_Level"] * subsidy
    result["Income_x_Concern"] = (
        result["Annual_Income_USD"] * result["Environmental_Concern_Level"]
    )
    result["HomeCharge_x_RangeAnxiety"] = (
        home_charge
        * result["Range_Anxiety_Level"].map({"Low": 1, "Medium": 2, "High": 3}).fillna(0)
    )
    result["Charging_Stations_Total"] = (
        result["Charging_Stations_Near_Home"]
        + result["Charging_Stations_Near_Work"]
    )
    result["Charging_Station_Gap"] = (
        result["Charging_Stations_Near_Home"]
        - result["Charging_Stations_Near_Work"]
    )
    result["Income_Per_Car"] = result["Annual_Income_USD"] / (
        result["Number_of_Cars_Owned"] + 1.0
    )
    result["Income_At_Floor"] = result["Annual_Income_USD"].eq(30_000).astype("int8")
    result["Commute_At_Floor"] = result["Daily_Commute_km"].eq(5.0).astype("int8")
    return result


def add_digit_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in NUMERIC_COLUMNS:
        values = np.abs(result[column].to_numpy(dtype="float64"))
        for power in range(-4, 4):
            digit = np.floor(values / (10.0**power)).astype("int64") % 10
            result[f"{column}_digit_{power:+d}"] = digit.astype("int8")
    return result


def _category_maps(reference: pd.DataFrame) -> dict[str, dict[object, int]]:
    maps: dict[str, dict[object, int]] = {}
    for column in CATEGORICAL_COLUMNS:
        values = sorted(reference[column].astype(str).unique())
        maps[column] = {value: index for index, value in enumerate(values)}
    return maps


def _apply_category_maps(
    frame: pd.DataFrame, maps: dict[str, dict[object, int]]
) -> pd.DataFrame:
    result = frame.copy()
    for column, mapping in maps.items():
        result[column] = result[column].astype(str).map(mapping).fillna(-1).astype("int16")
    return result


def add_global_frequency_features(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    selected = columns or FEATURE_COLUMNS
    denominator = float(len(reference))
    for column in selected:
        frequencies = reference[column].value_counts(dropna=False) / denominator
        result[f"{column}_GlobalFreq"] = (
            frame[column].map(frequencies).fillna(0.0).astype("float32")
        )
    return result


def build_static_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = pd.concat(
        [train[FEATURE_COLUMNS], test[FEATURE_COLUMNS]], ignore_index=True
    )
    train_features = add_domain_features(train)
    test_features = add_domain_features(test)

    if variant in {"enhanced", "digits"}:
        train_features = add_global_frequency_features(
            train_features, reference, FEATURE_COLUMNS
        )
        test_features = add_global_frequency_features(test_features, reference, FEATURE_COLUMNS)
    if variant == "digits":
        train_features = add_digit_features(train_features)
        test_features = add_digit_features(test_features)
    if variant not in {"base", "enhanced", "digits"}:
        raise ValueError(f"Unknown feature variant: {variant}")

    maps = _category_maps(reference)
    train_features = _apply_category_maps(train_features, maps)
    test_features = _apply_category_maps(test_features, maps)
    train_features = train_features.replace([np.inf, -np.inf], np.nan).fillna(-1)
    test_features = test_features.replace([np.inf, -np.inf], np.nan).fillna(-1)
    return train_features, test_features


def add_fold_target_encoding(
    train_fold: pd.DataFrame,
    validation_fold: pd.DataFrame,
    test: pd.DataFrame,
    target: pd.Series,
    columns: list[str],
    smoothing: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_encoded = train_fold.copy()
    validation_encoded = validation_fold.copy()
    test_encoded = test.copy()
    global_mean = float(target.mean())

    for column in columns:
        stats = pd.DataFrame({column: train_fold[column], "target": target.to_numpy()})
        grouped = stats.groupby(column, dropna=False)["target"].agg(["sum", "count"])
        smoothed = (grouped["sum"] + global_mean * smoothing) / (
            grouped["count"] + smoothing
        )
        feature_name = f"{column}_FoldTargetMean"
        group_sum = train_fold[column].map(grouped["sum"]).to_numpy(dtype="float64")
        group_count = train_fold[column].map(grouped["count"]).to_numpy(dtype="float64")
        # Leave each training row's own label out of its encoding. Validation
        # and test values use all labels in the corresponding outer train fold.
        numerator = (
            group_sum
            - target.to_numpy(dtype="float64")
            + global_mean * smoothing
        )
        denominator = group_count - 1.0 + smoothing
        leave_one_out = np.full(len(train_fold), global_mean, dtype="float64")
        np.divide(
            numerator,
            denominator,
            out=leave_one_out,
            where=denominator > 0,
        )
        train_encoded[feature_name] = leave_one_out
        validation_encoded[feature_name] = (
            validation_fold[column].map(smoothed).fillna(global_mean)
        )
        test_encoded[feature_name] = test[column].map(smoothed).fillna(global_mean)

    return train_encoded, validation_encoded, test_encoded
