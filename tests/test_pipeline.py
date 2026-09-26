import numpy as np
import pandas as pd

from evpurchase.data import encode_target
from evpurchase.features import add_fold_target_encoding, add_global_frequency_features


def test_encode_target() -> None:
    result = encode_target(pd.Series(["No", "Yes", "No"]))
    assert result.tolist() == [0, 1, 0]


def test_global_frequency_features_are_deterministic() -> None:
    frame = pd.DataFrame({"value": [1, 1, 2]})
    transformed = add_global_frequency_features(frame, frame, ["value"])
    assert np.allclose(transformed["value_GlobalFreq"], [2 / 3, 2 / 3, 1 / 3])


def test_fold_target_encoding_uses_training_fold_only() -> None:
    train = pd.DataFrame({"group": ["a", "a", "b"]})
    valid = pd.DataFrame({"group": ["a", "c"]})
    test = pd.DataFrame({"group": ["b", "c"]})
    target = pd.Series([1, 1, 0])
    _, valid_encoded, test_encoded = add_fold_target_encoding(
        train, valid, test, target, ["group"], smoothing=0
    )
    global_mean = 2 / 3
    assert valid_encoded["group_FoldTargetMean"].tolist() == [1.0, global_mean]
    assert test_encoded["group_FoldTargetMean"].tolist() == [0.0, global_mean]


def test_training_target_encoding_leaves_each_row_out() -> None:
    train = pd.DataFrame({"group": ["a", "a", "b"]})
    target = pd.Series([1, 0, 0])
    train_encoded, _, _ = add_fold_target_encoding(
        train, train.iloc[:1], train.iloc[:1], target, ["group"], smoothing=1
    )
    assert np.allclose(
        train_encoded["group_FoldTargetMean"],
        [1 / 6, 2 / 3, 1 / 3],
    )
