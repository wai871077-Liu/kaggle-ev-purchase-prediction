from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(
    submission_path: Path,
    test_path: Path,
    sample_path: Path,
    expected_sha256: str | None = None,
) -> None:
    submission = pd.read_csv(submission_path)
    test = pd.read_csv(test_path, usecols=["id"])
    sample = pd.read_csv(sample_path)
    assert list(submission.columns) == ["id", "Will_Buy_EV"]
    assert len(submission) == len(test) == len(sample) == 286_571
    assert submission["id"].equals(test["id"])
    assert submission["id"].equals(sample["id"])
    assert submission["id"].is_unique
    assert submission["Will_Buy_EV"].notna().all()
    assert submission["Will_Buy_EV"].between(0, 1).all()
    actual_sha256 = sha256_file(submission_path)
    if expected_sha256 is not None:
        assert actual_sha256 == expected_sha256, (
            f"SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    print(
        f"VALID: {submission_path} | rows={len(submission):,} | "
        f"range=[{submission['Will_Buy_EV'].min():.6f}, "
        f"{submission['Will_Buy_EV'].max():.6f}] | sha256={actual_sha256}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--test", type=Path, default=Path("data/raw/test.csv"))
    parser.add_argument(
        "--sample", type=Path, default=Path("data/raw/sample_submission.csv")
    )
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    validate(args.submission, args.test, args.sample, args.expected_sha256)
