"""Run the documented project workflow with the current Python environment.

Install requirements and this package first. Raw CSVs must be in data/raw.
No downloads, uploads or account mutations are performed by this script.
"""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    for name in ("train.csv", "test.csv", "sample_submission.csv", "original.csv"):
        if not (ROOT / "data/raw" / name).is_file():
            raise FileNotFoundError(
                f"Missing data/raw/{name}; see data/DATA_SOURCES.rst"
            )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    stages = [
        ["-m", "pytest", "-q"],
        ["-m", "evpurchase.eda"],
        ["-m", "evpurchase.train", "--models", "baseline_lgbm", "enhanced_lgbm",
         "xgb_digits", "xgb_digits_te", "catboost_diverse"],
        ["-m", "evpurchase.triple"],
        ["-m", "evpurchase.advanced"],
        ["-m", "evpurchase.triple", "--folds", "10"],
        ["-m", "evpurchase.triple", "--folds", "10", "--seed", "137"],
        ["-m", "evpurchase.triple", "--folds", "20", "--profile", "shallow"],
        ["-m", "evpurchase.ensemble"],
        ["scripts/audit_predictions.py"],
        ["scripts/validate_submission.py", "submissions/submission_best.csv"],
        ["-m", "jupyter", "nbconvert", "--execute", "--to", "notebook", "--inplace",
         "notebooks/01_ev_purchase_project.ipynb"],
        ["-m", "jupyter", "nbconvert", "--to", "html", "--output-dir", "reports",
         "notebooks/01_ev_purchase_project.ipynb"],
    ]
    for stage in stages:
        print("Running:", " ".join(stage), flush=True)
        subprocess.run([sys.executable, *stage], cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
