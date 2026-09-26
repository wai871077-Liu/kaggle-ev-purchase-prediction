from __future__ import annotations

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def make_model(kind: str, seed: int):
    if kind == "lgbm":
        return LGBMClassifier(
            objective="binary",
            n_estimators=2_000,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_alpha=0.05,
            reg_lambda=2.0,
            random_state=seed,
            n_jobs=8,
            verbosity=-1,
        )
    if kind == "xgb":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=2_000,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=8,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_alpha=0.05,
            reg_lambda=2.0,
            tree_method="hist",
            random_state=seed,
            n_jobs=8,
            early_stopping_rounds=100,
        )
    if kind == "catboost":
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=1_500,
            learning_rate=0.04,
            depth=7,
            l2_leaf_reg=5.0,
            random_seed=seed,
            thread_count=8,
            allow_writing_files=False,
            verbose=False,
        )
    raise ValueError(f"Unknown model kind: {kind}")

