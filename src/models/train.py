"""Train, calibrate, and register the churn model.

Pipeline (decisions grounded in the Phase-5 experiments, notebook 05):
- Plain LightGBM beat class-weighting and SMOTE on PR-AUC (SMOTE was worst:
  0.268 vs 0.288 CV) -> no resampling; imbalance is handled at the *threshold*.
- Hyperparameters from a 40-trial Optuna TPE study (seed 42, CV PR-AUC 0.3135).
- Isotonic calibration so predicted probabilities are trustworthy for the
  cost-based targeting decision.
- Decision threshold chosen on **out-of-fold train predictions** (never on the
  test set) by maximizing expected retention profit from config's business
  assumptions.

Run:  python -m src.models.train
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.utils.config import MODELS_DIR, PROJECT_ROOT, load_config

SEED = 42

# 40-trial Optuna TPE study (seed 42), objective = 5-fold CV PR-AUC. See notebook 05.
TUNED_PARAMS = {
    "n_estimators": 165,
    "learning_rate": 0.01564,
    "num_leaves": 8,
    "min_child_samples": 62,
    "subsample": 0.9585,
    "colsample_bytree": 0.7894,
    "reg_lambda": 0.00293,
}


def load_features() -> tuple[pd.DataFrame, pd.Series]:
    cfg = load_config()
    df = pd.read_parquet(PROJECT_ROOT / cfg["data"]["processed_dir"] / "churn_features.parquet")
    y = df[cfg["data"]["target"]]
    X = df.drop(columns=[cfg["data"]["target"], cfg["data"]["id_column"]])
    return X, y


def make_model(X: pd.DataFrame) -> Pipeline:
    cat_cols = X.select_dtypes(["category", "object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    prep = ColumnTransformer(
        [
            ("ohe", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    lgbm = LGBMClassifier(random_state=SEED, verbose=-1, **TUNED_PARAMS)
    return Pipeline([("prep", prep), ("model", lgbm)])


def profit(y_true: np.ndarray, proba: np.ndarray, threshold: float, cfg: dict) -> float:
    """Expected retention profit of targeting everyone above the threshold."""
    biz = cfg["business"]
    tp_value = biz["save_value"] * biz["save_rate"] - biz["offer_cost"]
    pred = proba >= threshold
    tp = int((pred & (y_true == 1)).sum())
    fp = int((pred & (y_true == 0)).sum())
    return tp_value * tp - biz["offer_cost"] * fp


def pick_threshold(y_true: np.ndarray, proba: np.ndarray, cfg: dict) -> float:
    grid = np.arange(0.05, 0.75, 0.01)
    profits = [profit(y_true, proba, t, cfg) for t in grid]
    return float(grid[int(np.argmax(profits))])


def main() -> None:
    import mlflow  # imported here so serving images don't need the mlflow stack

    cfg = load_config()
    X, y = load_features()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["modeling"]["test_size"], stratify=y, random_state=SEED
    )

    mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="lgbm-tuned-calibrated"):
        mlflow.log_params(TUNED_PARAMS)
        mlflow.log_params({f"biz_{k}": v for k, v in cfg["business"].items()})

        # Calibrated model (isotonic, 5-fold internal)
        cal = CalibratedClassifierCV(make_model(X), method="isotonic", cv=5)

        # Threshold from out-of-fold predictions on TRAIN only (no test peeking)
        oof = cross_val_predict(cal, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
        threshold = pick_threshold(y_tr.to_numpy(), oof, cfg)

        cal.fit(X_tr, y_tr)
        p_te = cal.predict_proba(X_te)[:, 1]

        metrics = {
            "test_roc_auc": roc_auc_score(y_te, p_te),
            "test_pr_auc": average_precision_score(y_te, p_te),
            "test_brier": brier_score_loss(y_te, p_te),
            "threshold": threshold,
            "test_profit_at_threshold": profit(y_te.to_numpy(), p_te, threshold, cfg),
            "test_profit_at_default_0.5": profit(y_te.to_numpy(), p_te, 0.5, cfg),
            "test_recall_at_threshold": recall_score(y_te, p_te >= threshold),
            "test_targeted_share": float((p_te >= threshold).mean()),
        }
        mlflow.log_metrics(metrics)

        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(cal, MODELS_DIR / "churn_model.joblib")
        meta = {"threshold": threshold, "features": list(X.columns), **metrics}
        (MODELS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
        mlflow.log_artifact(str(MODELS_DIR / "model_meta.json"))
        mlflow.sklearn.log_model(
            cal,
            name="model",
            skops_trusted_types=[
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMClassifier",
                "sklearn.calibration._CalibratedClassifier",
            ],
        )

        for k, v in metrics.items():
            print(f"{k:32s} {v:.4f}" if isinstance(v, float) else f"{k:32s} {v}")
        print(f"\nsaved -> {MODELS_DIR / 'churn_model.joblib'}")


if __name__ == "__main__":
    main()
