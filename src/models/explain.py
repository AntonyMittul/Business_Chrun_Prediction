"""SHAP explanations for the churn model.

SHAP runs on the *uncalibrated* tuned LightGBM pipeline: isotonic calibration is a
monotonic map, so feature attributions and risk *rankings* are identical, while
TreeExplainer stays exact and fast. Calibrated probabilities still come from the
deployed model; SHAP explains *why*, calibration fixes *how much*.

Run:  python -m src.models.explain   (prints global top drivers)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from src.models.train import SEED, load_features, make_model


class ChurnExplainer:
    """Fits the tuned pipeline and exposes global + per-customer SHAP views."""

    def __init__(self) -> None:
        X, y = load_features()
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
        self.pipeline = make_model(X_tr)
        self.pipeline.fit(X_tr, y_tr)

        self.prep = self.pipeline.named_steps["prep"]
        self.model = self.pipeline.named_steps["model"]
        self.feature_names = [n.split("__", 1)[1] for n in self.prep.get_feature_names_out()]
        self.explainer = shap.TreeExplainer(self.model)

    def explanation(self, X: pd.DataFrame) -> shap.Explanation:
        """SHAP Explanation object for a set of customers (churn class)."""
        Xt = self.prep.transform(X)
        Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
        exp = self.explainer(Xt)
        exp.feature_names = self.feature_names
        return exp

    def top_drivers(self, X_row: pd.DataFrame, k: int = 5) -> list[dict]:
        """Top-k churn drivers for ONE customer, as plain dicts (GenAI-ready)."""
        exp = self.explanation(X_row)
        vals = exp.values[0]
        data = exp.data[0]
        order = np.argsort(-np.abs(vals))[:k]
        return [
            {
                "feature": self.feature_names[i],
                "value": round(float(data[i]), 3),
                "shap": round(float(vals[i]), 4),
                "direction": "pushes toward churn" if vals[i] > 0 else "pushes toward staying",
            }
            for i in order
        ]

    def churn_probability(self, X_row: pd.DataFrame) -> float:
        return float(self.pipeline.predict_proba(X_row)[0, 1])


if __name__ == "__main__":
    X, y = load_features()
    ex = ChurnExplainer()
    exp = ex.explanation(X.sample(2000, random_state=SEED))
    global_imp = (
        pd.Series(np.abs(exp.values).mean(axis=0), index=ex.feature_names)
        .sort_values(ascending=False)
        .head(12)
    )
    print("global mean |SHAP| (top 12):")
    print(global_imp.round(4).to_string())
