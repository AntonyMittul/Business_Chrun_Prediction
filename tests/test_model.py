"""Smoke tests for the trained model artifact (skipped if not yet trained)."""
import json

import joblib
import pandas as pd
import pytest

from src.utils.config import MODELS_DIR, PROJECT_ROOT

MODEL_PATH = MODELS_DIR / "churn_model.joblib"
META_PATH = MODELS_DIR / "model_meta.json"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="model not trained yet (run python -m src.models.train)"
)


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def meta():
    return json.loads(META_PATH.read_text())


@pytest.fixture(scope="module")
def sample(meta):
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "churn_features.parquet")
    return df[meta["features"]].head(20)


def test_model_predicts_valid_probabilities(model, sample):
    proba = model.predict_proba(sample)[:, 1]
    assert proba.shape == (20,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_meta_is_consistent(meta):
    assert 0.05 <= meta["threshold"] <= 0.75
    assert meta["test_pr_auc"] > 0.25          # sanity floor vs 0.102 baseline
    assert meta["test_roc_auc"] > 0.75
    assert meta["test_profit_at_threshold"] > meta["test_profit_at_default_0.5"]
