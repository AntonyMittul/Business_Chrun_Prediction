"""Tests for the feature-engineering pipeline."""
import numpy as np
import pytest

from src.data.clean_data import clean
from src.data.load_data import load_raw
from src.features.build_features import DROP_NOISE, build_features


@pytest.fixture(scope="module")
def features():
    return build_features(clean(load_raw()))


def test_noise_columns_dropped(features):
    assert not set(DROP_NOISE) & set(features.columns)


def test_no_nan_or_inf(features):
    assert not features.isna().any().any()
    assert np.isfinite(features.select_dtypes("number")).all().all()


def test_threshold_flags_match_definitions(features):
    assert (features["is_new_customer"] == (features["tenure_months"] <= 6)).all()
    assert (features["payment_risk"] == (features["payment_failures"] >= 2)).all()
    assert (features["low_csat"] == (features["csat_score"] <= 2)).all()
    assert (features["inactive_30d"] == (features["last_login_days_ago"] > 30)).all()


def test_interaction_and_risk_count(features):
    both = (features["low_csat"] == 1) & (features["payment_risk"] == 1)
    assert (features.loc[both, "csat_x_payment_risk"] == 1).all()
    assert features["risk_factor_count"].between(0, 4).all()


def test_flags_reproduce_known_churn_rates(features):
    """The engineered flags must reproduce the Phase 2/3 findings."""
    assert features.loc[features["is_new_customer"] == 1, "churn"].mean() > 0.25
    assert features.loc[features["payment_risk"] == 1, "churn"].mean() > 0.20
    assert features.loc[features["low_csat"] == 1, "churn"].mean() > 0.20
    assert features.loc[features["risk_factor_count"] == 0, "churn"].mean() < 0.10
