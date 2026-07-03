"""API tests — run against the real model artifact (skipped if not trained).

GEMINI_API_KEY is cleared so tests are deterministic, offline, and free.
"""
import pytest

from src.utils.config import MODELS_DIR

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "churn_model.joblib").exists(),
    reason="model not trained yet (run python -m src.models.train)",
)

HIGH_RISK = {
    "age": 40, "tenure_months": 2, "monthly_logins": 1, "weekly_active_days": 0,
    "avg_session_time": 4.0, "features_used": 1, "last_login_days_ago": 45,
    "monthly_fee": 50, "payment_failures": 3, "support_tickets": 4,
    "avg_resolution_time": 40.0, "csat_score": 1.0, "escalations": 2,
    "survey_response": "Unsatisfied", "complaint_type": "Billing",
}
LOW_RISK = {
    "age": 40, "tenure_months": 48, "monthly_logins": 30, "weekly_active_days": 6,
    "avg_session_time": 25.0, "features_used": 10, "last_login_days_ago": 1,
    "monthly_fee": 30, "payment_failures": 0, "csat_score": 5.0,
}


@pytest.fixture(scope="module")
def client(module_mocker=None):
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as c:  # context manager triggers lifespan (model load)
        yield c


@pytest.fixture(autouse=True)
def _offline_genai(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_high_risk(client):
    r = client.post("/predict", json=HIGH_RISK)
    assert r.status_code == 200
    body = r.json()
    assert body["churn_probability"] > body["threshold"]
    assert body["decision"] == "target with retention offer"
    assert len(body["top_drivers"]) == 5
    assert body["retention_plan"]["recommended_actions"]


def test_predict_low_risk_scores_lower(client):
    hi = client.post("/predict", json=HIGH_RISK, params={"include_plan": False}).json()
    lo = client.post("/predict", json=LOW_RISK, params={"include_plan": False}).json()
    assert lo["churn_probability"] < hi["churn_probability"]
    assert "retention_plan" not in lo


def test_validation_rejects_bad_input(client):
    r = client.post("/predict", json={**HIGH_RISK, "csat_score": 9})
    assert r.status_code == 422
