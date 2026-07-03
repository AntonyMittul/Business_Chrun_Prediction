"""Tests for the GenAI advisor's offline (rule-based) path — no API key needed."""
import pytest

from src.genai.advisor import _fallback, _risk_level, advise

DRIVERS = [
    {"feature": "risk_factor_count", "value": 3, "shap": 1.2, "direction": "pushes toward churn"},
    {"feature": "csat_score", "value": 1.0, "shap": 0.4, "direction": "pushes toward churn"},
    {"feature": "payment_failures", "value": 3, "shap": 0.3, "direction": "pushes toward churn"},
    {"feature": "monthly_logins", "value": 25, "shap": -0.2, "direction": "pushes toward staying"},
]
CUSTOMER = {"customer_segment": "SME", "tenure_months": 3, "csat_score": 1.0}


def test_fallback_structure():
    plan = _fallback(CUSTOMER, DRIVERS, 0.42)
    assert plan["risk_level"] == "high"
    assert plan["recommended_actions"], "must always propose at least one action"
    assert all({"action", "rationale"} <= set(a) for a in plan["recommended_actions"])
    assert "rule-based" in plan["source"]


def test_fallback_only_acts_on_churn_pushing_drivers():
    plan = _fallback(CUSTOMER, DRIVERS, 0.42)
    assert not any("adoption session" in a["action"] for a in plan["recommended_actions"]), (
        "monthly_logins pushes toward staying - must not trigger a usage action"
    )


def test_risk_levels():
    assert _risk_level(0.40) == "high"
    assert _risk_level(0.20) == "medium"
    assert _risk_level(0.05) == "low"


def test_advise_without_key_uses_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    plan = advise(CUSTOMER, DRIVERS, 0.42)
    assert "rule-based" in plan["source"]
