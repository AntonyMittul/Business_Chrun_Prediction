"""GenAI retention advisor: SHAP drivers -> plain-English retention action plan.

Uses Gemini (model from config.yaml, key from .env GEMINI_API_KEY). Degrades
gracefully: without a key (or on API error) it produces a rule-based plan from the
same drivers, tagged with ``source`` so the two are never confused. The pipeline,
tests and demo therefore work fully offline.

Run a demo:  python -m src.genai.advisor
"""
from __future__ import annotations

import json
import os

from dotenv import load_dotenv

from src.utils.config import PROJECT_ROOT, load_config

load_dotenv(PROJECT_ROOT / ".env")

PROMPT_TEMPLATE = """You are a customer-retention specialist at a SaaS company.
A churn model flags this customer, with SHAP feature attributions explaining why.

Churn probability: {proba:.0%}
Customer profile: {customer}
Top model drivers (SHAP): {drivers}

Feature glossary: risk_factor_count = how many of 4 risk flags are active
(new customer, >=2 payment failures, CSAT<=2, inactive>30d); usage_minutes =
monthly logins x avg session; tenure in months; csat_score 1-5.

Respond with ONLY a JSON object:
{{
  "risk_level": "high" | "medium" | "low",
  "summary": "<2 sentences: who this customer is and why they are at risk>",
  "drivers_plain_english": ["<driver 1 in business language>", ...],
  "recommended_actions": [
    {{"action": "<concrete retention action>", "rationale": "<why this addresses a driver>"}}
  ]
}}
Ground every action in a driver. Never invent facts not present in the data."""

# Rule-based fallback: driver feature -> (plain english, action, rationale)
_RULES = {
    "payment": ("Repeated payment failures signal billing friction",
                "Trigger the dunning flow: retry schedule + payment-method update email",
                "Payment failures >=2 multiply churn risk ~2.4x"),
    "csat": ("Recent support experience left the customer dissatisfied",
             "Priority callback from senior support + goodwill credit",
             "CSAT <= 2 customers churn at 24-26% vs 10% baseline"),
    "new_customer": ("Customer is in the high-risk first 6 months",
                     "Enroll in guided onboarding with a success-milestone check-in",
                     "First-6-month cohort churns at 28% - onboarding is the strongest lever"),
    "tenure": ("Customer is early in their lifecycle",
               "Enroll in guided onboarding with a success-milestone check-in",
               "Churn hazard is concentrated in early tenure"),
    "inactive": ("Customer has gone quiet for 30+ days",
                 "Send a re-engagement campaign highlighting unused features",
                 "Inactivity past 30 days doubles churn risk"),
    "login": ("Product usage is below healthy levels",
              "Offer a feature-adoption session / usage tips nudge",
              "Low monthly logins is a confirmed churn driver (d=-0.33)"),
    "usage": ("Overall product engagement is low",
              "Offer a feature-adoption session / usage tips nudge",
              "Engagement depth is a confirmed churn driver"),
    "risk_factor_count": ("Multiple independent risk flags are active at once",
                          "Escalate to a human retention specialist for a save call",
                          "Compound risk (e.g. low CSAT + payment issues) reaches 37% churn"),
}


def _risk_level(proba: float) -> str:
    return "high" if proba >= 0.30 else "medium" if proba >= 0.15 else "low"


def _fallback(customer: dict, drivers: list[dict], proba: float) -> dict:
    plain, actions, seen = [], [], set()
    for d in drivers:
        if d["shap"] <= 0:  # only churn-pushing drivers warrant action
            continue
        for key, (text, action, why) in _RULES.items():
            if key in d["feature"] and key not in seen:
                seen.add(key)
                plain.append(f"{text} ({d['feature']}={d['value']})")
                actions.append({"action": action, "rationale": why})
    if not actions:
        actions = [{"action": "Add to the standard retention-watch list",
                    "rationale": "No single dominant driver; monitor next cycle"}]
    return {
        "risk_level": _risk_level(proba),
        "summary": f"Customer with {proba:.0%} churn probability; "
                   f"{len(actions)} driver-matched retention action(s) proposed.",
        "drivers_plain_english": plain or ["Risk is spread across weak signals"],
        "recommended_actions": actions,
        "source": "rule-based fallback (no GEMINI_API_KEY or API error)",
    }


def advise(customer: dict, drivers: list[dict], proba: float) -> dict:
    """Retention plan for one customer. Gemini if configured, rules otherwise."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return _fallback(customer, drivers, proba)

    cfg = load_config()
    prompt = PROMPT_TEMPLATE.format(
        proba=proba, customer=json.dumps(customer), drivers=json.dumps(drivers)
    )
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=cfg["genai"]["model"],
            contents=prompt,
            config={"response_mime_type": "application/json", "temperature": 0.3},
        )
        result = json.loads(resp.text)
        result["source"] = cfg["genai"]["model"]
        return result
    except Exception as err:  # API/parse failure -> still deliver a usable plan
        out = _fallback(customer, drivers, proba)
        out["source"] += f" | gemini error: {type(err).__name__}"
        return out


if __name__ == "__main__":
    import pandas as pd

    from src.models.explain import ChurnExplainer

    ex = ChurnExplainer()
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "churn_features.parquet")
    X = df.drop(columns=["churn", "customer_id"])

    proba_all = ex.pipeline.predict_proba(X)[:, 1]
    idx = int(proba_all.argmax())  # riskiest customer in the book
    row = X.iloc[[idx]]

    profile_cols = ["customer_segment", "tenure_months", "monthly_fee", "csat_score",
                    "payment_failures", "last_login_days_ago", "monthly_logins"]
    customer = {c: (v.item() if hasattr(v := row.iloc[0][c], "item") else str(v))
                for c in profile_cols}

    plan = advise(customer, ex.top_drivers(row), ex.churn_probability(row))
    print(json.dumps(plan, indent=2))
