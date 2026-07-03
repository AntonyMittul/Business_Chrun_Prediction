"""FastAPI churn-prediction service.

POST /predict takes a raw (clean-schema) customer record and returns:
score -> decision -> SHAP drivers -> retention plan. One call, the full stack.

Run locally:
    uvicorn src.api.main:app --reload
Docs (interactive): http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features.build_features import build_features
from src.genai.advisor import _risk_level, advise
from src.models.explain import ChurnExplainer
from src.utils.config import MODELS_DIR

RESOURCES: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    RESOURCES["model"] = joblib.load(MODELS_DIR / "churn_model.joblib")
    RESOURCES["meta"] = json.loads((MODELS_DIR / "model_meta.json").read_text())
    RESOURCES["explainer"] = ChurnExplainer()  # refit-on-start, ~seconds at this scale
    yield
    RESOURCES.clear()


app = FastAPI(
    title="Churn Prediction API",
    description=(
        "Calibrated churn score + profit-optimal decision + SHAP drivers "
        "+ GenAI retention plan"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class Customer(BaseModel):
    """Raw customer record (clean schema). Geography/gender are optional — the model drops them."""

    customer_segment: str = Field("Individual", examples=["SME"])
    signup_channel: str = Field("Web", examples=["Mobile"])
    contract_type: str = Field("Monthly", examples=["Yearly"])
    payment_method: str = Field("Card", examples=["PayPal"])
    complaint_type: str = Field("None", examples=["Billing"])
    survey_response: str = Field("Neutral", examples=["Unsatisfied"])
    age: int = Field(..., ge=18, le=100, examples=[42])
    tenure_months: int = Field(..., ge=1, examples=[3])
    monthly_logins: int = Field(..., ge=0, examples=[4])
    weekly_active_days: int = Field(..., ge=0, le=7, examples=[1])
    avg_session_time: float = Field(..., gt=0, examples=[8.5])
    features_used: int = Field(..., ge=1, examples=[2])
    usage_growth_rate: float = Field(0.0, ge=-1, le=1, examples=[-0.2])
    last_login_days_ago: int = Field(..., ge=0, examples=[35])
    monthly_fee: int = Field(..., gt=0, examples=[50])
    payment_failures: int = Field(0, ge=0, examples=[2])
    discount_applied: int = Field(0, ge=0, le=1)
    price_increase_last_3m: int = Field(0, ge=0, le=1)
    support_tickets: int = Field(0, ge=0, examples=[3])
    avg_resolution_time: float = Field(20.0, gt=0)
    csat_score: float = Field(3.0, ge=1, le=5, examples=[1.0])
    escalations: int = Field(0, ge=0)
    email_open_rate: float = Field(0.5, ge=0, le=1)
    marketing_click_rate: float = Field(0.25, ge=0, le=1)
    nps_score: int = Field(0, ge=-100, le=100)
    referral_count: int = Field(0, ge=0)
    # dropped by the pipeline, accepted for schema completeness
    country: str = "USA"
    city: str = "New York"
    gender: str = "Male"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in RESOURCES}


@app.post("/predict")
def predict(customer: Customer, include_plan: bool = True) -> dict:
    meta = RESOURCES["meta"]
    raw = pd.DataFrame([customer.model_dump()])
    feats = build_features(raw)[meta["features"]]

    proba = float(RESOURCES["model"].predict_proba(feats)[0, 1])
    threshold = meta["threshold"]
    drivers = RESOURCES["explainer"].top_drivers(feats, k=5)

    response = {
        "churn_probability": round(proba, 4),
        "risk_level": _risk_level(proba),
        "decision": "target with retention offer" if proba >= threshold else "no action",
        "threshold": threshold,
        "top_drivers": drivers,
    }
    if include_plan:
        profile = {k: v for k, v in customer.model_dump().items()
                   if k in ("customer_segment", "tenure_months", "monthly_fee", "csat_score",
                            "payment_failures", "last_login_days_ago", "monthly_logins")}
        response["retention_plan"] = advise(profile, drivers, proba)
    return response
