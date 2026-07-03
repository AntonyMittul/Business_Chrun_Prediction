"""Pandera schema contract for the cleaned churn dataset.

CLEAN_SCHEMA is the contract every downstream phase relies on. Any code
that claims to produce analysis-ready data must pass `validate_clean`.

Bounds marked "structural" are hard invariants of the measurement itself
(e.g. days per week); the rest are sanity ranges grounded in the observed
data, kept loose enough to survive a data refresh.
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column

# Allowed values for text categoricals (complaint_type gains "None" during cleaning)
CATEGORICAL_VALUES: dict[str, list[str]] = {
    "gender": ["Male", "Female"],
    "country": ["Australia", "Bangladesh", "Canada", "Germany", "India", "UK", "USA"],
    "city": ["Berlin", "Delhi", "Dhaka", "London", "New York", "Sydney", "Toronto"],
    "customer_segment": ["Enterprise", "Individual", "SME"],
    "signup_channel": ["Mobile", "Referral", "Web"],
    "contract_type": ["Monthly", "Quarterly", "Yearly"],
    "payment_method": ["Bank Transfer", "Card", "PayPal"],
    "complaint_type": ["Billing", "Service", "Technical", "None"],
    "survey_response": ["Neutral", "Satisfied", "Unsatisfied"],
}

CLEAN_SCHEMA = pa.DataFrameSchema(
    {
        "customer_id": Column(str, Check.str_matches(r"^CUST_\d{5}$"), unique=True),
        **{
            name: Column("category", Check.isin(values))
            for name, values in CATEGORICAL_VALUES.items()
        },
        "age": Column(int, Check.in_range(18, 100)),
        "tenure_months": Column(int, Check.in_range(1, 600)),
        "monthly_logins": Column(int, Check.ge(0)),
        "weekly_active_days": Column(int, Check.in_range(0, 7)),  # structural
        "avg_session_time": Column(float, Check.gt(0)),
        "features_used": Column(int, Check.ge(0)),
        "usage_growth_rate": Column(float, Check.in_range(-1.0, 1.0)),
        "last_login_days_ago": Column(int, Check.ge(0)),
        "monthly_fee": Column(int, Check.gt(0)),
        "payment_failures": Column(int, Check.ge(0)),
        "discount_applied": Column("int8", Check.isin([0, 1])),
        "price_increase_last_3m": Column("int8", Check.isin([0, 1])),
        "support_tickets": Column(int, Check.ge(0)),
        "avg_resolution_time": Column(float, Check.gt(0)),
        "csat_score": Column(float, Check.in_range(1, 5)),  # structural
        "escalations": Column(int, Check.ge(0)),
        "email_open_rate": Column(float, Check.in_range(0, 1)),  # structural
        "marketing_click_rate": Column(float, Check.in_range(0, 1)),  # structural
        "nps_score": Column(int, Check.in_range(-100, 100)),  # structural
        "referral_count": Column(int, Check.ge(0)),
        "churn": Column(int, Check.isin([0, 1])),
    },
    strict=True,  # no unexpected columns (catches total_revenue sneaking back in)
)


def validate_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate against the clean-data contract; raises with ALL failures listed."""
    return CLEAN_SCHEMA.validate(df, lazy=True)
