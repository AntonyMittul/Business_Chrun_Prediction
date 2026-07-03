"""Feature engineering — every feature is grounded in a Phase 2/3 finding.

Design principles:
- All transforms are **row-wise** (no fitted state), so computing them before the
  train/test split cannot leak target information.
- Threshold flags encode the effects the statistics notebook proved are
  threshold-shaped rather than linear (e.g. ``last_login_days_ago`` fails a global
  rank test but doubles churn past 30 days).
- Confirmed-noise columns are dropped: ``country``/``city`` (independently random —
  all 49 pairs uniform) and ``gender`` (n.s., and excluding it is fairness-positive).

Run:  python -m src.features.build_features
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import PROJECT_ROOT, load_config

DROP_NOISE = ["country", "city", "gender"]

TENURE_BINS = [0, 6, 12, 24, 36, np.inf]
TENURE_LABELS = ["01-06", "07-12", "13-24", "25-36", "37+"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features and drop proven-noise columns."""
    df = df.drop(columns=DROP_NOISE).copy()

    # --- Threshold flags (Phase 3: effects are step-shaped, not linear) ---
    df["is_new_customer"] = (df["tenure_months"] <= 6).astype("int8")      # 28% churn
    df["payment_risk"] = (df["payment_failures"] >= 2).astype("int8")      # 21-33% churn
    df["low_csat"] = (df["csat_score"] <= 2).astype("int8")                # 24-26% churn
    df["inactive_30d"] = (df["last_login_days_ago"] > 30).astype("int8")   # ~2x churn

    # --- Interaction & compound risk (low CSAT + payment risk -> 37% churn) ---
    df["csat_x_payment_risk"] = (df["low_csat"] & df["payment_risk"]).astype("int8")
    df["risk_factor_count"] = (
        df[["is_new_customer", "payment_risk", "low_csat", "inactive_30d"]]
        .sum(axis=1)
        .astype("int8")
    )

    # --- Lifecycle ---
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"], bins=TENURE_BINS, labels=TENURE_LABELS
    )

    # --- Intensity ratios (denominators are structurally >= 1 in this data) ---
    df["usage_minutes"] = df["monthly_logins"] * df["avg_session_time"]
    df["fee_per_feature"] = df["monthly_fee"] / df["features_used"]
    df["support_burden"] = df["support_tickets"] * df["avg_resolution_time"]
    df["failures_per_year"] = df["payment_failures"] / (df["tenure_months"] / 12)

    return df


def main() -> None:
    cfg = load_config()
    clean_path = PROJECT_ROOT / cfg["cleaning"]["clean_file"]
    out = PROJECT_ROOT / cfg["data"]["processed_dir"] / "churn_features.parquet"

    df = build_features(pd.read_parquet(clean_path))

    assert not df.isna().any().any(), "engineered features contain NaN"
    assert np.isfinite(df.select_dtypes("number")).all().all(), "non-finite values"

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"OK: wrote {len(df):,} rows x {df.shape[1]} cols -> {out}")


if __name__ == "__main__":
    main()
