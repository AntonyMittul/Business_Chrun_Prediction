"""Clean the raw churn dataset into an analysis-ready interim table.

Cleaning decisions (Phase 1):
- ``total_revenue`` is dropped — verified to equal ``monthly_fee * tenure_months``
  exactly for every row, so it carries no information and would only inject
  perfect multicollinearity.
- ``complaint_type`` nulls mean "customer never complained" — informative
  missingness, encoded as an explicit ``"None"`` category rather than imputed.
- Yes/No flags -> int8 0/1.
- Text categoricals -> pandas ``category`` dtype.
- Output is validated against the pandera contract before it is written.

Run:  python -m src.data.clean_data
"""
from __future__ import annotations

import pandas as pd

from src.data.load_data import load_raw
from src.data.schema import CATEGORICAL_VALUES, validate_clean
from src.utils.config import PROJECT_ROOT, load_config

YES_NO_COLUMNS = ["discount_applied", "price_increase_last_3m"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all Phase 1 cleaning steps and validate the result."""
    df = df.copy()

    # total_revenue == fee * tenure exactly; the assert makes sure we notice
    # if a future data refresh ever breaks that identity.
    identity = df["monthly_fee"] * df["tenure_months"]
    assert (df["total_revenue"] == identity).all(), "total_revenue != fee * tenure"
    df = df.drop(columns=["total_revenue"])

    # Informative missingness: no complaint filed -> its own category
    df["complaint_type"] = df["complaint_type"].fillna("None")

    for col in YES_NO_COLUMNS:
        df[col] = df[col].map({"No": 0, "Yes": 1}).astype("int8")

    for col, values in CATEGORICAL_VALUES.items():
        df[col] = pd.Categorical(df[col], categories=values)

    return validate_clean(df)


def main() -> None:
    cfg = load_config()
    cleaned = clean(load_raw())
    out = PROJECT_ROOT / cfg["cleaning"]["clean_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(out, index=False)
    print(f"OK: wrote {len(cleaned):,} rows x {cleaned.shape[1]} cols -> {out}")


if __name__ == "__main__":
    main()
