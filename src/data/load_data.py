"""Load and profile the raw churn dataset.

Run as a script for a quick data profile:

    python -m src.data.load_data
"""
from __future__ import annotations

import pandas as pd

from src.utils.config import PROJECT_ROOT, load_config


def load_raw() -> pd.DataFrame:
    """Load the raw customer churn CSV into a DataFrame."""
    cfg = load_config()
    path = PROJECT_ROOT / cfg["data"]["raw_file"]
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. "
            "Extract customer_churn_business_dataset.csv into data/raw/."
        )
    return pd.read_csv(path)


def profile(df: pd.DataFrame) -> None:
    """Print a quick, human-readable profile of the dataset."""
    cfg = load_config()
    target = cfg["data"]["target"]

    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    print("Missing values per column (non-zero only):")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    print("  none" if missing.empty else missing.to_string())

    print("\nDuplicate rows:", df.duplicated().sum())

    if target in df.columns:
        counts = df[target].value_counts().sort_index()
        rate = df[target].mean()
        print(f"\nTarget '{target}' balance:")
        print(counts.to_string())
        print(f"  churn rate: {rate:.2%}")

    print("\nDtypes:")
    print(df.dtypes.to_string())


if __name__ == "__main__":
    frame = load_raw()
    profile(frame)
