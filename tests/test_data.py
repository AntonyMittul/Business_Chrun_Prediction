"""Smoke tests for the raw dataset — guards against a broken/missing file."""
from src.data.load_data import load_raw
from src.utils.config import load_config


def test_raw_loads_with_expected_shape():
    df = load_raw()
    assert df.shape == (10000, 32), f"unexpected shape: {df.shape}"


def test_target_present_and_binary():
    cfg = load_config()
    df = load_raw()
    target = cfg["data"]["target"]
    assert target in df.columns
    assert set(df[target].unique()) <= {0, 1}


def test_id_column_unique():
    cfg = load_config()
    df = load_raw()
    id_col = cfg["data"]["id_column"]
    assert df[id_col].is_unique
