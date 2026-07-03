"""Tests for the Phase 1 cleaning pipeline and schema contract."""
import pytest

from src.data.clean_data import YES_NO_COLUMNS, clean
from src.data.load_data import load_raw


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def cleaned(raw):
    # clean() validates against CLEAN_SCHEMA internally, so this fixture
    # passing at all already proves the schema contract holds.
    return clean(raw)


def test_no_missing_values(cleaned):
    assert cleaned.isna().sum().sum() == 0


def test_identity_column_dropped(cleaned):
    assert "total_revenue" not in cleaned.columns


def test_row_count_preserved(raw, cleaned):
    assert len(cleaned) == len(raw)


def test_yes_no_flags_binary(cleaned):
    for col in YES_NO_COLUMNS:
        assert set(cleaned[col].unique()) <= {0, 1}, col


def test_no_complaint_becomes_none_category(raw, cleaned):
    n_missing = raw["complaint_type"].isna().sum()
    assert (cleaned["complaint_type"] == "None").sum() == n_missing
