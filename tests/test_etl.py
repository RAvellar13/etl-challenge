"""Unit tests for the ETL transform logic (etl.py).

These tests use small, hand-built DataFrames instead of the full 50k-row
CSV so the suite runs in well under a second and each case is easy to
reason about in isolation.
"""

import pandas as pd
import pytest

from etl import HIRE_THRESHOLD, transform


def make_raw_df(rows):
    """Build a raw (pre-transform) candidates DataFrame from row dicts,
    filling in any columns the test doesn't care about with defaults."""
    defaults = {
        "First Name": "Jane",
        "Last Name": "Doe",
        "Email": "JANE.DOE@Example.com",
        "Application Date": "2021-05-10",
        "Country": "Brazil",
        "YOE": 3,
        "Seniority": "Mid-Level",
        "Technology": "Data Engineer",
        "Code Challenge Score": 5,
        "Technical Interview Score": 5,
    }
    full_rows = [{**defaults, **row} for row in rows]
    return pd.DataFrame(full_rows)


def test_hired_requires_both_scores_at_or_above_threshold():
    df = make_raw_df([
        {"Code Challenge Score": HIRE_THRESHOLD, "Technical Interview Score": HIRE_THRESHOLD},
        {"Code Challenge Score": HIRE_THRESHOLD - 1, "Technical Interview Score": 10},
        {"Code Challenge Score": 10, "Technical Interview Score": HIRE_THRESHOLD - 1},
        {"Code Challenge Score": 10, "Technical Interview Score": 10},
    ])
    result = transform(df)
    assert result["Is Hired"].tolist() == [1, 0, 0, 1]


def test_application_year_is_derived_from_application_date():
    df = make_raw_df([
        {"Application Date": "2019-01-15"},
        {"Application Date": "2022-07-04"},
    ])
    result = transform(df)
    assert result["Application Year"].tolist() == [2019, 2022]


def test_rows_with_unparseable_dates_are_dropped():
    df = make_raw_df([
        {"Application Date": "2021-05-10"},
        {"Application Date": "not-a-date"},
    ])
    result = transform(df)
    assert len(result) == 1
    assert result.loc[0, "Application Date"] == pd.Timestamp("2021-05-10")


def test_rows_with_invalid_numeric_values_are_dropped():
    df = make_raw_df([
        {"YOE": 5},
        {"Code Challenge Score": "not-a-number"},
    ])
    result = transform(df)
    assert len(result) == 1


def test_scores_are_clipped_to_0_10_range():
    df = make_raw_df([
        {"Code Challenge Score": 15, "Technical Interview Score": -3},
    ])
    result = transform(df)
    assert result.loc[0, "Code Challenge Score"] == 10
    assert result.loc[0, "Technical Interview Score"] == 0


def test_exact_duplicate_rows_are_removed():
    df = make_raw_df([
        {"Email": "same@example.com"},
        {"Email": "same@example.com"},
    ])
    result = transform(df)
    assert len(result) == 1


def test_emails_are_lowercased_and_text_fields_trimmed():
    df = make_raw_df([
        {"Email": "  JANE.DOE@EXAMPLE.COM  ".strip(), "First Name": "  Jane "},
    ])
    result = transform(df)
    assert result.loc[0, "Email"] == "jane.doe@example.com"
    assert result.loc[0, "First Name"] == "Jane"


def test_negative_yoe_is_clipped_to_zero():
    df = make_raw_df([{"YOE": -5}])
    result = transform(df)
    assert result.loc[0, "YOE"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
