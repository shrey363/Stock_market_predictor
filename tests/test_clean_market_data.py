"""Market cleaning: timestamps, timezones, duplicates, ordering, missing data.

All fixtures are synthetic and tiny so every expected number is asserted exactly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_project.common.errors import DataSourceError
from stock_project.preprocessing.clean_market_data import MARKET_COLUMNS, clean_market_data


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _row(timestamp: str, close=99.0, volume=900, **overrides) -> dict:
    row = {
        "symbol": "SPY",
        "timestamp": timestamp,
        "open": 98.0,
        "high": 100.0,
        "low": 97.0,
        "close": close,
        "adj_close": close,
        "volume": volume,
        "source": "fixture",
    }
    row.update(overrides)
    return row


def test_fixture_counts_are_exact(market_raw: pd.DataFrame):
    cleaned, quality = clean_market_data(market_raw, symbol="SPY")
    assert quality.rows_read == 6
    assert quality.rows_dropped_unparsable_timestamp == 1
    assert quality.duplicates_removed == 2
    assert quality.rows_written == 3
    assert len(cleaned) == 3


def test_duplicate_key_documented_and_first_occurrence_kept(market_raw: pd.DataFrame):
    cleaned, quality = clean_market_data(market_raw, symbol="SPY")
    assert quality.duplicate_key == "symbol+timestamp"
    assert list(cleaned["timestamp"].dt.date.astype(str)) == ["2020-01-02", "2020-01-03", "2020-01-06"]
    assert any("duplicate" in warning for warning in quality.warnings)


def test_timezone_aware_forms_unify_to_one_session(market_raw: pd.DataFrame):
    """``...T00:00:00Z`` and ``...T00:00:00+00:00`` are the same instant, so they deduplicate."""
    cleaned, _ = clean_market_data(market_raw, symbol="SPY")
    assert int((cleaned["timestamp"].dt.date.astype(str) == "2020-01-06").sum()) == 1


def test_output_is_chronological_and_input_order_is_reported(market_raw: pd.DataFrame):
    cleaned, quality = clean_market_data(market_raw, symbol="SPY")
    assert quality.input_chronological is False
    assert cleaned["timestamp"].is_monotonic_increasing
    assert any("chronological" in warning for warning in quality.warnings)


def test_offset_timestamp_is_converted_to_utc_instant():
    frame = _frame([_row("2020-01-02T15:30:00+02:00")])
    cleaned, quality = clean_market_data(frame, symbol="SPY")
    assert cleaned.loc[0, "timestamp"] == pd.Timestamp("2020-01-02T13:30:00Z")
    assert any("converted to UTC" in note for note in quality.timezone_assumptions)


def test_naive_timestamp_is_read_as_utc_and_recorded():
    frame = _frame([_row("2020-01-02 15:30:00")])
    cleaned, quality = clean_market_data(frame, symbol="SPY")
    assert cleaned.loc[0, "timestamp"] == pd.Timestamp("2020-01-02T15:30:00Z")
    assert any("read as UTC" in note for note in quality.timezone_assumptions)


def test_ambiguous_date_is_rejected_not_guessed():
    frame = _frame([_row("01/02/2020"), _row("2020-01-03T00:00:00Z")])
    cleaned, quality = clean_market_data(frame, symbol="SPY")
    assert quality.rows_dropped_unparsable_timestamp == 1
    assert len(cleaned) == 1
    assert cleaned.loc[0, "timestamp"].date().isoformat() == "2020-01-03"


def test_missing_values_are_reported_and_never_imputed():
    frame = _frame(
        [
            _row("2020-01-02T00:00:00Z"),
            _row("2020-01-03T00:00:00Z", close=None, adj_close=None, volume="abc"),
            _row("2020-01-06T00:00:00Z"),
        ]
    )
    cleaned, quality = clean_market_data(frame, symbol="SPY")
    assert quality.missing_counts["close"] == 1
    assert quality.missing_counts["adj_close"] == 1
    assert quality.missing_counts["volume"] == 1
    assert quality.coerced_to_missing == {"volume": 1}
    assert quality.imputation_applied == []
    assert pd.isna(cleaned.loc[1, "close"])
    assert pd.isna(cleaned.loc[1, "volume"])
    assert quality.rows_written == 3


def test_absent_optional_column_is_flagged_not_computed():
    frame = _frame([_row("2020-01-02T00:00:00Z")]).drop(columns=["adj_close"])
    cleaned, quality = clean_market_data(frame, symbol="SPY")
    assert cleaned["adj_close"].isna().all()
    assert any("adj_close" in warning for warning in quality.warnings)
    assert quality.imputation_applied == []


def test_symbol_mismatch_is_reported():
    frame = _frame([_row("2020-01-02T00:00:00Z"), _row("2020-01-03T00:00:00Z", symbol="QQQ")])
    _, quality = clean_market_data(frame, symbol="SPY")
    assert any("other than the configured" in warning for warning in quality.warnings)


def test_price_sanity_warnings_are_raised():
    frame = _frame(
        [
            _row("2020-01-02T00:00:00Z", close=-1.0, volume=-5),
            _row("2020-01-03T00:00:00Z", high=90.0, low=95.0),
        ]
    )
    _, quality = clean_market_data(frame, symbol="SPY")
    warnings = " | ".join(quality.warnings)
    assert "non-positive close" in warnings
    assert "negative volume" in warnings
    assert "high < low" in warnings


def test_output_columns_match_the_documented_contract(market_raw: pd.DataFrame):
    cleaned, _ = clean_market_data(market_raw, symbol="SPY")
    assert list(cleaned.columns) == MARKET_COLUMNS
    assert cleaned.index.equals(pd.RangeIndex(len(cleaned)))


def test_missing_required_column_raises():
    frame = _frame([_row("2020-01-02T00:00:00Z")]).drop(columns=["volume"])
    with pytest.raises(DataSourceError, match="volume"):
        clean_market_data(frame, symbol="SPY")


def test_empty_table_raises():
    with pytest.raises(DataSourceError, match="no rows"):
        clean_market_data(_frame([_row("2020-01-02T00:00:00Z")]).iloc[0:0], symbol="SPY")


def test_report_serialises_to_json_safe_types(market_raw: pd.DataFrame):
    _, quality = clean_market_data(market_raw, symbol="SPY")
    payload = quality.to_dict()
    assert payload["date_range"] == ["2020-01-02T00:00:00Z", "2020-01-06T00:00:00Z"]
    assert isinstance(payload["missing_counts"], dict)
