"""Market data cleaning and validation.

Rules this step enforces (all of them tested):

* **Strict, documented time parsing.** Timestamps must be ISO-8601. Strings without
  an offset are read as UTC and that assumption is recorded; ambiguous formats such
  as ``01/02/2018`` are rejected as unparsable rather than guessed. Timezone-aware
  input is converted to UTC instants.
* **Chronological order.** Rows are sorted ascending, exactly once, and the report
  states whether the raw input was already in order. Nothing is shuffled.
* **Duplicates reported.** Rows are unique on ``(symbol, timestamp)``; the number
  removed is recorded, never silently dropped.
* **No imputation.** Missing prices stay missing and are counted per column. Gaps
  (weekends, holidays) are not filled, and no value is forward-filled.
"""

from __future__ import annotations

import pandas as pd

from ..common.errors import DataSourceError
from ..common.quality import QualityReport

#: Canonical column order of a cleaned market table.
MARKET_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "adj_close", "volume", "source"]

#: Columns that must exist before cleaning can proceed.
REQUIRED_COLUMNS = ("symbol", "timestamp", "open", "high", "low", "close", "volume")

PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close")

#: Timestamps must match ISO-8601; anything else is reported, not interpreted.
_ISO8601_EXAMPLE = "YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS(+HH:MM|Z)"


def _coerce_timestamp(series: pd.Series, quality: QualityReport) -> pd.Series:
    """Parse timestamps to UTC, recording how the timezone information was obtained."""
    if pd.api.types.is_datetime64_any_dtype(series):
        tz = getattr(series.dt, "tz", None)
        if tz is None:
            quality.add_timezone_assumption(
                "datetime64 input without a timezone was read as UTC (the project never guesses a local zone)"
            )
        else:
            quality.add_timezone_assumption(f"datetime64 input carried timezone {tz}; converted to UTC instants")
        return pd.to_datetime(series, utc=True, errors="coerce")

    text = series.astype("string").fillna("").str.strip()
    has_offset = int(text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).sum())
    quality.add_note(
        f"timestamp strings: {has_offset} with an explicit offset, {int((~text.eq('')).sum()) - has_offset} without"
    )
    if has_offset:
        quality.add_timezone_assumption(
            f"{has_offset} timestamp string(s) carried an explicit offset and were converted to UTC instants"
        )
    quality.add_timezone_assumption(
        f"timestamp strings without an offset were read as UTC; ISO-8601 is required ({_ISO8601_EXAMPLE}) and "
        "ambiguous formats are rejected instead of guessed"
    )
    return pd.to_datetime(series, utc=True, errors="coerce", format="ISO8601")


def clean_market_data(
    frame: pd.DataFrame,
    *,
    symbol: str | None = None,
    report: QualityReport | None = None,
) -> tuple[pd.DataFrame, QualityReport]:
    """Validate a raw market table and return ``(cleaned, quality_report)``."""
    quality = report if report is not None else QualityReport(table="market_clean")
    quality.table = "market_clean"
    quality.rows_read = int(len(frame))

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise DataSourceError(
            f"market table is missing required column(s) {missing_columns}; present columns: {list(frame.columns)}"
        )
    if frame.empty:
        raise DataSourceError("market table has no rows to clean")

    working = frame.copy()

    working["timestamp"] = _coerce_timestamp(working["timestamp"], quality)
    unparsable = working["timestamp"].isna()
    quality.rows_dropped_unparsable_timestamp = int(unparsable.sum())
    if quality.rows_dropped_unparsable_timestamp:
        quality.add_warning(
            f"{quality.rows_dropped_unparsable_timestamp} row(s) had a missing or non-ISO-8601 timestamp and were dropped"
        )
        working = working.loc[~unparsable]

    for column in (*PRICE_COLUMNS, "volume"):
        if column not in working.columns:
            working[column] = pd.NA
            quality.add_warning(f"column {column!r} was absent from the raw table; stored as missing (never computed)")
        original = working[column]
        coerced = pd.to_numeric(original, errors="coerce")
        newly_missing = int((original.notna() & coerced.isna()).sum())
        if newly_missing:
            quality.coerced_to_missing[column] = newly_missing
            quality.add_warning(f"{newly_missing} value(s) in {column!r} were not numeric and are stored as missing")
        working[column] = coerced

    working["symbol"] = working["symbol"].astype("string").str.strip().str.upper()
    if symbol is not None:
        unexpected = int((working["symbol"] != symbol.upper()).sum())
        if unexpected:
            quality.add_warning(f"{unexpected} row(s) carry a symbol other than the configured {symbol.upper()!r}")
    if "source" in working.columns:
        working["source"] = working["source"].astype("string")
    else:
        working["source"] = pd.Series(pd.NA, index=working.index, dtype="string")
        quality.add_warning("no 'source' column in the raw table; provenance column stored as missing")

    quality.input_chronological = bool(working["timestamp"].is_monotonic_increasing)
    if not quality.input_chronological:
        quality.add_warning("raw rows were not in chronological order; sorted ascending (no shuffling)")
    working = working.sort_values("timestamp", kind="stable")

    quality.duplicate_key = "symbol+timestamp"
    before = len(working)
    working = working.drop_duplicates(subset=["symbol", "timestamp"], keep="first")
    quality.duplicates_removed = int(before - len(working))
    if quality.duplicates_removed:
        quality.add_warning(
            f"{quality.duplicates_removed} duplicate (symbol, timestamp) row(s) removed, keeping the first occurrence"
        )

    non_positive_close = int((working["close"].notna() & (working["close"] <= 0)).sum())
    if non_positive_close:
        quality.add_warning(f"{non_positive_close} row(s) have a non-positive close price")
    negative_volume = int((working["volume"].notna() & (working["volume"] < 0)).sum())
    if negative_volume:
        quality.add_warning(f"{negative_volume} row(s) have negative volume")
    inverted_range = int((working["high"].notna() & working["low"].notna() & (working["high"] < working["low"])).sum())
    if inverted_range:
        quality.add_warning(f"{inverted_range} row(s) have high < low")

    working = working[MARKET_COLUMNS].reset_index(drop=True)
    quality.rows_written = int(len(working))
    quality.record_missing(working, MARKET_COLUMNS)
    quality.record_date_range(working["timestamp"])
    quality.add_note(
        "no imputation or forward-fill: missing prices stay missing, and non-trading days simply have no row (invariant 3)"
    )
    quality.add_note(
        "each row is one exchange session; the timestamp is a UTC date stamp for that session, not an intraday instant"
    )
    return working, quality


__all__ = ["MARKET_COLUMNS", "PRICE_COLUMNS", "REQUIRED_COLUMNS", "clean_market_data"]
