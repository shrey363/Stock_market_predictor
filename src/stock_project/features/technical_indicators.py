"""Market feature engineering for the Phase 3 baseline.

Cutoff contract (``docs/ARCHITECTURE.md`` §3, invariant 2)
----------------------------------------------------------
A feature row is indexed by the session it predicts, ``t``. Its
``decision_time_utc`` is that session's own timestamp — the pre-open decision
point (05:00 UTC in the cached Yahoo series). Every indicator in the row is
computed from bars up to and including ``t - 1`` only: each indicator is
evaluated at the close of its bar and then shifted forward by one row, so no
value from session ``t`` or later can reach the row. The test suite proves this
by mutating prices at and after ``t`` and re-checking the row
(``tests/test_technical_indicators.py``).

The target is the direction of session ``t`` itself, relative to the previous
close: from the pre-open decision point, session ``t`` *is* the next session.

``news_cutoff_utc`` is the latest news availability a future news-aware row may
use: the session date at the configured ``prediction.cutoff_time`` (00:00 by
default). Because the news source carries only a calendar date, a headline
published on day ``t-1`` becomes available at 00:00 of day ``t`` and is usable
for row ``t``; a headline published on day ``t`` is not. Phase 3 itself uses no
news data — the column is part of the contract for Phase 4.

No value is ever imputed: rows whose indicators are still undefined (the warm-up
window) are dropped and counted, never filled.
"""

from __future__ import annotations

import pandas as pd

from ..common.errors import DataSourceError
from ..common.quality import QualityReport

#: Columns the cleaned Phase 1 market table must provide.
REQUIRED_MARKET_COLUMNS = ("symbol", "timestamp", "open", "high", "low", "close", "volume")

#: Context columns carried next to the numeric features.
CONTEXT_COLUMNS = ("symbol", "session_date", "decision_time_utc", "news_cutoff_utc")

#: Target columns. Never features: they are what the model is asked to predict.
TARGET_COLUMNS = ("next_session_return", "target")


def simple_returns(close: pd.Series, window: int = 1) -> pd.Series:
    """``close[t] / close[t - window] - 1`` — a simple (not log) return."""
    if window < 1:
        raise ValueError(f"return window must be >= 1, got {window}")
    return close / close.shift(window) - 1.0


def sma_ratio(close: pd.Series, window: int) -> pd.Series:
    """Distance of the close from its simple moving average, as a fraction.

    ``close[t] / mean(close[t-window+1 .. t]) - 1``: positive above the average.
    """
    if window < 2:
        raise ValueError(f"moving-average window must be >= 2, got {window}")
    average = close.rolling(window=window).mean()
    return close / average - 1.0


def _rsi_from_averages(average_gain: float, average_loss: float) -> float:
    if average_loss == 0.0:
        # No downward movement in the window: 100 when there were gains, 50 when
        # the price never moved at all (an undefined ratio, reported as neutral).
        return 50.0 if average_gain == 0.0 else 100.0
    relative_strength = average_gain / average_loss
    return 100.0 - 100.0 / (1.0 + relative_strength)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index on closing prices.

    The first value needs ``window`` price changes; averages are then smoothed
    recursively as ``(previous * (window - 1) + current) / window``. NaN until
    the window is full — never filled.
    """
    if window < 2:
        raise ValueError(f"RSI window must be >= 2, got {window}")
    values = pd.Series(float("nan"), index=close.index, dtype="float64")
    if len(close) <= window:
        return values

    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    average_gain = float(gains.iloc[1 : window + 1].mean())
    average_loss = float(losses.iloc[1 : window + 1].mean())
    values.iloc[window] = _rsi_from_averages(average_gain, average_loss)

    for position in range(window + 1, len(close)):
        average_gain = (average_gain * (window - 1) + float(gains.iloc[position])) / window
        average_loss = (average_loss * (window - 1) + float(losses.iloc[position])) / window
        values.iloc[position] = _rsi_from_averages(average_gain, average_loss)
    return values


def rolling_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Per-session standard deviation of simple returns over ``window`` sessions.

    Deliberately *not* annualised: the value is a per-session volatility, and the
    report says so, so no hidden scaling factor is applied anywhere.
    """
    if window < 2:
        raise ValueError(f"volatility window must be >= 2, got {window}")
    return close.pct_change().rolling(window=window).std(ddof=1)


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """``volume[t] / mean(volume[t-window+1 .. t])`` — 1.0 means average volume."""
    if window < 2:
        raise ValueError(f"volume window must be >= 2, got {window}")
    return volume / volume.rolling(window=window).mean()


def intraday_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """``(high - low) / close``: how wide the session's range was, relative to price."""
    return (high - low) / close


def overnight_gap(open_: pd.Series, close: pd.Series) -> pd.Series:
    """``open[t] / close[t-1] - 1``: the gap between the previous close and the open."""
    return open_ / close.shift(1) - 1.0


def _parse_cutoff_offset(cutoff_time: str) -> pd.Timedelta:
    parts = cutoff_time.split(":")
    if len(parts) != 2 or not all(part.isdigit() and len(part) == 2 for part in parts):
        raise ValueError(f"cutoff_time must be HH:MM, got {cutoff_time!r}")
    hours, minutes = int(parts[0]), int(parts[1])
    if not (0 <= hours <= 23 and 0 <= minutes <= 59):
        raise ValueError(f"cutoff_time must be a valid 24-hour time, got {cutoff_time!r}")
    return pd.Timedelta(hours=hours, minutes=minutes)


def feature_definitions(
    *,
    return_windows: tuple[int, ...] | list[int] = (1, 5, 20),
    moving_average_windows: tuple[int, ...] | list[int] = (10, 50),
    rsi_window: int = 14,
    volatility_window: int = 20,
    volume_window: int = 20,
) -> dict[str, str]:
    """Human-readable definition of every feature column, for the report and docs."""
    definitions: dict[str, str] = {}
    for window in return_windows:
        definitions[f"return_{window}d"] = f"close[t-1] / close[t-1-{window}] - 1 (simple return over {window} session(s), known at close t-1)"
    for window in moving_average_windows:
        definitions[f"sma_ratio_{window}"] = f"close[t-1] / mean(close[t-{window} .. t-1]) - 1 at close t-1"
    definitions[f"rsi_{rsi_window}"] = f"Wilder RSI({rsi_window}) on closes up to t-1"
    definitions[f"volatility_{volatility_window}"] = f"stdev(ddof=1) of simple returns over {volatility_window} sessions ending at t-1 (not annualised)"
    definitions[f"volume_ratio_{volume_window}"] = f"volume[t-1] / mean(volume[t-{volume_window} .. t-1]) at close t-1"
    definitions["intraday_range"] = "(high - low) / close at t-1"
    definitions["overnight_gap"] = "open[t-1] / close[t-2] - 1 (gap into session t-1)"
    return definitions


def build_market_features(
    frame: pd.DataFrame,
    *,
    return_windows: tuple[int, ...] | list[int] = (1, 5, 20),
    moving_average_windows: tuple[int, ...] | list[int] = (10, 50),
    rsi_window: int = 14,
    volatility_window: int = 20,
    volume_window: int = 20,
    cutoff_time: str = "00:00",
    report: QualityReport | None = None,
) -> tuple[pd.DataFrame, QualityReport]:
    """Build one feature row per predicted session, plus its binary target.

    Returns ``(feature_frame, quality_report)``. The frame is chronological,
    has no missing feature values, and carries ``CONTEXT_COLUMNS``,
    the feature columns, then ``TARGET_COLUMNS``.
    """
    quality = report if report is not None else QualityReport(table="market_features")
    quality.table = "market_features"
    quality.rows_read = int(len(frame))

    missing = [column for column in REQUIRED_MARKET_COLUMNS if column not in frame.columns]
    if missing:
        raise DataSourceError(
            f"market table is missing required column(s) {missing}; present columns: {list(frame.columns)}"
        )
    if frame.empty:
        raise DataSourceError("market table has no rows to build features from")

    working = frame.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    if working["timestamp"].isna().any():
        raise DataSourceError("market table contains timestamps that could not be parsed as UTC")

    quality.input_chronological = bool(working["timestamp"].is_monotonic_increasing)
    working = working.sort_values("timestamp", kind="stable").reset_index(drop=True)

    close = working["close"].astype("float64")
    for column, series in (("open", working["open"]), ("high", working["high"]), ("low", working["low"]), ("volume", working["volume"])):
        if series.isna().any():
            quality.add_warning(f"{int(series.isna().sum())} missing value(s) in {column!r}; affected rows are dropped, never filled")

    indicators: dict[str, pd.Series] = {}
    for window in return_windows:
        indicators[f"return_{window}d"] = simple_returns(close, window)
    for window in moving_average_windows:
        indicators[f"sma_ratio_{window}"] = sma_ratio(close, window)
    indicators[f"rsi_{rsi_window}"] = rsi(close, rsi_window)
    indicators[f"volatility_{volatility_window}"] = rolling_volatility(close, volatility_window)
    indicators[f"volume_ratio_{volume_window}"] = volume_ratio(working["volume"].astype("float64"), volume_window)
    indicators["intraday_range"] = intraday_range(working["high"].astype("float64"), working["low"].astype("float64"), close)
    indicators["overnight_gap"] = overnight_gap(working["open"].astype("float64"), close)

    feature_columns = list(indicators)
    # Invariant 2: an indicator known at the close of bar s belongs to the row that
    # predicts s+1. Shifting by exactly one row is what makes the cutoff provable.
    feature_frame = pd.DataFrame(indicators).shift(1)

    decision_times = working["timestamp"]
    news_cutoff = decision_times.dt.normalize() + _parse_cutoff_offset(cutoff_time)

    next_session_return = close / close.shift(1) - 1.0
    target = (next_session_return > 0).astype("int64").where(next_session_return.notna())

    rows = pd.DataFrame(
        {
            "symbol": working["symbol"].astype("string"),
            "session_date": decision_times.dt.strftime("%Y-%m-%d"),
            "decision_time_utc": decision_times.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "news_cutoff_utc": news_cutoff.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    rows = pd.concat([rows, feature_frame], axis=1)
    rows["next_session_return"] = next_session_return
    rows["target"] = target

    warmup = feature_frame.isna().any(axis=1) | next_session_return.isna()
    quality.rows_dropped_missing_required = int(warmup.sum())
    if quality.rows_dropped_missing_required:
        quality.add_note(
            f"{quality.rows_dropped_missing_required} leading row(s) dropped: the indicators' warm-up window "
            "is not yet full; nothing was imputed"
        )
    rows = rows.loc[~warmup].reset_index(drop=True)

    if rows.empty:
        raise DataSourceError(
            "no feature rows remain after the warm-up window; the market series is shorter than the "
            "longest configured indicator window"
        )
    if rows[feature_columns].isna().any().any():  # pragma: no cover - defensive
        raise AssertionError("feature frame still contains missing values after the warm-up drop")

    rows["target"] = rows["target"].astype("int64")

    quality.rows_written = int(len(rows))
    quality.record_missing(rows, list(rows.columns))
    quality.record_date_range(pd.to_datetime(rows["session_date"]))
    quality.add_note(
        "every feature is computed from bars up to and including t-1 and shifted forward one row, "
        "so session t itself cannot influence its own row (invariant 2)"
    )
    quality.add_note(
        "target = 1 when the predicted session closes above the previous close; "
        "next_session_return is the exact return used to derive it"
    )
    quality.add_note(
        f"news_cutoff_utc = session date at {cutoff_time} UTC; Phase 4 may only use articles whose "
        "available_from is at or before it (day-precision news published the day before at the latest)"
    )
    quality.add_note("no news data is read in Phase 3; this table is the market-only baseline's input")
    return rows, quality


__all__ = [
    "CONTEXT_COLUMNS",
    "REQUIRED_MARKET_COLUMNS",
    "TARGET_COLUMNS",
    "build_market_features",
    "feature_definitions",
    "intraday_range",
    "overnight_gap",
    "rolling_volatility",
    "rsi",
    "simple_returns",
    "sma_ratio",
    "volume_ratio",
]
