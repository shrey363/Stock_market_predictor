"""Phase 4 news feature aggregation.

Cutoff contract (``docs/ARCHITECTURE.md`` §3, invariants 2 and 3)
----------------------------------------------------------------
A session row may use only articles whose ``available_from`` (publication date +
1 day, see ``preprocessing/clean_news_data.py``) falls in the window
``(news_cutoff_utc - lookback_days, news_cutoff_utc]``. ``news_cutoff_utc`` is
that session's decision cutoff, produced by ``features/technical_indicators.py``.
Inclusive at the cutoff is deliberate: day-precision news published the day
before becomes available at 00:00 of the prediction day and is usable; news
published on the prediction day is not.

No silent fills
---------------
A session with no article in the window gets all-zero counts, a sentiment sum of
zero, ``no_news = 1``, and ``news_days_since_last = lookback_days``. Sentiment is
aggregated as a **sum**, not a mean, precisely so that absence of news cannot
masquerade as neutral sentiment: the mean is recoverable as
``news_sentiment_sum / news_count_total`` only where the count is positive, and
``no_news`` states explicitly when there was nothing to average.
``news_days_since_last`` looks at all history at or before the cutoff (not just
the window), capped at ``lookback_days`` so the value stays bounded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.errors import DataSourceError
from ..common.quality import QualityReport
from ..nlp.taxonomy import EVENT_CATEGORIES

#: Columns the session frame (market features) must provide.
REQUIRED_SESSION_COLUMNS = ("news_cutoff_utc",)

#: Columns the classified news table must provide.
REQUIRED_NEWS_COLUMNS = ("available_from", "event_category", "sentiment_label", "sentiment_score")

#: Sentiment labels the lexicon scorer can produce.
SENTIMENT_LABELS = ("positive", "negative", "neutral")

#: Explicit missing-news indicator (spec §8; invariant 3).
NO_NEWS_COLUMN = "no_news"


def news_feature_columns() -> list[str]:
    """Every column :func:`build_news_features` adds, in contract order."""
    return [
        "news_count_total",
        *(f"news_count_{category}" for category in EVENT_CATEGORIES),
        "news_positive_count",
        "news_negative_count",
        "news_neutral_count",
        "news_sentiment_sum",
        "news_days_since_last",
        NO_NEWS_COLUMN,
    ]


def news_feature_definitions(lookback_days: int) -> dict[str, str]:
    """Human-readable definition of every news feature, for the report and docs."""
    window = f"(cutoff - {lookback_days} day(s), cutoff]"
    definitions = {
        "news_count_total": f"number of usable articles with available_from in {window}",
    }
    for category in EVENT_CATEGORIES:
        definitions[f"news_count_{category}"] = (
            f"number of usable {category} articles with available_from in {window}"
        )
    definitions.update(
        {
            "news_positive_count": f"articles labelled positive with available_from in {window}",
            "news_negative_count": f"articles labelled negative with available_from in {window}",
            "news_neutral_count": f"articles labelled neutral with available_from in {window}",
            "news_sentiment_sum": f"sum of sentiment_score over articles with available_from in {window}",
            "news_days_since_last": (
                "days from the most recent usable article's available_from to the cutoff, capped at "
                f"{lookback_days}; {lookback_days} when no article exists at or before the cutoff"
            ),
            NO_NEWS_COLUMN: f"1 when no usable article falls in {window}, else 0 (never imputed as neutral sentiment)",
        }
    )
    return definitions


def _as_utc(values: pd.Series, *, column: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise DataSourceError(
            f"{column} contains {int(parsed.isna().sum())} value(s) that could not be parsed as UTC; "
            "nothing is dropped silently"
        )
    return parsed


def build_news_features(
    sessions: pd.DataFrame,
    news: pd.DataFrame,
    *,
    lookback_days: int,
    report: QualityReport | None = None,
) -> tuple[pd.DataFrame, QualityReport]:
    """Aggregate pre-cutoff news into one row per session.

    ``sessions`` is the market feature table (or any frame carrying
    ``news_cutoff_utc``); ``news`` is the classified news table. The returned
    frame has the same index and row order as ``sessions`` and contains exactly
    :func:`news_feature_columns`.
    """
    if lookback_days < 1:
        raise ValueError(f"lookback_days must be >= 1, got {lookback_days}")
    quality = report if report is not None else QualityReport(table="news_features")
    quality.table = "news_features"
    quality.rows_read = int(len(news))

    missing_sessions = [column for column in REQUIRED_SESSION_COLUMNS if column not in sessions.columns]
    if missing_sessions:
        raise DataSourceError(
            f"session frame is missing required column(s) {missing_sessions}; present columns: {list(sessions.columns)}"
        )
    missing_news = [column for column in REQUIRED_NEWS_COLUMNS if column not in news.columns]
    if missing_news:
        raise DataSourceError(
            f"news table is missing required column(s) {missing_news}; present columns: {list(news.columns)}"
        )

    working_news = news.copy()
    working_news["available_from"] = _as_utc(working_news["available_from"], column="available_from")
    cutoffs = _as_utc(sessions["news_cutoff_utc"], column="news_cutoff_utc")

    unknown_categories = sorted(set(working_news["event_category"].astype("string")) - set(EVENT_CATEGORIES))
    if unknown_categories:
        raise DataSourceError(
            f"news table contains event_category value(s) outside the taxonomy: {unknown_categories}"
        )
    unknown_labels = sorted(set(working_news["sentiment_label"].astype("string")) - set(SENTIMENT_LABELS))
    if unknown_labels:
        raise DataSourceError(f"news table contains sentiment_label value(s) outside {list(SENTIMENT_LABELS)}: {unknown_labels}")

    scores = pd.to_numeric(working_news["sentiment_score"], errors="coerce")
    if scores.isna().any():
        raise DataSourceError(
            f"news table contains {int(scores.isna().sum())} non-numeric or missing sentiment_score value(s)"
        )
    working_news["sentiment_score"] = scores
    if len(working_news) and ((scores < -1.0) | (scores > 1.0)).any():
        quality.add_warning("sentiment_score values outside [-1, 1] were found; they are used unchanged")

    if not working_news.empty:
        quality.input_chronological = bool(working_news["available_from"].is_monotonic_increasing)
    working_news = working_news.sort_values("available_from", kind="stable").reset_index(drop=True)

    times = working_news["available_from"].to_numpy(dtype="datetime64[ns]")
    cutoff_values = cutoffs.to_numpy(dtype="datetime64[ns]")
    window_start = cutoff_values - np.timedelta64(int(lookback_days), "D")
    # (start, end] — start is exclusive, end is inclusive (available_from <= cutoff).
    starts = np.searchsorted(times, window_start, side="right")
    ends = np.searchsorted(times, cutoff_values, side="right")

    rows: list[dict[str, object]] = []
    for start, end, cutoff, last in zip(starts, ends, cutoff_values, ends - 1):
        chunk = working_news.iloc[int(start) : int(end)]
        category_counts = chunk["event_category"].value_counts()
        label_counts = chunk["sentiment_label"].value_counts()
        if last >= 0:
            days_since = float((cutoff - times[int(last)]) / np.timedelta64(1, "D"))
            days_since = min(days_since, float(lookback_days))
        else:
            days_since = float(lookback_days)
        row: dict[str, object] = {
            "news_count_total": int(len(chunk)),
            "news_positive_count": int(label_counts.get("positive", 0)),
            "news_negative_count": int(label_counts.get("negative", 0)),
            "news_neutral_count": int(label_counts.get("neutral", 0)),
            "news_sentiment_sum": round(float(chunk["sentiment_score"].sum()), 6),
            "news_days_since_last": round(days_since, 6),
            NO_NEWS_COLUMN: int(len(chunk) == 0),
        }
        for category in EVENT_CATEGORIES:
            row[f"news_count_{category}"] = int(category_counts.get(category, 0))
        rows.append(row)

    columns = news_feature_columns()
    result = pd.DataFrame(rows, index=sessions.index, columns=columns)
    float_columns = {"news_sentiment_sum", "news_days_since_last"}
    for column in columns:
        result[column] = result[column].astype("float64" if column in float_columns else "int64")

    no_news_sessions = int(result[NO_NEWS_COLUMN].sum())
    quality.rows_written = int(len(result))
    quality.add_note(
        f"aggregation window is (cutoff - {lookback_days} day(s), cutoff]; only articles whose "
        "available_from is at or before the cutoff are used (invariant 2)"
    )
    quality.add_note(
        f"{no_news_sessions} of {len(result)} session(s) have no article in the window and carry "
        "zero counts plus no_news=1; no sentiment value was imputed"
    )
    quality.add_note(
        "article membership is per-session: one article can fall inside several overlapping windows, "
        "so window counts are not unique-article totals"
    )
    return result, quality


__all__ = [
    "NO_NEWS_COLUMN",
    "REQUIRED_NEWS_COLUMNS",
    "REQUIRED_SESSION_COLUMNS",
    "SENTIMENT_LABELS",
    "build_news_features",
    "news_feature_columns",
    "news_feature_definitions",
]
