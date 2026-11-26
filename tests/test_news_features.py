"""Phase 4 news aggregation: cutoff window, no silent fills, and leakage guards."""

from __future__ import annotations

import pandas as pd
import pytest

from stock_project.common.errors import DataSourceError
from stock_project.features.news_features import (
    NO_NEWS_COLUMN,
    build_news_features,
    news_feature_columns,
    news_feature_definitions,
)
from stock_project.nlp.taxonomy import EVENT_CATEGORIES


def session_frame(*cutoffs: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_date": [cutoff[:10] for cutoff in cutoffs],
            "news_cutoff_utc": list(cutoffs),
        }
    )


def news_frame(*rows: dict[str, object]) -> pd.DataFrame:
    defaults = {
        "available_from": "2020-01-09T00:00:00Z",
        "event_category": "armed_conflict",
        "sentiment_label": "neutral",
        "sentiment_score": 0.0,
    }
    if not rows:
        return pd.DataFrame(columns=list(defaults))
    return pd.DataFrame([{**defaults, **row} for row in rows])


def build(sessions: pd.DataFrame, news: pd.DataFrame, *, lookback_days: int = 7) -> pd.DataFrame:
    result, _ = build_news_features(sessions, news, lookback_days=lookback_days)
    return result


def test_window_is_strictly_after_start_and_inclusive_at_cutoff():
    sessions = session_frame("2020-01-10T00:00:00Z")
    news = news_frame(
        {"available_from": "2020-01-03T00:00:00Z"},  # exactly 7 days before: excluded
        {"available_from": "2020-01-03T00:00:01Z"},  # inside
        {"available_from": "2020-01-10T00:00:00Z"},  # exactly at cutoff: included
        {"available_from": "2020-01-10T00:00:01Z"},  # after cutoff: excluded
    )
    row = build(sessions, news).iloc[0]
    assert row["news_count_total"] == 2
    assert row[NO_NEWS_COLUMN] == 0


def test_day_precision_news_published_the_day_before_is_usable():
    # The cleaned table sets available_from = published_at + 1 day.
    sessions = session_frame("2020-01-10T00:00:00Z")
    news = news_frame({"available_from": "2020-01-10T00:00:00Z"})
    assert build(sessions, news).iloc[0]["news_count_total"] == 1


def test_counts_sentiment_and_recency_match_hand_computation():
    sessions = session_frame("2020-01-10T00:00:00Z")
    news = news_frame(
        {"available_from": "2020-01-09T00:00:00Z", "event_category": "sanctions", "sentiment_label": "positive", "sentiment_score": 0.5},
        {"available_from": "2020-01-08T00:00:00Z", "event_category": "sanctions", "sentiment_label": "negative", "sentiment_score": -1.0},
        {"available_from": "2020-01-05T00:00:00Z", "event_category": "armed_conflict", "sentiment_label": "neutral", "sentiment_score": 0.0},
        {"available_from": "2020-01-04T00:00:00Z", "event_category": "other_or_uncertain", "sentiment_label": "positive", "sentiment_score": 1.0},
    )
    row = build(sessions, news).iloc[0]
    assert row["news_count_total"] == 4
    assert row["news_count_sanctions"] == 2
    assert row["news_count_armed_conflict"] == 1
    assert row["news_count_other_or_uncertain"] == 1
    assert row["news_count_trade_restriction"] == 0
    assert row["news_positive_count"] == 2
    assert row["news_negative_count"] == 1
    assert row["news_neutral_count"] == 1
    assert row["news_sentiment_sum"] == pytest.approx(0.5)
    assert row["news_days_since_last"] == pytest.approx(1.0)
    assert row[NO_NEWS_COLUMN] == 0


def test_session_without_news_is_explicit_not_imputed():
    sessions = session_frame("2020-01-10T00:00:00Z")
    row = build(sessions, news_frame()).iloc[0]
    assert row["news_count_total"] == 0
    assert all(row[f"news_count_{category}"] == 0 for category in EVENT_CATEGORIES)
    assert row["news_positive_count"] == row["news_negative_count"] == row["news_neutral_count"] == 0
    assert row["news_sentiment_sum"] == 0.0
    assert row["news_days_since_last"] == pytest.approx(7.0)
    assert row[NO_NEWS_COLUMN] == 1


def test_days_since_last_uses_history_but_is_capped():
    sessions = session_frame("2020-01-10T00:00:00Z")
    news = news_frame({"available_from": "2019-12-01T00:00:00Z"})  # 40 days earlier
    row = build(sessions, news, lookback_days=7).iloc[0]
    assert row[NO_NEWS_COLUMN] == 1  # outside the window
    assert row["news_days_since_last"] == pytest.approx(7.0)  # capped, documented


def test_post_cutoff_articles_cannot_change_a_session_row():
    sessions = session_frame("2020-01-10T00:00:00Z")
    cutoff_safe = news_frame({"available_from": "2020-01-09T00:00:00Z"})
    with_future = pd.concat(
        [
            cutoff_safe,
            news_frame(
                {
                    "available_from": "2020-01-10T00:00:01Z",
                    "event_category": "sanctions",
                    "sentiment_label": "negative",
                    "sentiment_score": -1.0,
                }
            ),
        ],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(build(sessions, cutoff_safe), build(sessions, with_future))


def test_feature_columns_and_definitions_agree():
    definitions = news_feature_definitions(7)
    assert list(definitions) == news_feature_columns()
    assert "7 day(s)" in definitions["news_count_total"]
    assert definitions[NO_NEWS_COLUMN].startswith("1 when no usable article")


def test_unknown_category_is_a_hard_error():
    sessions = session_frame("2020-01-10T00:00:00Z")
    news = news_frame({"event_category": "election"})
    with pytest.raises(DataSourceError, match="outside the taxonomy"):
        build(sessions, news)


def test_unparsable_available_from_is_a_hard_error():
    sessions = session_frame("2020-01-10T00:00:00Z")
    news = news_frame({"available_from": "not-a-date"})
    with pytest.raises(DataSourceError, match="could not be parsed as UTC"):
        build(sessions, news)


def test_missing_columns_are_hard_errors():
    sessions = session_frame("2020-01-10T00:00:00Z")
    with pytest.raises(DataSourceError, match="missing required column"):
        build(sessions.drop(columns=["news_cutoff_utc"]), news_frame())
    with pytest.raises(DataSourceError, match="missing required column"):
        build(sessions, news_frame().drop(columns=["sentiment_score"]))
