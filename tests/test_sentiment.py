"""Lexicon sentiment baseline: exact scores, negation, modes, and framing."""

from __future__ import annotations

import pandas as pd
import pytest

from stock_project.nlp.sentiment import (
    NEGATION_TERMS,
    SENTIMENT_COLUMNS,
    get_sentiment_scorer,
    score_frame,
    score_text,
    sentiment_summary,
)


@pytest.mark.parametrize(
    ("text", "label", "score"),
    [
        ("Shares plunge as fears of a recession grow", "negative", -1.0),
        ("Markets rally after a strong earnings beat", "positive", 1.0),
        ("Officials met on Tuesday to review the file", "neutral", 0.0),
        ("Stocks gain and losses narrow", "neutral", 0.0),
        ("A deal is reached and a crisis is averted", "neutral", 0.0),
    ],
)
def test_scores_are_exact(text: str, label: str, score: float):
    result = score_text(text)
    assert result.label == label
    assert result.score == score


def test_negation_flips_the_following_term():
    result = score_text("No damage reported at the plant")
    assert "damage" in result.positive_terms
    assert result.negative_terms == ()
    assert result.label == "positive"
    assert result.score == 1.0


def test_terms_are_recorded_with_polarity():
    result = score_text("Fears grow but a deal is close")
    assert "fears:-1" in result.terms_string()
    assert "deal:+1" in result.terms_string()
    assert result.term_count == 2


def test_empty_and_missing_text_is_neutral():
    for empty in ("", "   ", None):
        result = score_text(empty)
        assert result.label == "neutral"
        assert result.score == 0.0
        assert result.term_count == 0


def test_score_is_always_within_bounds():
    text = " ".join(["crisis", "plunge", "fears", "relief", "deal", "surge"])
    assert -1.0 <= score_text(text).score <= 1.0


def test_score_frame_adds_the_documented_columns_and_keeps_rows():
    frame = pd.DataFrame({"article_id": ["a1", "a2"], "headline": ["Shares plunge", "A deal is reached"]})
    scored = score_frame(frame)
    assert list(scored.columns) == [*frame.columns, *SENTIMENT_COLUMNS]
    assert list(scored["sentiment_label"]) == ["negative", "positive"]


def test_score_frame_rejects_a_missing_column():
    with pytest.raises(ValueError, match="headline"):
        score_frame(pd.DataFrame({"title": ["x"]}))


def test_mode_selector_exposes_only_the_implemented_scorer():
    assert get_sentiment_scorer("lexicon") is score_text
    with pytest.raises(NotImplementedError, match="finbert"):
        get_sentiment_scorer("finbert")
    with pytest.raises(ValueError, match="unknown sentiment_mode"):
        get_sentiment_scorer("gpt")


def test_summary_reports_labels_and_framing_note():
    frame = score_frame(
        pd.DataFrame({"headline": ["Shares plunge", "A deal is reached", "Officials met on Tuesday"]})
    )
    summary = sentiment_summary(frame)
    assert summary["label_distribution"]["negative"]["count"] == 1
    assert summary["label_distribution"]["neutral"]["count"] == 1
    assert summary["articles_with_no_term"] == 1
    assert "not severity" in summary["note"]


def test_summary_requires_sentiment_columns():
    with pytest.raises(ValueError, match="sentiment"):
        sentiment_summary(pd.DataFrame({"headline": ["x"]}))


def test_negation_terms_are_lowercase_tokens():
    assert all(term == term.lower().strip() for term in NEGATION_TERMS)
