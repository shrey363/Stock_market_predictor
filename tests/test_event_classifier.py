"""Event classifier behaviour: categories, tie-breaks, evidence flags, scopes.

Every expectation is hand-checked against the lexicons in ``nlp/taxonomy.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stock_project.nlp.event_classifier import (
    EVENT_COLUMNS,
    classification_input,
    classification_summary,
    classify_frame,
    classify_text,
    tokenize,
)
from stock_project.nlp.taxonomy import EVENT_CATEGORIES, OTHER_CATEGORY


@pytest.mark.parametrize(
    ("headline", "expected"),
    [
        ("Missiles strike port as troops advance", "armed_conflict"),
        ("EU imposes new sanctions on banks", "sanctions"),
        ("U.S. announces tariffs on steel imports", "trade_restriction"),
        ("Pipeline outage causes gas shortage in Europe", "energy_or_supply_disruption"),
        ("Leaders hold summit talks on security", "diplomatic_development"),
        ("City council debates parking rules", OTHER_CATEGORY),
        ("Trump threatens tariffs on Chinese goods", "trade_restriction"),
        ("Ceasefire agreed after talks", "armed_conflict"),
    ],
)
def test_categories_are_assigned_as_expected(headline: str, expected: str):
    assert classify_text(headline).category == expected


def test_tokens_and_punctuation_are_handled():
    assert tokenize("U.S. imposes sanctions on Russia!") == ["u", "s", "imposes", "sanctions", "on", "russia"]
    assert classify_text("U.S. imposes sanctions on Russia!").category == "sanctions"
    assert tokenize(None) == []


def test_priority_order_decides_ties_and_records_the_competition():
    match = classify_text("Sanctions imposed after summit collapse")
    assert match.category == "sanctions"
    assert match.priority < EVENT_CATEGORIES.index("diplomatic_development")
    assert "diplomatic_development" in match.competing_categories


def test_unmatched_text_falls_back_without_inventing_a_label():
    match = classify_text("Local election results announced")
    assert match.category == OTHER_CATEGORY
    assert match.matched_keywords == ()
    assert match.matched_term_count == 0
    assert match.competing_categories == ()


def test_empty_headlines_are_handled():
    for empty in ("", "   ", None, float("nan")):
        match = classify_text(empty)
        assert match.category == OTHER_CATEGORY
        assert match.matched_keywords == ()


def test_negated_evidence_is_flagged_but_keeps_the_event_type():
    match = classify_text("No new sanctions expected, official says")
    assert match.category == "sanctions"
    assert match.evidence_negated is True


def test_reversal_wording_counts_as_negated_evidence():
    match = classify_text("U.S. lifts sanctions on the bank")
    assert match.category == "sanctions"
    assert match.evidence_negated is True


def test_speculation_is_flagged_but_keeps_the_event_type():
    match = classify_text("Trump threatens tariffs on Chinese goods")
    assert match.category == "trade_restriction"
    assert match.conditional is True
    assert match.evidence_negated is False


def test_evidence_is_recorded_for_audit():
    match = classify_text("Missiles strike port as troops advance")
    assert set(match.matched_keywords) == {"missiles", "troops"}
    row = match.to_row()
    assert row["matched_keywords"] == "missiles; troops"
    assert row["event_priority"] == 0
    assert row["classified_scope"] == "headline"


@pytest.mark.parametrize(
    ("headline", "expected", "vetoed"),
    [
        (
            "Trade war escalates between the U.S. and China",
            "trade_restriction",
            "armed_conflict: trade war",
        ),
        (
            "California's Marijuana Legalization Aims To Repair Damage From The War On Drugs",
            OTHER_CATEGORY,
            "armed_conflict: war on drugs",
        ),
        (
            "Prosecutor Drops Felony Invasion Of Privacy Charge Against Missouri Governor",
            OTHER_CATEGORY,
            "armed_conflict: invasion of privacy",
        ),
        (
            "Hawaii Governor Assigns General To Review Botched Missile Alert",
            OTHER_CATEGORY,
            "armed_conflict: missile alert",
        ),
        ("Ukraine war continues as talks stall", "armed_conflict", ""),
    ],
)
def test_reviewed_false_positives_are_suppressed_and_recorded(headline: str, expected: str, vetoed: str):
    """The veto list exists because the first real-data review found these errors."""
    match = classify_text(headline)
    assert match.category == expected
    if vetoed:
        assert vetoed in match.excluded_evidence
    else:
        assert match.excluded_evidence == ()


def test_summary_counts_suppressed_evidence():
    frame = pd.DataFrame({"headline": ["Trade war escalates", "City council debates parking"]})
    summary = classification_summary(classify_frame(frame))
    assert summary["articles_with_excluded_evidence"] == 1


def test_classification_is_deterministic():
    headline = "Sanctions and tariffs follow missile strikes"
    first = classify_text(headline)
    second = classify_text(headline)
    assert first == second
    assert first.category == "armed_conflict"


def test_description_scope_adds_evidence_only_when_configured():
    frame = pd.DataFrame(
        [
            {
                "article_id": "a1",
                "headline": "Talks continue in the region",
                "article_text": "Officials discussed the missile strikes.",
            }
        ]
    )
    headline_only = classify_frame(frame, text_scope="headline")
    both = classify_frame(frame, text_scope="headline+description")
    assert headline_only.loc[0, "event_category"] == "diplomatic_development"
    assert both.loc[0, "event_category"] == "armed_conflict"
    assert both.loc[0, "classified_scope"] == "headline+description"


def test_classify_frame_adds_columns_and_preserves_rows(market_raw: pd.DataFrame):
    frame = pd.DataFrame(
        {
            "article_id": ["a1", "a2"],
            "headline": ["EU imposes new sanctions", "City council debates parking"],
        }
    )
    classified = classify_frame(frame)
    assert list(classified.columns) == [*frame.columns, *EVENT_COLUMNS]
    assert list(classified["article_id"]) == ["a1", "a2"]
    assert list(classified["event_category"]) == ["sanctions", OTHER_CATEGORY]


def test_classify_frame_rejects_bad_inputs():
    with pytest.raises(ValueError, match="text_scope"):
        classify_frame(pd.DataFrame({"headline": ["x"]}), text_scope="everything")
    with pytest.raises(ValueError, match="headline"):
        classify_frame(pd.DataFrame({"title": ["x"]}))


def test_classification_input_handles_missing_description():
    row = pd.Series({"headline": "Summit talks", "article_text": float("nan")})
    assert classification_input(row, text_scope="headline+description", headline_column="headline", description_column="article_text") == "Summit talks"
    assert classification_input(row, text_scope="headline", headline_column="headline", description_column="article_text") == "Summit talks"


def test_summary_reports_distribution_and_imbalance():
    frame = pd.DataFrame(
        {
            "headline": [
                "EU imposes new sanctions",
                "No new sanctions expected",
                "City council debates parking",
            ]
        }
    )
    summary = classification_summary(classify_frame(frame, text_scope="headline"))
    assert summary["articles_classified"] == 3
    assert summary["category_distribution"]["sanctions"]["count"] == 2
    assert summary["category_distribution"][OTHER_CATEGORY]["count"] == 1
    assert summary["class_imbalance"]["largest_class"] == "sanctions"
    assert summary["class_imbalance"]["largest_to_smallest_ratio"] == 2.0
    assert "balanced" in summary["class_imbalance"]["note"]


def test_summary_requires_a_classified_frame():
    with pytest.raises(ValueError, match="not been classified"):
        classification_summary(pd.DataFrame({"headline": ["x"]}))
