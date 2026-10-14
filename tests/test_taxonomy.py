"""Taxonomy integrity: the labels must match the specification and stay consistent."""

from __future__ import annotations

import re
from pathlib import Path

from stock_project.common.paths import PROJECT_ROOT
from stock_project.nlp import taxonomy

SPEC = PROJECT_ROOT / "PROJECT_DESCRIPTION.md"

SPEC_SECTION_START = "## 7. Initial Event Categories"


def _spec_categories() -> set[str]:
    """The categories listed in the specification's taxonomy section."""
    text = SPEC.read_text(encoding="utf-8")
    start = text.index(SPEC_SECTION_START)
    end = text.index("### Classification approach", start)
    section = text[start:end]
    return set(re.findall(r"`([a-z_]+)`", section))


def test_categories_match_the_specification_exactly():
    assert set(taxonomy.EVENT_CATEGORIES) == _spec_categories()
    assert len(taxonomy.EVENT_CATEGORIES) == 6
    assert taxonomy.OTHER_CATEGORY == "other_or_uncertain"


def test_uncertain_category_is_last_in_priority_order():
    """It is the fallback, so it must never pre-empt a confident category."""
    assert taxonomy.EVENT_CATEGORIES[-1] == taxonomy.OTHER_CATEGORY
    assert taxonomy.category_priority(taxonomy.OTHER_CATEGORY) == len(taxonomy.EVENT_CATEGORIES) - 1


def test_priority_order_is_documented_in_the_guidelines():
    guidelines = (PROJECT_ROOT / "docs" / "annotation_guidelines.md").read_text(encoding="utf-8")
    for category in taxonomy.EVENT_CATEGORIES:
        assert f"`{category}`" in guidelines, category


def test_every_category_has_a_definition():
    assert set(taxonomy.CATEGORY_DEFINITIONS) == set(taxonomy.EVENT_CATEGORIES)
    assert all(definition.strip() for definition in taxonomy.CATEGORY_DEFINITIONS.values())


def test_keyword_lists_exist_and_have_no_duplicates():
    for category in taxonomy.EVENT_CATEGORIES:
        if category == taxonomy.OTHER_CATEGORY:
            continue
        keywords = taxonomy.EVENT_KEYWORDS[category]
        assert len(keywords) >= 10, category
        assert len(set(keywords)) == len(keywords), f"duplicate keyword in {category}"
        assert all(keyword == keyword.lower().strip() for keyword in keywords), category


def test_the_uncertain_category_has_no_keywords():
    """It is the fallback: nothing can match into it by keyword."""
    assert taxonomy.OTHER_CATEGORY not in taxonomy.EVENT_KEYWORDS


def test_marker_lists_are_usable():
    assert taxonomy.NEGATION_WINDOW >= 1 and taxonomy.CONDITIONAL_WINDOW >= 1
    assert len(taxonomy.NEGATION_TERMS) >= 10
    assert len(taxonomy.CONDITIONAL_TERMS) >= 10
    assert all(term == term.lower().strip() for term in taxonomy.NEGATION_TERMS)
    assert all(term == term.lower().strip() for term in taxonomy.CONDITIONAL_TERMS)


def test_sentiment_lexicons_are_disjoint_and_defined():
    positive = set(taxonomy.SENTIMENT_POSITIVE)
    negative = set(taxonomy.SENTIMENT_NEGATIVE)
    assert len(positive) >= 40 and len(negative) >= 40
    assert not positive & negative, f"term in both polarities: {positive & negative}"


def test_taxonomy_version_is_recorded():
    assert taxonomy.TAXONOMY_VERSION and isinstance(taxonomy.TAXONOMY_VERSION, str)
