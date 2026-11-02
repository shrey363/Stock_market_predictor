"""Transparent keyword/rule-based event classifier — the Phase 2 baseline.

Design rules (all tested):

* **Auditable.** Every prediction carries the exact matched keyword(s), the priority
  order that decided it, which categories *also* matched, and whether the evidence
  was negated or conditional. Nothing is a black box.
* **No confident guess.** A headline that matches nothing becomes
  ``other_or_uncertain`` — the taxonomy's default, never a forced label.
* **Deterministic.** Same text in, same result out. No randomness, no model weights.
* **Headline-first.** The default scope is the headline alone (a documented Phase 2
  limitation); ``headline+description`` is an explicit, recorded variant.

This is a baseline to beat, not a claim of accuracy. Phase 2 evaluates it against a
hand-labelled review sample, and the misclassification listing is a required output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .taxonomy import (
    CONDITIONAL_TERMS,
    CONDITIONAL_WINDOW,
    EVENT_CATEGORIES,
    EVENT_KEYWORDS,
    EXCLUSION_PHRASES,
    NEGATION_TERMS,
    NEGATION_WINDOW,
    OTHER_CATEGORY,
    category_priority,
)

#: Tokenisation is deliberately crude: lowercase alphanumeric runs.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

#: Columns this module adds to a news table.
EVENT_COLUMNS = [
    "event_category",
    "event_priority",
    "matched_keywords",
    "competing_categories",
    "excluded_evidence",
    "evidence_negated",
    "conditional",
    "classified_scope",
    "matched_term_count",
]

SCOPES = ("headline", "headline+description")


def tokenize(text: str | None) -> list[str]:
    """Lowercase alphanumeric tokens; non-text becomes an empty list."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    return TOKEN_PATTERN.findall(str(text).lower())


def _keyword_tokens(keyword: str) -> tuple[str, ...]:
    return tuple(tokenize(keyword))


# Pre-tokenise the lexicons once so matching is a plain list comparison.
_TOKENISED_KEYWORDS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    category: [(keyword, _keyword_tokens(keyword)) for keyword in keywords]
    for category, keywords in EVENT_KEYWORDS.items()
}
_TOKENISED_EXCLUSIONS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    category: [(phrase, _keyword_tokens(phrase)) for phrase in phrases]
    for category, phrases in EXCLUSION_PHRASES.items()
}


def _find_hits(tokens: list[str], keyword_tokens: tuple[str, ...]) -> list[int]:
    """Start indices where ``keyword_tokens`` appears as a token-sequence match."""
    size = len(keyword_tokens)
    if size == 0 or size > len(tokens):
        return []
    return [
        index
        for index in range(len(tokens) - size + 1)
        if tuple(tokens[index : index + size]) == keyword_tokens
    ]


def _negated_before(tokens: list[str], start: int) -> bool:
    """True when a negation/reversal term sits within the look-behind window."""
    window = tokens[max(0, start - NEGATION_WINDOW) : start]
    return any(token in NEGATION_TERMS for token in window)


def _conditional_nearby(tokens: list[str], start: int, length: int) -> bool:
    """True when a threat/plan/possibility term sits near the match."""
    low = max(0, start - CONDITIONAL_WINDOW)
    high = min(len(tokens), start + length + CONDITIONAL_WINDOW)
    return any(token in CONDITIONAL_TERMS for token in tokens[low:high])


def _exclusions_present(tokens: list[str], category: str) -> tuple[str, ...]:
    """Context phrases that veto this category's match ("trade war" is not kinetic conflict)."""
    return tuple(
        phrase
        for phrase, phrase_tokens in _TOKENISED_EXCLUSIONS.get(category, [])
        if _find_hits(tokens, phrase_tokens)
    )


@dataclass(frozen=True)
class EventMatch:
    """One classification decision, with the evidence that produced it."""

    category: str
    priority: int
    matched_keywords: tuple[str, ...]
    competing_categories: tuple[str, ...]
    excluded_evidence: tuple[str, ...]
    evidence_negated: bool
    conditional: bool
    scope: str
    matched_term_count: int

    def to_row(self) -> dict[str, object]:
        return {
            "event_category": self.category,
            "event_priority": self.priority,
            "matched_keywords": "; ".join(self.matched_keywords),
            "competing_categories": "; ".join(self.competing_categories),
            "excluded_evidence": "; ".join(self.excluded_evidence),
            "evidence_negated": self.evidence_negated,
            "conditional": self.conditional,
            "classified_scope": self.scope,
            "matched_term_count": self.matched_term_count,
        }


def classify_text(text: str | None, *, scope: str = "headline") -> EventMatch:
    """Classify one piece of text into the event taxonomy."""
    tokens = tokenize(text)
    hits: dict[str, list[tuple[str, int, int]]] = {}
    excluded: list[str] = []
    for category in EVENT_CATEGORIES:
        if category == OTHER_CATEGORY:
            continue
        found: list[tuple[str, int, int]] = []
        for keyword, keyword_tokens in _TOKENISED_KEYWORDS[category]:
            for start in _find_hits(tokens, keyword_tokens):
                found.append((keyword, start, len(keyword_tokens)))
        if not found:
            continue
        exclusions = _exclusions_present(tokens, category)
        if exclusions:
            # The keyword fired, but the surrounding context rules the category out.
            # Recorded, not hidden, so a reviewer can see why the category was suppressed.
            excluded.extend(f"{category}: {phrase}" for phrase in exclusions)
            continue
        hits[category] = found

    if not hits:
        return EventMatch(
            category=OTHER_CATEGORY,
            priority=category_priority(OTHER_CATEGORY),
            matched_keywords=(),
            competing_categories=(),
            excluded_evidence=tuple(excluded),
            evidence_negated=False,
            conditional=False,
            scope=scope,
            matched_term_count=0,
        )

    # Priority order decides: the first category in EVENT_CATEGORIES that matched.
    selected = next(category for category in EVENT_CATEGORIES if category in hits)
    found = hits[selected]
    return EventMatch(
        category=selected,
        priority=category_priority(selected),
        matched_keywords=tuple(dict.fromkeys(keyword for keyword, _start, _length in found)),
        competing_categories=tuple(category for category in EVENT_CATEGORIES if category in hits and category != selected),
        excluded_evidence=tuple(excluded),
        evidence_negated=any(_negated_before(tokens, start) for _keyword, start, _length in found),
        conditional=any(_conditional_nearby(tokens, start, length) for _keyword, start, length in found),
        scope=scope,
        matched_term_count=len(found),
    )


def _is_blank(value: object) -> bool:
    """True for None, NaN, and whitespace-only strings."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str):
        return not value.strip()
    return pd.isna(value)


def classification_input(row: pd.Series, *, text_scope: str, headline_column: str, description_column: str) -> str:
    """Build the text to classify, honouring the configured scope."""
    if text_scope not in SCOPES:
        raise ValueError(f"text_scope must be one of {list(SCOPES)}, got {text_scope!r}")
    headline = "" if _is_blank(row.get(headline_column)) else str(row[headline_column])
    if text_scope == "headline":
        return headline
    description = row.get(description_column)
    if _is_blank(description):
        return headline
    return f"{headline}. {description}"


def classify_frame(
    frame: pd.DataFrame,
    *,
    text_scope: str = "headline",
    headline_column: str = "headline",
    description_column: str = "article_text",
) -> pd.DataFrame:
    """Return a copy of ``frame`` with the event columns added."""
    if text_scope not in SCOPES:
        raise ValueError(f"text_scope must be one of {list(SCOPES)}, got {text_scope!r}")
    if headline_column not in frame.columns:
        raise ValueError(f"news table has no {headline_column!r} column; columns are {list(frame.columns)}")

    working = frame.copy()
    rows: list[dict[str, object]] = []
    for _, row in working.iterrows():
        text = classification_input(
            row,
            text_scope=text_scope,
            headline_column=headline_column,
            description_column=description_column,
        )
        rows.append(classify_text(text, scope=text_scope).to_row())
    additions = pd.DataFrame(rows, index=working.index)
    for column in EVENT_COLUMNS:
        working[column] = additions[column]
    return working


def classification_summary(frame: pd.DataFrame, *, total_articles: int | None = None) -> dict[str, object]:
    """Category distribution and evidence statistics for the artifact."""
    if "event_category" not in frame.columns:
        raise ValueError("frame has not been classified; run classify_frame first")
    counts = frame["event_category"].value_counts()
    total = int(total_articles if total_articles is not None else len(frame))
    distribution = {
        category: {
            "count": int(counts.get(category, 0)),
            "share": round(int(counts.get(category, 0)) / total, 6) if total else 0.0,
        }
        for category in EVENT_CATEGORIES
    }
    matched = frame["event_category"] != OTHER_CATEGORY
    return {
        "articles_classified": int(len(frame)),
        "articles_in_scope": total,
        "category_distribution": distribution,
        "unmatched_share": round(float((~matched).sum()) / total, 6) if total else 0.0,
        "articles_with_competing_categories": int((frame["competing_categories"].astype("string").fillna("") != "").sum()),
        "articles_with_excluded_evidence": int((frame["excluded_evidence"].astype("string").fillna("") != "").sum()),
        "articles_with_negated_evidence": int(frame["evidence_negated"].sum()),
        "articles_with_conditional_evidence": int(frame["conditional"].sum()),
        "class_imbalance": _class_imbalance({category: values["count"] for category, values in distribution.items()}),
    }


def _class_imbalance(counts: dict[str, int]) -> dict[str, object]:
    """Report imbalance explicitly instead of leaving it to the reader."""
    positives = {category: count for category, count in counts.items() if count > 0}
    if not positives:
        return {"classes_present": 0, "largest_class": None, "smallest_class": None, "largest_to_smallest_ratio": None}
    largest = max(positives.values())
    smallest = min(positives.values())
    return {
        "classes_present": len(positives),
        "largest_class": max(positives, key=lambda category: positives[category]),
        "smallest_class": min(positives, key=lambda category: positives[category]),
        "largest_to_smallest_ratio": round(largest / smallest, 4),
        "note": (
            "Accuracy alone is misleading under this imbalance; per-class precision/recall and balanced "
            "accuracy are reported alongside it, and rare classes may have too few examples to evaluate."
        ),
    }


__all__ = [
    "EVENT_COLUMNS",
    "SCOPES",
    "EventMatch",
    "TOKEN_PATTERN",
    "classification_input",
    "classification_summary",
    "classify_frame",
    "classify_text",
    "tokenize",
]
