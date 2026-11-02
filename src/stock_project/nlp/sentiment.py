"""Lexicon sentiment baseline — the second Phase 2 component.

Why a lexicon and not FinBERT first: a small, auditable baseline with no model
weights is reproducible offline, costs nothing to run, and is honest about what it
is. The config keeps ``sentiment_mode`` as a switch; ``finbert`` is a documented,
not-yet-wired option (see :func:`get_sentiment_scorer`).

What the score means: **tone of the text**, on ``[-1, 1]``, computed as
``(positive_terms - negative_terms) / (positive_terms + negative_terms)`` after
negation handling. It is *not* severity, event magnitude, or expected market impact —
those must be evaluated, never inferred from tone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .taxonomy import SENTIMENT_NEGATIVE, SENTIMENT_POSITIVE

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

#: Terms that flip the polarity of a following sentiment term within the window.
NEGATION_TERMS: tuple[str, ...] = ("no", "not", "never", "without", "denies", "denied", "fail", "fails", "failed")

#: How many tokens before a sentiment term are searched for a negation.
NEGATION_WINDOW = 3

#: Columns this module adds to a news table.
SENTIMENT_COLUMNS = ["sentiment_label", "sentiment_score", "sentiment_terms"]


def _term_list(terms: tuple[str, ...]) -> list[tuple[str, tuple[str, ...]]]:
    return [(term, tuple(TOKEN_PATTERN.findall(term.lower()))) for term in terms]


_POSITIVE = _term_list(SENTIMENT_POSITIVE)
_NEGATIVE = _term_list(SENTIMENT_NEGATIVE)


@dataclass(frozen=True)
class SentimentResult:
    """One sentiment decision with the terms that produced it."""

    label: str
    score: float
    positive_terms: tuple[str, ...]
    negative_terms: tuple[str, ...]

    @property
    def term_count(self) -> int:
        return len(self.positive_terms) + len(self.negative_terms)

    def terms_string(self) -> str:
        entries = [f"{term}:+1" for term in self.positive_terms] + [f"{term}:-1" for term in self.negative_terms]
        return "; ".join(entries)

    def to_row(self) -> dict[str, object]:
        return {
            "sentiment_label": self.label,
            "sentiment_score": self.score,
            "sentiment_terms": self.terms_string(),
        }


def _find_hits(tokens: list[str], term_tokens: tuple[str, ...]) -> list[int]:
    size = len(term_tokens)
    if size == 0 or size > len(tokens):
        return []
    return [index for index in range(len(tokens) - size + 1) if tuple(tokens[index : index + size]) == term_tokens]


def score_text(text: str | None) -> SentimentResult:
    """Score one piece of text, flipping terms that follow a negation."""
    tokens = TOKEN_PATTERN.findall(str(text).lower()) if text is not None else []
    positive: list[str] = []
    negative: list[str] = []
    for terms, bucket, flipped_bucket in ((_POSITIVE, positive, negative), (_NEGATIVE, negative, positive)):
        for term, term_tokens in terms:
            for start in _find_hits(tokens, term_tokens):
                window = tokens[max(0, start - NEGATION_WINDOW) : start]
                target = flipped_bucket if any(token in NEGATION_TERMS for token in window) else bucket
                target.append(term)

    total = len(positive) + len(negative)
    score = round((len(positive) - len(negative)) / total, 6) if total else 0.0
    label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    return SentimentResult(
        label=label,
        score=score,
        positive_terms=tuple(positive),
        negative_terms=tuple(negative),
    )


def score_frame(frame: pd.DataFrame, *, column: str = "headline") -> pd.DataFrame:
    """Return a copy of ``frame`` with the sentiment columns added."""
    if column not in frame.columns:
        raise ValueError(f"news table has no {column!r} column; columns are {list(frame.columns)}")
    working = frame.copy()
    results = [score_text(value).to_row() for value in working[column]]
    additions = pd.DataFrame(results, index=working.index)
    for name in SENTIMENT_COLUMNS:
        working[name] = additions[name]
    return working


def get_sentiment_scorer(mode: str) -> Callable[[str | None], SentimentResult]:
    """Return the scorer for ``mode``.

    Only ``lexicon`` is implemented. ``finbert`` is a planned upgrade: it needs the
    ``transformers``/``torch`` stack (a large dependency) and a network download of
    model weights, so it is deliberately not a default and not wired up yet. Asking
    for it fails loudly instead of silently falling back to the lexicon.
    """
    if mode == "lexicon":
        return score_text
    if mode == "finbert":
        raise NotImplementedError(
            "sentiment_mode 'finbert' is not implemented yet. Use 'lexicon' for now; wiring FinBERT "
            "adds transformers+torch and downloads model weights (see docs/annotation_guidelines.md)."
        )
    raise ValueError(f"unknown sentiment_mode {mode!r}; expected 'lexicon' or 'finbert'")


def sentiment_summary(frame: pd.DataFrame) -> dict[str, object]:
    """Label distribution and score statistics for the artifact."""
    if "sentiment_label" not in frame.columns:
        raise ValueError("frame has no sentiment columns; run score_frame first")
    counts = frame["sentiment_label"].value_counts()
    total = int(len(frame))
    return {
        "label_distribution": {
            label: {
                "count": int(counts.get(label, 0)),
                "share": round(int(counts.get(label, 0)) / total, 6) if total else 0.0,
            }
            for label in ("positive", "neutral", "negative")
        },
        "mean_score": round(float(frame["sentiment_score"].mean()), 6) if total else 0.0,
        "articles_with_no_term": int((frame["sentiment_terms"].astype("string").fillna("") == "").sum()),
        "note": "Score is text tone, not severity or expected market impact.",
    }


__all__ = [
    "NEGATION_TERMS",
    "NEGATION_WINDOW",
    "SENTIMENT_COLUMNS",
    "SentimentResult",
    "get_sentiment_scorer",
    "score_frame",
    "score_text",
    "sentiment_summary",
]
