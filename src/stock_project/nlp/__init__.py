"""NLP layer (Phase 2): transparent event classification and sentiment baselines.

Both components are deliberately simple, auditable rule/lexicon systems. Their
outputs are traceable back to the exact matched terms, and every article keeps its
``article_id``, headline, timestamp, and source (AGENTS.md §4 invariant 4).
"""

from __future__ import annotations

__all__ = ["annotate", "classify_news", "event_classifier", "evaluate_classifier", "sentiment", "taxonomy"]
