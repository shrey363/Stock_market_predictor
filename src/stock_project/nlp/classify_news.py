"""CLI: classify and score the cleaned news table (Phase 2).

    python -m stock_project.nlp.classify_news --config configs/config.example.yaml

Reads the Phase 1 cleaned table (``data/processed/news_*_clean.csv``), runs the
keyword/rule event classifier and the lexicon sentiment scorer, writes the
traceable classified table to ``data/processed/news_classified_*.csv`` and the
Phase 2 report to ``reports/experiment_results/phase2_classification_report.json``.

Traceability is the point: every row keeps ``article_id``, ``published_at``,
``available_from``, ``headline``, ``source_url``, and ``source_category``, next to the
assigned category, the exact matched keywords, and the sentiment terms — so any
number in the dashboard can be walked back to a headline (AGENTS.md §4 invariant 4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..common.config import Config, load_config
from ..common.errors import DataNotFetchedError
from ..common.logging_utils import configure_cli_logging, get_logger
from ..common.manifest import make_dataset_entry, write_manifest
from ..common.paths import ensure_directories, processed_dir, relative_to_root, results_dir
from ..common.timeutils import utc_now_iso
from ..data.inspect_data import news_processed_path
from ..data.news_data import dataset_slug
from .event_classifier import classification_input, classification_summary, classify_frame
from .sentiment import SENTIMENT_COLUMNS, score_frame, sentiment_summary
from .taxonomy import EVENT_CATEGORIES, OTHER_CATEGORY, TAXONOMY_VERSION

logger = get_logger("stock_project.classify_news")

REPORT_FILENAME = "phase2_classification_report.json"

#: Final contract of the classified news table.
CLASSIFIED_COLUMNS = [
    "article_id",
    "published_at",
    "available_from",
    "headline",
    "article_text",
    "source",
    "source_url",
    "source_category",
    "event_category",
    "event_priority",
    "matched_keywords",
    "competing_categories",
    "excluded_evidence",
    "evidence_negated",
    "conditional",
    "classified_scope",
    "matched_term_count",
    "sentiment_label",
    "sentiment_score",
    "sentiment_terms",
    "affected_sector",
]

#: Columns that must survive classification untouched (traceability check).
TRACEABILITY_COLUMNS = ["article_id", "published_at", "available_from", "headline", "source_url", "source_category"]

#: How many illustrative rows to keep per category in the report.
EXAMPLES_PER_CATEGORY = 4


def classified_path(config: Config, *, root: Path | str | None = None) -> Path:
    slug = dataset_slug(config.news.dataset_id)
    label = config.news.source_category_filter.strip().lower().replace(" ", "_")
    return processed_dir(root) / f"news_classified_{slug}_{label}.csv"


def build_classified_table(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Classify + score a cleaned news table and return the documented contract."""
    for column in TRACEABILITY_COLUMNS:
        if column not in frame.columns:
            raise ValueError(f"cleaned news table is missing the traceability column {column!r}")

    classified = classify_frame(frame, text_scope=config.nlp.text_scope)

    # Sentiment is scored on exactly the text that was classified (so the two components
    # see the same input), replacing the Phase 1 placeholder columns, which are all missing.
    scope_text = pd.DataFrame(
        {
            "headline": [
                classification_input(
                    row,
                    text_scope=config.nlp.text_scope,
                    headline_column="headline",
                    description_column="article_text",
                )
                for _, row in frame.iterrows()
            ]
        },
        index=frame.index,
    )
    sentiment = score_frame(scope_text, column="headline")
    for column in SENTIMENT_COLUMNS:
        classified[column] = sentiment[column]

    if "affected_sector" not in classified.columns:
        # Spec §6 lists it as optional/manual; Phase 2 does not assign sectors.
        classified["affected_sector"] = pd.NA
    return classified[CLASSIFIED_COLUMNS]


def _examples(frame: pd.DataFrame, mask: pd.Series, *, limit: int) -> list[dict[str, object]]:
    subset = frame.loc[mask].head(limit)
    return [
        {
            "article_id": row["article_id"],
            "published_at": str(row["published_at"]),
            "event_category": row["event_category"],
            "matched_keywords": row["matched_keywords"],
            "competing_categories": row["competing_categories"],
            "excluded_evidence": row["excluded_evidence"],
            "sentiment_label": row["sentiment_label"],
            "headline": row["headline"],
        }
        for _, row in subset.iterrows()
    ]


def classification_examples(frame: pd.DataFrame, *, per_category: int = EXAMPLES_PER_CATEGORY) -> dict[str, object]:
    """Concrete rows behind the counts, for review and for the misclassification pass."""
    by_category = {
        category: _examples(frame, frame["event_category"] == category, limit=per_category)
        for category in EVENT_CATEGORIES
    }
    return {
        "per_category": by_category,
        "flagged_negated": _examples(frame, frame["evidence_negated"], limit=6),
        "flagged_conditional": _examples(frame, frame["conditional"], limit=6),
        "competing_categories": _examples(frame, frame["competing_categories"].astype("string").fillna("") != "", limit=6),
        "suppressed_evidence": _examples(frame, frame["excluded_evidence"].astype("string").fillna("") != "", limit=6),
    }


def classify_datasets(config: Config, *, root: Path | str | None = None, write: bool = True) -> dict[str, object]:
    """Classify the cleaned news table, write the artifact, and return the report."""
    source = news_processed_path(config, root=root)
    if not source.exists():
        raise DataNotFetchedError(
            f"no cleaned news table at {source}; run: python -m stock_project.data.inspect_data "
            "(after fetch_data) first"
        )

    cleaned = pd.read_csv(source)
    cleaned["published_at"] = pd.to_datetime(cleaned["published_at"], utc=True, errors="coerce")
    if "available_from" in cleaned.columns:
        cleaned["available_from"] = pd.to_datetime(cleaned["available_from"], utc=True, errors="coerce")

    classified = build_classified_table(cleaned, config)
    if len(classified) != len(cleaned):
        raise AssertionError("classification must not add or drop articles")
    empty_headlines = int(classified["headline"].astype("string").fillna("").str.strip().eq("").sum())

    report: dict[str, object] = {
        "generated_at_utc": utc_now_iso(),
        "phase": 2,
        "phase_name": "News classification and sentiment",
        "config": config.to_dict(),
        "inputs": {
            "cleaned_news_file": relative_to_root(source, root),
            "articles": int(len(cleaned)),
            "empty_headlines": empty_headlines,
        },
        "method": {
            "taxonomy_version": TAXONOMY_VERSION,
            "configured_taxonomy_version": config.nlp.taxonomy_version,
            "event_classifier": "transparent keyword/rule baseline; priority order decides ties",
            "exclusion_phrases_applied": True,
            "text_scope": config.nlp.text_scope,
            "sentiment_mode": config.nlp.sentiment_mode,
            "sentiment_scorer": "lexicon with negation handling, score in [-1, 1]",
            "sentiment_input": f"same text as classification ({config.nlp.text_scope})",
            "event_categories": list(EVENT_CATEGORIES),
        },
        "classification": classification_summary(classified),
        "sentiment": sentiment_summary(classified),
        "traceability": {
            "columns_preserved": TRACEABILITY_COLUMNS,
            "matched_keywords_recorded": True,
            "output_file": relative_to_root(classified_path(config, root=root), root),
        },
        "examples": classification_examples(classified),
        "limitations": [
            "Headline-scope classification by default: the headline is the only input, so events "
            "described only in the body text are missed (a documented Phase 2 limitation).",
            "The source is general-interest political news; a high other_or_uncertain share is expected "
            "and is not a bug.",
            "A keyword match is not proof an event happened: negation, threats, and speculation are only "
            "flagged (evidence_negated / conditional), not resolved.",
            "Sentiment is text tone, not severity or expected market impact.",
            "affected_sector is not assigned in Phase 2 (optional, manual field in the specification).",
            "Predictions have not been compared with human labels yet: run the review sheet through "
            "evaluate_classifier to obtain precision/recall/F1.",
            "The classifier has no notion of past vs new events; an anniversary headline reads the same "
            "as a fresh one.",
        ],
        "next_step": (
            "Label data/annotations/news_review_sample.csv by hand (see docs/annotation_guidelines.md), "
            "then run: python -m stock_project.nlp.evaluate_classifier"
        ),
    }
    if config.nlp.taxonomy_version != TAXONOMY_VERSION:
        report["method"]["taxonomy_version_mismatch"] = (
            f"config declares {config.nlp.taxonomy_version!r} but the code implements {TAXONOMY_VERSION!r}; "
            "both are recorded so results stay comparable."
        )

    if write:
        ensure_directories(root)
        target = classified_path(config, root=root)
        classified.to_csv(target, index=False, lineterminator="\n")
        write_manifest(
            root,
            key=f"classified:news:{config.news.dataset_id}",
            entry=make_dataset_entry(
                key=f"classified:news:{config.news.dataset_id}",
                name=f"classified + scored news ({config.news.source_category_filter})",
                file_path=target,
                root=root,
                source=config.news.dataset_url,
                dataset_id=config.news.dataset_id,
                revision=config.news.revision,
                license=config.news.license,
                license_url=config.news.license_url,
                rows=int(len(classified)),
                columns=list(classified.columns),
                date_coverage=(
                    str(classified["published_at"].min().date()),
                    str(classified["published_at"].max().date()),
                ),
                notes=[
                    "Derived by nlp.classify_news from the Phase 1 cleaned table.",
                    f"taxonomy_version={TAXONOMY_VERSION}; text_scope={config.nlp.text_scope}; "
                    f"sentiment_mode={config.nlp.sentiment_mode}.",
                    "Labels are machine-generated baseline outputs, not human judgements.",
                ],
            ),
        )
        report_path = results_dir(root) / REPORT_FILENAME
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logger.info("wrote %s", relative_to_root(report_path, root))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify events and score sentiment for the cleaned news table.")
    parser.add_argument("--config", type=Path, default=None, help="YAML config (default: configs/config.yaml or the example)")
    parser.add_argument("--root", type=Path, default=None, help="project root to read/write (default: this repository)")
    parser.add_argument("--print-json", action="store_true", help="also print the full report to stdout")
    args = parser.parse_args(argv)

    configure_cli_logging()
    config = load_config(args.config)
    root = Path(args.root) if args.root else None
    report = classify_datasets(config, root=root)

    distribution = report["classification"]["category_distribution"]
    print(f"articles classified: {report['inputs']['articles']} (scope: {config.nlp.text_scope})")
    for category in EVENT_CATEGORIES:
        entry = distribution[category]
        print(f"  {category:<28} {entry['count']:>6}  ({entry['share'] * 100:.1f}%)")
    imbalance = report["classification"]["class_imbalance"]
    print(f"unmatched share: {report['classification']['unmatched_share'] * 100:.1f}%")
    print(f"largest/smallest ratio: {imbalance['largest_to_smallest_ratio']} across {imbalance['classes_present']} classes")
    print(f"sentiment: {report['sentiment']['label_distribution']}")
    print(f"report: {relative_to_root(results_dir(root) / REPORT_FILENAME, root)}")
    if args.print_json:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
