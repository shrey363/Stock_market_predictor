"""CLI: build the review sheets the team labels by hand (Phase 2).

    python -m stock_project.nlp.annotate --config configs/config.example.yaml

Two sheets are produced under ``data/annotations/`` (committed, because hand
labelling is expensive work):

* ``news_review_sample.csv`` — a **random** sample, seeded and deterministic. This is
  the sheet whose labels produce the classifier's precision/recall/F1, because a
  random sample is the only kind that supports an unbiased estimate.
* ``news_review_focus.csv`` — a **deliberately stratified** sheet built from rows the
  baseline flagged (negated, conditional) and from its smallest categories, for
  qualitative review of error modes. Metrics computed on it are *not* unbiased.

Two deliberate safety properties:

* **No labels are ever invented.** The ``event_category`` column is written empty; nothing
  in this project fills it except a human reviewer.
* **Existing labels are never clobbered.** If the target sheet already contains labels,
  the command refuses to overwrite it unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from ..common.config import Config, load_config
from ..common.errors import DataNotFetchedError
from ..common.logging_utils import configure_cli_logging, get_logger
from ..common.paths import annotations_dir, ensure_directories, relative_to_root, results_dir
from ..common.timeutils import utc_now_iso
from .classify_news import classified_path
from .taxonomy import CATEGORY_DEFINITIONS, EVENT_CATEGORIES, TAXONOMY_VERSION

logger = get_logger("stock_project.annotate")

REVIEW_SAMPLE_FILENAME = "news_review_sample.csv"
REVIEW_FOCUS_FILENAME = "news_review_focus.csv"
REPORT_FILENAME = "phase2_annotation_sheets.json"

#: Columns a reviewer fills in. Everything else is context.
LABEL_COLUMNS = ["event_category", "reviewer", "reviewed_at", "notes"]

#: Context columns taken from the classified table.
CONTEXT_COLUMNS = ["article_id", "published_at", "headline", "source", "source_category"]

#: Only added with ``--show-predictions`` (off by default so labelling is blind).
PREDICTION_COLUMNS = [
    "predicted_event_category",
    "predicted_matched_keywords",
    "predicted_evidence_negated",
    "predicted_conditional",
]

SHEET_COLUMNS = [*CONTEXT_COLUMNS, "sampling_reason", *LABEL_COLUMNS]

GUIDELINES = "docs/annotation_guidelines.md"


def review_sheet_path(root: Path | str | None = None, filename: str = REVIEW_SAMPLE_FILENAME) -> Path:
    return annotations_dir(root) / filename


def _blank_labels(frame: pd.DataFrame) -> pd.DataFrame:
    for column in LABEL_COLUMNS:
        frame[column] = ""
    return frame


def build_review_sheet(
    classified: pd.DataFrame,
    *,
    size: int,
    seed: int,
    include_predictions: bool = False,
) -> pd.DataFrame:
    """A deterministic random sample of articles to label, labels left blank."""
    if size <= 0:
        raise ValueError(f"review sample size must be positive, got {size}")
    ordered = classified.sort_values(["published_at", "article_id"], kind="stable").reset_index(drop=True)
    take = min(size, len(ordered))
    indices = sorted(random.Random(seed).sample(range(len(ordered)), take))
    sheet = ordered.loc[indices, CONTEXT_COLUMNS].copy()
    sheet["sampling_reason"] = "random"
    sheet = _blank_labels(sheet)
    if include_predictions:
        predictions = ordered.loc[indices, ["event_category", "matched_keywords", "evidence_negated", "conditional"]]
        for target, source in zip(PREDICTION_COLUMNS, predictions.columns):
            sheet[target] = predictions[source].to_numpy()
        order = [*CONTEXT_COLUMNS, "sampling_reason", *PREDICTION_COLUMNS, *LABEL_COLUMNS]
        sheet = sheet[order]
    return sheet.reset_index(drop=True)


def build_focus_sheet(classified: pd.DataFrame, *, size: int, seed: int) -> pd.DataFrame:
    """A stratified sheet aimed at the baseline's known weak spots (not for metrics)."""
    if size <= 0:
        raise ValueError(f"focus sheet size must be positive, got {size}")
    ordered = classified.sort_values(["published_at", "article_id"], kind="stable").reset_index(drop=True)
    flagged_mask = ordered["evidence_negated"] | ordered["conditional"]
    counts = ordered["event_category"].value_counts()
    present = counts[counts > 0]
    if present.empty:  # pragma: no cover - defensive
        return _blank_labels(ordered.loc[:, CONTEXT_COLUMNS].head(0))
    median_size = float(present.median())
    rare_categories = [category for category, count in present.items() if count <= median_size]

    reasons: list[str] = []
    for index in range(len(ordered)):
        row_reasons = []
        if flagged_mask.iloc[index]:
            row_reasons.append("flagged_evidence")
        if ordered["event_category"].iloc[index] in rare_categories:
            row_reasons.append("small_category")
        reasons.append("+".join(row_reasons) if row_reasons else "stratified_other")

    candidate = ordered.loc[:, CONTEXT_COLUMNS].copy()
    candidate["sampling_reason"] = reasons
    candidate = candidate.replace({"sampling_reason": {"stratified_other": "context"}})
    # Flagged rows first (they carry the known error modes), then small-category rows;
    # ``random.Random(seed)`` only breaks ties so the sheet is reproducible.
    tie_break = random.Random(seed)
    candidate["_rank"] = [
        (0 if "flagged_evidence" in reason else 1 if "small_category" in reason else 2, tie_break.random())
        for reason in candidate["sampling_reason"]
    ]
    sheet = (
        candidate.sort_values("_rank", kind="stable")
        .drop(columns=["_rank"])
        .head(min(size, len(candidate)))
        .sort_values(["sampling_reason", "published_at"], kind="stable")
        .reset_index(drop=True)
    )
    return _blank_labels(sheet)


def has_labels(path: Path | str) -> bool:
    """True when a sheet exists and at least one reviewer-filled cell is non-empty."""
    target = Path(path)
    if not target.exists():
        return False
    existing = pd.read_csv(target, dtype=str).fillna("")
    for column in LABEL_COLUMNS:
        if column in existing.columns and (existing[column].str.strip() != "").any():
            return True
    return False


def annotate(
    config: Config,
    *,
    root: Path | str | None = None,
    size: int | None = None,
    focus_size: int = 60,
    include_predictions: bool = False,
    force: bool = False,
    write: bool = True,
) -> dict[str, object]:
    """Build and (optionally) write both review sheets; return the report."""
    source = classified_path(config, root=root)
    if not source.exists():
        raise DataNotFetchedError(
            f"no classified news table at {source}; run: python -m stock_project.nlp.classify_news first"
        )
    classified = pd.read_csv(source)
    resolved_size = int(size if size is not None else config.nlp.review_sample_size)

    sample = build_review_sheet(classified, size=resolved_size, seed=config.nlp.review_sample_seed, include_predictions=include_predictions)
    focus = build_focus_sheet(classified, size=focus_size, seed=config.nlp.review_sample_seed)

    sample_path = review_sheet_path(root, REVIEW_SAMPLE_FILENAME)
    focus_path = review_sheet_path(root, REVIEW_FOCUS_FILENAME)
    report: dict[str, object] = {
        "generated_at_utc": utc_now_iso(),
        "phase": 2,
        "taxonomy_version": TAXONOMY_VERSION,
        "guidelines": GUIDELINES,
        "seed": config.nlp.review_sample_seed,
        "blind_labelling": not include_predictions,
        "allowed_labels": list(EVENT_CATEGORIES),
        "category_definitions": CATEGORY_DEFINITIONS,
        "sheets": {
            "random_sample": {
                "file": relative_to_root(sample_path, root),
                "rows": int(len(sample)),
                "purpose": "unbiased estimate of classifier precision/recall/F1",
            },
            "focus_sample": {
                "file": relative_to_root(focus_path, root),
                "rows": int(len(focus)),
                "purpose": "qualitative review of flagged evidence and small classes; NOT for headline metrics",
                "sampling_reason_counts": focus["sampling_reason"].value_counts().to_dict() if len(focus) else {},
            },
        },
        "labels_written": 0,
        "notes": [
            "The event_category column is written empty: labels come only from human review.",
            "Sampling is deterministic: the same classified table + seed reproduces the same sheets.",
            "Only the random sheet supports unbiased metrics; the focus sheet is deliberately stratified.",
        ],
        "next_step": (
            f"Label the sheets by hand against {GUIDELINES}, then run: "
            "python -m stock_project.nlp.evaluate_classifier"
        ),
    }

    if write:
        ensure_directories(root)
        for target in (sample_path, focus_path):
            if has_labels(target) and not force:
                raise FileExistsError(
                    f"{relative_to_root(target, root)} already contains labels; refusing to overwrite hand "
                    "labelling. Move or rename it, or pass --force to discard it."
                )
        sample.to_csv(sample_path, index=False, lineterminator="\n")
        focus.to_csv(focus_path, index=False, lineterminator="\n")
        report_path = results_dir(root) / REPORT_FILENAME
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logger.info("wrote %s and %s", relative_to_root(sample_path, root), relative_to_root(focus_path, root))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Phase 2 hand-labelling review sheets (labels left empty).")
    parser.add_argument("--config", type=Path, default=None, help="YAML config (default: configs/config.yaml or the example)")
    parser.add_argument("--root", type=Path, default=None, help="project root to read/write (default: this repository)")
    parser.add_argument("--size", type=int, default=None, help="rows in the random review sample (default from config)")
    parser.add_argument("--focus-size", type=int, default=60, help="rows in the stratified focus sheet")
    parser.add_argument("--show-predictions", action="store_true", help="include the baseline's predictions as context (not blind)")
    parser.add_argument("--force", action="store_true", help="overwrite sheets that already contain labels")
    args = parser.parse_args(argv)

    configure_cli_logging()
    config = load_config(args.config)
    root = Path(args.root) if args.root else None
    report = annotate(
        config,
        root=root,
        size=args.size,
        focus_size=args.focus_size,
        include_predictions=args.show_predictions,
        force=args.force,
    )
    print(f"random sample sheet: {report['sheets']['random_sample']['file']} ({report['sheets']['random_sample']['rows']} rows)")
    print(f"focus sheet        : {report['sheets']['focus_sample']['file']} ({report['sheets']['focus_sample']['rows']} rows)")
    print(f"review guidelines  : {GUIDELINES}")
    print("Labels are left empty on purpose — a human reviewer fills them in.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
