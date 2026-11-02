"""CLI: evaluate the Phase 2 baseline against the hand-labelled review sheet.

    python -m stock_project.nlp.evaluate_classifier --config configs/config.example.yaml

Reads the labels a human wrote into ``data/annotations/news_review_sample.csv``,
joins them to the classifier's predictions, and reports per-class
precision/recall/F1, accuracy, balanced accuracy, Cohen's kappa, the confusion
matrix, a naive majority-class baseline, and an explicit class-imbalance picture —
plus a misclassification listing for the error review the specification requires.

Honesty rules enforced here:

* Unreviewed rows (blank labels) are counted and excluded, never treated as correct.
* A label outside the taxonomy is a hard error, so typos cannot silently change metrics.
* Metrics from a stratified sheet are labelled as biased wherever they appear.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ..common.config import Config, load_config
from ..common.errors import DataNotFetchedError
from ..common.logging_utils import configure_cli_logging, get_logger
from ..common.paths import ensure_directories, relative_to_root, results_dir
from ..common.timeutils import utc_now_iso
from .annotate import REVIEW_FOCUS_FILENAME, REVIEW_SAMPLE_FILENAME, review_sheet_path
from .classify_news import classified_path
from .taxonomy import EVENT_CATEGORIES, TAXONOMY_VERSION

logger = get_logger("stock_project.evaluate_classifier")

EVALUATION_FILENAME = "phase2_classifier_evaluation.json"
MISCLASSIFICATIONS_FILENAME = "phase2_misclassifications.csv"

LABEL_COLUMN = "event_category"
OUTPUT_COLUMNS = [
    "article_id",
    "published_at",
    "headline",
    "true_event_category",
    "predicted_event_category",
    "matched_keywords",
    "competing_categories",
    "evidence_negated",
    "conditional",
    "sentiment_label",
]


def load_labelled_sheet(path: Path | str, *, labels: tuple[str, ...] = EVENT_CATEGORIES) -> tuple[pd.DataFrame, dict]:
    """Load a review sheet, returning ``(labelled_rows, stats)``.

    Blank labels are simply unreviewed. Anything else must be a taxonomy label.
    """
    target = Path(path)
    if not target.exists():
        raise DataNotFetchedError(f"no review sheet at {target}; run: python -m stock_project.nlp.annotate")
    sheet = pd.read_csv(target, dtype=str).fillna("")
    if "article_id" not in sheet.columns or LABEL_COLUMN not in sheet.columns:
        raise ValueError(f"review sheet {target} must contain 'article_id' and {LABEL_COLUMN!r} columns")

    normalised = sheet[LABEL_COLUMN].str.strip().str.lower()
    invalid_mask = (normalised != "") & (~normalised.isin(labels))
    if invalid_mask.any():
        offenders = sheet.loc[invalid_mask, ["article_id", LABEL_COLUMN]].head(10).to_dict("records")
        raise ValueError(
            f"{int(invalid_mask.sum())} row(s) in {target.name} carry a label outside the taxonomy "
            f"{list(labels)}: {offenders}. Fix the sheet (allowed values are listed in the header notes "
            "and docs/annotation_guidelines.md)."
        )

    sheet = sheet.assign(**{LABEL_COLUMN: normalised})
    labelled = sheet.loc[sheet[LABEL_COLUMN] != ""].copy()
    resolved_columns = [column for column in ("article_id", LABEL_COLUMN, "reviewer", "reviewed_at") if column in sheet.columns]
    stats = {
        "sheet": str(target),
        "sheet_name": target.name,
        "rows_in_sheet": int(len(sheet)),
        "labelled_rows": int(len(labelled)),
        "unreviewed_rows": int(len(sheet) - len(labelled)),
        "reviewers": sorted({value for value in sheet.get("reviewer", pd.Series(dtype=str)) if value}),
        "columns_used": resolved_columns,
    }
    return labelled[[c for c in ("article_id", LABEL_COLUMN) if c in labelled.columns]], stats


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, *, labels: tuple[str, ...] = EVENT_CATEGORIES) -> dict[str, object]:
    """Per-class and aggregate metrics. Hand-checkable on tiny inputs."""
    label_list = list(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_list, zero_division=0.0
    )
    # Macro averages are taken over the labels the sample actually contains: averaging in a
    # class nobody labelled would report a failure for something that cannot be evaluated.
    present_labels = [label for index, label in enumerate(label_list) if support[index] > 0]
    if present_labels:
        macro = precision_recall_fscore_support(
            y_true, y_pred, labels=present_labels, average="macro", zero_division=0.0
        )
    else:  # pragma: no cover - guarded by the zero-label path
        macro = (0.0, 0.0, 0.0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=label_list, average="weighted", zero_division=0.0)
    predicted_counts = Counter(y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=label_list)

    per_class = {
        label: {
            "precision": round(float(precision[index]), 6),
            "recall": round(float(recall[index]), 6),
            "f1": round(float(f1[index]), 6),
            "support": int(support[index]),
            "predicted": int(predicted_counts.get(label, 0)),
        }
        for index, label in enumerate(label_list)
    }
    # Balanced accuracy = macro recall over the labels present in the sample. Computed here rather
    # than taken from sklearn so that classes present in the predictions but never labelled cannot
    # quietly change the denominator.
    present_recall = [float(recall[index]) for index, label in enumerate(label_list) if label in present_labels]
    balanced_accuracy = float(np.mean(present_recall)) if present_recall else 0.0

    return {
        "n_evaluated": int(len(y_true)),
        "per_class": per_class,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": round(balanced_accuracy, 6),
        "cohen_kappa": round(float(cohen_kappa_score(y_true, y_pred, labels=label_list)), 6),
        "macro_average": {
            "precision": round(float(macro[0]), 6),
            "recall": round(float(macro[1]), 6),
            "f1": round(float(macro[2]), 6),
            "labels_used": present_labels,
            "labels_excluded_no_support": [label for label in label_list if label not in present_labels],
        },
        "weighted_average": {
            "precision": round(float(weighted[0]), 6),
            "recall": round(float(weighted[1]), 6),
            "f1": round(float(weighted[2]), 6),
        },
        "confusion_matrix": {
            "rows": "true label",
            "columns": "predicted label",
            "labels": label_list,
            "matrix": matrix.tolist(),
        },
        "classes_without_support": [label for label in label_list if per_class[label]["support"] == 0],
        "classes_never_predicted": [label for label in label_list if per_class[label]["predicted"] == 0],
        "notes": [
            "Accuracy is reported alongside balanced accuracy because the classes are imbalanced.",
            "Classes with zero support cannot be evaluated: their precision/recall are shown as 0.0 and must "
            "not be read as failures.",
            "Balanced accuracy is the mean recall over the labels present in the sample (macro recall).",
            f"Macro averages use only the {len(present_labels)} label(s) present in this sample: "
            f"{present_labels}.",
        ],
    }


def naive_majority_baseline(y_true: pd.Series) -> dict[str, object]:
    """The always-predict-the-most-common-class baseline the model must beat."""
    counts = Counter(y_true)
    if not counts:  # pragma: no cover - guarded by callers
        return {"class": None, "accuracy": None}
    majority, count = counts.most_common(1)[0]
    return {
        "class": majority,
        "accuracy": round(count / len(y_true), 6),
        "note": "Under imbalance this baseline can look strong; compare macro/balanced metrics, not just accuracy.",
    }


def error_patterns(errors: pd.DataFrame) -> dict[str, object]:
    """Where the baseline goes wrong, in aggregate."""
    if errors.empty:
        return {"error_count": 0, "top_confusions": [], "errors_with_negated_evidence": 0, "errors_with_conditional_evidence": 0}
    pairs = Counter(zip(errors["true_event_category"], errors["predicted_event_category"]))
    return {
        "error_count": int(len(errors)),
        "top_confusions": [
            {"true": true, "predicted": predicted, "count": count}
            for (true, predicted), count in pairs.most_common(8)
        ],
        "errors_with_negated_evidence": int(
            errors["evidence_negated"].astype(str).str.lower().isin(["true", "1"]).sum()
        ),
        "errors_with_conditional_evidence": int(
            errors["conditional"].astype(str).str.lower().isin(["true", "1"]).sum()
        ),
        "errors_where_baseline_found_no_keyword": int(errors["predicted_event_category"].eq("other_or_uncertain").sum()),
    }


def _evaluate_sheet(
    sheet_path: Path,
    classified: pd.DataFrame,
    *,
    root: Path | str | None,
    write_errors_to: Path | None,
) -> tuple[dict[str, object], pd.DataFrame]:
    labelled, stats = load_labelled_sheet(sheet_path)
    if stats["labelled_rows"] == 0:
        # Nothing to score yet. Report it plainly rather than computing metrics on an empty set.
        return (
            {
                "sheet": relative_to_root(sheet_path, root),
                "rows_in_sheet": stats["rows_in_sheet"],
                "labelled_rows": 0,
                "unreviewed_rows": stats["unreviewed_rows"],
                "reviewers": stats["reviewers"],
                "metrics": None,
                "naive_majority_baseline": None,
                "error_patterns": None,
                "note": "No labels yet: the sheet is empty until a human reviews it. No metrics were computed.",
            },
            pd.DataFrame(columns=OUTPUT_COLUMNS),
        )

    predictions = classified[
        [
            "article_id",
            "published_at",
            "headline",
            "event_category",
            "matched_keywords",
            "competing_categories",
            "evidence_negated",
            "conditional",
            "sentiment_label",
        ]
    ].rename(columns={"event_category": "predicted_event_category"})

    merged = labelled.rename(columns={LABEL_COLUMN: "true_event_category"}).merge(
        predictions, on="article_id", how="left", validate="one_to_one"
    )
    unresolved = merged["predicted_event_category"].isna()
    if unresolved.any():
        missing = merged.loc[unresolved, "article_id"].head(5).tolist()
        raise ValueError(
            f"{int(unresolved.sum())} labelled article(s) from {Path(sheet_path).name} are not in the "
            f"classified table (first ids: {missing}). Re-run classify_news after regenerating the sheet."
        )

    metrics = compute_metrics(merged["true_event_category"], merged["predicted_event_category"])
    errors = merged.loc[merged["true_event_category"] != merged["predicted_event_category"], OUTPUT_COLUMNS]
    errors = errors.sort_values(["true_event_category", "predicted_event_category"], kind="stable")
    if write_errors_to is not None:
        errors.to_csv(write_errors_to, index=False, lineterminator="\n")

    payload: dict[str, object] = {
        "sheet": relative_to_root(sheet_path, root),
        "rows_in_sheet": stats["rows_in_sheet"],
        "labelled_rows": stats["labelled_rows"],
        "unreviewed_rows": stats["unreviewed_rows"],
        "reviewers": stats["reviewers"],
        "observed_label_distribution": merged["true_event_category"].value_counts().to_dict(),
        "metrics": metrics,
        "naive_majority_baseline": naive_majority_baseline(merged["true_event_category"]),
        "error_patterns": error_patterns(errors),
    }
    return payload, errors


def evaluate_classifier(
    config: Config,
    *,
    root: Path | str | None = None,
    sheet_path: Path | str | None = None,
    write: bool = True,
) -> dict[str, object]:
    """Evaluate the baseline on the labelled sheets and write the artifacts."""
    classified_file = classified_path(config, root=root)
    if not classified_file.exists():
        raise DataNotFetchedError(
            f"no classified news table at {classified_file}; run: python -m stock_project.nlp.classify_news first"
        )
    classified = pd.read_csv(classified_file)

    primary_sheet = Path(sheet_path) if sheet_path is not None else review_sheet_path(root, REVIEW_SAMPLE_FILENAME)
    misclassification_path = results_dir(root) / MISCLASSIFICATIONS_FILENAME
    if write:
        ensure_directories(root)
    primary, errors = _evaluate_sheet(
        primary_sheet, classified, root=root, write_errors_to=misclassification_path if write else None
    )

    focus_sheet = review_sheet_path(root, REVIEW_FOCUS_FILENAME)
    additional: dict[str, object] = {}
    if sheet_path is None and focus_sheet.exists():
        focus_labelled, focus_stats = load_labelled_sheet(focus_sheet)
        if focus_stats["labelled_rows"]:
            focus_payload, _ = _evaluate_sheet(focus_sheet, classified, root=root, write_errors_to=None)
            focus_payload["warning"] = (
                "This sheet is deliberately stratified (flagged evidence and small classes), so its metrics "
                "are biased and must not be compared against the random sample's."
            )
            additional["focus_sheet"] = focus_payload
        else:
            additional["focus_sheet"] = {"sheet": relative_to_root(focus_sheet, root), "labelled_rows": 0, "note": "not labelled yet"}

    if primary.get("labelled_rows"):
        next_step = "Phase 3 - market features and the market-only baseline, using these labels where they help."
    else:
        next_step = (
            f"Label data/annotations/{REVIEW_SAMPLE_FILENAME} by hand (rules: docs/annotation_guidelines.md), "
            "then re-run this command to compute precision/recall/F1. Phase 3 comes after that."
        )

    report: dict[str, object] = {
        "generated_at_utc": utc_now_iso(),
        "phase": 2,
        "taxonomy_version": TAXONOMY_VERSION,
        "config": config.to_dict(),
        "inputs": {
            "classified_news_file": relative_to_root(classified_file, root),
            "primary_sheet": primary["sheet"],
        },
        "primary_sheet": primary,
        "additional_sheets": additional,
        "misclassifications_file": (
            relative_to_root(misclassification_path, root) if write and primary.get("labelled_rows") else None
        ),
        "limitations": [
            "Metrics describe this labelled sample only; they are not an estimate for future articles.",
            f"{primary['unreviewed_rows']} row(s) in the sheet are unreviewed and were excluded, not counted as correct.",
            "Labelled articles come from one publisher category, so the sample is not representative of all news.",
            "Classification uses the headline (or headline+description) only; body text is not available in this dataset.",
            "Per-class numbers for classes with few labels are unstable; treat small-support rows as indicative only.",
            "This is a baseline, not a validated model: a result close to the naive baseline is a valid outcome to report.",
        ],
        "next_step": next_step,
    }

    if write:
        report_path = results_dir(root) / EVALUATION_FILENAME
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logger.info("wrote %s", relative_to_root(report_path, root))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the Phase 2 classifier against hand-labelled review sheets.")
    parser.add_argument("--config", type=Path, default=None, help="YAML config (default: configs/config.yaml or the example)")
    parser.add_argument("--root", type=Path, default=None, help="project root to read/write (default: this repository)")
    parser.add_argument("--sheet", type=Path, default=None, help="review sheet to evaluate (default: the random sample sheet)")
    parser.add_argument("--print-json", action="store_true", help="also print the full report to stdout")
    args = parser.parse_args(argv)

    configure_cli_logging()
    config = load_config(args.config)
    root = Path(args.root) if args.root else None
    report = evaluate_classifier(config, root=root, sheet_path=args.sheet)

    primary = report["primary_sheet"]
    metrics = primary["metrics"]
    print(f"sheet             : {primary['sheet']}")
    print(f"labelled rows     : {primary['labelled_rows']} (unreviewed: {primary['unreviewed_rows']})")
    if not primary["labelled_rows"]:
        print("\nNo labels yet — the sheets are empty by design until a human reviews them.")
        print(report["next_step"])
        return 0
    print(f"accuracy          : {metrics['accuracy']}")
    print(f"balanced accuracy : {metrics['balanced_accuracy']}")
    print(f"macro F1          : {metrics['macro_average']['f1']}")
    print(f"cohen kappa       : {metrics['cohen_kappa']}")
    print(f"naive majority    : {report['primary_sheet']['naive_majority_baseline']}")
    print(f"errors            : {primary['error_patterns']['error_count']} -> {report['misclassifications_file']}")
    if args.print_json:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
