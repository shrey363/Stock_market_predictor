"""Classifier evaluation: hand-computed metrics, label validation, error listing.

The numbers asserted here are computed by hand from the tiny fixtures, so a change in
the evaluation code cannot silently move them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stock_project.common.config import Config, load_config
from stock_project.common.errors import DataNotFetchedError
from stock_project.data.inspect_data import news_processed_path
from stock_project.nlp.annotate import REVIEW_SAMPLE_FILENAME, review_sheet_path
from stock_project.nlp.classify_news import classify_datasets, classified_path
from stock_project.nlp.evaluate_classifier import (
    EVALUATION_FILENAME,
    MISCLASSIFICATIONS_FILENAME,
    compute_metrics,
    error_patterns,
    load_labelled_sheet,
    main,
    naive_majority_baseline,
)
from stock_project.preprocessing.clean_news_data import clean_news_data

# Hand-checked example:
#   true : armed_conflict, sanctions, sanctions, other_or_uncertain
#   pred : armed_conflict, sanctions, other_or_uncertain, other_or_uncertain
#   accuracy = 3/4 = 0.75; recalls = 1.0, 0.5, 1.0 -> balanced accuracy = 2.5/3
#   sanctions: precision 1.0, recall 0.5, f1 0.667; other: precision 0.5, recall 1.0
#   kappa: p_o = 0.75, p_e = 5/16 = 0.3125 -> (0.75 - 0.3125) / (1 - 0.3125) = 0.636364
Y_TRUE = pd.Series(["armed_conflict", "sanctions", "sanctions", "other_or_uncertain"])
Y_PRED = pd.Series(["armed_conflict", "sanctions", "other_or_uncertain", "other_or_uncertain"])


def test_metrics_match_hand_computed_values():
    metrics = compute_metrics(Y_TRUE, Y_PRED)
    assert metrics["n_evaluated"] == 4
    assert metrics["accuracy"] == 0.75
    assert metrics["balanced_accuracy"] == pytest.approx(0.833333, abs=1e-6)
    assert metrics["cohen_kappa"] == pytest.approx(0.636364, abs=1e-6)
    assert metrics["per_class"]["sanctions"] == pytest.approx(
        {"precision": 1.0, "recall": 0.5, "f1": 0.666667, "support": 2, "predicted": 1}, abs=1e-6
    )
    assert metrics["per_class"]["other_or_uncertain"]["precision"] == 0.5
    assert metrics["per_class"]["trade_restriction"]["support"] == 0


def test_metrics_are_zeroed_for_absent_classes_without_crashing():
    metrics = compute_metrics(Y_TRUE, Y_PRED)
    assert metrics["per_class"]["trade_restriction"]["precision"] == 0.0
    assert set(metrics["classes_without_support"]) == {
        "trade_restriction",
        "energy_or_supply_disruption",
        "diplomatic_development",
    }
    assert metrics["classes_never_predicted"] == metrics["classes_without_support"]
    assert any("must not be read as failures" in note for note in metrics["notes"])


def test_confusion_matrix_is_laid_out_as_documented():
    metrics = compute_metrics(Y_TRUE, Y_PRED)
    labels = metrics["confusion_matrix"]["labels"]
    matrix = metrics["confusion_matrix"]["matrix"]
    assert metrics["confusion_matrix"]["rows"] == "true label"
    assert len(matrix) == len(labels)
    sanctions_index = labels.index("sanctions")
    other_index = labels.index("other_or_uncertain")
    assert matrix[sanctions_index][sanctions_index] == 1
    assert matrix[sanctions_index][other_index] == 1


def test_perfect_agreement_is_reported_as_such():
    metrics = compute_metrics(Y_TRUE, Y_TRUE)
    assert metrics["accuracy"] == 1.0
    assert metrics["cohen_kappa"] == 1.0
    assert metrics["macro_average"]["f1"] == pytest.approx(1.0, abs=1e-6)


def test_naive_baseline_uses_the_majority_class():
    baseline = naive_majority_baseline(Y_TRUE)
    assert baseline["class"] == "sanctions"
    assert baseline["accuracy"] == 0.5


def test_error_patterns_summarise_confusions():
    errors = pd.DataFrame(
        [
            {
                "true_event_category": "sanctions",
                "predicted_event_category": "other_or_uncertain",
                "evidence_negated": True,
                "conditional": False,
            },
            {
                "true_event_category": "sanctions",
                "predicted_event_category": "other_or_uncertain",
                "evidence_negated": False,
                "conditional": True,
            },
        ]
    )
    patterns = error_patterns(errors)
    assert patterns["error_count"] == 2
    assert patterns["top_confusions"][0] == {"true": "sanctions", "predicted": "other_or_uncertain", "count": 2}
    assert patterns["errors_with_negated_evidence"] == 1
    assert patterns["errors_where_baseline_found_no_keyword"] == 2


def test_error_patterns_handle_no_errors():
    assert error_patterns(pd.DataFrame(columns=["true_event_category"])).get("error_count") == 0


def test_load_labelled_sheet_counts_blank_rows_as_unreviewed(tmp_path: Path):
    sheet = tmp_path / "sheet.csv"
    sheet.write_text(
        "article_id,event_category,reviewer\n"
        "a1,sanctions,reviewer-1\n"
        "a2,,reviewer-1\n"
        "a3,ARMED_CONFLICT,reviewer-2\n",
        encoding="utf-8",
    )
    labelled, stats = load_labelled_sheet(sheet)
    assert stats["rows_in_sheet"] == 3
    assert stats["labelled_rows"] == 2
    assert stats["unreviewed_rows"] == 1
    assert stats["reviewers"] == ["reviewer-1", "reviewer-2"]
    assert list(labelled["event_category"]) == ["sanctions", "armed_conflict"]


def test_labels_outside_the_taxonomy_are_rejected(tmp_path: Path):
    sheet = tmp_path / "sheet.csv"
    sheet.write_text("article_id,event_category\na1,trade_war\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the taxonomy"):
        load_labelled_sheet(sheet)


def test_missing_sheet_raises_with_the_next_command(tmp_path: Path):
    with pytest.raises(DataNotFetchedError, match="annotate"):
        load_labelled_sheet(tmp_path / "absent.csv")


def prepare_labelled_project(root: Path, config_path: Path, raw_frame: pd.DataFrame, *, label_override: dict[str, str] | None = None) -> Config:
    """Cache cleaned news, classify it, and write labels for every sampled row."""
    config = load_config(config_path)
    cleaned, _ = clean_news_data(raw_frame)
    target = news_processed_path(config, root=root)
    target.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(target, index=False, lineterminator="\n")
    classify_datasets(config, root=root)

    from stock_project.nlp.annotate import annotate

    annotate(config, root=root, size=100, focus_size=4)
    sheet_path = review_sheet_path(root, REVIEW_SAMPLE_FILENAME)
    sheet = pd.read_csv(sheet_path, dtype=str).fillna("")
    predictions = pd.read_csv(classified_path(config, root=root)).set_index("article_id")["event_category"]
    sheet["event_category"] = [predictions[article_id] for article_id in sheet["article_id"]]
    sheet["reviewer"] = "reviewer-1"
    sheet["reviewed_at"] = "2026-10-13"
    if label_override:
        for article_id, label in label_override.items():
            sheet.loc[sheet["article_id"] == article_id, "event_category"] = label
    sheet.to_csv(sheet_path, index=False, lineterminator="\n")
    return config


def test_end_to_end_evaluation_writes_metrics_and_error_list(
    news_raw_classifiable: pd.DataFrame, config_path: Path, tmp_path: Path
):
    from stock_project.nlp.evaluate_classifier import evaluate_classifier

    config = load_config(config_path)
    cleaned, _ = clean_news_data(news_raw_classifiable)
    target = news_processed_path(config, root=tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(target, index=False, lineterminator="\n")
    classify_datasets(config, root=tmp_path)

    from stock_project.nlp.annotate import annotate

    annotate(config, root=tmp_path, size=100, focus_size=4)
    sheet_path = review_sheet_path(tmp_path, REVIEW_SAMPLE_FILENAME)
    sheet = pd.read_csv(sheet_path, dtype=str).fillna("")
    predictions = pd.read_csv(classified_path(config, root=tmp_path)).set_index("article_id")["event_category"]
    sheet["event_category"] = [predictions[article_id] for article_id in sheet["article_id"]]
    # One deliberate disagreement so the error path is exercised.
    first_id = sheet.loc[0, "article_id"]
    sheet.loc[0, "event_category"] = "other_or_uncertain" if predictions[first_id] != "other_or_uncertain" else "sanctions"
    sheet.to_csv(sheet_path, index=False, lineterminator="\n")

    report = evaluate_classifier(config, root=tmp_path)
    primary = report["primary_sheet"]
    assert primary["labelled_rows"] == 8
    assert primary["unreviewed_rows"] == 0
    assert primary["metrics"]["n_evaluated"] == 8
    assert primary["metrics"]["accuracy"] == pytest.approx(7 / 8, abs=1e-6)
    assert primary["error_patterns"]["error_count"] == 1
    baseline = primary["naive_majority_baseline"]
    assert baseline["accuracy"] == pytest.approx(2 / 8, abs=1e-6)
    assert baseline["class"] in {"sanctions", "trade_restriction", "other_or_uncertain"}

    errors = pd.read_csv(tmp_path / "reports" / "experiment_results" / MISCLASSIFICATIONS_FILENAME)
    assert len(errors) == 1
    assert errors.loc[0, "article_id"] == first_id
    assert errors.loc[0, "headline"]

    on_disk = json.loads((tmp_path / "reports" / "experiment_results" / EVALUATION_FILENAME).read_text(encoding="utf-8"))
    assert on_disk["primary_sheet"]["metrics"]["accuracy"] == pytest.approx(7 / 8, abs=1e-6)
    assert on_disk["limitations"]


def test_unreviewed_rows_are_excluded_not_counted_as_correct(
    news_raw_classifiable: pd.DataFrame, config_path: Path, tmp_path: Path
):
    from stock_project.nlp.evaluate_classifier import evaluate_classifier

    config = prepare_labelled_project(tmp_path, config_path, news_raw_classifiable)
    sheet_path = review_sheet_path(tmp_path, REVIEW_SAMPLE_FILENAME)
    sheet = pd.read_csv(sheet_path, dtype=str).fillna("")
    sheet.loc[0:1, "event_category"] = ""
    sheet.to_csv(sheet_path, index=False, lineterminator="\n")

    report = evaluate_classifier(config, root=tmp_path)
    assert report["primary_sheet"]["labelled_rows"] == 6
    assert report["primary_sheet"]["unreviewed_rows"] == 2
    assert report["primary_sheet"]["metrics"]["n_evaluated"] == 6
    assert any("unreviewed" in limitation for limitation in report["limitations"])


def test_unlabelled_sheet_produces_a_report_without_metrics(
    news_raw_classifiable: pd.DataFrame, config_path: Path, tmp_path: Path, capsys
):
    config = load_config(config_path)
    cleaned, _ = clean_news_data(news_raw_classifiable)
    target = news_processed_path(config, root=tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(target, index=False, lineterminator="\n")
    classify_datasets(config, root=tmp_path)

    from stock_project.nlp.annotate import annotate

    annotate(config, root=tmp_path, size=4, focus_size=4)
    exit_code = main(["--config", str(config_path), "--root", str(tmp_path)])
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "labelled rows     : 0" in captured
    assert "No labels yet" in captured

    report = json.loads((tmp_path / "reports" / "experiment_results" / EVALUATION_FILENAME).read_text(encoding="utf-8"))
    assert report["primary_sheet"]["metrics"] is None


def test_missing_classified_table_raises(config_path: Path, tmp_path: Path):
    from stock_project.nlp.evaluate_classifier import evaluate_classifier

    config = load_config(config_path)
    with pytest.raises(DataNotFetchedError, match="classify_news"):
        evaluate_classifier(config, root=tmp_path)
