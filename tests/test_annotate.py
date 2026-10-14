"""Review sheets: determinism, empty labels, blindness, and label protection."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stock_project.common.config import Config, load_config
from stock_project.common.errors import DataNotFetchedError
from stock_project.data.inspect_data import news_processed_path
from stock_project.nlp.annotate import (
    LABEL_COLUMNS,
    PREDICTION_COLUMNS,
    REPORT_FILENAME,
    REVIEW_SAMPLE_FILENAME,
    build_focus_sheet,
    build_review_sheet,
    has_labels,
    main,
    review_sheet_path,
)
from stock_project.nlp.classify_news import classify_datasets, classified_path
from stock_project.nlp.taxonomy import EVENT_CATEGORIES
from stock_project.preprocessing.clean_news_data import clean_news_data


def prepare_phase2(root: Path, config_path: Path, raw_frame: pd.DataFrame) -> Config:
    """Cache the cleaned news table, then classify it, so the annotate step has input."""
    config = load_config(config_path)
    cleaned, _ = clean_news_data(raw_frame)
    target = news_processed_path(config, root=root)
    target.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(target, index=False, lineterminator="\n")
    classify_datasets(config, root=root)
    return config


def classified_frame(config: Config, root: Path) -> pd.DataFrame:
    return pd.read_csv(classified_path(config, root=root))


def test_random_sheet_is_reproducible(news_raw_classifiable: pd.DataFrame, config_path: Path, tmp_path: Path):
    config = prepare_phase2(tmp_path, config_path, news_raw_classifiable)
    frame = classified_frame(config, tmp_path)
    first = build_review_sheet(frame, size=5, seed=11)
    second = build_review_sheet(frame, size=5, seed=11)
    third = build_review_sheet(frame, size=5, seed=12)
    assert list(first["article_id"]) == list(second["article_id"])
    assert len(first) == 5 and len(third) == 5
    assert set(first["article_id"]).issubset(set(frame["article_id"]))


def test_sheet_has_blank_labels_and_context_columns(news_raw_classifiable: pd.DataFrame, config_path: Path, tmp_path: Path):
    config = prepare_phase2(tmp_path, config_path, news_raw_classifiable)
    frame = classified_frame(config, tmp_path)
    sheet = build_review_sheet(frame, size=5, seed=1)
    assert set(LABEL_COLUMNS).issubset(sheet.columns)
    for column in LABEL_COLUMNS:
        assert (sheet[column].astype(str) == "").all(), column
    assert list(sheet["sampling_reason"]) == ["random"] * 5
    assert not set(PREDICTION_COLUMNS) & set(sheet.columns)


def test_predictions_are_opt_in(news_raw_classifiable: pd.DataFrame, config_path: Path, tmp_path: Path):
    config = prepare_phase2(tmp_path, config_path, news_raw_classifiable)
    frame = classified_frame(config, tmp_path)
    sheet = build_review_sheet(frame, size=4, seed=1, include_predictions=True)
    assert set(PREDICTION_COLUMNS).issubset(sheet.columns)
    assert (sheet["event_category"].astype(str) == "").all()


def test_size_is_capped_by_available_articles(news_raw_classifiable: pd.DataFrame, config_path: Path, tmp_path: Path):
    config = prepare_phase2(tmp_path, config_path, news_raw_classifiable)
    frame = classified_frame(config, tmp_path)
    assert len(build_review_sheet(frame, size=100, seed=1)) == len(frame)


def test_invalid_sizes_raise(news_raw_classifiable: pd.DataFrame, config_path: Path, tmp_path: Path):
    config = prepare_phase2(tmp_path, config_path, news_raw_classifiable)
    frame = classified_frame(config, tmp_path)
    with pytest.raises(ValueError, match="positive"):
        build_review_sheet(frame, size=0, seed=1)
    with pytest.raises(ValueError, match="positive"):
        build_focus_sheet(frame, size=0, seed=1)


def test_focus_sheet_prioritises_flagged_rows(news_raw_classifiable: pd.DataFrame, config_path: Path, tmp_path: Path):
    config = prepare_phase2(tmp_path, config_path, news_raw_classifiable)
    frame = classified_frame(config, tmp_path)
    focus = build_focus_sheet(frame, size=len(frame), seed=1)
    reasons = set(focus["sampling_reason"])
    assert reasons <= {"flagged_evidence", "small_category", "flagged_evidence+small_category", "context"}
    flagged = set(frame.loc[frame["evidence_negated"] | frame["conditional"], "article_id"])
    assert flagged.issubset(set(focus["article_id"]))


def test_written_sheet_is_blank_and_report_describes_it(
    news_raw_classifiable: pd.DataFrame, config_path: Path, tmp_path: Path
):
    config = prepare_phase2(tmp_path, config_path, news_raw_classifiable)
    report = annotate_report(config, tmp_path)

    sheet_path = review_sheet_path(tmp_path, REVIEW_SAMPLE_FILENAME)
    on_disk = pd.read_csv(sheet_path, dtype=str).fillna("")
    assert len(on_disk) == report["sheets"]["random_sample"]["rows"]
    for column in LABEL_COLUMNS:
        assert (on_disk[column].str.strip() == "").all(), column
    assert report["labels_written"] == 0
    assert report["allowed_labels"] == list(EVENT_CATEGORIES)
    assert report["blind_labelling"] is True
    assert any("human review" in note for note in report["notes"])
    assert (tmp_path / "reports" / "experiment_results" / REPORT_FILENAME).exists()


def test_existing_labels_are_never_overwritten(
    news_raw_classifiable: pd.DataFrame, config_path: Path, tmp_path: Path
):
    from stock_project.nlp.annotate import annotate

    config = prepare_phase2(tmp_path, config_path, news_raw_classifiable)
    annotate(config, root=tmp_path, size=4, focus_size=4)
    sheet_path = review_sheet_path(tmp_path, REVIEW_SAMPLE_FILENAME)
    sheet = pd.read_csv(sheet_path, dtype=str).fillna("")
    sheet.loc[0, "event_category"] = "sanctions"
    sheet.loc[0, "reviewer"] = "reviewer-1"
    sheet.to_csv(sheet_path, index=False, lineterminator="\n")

    assert has_labels(sheet_path) is True
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        annotate(config, root=tmp_path, size=4, focus_size=4)
    # The hand-written label is still there after the refusal.
    assert pd.read_csv(sheet_path, dtype=str).loc[0, "event_category"] == "sanctions"

    annotate(config, root=tmp_path, size=4, focus_size=4, force=True)
    assert pd.read_csv(sheet_path, dtype=str)["event_category"].isna().all()


def test_has_labels_is_false_for_fresh_and_missing_sheets(tmp_path: Path):
    assert has_labels(tmp_path / "absent.csv") is False
    fresh = tmp_path / "fresh.csv"
    fresh.write_text("article_id,event_category,reviewer\n, ,\n", encoding="utf-8")
    assert has_labels(fresh) is False


def test_missing_classified_table_raises(config_path: Path, tmp_path: Path):
    from stock_project.nlp.annotate import annotate

    config = load_config(config_path)
    with pytest.raises(DataNotFetchedError, match="classify_news"):
        annotate(config, root=tmp_path)


def test_cli_exits_zero_and_says_labels_are_empty(
    news_raw_classifiable: pd.DataFrame, config_path: Path, tmp_path: Path, capsys
):
    prepare_phase2(tmp_path, config_path, news_raw_classifiable)
    exit_code = main(["--config", str(config_path), "--root", str(tmp_path), "--size", "4", "--focus-size", "4"])
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "Labels are left empty on purpose" in captured
    assert "random sample sheet" in captured


def annotate_report(config: Config, root: Path) -> dict:
    from stock_project.nlp.annotate import annotate

    return annotate(config, root=root, size=5, focus_size=5, write=True)
