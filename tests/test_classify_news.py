"""Classification CLI: traceable output table, report contents, and guards."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stock_project.common.config import Config, load_config
from stock_project.common.errors import DataNotFetchedError
from stock_project.data.inspect_data import news_processed_path
from stock_project.nlp.classify_news import (
    CLASSIFIED_COLUMNS,
    REPORT_FILENAME,
    build_classified_table,
    classified_path,
    classify_datasets,
    main,
)
from stock_project.nlp.taxonomy import EVENT_CATEGORIES, OTHER_CATEGORY
from stock_project.preprocessing.clean_news_data import clean_news_data


def prepare_cleaned_news(root: Path, config_path: Path, raw_frame: pd.DataFrame) -> Config:
    """Run the Phase 1 news cleaning and cache the result where Phase 2 expects it."""
    config = load_config(config_path)
    cleaned, _ = clean_news_data(raw_frame)
    target = news_processed_path(config, root=root)
    target.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(target, index=False, lineterminator="\n")
    return config


def test_classified_table_matches_the_documented_contract(news_raw_classifiable: pd.DataFrame, config_path: Path, tmp_path: Path):
    config = prepare_cleaned_news(tmp_path, config_path, news_raw_classifiable)
    cleaned = pd.read_csv(news_processed_path(config, root=tmp_path))
    classified = build_classified_table(cleaned, config)
    assert list(classified.columns) == CLASSIFIED_COLUMNS
    assert classified["affected_sector"].isna().all()
    assert len(classified) == len(cleaned)


def test_traceability_columns_survive_classification(news_raw_classifiable: pd.DataFrame, config_path: Path, tmp_path: Path):
    config = prepare_cleaned_news(tmp_path, config_path, news_raw_classifiable)
    cleaned = pd.read_csv(news_processed_path(config, root=tmp_path))
    classified = build_classified_table(cleaned, config)
    for column in ("article_id", "published_at", "available_from", "headline", "source_url", "source_category"):
        assert column in classified.columns
        assert classified[column].notna().all(), column
    conflict_evidence = classified.loc[classified["event_category"] == "armed_conflict", "matched_keywords"]
    assert conflict_evidence.str.contains("missiles").any()
    assert classified["matched_term_count"].sum() > 0


def test_end_to_end_classification_writes_table_and_report(
    news_raw_classifiable: pd.DataFrame, config_path: Path, tmp_path: Path
):
    config = prepare_cleaned_news(tmp_path, config_path, news_raw_classifiable)
    report = classify_datasets(config, root=tmp_path)

    table = pd.read_csv(classified_path(config, root=tmp_path))
    assert len(table) == 8
    distribution = table["event_category"].value_counts().to_dict()
    assert distribution["armed_conflict"] == 1
    assert distribution["sanctions"] == 2
    assert distribution["trade_restriction"] == 2
    assert distribution["energy_or_supply_disruption"] == 1
    assert distribution["diplomatic_development"] == 1
    assert distribution[OTHER_CATEGORY] == 1

    assert report["phase"] == 2
    assert report["inputs"]["articles"] == 8
    assert report["classification"]["articles_with_negated_evidence"] == 1
    assert report["classification"]["articles_with_conditional_evidence"] == 2
    assert report["method"]["text_scope"] == "headline"
    assert report["limitations"]
    assert report["next_step"].startswith("Label data/annotations")

    report_file = tmp_path / "reports" / "experiment_results" / REPORT_FILENAME
    assert report_file.exists()
    on_disk = json.loads(report_file.read_text(encoding="utf-8"))
    assert on_disk["classification"]["category_distribution"]["sanctions"]["count"] == 2
    assert on_disk["traceability"]["matched_keywords_recorded"] is True


def test_report_distribution_covers_every_taxonomy_category(
    news_raw_classifiable: pd.DataFrame, config_path: Path, tmp_path: Path
):
    config = prepare_cleaned_news(tmp_path, config_path, news_raw_classifiable)
    report = classify_datasets(config, root=tmp_path)
    assert list(report["classification"]["category_distribution"]) == list(EVENT_CATEGORIES)


def test_manifest_entry_records_the_classified_table(
    news_raw_classifiable: pd.DataFrame, config_path: Path, tmp_path: Path
):
    config = prepare_cleaned_news(tmp_path, config_path, news_raw_classifiable)
    classify_datasets(config, root=tmp_path)
    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["datasets"][f"classified:news:{config.news.dataset_id}"]
    assert entry["rows"] == 8
    assert entry["license"] == "CC BY 4.0"
    assert "human judgements" in " ".join(entry["notes"])


def test_empty_headlines_are_counted_and_fall_back(
    news_raw_classifiable: pd.DataFrame, config_path: Path, tmp_path: Path
):
    config = prepare_cleaned_news(tmp_path, config_path, news_raw_classifiable)
    target = news_processed_path(config, root=tmp_path)
    frame = pd.read_csv(target)
    frame.loc[0, "headline"] = ""
    frame.to_csv(target, index=False, lineterminator="\n")

    report = classify_datasets(config, root=tmp_path)
    assert report["inputs"]["empty_headlines"] == 1
    table = pd.read_csv(classified_path(config, root=tmp_path))
    assert table.loc[0, "event_category"] == OTHER_CATEGORY


def test_missing_cleaned_table_raises_with_the_next_command(config_path: Path, tmp_path: Path):
    config = load_config(config_path)
    with pytest.raises(DataNotFetchedError, match="inspect_data"):
        classify_datasets(config, root=tmp_path)


def test_cli_exits_zero_and_prints_the_distribution(
    news_raw_classifiable: pd.DataFrame, config_path: Path, tmp_path: Path, capsys
):
    prepare_cleaned_news(tmp_path, config_path, news_raw_classifiable)
    exit_code = main(["--config", str(config_path), "--root", str(tmp_path)])
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "articles classified: 8" in captured
    assert "trade_restriction" in captured
    assert "report:" in captured
