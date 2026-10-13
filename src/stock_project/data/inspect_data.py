"""CLI: inspect the Phase 1 datasets and write the reproducibility artifact.

    python -m stock_project.data.inspect_data --config configs/config.example.yaml

Reads the cached raw files (no network needed), runs the cleaning/validation steps,
writes cleaned tables to ``data/processed/`` and the full report to
``reports/experiment_results/phase1_data_report.json``.

The report is the artifact the next phase consumes: per-table quality reports
(missing values, duplicates, timezone assumptions), coverage, explicit pass/fail
checks, the columns still unassigned, and the known limitations.
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
from ..common.quality import QualityReport
from ..common.timeutils import utc_now_iso
from ..preprocessing.clean_market_data import clean_market_data
from ..preprocessing.clean_news_data import NEWS_COLUMNS, UNASSIGNED_COLUMNS, clean_news_data
from .market_data import load_market_data, market_file_path
from .news_data import dataset_slug, load_news_subset, subset_path

logger = get_logger("stock_project.inspect_data")

REPORT_FILENAME = "phase1_data_report.json"


def market_processed_path(config: Config, *, root: Path | None = None) -> Path:
    return processed_dir(root) / f"market_{config.market.symbol.upper()}_{config.market.interval}_clean.csv"


def news_processed_path(config: Config, *, root: Path | None = None) -> Path:
    slug = dataset_slug(config.news.dataset_id)
    label = config.news.source_category_filter.strip().lower().replace(" ", "_")
    return processed_dir(root) / f"news_{slug}_{label}_clean.csv"


def _iso_day(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def build_checks(
    market_clean: pd.DataFrame,
    market_quality: QualityReport,
    news_clean: pd.DataFrame,
    news_quality: QualityReport,
) -> list[dict[str, object]]:
    """Phase 1 acceptance checks, each with the evidence that justifies its verdict."""
    checks: list[dict[str, object]] = []

    checks.append(
        {
            "name": "market_rows_present",
            "passed": len(market_clean) > 0,
            "detail": f"{len(market_clean)} session row(s) after cleaning",
        }
    )
    checks.append(
        {
            "name": "market_chronological",
            "passed": bool(market_clean["timestamp"].is_monotonic_increasing),
            "detail": "timestamps are strictly ascending after cleaning; no shuffling",
        }
    )
    unique_sessions = int(market_clean.duplicated(subset=["symbol", "timestamp"]).sum())
    checks.append(
        {
            "name": "market_sessions_unique",
            "passed": unique_sessions == 0,
            "detail": f"{market_quality.duplicates_removed} duplicate row(s) removed; {unique_sessions} remain",
        }
    )
    checks.append(
        {
            "name": "market_no_imputation",
            "passed": not market_quality.imputation_applied,
            "detail": (
                "no values imputed or forward-filled; reported missing cells: "
                f"{market_quality.columns_with_missing() or 'none'}"
            ),
        }
    )

    checks.append(
        {
            "name": "news_rows_present",
            "passed": len(news_clean) > 0,
            "detail": f"{len(news_clean)} article row(s) after cleaning",
        }
    )
    checks.append(
        {
            "name": "news_published_at_parsed",
            "passed": int(news_clean["published_at"].isna().sum()) == 0,
            "detail": (
                f"{news_quality.rows_dropped_unparsable_timestamp} row(s) dropped as unparsable before cleaning; "
                f"{int(news_clean['published_at'].isna().sum())} unparsable remain"
            ),
        }
    )
    available_guard = bool((news_clean["available_from"] > news_clean["published_at"]).all())
    checks.append(
        {
            "name": "news_available_from_after_published_at",
            "passed": available_guard,
            "detail": (
                "every article is available from the day after its (day-precision) publication date, so a "
                "same-day cutoff cannot leak post-cutoff news"
            ),
        }
    )
    checks.append(
        {
            "name": "news_article_ids_unique",
            "passed": int(news_clean["article_id"].duplicated().sum()) == 0,
            "detail": f"{news_quality.duplicates_removed} duplicate article(s) removed",
        }
    )
    unassigned_filled = {
        column: int(news_clean[column].notna().sum()) for column in UNASSIGNED_COLUMNS if news_clean[column].notna().any()
    }
    checks.append(
        {
            "name": "news_labels_not_invented",
            "passed": not unassigned_filled,
            "detail": (
                f"unassigned columns stay empty: {list(UNASSIGNED_COLUMNS)}"
                if not unassigned_filled
                else f"unexpectedly populated: {unassigned_filled}"
            ),
        }
    )

    market_days = set(market_clean["timestamp"].dt.date)
    news_days = set(news_clean["published_at"].dt.date)
    overlap = sorted(market_days & news_days)
    checks.append(
        {
            "name": "market_news_windows_overlap",
            "passed": bool(overlap),
            "detail": (
                f"{len(overlap)} calendar day(s) present in both tables"
                + (f", first {overlap[0]} last {overlap[-1]}" if overlap else " — the two windows do not overlap")
            ),
        }
    )
    return checks


def inspect_datasets(config: Config, *, root: Path | None = None, write: bool = True) -> dict[str, object]:
    """Clean the cached datasets, write the artifact, and return the report."""
    market_raw_path = market_file_path(
        config.market.symbol, config.market.interval, config.market.start, config.market.end, root=root
    )
    if not market_raw_path.exists():
        raise DataNotFetchedError(
            f"no cached market data at {market_raw_path}; run: python -m stock_project.data.fetch_data"
        )
    news_subset_file = subset_path(config.news.source_category_filter, config.news.start, config.news.end, root=root)
    if not news_subset_file.exists():
        raise DataNotFetchedError(
            f"no cached news subset at {news_subset_file}; run: python -m stock_project.data.fetch_data"
        )

    market_input = load_market_data(market_raw_path)
    news_input = load_news_subset(news_subset_file)
    market_clean, market_quality = clean_market_data(market_input, symbol=config.market.symbol)
    news_clean, news_quality = clean_news_data(news_input, source_label=None)
    if list(news_clean.columns) != NEWS_COLUMNS:  # pragma: no cover - defensive contract check
        raise AssertionError("cleaned news table does not match the documented column contract")

    market_out = market_processed_path(config, root=root)
    news_out = news_processed_path(config, root=root)
    report: dict[str, object] = {
        "generated_at_utc": utc_now_iso(),
        "phase": 1,
        "phase_name": "Define and inspect data",
        "config": config.to_dict(),
        "datasets": {
            "market": {
                "symbol": config.market.symbol,
                "interval": config.market.interval,
                "raw_file": relative_to_root(market_raw_path, root),
                "processed_file": relative_to_root(market_out, root),
                "raw_rows": int(len(market_input)),
                "sessions": int(len(market_clean)),
                "first_session": _iso_day(market_clean["timestamp"].min()),
                "last_session": _iso_day(market_clean["timestamp"].max()),
                "quality": market_quality.to_dict(),
            },
            "news": {
                "dataset_id": config.news.dataset_id,
                "revision": config.news.revision,
                "license": config.news.license,
                "subset_file": relative_to_root(news_subset_file, root),
                "processed_file": relative_to_root(news_out, root),
                "raw_rows": int(len(news_input)),
                "articles": int(len(news_clean)),
                "first_published": _iso_day(news_clean["published_at"].min()),
                "last_published": _iso_day(news_clean["published_at"].max()),
                "quality": news_quality.to_dict(),
            },
        },
        "summary": {
            "market_rows": int(len(market_clean)),
            "market_columns": list(market_clean.columns),
            "market_missing_cells": market_quality.columns_with_missing(),
            "news_rows": int(len(news_clean)),
            "news_columns": list(news_clean.columns),
            "news_missing_cells": news_quality.columns_with_missing(),
            "news_publishers": news_clean["source"].value_counts().to_dict(),
            "class_balance_note": (
                "Phase 1 has no prediction target yet, so target class balance is reported in Phase 3. "
                "Event-category balance depends on the Phase 2 annotation step; the single "
                f"{config.news.source_category_filter!r} source category is a filter, not our taxonomy."
            ),
        },
        "checks": [],
        "unassigned_columns": list(UNASSIGNED_COLUMNS),
        "limitations": [
            "News timestamps are day-level: time of day is unknown, so features must use available_from (Phase 4).",
            "The source is general-interest political news, not a curated geopolitical corpus; a source category "
            "filter does not make every headline geopolitical.",
            "Only one instrument (SPY) is covered in Phase 1; sector or multi-symbol coverage is out of scope for now.",
            "event_category and sentiment columns are intentionally empty until Phase 2.",
            "Coverage differs between the two tables where configured dates differ; overlapping coverage is reported above.",
            "Market data comes from Yahoo Finance via yfinance and is used for research/education only; see docs/dataset_notes.md.",
        ],
        "next_step": "Phase 2 — define the event taxonomy, then label a small reviewed sample and classify headlines.",
    }
    report["checks"] = build_checks(market_clean, market_quality, news_clean, news_quality)

    if write:
        ensure_directories(root)
        market_clean.to_csv(market_out, index=False, lineterminator="\n")
        news_clean.to_csv(news_out, index=False, lineterminator="\n")
        write_manifest(
            root,
            key=f"processed:market:{config.market.symbol.upper()}:{config.market.interval}",
            entry=make_dataset_entry(
                key=f"processed:market:{config.market.symbol.upper()}:{config.market.interval}",
                name=f"cleaned {config.market.symbol.upper()} {config.market.interval}",
                file_path=market_out,
                root=root,
                source=config.market.source,
                rows=int(len(market_clean)),
                columns=list(market_clean.columns),
                date_coverage=(
                    _iso_day(market_clean["timestamp"].min()),
                    _iso_day(market_clean["timestamp"].max()),
                ),
                notes=[
                    "Derived by preprocessing.clean_market_data from the raw file listed above.",
                    f"{market_quality.duplicates_removed} duplicate row(s) removed; no imputation applied.",
                ],
            ),
        )
        write_manifest(
            root,
            key=f"processed:news:{config.news.dataset_id}",
            entry=make_dataset_entry(
                key=f"processed:news:{config.news.dataset_id}",
                name=f"cleaned news subset ({config.news.source_category_filter})",
                file_path=news_out,
                root=root,
                source=config.news.dataset_url,
                dataset_id=config.news.dataset_id,
                revision=config.news.revision,
                license=config.news.license,
                license_url=config.news.license_url,
                rows=int(len(news_clean)),
                columns=list(news_clean.columns),
                date_coverage=(
                    _iso_day(news_clean["published_at"].min()),
                    _iso_day(news_clean["published_at"].max()),
                ),
                notes=[
                    "Derived by preprocessing.clean_news_data from the interim subset listed above.",
                    "event_category/sentiment columns are unassigned until Phase 2.",
                    f"licence {config.news.license} — attribution required ({config.news.license_url}).",
                ],
            ),
        )
        report_path = results_dir(root) / REPORT_FILENAME
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logger.info("wrote %s", relative_to_root(report_path, root))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean and inspect the cached Phase 1 datasets; write the JSON report.")
    parser.add_argument("--config", type=Path, default=None, help="YAML config (default: configs/config.yaml or the example)")
    parser.add_argument("--print-json", action="store_true", help="also print the full report to stdout")
    parser.add_argument("--root", type=Path, default=None, help="project root to read/write (default: this repository)")
    args = parser.parse_args(argv)

    configure_cli_logging()
    config = load_config(args.config)
    root = Path(args.root) if args.root else None
    report = inspect_datasets(config, root=root)

    market = report["datasets"]["market"]
    news = report["datasets"]["news"]
    print(f"market : {market['sessions']} sessions {market['first_session']}..{market['last_session']} ({config.market.symbol})")
    print(f"news   : {news['articles']} articles {news['first_published']}..{news['last_published']} ({config.news.source_category_filter})")
    print(f"report : {relative_to_root(results_dir(root) / REPORT_FILENAME, root)}")
    failures = [check for check in report["checks"] if not check["passed"]]
    for check in report["checks"]:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['name']}: {check['detail']}")
    if args.print_json:
        print(json.dumps(report, indent=2))
    if failures:
        print(f"\n{len(failures)} check(s) failed — the data is not ready for the next phase.")
        return 1
    print(f"\nAll {len(report['checks'])} Phase 1 checks passed.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
