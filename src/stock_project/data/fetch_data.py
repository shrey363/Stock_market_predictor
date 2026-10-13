"""CLI: fetch the Phase 1 datasets.

    python -m stock_project.data.fetch_data --config configs/config.example.yaml

Downloads (network required) and caches:

* ``data/raw/market/<symbol>_<interval>_<start>_<end>.csv`` — daily OHLCV bars,
* ``data/raw/news/<dataset>/<source_file>`` — the pinned raw news file, and
* ``data/interim/news_<category>_<start>_<end>.csv`` — the small filtered subset
  the rest of the project reads.

Existing files are reused unless ``--force`` is passed, so re-runs are cheap and
deterministic. Provenance for the raw downloads is written to ``data/manifest.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..common.config import Config, load_config
from ..common.logging_utils import configure_cli_logging, get_logger
from ..common.paths import ensure_directories, relative_to_root
from .market_data import fetch_market_data
from .news_data import extract_news_subset, fetch_news_dataset, subset_path, write_news_subset

logger = get_logger("stock_project.fetch_data")


def fetch_market(config: Config, *, root: Path | None = None, force: bool = False, downloader=None) -> Path:
    return fetch_market_data(
        config.market.symbol,
        config.market.start,
        config.market.end,
        interval=config.market.interval,
        root=root,
        source_label=config.market.source,
        downloader=downloader,
        force=force,
    )


def fetch_news(config: Config, *, root: Path | None = None, force: bool = False, downloader=None) -> tuple[Path, Path]:
    """Return ``(raw_file, subset_file)``."""
    raw = fetch_news_dataset(
        config.news.dataset_id,
        config.news.source_file,
        revision=config.news.revision,
        dataset_url=config.news.dataset_url,
        license_name=config.news.license,
        license_url=config.news.license_url,
        root=root,
        downloader=downloader,
        force=force,
    )
    subset = subset_path(config.news.source_category_filter, config.news.start, config.news.end, root=root)
    if subset.exists() and not force:
        logger.info("reusing cached news subset: %s", relative_to_root(subset, root))
        return raw, subset
    frame = extract_news_subset(
        raw,
        category=config.news.source_category_filter,
        start=config.news.start,
        end=config.news.end,
        max_articles=config.news.max_articles,
    )
    write_news_subset(frame, subset)
    logger.info("wrote %s news rows to %s", len(frame), relative_to_root(subset, root))
    return raw, subset


def fetch_all(config: Config, *, root: Path | None = None, force: bool = False) -> dict[str, Path]:
    ensure_directories(root)
    market = fetch_market(config, root=root, force=force)
    raw_news, subset = fetch_news(config, root=root, force=force)
    return {"market": market, "news_raw": raw_news, "news_subset": subset}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the Phase 1 market and news datasets (network required).")
    parser.add_argument("--config", type=Path, default=None, help="YAML config (default: configs/config.yaml or the example)")
    parser.add_argument("--force", action="store_true", help="re-download and rewrite already-cached files")
    parser.add_argument("--root", type=Path, default=None, help="project root to write into (default: this repository)")
    args = parser.parse_args(argv)

    configure_cli_logging()
    config = load_config(args.config)
    root = Path(args.root) if args.root else None
    paths = fetch_all(config, root=root, force=args.force)
    for label, path in paths.items():
        print(f"{label:>12}: {path}")
    print("\nNext step: python -m stock_project.data.inspect_data")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
