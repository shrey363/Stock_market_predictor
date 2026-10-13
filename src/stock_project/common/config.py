"""Typed, strictly validated access to the YAML configuration.

Validation is deliberately strict: a typo in a config key raises instead of
silently falling back to a default, because a silently wrong cutoff date or
symbol would corrupt every later phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ConfigError
from .paths import configs_dir, relative_to_root

SUPPORTED_TIMEZONE = "UTC"
REQUIRED_SECTIONS = ("project", "data", "prediction", "trading")


def parse_iso_date(value: Any, *, where: str) -> date:
    """Parse a ``YYYY-MM-DD`` string, raising :class:`ConfigError` otherwise."""
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ConfigError(f"{where} must be a YYYY-MM-DD string, got {type(value).__name__}")
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ConfigError(f"{where} must be a valid YYYY-MM-DD date, got {value!r}") from exc


def _require_mapping(parent: Mapping[str, Any], key: str, *, where: str) -> Mapping[str, Any]:
    if key not in parent:
        raise ConfigError(f"missing required section {key!r} in {where}")
    value = parent[key]
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key!r} in {where} must be a mapping, got {type(value).__name__}")
    return value


def _require(section: Mapping[str, Any], key: str, *, where: str) -> Any:
    if key not in section or section[key] is None:
        raise ConfigError(f"missing required key {key!r} in {where}")
    return section[key]


def _require_str(section: Mapping[str, Any], key: str, *, where: str) -> str:
    value = _require(section, key, where=where)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key!r} in {where} must be a non-empty string")
    return value.strip()


def _require_int(section: Mapping[str, Any], key: str, *, where: str) -> int:
    value = _require(section, key, where=where)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key!r} in {where} must be an integer, got {value!r}")
    return value


def _require_number(section: Mapping[str, Any], key: str, *, where: str) -> float:
    value = _require(section, key, where=where)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key!r} in {where} must be a number, got {value!r}")
    return float(value)


def _check_window(start: date, end: date, *, where: str) -> None:
    if start >= end:
        raise ConfigError(f"{where}: start ({start.isoformat()}) must be earlier than end ({end.isoformat()})")


def _check_cutoff_time(value: str, *, where: str) -> str:
    parts = value.split(":")
    valid = len(parts) == 2 and all(p.isdigit() and len(p) == 2 for p in parts)
    if not valid or not (0 <= int(parts[0]) <= 23 and 0 <= int(parts[1]) <= 59):
        raise ConfigError(f"{where} must be HH:MM (24-hour), got {value!r}")
    return value


@dataclass(frozen=True)
class ProjectSettings:
    name: str
    random_seed: int


@dataclass(frozen=True)
class MarketSettings:
    symbol: str
    start: date
    end: date
    interval: str
    timezone: str
    source: str


@dataclass(frozen=True)
class NewsSettings:
    dataset_id: str
    revision: str
    source_file: str
    license: str
    license_url: str
    dataset_url: str
    source_category_filter: str
    start: date
    end: date
    max_articles: int
    timestamp_precision: str


@dataclass(frozen=True)
class PredictionSettings:
    cutoff_time: str
    target: str


@dataclass(frozen=True)
class TradingSettings:
    initial_capital: float
    transaction_cost_bps: float
    position_fraction: float


@dataclass(frozen=True)
class Config:
    """A validated configuration plus the path it was loaded from."""

    project: ProjectSettings
    market: MarketSettings
    news: NewsSettings
    prediction: PredictionSettings
    trading: TradingSettings
    path: Path
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable snapshot, recorded next to every experiment artifact."""
        return {
            "config_path": relative_to_root(self.path),
            "project": {"name": self.project.name, "random_seed": self.project.random_seed},
            "data": {
                "market": {
                    "symbol": self.market.symbol,
                    "start": self.market.start.isoformat(),
                    "end": self.market.end.isoformat(),
                    "interval": self.market.interval,
                    "timezone": self.market.timezone,
                    "source": self.market.source,
                },
                "news": {
                    "dataset_id": self.news.dataset_id,
                    "revision": self.news.revision,
                    "source_file": self.news.source_file,
                    "license": self.news.license,
                    "license_url": self.news.license_url,
                    "dataset_url": self.news.dataset_url,
                    "source_category_filter": self.news.source_category_filter,
                    "start": self.news.start.isoformat(),
                    "end": self.news.end.isoformat(),
                    "max_articles": self.news.max_articles,
                    "timestamp_precision": self.news.timestamp_precision,
                },
            },
            "prediction": {"cutoff_time": self.prediction.cutoff_time, "target": self.prediction.target},
            "trading": {
                "initial_capital": self.trading.initial_capital,
                "transaction_cost_bps": self.trading.transaction_cost_bps,
                "position_fraction": self.trading.position_fraction,
            },
        }


def default_config_path(root: Path | str | None = None) -> Path:
    """``configs/config.yaml`` when present, otherwise the committed example."""
    base = configs_dir(root)
    candidate = base / "config.yaml"
    return candidate if candidate.exists() else base / "config.example.yaml"


def load_config(path: Path | str | None = None, *, root: Path | str | None = None) -> Config:
    """Read and validate a YAML configuration file."""
    resolved = Path(path) if path is not None else default_config_path(root)
    if not resolved.exists():
        raise ConfigError(f"config file not found: {resolved}")
    try:
        loaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - exact message varies by version
        raise ConfigError(f"config file {resolved} is not valid YAML: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"config file {resolved} must contain a YAML mapping at the top level")

    where = str(resolved)
    for section in REQUIRED_SECTIONS:
        _require_mapping(loaded, section, where=where)
    data = _require_mapping(loaded, "data", where=where)
    market_raw = _require_mapping(data, "market", where=f"{where}: data")
    news_raw = _require_mapping(data, "news", where=f"{where}: data")
    project_raw = _require_mapping(loaded, "project", where=where)
    prediction_raw = _require_mapping(loaded, "prediction", where=where)
    trading_raw = _require_mapping(loaded, "trading", where=where)

    market_where = f"{where}: data.market"
    market_start = parse_iso_date(_require(market_raw, "start", where=market_where), where=f"{market_where}.start")
    market_end = parse_iso_date(_require(market_raw, "end", where=market_where), where=f"{market_where}.end")
    _check_window(market_start, market_end, where=market_where)
    timezone = _require_str(market_raw, "timezone", where=market_where)
    if timezone.upper() != SUPPORTED_TIMEZONE:
        raise ConfigError(
            f"{market_where}.timezone must be {SUPPORTED_TIMEZONE}: the project normalises every "
            f"timestamp to UTC (see AGENTS.md §4), got {timezone!r}"
        )
    symbol = _require_str(market_raw, "symbol", where=market_where)
    if any(ch.isspace() for ch in symbol):
        raise ConfigError(f"{market_where}.symbol must not contain whitespace, got {symbol!r}")
    market = MarketSettings(
        symbol=symbol.upper(),
        start=market_start,
        end=market_end,
        interval=_require_str(market_raw, "interval", where=market_where),
        timezone=SUPPORTED_TIMEZONE,
        source=_require_str(market_raw, "source", where=market_where),
    )

    news_where = f"{where}: data.news"
    news_start = parse_iso_date(_require(news_raw, "start", where=news_where), where=f"{news_where}.start")
    news_end = parse_iso_date(_require(news_raw, "end", where=news_where), where=f"{news_where}.end")
    _check_window(news_start, news_end, where=news_where)
    max_articles = _require_int(news_raw, "max_articles", where=news_where)
    if max_articles <= 0:
        raise ConfigError(f"{news_where}.max_articles must be positive, got {max_articles}")
    news = NewsSettings(
        dataset_id=_require_str(news_raw, "dataset_id", where=news_where),
        revision=_require_str(news_raw, "revision", where=news_where),
        source_file=_require_str(news_raw, "source_file", where=news_where),
        license=_require_str(news_raw, "license", where=news_where),
        license_url=_require_str(news_raw, "license_url", where=news_where),
        dataset_url=_require_str(news_raw, "dataset_url", where=news_where),
        source_category_filter=_require_str(news_raw, "source_category_filter", where=news_where),
        start=news_start,
        end=news_end,
        max_articles=max_articles,
        timestamp_precision=_require_str(news_raw, "timestamp_precision", where=news_where),
    )

    project_where = f"{where}: project"
    prediction_where = f"{where}: prediction"
    trading_where = f"{where}: trading"
    prediction = PredictionSettings(
        cutoff_time=_check_cutoff_time(
            _require_str(prediction_raw, "cutoff_time", where=prediction_where), where=f"{prediction_where}.cutoff_time"
        ),
        target=_require_str(prediction_raw, "target", where=prediction_where),
    )
    initial_capital = _require_number(trading_raw, "initial_capital", where=trading_where)
    if initial_capital <= 0:
        raise ConfigError(f"{trading_where}.initial_capital must be positive, got {initial_capital}")
    cost_bps = _require_number(trading_raw, "transaction_cost_bps", where=trading_where)
    if cost_bps < 0:
        raise ConfigError(f"{trading_where}.transaction_cost_bps must not be negative, got {cost_bps}")
    position_fraction = _require_number(trading_raw, "position_fraction", where=trading_where)
    if not 0 < position_fraction <= 1:
        raise ConfigError(f"{trading_where}.position_fraction must be in (0, 1], got {position_fraction}")
    trading = TradingSettings(
        initial_capital=initial_capital,
        transaction_cost_bps=cost_bps,
        position_fraction=position_fraction,
    )

    return Config(
        project=ProjectSettings(
            name=_require_str(project_raw, "name", where=project_where),
            random_seed=_require_int(project_raw, "random_seed", where=project_where),
        ),
        market=market,
        news=news,
        prediction=prediction,
        trading=trading,
        path=resolved,
        raw=loaded,
    )


__all__ = [
    "Config",
    "ConfigError",
    "MarketSettings",
    "NewsSettings",
    "PredictionSettings",
    "ProjectSettings",
    "SUPPORTED_TIMEZONE",
    "TradingSettings",
    "default_config_path",
    "load_config",
    "parse_iso_date",
]
