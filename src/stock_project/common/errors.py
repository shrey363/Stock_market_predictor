"""Project exception types.

Failures are raised with actionable messages instead of being swallowed: the
project's rules forbid hiding missing data or silently falling back to fabricated
values (see AGENTS.md §4 and §8).
"""

from __future__ import annotations


class StockProjectError(Exception):
    """Base class for project errors."""


class ConfigError(StockProjectError, ValueError):
    """The configuration file is missing, malformed, or internally inconsistent."""


class DataSourceError(StockProjectError):
    """A remote or local data source could not be read."""


class DataNotFetchedError(StockProjectError, FileNotFoundError):
    """A cached dataset is required for this step but has not been fetched yet."""


__all__ = ["StockProjectError", "ConfigError", "DataSourceError", "DataNotFetchedError"]
