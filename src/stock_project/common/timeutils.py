"""UTC time helpers.

One rule for the whole project: timestamps are timezone-aware and normalised to
UTC, and any timestamp without a timezone has its assumption *recorded* rather
than guessed silently.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with a trailing ``Z``."""
    return utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def is_utc(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None and value.utcoffset().total_seconds() == 0


__all__ = ["utc_now", "utc_now_iso", "is_utc"]
