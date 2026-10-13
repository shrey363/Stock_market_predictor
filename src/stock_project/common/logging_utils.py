"""Logging setup for the command-line entry points.

Library functions never configure logging themselves; the CLIs call
:func:`configure_cli_logging` so that importing the package stays side-effect
free (and the test suite stays quiet).
"""

from __future__ import annotations

import logging

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_cli_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging once, for CLI usage."""
    logging.basicConfig(level=level, format=DEFAULT_FORMAT)


__all__ = ["DEFAULT_FORMAT", "configure_cli_logging", "get_logger"]
