"""Filesystem locations used across the project.

All paths are derived from the repository root so that the same commands behave
identically from the project root, a notebook, or a test. Functions accept an
optional ``root`` override, which is what keeps the test suite hermetic: tests
point the whole project at ``tmp_path`` instead of the real data directories.
"""

from __future__ import annotations

from pathlib import Path

#: Repository root, derived from this file (``<root>/src/stock_project/common/paths.py``).
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def project_root(root: Path | str | None = None) -> Path:
    """Return the repository root, or ``root`` when one is supplied."""
    return Path(root) if root is not None else PROJECT_ROOT


def data_dir(root: Path | str | None = None) -> Path:
    return project_root(root) / "data"


def raw_dir(root: Path | str | None = None) -> Path:
    """Untouched downloads. Git-ignored; never committed."""
    return data_dir(root) / "raw"


def raw_market_dir(root: Path | str | None = None) -> Path:
    return raw_dir(root) / "market"


def raw_news_dir(root: Path | str | None = None) -> Path:
    return raw_dir(root) / "news"


def interim_dir(root: Path | str | None = None) -> Path:
    """Intermediate, still-unvalidated datasets derived from raw downloads."""
    return data_dir(root) / "interim"


def processed_dir(root: Path | str | None = None) -> Path:
    """Cleaned, validated tables ready for later phases."""
    return data_dir(root) / "processed"


def configs_dir(root: Path | str | None = None) -> Path:
    return project_root(root) / "configs"


def reports_dir(root: Path | str | None = None) -> Path:
    return project_root(root) / "reports"


def results_dir(root: Path | str | None = None) -> Path:
    """Experiment artifacts consumed by the backend/dashboard (invariant 7)."""
    return reports_dir(root) / "experiment_results"


def figures_dir(root: Path | str | None = None) -> Path:
    return reports_dir(root) / "figures"


def manifest_path(root: Path | str | None = None) -> Path:
    """Committed provenance manifest for every dataset in the project."""
    return data_dir(root) / "manifest.json"


def all_data_dirs(root: Path | str | None = None) -> list[Path]:
    """Every directory the pipeline writes to, in creation order."""
    return [
        raw_dir(root),
        raw_market_dir(root),
        raw_news_dir(root),
        interim_dir(root),
        processed_dir(root),
        results_dir(root),
        figures_dir(root),
    ]


def ensure_directories(root: Path | str | None = None) -> list[Path]:
    """Create the data/report directories if needed and return them."""
    created: list[Path] = []
    for directory in all_data_dirs(root):
        directory.mkdir(parents=True, exist_ok=True)
        created.append(directory)
    return created


def relative_to_root(path: Path | str, root: Path | str | None = None) -> str:
    """Return ``path`` relative to the project root, when it is inside it.

    Report artifacts use this so they never embed machine-specific absolute
    paths, which keeps them diffable and reproducible on another checkout.
    """
    base = project_root(root)
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


__all__ = [
    "PROJECT_ROOT",
    "all_data_dirs",
    "configs_dir",
    "data_dir",
    "ensure_directories",
    "figures_dir",
    "interim_dir",
    "manifest_path",
    "processed_dir",
    "project_root",
    "raw_dir",
    "raw_market_dir",
    "raw_news_dir",
    "relative_to_root",
    "reports_dir",
    "results_dir",
]
