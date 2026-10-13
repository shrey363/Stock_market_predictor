"""Provenance manifest for downloaded datasets.

Every fetch records source, licence, revision, retrieval time, row count, date
coverage, and a SHA-256 of the raw file in ``data/manifest.json``. The manifest
is small and committed, so the project can always answer "where did this data
come from, and exactly which snapshot?" without shipping the raw files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .paths import manifest_path, relative_to_root
from .timeutils import utc_now_iso

MANIFEST_VERSION = 1


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks so large manifests stay cheap."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def make_dataset_entry(
    *,
    key: str,
    name: str,
    file_path: Path | str,
    root: Path | str | None = None,
    source: str | None = None,
    url: str | None = None,
    dataset_id: str | None = None,
    revision: str | None = None,
    license: str | None = None,
    license_url: str | None = None,
    rows: int | None = None,
    columns: list[str] | None = None,
    date_coverage: tuple[str, str] | None = None,
    retrieved_at: str | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Build one manifest entry, hashing the file it describes."""
    path = Path(file_path)
    return {
        "key": key,
        "name": name,
        "file": relative_to_root(path, root),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "retrieved_at_utc": retrieved_at or utc_now_iso(),
        "source": source,
        "url": url,
        "dataset_id": dataset_id,
        "revision": revision,
        "license": license,
        "license_url": license_url,
        "rows": rows,
        "columns": columns,
        "date_coverage": list(date_coverage) if date_coverage else None,
        "notes": list(notes) if notes else [],
    }


def read_manifest(root: Path | str | None = None, path: Path | str | None = None) -> dict[str, Any]:
    """Read the manifest, returning an empty skeleton when it does not exist yet."""
    target = Path(path) if path is not None else manifest_path(root)
    if not target.exists():
        return {"manifest_version": MANIFEST_VERSION, "datasets": {}}
    loaded = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):  # pragma: no cover - defensive
        raise ValueError(f"manifest at {target} must contain a JSON object")
    loaded.setdefault("manifest_version", MANIFEST_VERSION)
    loaded.setdefault("datasets", {})
    return loaded


def write_manifest(
    root: Path | str | None = None,
    *,
    key: str,
    entry: dict[str, Any],
    path: Path | str | None = None,
) -> Path:
    """Insert/replace one dataset entry, preserving all other entries."""
    target = Path(path) if path is not None else manifest_path(root)
    manifest = read_manifest(root, target)
    manifest["datasets"][key] = entry
    manifest["updated_at_utc"] = utc_now_iso()
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = {
        "manifest_version": manifest["manifest_version"],
        "updated_at_utc": manifest["updated_at_utc"],
        "datasets": {name: manifest["datasets"][name] for name in sorted(manifest["datasets"])},
    }
    target.write_text(json.dumps(ordered, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return target


def get_dataset_entry(key: str, root: Path | str | None = None) -> dict[str, Any] | None:
    return read_manifest(root)["datasets"].get(key)


__all__ = ["MANIFEST_VERSION", "get_dataset_entry", "make_dataset_entry", "read_manifest", "sha256_file", "write_manifest"]
