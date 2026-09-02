"""Content-addressed provenance helpers for scored campaign artifacts."""

from __future__ import annotations

import functools
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict


@functools.lru_cache(maxsize=64)
def _sha256_at_version(path: Path, mtime_ns: int, ctime_ns: int,
                       size: int) -> str:
    """Hash one immutable-on-disk version of an artifact."""
    del mtime_ns, ctime_ns, size
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Return a cached digest that invalidates when file identity changes."""
    resolved = path.resolve()
    stat = resolved.stat()
    return _sha256_at_version(
        resolved, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)


def artifact_record(path: Path) -> Dict[str, str]:
    """Describe one local artifact by resolved path and content digest."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Provenance artifact not found: {resolved}")
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def artifact_record_is_current(value: object) -> bool:
    """Whether a persisted artifact record still names identical bytes."""
    if not isinstance(value, dict):
        return False
    path_value = value.get("path")
    digest = value.get("sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str):
        return False
    path = Path(path_value)
    try:
        return path.is_file() and file_sha256(path) == digest
    except OSError:
        return False


@functools.lru_cache(maxsize=8)
def executable_version(path: Path) -> str:
    """Return the first informative line from ``<path> --version``."""
    result = subprocess.run(
        [str(path), "--version"], check=True, capture_output=True, text=True,
    )
    lines = [line.strip(" *") for line in result.stdout.splitlines()
             if line.strip(" *")]
    if not lines:
        raise RuntimeError(f"No version text returned by {path}")
    return lines[0]


def executable_record(path: Path) -> Dict[str, str]:
    """Describe an executable by path, bytes, and reported version."""
    record = artifact_record(path)
    record["version"] = executable_version(path.resolve())
    return record


def executable_record_is_current(value: object) -> bool:
    """Whether executable bytes and reported version still match a record."""
    if not artifact_record_is_current(value) or not isinstance(value, dict):
        return False
    path_value = value.get("path")
    version = value.get("version")
    if not isinstance(path_value, str) or not isinstance(version, str):
        return False
    try:
        return executable_version(Path(path_value).resolve()) == version
    except (OSError, subprocess.SubprocessError, RuntimeError):
        return False
