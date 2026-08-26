#!/usr/bin/env python3
"""Create one immutable provenance manifest for a re-gate campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def _jobs(path: Path) -> Tuple[List[str], Set[Tuple[str, str, str]]]:
    lines = [line.strip() for line in path.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    groups: Set[Tuple[str, str, str]] = set()
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"invalid campaign job: {line}")
        tag, variant, tech, _suite, _omp = parts
        groups.add((tag, variant, tech.lower()))
    return lines, groups


def _artifact_hashes(
    checkpoints: Path,
    groups: Set[Tuple[str, str, str]],
) -> Dict[str, Optional[str]]:
    hashes: Dict[str, Optional[str]] = {}
    for tag, variant, tech in sorted(groups):
        for device in ("nmos", "pmos"):
            stem = f"{tech}_{tag}_{variant}_{device}"
            suffixes = ["_best.pt", "_norm.npz", "_best.pt.complete"]
            if tag == "tf":
                suffixes.append("_config.npz")
            for suffix in suffixes:
                path = checkpoints / f"{stem}{suffix}"
                hashes[path.name] = _sha256(path) if path.is_file() else None
    return hashes


def _pdk_hashes(pdk_root: Path, techs: Set[str]) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for tech in sorted(techs):
        for path in sorted((pdk_root / tech.upper()).rglob("*.l")):
            hashes[str(path.relative_to(pdk_root))] = _sha256(path)
    return hashes


def _verify_group(
    manifest_path: Path,
    checkpoints: Path,
    group: Tuple[str, str, str],
) -> None:
    manifest = json.loads(manifest_path.read_text())
    expected_all = manifest.get("checkpoint_sha256")
    if not isinstance(expected_all, dict):
        raise ValueError("campaign manifest has no checkpoint hash map")
    observed = _artifact_hashes(checkpoints, {group})
    expected = {name: expected_all.get(name) for name in observed}
    if observed != expected or any(value is None for value in observed.values()):
        raise ValueError(
            f"checkpoint artifacts drifted for {'/'.join(group)}"
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--checkpoints", required=True, type=Path)
    parser.add_argument("--ngspice", type=Path)
    parser.add_argument("--osdi", type=Path)
    parser.add_argument("--pdk-root", type=Path)
    parser.add_argument(
        "--verify-group", nargs=3, metavar=("TAG", "VARIANT", "TECH"),
    )
    args = parser.parse_args(argv)

    if args.verify_group is not None:
        try:
            tag, variant, tech = args.verify_group
            _verify_group(
                args.output, args.checkpoints, (tag, variant, tech.lower()),
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"campaign provenance error: {exc}", file=sys.stderr)
            return 2
        return 0

    if any(value is None for value in (
        args.jobs, args.ngspice, args.osdi, args.pdk_root,
    )):
        parser.error(
            "--jobs, --ngspice, --osdi, and --pdk-root are required "
            "when creating a manifest"
        )
    assert args.jobs is not None
    assert args.ngspice is not None
    assert args.osdi is not None
    assert args.pdk_root is not None

    root = Path(__file__).resolve().parents[1]
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit(
            "campaign provenance requires a clean tracked worktree; commit first"
        )
    untracked = _git(root, "ls-files", "--others", "--exclude-standard")
    unsafe_suffixes = {".py", ".sh", ".so", ".toml", ".yaml", ".yml"}
    unsafe_untracked = [
        path for path in untracked.splitlines()
        if Path(path).suffix.lower() in unsafe_suffixes
    ]
    if unsafe_untracked:
        raise SystemExit(
            "campaign provenance found untracked executable/config source: "
            + ", ".join(unsafe_untracked)
        )
    lines, groups = _jobs(args.jobs)
    manifest = {
        "schema": 1,
        "source_commit": _git(root, "rev-parse", "HEAD"),
        "source_dirty": False,
        "python": str(Path(sys.executable).resolve()),
        "jobs_sha256": _sha256(args.jobs),
        "job_count": len(lines),
        "ngspice_path": str(args.ngspice.resolve()),
        "ngspice_sha256": _sha256(args.ngspice),
        "osdi_path": str(args.osdi.resolve()),
        "osdi_sha256": _sha256(args.osdi),
        "checkpoint_dir": str(args.checkpoints.resolve()),
        "checkpoint_sha256": _artifact_hashes(args.checkpoints, groups),
        "pdk_sha256": _pdk_hashes(
            args.pdk_root, {tech for _tag, _variant, tech in groups},
        ),
    }
    content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_text() != content:
        raise SystemExit(
            f"campaign manifest drifted: {args.output}; use a new output root"
        )
    if not args.output.exists():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(content)
        temporary.replace(args.output)
    print(_sha256(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
