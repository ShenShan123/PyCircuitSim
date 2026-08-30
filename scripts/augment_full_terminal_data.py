#!/usr/bin/env python3
"""Append only new terminal-length groups to a canonical DNF dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_compact_models" / "bsim_cmg"))
sys.path.insert(0, str(ROOT / "external_compact_models"))

from neural_network.data.dataset import validate_canonical_dataset  # noqa: E402
from pycmg.sweep import save_npz  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return commit, bool(status)


def _metadata(data: np.lib.npyio.NpzFile) -> Dict[str, object]:
    return {
        name.removeprefix("meta_"): np.asarray(data[name])
        for name in data.files
        if name.startswith("meta_")
    }


def augment(base_path: Path, terminal_path: Path, output_path: Path) -> dict:
    """Write base rows plus terminal-length groups absent from ``base``."""
    validate_canonical_dataset(base_path)
    validate_canonical_dataset(terminal_path)
    source_commit, source_dirty = _source_identity()
    if source_dirty:
        raise RuntimeError("augmentation source has tracked changes")

    with np.load(base_path, allow_pickle=False) as base, np.load(
        terminal_path, allow_pickle=False
    ) as terminal:
        base_max_l = float(np.max(base["geometry"][:, 1]))
        new_mask = terminal["geometry"][:, 1] > base_max_l
        new_rows = int(np.sum(new_mask))
        if new_rows == 0:
            raise ValueError("terminal artifact contains no new length groups")

        inputs = np.concatenate([base["inputs"], terminal["inputs"][new_mask]])
        geometry = np.concatenate(
            [base["geometry"], terminal["geometry"][new_mask]])
        outputs = np.concatenate(
            [base["outputs"], terminal["outputs"][new_mask]])
        sample_class = np.concatenate(
            [base["sample_class"], terminal["sample_class"][new_mask]])
        if len(outputs) != len(terminal["outputs"]):
            raise ValueError(
                "matched append row count differs from the regenerated artifact")

        metadata = _metadata(terminal)
        metadata.update({
            "dataset_variant": "v764_full_terminal_matched_append",
            "generator_release": "V7.6.4",
            "source_commit": source_commit,
            "source_dirty": np.bool_(False),
            "generator_command": shlex.join([sys.executable, *sys.argv]),
            "matched_base_dataset": base_path.name,
            "matched_base_dataset_sha256": _sha256(base_path),
            "terminal_dataset": terminal_path.name,
            "terminal_dataset_sha256": _sha256(terminal_path),
            "matched_base_rows": np.int64(len(base["outputs"])),
            "appended_terminal_rows": np.int64(new_rows),
            "matched_base_max_l": np.float64(base_max_l),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(
        inputs, geometry, outputs, output_path,
        metadata=metadata, sample_class=sample_class,
    )
    marker_path = output_path.with_suffix(output_path.suffix + ".complete")
    marker = {
        "dataset": output_path.name,
        "dataset_sha256": _sha256(output_path),
        "rows": len(outputs),
        "source_commit": source_commit,
        "source_dirty": False,
        "generator_release": "V7.6.4",
    }
    marker_path.write_text(json.dumps(marker, sort_keys=True, indent=2) + "\n")
    validate_canonical_dataset(output_path)
    return {
        "output": str(output_path),
        "rows": len(outputs),
        "base_rows": len(outputs) - new_rows,
        "terminal_rows": new_rows,
        "base_max_l_nm": base_max_l * 1e9,
        "output_sha256": marker["dataset_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        augment(args.base.resolve(), args.terminal.resolve(),
                args.output.resolve()),
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
