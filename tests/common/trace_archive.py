"""Persist the candidate and reference traces a gate compares (V7.7.6).

Campaign waveform plots must show the exact traces the verdicts scored, not a
re-simulation. When ``PYCIRCUITSIM_TRACE_ARCHIVE`` names a directory, each
comparison site writes one ``.npz`` holding both engines' raw axes and
signals; unset, nothing is written. Archiving never changes a result: a write
failure is printed and the gate continues.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

ENV = "PYCIRCUITSIM_TRACE_ARCHIVE"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", str(text)).strip("_") or "x"


def archive_trace_pair(
    case: str,
    tech: str,
    analysis: str,
    axis_name: str,
    candidate_axis: np.ndarray,
    candidate: Mapping[str, np.ndarray],
    reference_axis: np.ndarray,
    reference: Mapping[str, np.ndarray],
    *,
    corner: str = "nominal",
) -> Optional[Path]:
    """Write one compared trace pair; return its path, or None when off.

    ``candidate`` and ``reference`` map signal names to arrays on their own
    axes (complex for AC). Only signals present in both are archived.
    """
    root = os.environ.get(ENV)
    if not root:
        return None
    signals = [name for name in candidate if name in reference]
    stem = "__".join(_slug(part) for part in (case, tech, corner, analysis))
    path = Path(root) / f"{stem}.npz"
    arrays = {
        "candidate_axis": np.asarray(candidate_axis),
        "reference_axis": np.asarray(reference_axis),
    }
    for index, name in enumerate(signals):
        arrays[f"candidate_{index}"] = np.asarray(candidate[name])
        arrays[f"reference_{index}"] = np.asarray(reference[name])
    meta = {"case": case, "tech": tech, "corner": corner,
            "analysis": analysis, "axis": axis_name, "signals": signals}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, meta=np.asarray(json.dumps(meta)), **arrays)
    except (OSError, ValueError) as exc:
        print(f"[trace-archive] WARNING: {path}: {exc}", file=sys.stderr)
        return None
    return path


def load_trace_pair(path: Path) -> dict:
    """Read an archive back as {meta, candidate_axis, reference_axis, signals}."""
    with np.load(path) as data:
        meta = json.loads(str(data["meta"]))
        signals = {
            name: (data[f"candidate_{index}"], data[f"reference_{index}"])
            for index, name in enumerate(meta["signals"])
        }
        return {"meta": meta, "candidate_axis": data["candidate_axis"],
                "reference_axis": data["reference_axis"], "signals": signals}
