#!/usr/bin/env python3
"""Plot every archived candidate trace against its NGSPICE reference.

Reads the ``<cell>/traces/*.npz`` archives that ``scripts/v710_regate.sh``
cells write under ``V710_SCRATCH`` (``tests/common/trace_archive.py``) and
draws one figure per (suite, case, technology, corner, analysis): one column
per NN family, one row per signal, NGSPICE LEVEL=72 in black under the four
model sizes. AC signals are drawn as |H| in dB on a log frequency axis; drain
currents as |Id| on a log axis. A size with no archive (candidate ERROR before
its comparison) is named in the legend as missing.

Usage:
  python scripts/campaign_waveform_plots.py --scratch <V710_SCRATCH> \
      [--scratch ...] --out results/<campaign>/waveforms [--jobs 16]

Writes ``<out>/<level>/<case>/<tech>/<corner>__<analysis>.png`` (levels
L0-L4 as in ``circuit_templates/``, plus ``hier``) and ``<out>/index.csv``
(figure, level, missing sizes per family).
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.common.trace_archive import load_trace_pair  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from campaign_case_matrix import label, suite_tiers  # noqa: E402

FAMILIES = (("dnf", "DirectNet-Full (L75)"), ("tff", "BSIM-AR-Full (L76)"))
SIZES = ("small", "medium", "large", "xl")
# Validated categorical slots 1-4 (dataviz reference palette, light mode);
# line style is the secondary encoding for the two low-contrast slots.
STYLE = {
    "small": ("#2a78d6", "-"),
    "medium": ("#eb6834", "--"),
    "large": ("#1baf7a", "-."),
    "xl": ("#eda100", ":"),
}
REFERENCE_INK = "#0b0b0b"
CELL = re.compile(r"^(dnf|tff)_(small|medium|large|xl)_(tsmc\d+)_(.+)_omp(\d+)$")

Key = Tuple[str, str, str, str, str]  # suite, case, tech, corner, analysis


def collect(scratch: List[Path]) -> Dict[Key, Dict[Tuple[str, str], Path]]:
    """Group archives by figure key; OMP=1 cells only (2/4 repeat them)."""
    groups: Dict[Key, Dict[Tuple[str, str], Path]] = defaultdict(dict)
    for root in scratch:
        for cell in sorted(root.iterdir()):
            match = CELL.match(cell.name)
            if not match or match.group(5) != "1":
                continue
            tag, size, tech, suite, _omp = match.groups()
            for path in sorted((cell / "traces").glob("*.npz")):
                case, _tech, corner, analysis = path.stem.split("__")
                key = (suite, case, tech.upper(), corner, analysis)
                groups[key][(tag, size)] = path
    return groups


AXIS_LABEL = {"time": "time (s)", "frequency": "frequency (Hz)"}


def _is_current(signal: str) -> bool:
    return signal.lower().startswith(("id", "i("))


def _panels(axis_name: str, signals: List[str]) -> List[Tuple[str, str]]:
    """(signal, view) rows: AC gets magnitude and phase, others one view."""
    views = ("mag_db", "phase_deg") if axis_name == "frequency" else ("value",)
    return [(signal, view) for signal in signals for view in views]


def _transform(values: np.ndarray, view: str, signal: str) -> np.ndarray:
    if view == "mag_db":
        return 20.0 * np.log10(np.maximum(np.abs(values), 1e-30))
    if view == "phase_deg":
        return np.rad2deg(np.unwrap(np.angle(values)))
    if _is_current(signal):
        return np.maximum(np.abs(np.real(values)), 1e-18)
    return np.real(values)


def _ylabel(view: str, signal: str) -> str:
    if view == "mag_db":
        return f"|{signal}| (dB)"
    if view == "phase_deg":
        return f"∠{signal} (deg)"
    if _is_current(signal):
        return f"|{signal}| (A)"
    return f"{signal} (V)"


def plot_one(item: Tuple[Key, Dict[Tuple[str, str], Path], Path]) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key, members, out, level = item
    suite, case, tech, corner, analysis = key
    loaded = {member: load_trace_pair(path) for member, path in members.items()}
    first = next(iter(loaded.values()))
    axis_name = first["meta"]["axis"]
    panels = _panels(axis_name, first["meta"]["signals"])
    fig, axes = plt.subplots(
        len(panels), 2, figsize=(11, 3.0 * len(panels)),
        squeeze=False, sharex="col",
    )
    missing_by_family = []
    for col, (tag, family_label) in enumerate(FAMILIES):
        present = [size for size in SIZES if (tag, size) in loaded]
        missing_by_family.append(
            " ".join(size for size in SIZES if size not in present) or "-")
        ref_source = loaded[(tag, present[0])] if present else first
        for row, (signal, view) in enumerate(panels):
            ax = axes[row][col]
            ref = ref_source["signals"].get(signal)
            if ref is not None:
                ax.plot(ref_source["reference_axis"],
                        _transform(ref[1], view, signal),
                        color=REFERENCE_INK, linewidth=2.6,
                        label="NGSPICE L72")
            for size in SIZES:
                color, style = STYLE[size]
                data = loaded.get((tag, size))
                if data is None or signal not in data["signals"]:
                    ax.plot([], [], color=color, linestyle=style,
                            label=f"{size}: no trace (ERROR)")
                    continue
                ax.plot(data["candidate_axis"],
                        _transform(data["signals"][signal][0], view, signal),
                        color=color, linestyle=style, linewidth=1.6, label=size)
            if axis_name == "frequency":
                ax.set_xscale("log")
            if view == "value" and _is_current(signal):
                ax.set_yscale("log")
            ax.set_ylabel(_ylabel(view, signal), fontsize=9)
            ax.grid(True, color="#e4e3df", linewidth=0.6)
            ax.tick_params(labelsize=8)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if row == 0:
                ax.set_title(family_label, fontsize=10)
                ax.legend(fontsize=7, frameon=False, loc="best")
        axes[-1][col].set_xlabel(AXIS_LABEL.get(axis_name, f"{axis_name} (V)"),
                                 fontsize=9)
    fig.suptitle(f"{case} · {tech} · {analysis} · {corner}", fontsize=11)
    fig.tight_layout()
    dest = out / level / label(suite) / tech / f"{case}__{corner}__{analysis}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=110)
    plt.close(fig)
    return [str(dest.relative_to(out)), level, suite, case, tech, corner,
            analysis, *missing_by_family]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scratch", action="append", required=True, type=Path,
                    help="V710_SCRATCH directory holding per-cell folders")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    groups = collect(args.scratch)
    if not groups:
        print("no trace archives found", file=sys.stderr)
        return 2
    tiers = suite_tiers()
    unplaced = sorted({key[0] for key in groups} - set(tiers))
    if unplaced:
        print(f"suites without a level: {unplaced}", file=sys.stderr)
        return 2
    items = [(key, members, args.out, tiers[key[0]])
             for key, members in sorted(groups.items())]
    with multiprocessing.Pool(args.jobs) as pool:
        rows = pool.map(plot_one, items, chunksize=4)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "index.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["figure", "level", "suite", "case", "tech", "corner",
                         "analysis",
                         "dnf_missing", "tff_missing"])
        writer.writerows(rows)
    print(f"{len(rows)} figures -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
