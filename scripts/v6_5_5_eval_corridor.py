#!/usr/bin/env python3
"""V6.5.5 — A/B gate the corridor checkpoints vs the production baseline.

For each corridor config (tsmc5 ring / tsmc7 opamp × seeds), pin the NN
checkpoint via PYCIRCUITSIM_NN_CHECKPOINT_DN_{NMOS,PMOS}=<stem> and run the 4
complex gates for that tech in a CLEAN subprocess (CPU-pinned — the multistable
opamp lands different NR basins on CUDA vs CPU; memory v648-gate-cpu-vs-cuda).
Baseline = no override (the production tsmc{X}_dn_medium symlink).

Pass criteria mirror the gate scripts (ring ±5% period, opamp ±10% gain,
switchcap charge+droop, sram-snm). Prints a per-tech before/after table so the
ring<->opamp trade (tsmc5) and the opamp recovery (tsmc7) are explicit.

Usage:
    conda run -n pycircuitsim python scripts/v6_5_5_eval_corridor.py \
        --seeds 42,17,7 [--baseline-only]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
NGSPICE = ROOT / "tools" / "ngspice-45.2" / "bin" / "ngspice"
CKPT = ROOT / "external_compact_models" / "bsimar" / "checkpoints"

GATES = [
    ("opamp", "tests/verify_complex_opamp.py", r"gain error = ([\d.]+)%"),
    ("ring", "tests/verify_complex_ring_osc.py", r"period error = ([\d.]+)%"),
    ("swcap", "tests/verify_complex_switchcap.py", None),
    ("sram", "tests/verify_complex_sram_snm.py", None),
]
# tech -> corridor tag
TAG = {"tsmc5": "corring", "tsmc7": "coropamp"}


def _env(stem: Optional[str]) -> Dict[str, str]:
    e = dict(os.environ)
    e.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
             NGSPICE_BIN=str(NGSPICE),
             PYTHONPATH=f"{ROOT}/external_compact_models:"
                        f"{ROOT}/external_compact_models/PyCMG")
    for k in ("PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS",
              "PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS"):
        e.pop(k, None)
    if stem:
        e["PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS"] = stem
        e["PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS"] = stem
    return e


def _run_gate(script: str, tech: str, metric_re: Optional[str],
              env: Dict[str, str]) -> Tuple[Optional[bool], str]:
    techU = tech.upper()
    try:
        out = subprocess.run(
            [sys.executable, str(ROOT / script), "--tech", techU],
            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1200).stdout
    except subprocess.TimeoutExpired:
        return None, "timeout"
    # pass count from the SUMMARY "N/M ..." line (single tech -> M=1)
    passed: Optional[bool] = None
    for m in re.finditer(r"(\d+)\s*/\s*(\d+)\b", out):
        n, d = int(m.group(1)), int(m.group(2))
        if d == 1:
            passed = (n == 1)
    metric = ""
    if metric_re:
        mm = re.findall(metric_re, out)
        if mm:
            metric = mm[-1]
    return passed, metric


def eval_config(tech: str, stem: Optional[str], label: str) -> Dict:
    env = _env(stem)
    row = {"label": label, "tech": tech, "npass": 0}
    for name, script, mre in GATES:
        passed, metric = _run_gate(script, tech, mre, env)
        row[name] = passed
        row[f"{name}_m"] = metric
        row["npass"] += int(bool(passed))
    return row


def _fmt(row: Dict) -> str:
    def cell(g):
        p = row.get(g)
        m = row.get(f"{g}_m", "")
        tag = "P" if p else ("F" if p is False else "?")
        return f"{tag}{(' '+m) if m else ''}".strip()
    return (f"  {row['label']:28s} | {row['npass']}/4 | "
            f"opamp={cell('opamp'):>9} ring={cell('ring'):>9} "
            f"swcap={cell('swcap'):>3} sram={cell('sram'):>3}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,17,7")
    ap.add_argument("--techs", default="tsmc5,tsmc7")
    ap.add_argument("--baseline-only", action="store_true")
    args = ap.parse_args()
    seeds = [s.strip() for s in args.seeds.split(",")]
    techs = [t.strip() for t in args.techs.split(",")]

    print("=" * 78)
    print("V6.5.5 corridor A/B — 4 complex gates per tech (CPU-pinned)")
    print("=" * 78)
    for tech in techs:
        print(f"\n### {tech} ###")
        print(_fmt(eval_config(tech, None, "BASELINE (production)")))
        if args.baseline_only:
            continue
        # auto-discover every corridor checkpoint for this tech that has both
        # nmos+pmos siblings (stem = filename minus _nmos_best.pt).
        stems = sorted({p.name[:-len("_nmos_best.pt")]
                        for p in CKPT.glob(f"{tech}_dn_cor*_nmos_best.pt")
                        if (CKPT / f"{p.name[:-len('_nmos_best.pt')]}_pmos_best.pt").exists()})
        if seeds:
            stems = [s for s in stems if any(f"_s{sd}" in s for sd in seeds)]
        for stem in stems:
            print(_fmt(eval_config(tech, stem, stem.replace(f"{tech}_dn_", ""))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
