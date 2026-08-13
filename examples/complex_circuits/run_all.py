#!/usr/bin/env python3
"""Run every design in one per-tech tree and collect its measurements.

Which tree is resolved by ``tools/pycmg_lib._resolve_tree`` — the working
directory, or AG_TECH / AG_TREE. Since V7.5.8 one copy of this script and of
``tools/`` serves all five techs (they were byte-identical duplicates).

    cd designs_tsmc7 && python3 ../run_all.py        # everything, tsmc7
    cd designs_tsmc7 && python3 ../run_all.py amplifier ldo
    AG_TECH=tsmc7 python3 run_all.py                 # same, from anywhere
    NGSPICE=/path/to/ngspice python3 ../run_all.py

Writes results/<category>/<design>/<deck>.log plus results/summary.csv and
results/run_all.json.

Every row is produced by ``tools/finalize.py`` -- the same measurement code,
metric definitions and convergence ladder (hot/cold temperature
continuations, nodeset-fallback slew decks, wrap-aware AC stability, sensor
full-sweep gates) that writes each design's result.json and feeds RESULTS.md.
One semantics, one set of numbers: summary.csv can never disagree with
result.json because both come from the same finalize pass.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))

from pycmg_lib import ROOT, TECH               # noqa: E402
from finalize import CONTROLS, finalize        # noqa: E402

# Headline metric per category, in report order.  ``pm_true*`` is the
# wrap-aware stability margin (tools/acstab.py); ``phase_in_deg`` / ``pm_max``
# / ``pm_min`` remain the raw principal-value crossover readings so the
# tables stay auditable against the deck ``.meas`` output.
HEADLINE: Dict[str, List[str]] = {
    "amplifier": ["dcgain", "gain_bandwidth_product", "phase_in_deg",
                  "pm_true", "ph_xover", "power",
                  "cmrrdc", "dcpsrp", "dcpsrn", "vos25", "tc",
                  "sr_rise", "sr_fall"],
    "ldo": ["vout_max", "vout_min", "lnrmax", "lnrmin", "lr", "power_max",
            "power_min", "dcgain_max", "gbw_max", "pm_max", "pm_true_max",
            "dcgain_min", "gbw_min", "pm_min", "pm_true_min",
            "psrr_max", "psrr_min", "undershoot", "overshoot"],
    "sensing_front_end": ["vout25", "lsb_25_75c", "ppval", "vout0", "vout100",
                          "mono_violations", "min_slope_25_75c",
                          "max_step_frac_25_75c", "dc_fallback_points",
                          # SMCNR_SE_2st_AMP (AC bench) only:
                          "dcgain", "gain_bandwidth_product", "phase_in_deg",
                          "pm_true", "power"],
    "voltage_reference": ["vref1_at25", "vref1_tc", "vref2_at25", "vref2_tc",
                          "vref3_at25", "vref3_tc"],
    "charge_pump": ["up_iavg", "lo_iavg", "up_imin", "up_imax",
                    "lo_imin", "lo_imax"],
}


def run_design(args: Tuple[str, Path, Path]) -> Dict:
    """Measure one design through finalize and copy its fresh logs."""
    category, design_dir, results_dir = args
    t0 = time.time()
    res = finalize((category, design_dir))

    out = results_dir / category / design_dir.name
    out.mkdir(parents=True, exist_ok=True)
    # Only logs this pass produced: stale logs from older runs stay behind.
    for log in sorted(design_dir.glob("*.log")):
        if log.stat().st_mtime >= t0:
            shutil.copy(log, out / log.name)

    return {"category": category, "design": design_dir.name,
            "seconds": round(time.time() - t0, 1),
            "metrics": res["metrics"], "pass": res["pass"],
            "score": res["score"], "errors": res["errors"]}


def main() -> None:
    wanted = sys.argv[1:] or list(CONTROLS)
    results_dir = ROOT / "results"
    jobs: List[Tuple[str, Path, Path]] = []
    for category in wanted:
        cdir = ROOT / category
        if not cdir.is_dir():
            print(f"skip {category}: not built yet")
            continue
        for d in sorted(cdir.iterdir()):
            if d.is_dir() and (d / "netlist.spice").exists():
                jobs.append((category, d, results_dir))

    if not jobs:
        print("nothing to run")
        return

    workers = int(os.environ.get("RUN_ALL_WORKERS", "8"))
    rows: List[Dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_design, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            p = r["pass"]
            flag = f"  ERRORS: {len(r['errors'])}" if r["errors"] else ""
            print(f"[{len(rows):2d}/{len(jobs)}] {r['category']:18s} "
                  f"{r['design']:36s} {r['seconds']:7.1f}s "
                  f"pass={sum(bool(v) for v in p.values())}/{len(p) or 1}"
                  f"{flag}", flush=True)

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "run_all.json").write_text(
        json.dumps(sorted(rows, key=lambda r: (r["category"], r["design"])),
                   indent=2, default=str))

    with (results_dir / "summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "design", "metric", "value"])
        for r in sorted(rows, key=lambda r: (r["category"], r["design"])):
            for key in HEADLINE[r["category"]]:
                if key in r["metrics"]:
                    w.writerow([r["category"], r["design"], key,
                                r["metrics"][key]])
    print(f"\n{len(rows)} designs -> {results_dir/'summary.csv'}")


if __name__ == "__main__":
    main()
