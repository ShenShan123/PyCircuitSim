#!/usr/bin/env python3
"""V7.2.0 §8.4 T4 gate — bistable latch basin agreement for the
perturbing NN-eval configurations.

The complex suite has no *gated* bistable test (the SRAM ``force_ic``
probe is a printed diagnostic), yet the failure mode the V7.2.0 plan
fears most is exactly a bistable one: a last-float32-bit perturbation
(batched-GEMM row vs single-row eval, GPU vs CPU kernels) amplified by a
latch into the *other* stored state. This gate makes that failure
binding: a full 6T cell in retention (wordline OFF — the valid probe per
the V6.4.7 S11 finding; wordline-ON read-disturb fails even for the L72
ground truth), both stored states, all requested techs.

Reference is the production path: CPU, all perturbing flags unset.
Candidate is one of:

  commit      PYCIRCUITSIM_TRAN_BATCH_COMMIT=1 (Phase 2t), CPU
  gpu         PYCIRCUITSIM_NN_DEVICE=cuda (Phase 3a), commit flag unset
  commit+gpu  both

Verdict: PASS requires 100% basin agreement — every (tech x state) cell
must end in the state it started in, in BOTH runs. Waveform NRMSE and
final-voltage deltas are reported as diagnostics. Any reference run
losing retention is an ERROR (infrastructure/model problem, not a
candidate failure). Per plan §8.4: 100% same-basin or no promotion; the
numbers are per-hardware — re-run on any driver change.

Usage:
    conda run -n pycircuitsim python tests/perf/verify_latch_basin_gpu.py
    ... verify_latch_basin_gpu.py --config commit+gpu --gpu 2
    ... verify_latch_basin_gpu.py --tech TSMC5,TSMC7
"""
from __future__ import annotations

import argparse
import csv
import functools
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

print = functools.partial(print, flush=True)  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.circuit_benchmarks import BENCH  # noqa: E402

DEFAULT_TECHS = ["TSMC5", "TSMC7", "TSMC12", "TSMC16"]
TRAN = ".tran 0.05n 2n"


def render_cell(tech: str, state_q1: bool) -> str:
    """6T bitcell in retention: WL off, both bitlines at VDD."""
    bt = BENCH[tech]
    vdd = bt.vdd
    q0 = vdd if state_q1 else 0.0
    qb0 = 0.0 if state_q1 else vdd
    ln = f"{bt.l_nmos * 1e9:.0f}n"
    lp = f"{bt.l_pmos * 1e9:.0f}n"
    nf, nfp = bt.nfin, bt.effective_nfin_p
    return "\n".join([
        f"* 6T retention latch — {tech} state q={'1' if state_q1 else '0'}",
        f"Vdd vdd 0 {vdd}",
        "Vwl wl 0 0.0",
        f"Vbl bl 0 {vdd}",
        f"Vblb blb 0 {vdd}",
        f"Mpl q qb vdd vdd pmos_nn L={lp} NFIN={nfp}",
        f"Mnl q qb 0 0 nmos_nn L={ln} NFIN={nf}",
        f"Mpr qb q vdd vdd pmos_nn L={lp} NFIN={nfp}",
        f"Mnr qb q 0 0 nmos_nn L={ln} NFIN={nf}",
        f"Mal bl wl q 0 nmos_nn L={ln} NFIN={nf}",
        f"Mar blb wl qb 0 nmos_nn L={ln} NFIN={nf}",
        f".model nmos_nn NMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.vt})",
        f".model pmos_nn PMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.vt})",
        f".ic V(q)={q0} V(qb)={qb0}",
        TRAN,
        ".end",
        "",
    ])


def run_deck(deck: Path, outdir: Path, extra_env: Dict[str, str]) -> Path:
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    env.update(extra_env)
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "main.py"), str(deck),
         "-o", str(outdir)],
        capture_output=True, text=True, env=env, timeout=1800)
    csvs = list(outdir.glob("*/tran/*_transient.csv"))
    if r.returncode != 0 or not csvs:
        raise RuntimeError(
            f"simulation failed rc={r.returncode}\n"
            f"stdout tail: {r.stdout[-800:]}\nstderr tail: {r.stderr[-800:]}")
    return csvs[0]


def load_csv(path: Path) -> Dict[str, List[float]]:
    with open(path) as f:
        rows = list(csv.reader(f))
    return {h: [float(r[i]) for r in rows[1:]]
            for i, h in enumerate(rows[0])}


def col(d: Dict[str, List[float]], name: str) -> List[float]:
    for k in d:
        if k.lower() in (name, f"v({name})"):
            return d[k]
    raise KeyError(f"column {name} not in {list(d)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", default=",".join(DEFAULT_TECHS))
    ap.add_argument("--config", default="commit",
                    choices=["commit", "gpu", "commit+gpu", "stamp",
                             "order", "commit+stamp", "stamp+order",
                             "commit+stamp+order", "commit+gpu+stamp",
                             "commit+gpu+stamp+order"])
    ap.add_argument("--gpu", default="0",
                    help="CUDA_VISIBLE_DEVICES for gpu configs")
    ap.add_argument("--ordering", default="NATURAL",
                    help="permc_spec for 'order' configs (Phase 4a')")
    args = ap.parse_args()
    techs = [t.strip().upper() for t in args.tech.split(",") if t.strip()]

    cand_env: Dict[str, str] = {}
    if "commit" in args.config:
        cand_env["PYCIRCUITSIM_TRAN_BATCH_COMMIT"] = "1"
    if "gpu" in args.config:
        cand_env["PYCIRCUITSIM_NN_DEVICE"] = "cuda"
        cand_env["CUDA_VISIBLE_DEVICES"] = args.gpu
    if "stamp" in args.config:
        cand_env["PYCIRCUITSIM_BATCHED_STAMP"] = "1"
    if "order" in args.config:
        cand_env["PYCIRCUITSIM_MNA_ORDERING"] = args.ordering

    print(f"T4 latch-basin gate — candidate config '{args.config}' "
          f"env={cand_env}")
    results: List[Tuple[str, str, str]] = []  # (case, verdict, detail)
    n_pass = n_fail = n_err = 0

    with tempfile.TemporaryDirectory(prefix="latch_basin_") as td:
        tmp = Path(td)
        for tech in techs:
            vdd = BENCH[tech].vdd
            for state_q1 in (True, False):
                case = f"{tech} q={'1' if state_q1 else '0'}"
                deck = tmp / f"cell_{tech}_{int(state_q1)}.sp"
                deck.write_text(render_cell(tech, state_q1))
                try:
                    ref_csv = run_deck(deck, tmp / f"ref_{deck.stem}", {})
                    can_csv = run_deck(
                        deck, tmp / f"can_{deck.stem}", cand_env)
                    ref = load_csv(ref_csv)
                    can = load_csv(can_csv)
                    rq, cq = col(ref, "q"), col(can, "q")
                    rqb, cqb = col(ref, "qb"), col(can, "qb")
                    half = vdd / 2.0
                    ref_kept = ((rq[-1] > half) == state_q1
                                and (rqb[-1] > half) != state_q1)
                    can_kept = ((cq[-1] > half) == state_q1
                                and (cqb[-1] > half) != state_q1)
                    dmax = max(
                        abs(a - b)
                        for ra, ca in ((rq, cq), (rqb, cqb))
                        for a, b in zip(ra, ca))
                    nrmse = 100.0 / vdd * math.sqrt(
                        sum((a - b) ** 2 for a, b in zip(rq, cq))
                        / len(rq))
                    detail = (f"ref_kept={ref_kept} cand_kept={can_kept} "
                              f"max|dV|={dmax * 1e3:.4f}mV "
                              f"q-NRMSE={nrmse:.4f}%VDD")
                    if not ref_kept:
                        verdict = "ERROR"   # reference lost retention
                        n_err += 1
                    elif can_kept:
                        verdict = "PASS"
                        n_pass += 1
                    else:
                        verdict = "FAIL"    # basin flip — the feared mode
                        n_fail += 1
                except Exception as e:  # noqa: BLE001 — report, don't die
                    verdict, detail = "ERROR", f"EXCEPTION: {e}"
                    n_err += 1
                results.append((case, verdict, detail))
                print(f"  [{verdict}] {case:14s} {detail}")

    total = len(results)
    print("=" * 70)
    print(f"RESULT: {n_pass}/{total} PASS, {n_fail} FAIL (basin flip), "
          f"{n_err} ERROR — config '{args.config}'")
    print("Gate rule: 100% same-basin or no promotion (plan §8.4 T4).")
    return 0 if (n_fail == 0 and n_err == 0 and n_pass == total) else 1


if __name__ == "__main__":
    sys.exit(main())
