#!/usr/bin/env python3
"""V6.4.7 S9b — control-v2 gate vs baseline_v6_4_7_pre.json.

Runs the multi-circuit scorer (``eval_v6_4_5_candidate.py``) for every
(tech, seed) control-v2 checkpoint pair, parses the ``RESULT`` JSON, picks
the best seed per (tech, scorer-cell), and reports whether control-v2 holds
the S8 protected gates. Per plan S9b step 7 / rev-3 sequencing: control-v2
(stock recipe on the regen-v2 unfiltered data) must clear the baseline bar
before any S10+ arm trains; the inverter VTC is the s_id-drift canary.

Scorer cells covered: ring_osc, opamp, switchcap (3 of the 16 headline cells
x 4 techs = 12) + inverter (extended gate). sram_butterfly + force_ic come
from verify_complex_sram_snm.py and are checked separately.

Usage:
    NGSPICE_BIN=/.../ngspice PYTHONPATH=.../external_compact_models:.../PyCMG \
      conda run -n pycircuitsim python scripts/v6_4_7_s9b_gate_controlv2.py \
        --prefix v6_4_7_ctlv2 --seeds 42,17,7,31 \
        --techs tsmc5,tsmc7,tsmc12,tsmc16 \
        --out results/v6_4_7/S9b_controlv2_gate.md
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = PROJECT_ROOT / "external_compact_models" / "bsimar" / "checkpoints"

# Pass thresholds (mirror the harness gates).
RO_TOL = 5.0           # ring_osc_period_err %
OPAMP_TOL = 10.0       # opamp_gain_err %
INV_VTC_TOL = 5.0      # inv_vtc_nrmse % (baseline 0.96-2.36; 5% = generous gate)
INV_TRAN_TOL = 5.0     # inv_tran_post_nrmse %


def _run_scorer(tech: str, nmos: str, pmos: str,
                deriv: bool = False, timeout_s: int = 2400,
                cpu: bool = False) -> Optional[dict]:
    """Run the scorer for one (tech, seed) pair; return parsed RESULT dict."""
    # Scorer keys ALL_TEST_TECHS by UPPERCASE tech; checkpoint stems are
    # lowercase. deriv-fidelity lowercases tech for the npz filename, so
    # uppercase is safe everywhere.
    cmd = [
        "conda", "run", "-n", "pycircuitsim", "python",
        str(PROJECT_ROOT / "scripts" / "eval_v6_4_5_candidate.py"),
        "--tech", tech.upper(), "--nmos", nmos, "--pmos", pmos, "--json",
    ]
    if deriv:
        cmd += ["--deriv-fidelity", "--deriv-data-suffix", "v2",
                "--no-deriv-apply-filter"]
    env = dict(os.environ)
    env.setdefault(
        "NGSPICE_BIN",
        str(PROJECT_ROOT / "tools" / "ngspice-45.2" / "bin" / "ngspice"))
    # GPU vs CPU: a SINGLE GPU scorer co-exists with the training arms fine
    # (validated by the S9b smoke), but >1 concurrent GPU scorer + training
    # triggers CUDA device-side asserts / OOM. So GPU mode REQUIRES
    # workers=1; CPU mode (--cpu) avoids the GPU entirely but the full
    # harness is ~5x slower (ring_osc/opamp transients dominate).
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT/'external_compact_models'}:"
        f"{PROJECT_ROOT/'external_compact_models'/'PyCMG'}:"
        + env.get("PYTHONPATH", ""))
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired:
        return {"_error": "timeout", "tech": tech, "nmos": nmos}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            try:
                return json.loads(line[len("RESULT "):])
            except json.JSONDecodeError:
                pass
    return {"_error": "no RESULT line", "tech": tech, "nmos": nmos,
            "stderr_tail": p.stderr[-400:]}


def _cell_pass(res: dict, cell: str) -> Optional[bool]:
    """Pass flag for a scorer cell, or None if the metric is missing/NaN."""
    def ok(v) -> bool:
        return v is not None and v == v  # not None, not NaN

    if cell == "ring_osc":
        e = res.get("ring_osc_period_err")
        return (e <= RO_TOL) if ok(e) else None
    if cell == "opamp":
        flat = res.get("opamp_flat_flag")
        ge = res.get("opamp_gain_err")
        if not ok(ge) or flat is None:
            return None
        return (flat == 0) and (ge <= OPAMP_TOL)
    if cell == "switchcap":
        sp = res.get("sc_pass")
        return bool(sp) if sp is not None else None
    if cell == "inverter":
        v = res.get("inv_vtc_nrmse"); t = res.get("inv_tran_post_nrmse")
        if not (ok(v) and ok(t)):
            return None
        return (v <= INV_VTC_TOL) and (t <= INV_TRAN_TOL)
    return None


def _cell_metric(res: dict, cell: str) -> float:
    """Lower-is-better metric for best-seed selection within a cell."""
    return {
        "ring_osc": res.get("ring_osc_period_err", float("inf")),
        "opamp": res.get("opamp_gain_err", float("inf")),
        "switchcap": res.get("sc_charge_err_pct", float("inf")),
        "inverter": res.get("inv_vtc_nrmse", float("inf")),
    }.get(cell, float("inf"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="v6_4_7_ctlv2")
    ap.add_argument("--seeds", default="42,17,7,31")
    ap.add_argument("--techs", default="tsmc5,tsmc7,tsmc12,tsmc16")
    ap.add_argument("--baseline",
                    default=str(PROJECT_ROOT / "results" / "v6_4_7"
                                / "baseline_v6_4_7_pre.json"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "v6_4_7"
                                         / "S9b_controlv2_gate.md"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--deriv", action="store_true",
                    help="also compute deriv-fidelity (slower; off for the "
                         "protected-gate check, compute separately for P4)")
    ap.add_argument("--timeout", type=int, default=2400,
                    help="per-(tech,seed) scorer timeout seconds (CPU is slow)")
    ap.add_argument("--cpu", action="store_true",
                    help="run scorers on CPU (no GPU contention but ~5x "
                         "slower); default GPU REQUIRES --workers 1")
    args = ap.parse_args()
    if not args.cpu and args.workers != 1:
        print(f"[gate] GPU mode forces workers=1 (was {args.workers}) to "
              f"avoid CUDA contention with training", file=sys.stderr)
        args.workers = 1

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    techs = [t.strip() for t in args.techs.split(",") if t.strip()]
    baseline = json.load(open(args.baseline))

    # Build the (tech, seed) job list for pairs that have both checkpoints.
    jobs: List[Tuple[str, str, str, str]] = []
    for tech in techs:
        for seed in seeds:
            nmos = f"{args.prefix}_s{seed}_{tech}_nmos"
            pmos = f"{args.prefix}_s{seed}_{tech}_pmos"
            if (CKPT_DIR / f"{nmos}_best.pt").exists() and \
               (CKPT_DIR / f"{pmos}_best.pt").exists():
                jobs.append((tech, seed, nmos, pmos))
            else:
                print(f"[skip] {tech} s{seed}: missing checkpoint(s)",
                      file=sys.stderr)

    print(f"[gate] scoring {len(jobs)} (tech,seed) pairs "
          f"with {args.workers} workers", file=sys.stderr)
    results: Dict[Tuple[str, str], dict] = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut = {ex.submit(_run_scorer, t, n, p, args.deriv, args.timeout,
                         args.cpu): (t, s)
               for (t, s, n, p) in jobs}
        for f in cf.as_completed(fut):
            t, s = fut[f]
            results[(t, s)] = f.result()
            print(f"[gate] scored {t} s{s}", file=sys.stderr)

    # Aggregate: per (tech, cell) pick best-passing seed (then best metric).
    cells = ["ring_osc", "opamp", "switchcap", "inverter"]
    lines: List[str] = []
    lines.append("# V6.4.7 S9b — control-v2 gate vs baseline_v6_4_7_pre.json\n")
    lines.append(f"Scored {len(jobs)} (tech,seed) pairs. Pass tol: "
                 f"RO≤{RO_TOL}%, opamp≤{OPAMP_TOL}%, SC harness, "
                 f"inv VTC/tran≤{INV_VTC_TOL}/{INV_TRAN_TOL}%.\n")
    lines.append("| tech | cell | baseline | ctl-v2 best | best-seed | "
                 "metric | verdict |")
    lines.append("|---|---|---|---|---|---|---|")

    regressions: List[str] = []
    headline_pass = 0
    headline_total = 0
    for tech in techs:
        for cell in cells:
            # baseline pass for this cell
            base_section = baseline.get(
                {"ring_osc": "ring_osc", "opamp": "opamp",
                 "switchcap": "switchcap", "inverter": None}[cell] or "", {})
            if cell == "inverter":
                base_pass = True  # part of the 8/8 extended gate
            else:
                bt = base_section.get(tech.upper(), {}) \
                     if isinstance(base_section, dict) else {}
                base_pass = bool(bt.get("pass", False))

            # best ctl-v2 seed for this (tech, cell)
            best_seed = None; best_pass = None; best_metric = float("inf")
            for seed in seeds:
                res = results.get((tech, seed))
                if not res or "_error" in res:
                    continue
                cp = _cell_pass(res, cell)
                cm = _cell_metric(res, cell)
                # prefer a passing seed; among ties, lowest metric
                better = False
                if cp and not best_pass:
                    better = True
                elif cp == best_pass and cm < best_metric:
                    better = True
                if better or best_seed is None:
                    best_seed, best_pass, best_metric = seed, cp, cm

            if cell != "inverter":
                headline_total += 1
                if best_pass:
                    headline_pass += 1
            verdict = "OK"
            if base_pass and not best_pass:
                verdict = "REGRESSION"
                regressions.append(f"{tech} {cell}")
            elif not base_pass and best_pass:
                verdict = "NEW-PASS"
            lines.append(
                f"| {tech} | {cell} | {'pass' if base_pass else 'FAIL'} | "
                f"{'pass' if best_pass else 'fail' if best_pass is not None else '?'} | "
                f"s{best_seed} | {best_metric:.2f} | {verdict} |")

    lines.append(f"\n**Scorer headline (ro+opamp+sc, 12 cells): "
                 f"{headline_pass}/{headline_total}**")
    lines.append(f"\n**Protected-gate regressions: "
                 f"{len(regressions)}** "
                 f"{'— ' + ', '.join(regressions) if regressions else '(none)'}")
    lines.append("\n> sram_butterfly (4) + force_ic checked separately via "
                 "verify_complex_sram_snm.py on the selected mix.\n")
    lines.append("\n## Raw per-(tech,seed) RESULT vectors\n")
    lines.append("```json")
    lines.append(json.dumps(
        {f"{t}_s{s}": r for (t, s), r in sorted(results.items())}, indent=1))
    lines.append("```")

    out = "\n".join(lines)
    Path(args.out).write_text(out)
    print(out)
    print(f"\n[gate] wrote {args.out}", file=sys.stderr)
    print(f"[gate] GATE {'PASS' if not regressions else 'REVIEW'}: "
          f"{len(regressions)} regressions", file=sys.stderr)
    return 0 if not regressions else 2


if __name__ == "__main__":
    sys.exit(main())
