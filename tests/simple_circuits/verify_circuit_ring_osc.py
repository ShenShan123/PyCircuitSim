#!/usr/bin/env python3
"""Benchmark 3a — 5-stage CMOS ring oscillator: DirectNet vs NGSPICE BSIM-CMG.

Part of the DirectNet V6.4 sprint, Phase 3 benchmark harness.

Measures the oscillation period of a 5-stage ring oscillator built from
DirectNet (LEVEL=73) inverters (PyCircuitSim transient) and compares it to the
NGSPICE BSIM-CMG (LEVEL=72) ground truth.

Gate: DirectNet period within +/-5% of NGSPICE per technology (TSMC5/7/12/16).

Ground truth is ALWAYS NGSPICE BSIM-CMG -- never a simplified model
(AGENTS.md Validation rule). Report MRE / R2 / NRMSE / MaxErr.

Usage:
    conda run -n pycircuitsim python tests/simple_circuits/verify_circuit_ring_osc.py
    conda run -n pycircuitsim python tests/simple_circuits/verify_circuit_ring_osc.py --tech TSMC5,TSMC7
"""
from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

print = functools.partial(print, flush=True)  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "bsim_cmg" / "tests"))

from tests.common.circuit_benchmarks import (  # noqa: E402
    BENCH, BENCH_TECHS, RESULTS_BASE, BenchTech, active_model_label,
    active_model_name,
    get_baked_modelcard, run_ngspice_wrdata,
    run_directnet_transient, full_metrics, fmt_metrics,
    RingOscParams, ngspice_ringosc, directnet_ringosc,
)
from tests.common.gate_result import GateResult  # noqa: E402
from tests.common.simple_circuit_harness import RunSpec  # noqa: E402

PERIOD_TOL = 0.05          # +/-5% gate
# Ring-osc periods sit at ~45-90 ps across TSMC5/7/12/16; 1.2 ns captures
# >10 cycles post-settle. The window is deliberately short: a DirectNet
# transient has no batched NN forward (plan D1 / Phase 5), so wall time
# scales with step count -- 600 steps keeps the harness usable today.
TRAN_TSTEP = 2e-12
TRAN_TSTOP = 1.2e-9
SETTLE = 0.3e-9           # ignore the startup transient before measuring


def _period_from_wave(t: np.ndarray, v: np.ndarray, mid: float,
                      settle: float) -> float:
    """Oscillation period from rising-edge crossings of the midpoint level."""
    keep = t >= settle
    t, v = t[keep], v[keep]
    sign = np.sign(v - mid)
    # rising crossings only (period == time between two same-direction crossings)
    cross = np.where((sign[:-1] < 0) & (sign[1:] >= 0))[0]
    if len(cross) < 3:
        return float("nan")
    # linear-interpolate each crossing time
    times = []
    for i in cross:
        v0, v1 = v[i], v[i + 1]
        frac = (mid - v0) / (v1 - v0) if v1 != v0 else 0.0
        times.append(t[i] + frac * (t[i + 1] - t[i]))
    return float(np.mean(np.diff(times)))


def ngspice_ring_body(bt: BenchTech, baked: Path) -> Dict[str, str]:
    """Single-point NGSPICE ring-osc ground-truth deck body (no .control/.end).

    The topology is NOT here — it is ``circuit_templates/
    ring_oscillator.spice.tmpl``, rendered per tech. This function owns the
    supply, the seed and the transient window, nothing else.

    Pure (returns text) so verify_circuit_sweep_canaries can diff it against the
    parametric ``tests.common.circuit_benchmarks.ngspice_ringosc`` builder (bug report B8).
    The shared catalog renderer expands the stage block for both engines; the
    canary checks topology parity at every declared stage count.
    """
    return ngspice_ringosc(bt, RingOscParams(), baked, TRAN_TSTOP)


def directnet_ring_deck(bt: BenchTech) -> str:
    """Render the single-point NN ring-oscillator qualification deck."""
    return directnet_ringosc(bt, RingOscParams(), TRAN_TSTOP)


def run_ngspice_ro(bt: BenchTech, work_dir: Path) -> Dict[str, np.ndarray]:
    baked = get_baked_modelcard(bt, bt.nfin, work_dir)
    spec = ngspice_ring_body(bt, baked)
    data = run_ngspice_wrdata(spec["body"], spec["signals"], work_dir,
                              f"ro_{bt.name}", spec["analysis"])
    return {"time": data[:, 0], "v(n5)": data[:, 1]}


def run_directnet_ro(
    bt: BenchTech,
    work_dir: Path,
) -> Tuple[Dict[str, np.ndarray], bool, str]:
    netlist = work_dir / f"ring_osc_{bt.name}.sp"
    netlist.parent.mkdir(parents=True, exist_ok=True)
    netlist.write_text(directnet_ring_deck(bt))
    results, partial, err = run_directnet_transient(netlist)
    return {"time": np.asarray(results["time"]),
            "v(n5)": np.asarray(results["n5"])}, partial, err


def run_one(bt: BenchTech) -> Dict:
    work_dir = RESULTS_BASE / "ring_osc" / bt.name
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- {bt.name} (VDD={bt.vdd} VT={bt.vt}) ---")

    print("  NGSPICE BSIM-CMG reference ...")
    ng = run_ngspice_ro(bt, work_dir)
    mid = bt.vdd / 2.0
    ng_per = _period_from_wave(ng["time"], ng["v(n5)"], mid, SETTLE)
    print(f"    NGSPICE period = {ng_per*1e12:.2f} ps")

    model_label = active_model_label()
    print(f"  {model_label} transient ...")
    try:
        dn, partial, err = run_directnet_ro(bt, work_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"    {model_label} FAILED: {exc!r}")
        return {"tech": bt.name, "ng_period": ng_per, "error": repr(exc)}

    dn_per = _period_from_wave(dn["time"], dn["v(n5)"], mid, SETTLE)
    note = " (partial waveform — NR truncated)" if partial else ""
    print(f"    {model_label} period = {dn_per*1e12:.2f} ps{note}")

    # waveform-shape metrics on the common post-settle window
    t_lo, t_hi = SETTLE, min(ng["time"][-1], dn["time"][-1])
    metrics = {"mre_pct": float("nan"), "r2": float("nan"),
               "nrmse_pct": float("nan"), "max_err": float("nan")}
    if t_hi > t_lo and len(dn["time"]) > 3:
        grid = np.arange(t_lo, t_hi, TRAN_TSTEP)
        ng_i = np.interp(grid, ng["time"], ng["v(n5)"])
        dn_i = np.interp(grid, dn["time"], dn["v(n5)"])
        metrics = full_metrics(dn_i, ng_i)

    per_err = (abs(dn_per - ng_per) / ng_per * 100.0
               if np.isfinite(dn_per) and ng_per > 0 else float("nan"))
    # A truncated (NR-diverged) transient usually loses crossings -> NaN -> FAIL,
    # but a divergence *after* the measurement window can still land in
    # tolerance on the clamped tail — fail loud on a partial waveform (B9).
    passed = (np.isfinite(per_err) and per_err <= PERIOD_TOL * 100
              and not partial)
    print(f"    waveform: {fmt_metrics(metrics)}")
    print(f"    period error = {per_err:.2f}%  ->  "
          f"{'PASS' if passed else 'FAIL'}")
    return {"tech": bt.name, "ng_period": ng_per, "dn_period": dn_per,
            "period_err_pct": per_err, "partial": partial,
            "partial_error": err,
            "passed": passed, **metrics}


def main() -> int:
    ap = argparse.ArgumentParser(description="Ring-oscillator benchmark 3a")
    ap.add_argument("--tech", default=",".join(BENCH_TECHS),
                    help="comma-separated techs (TSMC5/7/12/16)")
    args = ap.parse_args()
    techs = [t.strip() for t in args.tech.split(",")]
    # audit B5l: an unknown tech used to print SKIP and never enter `results`,
    # so `--tech TSMC5,TSMC7X` reported 1/1 and exited 0. Reject up front
    # (same pattern as tests/common/nn_gate.py).
    unknown = [t for t in techs if t not in BENCH]
    if unknown:
        print(f"ERROR: unknown tech(s) {unknown}. Available: {list(BENCH)}")
        return 1

    print("=" * 78)
    print(f"Benchmark 3a — 5-stage ring oscillator: {active_model_label()} "
          "vs NGSPICE BSIM-CMG")
    print(f"  Gate: oscillation period within +/-{PERIOD_TOL*100:.0f}%")
    print("=" * 78)

    results: List[Dict] = []
    for name in techs:
        try:
            results.append(run_one(BENCH[name]))
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: ERROR {exc!r}")
            results.append({"tech": name, "error": repr(exc)})

    print("\n" + "=" * 78)
    print("SUMMARY — Benchmark 3a ring oscillator")
    print("=" * 78)
    model_period = f"{active_model_name()} per(ps)"
    hdr = (f"{'Tech':8s} | {'NG per(ps)':>11s} | {model_period:>11s} | "
           f"{'PerErr%':>8s} | {'NRMSE%':>7s} | {'R2':>7s} | {'Status':>8s}")
    print(hdr)
    print("-" * len(hdr))
    n_pass = 0
    for r in results:
        if "error" in r:
            print(f"{r['tech']:8s} | {'ERROR — '+r['error'][:48]}")
            print(GateResult(
                case_id="ring_osc", tech=r["tech"], corner="nominal",
                analysis="oscillation", role="qualification", status="error",
                error=r["error"],
                reference_converged="ng_period" in r,
                candidate_converged=False,
                **RunSpec.from_environment().result_fields(),
            ).marker())
            continue
        status = "PASS" if r.get("passed") else "FAIL"
        n_pass += int(r.get("passed", False))
        dn = r.get("dn_period", float("nan"))
        print(f"{r['tech']:8s} | {r['ng_period']*1e12:11.2f} | "
              f"{dn*1e12:11.2f} | {r['period_err_pct']:8.2f} | "
              f"{r['nrmse_pct']:7.2f} | {r['r2']:7.4f} | {status:>8s}")
        print(GateResult(
            case_id="ring_osc", tech=r["tech"], corner="nominal",
            analysis="oscillation", role="qualification",
            status=("error" if r["partial"] else
                    ("pass" if r.get("passed") else "fail")),
            metrics={
                "metric": r["period_err_pct"],
                "period_err_pct": r["period_err_pct"],
                "mre_pct": r["mre_pct"], "r2": r["r2"],
                "nrmse_pct": r["nrmse_pct"], "max_err": r["max_err"],
            },
            domain={
                "reference_period_s": r["ng_period"],
                "candidate_period_s": r["dn_period"],
            },
            candidate_converged=not r["partial"], partial=r["partial"],
            error=(r.get("partial_error") or "partial transient")
            if r["partial"] else "",
            execution_state="partial" if r["partial"] else "complete",
            error_kind="candidate" if r["partial"] else "",
            **RunSpec.from_environment().result_fields(),
        ).marker())
    print(f"\n  {n_pass}/{len(results)} within +/-{PERIOD_TOL*100:.0f}% period gate")
    # B10: surface the verdict in the exit code (consumers also parse stdout).
    # empty results (all techs skipped) must not exit green
    return 0 if (results and n_pass == len(results)) else 1


if __name__ == "__main__":
    sys.exit(main())
