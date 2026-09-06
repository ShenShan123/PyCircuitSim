#!/usr/bin/env python3
"""Lifted-source full-terminal NN NMOS/PMOS Id-Vgs vs NGSPICE BSIM-CMG.

Guards the source-relative inference contract: a source lifted above ground
must be evaluated at shifted Vd/Vg/Vb while the NN's Vs input remains zero.

Sweep: NMOS Id-Vgs per tech (TSMC5/6/7/12/16) at tech-default L/NFIN/VT with
absolute terminal voltages Vs=vs0, Vb=0, Vd=VDD, Vg = 0..VDD (5 mV grid),
for vs0 in {0, 0.1, 0.2}*VDD. vs0=0 is the grounded control. NOTE: the drain
sits at VDD here (the 55-config gate biases Vds=VDD/2), so the control
matches the existing grounded baseline qualitatively (few-% NRMSE), not
bit-for-bit. PMOS mirrors every voltage around VDD: Vs=VDD-vs0, Vb=VDD,
Vd=0, Vg=VDD..0. Its reported sweep axis is VDD-Vg. Signed terminal currents
are oriented positive for each device polarity; a sign error is not hidden.

The NN side runs through the selected LEVEL=75/76 ``.dc`` path with four
independent DC V-sources, so `_raw_voltages` sees absolute node voltages.
Ground truth is the same NGSPICE OSDI BSIM-CMG mechanism as
``verify_nn_multi_tech_dc.py`` — never a simplified equation.

Permanent gate: NRMSE <= 10% per config (`DC_NRMSE_PASS`) — the vs input had
zero verification coverage before V6.4.7; this sweep is the canary for the
P0 frame fix and the P2 reverse-clamp work. The pre-fix capture lives at
``results/v6_4_7/lifted_source_sweep_prefix.{csv,md}`` (run with
``--label prefix --no-gate``).

V7.7.2: the gate is dispatched by the campaign ``canary`` pool (one cell per
checkpoint group, ``--tech <ONE>``), emits one structured result marker per
(technology, device, lift) so the collector can verify the cell is complete, and exits
through ``result_exit_code`` like every other campaign suite.

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      conda run -n pycircuitsim python tests/single_devices/verify_nn_lifted_source_dc.py \\
      [--tech TSMC5,TSMC7] [--label postfix] [--no-gate]
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.nn_sweep import (  # noqa: E402
    NN_TECHS, curve_metrics, validate_curve_trace,
)
from tests.common.base import (  # noqa: E402
    DEVICE_DECKS,
    parse_csv_choices,
    render_template,
    run_ngspice_subprocess,
)
from tests.common.gate_result import GateResult, result_exit_code  # noqa: E402
from tests.common.nn_gate import (  # noqa: E402
    ALL_TEST_TECHS,
    OSDI_PATH,
    TestTechConfig,
    create_baked_modelcard,
    create_baked_pmos_modelcard,
)
from tests.common.circuit_benchmarks import (  # noqa: E402
    active_model_label,
    active_model_level,
    nn_model_parameters,
)
from tests.common.simple_circuit_harness import RunSpec  # noqa: E402

VS0_FRACTIONS: List[float] = [0.0, 0.1, 0.2]   # source lift, fraction of VDD
VG_STEP = 0.005                                 # same grid as the 55-cfg gate
DC_NRMSE_PASS = 10.0                            # % — same gate as the 55-cfg DC
#: Structured-marker identity shared with the campaign collector.
LIFTED_CASE_ID = "nn_lifted_source_dc"
LIFTED_DEVICES = ("nmos", "pmos")
RESULTS_DIR = Path(os.environ.get(
    "PYCIRCUITSIM_NN_RESULTS",
    str(PROJECT_ROOT / "results" / "tests" / "nn_lifted_source_dc"),
))


def lifted_analysis_name(fraction: float, device: str = "nmos") -> str:
    """Keep historical NMOS marker names and distinguish mirrored PMOS rows."""
    prefix = "" if device == "nmos" else "pmos_"
    return f"{prefix}vs0_{round(fraction * 100)}pct"


def _lifted_deck(
    tech: TestTechConfig, vs0: float, device: str, *,
    model_setup: str, model: str, device_name: str,
) -> str:
    """Both adapters use identical biases and a mirrored PMOS experiment."""
    if device not in LIFTED_DEVICES:
        raise ValueError(f"unknown canary device {device!r}")
    is_nmos = device == "nmos"
    start, stop, step = ((0.0, tech.vdd, VG_STEP) if is_nmos
                         else (tech.vdd, 0.0, -VG_STEP))
    return render_template(DEVICE_DECKS / "mosfet.spice.tmpl", {
            "MODEL_SETUP": model_setup, "TEMP": "27",
            "DRAIN_BIAS": f"Vd d 0 {tech.vdd if is_nmos else 0.0:g}",
            "GATE_BIAS": f"Vg g 0 {start:g}",
            "SOURCE_BIAS": f"Vs s 0 {vs0 if is_nmos else tech.vdd - vs0:g}",
            "BULK_BIAS": f"Vb b 0 {0.0 if is_nmos else tech.vdd:g}",
            "DEVICE_NAME": device_name,
            "DRAIN_NODE": "d", "GATE_NODE": "g", "SOURCE_NODE": "s",
            "BULK_NODE": "b", "DEVICE": model,
            "EXTRA_DEVICES": "", "LOAD": "",
            "ANALYSIS": f".dc Vg {start:g} {stop:g} {step:g}",
        })


def run_ngspice_dc_lifted(
    tech: TestTechConfig, work_dir: Path, vs0: float, device: str,
) -> Dict[str, np.ndarray]:
    """Run fresh NGSPICE OSDI ground truth for one source-frame experiment."""
    is_nmos = device == "nmos"
    baked = (create_baked_modelcard if is_nmos else create_baked_pmos_modelcard)(
        tech, work_dir,
    )
    tag = f"{device}_lifted_{tech.name}_vs{round(vs0 * 1e3)}mV"
    netlist = work_dir / f"ngspice_{tag}.cir"
    netlist.write_text(_lifted_deck(
        tech, vs0, device, model_setup=f'.include "{baked}"',
        model=tech.nmos_model if is_nmos else tech.pmos_model, device_name="Ndut",
    ))
    csv_path = work_dir / f"ngspice_{tag}.csv"
    log_path = work_dir / f"ngspice_{tag}.log"
    runner = work_dir / f"ngspice_{tag}_runner.cir"
    runner.write_text(
        f"* NGSPICE DC runner ({tag})\n"
        f".control\n"
        f"osdi {OSDI_PATH}\n"
        f"source {netlist}\n"
        f"set filetype=ascii\nset wr_vecnames\nrun\n"
        f"wrdata {csv_path} i(Vd)\n"
        f".endc\n.end\n")
    lines = run_ngspice_subprocess(runner, log_path, csv_path)
    try:
        data = np.array([[float(x) for x in line.split()]
                         for line in lines[1:] if line.strip()])
    except ValueError as exc:
        raise RuntimeError(f"malformed NGSPICE output for {tag}") from exc
    if data.ndim != 2 or data.shape[1] != 2:
        raise RuntimeError(f"NGSPICE output has no aligned sweep/current pair for {tag}")
    return {"sweep": data[:, 0] if is_nmos else tech.vdd - data[:, 0],
            "id": -data[:, 1] if is_nmos else data[:, 1]}


def run_nn_dc_lifted(
    tech: TestTechConfig, work_dir: Path, vs0: float, device: str,
) -> Dict[str, np.ndarray]:
    """Run the selected full-terminal NN through the real ``.dc`` path."""
    from pycircuitsim.parser import Parser
    from pycircuitsim.simulation import run_dc_sweep
    from pycircuitsim.visualizer import Visualizer

    is_nmos = device == "nmos"
    length = tech.l_nmos if is_nmos else tech.effective_l_pmos
    vt = tech.nn_vt if is_nmos else tech.effective_pmos_vt
    tag = f"{device}_lifted_{tech.name}_vs{round(vs0 * 1e3)}mV"
    netlist = work_dir / f"nn_{tag}.sp"
    netlist.write_text(_lifted_deck(
        tech, vs0, device,
        model_setup=f".model dut_nn {device.upper()} "
                    f"({nn_model_parameters(active_model_level(), tech.nn_tech_key, vt)})",
        model=f"dut_nn L={length * 1e9:.0f}n NFIN={tech.nfin}", device_name="Mdut",
    ))
    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        parser = Parser()
        parser.parse_file(str(netlist))
        out_dir = work_dir / f"nn_dc_{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        results = run_dc_sweep(
            parser.circuit, parser.analysis_params, Visualizer(), out_dir,
            f"nn_{tag}", require_convergence=True,
        )
    finally:
        logging.disable(previous_disable)
    gate = np.asarray(results["g"])
    current = np.asarray(results["i(Mdut)"])
    return {"sweep": gate if is_nmos else tech.vdd - gate,
            "id": current if is_nmos else -current}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="postfix",
                    help="output-file label (results/v6_4_7/"
                         "lifted_source_sweep_<label>.{csv,md})")
    ap.add_argument("--no-gate", action="store_true",
                    help="diagnostic mode: report metrics, no PASS/FAIL gate")
    ap.add_argument("--tech", "--techs", dest="tech", default=None,
                    help="comma-separated tech filter (e.g. TSMC5,TSMC7); "
                         "needed when env-pinning per-tech recipe checkpoints")
    args = ap.parse_args(argv)
    techs = list(NN_TECHS)
    if args.tech is not None:
        techs = parse_csv_choices(
            ap, args.tech, flag="--tech", choices=NN_TECHS,
            normalize=str.upper,
        )
    try:
        run_spec = RunSpec.from_environment()
        run_spec.validate_checkpoint_pins(Path(os.environ.get(
            "BSIMAR_CHECKPOINT_DIR",
            PROJECT_ROOT / "external_compact_models" / "neural_network"
            / "checkpoints",
        )))
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    provenance = run_spec.result_fields()

    print("=" * 78)
    print(f"  V7.7.2 — lifted-source NMOS/PMOS Id-Vgs ({args.label})")
    print(f"  {run_spec.model_family} LEVEL={run_spec.model_level}")
    print("=" * 78)
    rows: List[Dict[str, object]] = []
    curve_rows: List[List[object]] = []
    results: List[GateResult] = []
    n_pass = 0
    for tk in techs:
        tech = ALL_TEST_TECHS[tk]
        for device, frac in product(LIFTED_DEVICES, VS0_FRACTIONS):
            vs0 = round(tech.vdd * frac, 4)
            analysis = lifted_analysis_name(frac, device)
            identity = {
                "case_id": LIFTED_CASE_ID, "tech": tk, "corner": "nominal",
                "analysis": analysis,
                "role": "diagnostic" if args.no_gate else "qualification",
            }
            wd = RESULTS_DIR / tk / analysis
            wd.mkdir(parents=True, exist_ok=True)
            try:
                ref = run_ngspice_dc_lifted(tech, wd, vs0, device)
                validate_curve_trace(
                    ref["sweep"], ref["id"], expected_start=0.0,
                    expected_stop=tech.vdd, max_step=VG_STEP,
                    label=f"{device} reference",
                )
            except (RuntimeError, ValueError, OSError) as exc:
                print(f"  {tk:<7s} {device} vs0={vs0:6.3f}V  ERROR reference: {exc}")
                results.append(GateResult(
                    **identity, status="error", error=str(exc),
                    reference_converged=False, error_kind="reference",
                    **provenance,
                ))
                continue
            try:
                test = run_nn_dc_lifted(tech, wd, vs0, device)
                validate_curve_trace(
                    test["sweep"], test["id"], expected_start=0.0,
                    expected_stop=tech.vdd, max_step=VG_STEP,
                    label=f"{device} candidate",
                )
            except OSError as exc:
                results.append(GateResult(
                    **identity, status="error", error=str(exc),
                    execution_state="infrastructure_error", error_kind="infrastructure",
                    **provenance,
                ))
                continue
            except (RuntimeError, ValueError) as exc:
                print(f"  {tk:<7s} {device} vs0={vs0:6.3f}V  ERROR candidate: {exc}")
                results.append(GateResult(
                    **identity, status="error", error=str(exc),
                    candidate_converged=False, error_kind="candidate",
                    **provenance,
                ))
                continue
            m = curve_metrics(ref["sweep"], ref["id"], test["sweep"], test["id"])
            ok = m["nrmse"] <= DC_NRMSE_PASS
            n_pass += int(ok)
            verdict = "" if args.no_gate else ("  PASS" if ok else "  FAIL")
            rows.append({"tech": tk, "device": device, "frac": frac, "vs0": vs0, **m,
                         "verdict": verdict.strip()})
            results.append(GateResult(
                **identity,
                status="diagnostic" if args.no_gate else "pass" if ok else "fail",
                metrics={
                    "nrmse_pct": float(m["nrmse"]), "mre_pct": float(m["mre"]),
                    "r2": float(m["r2"]), "max_err": float(m["max_err"]),
                },
                domain={"device": device, "vs0_v": float(vs0),
                        "vs0_fraction": float(frac)},
                **provenance,
            ))
            print(f"  {tk:<7s} {device} vs0={vs0:6.3f}V ({frac:.1f}*VDD)  "
                  f"NRMSE={m['nrmse']:7.2f}%  MRE={m['mre']:7.2f}%  "
                  f"R2={m['r2']:8.5f}  MaxErr={m['max_err'] * 1e6:9.3f}uA"
                  f"{verdict}")
            id_nn = np.interp(ref["sweep"], test["sweep"], test["id"])
            curve_rows += [[tk, device, frac, vs0, f"{vg:.4f}", f"{ig:.6e}", f"{inn:.6e}"]
                           for vg, ig, inn in zip(ref["sweep"], ref["id"], id_nn)]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"lifted_source_sweep_{args.label}.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tech", "device", "vs0_frac", "vs0_V", "gate_sweep_V",
                    "id_ngspice_A", "id_nn_A"])
        w.writerows(curve_rows)
    print(f"\n  [CSV] Raw curves saved: {csv_path}")

    md_path = RESULTS_DIR / f"lifted_source_sweep_{args.label}.md"
    lines = [
        f"# Lifted-source NMOS/PMOS Id-Vgs sweep — {args.label} (V7.7.2)",
        "",
        f"{active_model_label()} vs NGSPICE OSDI BSIM-CMG, absolute terminals"
        " Vs=vs0, Vb=0, Vd=VDD, Vg=0..VDD (5 mV grid)."
        " PMOS mirrors each voltage around VDD; its axis is VDD-Vg."
        " Currents retain polarity after device orientation.",
        "Note: Vd=VDD here; the 55-config grounded gate biases Vds=VDD/2,"
        " so vs0=0 is a qualitative (not bit-exact) control.",
        "",
        "| tech | device | vs0/VDD | vs0 (V) | NRMSE (%) | MRE (%) | R2 | MaxErr (uA)"
        " | verdict |",
        "|------|--------|---------|---------|-----------|---------|----|----------"
        "|---------|",
    ]
    lines += [f"| {r['tech']} | {r['device']} | {r['frac']:.1f} | {r['vs0']:.3f} "
              f"| {r['nrmse']:.2f} | {r['mre']:.2f} | {r['r2']:.5f} "
              f"| {r['max_err'] * 1e6:.3f} | {r['verdict'] or '-'} |"
              for r in rows]
    lines += [f"| {result.tech} | {result.analysis} | — | — | — | — | — | — "
              f"| ERROR: {result.error.replace('|', '/').replace(chr(10), ' ')} |"
              for result in results if result.status == "error"]
    md_path.write_text("\n".join(lines) + "\n")
    print(f"  [MD]  Summary saved: {md_path}")

    # audit B5n, defence in depth: `0 == 0` would otherwise exit green on an
    # empty run (e.g. a future tech-gating change or an emptied
    # VS0_FRACTIONS) even though nothing was measured.
    if not results:
        print("\nERROR: no configs ran — nothing under test")
        return 2
    for result in results:
        print(result.marker())
    total = len(results)
    if args.no_gate:
        print(f"\nRESULT: {total} diagnostic configs, "
              f"{sum(result.status == 'error' for result in results)} errors")
    else:
        print(f"\nRESULT: {n_pass}/{total} configs PASSED "
              f"(NRMSE <= {DC_NRMSE_PASS:.0f}%)")
    return result_exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
