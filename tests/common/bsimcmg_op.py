"""Shared infrastructure for the BSIM-CMG (LEVEL=72) operating-point gates.

Split out of the former monolithic flat ``tests/verify_bsimcmg_op.py`` in
V7.5.8, when gates were regrouped by circuit tier: the NMOS and PMOS checks are
single-device gates and the inverter check is a simple-circuit gate, so they
live in different packages now and share this module instead of a file.

Everything here is ASAP7 at one geometry (L=30 nm, NFIN=10, VDD=0.7 V). Both
gates pin the SAME geometry deliberately — it is what makes the inverter
result attributable: if the devices agree and the inverter does not, the gap
is in the circuit, not the model.

Key NGSPICE OSDI notes:
  - OSDI devices use the generic prefix (N), NOT the MOSFET prefix (M).
  - The .model type must be 'bsimcmg' (bake_inst_params does this).
  - DEVTYPE must be baked: 1=NMOS, 0=PMOS (the ASAP7 modelcard lacks it).
  - Instance params (L, NFIN) cannot go on the device line for OSDI.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "bsim_cmg"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "bsim_cmg" / "tests"))

OSDI_PATH = (PROJECT_ROOT / "external_compact_models" / "bsim_cmg" / "build"
             / "osdi" / "bsimcmg.osdi")
MODELCARD_PATH = (PROJECT_ROOT / "PDKs" / "ASAP7"
                  / "7nm_TT_160803.pm")
NGSPICE_BIN = os.environ.get("NGSPICE_BIN",
                             "/usr/local/ngspice-45.2/bin/ngspice")
RESULTS_DIR = PROJECT_ROOT / "tests" / "verify_bsimcmg_op_results"

from helpers import bake_inst_params, run_ngspice_op  # noqa: E402,F401

from pycircuitsim.parser import Parser  # noqa: E402
from pycircuitsim.solver import DCSolver  # noqa: E402

# -- Shared test parameters --------------------------------------------------
L = 30e-9        # 30nm channel length
NFIN = 10        # 10 fins
VDD = 0.7        # ASAP7 nominal Vdd
REL_TOL = 0.01   # 1% relative tolerance
ABS_TOL_I = 1e-9  # 1 nA absolute tolerance floor (for near-zero currents)
ABS_TOL_V = 1e-4  # 0.1 mV absolute tolerance floor (for voltages)

# BSIM-CMG DEVTYPE distinguishes NMOS (1) from PMOS (0). The ASAP7 modelcard
# does NOT carry it; PyCMG auto-injects it but NGSPICE OSDI does not, so it
# must be baked into the modelcard for the NGSPICE side.
NMOS_INST_PARAMS = {"L": L, "NFIN": float(NFIN), "DEVTYPE": 1}
PMOS_INST_PARAMS = {"L": L, "NFIN": float(NFIN), "DEVTYPE": 0}


def relative_error(measured: float, reference: float, abs_tol: float) -> float:
    """Relative error with an absolute-tolerance floor."""
    diff = abs(measured - reference)
    denom = max(abs(reference), abs_tol)
    return diff / denom


def pass_fail(rel_err: float, threshold: float = REL_TOL) -> str:
    """Return the PASS/FAIL string for a relative error."""
    return "PASS" if rel_err <= threshold else "FAIL"


def run_pycircuitsim_op(netlist_content: str) -> Dict[str, float]:
    """Run a PyCircuitSim OP analysis on netlist text."""
    tmpdir = tempfile.mkdtemp(prefix="pycircuitsim_op_")
    try:
        netlist_path = Path(tmpdir) / "circuit.sp"
        netlist_path.write_text(netlist_content)

        parser = Parser()
        parser.parse_file(str(netlist_path))
        circuit = parser.circuit

        initial_guess = (circuit.initial_conditions
                         if circuit.initial_conditions else None)
        solver = DCSolver(
            circuit,
            initial_guess=initial_guess,
            use_source_stepping=True,
            source_stepping_steps=20,
        )
        return solver.solve()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def get_mosfet_current_from_solution(
    netlist_content: str,
    solution: Dict[str, float],
    mosfet_name: str,
) -> float:
    """Extract a MOSFET's drain current from a solved circuit."""
    tmpdir = tempfile.mkdtemp(prefix="pycircuitsim_cur_")
    try:
        netlist_path = Path(tmpdir) / "circuit.sp"
        netlist_path.write_text(netlist_content)

        parser = Parser()
        parser.parse_file(str(netlist_path))
        circuit = parser.circuit

        for comp in circuit.components:
            if comp.name.lower() == mosfet_name.lower():
                return comp.calculate_current(solution)

        raise ValueError(f"MOSFET '{mosfet_name}' not found in circuit")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_ngspice_custom(runner_path: Path, log_path: Path,
                       csv_path: Path) -> Dict[str, float]:
    """Run NGSPICE with a custom runner script and parse its CSV output."""
    res = subprocess.run(
        [NGSPICE_BIN, "-b", "-o", str(log_path), str(runner_path)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        log_content = log_path.read_text() if log_path.exists() else ""
        raise RuntimeError(
            f"NGSPICE failed (rc={res.returncode}):\n"
            f"  stdout: {res.stdout[:300]}\n"
            f"  stderr: {res.stderr[:300]}\n"
            f"  log (last 500 chars): ...{log_content[-500:]}\n"
        )

    if not csv_path.exists():
        log_content = log_path.read_text() if log_path.exists() else "(no log)"
        raise RuntimeError(
            f"NGSPICE produced no CSV output: {csv_path}\n"
            f"  log (last 500 chars): ...{log_content[-500:]}\n"
        )

    with csv_path.open() as f:
        csv_lines = f.readlines()
    if not csv_lines:
        raise RuntimeError(f"Empty NGSPICE CSV: {csv_path}")
    headers = csv_lines[0].split()
    values = [float(x) for x in csv_lines[1].split()]
    return {name: values[i] for i, name in enumerate(headers)}
