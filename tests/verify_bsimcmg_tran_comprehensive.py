#!/usr/bin/env python3
"""
Comprehensive parametric transient verification: PyCircuitSim vs NGSPICE.

Sweeps VDD, Cload, input slew (tr/tf), and pulse width across 14 unique
configurations built from a shared baseline.  Each configuration runs an
NGSPICE reference and a PyCircuitSim simulation, then computes NRMSE.

Test Parameter Matrix (14 configs, 4 sweeps with shared baseline):
  Sweep 1 – VDD:   0.5, 0.6, *0.7*, 0.8 V        (Cload=10fF, tr=100ps, pw=0.8ns)
  Sweep 2 – Cload:  1, 5, *10*, 50, 100 fF         (VDD=0.7V, tr=100ps, pw=0.8ns)
  Sweep 3 – Slew:   10, 50, *100*, 500 ps           (VDD=0.7V, Cload=10fF, pw=0.8ns)
  Sweep 4 – PW:     0.2, 0.5, *0.8*, 2.0 ns         (VDD=0.7V, Cload=10fF, tr=100ps)

  *baseline* config (VDD=0.7V, 10fF, 100ps, 0.8ns) is shared across sweeps.
"""
from __future__ import annotations

import os
import sys
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project paths (same convention as verify_bsimcmg_tran.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "PyCMG"))

OSDI_PATH = (
    PROJECT_ROOT / "models" / "PyCMG" / "build-deep-verify" / "osdi" / "bsimcmg.osdi"
)
MODELCARD_PATH = (
    PROJECT_ROOT
    / "models"
    / "PyCMG"
    / "tech_model_cards"
    / "ASAP7"
    / "7nm_TT_160803.pm"
)
NGSPICE_BIN = "/usr/local/ngspice-45.2/bin/ngspice"
RESULTS_DIR = PROJECT_ROOT / "tests" / "verify_bsimcmg_tran_results" / "comprehensive"

# Baked-modelcard helper
from pycmg.testing import bake_inst_params

# PyCircuitSim imports (deferred to run-time to keep import lightweight)
# from pycircuitsim.parser import Parser
# from pycircuitsim.solver import DCSolver, TransientSolver

# ---------------------------------------------------------------------------
# Device constants
# ---------------------------------------------------------------------------
L: float = 30e-9
NFIN: int = 10
NMOS_INST_PARAMS: Dict[str, Any] = {"L": L, "NFIN": float(NFIN), "DEVTYPE": 1}
PMOS_INST_PARAMS: Dict[str, Any] = {"L": L, "NFIN": float(NFIN), "DEVTYPE": 0}

# Acceptance criterion
NRMSE_THRESHOLD: float = 0.05  # 5% of Vdd

# Startup exclusion (Gmin-stepping / pseudo-transient artifact)
STARTUP_EXCLUSION: float = 0.1e-9  # 0.1 ns


# ---------------------------------------------------------------------------
# TestConfig dataclass
# ---------------------------------------------------------------------------
@dataclass
class TestConfig:
    """One parametric test point for the inverter transient sweep."""

    vdd: float          # Supply voltage [V]
    cload: float        # Load capacitance [F]
    tr: float           # Input rise time [s]
    tf: float           # Input fall time [s]
    pw: float           # Pulse width [s]
    name: str           # Human-readable tag (e.g. "baseline", "vdd_0p5")
    sweep_type: str     # Grouping key: "vdd", "cload", "slew", "pw"

    # PULSE low / high voltages
    pulse_v1: float = 0.0
    pulse_v2: float = 0.0  # set in __post_init__

    def __post_init__(self) -> None:
        if self.pulse_v2 == 0.0:
            self.pulse_v2 = self.vdd

    # -- Adaptive timing (computed properties) --------------------------------

    @property
    def tau_est(self) -> float:
        """Rough RC time constant estimate."""
        return max(self.cload / 1e-4, 0.1e-9)

    @property
    def td(self) -> float:
        """Delay before first input edge."""
        return max(0.5e-9, 10.0 * self.tr)

    @property
    def rest(self) -> float:
        """Relaxation time after pulse high phase."""
        return max(self.pw, 5.0 * self.tau_est)

    @property
    def per(self) -> float:
        """Pulse period."""
        return self.tr + self.pw + self.tf + self.rest

    @property
    def tstop(self) -> float:
        """Total simulation time (2.5 cycles after initial delay)."""
        return self.td + 2.5 * self.per

    @property
    def tstep(self) -> float:
        """Simulation time step (resolves input transitions)."""
        return min(10e-12, self.tr / 10.0)

    # -- Pretty-print helpers -------------------------------------------------

    def summary(self) -> str:
        """One-line summary string."""
        return (
            f"{self.name:20s}  VDD={self.vdd:.2f}V  "
            f"Cload={self.cload*1e15:.0f}fF  "
            f"tr/tf={self.tr*1e12:.0f}ps  "
            f"pw={self.pw*1e9:.1f}ns  "
            f"tstop={self.tstop*1e9:.1f}ns  "
            f"tstep={self.tstep*1e12:.1f}ps"
        )


# ---------------------------------------------------------------------------
# Build the 14-config parameter matrix
# ---------------------------------------------------------------------------
def _build_configs() -> List[TestConfig]:
    """Construct the full test matrix (14 unique points)."""
    configs: List[TestConfig] = []

    # ---- Sweep 1: VDD (4 tests) -------------------------------------------
    for vdd in [0.5, 0.6, 0.7, 0.8]:
        tag = "baseline" if vdd == 0.7 else f"vdd_{vdd:.1f}".replace(".", "p")
        configs.append(
            TestConfig(
                vdd=vdd,
                cload=10e-15,
                tr=100e-12,
                tf=100e-12,
                pw=0.8e-9,
                name=tag,
                sweep_type="vdd",
            )
        )

    # ---- Sweep 2: Cload (4 more, 10fF is already the baseline) ------------
    for cload_fF in [1, 5, 50, 100]:
        configs.append(
            TestConfig(
                vdd=0.7,
                cload=cload_fF * 1e-15,
                tr=100e-12,
                tf=100e-12,
                pw=0.8e-9,
                name=f"cload_{cload_fF}fF",
                sweep_type="cload",
            )
        )

    # ---- Sweep 3: Input slew / tr+tf (3 more, 100ps is baseline) ----------
    for tr_ps in [10, 50, 500]:
        configs.append(
            TestConfig(
                vdd=0.7,
                cload=10e-15,
                tr=tr_ps * 1e-12,
                tf=tr_ps * 1e-12,
                pw=0.8e-9,
                name=f"slew_{tr_ps}ps",
                sweep_type="slew",
            )
        )

    # ---- Sweep 4: Pulse width (3 more, 0.8ns is baseline) -----------------
    for pw_ns in [0.2, 0.5, 2.0]:
        pw_tag = f"{pw_ns:.1f}".replace(".", "p")
        configs.append(
            TestConfig(
                vdd=0.7,
                cload=10e-15,
                tr=100e-12,
                tf=100e-12,
                pw=pw_ns * 1e-9,
                name=f"pw_{pw_tag}ns",
                sweep_type="pw",
            )
        )

    return configs


ALL_CONFIGS: List[TestConfig] = _build_configs()


# ---------------------------------------------------------------------------
# Baked modelcard (shared across all configs)
# ---------------------------------------------------------------------------
def create_baked_modelcard() -> Path:
    """Create a combined baked modelcard for NGSPICE OSDI (once per run)."""
    combined = RESULTS_DIR / "combined_baked.lib"
    bake_inst_params(MODELCARD_PATH, combined, "nmos_rvt", NMOS_INST_PARAMS)
    bake_inst_params(combined, combined, "pmos_rvt", PMOS_INST_PARAMS)
    print(f"[NGSPICE] Baked modelcard: {combined}")
    return combined


# ---------------------------------------------------------------------------
# NGSPICE netlist generation & runner
# ---------------------------------------------------------------------------
def create_ngspice_netlist(config: TestConfig, baked_lib: Path) -> Path:
    """Generate an NGSPICE inverter transient netlist for *one* config.

    The netlist uses OSDI-style device names (N prefix) with instance params
    baked into the modelcard.  All timing / voltage values are drawn from
    ``config`` so the same function works for every parametric point.

    Returns the path to the written ``.cir`` file.
    """
    netlist_path = RESULTS_DIR / f"ngspice_{config.name}.cir"
    content = f"""\
* BSIM-CMG CMOS Inverter Transient - NGSPICE ({config.name})
.include "{baked_lib}"
.temp 27
Vdd vdd 0 {config.vdd}
Vin in 0 PULSE({config.pulse_v1} {config.pulse_v2} {config.td} {config.tr} {config.tf} {config.pw} {config.per})
Np out in vdd vdd pmos_rvt
Nn out in 0 0 nmos_rvt
Cload out 0 {config.cload}
.ic V(out)={config.vdd}
.tran {config.tstep} {config.tstop} uic
.end
"""
    netlist_path.write_text(content)
    print(f"[NGSPICE] Netlist ({config.name}): {netlist_path}")
    return netlist_path


def run_ngspice(netlist_path: Path, config: TestConfig) -> Dict[str, np.ndarray]:
    """Run NGSPICE transient simulation and parse ``wrdata`` output.

    Creates a lightweight runner script that loads the OSDI binary, sources the
    netlist, runs the simulation, and writes the result to a CSV via ``wrdata``.

    The wrdata format has a header line, then rows with:
        time  v(out)  time  v(in)
    Columns 0/1 carry time & v(out); column 3 carries v(in).

    Returns a dict with numpy arrays: ``time``, ``v(out)``, ``v(in)``.
    """
    csv_path = RESULTS_DIR / f"ngspice_{config.name}.csv"
    log_path = RESULTS_DIR / f"ngspice_{config.name}.log"
    runner_path = RESULTS_DIR / f"ngspice_{config.name}_runner.cir"

    runner_content = f"""\
* NGSPICE transient runner ({config.name})
.control
osdi {OSDI_PATH}
source {netlist_path}
set filetype=ascii
set wr_vecnames
run
wrdata {csv_path} v(out) v(in)
.endc
.end
"""
    runner_path.write_text(runner_content)

    print(f"[NGSPICE] Running simulation ({config.name})...")
    res = subprocess.run(
        [NGSPICE_BIN, "-b", "-o", str(log_path), str(runner_path)],
        capture_output=True,
        text=True,
    )

    if not csv_path.exists():
        log_content = log_path.read_text() if log_path.exists() else "(no log)"
        raise RuntimeError(
            f"NGSPICE produced no output: {csv_path}\n"
            f"RC={res.returncode}, log (tail): ...{log_content[-500:]}\n"
        )

    # Parse wrdata: header + data rows  (time, v(out), time, v(in))
    with csv_path.open() as f:
        lines = f.readlines()

    data_rows: list[list[float]] = []
    for line in lines[1:]:          # skip header
        stripped = line.strip()
        if not stripped:
            continue
        vals = [float(x) for x in stripped.split()]
        data_rows.append(vals)

    data = np.array(data_rows)
    result: Dict[str, np.ndarray] = {
        "time": data[:, 0],
        "v(out)": data[:, 1],
        "v(in)": data[:, 3],
    }
    print(
        f"[NGSPICE] Done ({config.name}): {len(result['time'])} pts, "
        f"V(out) [{result['v(out)'].min():.4f}, {result['v(out)'].max():.4f}]V"
    )
    return result


# ---------------------------------------------------------------------------
# Stub: single-test runner (to be implemented in subsequent tasks)
# ---------------------------------------------------------------------------
def run_single_test(
    config: TestConfig,
    baked_lib: Path,
) -> Dict[str, Any]:
    """Run NGSPICE + PyCircuitSim for *one* config and return metrics.

    Returns a dict with at least:
        "config": TestConfig
        "nrmse_post": float          # post-settling NRMSE (fraction)
        "nrmse_full": float          # full-range NRMSE (fraction)
        "max_err_mV": float          # max |error| in mV
        "passed": bool               # nrmse_post < NRMSE_THRESHOLD
    """
    # TODO: implement in subsequent tasks
    raise NotImplementedError(f"run_single_test not yet implemented for {config.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    """Run the full parametric transient verification suite."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("BSIM-CMG Comprehensive Transient Verification")
    print(f"  {len(ALL_CONFIGS)} configurations across 4 parameter sweeps")
    print(f"  Acceptance: NRMSE < {NRMSE_THRESHOLD*100:.0f}% of Vdd (post-settling)")
    print(f"  Startup exclusion: {STARTUP_EXCLUSION*1e9:.1f}ns")
    print("=" * 78)

    # Print the full test matrix
    print("\nTest matrix:")
    for i, cfg in enumerate(ALL_CONFIGS):
        print(f"  [{i+1:2d}] {cfg.summary()}")

    # Bake modelcard once
    print("\n--- Baking modelcard ---")
    baked_lib = create_baked_modelcard()

    # Run each configuration
    results: List[Dict[str, Any]] = []
    n_pass = 0
    n_fail = 0
    n_error = 0

    for i, cfg in enumerate(ALL_CONFIGS):
        print(f"\n{'='*78}")
        print(f"[{i+1}/{len(ALL_CONFIGS)}] {cfg.name}  (sweep={cfg.sweep_type})")
        print(f"  VDD={cfg.vdd:.2f}V  Cload={cfg.cload*1e15:.0f}fF  "
              f"tr/tf={cfg.tr*1e12:.0f}ps  pw={cfg.pw*1e9:.1f}ns")
        print(f"  td={cfg.td*1e9:.1f}ns  per={cfg.per*1e9:.1f}ns  "
              f"tstop={cfg.tstop*1e9:.1f}ns  tstep={cfg.tstep*1e12:.1f}ps")

        try:
            result = run_single_test(cfg, baked_lib)
            results.append(result)
            if result["passed"]:
                n_pass += 1
                print(f"  => PASS  NRMSE={result['nrmse_post']*100:.2f}%")
            else:
                n_fail += 1
                print(f"  => FAIL  NRMSE={result['nrmse_post']*100:.2f}%")
        except NotImplementedError:
            n_error += 1
            print("  => SKIPPED (run_single_test not yet implemented)")
        except Exception as exc:
            n_error += 1
            print(f"  => ERROR: {exc}")
            results.append({"config": cfg, "error": str(exc), "passed": False})

    # Summary
    print(f"\n{'='*78}")
    print("SUMMARY")
    print(f"  Total : {len(ALL_CONFIGS)}")
    print(f"  Pass  : {n_pass}")
    print(f"  Fail  : {n_fail}")
    print(f"  Error : {n_error}")
    print(f"{'='*78}")

    if n_fail > 0 or n_error > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
