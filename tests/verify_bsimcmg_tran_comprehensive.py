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
# Stub: single-test runner (to be implemented in Task 2+)
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
