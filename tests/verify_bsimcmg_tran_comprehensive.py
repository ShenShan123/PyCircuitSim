#!/usr/bin/env python3
"""
Comprehensive parametric transient verification: PyCircuitSim vs NGSPICE.

Sweeps VDD, Cload, input slew (tr/tf), pulse width, NFIN scaling, and P/N
ratio across 21 unique configurations built from a shared baseline.  Each
configuration runs an NGSPICE reference and a PyCircuitSim simulation, then
computes NRMSE.

Test Parameter Matrix (21 configs, 6 sweeps with shared baseline):
  Sweep 1 – VDD:      0.5, 0.6, *0.7*, 0.8 V          (NFIN=10/10, Cload=10fF, tr=100ps, pw=0.8ns)
  Sweep 2 – Cload:    1, 5, *10*, 50, 100 fF            (NFIN=10/10, VDD=0.7V, tr=100ps, pw=0.8ns)
  Sweep 3 – Slew:     10, 50, *100*, 500 ps              (NFIN=10/10, VDD=0.7V, Cload=10fF, pw=0.8ns)
  Sweep 4 – PW:       0.2, 0.5, *0.8*, 2.0 ns            (NFIN=10/10, VDD=0.7V, Cload=10fF, tr=100ps)
  Sweep 5 – NFIN:     1, 2, 5, *10*, 20  (equal P/N)     (VDD=0.7V, Cload=10fF, tr=100ps, pw=0.8ns)
  Sweep 6 – P/N ratio: 0.5, *1.0*, 1.5, 2.0              (NFIN_N=10, VDD=0.7V, Cload=10fF, tr=100ps, pw=0.8ns)

  *baseline* config (VDD=0.7V, NFIN=10/10, 10fF, 100ps, 0.8ns) is shared across sweeps.
"""
from __future__ import annotations

import os
import sys
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project paths (same convention as verify_bsimcmg_tran.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))

OSDI_PATH = (
    PROJECT_ROOT / "external_compact_models" / "PyCMG" / "build-deep-verify" / "osdi" / "bsimcmg.osdi"
)
MODELCARD_PATH = (
    PROJECT_ROOT
    / "external_compact_models"
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
    sweep_type: str     # Grouping key: "vdd", "cload", "slew", "pw", "nfin", "pn_ratio"

    # Geometry (fin counts)
    nfin_n: int = 10    # NMOS fin count
    nfin_p: int = 10    # PMOS fin count

    # PULSE low / high voltages
    pulse_v1: float = 0.0
    pulse_v2: float = 0.0  # set in __post_init__

    def __post_init__(self) -> None:
        if self.pulse_v2 == 0.0:
            self.pulse_v2 = self.vdd

    # -- Adaptive timing (computed properties) --------------------------------

    @property
    def tau_est(self) -> float:
        """Rough RC time constant estimate, scaled by fin count."""
        nfin_min = min(self.nfin_n, self.nfin_p)
        i_est = nfin_min * 1e-5  # ~10µA per fin (ASAP7 7nm estimate)
        return max(self.cload * self.vdd / i_est, 0.1e-9)

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
            f"NFIN={self.nfin_n}/{self.nfin_p}  "
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

    # ---- Sweep 5: NFIN scaling, equal P/N (4 more, NFIN=10 is baseline) --
    for nfin in [1, 2, 5, 20]:
        configs.append(
            TestConfig(
                vdd=0.7,
                cload=10e-15,
                tr=100e-12,
                tf=100e-12,
                pw=0.8e-9,
                name=f"nfin_{nfin}",
                sweep_type="nfin",
                nfin_n=nfin,
                nfin_p=nfin,
            )
        )

    # ---- Sweep 6: P/N ratio (3 more, P/N=1.0 is baseline) ---------------
    for nfin_p, tag in [(5, "0p5"), (15, "1p5"), (20, "2p0")]:
        configs.append(
            TestConfig(
                vdd=0.7,
                cload=10e-15,
                tr=100e-12,
                tf=100e-12,
                pw=0.8e-9,
                name=f"pn_{tag}",
                sweep_type="pn_ratio",
                nfin_n=10,
                nfin_p=nfin_p,
            )
        )

    return configs


ALL_CONFIGS: List[TestConfig] = _build_configs()


# ---------------------------------------------------------------------------
# Per-geometry baked modelcard cache
# ---------------------------------------------------------------------------
_baked_cache: Dict[Tuple[int, int], Path] = {}


def get_or_create_baked_modelcard(nfin_n: int, nfin_p: int) -> Path:
    """Return a baked modelcard for the given NFIN geometry, creating if needed."""
    key = (nfin_n, nfin_p)
    if key in _baked_cache:
        return _baked_cache[key]
    baked = RESULTS_DIR / f"baked_nfin_n{nfin_n}_p{nfin_p}.lib"
    nmos_params: Dict[str, Any] = {"L": L, "NFIN": float(nfin_n), "DEVTYPE": 1}
    pmos_params: Dict[str, Any] = {"L": L, "NFIN": float(nfin_p), "DEVTYPE": 0}
    bake_inst_params(MODELCARD_PATH, baked, "nmos_rvt", nmos_params)
    bake_inst_params(baked, baked, "pmos_rvt", pmos_params)
    _baked_cache[key] = baked
    print(f"[NGSPICE] Baked modelcard (NFIN_N={nfin_n}, NFIN_P={nfin_p}): {baked}")
    return baked


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
# PyCircuitSim netlist generation & runner
# ---------------------------------------------------------------------------
def create_pycircuitsim_netlist(config: TestConfig) -> Path:
    """Generate a PyCircuitSim inverter transient netlist for *one* config.

    Node mapping: 1=Vdd, 2=Vin, 3=Vout.
    Uses LEVEL=72 model definitions for BSIM-CMG via PyCMG.

    Returns the path to the written ``.sp`` file.
    """
    netlist_path = RESULTS_DIR / f"pycircuitsim_{config.name}.sp"
    content = f"""\
* BSIM-CMG Inverter Transient - PyCircuitSim ({config.name})
* VDD={config.vdd}V, L={L*1e9:.0f}n, NFIN_N={config.nfin_n}, NFIN_P={config.nfin_p}, Cload={config.cload*1e15:.0f}fF

* Power supply
Vdd 1 0 {config.vdd}

* Input pulse: {config.pulse_v1} -> {config.pulse_v2}V
Vin 2 0 PULSE {config.pulse_v1} {config.pulse_v2} {config.td} {config.tr} {config.tf} {config.pw} {config.per}

* PMOS (drain=out, gate=in, source=Vdd, bulk=Vdd)
Mp1 3 2 1 1 pmos1 L={L*1e9:.0f}n NFIN={config.nfin_p}

* NMOS (drain=out, gate=in, source=GND, bulk=GND)
Mn1 3 2 0 0 nmos1 L={L*1e9:.0f}n NFIN={config.nfin_n}

* Load capacitance
Cload 3 0 {config.cload}

* Initial condition: output starts high (PMOS on, NMOS off when Vin=0)
.ic V(3)={config.vdd}

* Model definitions (LEVEL=72 BSIM-CMG)
.model nmos1 NMOS (LEVEL=72)
.model pmos1 PMOS (LEVEL=72)

* Transient: {config.tstep*1e12:.1f}ps step, {config.tstop*1e9:.1f}ns total
.tran {config.tstep} {config.tstop}

.end
"""
    netlist_path.write_text(content)
    print(f"[PySim] Netlist ({config.name}): {netlist_path}")
    return netlist_path


def run_pycircuitsim(netlist_path: Path, config: TestConfig) -> Dict[str, np.ndarray]:
    """Run PyCircuitSim transient simulation for *one* config.

    Parses the netlist, runs DC operating point (with source stepping),
    then runs transient analysis with Gmin stepping and pseudo-transient
    initialization.

    Returns a dict with numpy arrays: ``time``, ``v(out)``, ``v(in)``.
    """
    import logging

    from pycircuitsim.parser import Parser
    from pycircuitsim.solver import DCSolver, TransientSolver

    # Suppress verbose logging during simulation
    logging.disable(logging.CRITICAL)

    parser = Parser()
    parser.parse_file(str(netlist_path))
    circuit = parser.circuit

    time_step: float = parser.analysis_params["tstep"]
    final_time: float = parser.analysis_params["tstop"]

    # Stage 1: DC operating point
    initial_guess = circuit.initial_conditions if circuit.initial_conditions else None
    op_solver = DCSolver(circuit, initial_guess=initial_guess, use_source_stepping=True)
    op_solution = op_solver.solve()

    # Stage 2: Transient analysis
    solver = TransientSolver(
        circuit,
        t_stop=final_time,
        dt=time_step,
        initial_guess=op_solution,
        use_gmin_stepping=True,
        gmin_initial=1e-9,
        gmin_final=1e-12,
        gmin_steps=5,
        use_pseudo_transient=True,
        pseudo_transient_steps=5,
        pseudo_transient_cap=1e-12,
        debug=False,
    )
    results = solver.solve()

    # Restore logging
    logging.disable(logging.NOTSET)

    # Node mapping: '1'=Vdd, '2'=Vin, '3'=Vout
    result: Dict[str, np.ndarray] = {
        "time": results["time"],
        "v(out)": results["3"],
        "v(in)": results["2"],
    }
    print(
        f"[PySim] Done ({config.name}): {len(result['time'])} pts, "
        f"V(out) [{result['v(out)'].min():.4f}, {result['v(out)'].max():.4f}]V"
    )
    return result


# ---------------------------------------------------------------------------
# Comparison, metrics, and plotting
# ---------------------------------------------------------------------------
SWEEP_COLORS: Dict[str, str] = {
    "vdd": "tab:blue",
    "cload": "tab:green",
    "slew": "tab:orange",
    "pw": "tab:purple",
    "nfin": "tab:red",
    "pn_ratio": "tab:brown",
}


def interpolate_to_common_time(
    ng_data: Dict[str, np.ndarray],
    py_data: Dict[str, np.ndarray],
    config: TestConfig,
    t_start: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate both datasets to a common uniform time grid.

    Uses ``config.tstep`` for the grid spacing (varies per config).

    Args:
        ng_data: NGSPICE results with keys ``time``, ``v(out)``, ``v(in)``.
        py_data: PyCircuitSim results with the same keys.
        config: Test configuration (provides ``tstep``).
        t_start: Start time for comparison (excludes startup artifacts).

    Returns:
        (time_common, ng_vout, py_vout, ng_vin, py_vin)
    """
    t_max = min(ng_data["time"][-1], py_data["time"][-1])
    t_common = np.arange(max(t_start, ng_data["time"][0]), t_max, config.tstep)

    ng_vout = np.interp(t_common, ng_data["time"], ng_data["v(out)"])
    py_vout = np.interp(t_common, py_data["time"], py_data["v(out)"])
    ng_vin = np.interp(t_common, ng_data["time"], ng_data["v(in)"])
    py_vin = np.interp(t_common, py_data["time"], py_data["v(in)"])

    return t_common, ng_vout, py_vout, ng_vin, py_vin


def compute_metrics(
    ng_vout: np.ndarray,
    py_vout: np.ndarray,
    vdd: float,
) -> Dict[str, float]:
    """Compute error metrics between NGSPICE and PyCircuitSim waveforms.

    Returns dict with RMSE (V, mV), NRMSE (% of Vdd), max |error| (V, mV, % of Vdd).
    """
    diff = py_vout - ng_vout
    rmse = np.sqrt(np.mean(diff ** 2))
    nrmse = rmse / vdd
    max_abs_err = np.max(np.abs(diff))

    return {
        "RMSE (V)": rmse,
        "RMSE (mV)": rmse * 1e3,
        "NRMSE (% of Vdd)": nrmse * 100,
        "Max |error| (V)": max_abs_err,
        "Max |error| (mV)": max_abs_err * 1e3,
        "Max |error| (% of Vdd)": max_abs_err / vdd * 100,
    }


def plot_single_comparison(
    ng_data: Dict[str, np.ndarray],
    py_data: Dict[str, np.ndarray],
    config: TestConfig,
    metrics_full: Dict[str, float],
    metrics_post: Dict[str, float],
    save_path: Path,
) -> None:
    """Generate a 3-panel comparison plot for one config.

    Panels:
        1. V(in) — input stimulus overlay
        2. V(out) — output response overlay with metrics text box
        3. Error trace (post-settling)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9),
                             gridspec_kw={"height_ratios": [0.8, 1.2, 0.8]})

    ng_t_ns = ng_data["time"] * 1e9
    py_t_ns = py_data["time"] * 1e9

    # --- Panel 1: V(in) ---
    ax1 = axes[0]
    ax1.plot(ng_t_ns, ng_data["v(in)"], "b-", label="NGSPICE", linewidth=1.5)
    ax1.plot(py_t_ns, py_data["v(in)"], "r--", label="PyCircuitSim",
             linewidth=1.2, alpha=0.8)
    ax1.set_ylabel("V(in) [V]")
    ax1.set_title(
        f"BSIM-CMG Inverter Transient: {config.name}\n"
        f"VDD={config.vdd:.2f}V  NFIN={config.nfin_n}/{config.nfin_p}  "
        f"Cload={config.cload*1e15:.0f}fF  "
        f"tr/tf={config.tr*1e12:.0f}ps  pw={config.pw*1e9:.1f}ns"
    )
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, config.vdd + 0.1)

    # --- Panel 2: V(out) ---
    ax2 = axes[1]
    ax2.plot(ng_t_ns, ng_data["v(out)"], "b-", label="NGSPICE", linewidth=1.5)
    ax2.plot(py_t_ns, py_data["v(out)"], "r--", label="PyCircuitSim",
             linewidth=1.2, alpha=0.8)
    ax2.axvline(x=STARTUP_EXCLUSION * 1e9, color="gray", linewidth=1,
                linestyle=":", label=f"Startup excl ({STARTUP_EXCLUSION*1e9:.1f}ns)")
    ax2.set_ylabel("V(out) [V]")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, config.vdd + 0.15)

    # Metrics text box
    txt = (
        f"Full: NRMSE={metrics_full['NRMSE (% of Vdd)']:.2f}%\n"
        f"Post-settling: NRMSE={metrics_post['NRMSE (% of Vdd)']:.2f}%, "
        f"Max|err|={metrics_post['Max |error| (mV)']:.1f}mV"
    )
    ax2.text(
        0.02, 0.05, txt, transform=ax2.transAxes, fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # --- Panel 3: Error trace (post-settling) ---
    ax3 = axes[2]
    t_c, ng_v, py_v, _, _ = interpolate_to_common_time(
        ng_data, py_data, config, t_start=STARTUP_EXCLUSION,
    )
    error_mv = (py_v - ng_v) * 1e3
    ax3.plot(t_c * 1e9, error_mv, "g-", linewidth=0.8)
    ax3.axhline(y=0, color="k", linewidth=0.5)
    threshold_mv = config.vdd * NRMSE_THRESHOLD * 1e3
    ax3.axhline(y=threshold_mv, color="r", linewidth=0.5, linestyle="--",
                label=f"{NRMSE_THRESHOLD*100:.0f}% Vdd = {threshold_mv:.0f}mV")
    ax3.axhline(y=-threshold_mv, color="r", linewidth=0.5, linestyle="--")
    ax3.set_ylabel("Error [mV]")
    ax3.set_xlabel("Time [ns]")
    ax3.set_title(f"Error (post {STARTUP_EXCLUSION*1e9:.1f}ns settling)", fontsize=10)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {save_path}")


# ---------------------------------------------------------------------------
# Single-test orchestrator
# ---------------------------------------------------------------------------
def run_single_test(config: TestConfig) -> Dict[str, Any]:
    """Run NGSPICE + PyCircuitSim for *one* config and return metrics.

    Orchestrates the full pipeline for a single parametric point:
        1. Create NGSPICE netlist and run simulation
        2. Create PyCircuitSim netlist and run simulation
        3. Compute full-range and post-settling metrics
        4. Generate comparison plot
        5. Return metrics dict

    Returns a dict with:
        "config": TestConfig
        "nrmse_post": float          # post-settling NRMSE (fraction)
        "nrmse_full": float          # full-range NRMSE (fraction)
        "max_err_mV": float          # max |error| in mV
        "passed": bool               # nrmse_post < NRMSE_THRESHOLD
    """
    # 1. NGSPICE reference
    baked_lib = get_or_create_baked_modelcard(config.nfin_n, config.nfin_p)
    ng_netlist = create_ngspice_netlist(config, baked_lib)
    ng_data = run_ngspice(ng_netlist, config)

    # 2. PyCircuitSim
    py_netlist = create_pycircuitsim_netlist(config)
    py_data = run_pycircuitsim(py_netlist, config)

    # 3. Compute metrics — full range
    _, ng_vout_full, py_vout_full, _, _ = interpolate_to_common_time(
        ng_data, py_data, config, t_start=0.0,
    )
    metrics_full = compute_metrics(ng_vout_full, py_vout_full, config.vdd)

    # 3b. Compute metrics — post-settling
    _, ng_vout_post, py_vout_post, _, _ = interpolate_to_common_time(
        ng_data, py_data, config, t_start=STARTUP_EXCLUSION,
    )
    metrics_post = compute_metrics(ng_vout_post, py_vout_post, config.vdd)

    nrmse_post = metrics_post["NRMSE (% of Vdd)"] / 100.0
    nrmse_full = metrics_full["NRMSE (% of Vdd)"] / 100.0
    max_err_mV = metrics_post["Max |error| (mV)"]
    passed = nrmse_post < NRMSE_THRESHOLD

    # 4. Plot
    plot_path = RESULTS_DIR / f"comparison_{config.name}.png"
    plot_single_comparison(ng_data, py_data, config, metrics_full, metrics_post, plot_path)

    # Print metrics summary for this config
    print(f"  Full-range  : NRMSE={nrmse_full*100:.2f}%")
    print(f"  Post-settling: NRMSE={nrmse_post*100:.2f}%, Max|err|={max_err_mV:.1f}mV")

    return {
        "config": config,
        "nrmse_post": nrmse_post,
        "nrmse_full": nrmse_full,
        "max_err_mV": max_err_mV,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------
def plot_summary(results: List[Dict[str, Any]], save_path: Path) -> None:
    """Generate a bar chart summarising NRMSE across all configs.

    Bars are colour-coded by sweep type and a horizontal line marks the
    acceptance threshold.
    """
    # Filter to results that have actual metrics (skip errors)
    valid = [r for r in results if "nrmse_post" in r]
    if not valid:
        print("[Plot] No valid results to plot summary.")
        return

    names = [r["config"].name for r in valid]
    nrmse_pct = [r["nrmse_post"] * 100 for r in valid]
    colors = [SWEEP_COLORS.get(r["config"].sweep_type, "tab:gray") for r in valid]

    fig, ax = plt.subplots(figsize=(max(10, len(valid) * 0.8), 5))
    x = np.arange(len(valid))
    bars = ax.bar(x, nrmse_pct, color=colors, edgecolor="black", linewidth=0.5)

    # Threshold line
    ax.axhline(y=NRMSE_THRESHOLD * 100, color="red", linewidth=1.5, linestyle="--",
               label=f"Threshold ({NRMSE_THRESHOLD*100:.0f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("NRMSE (% of Vdd)")
    ax.set_title("Comprehensive Transient Verification: NRMSE Summary")

    # Legend for sweep types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SWEEP_COLORS["vdd"], edgecolor="black", label="VDD sweep"),
        Patch(facecolor=SWEEP_COLORS["cload"], edgecolor="black", label="Cload sweep"),
        Patch(facecolor=SWEEP_COLORS["slew"], edgecolor="black", label="Slew sweep"),
        Patch(facecolor=SWEEP_COLORS["pw"], edgecolor="black", label="PW sweep"),
        Patch(facecolor=SWEEP_COLORS["nfin"], edgecolor="black", label="NFIN sweep"),
        Patch(facecolor=SWEEP_COLORS["pn_ratio"], edgecolor="black", label="P/N ratio sweep"),
        plt.Line2D([0], [0], color="red", linewidth=1.5, linestyle="--",
                    label=f"Threshold ({NRMSE_THRESHOLD*100:.0f}%)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Summary saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    """Run the full parametric transient verification suite."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("BSIM-CMG Comprehensive Transient Verification")
    print(f"  {len(ALL_CONFIGS)} configurations across 6 parameter sweeps")
    print(f"  Acceptance: NRMSE < {NRMSE_THRESHOLD*100:.0f}% of Vdd (post-settling)")
    print(f"  Startup exclusion: {STARTUP_EXCLUSION*1e9:.1f}ns")
    print("=" * 78)

    # Print the full test matrix
    print("\nTest matrix:")
    for i, cfg in enumerate(ALL_CONFIGS):
        print(f"  [{i+1:2d}] {cfg.summary()}")

    # Run each configuration
    results: List[Dict[str, Any]] = []
    n_pass = 0
    n_fail = 0
    n_error = 0

    for i, cfg in enumerate(ALL_CONFIGS):
        print(f"\n{'='*78}")
        print(f"[{i+1}/{len(ALL_CONFIGS)}] {cfg.name}  (sweep={cfg.sweep_type})")
        print(f"  VDD={cfg.vdd:.2f}V  NFIN={cfg.nfin_n}/{cfg.nfin_p}  "
              f"Cload={cfg.cload*1e15:.0f}fF  tr/tf={cfg.tr*1e12:.0f}ps  pw={cfg.pw*1e9:.1f}ns")
        print(f"  td={cfg.td*1e9:.1f}ns  per={cfg.per*1e9:.1f}ns  "
              f"tstop={cfg.tstop*1e9:.1f}ns  tstep={cfg.tstep*1e12:.1f}ps")

        try:
            result = run_single_test(cfg)
            results.append(result)
            if result["passed"]:
                n_pass += 1
                print(f"  => PASS  NRMSE={result['nrmse_post']*100:.2f}%")
            else:
                n_fail += 1
                print(f"  => FAIL  NRMSE={result['nrmse_post']*100:.2f}%")
        except Exception as exc:
            n_error += 1
            print(f"  => ERROR: {exc}")
            results.append({
                "config": cfg,
                "error": str(exc),
                "passed": False,
            })

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n{'='*78}")
    print("SUMMARY TABLE")
    print(f"{'='*78}")
    header = (
        f"{'Config Name':20s} | {'VDD':>5s} | {'NFIN_N':>6s} | {'NFIN_P':>6s} | "
        f"{'Cload':>7s} | {'tr/tf':>7s} | {'pw':>6s} | "
        f"{'NRMSE(%)':>9s} | {'MaxErr(mV)':>10s} | {'Status':>6s}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        cfg: TestConfig = r["config"]
        if "error" in r:
            print(
                f"{cfg.name:20s} | {cfg.vdd:5.2f} | {cfg.nfin_n:6d} | {cfg.nfin_p:6d} | "
                f"{cfg.cload*1e15:5.0f}fF | {cfg.tr*1e12:5.0f}ps | {cfg.pw*1e9:4.1f}ns | "
                f"{'ERROR':>9s} | {'ERROR':>10s} | {'ERROR':>6s}"
            )
        else:
            status = "PASS" if r["passed"] else "FAIL"
            print(
                f"{cfg.name:20s} | {cfg.vdd:5.2f} | {cfg.nfin_n:6d} | {cfg.nfin_p:6d} | "
                f"{cfg.cload*1e15:5.0f}fF | {cfg.tr*1e12:5.0f}ps | {cfg.pw*1e9:4.1f}ns | "
                f"{r['nrmse_post']*100:9.2f} | {r['max_err_mV']:10.1f} | "
                f"{status:>6s}"
            )

    print(f"\n  Total : {len(ALL_CONFIGS)}")
    print(f"  Pass  : {n_pass}")
    print(f"  Fail  : {n_fail}")
    print(f"  Error : {n_error}")

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    csv_path = RESULTS_DIR / "comprehensive_summary.csv"
    with csv_path.open("w") as f:
        f.write("name,sweep_type,vdd,nfin_n,nfin_p,cload_fF,tr_ps,pw_ns,nrmse_pct,max_err_mV,status\n")
        for r in results:
            cfg = r["config"]
            if "error" in r:
                f.write(
                    f"{cfg.name},{cfg.sweep_type},{cfg.vdd:.2f},"
                    f"{cfg.nfin_n},{cfg.nfin_p},"
                    f"{cfg.cload*1e15:.1f},{cfg.tr*1e12:.1f},{cfg.pw*1e9:.2f},"
                    f",,ERROR\n"
                )
            else:
                status = "PASS" if r["passed"] else "FAIL"
                f.write(
                    f"{cfg.name},{cfg.sweep_type},{cfg.vdd:.2f},"
                    f"{cfg.nfin_n},{cfg.nfin_p},"
                    f"{cfg.cload*1e15:.1f},{cfg.tr*1e12:.1f},{cfg.pw*1e9:.2f},"
                    f"{r['nrmse_post']*100:.4f},{r['max_err_mV']:.2f},{status}\n"
                )
    print(f"\n[CSV] Summary saved: {csv_path}")

    # -----------------------------------------------------------------------
    # Summary plot
    # -----------------------------------------------------------------------
    summary_plot_path = RESULTS_DIR / "comprehensive_summary.png"
    plot_summary(results, summary_plot_path)

    print(f"\n{'='*78}")
    if n_fail > 0 or n_error > 0:
        print(f"RESULT: {n_fail} FAIL, {n_error} ERROR out of {len(ALL_CONFIGS)} tests")
        return 1
    print(f"RESULT: ALL {n_pass} tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
