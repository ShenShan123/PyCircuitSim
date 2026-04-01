#!/usr/bin/env python3
"""
Multi-technology transient verification: PyCircuitSim vs NGSPICE.

Validates PyCircuitSim accuracy across 5 FinFET technologies (ASAP7, TSMC5,
TSMC7, TSMC12, TSMC16) by comparing CMOS inverter transient waveforms against
NGSPICE using the same OSDI binary.

Strategy:
  Phase 1 — Baseline: 1 config per tech (nominal VDD, default NFIN, 10fF, 100ps)
  Phase 2 — Parametric sweep: VDD sweep (3 points) + Cload sweep (5 points)
            Only runs for techs that pass baseline.

Each technology uses its own modelcard files, geometry (L, NFIN, TFIN), and VDD.
TSMC technologies have asymmetric NMOS/PMOS channel lengths (L_n=16nm, L_p=20nm).
"""
from __future__ import annotations

import os
import sys
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))

OSDI_PATH = (
    PROJECT_ROOT / "external_compact_models" / "PyCMG"
    / "build" / "osdi" / "bsimcmg.osdi"
)
TECH_MODEL_CARDS = (
    PROJECT_ROOT / "external_compact_models" / "PyCMG" / "modelcards"
)
NGSPICE_BIN = "/usr/local/ngspice-45.2/bin/ngspice"
RESULTS_BASE = PROJECT_ROOT / "tests" / "verify_multi_tech_tran_results"

# Baked-modelcard helper
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))
from helpers import bake_inst_params

# Acceptance criterion
NRMSE_THRESHOLD: float = 0.05  # 5% of Vdd

# Startup exclusion (Gmin-stepping / pseudo-transient artifact)
STARTUP_EXCLUSION: float = 0.1e-9  # 0.1 ns


# ---------------------------------------------------------------------------
# TechConfig dataclass
# ---------------------------------------------------------------------------
@dataclass
class TechConfig:
    """Technology-specific configuration for multi-tech verification."""

    name: str           # e.g. "TSMC7"
    vdd: float          # Core supply voltage [V]
    nmos_model: str     # Model name in modelcard (e.g. "nch_svt_mac")
    pmos_model: str     # Model name in modelcard (e.g. "pch_lvt_mac")
    nmos_file: str      # Naive modelcard filename for NMOS
    pmos_file: str      # Naive modelcard filename for PMOS
    tech_dir: str       # Subdir under modelcards/ (e.g. "TSMC7/naive")
    l_nmos: float       # NMOS channel length [m]
    l_pmos: float       # PMOS channel length [m]
    nfin: int           # Default NFIN
    tfin: float         # Fin thickness [m]
    single_file: bool = False  # True if NMOS+PMOS are in the same file (ASAP7)
    i_per_fin: float = 1e-5    # Estimated Idsat per fin [A] for tau_est


# ---------------------------------------------------------------------------
# Technology definitions
# ---------------------------------------------------------------------------
ALL_TECHS: List[TechConfig] = [
    TechConfig(
        name="ASAP7",
        vdd=0.7,
        nmos_model="nmos_rvt",
        pmos_model="pmos_rvt",
        nmos_file="7nm_TT_160803.pm",
        pmos_file="7nm_TT_160803.pm",
        tech_dir="ASAP7",
        l_nmos=30e-9,
        l_pmos=30e-9,
        nfin=10,
        tfin=6.5e-9,
        single_file=True,
        i_per_fin=1e-5,
    ),
    TechConfig(
        name="TSMC5",
        vdd=0.65,
        nmos_model="nch_svt_mac",
        pmos_model="pch_lvt_mac",
        nmos_file="nch_svt_mac_l16nm.l",
        pmos_file="pch_lvt_mac_l20nm.l",
        tech_dir="TSMC5/naive",
        l_nmos=16e-9,
        l_pmos=20e-9,
        nfin=2,
        tfin=6e-9,
        i_per_fin=1e-5,
    ),
    TechConfig(
        name="TSMC7",
        vdd=0.75,
        nmos_model="nch_svt_mac",
        pmos_model="pch_svt_mac",  # LVT PMOS has PDIBL2_i bug; use SVT
        nmos_file="nch_svt_mac_l16nm.l",
        pmos_file="pch_svt_mac_l20nm.l",  # SVT variant (LVT pch_lvt_mac has PDIBL2_i<0)
        tech_dir="TSMC7/naive",
        l_nmos=16e-9,
        l_pmos=20e-9,
        nfin=2,
        tfin=6e-9,
        i_per_fin=1e-5,
    ),
    TechConfig(
        name="TSMC12",
        vdd=0.80,
        nmos_model="nch_svt_mac",
        pmos_model="pch_lvt_mac",
        nmos_file="nch_svt_mac_l16nm.l",
        pmos_file="pch_lvt_mac_l20nm.l",
        tech_dir="TSMC12/naive",
        l_nmos=16e-9,
        l_pmos=20e-9,
        nfin=2,
        tfin=6e-9,
        i_per_fin=1e-5,
    ),
    TechConfig(
        name="TSMC16",
        vdd=0.80,
        nmos_model="nch_svt_mac",
        pmos_model="pch_lvt_mac",
        nmos_file="nch_svt_mac_l16nm.l",
        pmos_file="pch_lvt_mac_l20nm.l",
        tech_dir="TSMC16/naive",
        l_nmos=16e-9,
        l_pmos=20e-9,
        nfin=2,
        tfin=6e-9,
        i_per_fin=1e-5,
    ),
]


# ---------------------------------------------------------------------------
# TestConfig dataclass (per-simulation configuration)
# ---------------------------------------------------------------------------
@dataclass
class TestConfig:
    """One parametric test point for the inverter transient sweep."""

    tech: TechConfig    # Technology
    vdd: float          # Supply voltage [V]
    cload: float        # Load capacitance [F]
    tr: float           # Input rise time [s]
    tf: float           # Input fall time [s]
    pw: float           # Pulse width [s]
    config_name: str    # Human-readable tag (e.g. "baseline", "vdd_0p6")
    sweep_type: str     # Grouping key: "baseline", "vdd", "cload"
    nfin_n: int = 0     # NMOS fin count (0 = use tech default)
    nfin_p: int = 0     # PMOS fin count (0 = use tech default)

    # PULSE low / high voltages
    pulse_v1: float = 0.0
    pulse_v2: float = 0.0  # set in __post_init__

    def __post_init__(self) -> None:
        if self.nfin_n == 0:
            self.nfin_n = self.tech.nfin
        if self.nfin_p == 0:
            self.nfin_p = self.tech.nfin
        if self.pulse_v2 == 0.0:
            self.pulse_v2 = self.vdd

    # -- Adaptive timing (computed properties) --------------------------------

    @property
    def tau_est(self) -> float:
        """Rough RC time constant estimate, scaled by fin count."""
        nfin_min = min(self.nfin_n, self.nfin_p)
        i_est = nfin_min * self.tech.i_per_fin
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

    @property
    def full_name(self) -> str:
        """Full name including tech prefix."""
        return f"{self.tech.name}_{self.config_name}"

    def summary(self) -> str:
        """One-line summary string."""
        return (
            f"{self.full_name:30s}  VDD={self.vdd:.2f}V  "
            f"NFIN={self.nfin_n}/{self.nfin_p}  "
            f"Cload={self.cload*1e15:.0f}fF  "
            f"tr/tf={self.tr*1e12:.0f}ps  "
            f"pw={self.pw*1e9:.1f}ns  "
            f"tstop={self.tstop*1e9:.1f}ns"
        )


# ---------------------------------------------------------------------------
# Modelcard helpers
# ---------------------------------------------------------------------------
def get_results_dir(tech: TechConfig) -> Path:
    """Get results directory for a technology."""
    d = RESULTS_BASE / tech.name
    d.mkdir(parents=True, exist_ok=True)
    return d


def create_merged_modelcard(tech: TechConfig) -> Path:
    """Create a single modelcard file containing both NMOS and PMOS models.

    For ASAP7 (single_file=True), returns the original file path.
    For TSMC (separate files), concatenates NMOS + PMOS into one file.
    """
    if tech.single_file:
        return TECH_MODEL_CARDS / tech.tech_dir / tech.nmos_file

    nmos_src = TECH_MODEL_CARDS / tech.tech_dir / tech.nmos_file
    pmos_src = TECH_MODEL_CARDS / tech.tech_dir / tech.pmos_file

    if not nmos_src.exists():
        raise FileNotFoundError(f"NMOS modelcard not found: {nmos_src}")
    if not pmos_src.exists():
        raise FileNotFoundError(f"PMOS modelcard not found: {pmos_src}")

    merged = get_results_dir(tech) / "merged_modelcard.lib"
    content = nmos_src.read_text() + "\n" + pmos_src.read_text()
    merged.write_text(content)
    return merged


_baked_cache: Dict[Tuple[str, int, int], Path] = {}


def get_or_create_baked_modelcard(
    tech: TechConfig, nfin_n: int, nfin_p: int
) -> Path:
    """Return a baked modelcard for the given tech + NFIN geometry.

    Creates the baked file on first call, caches for subsequent calls.
    """
    key = (tech.name, nfin_n, nfin_p)
    if key in _baked_cache:
        return _baked_cache[key]

    merged = create_merged_modelcard(tech)
    baked = get_results_dir(tech) / f"baked_nfin_n{nfin_n}_p{nfin_p}.lib"

    nmos_params: Dict[str, Any] = {
        "L": tech.l_nmos, "NFIN": float(nfin_n), "TFIN": tech.tfin, "DEVTYPE": 1,
    }
    pmos_params: Dict[str, Any] = {
        "L": tech.l_pmos, "NFIN": float(nfin_p), "TFIN": tech.tfin, "DEVTYPE": 0,
    }

    bake_inst_params(merged, baked, tech.nmos_model, nmos_params)
    bake_inst_params(baked, baked, tech.pmos_model, pmos_params)

    _baked_cache[key] = baked
    print(f"[NGSPICE] Baked modelcard ({tech.name}, NFIN_N={nfin_n}, NFIN_P={nfin_p}): {baked}")
    return baked


# ---------------------------------------------------------------------------
# NGSPICE netlist generation & runner
# ---------------------------------------------------------------------------
def create_ngspice_netlist(config: TestConfig) -> Path:
    """Generate an NGSPICE inverter transient netlist for one config."""
    tech = config.tech
    results_dir = get_results_dir(tech)
    baked_lib = get_or_create_baked_modelcard(tech, config.nfin_n, config.nfin_p)

    netlist_path = results_dir / f"ngspice_{config.config_name}.cir"
    content = f"""\
* BSIM-CMG CMOS Inverter Transient - NGSPICE ({config.full_name})
.include "{baked_lib}"
.temp 27
Vdd vdd 0 {config.vdd}
Vin in 0 PULSE({config.pulse_v1} {config.pulse_v2} {config.td} {config.tr} {config.tf} {config.pw} {config.per})
Np out in vdd vdd {tech.pmos_model}
Nn out in 0 0 {tech.nmos_model}
Cload out 0 {config.cload}
.ic V(out)={config.vdd}
.tran {config.tstep} {config.tstop} uic
.end
"""
    netlist_path.write_text(content)
    return netlist_path


def run_ngspice(config: TestConfig) -> Dict[str, np.ndarray]:
    """Run NGSPICE transient simulation and parse wrdata output."""
    tech = config.tech
    results_dir = get_results_dir(tech)
    netlist_path = create_ngspice_netlist(config)

    csv_path = results_dir / f"ngspice_{config.config_name}.csv"
    log_path = results_dir / f"ngspice_{config.config_name}.log"
    runner_path = results_dir / f"ngspice_{config.config_name}_runner.cir"

    runner_content = f"""\
* NGSPICE transient runner ({config.full_name})
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

    print(f"  [NGSPICE] Running {config.full_name}...")
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

    # Parse wrdata: header + data rows (time, v(out), time, v(in))
    with csv_path.open() as f:
        lines = f.readlines()

    data_rows: List[List[float]] = []
    for line in lines[1:]:  # skip header
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
        f"  [NGSPICE] Done: {len(result['time'])} pts, "
        f"V(out) [{result['v(out)'].min():.4f}, {result['v(out)'].max():.4f}]V"
    )
    return result


# ---------------------------------------------------------------------------
# PyCircuitSim netlist generation & runner
# ---------------------------------------------------------------------------
def create_pycircuitsim_netlist(config: TestConfig) -> Path:
    """Generate a PyCircuitSim inverter transient netlist for one config."""
    tech = config.tech
    results_dir = get_results_dir(tech)

    netlist_path = results_dir / f"pycircuitsim_{config.config_name}.sp"

    # Format L values — use nm for readability
    l_nmos_nm = tech.l_nmos * 1e9
    l_pmos_nm = tech.l_pmos * 1e9

    content = f"""\
* BSIM-CMG Inverter Transient - PyCircuitSim ({config.full_name})
* Tech={tech.name}, VDD={config.vdd}V, L_n={l_nmos_nm:.0f}nm, L_p={l_pmos_nm:.0f}nm
* NFIN_N={config.nfin_n}, NFIN_P={config.nfin_p}, Cload={config.cload*1e15:.0f}fF

* Power supply
Vdd 1 0 {config.vdd}

* Input pulse: {config.pulse_v1} -> {config.pulse_v2}V
Vin 2 0 PULSE {config.pulse_v1} {config.pulse_v2} {config.td} {config.tr} {config.tf} {config.pw} {config.per}

* PMOS (drain=out, gate=in, source=Vdd, bulk=Vdd)
Mp1 3 2 1 1 {tech.pmos_model} L={l_pmos_nm:.0f}n NFIN={config.nfin_p} TFIN={tech.tfin*1e9:.1f}n

* NMOS (drain=out, gate=in, source=GND, bulk=GND)
Mn1 3 2 0 0 {tech.nmos_model} L={l_nmos_nm:.0f}n NFIN={config.nfin_n} TFIN={tech.tfin*1e9:.1f}n

* Load capacitance
Cload 3 0 {config.cload}

* Initial condition: output starts high (PMOS on, NMOS off when Vin=0)
.ic V(3)={config.vdd}

* Model definitions (LEVEL=72 BSIM-CMG)
.model {tech.nmos_model} NMOS (LEVEL=72)
.model {tech.pmos_model} PMOS (LEVEL=72)

* Transient: {config.tstep*1e12:.1f}ps step, {config.tstop*1e9:.1f}ns total
.tran {config.tstep} {config.tstop}

.end
"""
    netlist_path.write_text(content)
    return netlist_path


def run_pycircuitsim(config: TestConfig) -> Dict[str, np.ndarray]:
    """Run PyCircuitSim transient simulation for one config."""
    import logging

    from pycircuitsim.parser import Parser
    from pycircuitsim.solver import DCSolver, TransientSolver

    tech = config.tech
    netlist_path = create_pycircuitsim_netlist(config)

    # Create merged modelcard for this tech
    merged = create_merged_modelcard(tech)

    # Suppress verbose logging during simulation
    logging.disable(logging.CRITICAL)
    try:
        parser = Parser(
            modelcard_path=str(merged),
            model_name_map={"NMOS": tech.nmos_model, "PMOS": tech.pmos_model},
        )
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
            nr_tolerance=1e-7,
        )
        results = solver.solve()
    finally:
        logging.disable(logging.NOTSET)

    # Node mapping: '1'=Vdd, '2'=Vin, '3'=Vout
    result: Dict[str, np.ndarray] = {
        "time": results["time"],
        "v(out)": results["3"],
        "v(in)": results["2"],
    }
    print(
        f"  [PySim] Done: {len(result['time'])} pts, "
        f"V(out) [{result['v(out)'].min():.4f}, {result['v(out)'].max():.4f}]V"
    )
    return result


# ---------------------------------------------------------------------------
# Comparison, metrics, and plotting
# ---------------------------------------------------------------------------
def interpolate_to_common_time(
    ng_data: Dict[str, np.ndarray],
    py_data: Dict[str, np.ndarray],
    config: TestConfig,
    t_start: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate both datasets to a common uniform time grid."""
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
    """Compute error metrics between NGSPICE and PyCircuitSim waveforms."""
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
    """Generate a 3-panel comparison plot for one config."""
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 9),
        gridspec_kw={"height_ratios": [0.8, 1.2, 0.8]},
    )

    ng_t_ns = ng_data["time"] * 1e9
    py_t_ns = py_data["time"] * 1e9

    # --- Panel 1: V(in) ---
    ax1 = axes[0]
    ax1.plot(ng_t_ns, ng_data["v(in)"], "b-", label="NGSPICE", linewidth=1.5)
    ax1.plot(py_t_ns, py_data["v(in)"], "r--", label="PyCircuitSim",
             linewidth=1.2, alpha=0.8)
    ax1.set_ylabel("V(in) [V]")
    ax1.set_title(
        f"BSIM-CMG Inverter Transient: {config.full_name}\n"
        f"Tech={config.tech.name}  VDD={config.vdd:.2f}V  "
        f"NFIN={config.nfin_n}/{config.nfin_p}  "
        f"Cload={config.cload*1e15:.0f}fF  "
        f"L_n={config.tech.l_nmos*1e9:.0f}nm  L_p={config.tech.l_pmos*1e9:.0f}nm"
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


# ---------------------------------------------------------------------------
# Single-test orchestrator
# ---------------------------------------------------------------------------
def run_single_test(config: TestConfig) -> Dict[str, Any]:
    """Run NGSPICE + PyCircuitSim for one config and return metrics."""
    # 1. NGSPICE reference
    ng_data = run_ngspice(config)

    # 2. PyCircuitSim
    py_data = run_pycircuitsim(config)

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
    results_dir = get_results_dir(config.tech)
    plot_path = results_dir / f"comparison_{config.config_name}.png"
    plot_single_comparison(ng_data, py_data, config, metrics_full, metrics_post, plot_path)

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
# Parametric sweep builder
# ---------------------------------------------------------------------------
def build_baseline_config(tech: TechConfig) -> TestConfig:
    """Build the baseline test config for a technology."""
    return TestConfig(
        tech=tech,
        vdd=tech.vdd,
        cload=10e-15,
        tr=100e-12,
        tf=100e-12,
        pw=0.8e-9,
        config_name="baseline",
        sweep_type="baseline",
    )


def build_parametric_configs(tech: TechConfig) -> List[TestConfig]:
    """Build parametric sweep configs for a technology.

    VDD sweep: nominal-0.1V, nominal+0.1V (baseline already at nominal)
    Cload sweep: 1fF, 5fF, 50fF, 100fF (baseline already at 10fF)
    """
    configs: List[TestConfig] = []

    # VDD sweep (2 extra points around nominal)
    for delta_v in [-0.1, 0.1]:
        vdd_val = tech.vdd + delta_v
        if vdd_val <= 0:
            continue
        tag = f"vdd_{vdd_val:.1f}".replace(".", "p")
        configs.append(
            TestConfig(
                tech=tech,
                vdd=vdd_val,
                cload=10e-15,
                tr=100e-12,
                tf=100e-12,
                pw=0.8e-9,
                config_name=tag,
                sweep_type="vdd",
            )
        )

    # Cload sweep (4 extra points)
    for cload_fF in [1, 5, 50, 100]:
        configs.append(
            TestConfig(
                tech=tech,
                vdd=tech.vdd,
                cload=cload_fF * 1e-15,
                tr=100e-12,
                tf=100e-12,
                pw=0.8e-9,
                config_name=f"cload_{cload_fF}fF",
                sweep_type="cload",
            )
        )

    return configs


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------
TECH_COLORS: Dict[str, str] = {
    "ASAP7": "tab:blue",
    "TSMC5": "tab:green",
    "TSMC7": "tab:orange",
    "TSMC12": "tab:purple",
    "TSMC16": "tab:red",
}


def plot_summary(results: List[Dict[str, Any]], save_path: Path) -> None:
    """Generate a bar chart summarising NRMSE across all configs by tech."""
    valid = [r for r in results if "nrmse_post" in r]
    if not valid:
        print("[Plot] No valid results to plot summary.")
        return

    names = [r["config"].full_name for r in valid]
    nrmse_pct = [r["nrmse_post"] * 100 for r in valid]
    colors = [TECH_COLORS.get(r["config"].tech.name, "tab:gray") for r in valid]

    fig, ax = plt.subplots(figsize=(max(14, len(valid) * 0.8), 6))
    x = np.arange(len(valid))
    ax.bar(x, nrmse_pct, color=colors, edgecolor="black", linewidth=0.5)

    # Threshold line
    ax.axhline(
        y=NRMSE_THRESHOLD * 100, color="red", linewidth=1.5, linestyle="--",
        label=f"Threshold ({NRMSE_THRESHOLD*100:.0f}%)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("NRMSE (% of Vdd)")
    ax.set_title("Multi-Technology Transient Verification: NRMSE Summary")

    # Legend for tech types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=TECH_COLORS[t], edgecolor="black", label=t)
        for t in TECH_COLORS
    ]
    legend_elements.append(
        plt.Line2D([0], [0], color="red", linewidth=1.5, linestyle="--",
                    label=f"Threshold ({NRMSE_THRESHOLD*100:.0f}%)")
    )
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
    """Run multi-technology transient verification.

    Phase 1: Baseline test per tech.
    Phase 2: Parametric sweep for techs that pass baseline.
    """
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Multi-Technology Transient Verification")
    print(f"  Technologies: {', '.join(t.name for t in ALL_TECHS)}")
    print(f"  Acceptance: NRMSE < {NRMSE_THRESHOLD*100:.0f}% of Vdd (post-settling)")
    print(f"  Startup exclusion: {STARTUP_EXCLUSION*1e9:.1f}ns")
    print("=" * 78)

    # Print tech summary
    print("\nTechnology parameters:")
    print(f"  {'Tech':8s} | {'VDD':>5s} | {'L_n':>5s} | {'L_p':>5s} | "
          f"{'NFIN':>4s} | {'TFIN':>5s} | {'NMOS model':15s} | {'PMOS model':15s}")
    print("-" * 85)
    for tech in ALL_TECHS:
        print(
            f"  {tech.name:8s} | {tech.vdd:5.2f} | "
            f"{tech.l_nmos*1e9:4.0f}n | {tech.l_pmos*1e9:4.0f}n | "
            f"{tech.nfin:4d} | {tech.tfin*1e9:4.1f}n | "
            f"{tech.nmos_model:15s} | {tech.pmos_model:15s}"
        )

    all_results: List[Dict[str, Any]] = []
    tech_status: Dict[str, str] = {}

    for tech in ALL_TECHS:
        print(f"\n{'='*78}")
        print(f"Technology: {tech.name}")
        print(f"{'='*78}")

        # Phase 1: Baseline
        print(f"\n--- Phase 1: Baseline ({tech.name}) ---")
        baseline_cfg = build_baseline_config(tech)
        print(f"  {baseline_cfg.summary()}")

        try:
            result = run_single_test(baseline_cfg)
            all_results.append(result)

            if result["passed"]:
                print(f"  => BASELINE PASS  NRMSE={result['nrmse_post']*100:.2f}%")
                tech_status[tech.name] = "PASS"
            else:
                print(f"  => BASELINE FAIL  NRMSE={result['nrmse_post']*100:.2f}%")
                print(f"  {tech.name} BASELINE FAILED — skipping parametric sweep")
                tech_status[tech.name] = "BASELINE_FAIL"
                continue
        except Exception as exc:
            print(f"  => BASELINE ERROR: {exc}")
            all_results.append({
                "config": baseline_cfg,
                "error": str(exc),
                "passed": False,
            })
            tech_status[tech.name] = "BASELINE_ERROR"
            continue

        # Phase 2: Parametric sweep
        print(f"\n--- Phase 2: Parametric sweep ({tech.name}) ---")
        sweep_configs = build_parametric_configs(tech)

        for i, cfg in enumerate(sweep_configs):
            print(f"\n  [{i+1}/{len(sweep_configs)}] {cfg.summary()}")
            try:
                result = run_single_test(cfg)
                all_results.append(result)
                status = "PASS" if result["passed"] else "FAIL"
                print(f"  => {status}  NRMSE={result['nrmse_post']*100:.2f}%")
            except Exception as exc:
                print(f"  => ERROR: {exc}")
                all_results.append({
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
        f"{'Config':30s} | {'VDD':>5s} | {'L_n':>5s} | {'L_p':>5s} | "
        f"{'NFIN':>4s} | {'Cload':>7s} | {'tr/tf':>7s} | "
        f"{'NRMSE(%)':>9s} | {'MaxErr(mV)':>10s} | {'Status':>6s}"
    )
    print(header)
    print("-" * len(header))

    n_pass = 0
    n_fail = 0
    n_error = 0

    for r in all_results:
        cfg: TestConfig = r["config"]
        tech = cfg.tech
        if "error" in r:
            n_error += 1
            print(
                f"{cfg.full_name:30s} | {cfg.vdd:5.2f} | "
                f"{tech.l_nmos*1e9:4.0f}n | {tech.l_pmos*1e9:4.0f}n | "
                f"{cfg.nfin_n:4d} | {cfg.cload*1e15:5.0f}fF | "
                f"{cfg.tr*1e12:5.0f}ps | "
                f"{'ERROR':>9s} | {'ERROR':>10s} | {'ERROR':>6s}"
            )
        else:
            status = "PASS" if r["passed"] else "FAIL"
            if r["passed"]:
                n_pass += 1
            else:
                n_fail += 1
            print(
                f"{cfg.full_name:30s} | {cfg.vdd:5.2f} | "
                f"{tech.l_nmos*1e9:4.0f}n | {tech.l_pmos*1e9:4.0f}n | "
                f"{cfg.nfin_n:4d} | {cfg.cload*1e15:5.0f}fF | "
                f"{cfg.tr*1e12:5.0f}ps | "
                f"{r['nrmse_post']*100:9.2f} | {r['max_err_mV']:10.1f} | "
                f"{status:>6s}"
            )

    total = len(all_results)
    print(f"\n  Total : {total}")
    print(f"  Pass  : {n_pass}")
    print(f"  Fail  : {n_fail}")
    print(f"  Error : {n_error}")

    # Tech-level status
    print(f"\nTechnology status:")
    for tech_name, status in tech_status.items():
        print(f"  {tech_name:8s}: {status}")

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    csv_path = RESULTS_BASE / "multi_tech_summary.csv"
    with csv_path.open("w") as f:
        f.write(
            "tech,config,sweep_type,vdd,l_nmos_nm,l_pmos_nm,"
            "nfin_n,nfin_p,cload_fF,tr_ps,pw_ns,"
            "nrmse_pct,max_err_mV,status\n"
        )
        for r in all_results:
            cfg = r["config"]
            tech = cfg.tech
            if "error" in r:
                f.write(
                    f"{tech.name},{cfg.config_name},{cfg.sweep_type},"
                    f"{cfg.vdd:.2f},{tech.l_nmos*1e9:.0f},{tech.l_pmos*1e9:.0f},"
                    f"{cfg.nfin_n},{cfg.nfin_p},"
                    f"{cfg.cload*1e15:.1f},{cfg.tr*1e12:.1f},{cfg.pw*1e9:.2f},"
                    f",,ERROR\n"
                )
            else:
                status = "PASS" if r["passed"] else "FAIL"
                f.write(
                    f"{tech.name},{cfg.config_name},{cfg.sweep_type},"
                    f"{cfg.vdd:.2f},{tech.l_nmos*1e9:.0f},{tech.l_pmos*1e9:.0f},"
                    f"{cfg.nfin_n},{cfg.nfin_p},"
                    f"{cfg.cload*1e15:.1f},{cfg.tr*1e12:.1f},{cfg.pw*1e9:.2f},"
                    f"{r['nrmse_post']*100:.4f},{r['max_err_mV']:.2f},{status}\n"
                )
    print(f"\n[CSV] Summary saved: {csv_path}")

    # -----------------------------------------------------------------------
    # Summary plot
    # -----------------------------------------------------------------------
    summary_plot_path = RESULTS_BASE / "multi_tech_summary.png"
    plot_summary(all_results, summary_plot_path)

    print(f"\n{'='*78}")
    if n_fail > 0 or n_error > 0:
        print(f"RESULT: {n_fail} FAIL, {n_error} ERROR out of {total} tests")
        return 1
    print(f"RESULT: ALL {n_pass} tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
