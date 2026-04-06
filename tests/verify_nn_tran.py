#!/usr/bin/env python3
"""
NN compact model (LEVEL=73) transient verification: PyCircuitSim vs NGSPICE.

Validates NN-based MOSFET transient simulation across 5 FinFET technologies
(ASAP7, TSMC5, TSMC7, TSMC12, TSMC16) by comparing CMOS inverter transient
waveforms against NGSPICE (using BSIM-CMG OSDI as ground truth).

Strategy:
  1. Baseline transient test per tech (nominal VDD, NFIN=10, Cload=10fF)
  2. Ground truth = NGSPICE with BSIM-CMG OSDI (same as Phase 7-10)
  3. PyCircuitSim uses LEVEL=73 (NN) instead of LEVEL=72 (BSIM-CMG)

Acceptance: NRMSE < 15% of Vdd (post-settling).
  NN DC accuracy is 1-7%, so transient will be worse than BSIM-CMG's 0.2%.
"""
from __future__ import annotations

import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
RESULTS_BASE = PROJECT_ROOT / "tests" / "verify_nn_tran_results"
CHECKPOINT_DIR = (
    PROJECT_ROOT / "external_compact_models" / "BSIMAR" / "checkpoints"
)

sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))
from helpers import bake_inst_params

# Acceptance criterion — loose for NN (DC accuracy is 1-7%)
NRMSE_THRESHOLD: float = 0.15  # 15% of Vdd

# Startup exclusion (Gmin-stepping / pseudo-transient artifact)
STARTUP_EXCLUSION: float = 0.1e-9  # 0.1 ns


# ---------------------------------------------------------------------------
# TechConfig dataclass (replicates verify_multi_tech_tran.py structure)
# ---------------------------------------------------------------------------
@dataclass
class TechConfig:
    """Technology-specific configuration."""
    name: str           # e.g. "TSMC7"
    vdd: float
    nmos_model: str     # Model name in modelcard
    pmos_model: str
    nmos_file: str      # Modelcard filename for NMOS
    pmos_file: str
    tech_dir: str       # Subdir under modelcards/
    l_nmos: float       # NMOS channel length [m]
    l_pmos: float       # PMOS channel length [m]
    nfin: int           # Default NFIN
    tfin: float         # Fin thickness [m]
    single_file: bool = False  # True if NMOS+PMOS in same file (ASAP7)
    tech_key: str = ""  # Key for LEVEL=73 TECH parameter


ALL_TECHS: List[TechConfig] = [
    TechConfig(
        name="ASAP7", vdd=0.7,
        nmos_model="nmos_rvt", pmos_model="pmos_rvt",
        nmos_file="7nm_TT_160803.pm", pmos_file="7nm_TT_160803.pm",
        tech_dir="ASAP7",
        l_nmos=30e-9, l_pmos=30e-9, nfin=10, tfin=6.5e-9,
        single_file=True, tech_key="asap7",
    ),
    TechConfig(
        name="TSMC5", vdd=0.65,
        nmos_model="nch_svt_mac", pmos_model="pch_svt_mac",
        nmos_file="nch_svt_mac_l16nm.l", pmos_file="pch_svt_mac_l20nm.l",
        tech_dir="TSMC5/naive",
        l_nmos=16e-9, l_pmos=20e-9, nfin=10, tfin=6e-9,
        tech_key="tsmc5",
    ),
    TechConfig(
        name="TSMC7", vdd=0.75,
        nmos_model="nch_svt_mac", pmos_model="pch_svt_mac",
        nmos_file="nch_svt_mac_l16nm.l", pmos_file="pch_svt_mac_l20nm.l",
        tech_dir="TSMC7/naive",
        l_nmos=16e-9, l_pmos=20e-9, nfin=10, tfin=6e-9,
        tech_key="tsmc7",
    ),
    TechConfig(
        name="TSMC12", vdd=0.80,
        nmos_model="nch_svt_mac", pmos_model="pch_svt_mac",
        nmos_file="nch_svt_mac_l16nm.l", pmos_file="pch_svt_mac_l20nm.l",
        tech_dir="TSMC12/naive",
        l_nmos=16e-9, l_pmos=20e-9, nfin=10, tfin=6e-9,
        tech_key="tsmc12",
    ),
    TechConfig(
        name="TSMC16", vdd=0.80,
        nmos_model="nch_svt_mac", pmos_model="pch_svt_mac",
        nmos_file="nch_svt_mac_l16nm.l", pmos_file="pch_svt_mac_l20nm.l",
        tech_dir="TSMC16/naive",
        l_nmos=16e-9, l_pmos=20e-9, nfin=10, tfin=6e-9,
        tech_key="tsmc16",
    ),
]


# ---------------------------------------------------------------------------
# Transient test parameters
# ---------------------------------------------------------------------------
NFIN = 10
CLOAD = 10e-15      # 10fF
TR = 100e-12         # 100ps rise/fall
TF = 100e-12
PW = 0.8e-9          # 800ps pulse width
TD = 0.5e-9          # 500ps delay
TSTEP = 10e-12       # 10ps timestep
TSTOP = 5e-9         # 5ns total


# ---------------------------------------------------------------------------
# Checkpoint existence check
# ---------------------------------------------------------------------------
def check_checkpoints(tech: TechConfig) -> bool:
    """Check if NN checkpoints exist for this technology."""
    if tech.tech_key == "asap7":
        nmos_ckpt = CHECKPOINT_DIR / "nmos_best.pt"
        pmos_ckpt = CHECKPOINT_DIR / "pmos_best.pt"
    else:
        nmos_ckpt = CHECKPOINT_DIR / f"{tech.tech_key}_nmos_best.pt"
        pmos_ckpt = CHECKPOINT_DIR / f"{tech.tech_key}_pmos_best.pt"
    return nmos_ckpt.exists() and pmos_ckpt.exists()


# ---------------------------------------------------------------------------
# Modelcard helpers (for NGSPICE ground truth)
# ---------------------------------------------------------------------------
def get_results_dir(tech: TechConfig) -> Path:
    d = RESULTS_BASE / tech.name
    d.mkdir(parents=True, exist_ok=True)
    return d


def create_merged_modelcard(tech: TechConfig) -> Path:
    """Create a single modelcard file containing both NMOS and PMOS."""
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


def create_baked_modelcard(tech: TechConfig) -> Path:
    """Create baked modelcard with instance parameters for NGSPICE OSDI."""
    merged = create_merged_modelcard(tech)
    baked = get_results_dir(tech) / "baked_modelcard.lib"

    nmos_params: Dict[str, Any] = {
        "L": tech.l_nmos, "NFIN": float(NFIN), "TFIN": tech.tfin, "DEVTYPE": 1,
    }
    pmos_params: Dict[str, Any] = {
        "L": tech.l_pmos, "NFIN": float(NFIN), "TFIN": tech.tfin, "DEVTYPE": 0,
    }

    bake_inst_params(merged, baked, tech.nmos_model, nmos_params)
    bake_inst_params(baked, baked, tech.pmos_model, pmos_params)
    return baked


# ---------------------------------------------------------------------------
# NGSPICE (ground truth — BSIM-CMG OSDI)
# ---------------------------------------------------------------------------
def create_ngspice_netlist(tech: TechConfig, baked_lib: Path) -> Path:
    """Create NGSPICE inverter transient netlist."""
    results_dir = get_results_dir(tech)
    netlist_path = results_dir / "ngspice_tran.cir"

    per = TR + PW + TF + max(PW, 1.0e-9)

    content = f"""\
* BSIM-CMG CMOS Inverter Transient - NGSPICE ({tech.name})
.include "{baked_lib}"
.temp 27
Vdd vdd 0 {tech.vdd}
Vin in 0 PULSE(0 {tech.vdd} {TD} {TR} {TF} {PW} {per})
Np out in vdd vdd {tech.pmos_model}
Nn out in 0 0 {tech.nmos_model}
Cload out 0 {CLOAD}
.ic V(out)={tech.vdd}
.tran {TSTEP} {TSTOP} uic
.end
"""
    netlist_path.write_text(content)
    return netlist_path


def run_ngspice(tech: TechConfig) -> Dict[str, np.ndarray]:
    """Run NGSPICE transient simulation and parse wrdata output."""
    results_dir = get_results_dir(tech)
    baked_lib = create_baked_modelcard(tech)
    netlist_path = create_ngspice_netlist(tech, baked_lib)

    csv_path = results_dir / "ngspice_tran.csv"
    log_path = results_dir / "ngspice_tran.log"
    runner_path = results_dir / "ngspice_tran_runner.cir"

    runner_content = f"""\
* NGSPICE transient runner ({tech.name})
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

    print(f"  [NGSPICE] Running {tech.name}...")
    res = subprocess.run(
        [NGSPICE_BIN, "-b", "-o", str(log_path), str(runner_path)],
        capture_output=True, text=True,
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
    for line in lines[1:]:
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
    print(f"  [NGSPICE] Done: {len(result['time'])} pts, "
          f"V(out) [{result['v(out)'].min():.4f}, {result['v(out)'].max():.4f}]V")
    return result


# ---------------------------------------------------------------------------
# PyCircuitSim with LEVEL=73 (NN)
# ---------------------------------------------------------------------------
def create_nn_netlist(tech: TechConfig) -> Path:
    """Create PyCircuitSim inverter transient netlist using LEVEL=73."""
    results_dir = get_results_dir(tech)
    netlist_path = results_dir / "nn_inverter_tran.sp"

    l_nmos_nm = tech.l_nmos * 1e9
    l_pmos_nm = tech.l_pmos * 1e9

    per = TR + PW + TF + max(PW, 1.0e-9)

    # TECH parameter: omit for ASAP7 (default), include for TSMC
    tech_param = "" if tech.tech_key == "asap7" else f" TECH={tech.tech_key}"

    content = f"""\
* NN CMOS Inverter Transient - PyCircuitSim ({tech.name}, LEVEL=73)
* Tech={tech.name}, VDD={tech.vdd}V, L_n={l_nmos_nm:.0f}nm, L_p={l_pmos_nm:.0f}nm, NFIN={NFIN}

* Power supply
Vdd 1 0 {tech.vdd}

* Input pulse: 0 -> {tech.vdd}V
Vin 2 0 PULSE 0 {tech.vdd} {TD} {TR} {TF} {PW} {per}

* PMOS (drain=out, gate=in, source=Vdd, bulk=Vdd)
Mp1 3 2 1 1 pmos1 L={l_pmos_nm:.0f}n NFIN={NFIN}

* NMOS (drain=out, gate=in, source=GND, bulk=GND)
Mn1 3 2 0 0 nmos1 L={l_nmos_nm:.0f}n NFIN={NFIN}

* Load capacitance
Cload 3 0 {CLOAD}

* Initial condition: output starts high
.ic V(3)={tech.vdd}

* Model definitions (LEVEL=73 NN)
.model nmos1 NMOS (LEVEL=73{tech_param})
.model pmos1 PMOS (LEVEL=73{tech_param})

* Transient: {TSTEP*1e12:.0f}ps step, {TSTOP*1e9:.0f}ns total
.tran {TSTEP} {TSTOP}

.end
"""
    netlist_path.write_text(content)
    return netlist_path


def run_pycircuitsim_nn(tech: TechConfig) -> Dict[str, np.ndarray]:
    """Run PyCircuitSim transient simulation with LEVEL=73 (NN)."""
    import logging
    from pycircuitsim.parser import Parser
    from pycircuitsim.solver import DCSolver, TransientSolver

    netlist_path = create_nn_netlist(tech)

    logging.disable(logging.CRITICAL)
    try:
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
    print(f"  [NN-Sim] Done: {len(result['time'])} pts, "
          f"V(out) [{result['v(out)'].min():.4f}, {result['v(out)'].max():.4f}]V")
    return result


# ---------------------------------------------------------------------------
# Comparison functions
# ---------------------------------------------------------------------------
def interpolate_to_common_time(
    ng_data: Dict[str, np.ndarray],
    nn_data: Dict[str, np.ndarray],
    t_start: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate both datasets to common uniform time grid."""
    t_max = min(ng_data["time"][-1], nn_data["time"][-1])
    t_common = np.arange(max(t_start, ng_data["time"][0]), t_max, TSTEP)

    ng_vout = np.interp(t_common, ng_data["time"], ng_data["v(out)"])
    nn_vout = np.interp(t_common, nn_data["time"], nn_data["v(out)"])
    ng_vin = np.interp(t_common, ng_data["time"], ng_data["v(in)"])
    nn_vin = np.interp(t_common, nn_data["time"], nn_data["v(in)"])

    return t_common, ng_vout, nn_vout, ng_vin, nn_vin


def compute_metrics(
    ng_vout: np.ndarray,
    nn_vout: np.ndarray,
    vdd: float,
) -> Dict[str, float]:
    """Compute error metrics between NGSPICE and NN waveforms."""
    diff = nn_vout - ng_vout
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


def plot_comparison(
    tech: TechConfig,
    ng_data: Dict[str, np.ndarray],
    nn_data: Dict[str, np.ndarray],
    metrics_full: Dict[str, float],
    metrics_post: Dict[str, float],
    save_path: Path,
) -> None:
    """Generate 3-panel comparison plot for one tech."""
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 9),
        gridspec_kw={"height_ratios": [0.8, 1.2, 0.8]},
    )

    ng_t_ns = ng_data["time"] * 1e9
    nn_t_ns = nn_data["time"] * 1e9

    # Panel 1: V(in)
    ax1 = axes[0]
    ax1.plot(ng_t_ns, ng_data["v(in)"], "b-", label="NGSPICE (BSIM-CMG)", linewidth=1.5)
    ax1.plot(nn_t_ns, nn_data["v(in)"], "r--", label="PyCircuitSim (NN)", linewidth=1.2, alpha=0.8)
    ax1.set_ylabel("V(in) [V]")
    ax1.set_title(
        f"NN Inverter Transient: {tech.name}\n"
        f"VDD={tech.vdd:.2f}V  NFIN={NFIN}  Cload={CLOAD*1e15:.0f}fF  "
        f"L_n={tech.l_nmos*1e9:.0f}nm  L_p={tech.l_pmos*1e9:.0f}nm"
    )
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, tech.vdd + 0.1)

    # Panel 2: V(out)
    ax2 = axes[1]
    ax2.plot(ng_t_ns, ng_data["v(out)"], "b-", label="NGSPICE (BSIM-CMG)", linewidth=1.5)
    ax2.plot(nn_t_ns, nn_data["v(out)"], "r--", label="PyCircuitSim (NN)", linewidth=1.2, alpha=0.8)
    ax2.axvline(x=STARTUP_EXCLUSION * 1e9, color="gray", linewidth=1,
                linestyle=":", label=f"Startup excl ({STARTUP_EXCLUSION*1e9:.1f}ns)")
    ax2.set_ylabel("V(out) [V]")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, tech.vdd + 0.15)

    txt = (
        f"Full: NRMSE={metrics_full['NRMSE (% of Vdd)']:.2f}%\n"
        f"Post-settling: NRMSE={metrics_post['NRMSE (% of Vdd)']:.2f}%, "
        f"Max|err|={metrics_post['Max |error| (mV)']:.1f}mV"
    )
    ax2.text(0.02, 0.05, txt, transform=ax2.transAxes, fontsize=9,
             verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # Panel 3: Error (post-settling)
    ax3 = axes[2]
    t_c, ng_v, nn_v, _, _ = interpolate_to_common_time(
        ng_data, nn_data, t_start=STARTUP_EXCLUSION,
    )
    error_mv = (nn_v - ng_v) * 1e3
    ax3.plot(t_c * 1e9, error_mv, "g-", linewidth=0.8)
    ax3.axhline(y=0, color="k", linewidth=0.5)
    threshold_mv = tech.vdd * NRMSE_THRESHOLD * 1e3
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
    """Generate bar chart summarizing NRMSE across all techs."""
    valid = [r for r in results if "nrmse_post" in r]
    if not valid:
        return

    names = [r["tech"].name for r in valid]
    nrmse_pct = [r["nrmse_post"] * 100 for r in valid]
    colors = [TECH_COLORS.get(r["tech"].name, "tab:gray") for r in valid]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(valid))
    bars = ax.bar(x, nrmse_pct, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, nrmse_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.axhline(y=NRMSE_THRESHOLD * 100, color="red", linewidth=1.5, linestyle="--",
               label=f"Threshold ({NRMSE_THRESHOLD*100:.0f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("NRMSE (% of Vdd)")
    ax.set_title("NN (LEVEL=73) Transient Verification: NRMSE by Technology")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=TECH_COLORS.get(n, "tab:gray"), edgecolor="black", label=n)
        for n in names
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
def run_single_tech(tech: TechConfig) -> Dict[str, Any]:
    """Run NGSPICE + NN transient for one tech and return metrics."""
    # 1. NGSPICE reference (BSIM-CMG OSDI)
    ng_data = run_ngspice(tech)

    # 2. PyCircuitSim with LEVEL=73 (NN)
    nn_data = run_pycircuitsim_nn(tech)

    # 3. Metrics — full range
    _, ng_vout_full, nn_vout_full, _, _ = interpolate_to_common_time(
        ng_data, nn_data, t_start=0.0,
    )
    metrics_full = compute_metrics(ng_vout_full, nn_vout_full, tech.vdd)

    # 4. Metrics — post-settling
    _, ng_vout_post, nn_vout_post, _, _ = interpolate_to_common_time(
        ng_data, nn_data, t_start=STARTUP_EXCLUSION,
    )
    metrics_post = compute_metrics(ng_vout_post, nn_vout_post, tech.vdd)

    nrmse_post = metrics_post["NRMSE (% of Vdd)"] / 100.0
    nrmse_full = metrics_full["NRMSE (% of Vdd)"] / 100.0
    max_err_mV = metrics_post["Max |error| (mV)"]
    passed = nrmse_post < NRMSE_THRESHOLD

    # 5. Plot
    results_dir = get_results_dir(tech)
    plot_path = results_dir / "comparison_tran.png"
    plot_comparison(tech, ng_data, nn_data, metrics_full, metrics_post, plot_path)

    return {
        "tech": tech,
        "nrmse_post": nrmse_post,
        "nrmse_full": nrmse_full,
        "max_err_mV": max_err_mV,
        "passed": passed,
    }


def main() -> int:
    """Run NN transient verification across all technologies."""
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("NN Compact Model (LEVEL=73) Transient Verification")
    print(f"  Ground truth: NGSPICE with BSIM-CMG OSDI")
    print(f"  Circuit: CMOS inverter, NFIN={NFIN}, Cload={CLOAD*1e15:.0f}fF")
    print(f"  Transient: tstep={TSTEP*1e12:.0f}ps, tstop={TSTOP*1e9:.0f}ns")
    print(f"  Acceptance: NRMSE < {NRMSE_THRESHOLD*100:.0f}% of Vdd (post-settling)")
    print("=" * 72)

    all_results: List[Dict[str, Any]] = []
    n_pass = 0
    n_fail = 0
    n_skip = 0
    n_error = 0

    for tech in ALL_TECHS:
        print(f"\n{'='*60}")
        print(f"  {tech.name} (VDD={tech.vdd}V, L_n={tech.l_nmos*1e9:.0f}nm, "
              f"L_p={tech.l_pmos*1e9:.0f}nm)")
        print(f"{'='*60}")

        # Check checkpoints
        if not check_checkpoints(tech):
            print(f"  SKIP: NN checkpoints not found for {tech.name}")
            n_skip += 1
            continue

        try:
            result = run_single_tech(tech)
            all_results.append(result)

            nrmse_pct = result["nrmse_post"] * 100
            if result["passed"]:
                print(f"\n  PASS: NRMSE={nrmse_pct:.2f}% < {NRMSE_THRESHOLD*100:.0f}%, "
                      f"Max|err|={result['max_err_mV']:.1f}mV")
                n_pass += 1
            else:
                print(f"\n  FAIL: NRMSE={nrmse_pct:.2f}% >= {NRMSE_THRESHOLD*100:.0f}%")
                n_fail += 1
        except Exception as exc:
            print(f"\n  ERROR: {exc}")
            import traceback
            traceback.print_exc()
            all_results.append({"tech": tech, "error": str(exc), "passed": False})
            n_error += 1

    # Summary
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Tech':<8s} | {'VDD':>5s} | {'NRMSE(%)':>10s} | {'MaxErr(mV)':>10s} | {'Status':>6s}")
    print(f"  {'-'*8} | {'-'*5} | {'-'*10} | {'-'*10} | {'-'*6}")

    for r in all_results:
        tech = r["tech"]
        if "error" in r:
            print(f"  {tech.name:<8s} | {tech.vdd:>5.2f} | {'ERROR':>10s} | {'---':>10s} | {'FAIL':>6s}")
        else:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  {tech.name:<8s} | {tech.vdd:>5.2f} | "
                  f"{r['nrmse_post']*100:>10.2f} | {r['max_err_mV']:>10.1f} | {status:>6s}")

    total = n_pass + n_fail + n_error
    print(f"\n  Result: {n_pass}/{total} PASS, {n_fail} FAIL, {n_error} ERROR, {n_skip} SKIP")

    # Summary plot
    if all_results:
        plot_summary(all_results, RESULTS_BASE / "nn_tran_summary.png")

    if n_fail + n_error > 0:
        print(f"\nSome tests FAILED. Consider retraining with charge-emphasized weights:")
        print(f"  python -m nn_model.train --mode finetune --w-charges 1.5 --w-caps 1.0")
        return 1
    elif n_pass == 0:
        print("\nNo tests were run. Make sure NN checkpoints exist.")
        return 1
    else:
        print(f"\nAll {n_pass} tests PASSED!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
