"""Shared infrastructure for BSIM-CMG DC sweep verification.

Provides DC-specific test configuration, NGSPICE/PyCircuitSim DC runners,
comparison metrics, and plotting for the 3-level DC verification suite:
  Level 1: verify_bsimcmg_dc.py              (simple NMOS/PMOS Id-Vgs)
  Level 2: verify_bsimcmg_dc_comprehensive.py (VT/L/NFIN sweeps, all techs)
  Level 3: verify_multi_tech_dc.py            (inverter VTC, multi-tech parametric)

Technology profiles (TechProfile, VtPair, ALL_TECHS) and generic helpers
are imported from test_common.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Shared test infrastructure (from test_common)
# ---------------------------------------------------------------------------
from tests.common.base import (
    PROJECT_ROOT, OSDI_PATH, MODELCARDS_DIR, NGSPICE_BIN,
    VtPair, TechProfile, ALL_TECHS, TECH_ORDER, TECH_COLORS,
    bake_inst_params,
    run_ngspice_subprocess,
    plot_summary_bar as _plot_summary_bar,
    run_test_suite as _run_test_suite,
)

# ---------------------------------------------------------------------------
# DC-specific constants
# ---------------------------------------------------------------------------
NRMSE_THRESHOLD: float = 0.01        # 1% of normalization value
MAX_REL_ERR_THRESHOLD: float = 0.05  # 5%
NFIN_SWEEP_VALUES: List[int] = [2, 5, 10]

# ---------------------------------------------------------------------------
# Test types
# ---------------------------------------------------------------------------
NMOS_IDVGS = "nmos_idvgs"
PMOS_IDVGS = "pmos_idvgs"
INVERTER_VTC = "inverter_vtc"

_TYPE_SHORT = {NMOS_IDVGS: "nmos", PMOS_IDVGS: "pmos", INVERTER_VTC: "inv"}


# ---------------------------------------------------------------------------
# DCTestConfig
# ---------------------------------------------------------------------------
@dataclass
class DCTestConfig:
    """Configuration for one DC sweep test."""
    tech: TechProfile
    vt: VtPair
    test_type: str           # NMOS_IDVGS, PMOS_IDVGS, INVERTER_VTC
    l_nmos: float            # NMOS channel length [m]
    l_pmos: float            # PMOS channel length [m]
    nfin_n: int              # NMOS fin count
    nfin_p: int              # PMOS fin count
    vdd: float               # Supply voltage [V]
    vds_bias: float = 0.0    # |Vds| bias for single-device tests [V]; 0 = auto
    sweep_start: float = 0.0
    sweep_stop: float = 0.0  # 0 = auto (VDD)
    sweep_step: float = 0.01
    config_name: str = "baseline"
    sweep_type: str = "baseline"

    def __post_init__(self) -> None:
        if self.sweep_stop == 0.0:
            self.sweep_stop = self.vdd
        if self.vds_bias == 0.0:
            self.vds_bias = round(self.vdd * 0.5, 2)

    @property
    def label(self) -> str:
        short = _TYPE_SHORT.get(self.test_type, self.test_type)
        return f"{self.tech.name}_{self.vt.vt_name}_{short}_{self.config_name}"

    def summary(self) -> str:
        if self.test_type == INVERTER_VTC:
            return (
                f"{self.label:40s}  VDD={self.vdd:.2f}V  "
                f"L={self.l_nmos*1e9:.0f}/{self.l_pmos*1e9:.0f}nm  "
                f"NFIN={self.nfin_n}/{self.nfin_p}"
            )
        elif self.test_type == NMOS_IDVGS:
            return (
                f"{self.label:40s}  VDD={self.vdd:.2f}V  "
                f"Vds={self.vds_bias:.2f}V  "
                f"L={self.l_nmos*1e9:.0f}nm  NFIN={self.nfin_n}"
            )
        else:
            return (
                f"{self.label:40s}  VDD={self.vdd:.2f}V  "
                f"|Vds|={self.vds_bias:.2f}V  "
                f"L={self.l_pmos*1e9:.0f}nm  NFIN={self.nfin_p}"
            )


def make_dc_config(
    tech: TechProfile,
    test_type: str,
    vt: Optional[VtPair] = None,
    config_name: str = "baseline",
    sweep_type: str = "baseline",
    **overrides: Any,
) -> DCTestConfig:
    """Create a DCTestConfig with tech defaults and optional overrides."""
    if vt is None:
        vt = tech.default_vt_pair
    kwargs: Dict[str, Any] = dict(
        tech=tech, vt=vt, test_type=test_type,
        l_nmos=tech.default_l_nmos, l_pmos=tech.default_l_pmos,
        nfin_n=tech.default_nfin, nfin_p=tech.default_nfin,
        vdd=tech.vdd, config_name=config_name, sweep_type=sweep_type,
    )
    kwargs.update(overrides)
    return DCTestConfig(**kwargs)


def is_config_available(config: DCTestConfig) -> bool:
    """Check if required modelcard(s) exist for this config."""
    tech, vt = config.tech, config.vt
    if tech.single_file:
        return True
    if config.test_type == NMOS_IDVGS:
        return tech.get_nmos_modelcard(vt, config.l_nmos).exists()
    elif config.test_type == PMOS_IDVGS:
        return tech.get_pmos_modelcard(vt, config.l_pmos).exists()
    else:
        return tech.is_combo_available(vt, config.l_nmos, config.l_pmos)


# ---------------------------------------------------------------------------
# Modelcard helpers
# ---------------------------------------------------------------------------
_baked_cache: Dict[tuple, Path] = {}


def get_baked_modelcard(config: DCTestConfig, work_dir: Path) -> Path:
    """Create baked modelcard for NGSPICE with L/NFIN/TFIN/DEVTYPE injected."""
    tech, vt = config.tech, config.vt
    key = (tech.name, vt.vt_name, config.test_type,
           config.l_nmos, config.l_pmos, config.nfin_n, config.nfin_p)
    if key in _baked_cache:
        return _baked_cache[key]

    baked = work_dir / f"baked_{config.label}.lib"

    if config.test_type == NMOS_IDVGS:
        src = tech.get_nmos_modelcard(vt, config.l_nmos)
        if not src.exists():
            raise FileNotFoundError(f"NMOS modelcard: {src}")
        baked.write_text(src.read_text())
        bake_inst_params(baked, baked, vt.nmos_model,
                         {"L": config.l_nmos, "NFIN": float(config.nfin_n),
                          "TFIN": tech.tfin, "DEVTYPE": 1})

    elif config.test_type == PMOS_IDVGS:
        src = tech.get_pmos_modelcard(vt, config.l_pmos)
        if not src.exists():
            raise FileNotFoundError(f"PMOS modelcard: {src}")
        baked.write_text(src.read_text())
        bake_inst_params(baked, baked, vt.pmos_model,
                         {"L": config.l_pmos, "NFIN": float(config.nfin_p),
                          "TFIN": tech.tfin, "DEVTYPE": 0})

    else:  # INVERTER_VTC
        if tech.single_file:
            src = tech.get_nmos_modelcard(vt, config.l_nmos)
            baked.write_text(src.read_text())
        else:
            nmos_src = tech.get_nmos_modelcard(vt, config.l_nmos)
            pmos_src = tech.get_pmos_modelcard(vt, config.l_pmos)
            if not nmos_src.exists():
                raise FileNotFoundError(f"NMOS modelcard: {nmos_src}")
            if not pmos_src.exists():
                raise FileNotFoundError(f"PMOS modelcard: {pmos_src}")
            baked.write_text(nmos_src.read_text() + "\n" + pmos_src.read_text())
        bake_inst_params(baked, baked, vt.nmos_model,
                         {"L": config.l_nmos, "NFIN": float(config.nfin_n),
                          "TFIN": tech.tfin, "DEVTYPE": 1})
        bake_inst_params(baked, baked, vt.pmos_model,
                         {"L": config.l_pmos, "NFIN": float(config.nfin_p),
                          "TFIN": tech.tfin, "DEVTYPE": 0})

    _baked_cache[key] = baked
    return baked


def get_modelcard_for_pycircuitsim(config: DCTestConfig, work_dir: Path) -> Path:
    """Get modelcard path for PyCircuitSim (unbaked)."""
    tech, vt = config.tech, config.vt
    if tech.single_file:
        return tech.get_nmos_modelcard(vt, config.l_nmos)

    if config.test_type == NMOS_IDVGS:
        return tech.get_nmos_modelcard(vt, config.l_nmos)
    elif config.test_type == PMOS_IDVGS:
        return tech.get_pmos_modelcard(vt, config.l_pmos)
    else:
        nmos_src = tech.get_nmos_modelcard(vt, config.l_nmos)
        pmos_src = tech.get_pmos_modelcard(vt, config.l_pmos)
        merged = work_dir / f"merged_{config.label}.lib"
        if not merged.exists():
            merged.write_text(nmos_src.read_text() + "\n" + pmos_src.read_text())
        return merged


# ---------------------------------------------------------------------------
# NGSPICE DC runner
# ---------------------------------------------------------------------------
def create_ngspice_dc_netlist(
    config: DCTestConfig, work_dir: Path,
) -> Tuple[Path, List[str]]:
    """Generate NGSPICE DC sweep netlist. Returns (netlist_path, signal_names)."""
    baked = get_baked_modelcard(config, work_dir)
    netlist_path = work_dir / f"ngspice_{config.label}.cir"
    vt = config.vt

    if config.test_type == NMOS_IDVGS:
        content = (
            f"* NMOS Id-Vgs ({config.label})\n"
            f'.include "{baked}"\n'
            f".temp 27\n"
            f"Vds d 0 {config.vds_bias}\n"
            f"Vgs g 0 0.0\n"
            f"N1 d g 0 0 {vt.nmos_model}\n"
            f".dc Vgs {config.sweep_start} {config.sweep_stop} {config.sweep_step}\n"
            f".end\n"
        )
        signals = ["i(Vds)"]

    elif config.test_type == PMOS_IDVGS:
        vd = round(config.vdd - config.vds_bias, 4)
        content = (
            f"* PMOS |Id| vs Vg ({config.label})\n"
            f'.include "{baked}"\n'
            f".temp 27\n"
            f"Vdd vdd 0 {config.vdd}\n"
            f"Vdrain drain 0 {vd}\n"
            f"Vg gate 0 0.0\n"
            f"N1 drain gate vdd vdd {vt.pmos_model}\n"
            f".dc Vg {config.sweep_start} {config.sweep_stop} {config.sweep_step}\n"
            f".end\n"
        )
        signals = ["i(Vdrain)"]

    else:  # INVERTER_VTC
        content = (
            f"* CMOS Inverter VTC ({config.label})\n"
            f'.include "{baked}"\n'
            f".temp 27\n"
            f"Vdd vdd 0 {config.vdd}\n"
            f"Vin in 0 0.0\n"
            f"Np out in vdd vdd {vt.pmos_model}\n"
            f"Nn out in 0 0 {vt.nmos_model}\n"
            f".dc Vin {config.sweep_start} {config.sweep_stop} {config.sweep_step}\n"
            f".end\n"
        )
        signals = ["v(out)"]

    netlist_path.write_text(content)
    return netlist_path, signals


def run_ngspice_dc(config: DCTestConfig, work_dir: Path) -> Dict[str, np.ndarray]:
    """Run NGSPICE DC sweep and parse wrdata output.

    Returns dict with 'sweep' and signal values.
    """
    netlist_path, signals = create_ngspice_dc_netlist(config, work_dir)
    csv_path = work_dir / f"ngspice_{config.label}.csv"
    log_path = work_dir / f"ngspice_{config.label}.log"
    runner_path = work_dir / f"ngspice_{config.label}_runner.cir"

    signal_str = " ".join(signals)
    runner_content = (
        f"* NGSPICE DC runner ({config.label})\n"
        f".control\n"
        f"osdi {OSDI_PATH}\n"
        f"source {netlist_path}\n"
        f"set filetype=ascii\n"
        f"set wr_vecnames\n"
        f"run\n"
        f"wrdata {csv_path} {signal_str}\n"
        f".endc\n"
        f".end\n"
    )
    runner_path.write_text(runner_content)

    lines = run_ngspice_subprocess(runner_path, log_path, csv_path)

    # Parse wrdata: header + data rows
    data_rows = []
    for line in lines[1:]:
        stripped = line.strip()
        if stripped:
            data_rows.append([float(x) for x in stripped.split()])
    data = np.array(data_rows)

    # Sanity check: NaN/Inf indicates OSDI parameter issues
    if not np.all(np.isfinite(data)):
        raise RuntimeError("NGSPICE output contains NaN/Inf (likely OSDI parameter issues)")

    result: Dict[str, np.ndarray] = {"sweep": data[:, 0]}
    if len(signals) == 1:
        result[signals[0]] = data[:, 1]
    else:
        for i, sig in enumerate(signals):
            result[sig] = data[:, i * 2 + 1]
    return result


# ---------------------------------------------------------------------------
# PyCircuitSim DC runner
# ---------------------------------------------------------------------------
def create_pycircuitsim_dc_netlist(config: DCTestConfig, work_dir: Path) -> Path:
    """Generate PyCircuitSim DC sweep netlist."""
    tech, vt = config.tech, config.vt
    netlist_path = work_dir / f"pycircuitsim_{config.label}.sp"

    if config.test_type == NMOS_IDVGS:
        l_nm = config.l_nmos * 1e9
        content = (
            f"* NMOS Id-Vgs ({config.label})\n"
            f"Vds 1 0 {config.vds_bias}\n"
            f"Vgs 2 0 0.0\n"
            f"Mn1 1 2 0 0 {vt.nmos_model} L={l_nm:.0f}n"
            f" NFIN={config.nfin_n} TFIN={tech.tfin*1e9:.1f}n\n"
            f".model {vt.nmos_model} NMOS (LEVEL=72)\n"
            f".dc Vgs {config.sweep_start} {config.sweep_stop} {config.sweep_step}\n"
            f".end\n"
        )

    elif config.test_type == PMOS_IDVGS:
        l_nm = config.l_pmos * 1e9
        vd = round(config.vdd - config.vds_bias, 4)
        content = (
            f"* PMOS |Id| vs Vg ({config.label})\n"
            f"Vdd 1 0 {config.vdd}\n"
            f"Vdrain 3 0 {vd}\n"
            f"Vg 2 0 0.0\n"
            f"Mp1 3 2 1 1 {vt.pmos_model} L={l_nm:.0f}n"
            f" NFIN={config.nfin_p} TFIN={tech.tfin*1e9:.1f}n\n"
            f".model {vt.pmos_model} PMOS (LEVEL=72)\n"
            f".dc Vg {config.sweep_start} {config.sweep_stop} {config.sweep_step}\n"
            f".end\n"
        )

    else:  # INVERTER_VTC
        l_n_nm = config.l_nmos * 1e9
        l_p_nm = config.l_pmos * 1e9
        content = (
            f"* CMOS Inverter VTC ({config.label})\n"
            f"Vdd 1 0 {config.vdd}\n"
            f"Vin 2 0 0.0\n"
            f"Mp1 3 2 1 1 {vt.pmos_model} L={l_p_nm:.0f}n"
            f" NFIN={config.nfin_p} TFIN={tech.tfin*1e9:.1f}n\n"
            f"Mn1 3 2 0 0 {vt.nmos_model} L={l_n_nm:.0f}n"
            f" NFIN={config.nfin_n} TFIN={tech.tfin*1e9:.1f}n\n"
            f".model {vt.nmos_model} NMOS (LEVEL=72)\n"
            f".model {vt.pmos_model} PMOS (LEVEL=72)\n"
            f".dc Vin {config.sweep_start} {config.sweep_stop} {config.sweep_step}\n"
            f".end\n"
        )

    netlist_path.write_text(content)
    return netlist_path


def run_pycircuitsim_dc(config: DCTestConfig, work_dir: Path) -> Dict[str, np.ndarray]:
    """Run PyCircuitSim DC sweep. Returns {sweep, signal: values}."""
    from pycircuitsim.parser import Parser
    from pycircuitsim.simulation import run_dc_sweep
    from pycircuitsim.visualizer import Visualizer

    netlist_path = create_pycircuitsim_dc_netlist(config, work_dir)
    modelcard = get_modelcard_for_pycircuitsim(config, work_dir)

    if config.test_type == NMOS_IDVGS:
        name_map = {"NMOS": config.vt.nmos_model}
    elif config.test_type == PMOS_IDVGS:
        name_map = {"PMOS": config.vt.pmos_model}
    else:
        name_map = {"NMOS": config.vt.nmos_model, "PMOS": config.vt.pmos_model}

    logging.disable(logging.CRITICAL)
    try:
        parser = Parser(
            modelcard_path=str(modelcard),
            model_name_map=name_map,
        )
        parser.parse_file(str(netlist_path))
        circuit = parser.circuit

        vis = Visualizer()
        out_dir = work_dir / f"pycircuitsim_{config.label}_out"
        out_dir.mkdir(parents=True, exist_ok=True)

        results = run_dc_sweep(
            circuit, parser.analysis_params, vis, out_dir, config.label,
        )
    finally:
        logging.disable(logging.NOTSET)

    # Extract sweep (node 2 = gate/input) and signal
    sweep = np.array(results["2"])

    if config.test_type == NMOS_IDVGS:
        signal = np.abs(np.array(results["i(Mn1)"]))
    elif config.test_type == PMOS_IDVGS:
        signal = np.abs(np.array(results["i(Mp1)"]))
    else:
        signal = np.array(results["3"])

    return {"sweep": sweep, "signal": signal}


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------
def compute_dc_metrics(
    ng_sweep: np.ndarray,
    ng_values: np.ndarray,
    py_sweep: np.ndarray,
    py_values: np.ndarray,
) -> Dict[str, float]:
    """Compare NGSPICE vs PyCircuitSim DC curves.

    Interpolates PyCircuitSim onto NGSPICE sweep points.
    """
    common_start = max(ng_sweep[0], py_sweep[0])
    common_stop = min(ng_sweep[-1], py_sweep[-1])
    mask = (ng_sweep >= common_start - 1e-10) & (ng_sweep <= common_stop + 1e-10)
    ng_common = ng_sweep[mask]
    ng_vals = ng_values[mask]
    py_interp = np.interp(ng_common, py_sweep, py_values)

    diff = np.abs(py_interp - ng_vals)
    max_val = np.max(np.abs(ng_vals))

    rmse = float(np.sqrt(np.mean(diff ** 2)))
    nrmse = rmse / max_val if max_val > 0 else float("inf")
    max_abs_err = float(np.max(diff))

    # Relative error at significant points (> 1% of max)
    significant = np.abs(ng_vals) > 0.01 * max_val
    if np.any(significant):
        rel_errs = diff[significant] / np.maximum(np.abs(ng_vals[significant]), 1e-30)
        max_rel_err = float(np.max(rel_errs))
    else:
        max_rel_err = 0.0

    return {
        "rmse": rmse,
        "nrmse": nrmse,
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "n_common_points": len(ng_common),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_dc_comparison(
    ng_sweep: np.ndarray,
    ng_values: np.ndarray,
    py_sweep: np.ndarray,
    py_values: np.ndarray,
    config: DCTestConfig,
    metrics: Dict[str, float],
    save_path: Path,
    log_scale: bool = False,
) -> None:
    """2-panel comparison: overlay + error trace."""
    if config.test_type in (NMOS_IDVGS, PMOS_IDVGS):
        xlabel, ylabel = "Vgs (V)", "|Id| (A)"
    else:
        xlabel, ylabel = "Vin (V)", "Vout (V)"

    fig, axes = plt.subplots(2, 1, figsize=(10, 8),
                              gridspec_kw={"height_ratios": [3, 1]})

    ax1 = axes[0]
    ax1.plot(ng_sweep, ng_values, "b-", lw=2, label="NGSPICE")
    ax1.plot(py_sweep, py_values, "r--", lw=1.5, label="PyCircuitSim")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(f"DC: {config.label}")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    if log_scale and np.any(ng_values > 0):
        ax1.set_yscale("log")
        pos = ng_values[ng_values > 0]
        ax1.set_ylim(bottom=max(1e-12, np.min(pos) * 0.1))

    txt = (
        f"NRMSE: {metrics['nrmse']*100:.4f}%\n"
        f"Max |err|: {metrics['max_abs_err']:.4e}\n"
        f"Max rel err: {metrics['max_rel_err']*100:.2f}%"
    )
    ax1.text(0.02, 0.98, txt, transform=ax1.transAxes, fontsize=9,
             va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # Error trace
    ax2 = axes[1]
    common_start = max(ng_sweep[0], py_sweep[0])
    common_stop = min(ng_sweep[-1], py_sweep[-1])
    mask = (ng_sweep >= common_start - 1e-10) & (ng_sweep <= common_stop + 1e-10)
    ng_c = ng_sweep[mask]
    ng_v = ng_values[mask]
    py_i = np.interp(ng_c, py_sweep, py_values)
    ax2.plot(ng_c, py_i - ng_v, "g-", lw=1)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Error (PySim - NGSPICE)")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="k", lw=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dc_summary_bar(results: List[Dict[str, Any]], save_path: Path,
                        title: str = "DC Verification Summary") -> None:
    """Bar chart of NRMSE across all configs, colored by tech."""
    _plot_summary_bar(results, save_path, title,
                      nrmse_key="nrmse", threshold=NRMSE_THRESHOLD,
                      y_label="NRMSE (%)")


# ---------------------------------------------------------------------------
# Single-test orchestrator
# ---------------------------------------------------------------------------
def run_single_dc_test(config: DCTestConfig, work_dir: Path,
                       verbose: bool = True) -> Dict[str, Any]:
    """Run one DC test: NGSPICE + PyCircuitSim + metrics + plot."""
    work_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n  [{config.label}]")
        print(f"    {config.summary()}")

    # 1. NGSPICE
    if verbose:
        print("    Running NGSPICE ...")
    ng_data = run_ngspice_dc(config, work_dir)
    ng_sweep = ng_data["sweep"]

    if config.test_type == NMOS_IDVGS:
        ng_signal = np.abs(ng_data["i(Vds)"])
    elif config.test_type == PMOS_IDVGS:
        ng_signal = np.abs(ng_data["i(Vdrain)"])
    else:
        ng_signal = ng_data["v(out)"]

    # 2. PyCircuitSim
    if verbose:
        print("    Running PyCircuitSim ...")
    py_data = run_pycircuitsim_dc(config, work_dir)
    py_sweep = py_data["sweep"]
    py_signal = py_data["signal"]

    # 3. Metrics
    metrics = compute_dc_metrics(ng_sweep, ng_signal, py_sweep, py_signal)
    nrmse = metrics["nrmse"]
    max_rel = metrics["max_rel_err"]

    # Sanity: extreme NRMSE or MaxRelErr means garbage output (OSDI modelcard issues)
    if nrmse > 1.0 or max_rel > 1.0 or not np.isfinite(nrmse):
        if verbose:
            print(f"    Output validation failed: NRMSE={nrmse*100:.1f}%,"
                  f" MaxRel={max_rel*100:.1f}%"
                  " (likely OSDI parameter issues)")
        return {"config": config,
                "error": f"garbage output (NRMSE={nrmse*100:.1f}%)",
                "passed": False}

    passed = nrmse < NRMSE_THRESHOLD and max_rel < MAX_REL_ERR_THRESHOLD

    # 4. Plots
    plot_path = work_dir / f"comparison_{config.label}.png"
    plot_dc_comparison(ng_sweep, ng_signal, py_sweep, py_signal,
                       config, metrics, plot_path)
    if config.test_type in (NMOS_IDVGS, PMOS_IDVGS):
        log_path = work_dir / f"comparison_{config.label}_log.png"
        plot_dc_comparison(ng_sweep, ng_signal, py_sweep, py_signal,
                           config, metrics, log_path, log_scale=True)

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"    NRMSE={nrmse*100:.4f}%, MaxRel={max_rel*100:.2f}% -> {status}")

    return {
        "config": config,
        "nrmse": nrmse,
        "max_rel_err": max_rel,
        "max_abs_err": metrics["max_abs_err"],
        "n_points": metrics["n_common_points"],
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Summary output helpers
# ---------------------------------------------------------------------------
def print_dc_summary_table(results: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    """Print summary table. Returns (n_pass, n_fail, n_error)."""
    header = (
        f"{'Label':40s} | {'Type':>5s} | {'VDD':>5s} | {'L_n':>5s} | {'L_p':>5s} | "
        f"{'NFIN':>7s} | {'NRMSE%':>7s} | {'MaxRel%':>8s} | {'Status':>6s}"
    )
    print(header)
    print("-" * len(header))

    n_pass = n_fail = n_error = 0
    for r in results:
        cfg: DCTestConfig = r["config"]
        short = _TYPE_SHORT.get(cfg.test_type, "?")
        if "error" in r:
            n_error += 1
            print(
                f"{cfg.label:40s} | {short:>5s} | {cfg.vdd:5.2f} | "
                f"{cfg.l_nmos*1e9:4.0f}n | {cfg.l_pmos*1e9:4.0f}n | "
                f"{cfg.nfin_n:3d}/{cfg.nfin_p:<3d} | "
                f"{'ERR':>7s} | {'ERR':>8s} | {'ERROR':>6s}"
            )
        else:
            if r["passed"]:
                n_pass += 1
            else:
                n_fail += 1
            status = "PASS" if r["passed"] else "FAIL"
            print(
                f"{cfg.label:40s} | {short:>5s} | {cfg.vdd:5.2f} | "
                f"{cfg.l_nmos*1e9:4.0f}n | {cfg.l_pmos*1e9:4.0f}n | "
                f"{cfg.nfin_n:3d}/{cfg.nfin_p:<3d} | "
                f"{r['nrmse']*100:7.4f} | {r['max_rel_err']*100:7.2f}% | "
                f"{status:>6s}"
            )
    return n_pass, n_fail, n_error


def save_dc_summary_csv(results: List[Dict[str, Any]], csv_path: Path) -> None:
    """Save results to CSV file."""
    with csv_path.open("w") as f:
        f.write(
            "tech,vt,test_type,config,sweep_type,vdd,"
            "l_nmos_nm,l_pmos_nm,nfin_n,nfin_p,"
            "nrmse_pct,max_rel_err_pct,max_abs_err,status\n"
        )
        for r in results:
            cfg: DCTestConfig = r["config"]
            if "error" in r:
                status = "ERROR"
                nrmse_s = max_r_s = max_a_s = ""
            else:
                status = "PASS" if r["passed"] else "FAIL"
                nrmse_s = f"{r['nrmse']*100:.4f}"
                max_r_s = f"{r['max_rel_err']*100:.4f}"
                max_a_s = f"{r['max_abs_err']:.6e}"
            f.write(
                f"{cfg.tech.name},{cfg.vt.vt_name},{cfg.test_type},"
                f"{cfg.config_name},{cfg.sweep_type},{cfg.vdd:.2f},"
                f"{cfg.l_nmos*1e9:.0f},{cfg.l_pmos*1e9:.0f},"
                f"{cfg.nfin_n},{cfg.nfin_p},"
                f"{nrmse_s},{max_r_s},{max_a_s},{status}\n"
            )
    print(f"[CSV] Summary saved: {csv_path}")


def run_dc_test_suite(
    configs: List[DCTestConfig],
    results_dir: Path,
    title: str = "DC Verification",
) -> int:
    """Run a list of DCTestConfigs and produce summary. Returns exit code."""
    return _run_test_suite(
        configs, results_dir, title,
        acceptance_msg=(f"NRMSE < {NRMSE_THRESHOLD*100:.0f}%,"
                        f" MaxRelErr < {MAX_REL_ERR_THRESHOLD*100:.0f}%"),
        run_single_fn=run_single_dc_test,
        print_summary_fn=print_dc_summary_table,
        save_csv_fn=save_dc_summary_csv,
        plot_bar_fn=plot_dc_summary_bar,
    )
