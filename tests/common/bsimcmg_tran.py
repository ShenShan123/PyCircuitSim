"""Shared infrastructure for BSIM-CMG transient verification.

Provides transient-specific test configuration, NGSPICE/PyCircuitSim runners,
comparison metrics, and plotting for the 3-level verification suite:
  Level 1: verify_bsimcmg_tran.py          (simple baseline)
  Level 2: verify_bsimcmg_tran_comprehensive.py (VT/L/NFIN sweeps)
  Level 3: verify_multi_tech_tran.py        (multi-tech, P/N ratios, geometry)

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
# Transient-specific constants
# ---------------------------------------------------------------------------
NRMSE_THRESHOLD: float = 0.05      # 5% of VDD
STARTUP_EXCLUSION: float = 0.1e-9  # 0.1 ns


# ---------------------------------------------------------------------------
# TestConfig — per-simulation configuration
# ---------------------------------------------------------------------------
@dataclass
class TestConfig:
    """One parametric test point for the inverter transient sweep."""
    tech: TechProfile
    vt: VtPair
    l_nmos: float        # NMOS channel length [m]
    l_pmos: float        # PMOS channel length [m]
    nfin_n: int          # NMOS fin count
    nfin_p: int          # PMOS fin count
    vdd: float           # Supply voltage [V]
    cload: float = 10e-15
    tr: float = 100e-12
    tf: float = 100e-12
    pw: float = 0.8e-9
    config_name: str = "baseline"
    sweep_type: str = "baseline"

    # PULSE voltages (auto-set in __post_init__)
    pulse_v1: float = 0.0
    pulse_v2: float = 0.0

    def __post_init__(self) -> None:
        if self.pulse_v2 == 0.0:
            self.pulse_v2 = self.vdd

    # -- Adaptive timing ---------------------------------------------------

    @property
    def tau_est(self) -> float:
        nfin_min = min(self.nfin_n, self.nfin_p)
        i_est = nfin_min * self.tech.i_per_fin
        return max(self.cload * self.vdd / i_est, 0.1e-9)

    @property
    def td(self) -> float:
        return max(0.5e-9, 10.0 * self.tr)

    @property
    def rest(self) -> float:
        return max(self.pw, 5.0 * self.tau_est)

    @property
    def per(self) -> float:
        return self.tr + self.pw + self.tf + self.rest

    @property
    def tstop(self) -> float:
        return self.td + 2.5 * self.per

    @property
    def tstep(self) -> float:
        return min(10e-12, self.tr / 10.0)

    @property
    def label(self) -> str:
        """Unique label for file naming: Tech_VT_ConfigName."""
        return f"{self.tech.name}_{self.vt.vt_name}_{self.config_name}"

    def summary(self) -> str:
        return (
            f"{self.label:35s}  VDD={self.vdd:.2f}V  "
            f"L={self.l_nmos*1e9:.0f}/{self.l_pmos*1e9:.0f}nm  "
            f"NFIN={self.nfin_n}/{self.nfin_p}  "
            f"Cload={self.cload*1e15:.0f}fF  "
            f"tr={self.tr*1e12:.0f}ps"
        )


def make_baseline(tech: TechProfile,
                  vt: Optional[VtPair] = None,
                  config_name: str = "baseline",
                  sweep_type: str = "baseline",
                  **overrides: Any) -> TestConfig:
    """Create a baseline TestConfig for a tech with optional overrides."""
    if vt is None:
        vt = tech.default_vt_pair
    kwargs: Dict[str, Any] = dict(
        tech=tech, vt=vt,
        l_nmos=tech.default_l_nmos, l_pmos=tech.default_l_pmos,
        nfin_n=tech.default_nfin, nfin_p=tech.default_nfin,
        vdd=tech.vdd, config_name=config_name, sweep_type=sweep_type,
    )
    kwargs.update(overrides)
    return TestConfig(**kwargs)


# ---------------------------------------------------------------------------
# Modelcard helpers
# ---------------------------------------------------------------------------
_merged_cache: Dict[Tuple[str, str, float, float], Path] = {}
_baked_cache: Dict[Tuple[str, str, float, float, int, int], Path] = {}


def get_merged_modelcard(config: TestConfig, work_dir: Path) -> Path:
    """Get or create merged NMOS+PMOS modelcard (unbaked, for PyCircuitSim)."""
    tech, vt = config.tech, config.vt
    key = (tech.name, vt.vt_name, config.l_nmos, config.l_pmos)
    if key in _merged_cache:
        return _merged_cache[key]

    if tech.single_file:
        path = tech.get_nmos_modelcard(vt, config.l_nmos)
        _merged_cache[key] = path
        return path

    nmos_src = tech.get_nmos_modelcard(vt, config.l_nmos)
    pmos_src = tech.get_pmos_modelcard(vt, config.l_pmos)
    if not nmos_src.exists():
        raise FileNotFoundError(f"NMOS modelcard not found: {nmos_src}")
    if not pmos_src.exists():
        raise FileNotFoundError(f"PMOS modelcard not found: {pmos_src}")

    l_n_nm = round(config.l_nmos * 1e9)
    l_p_nm = round(config.l_pmos * 1e9)
    merged = work_dir / f"merged_{tech.name}_{vt.vt_name}_ln{l_n_nm}_lp{l_p_nm}.lib"
    merged.write_text(nmos_src.read_text() + "\n" + pmos_src.read_text())
    _merged_cache[key] = merged
    return merged


def get_baked_modelcard(config: TestConfig, work_dir: Path) -> Path:
    """Get or create baked modelcard (for NGSPICE, with L/NFIN/TFIN injected)."""
    tech, vt = config.tech, config.vt
    key = (tech.name, vt.vt_name, config.l_nmos, config.l_pmos,
           config.nfin_n, config.nfin_p)
    if key in _baked_cache:
        return _baked_cache[key]

    merged = get_merged_modelcard(config, work_dir)

    l_n_nm = round(config.l_nmos * 1e9)
    l_p_nm = round(config.l_pmos * 1e9)
    baked = work_dir / (
        f"baked_{tech.name}_{vt.vt_name}_ln{l_n_nm}_lp{l_p_nm}"
        f"_nn{config.nfin_n}_np{config.nfin_p}.lib"
    )

    nmos_params: Dict[str, Any] = {
        "L": config.l_nmos, "NFIN": float(config.nfin_n),
        "TFIN": tech.tfin, "DEVTYPE": 1,
    }
    pmos_params: Dict[str, Any] = {
        "L": config.l_pmos, "NFIN": float(config.nfin_p),
        "TFIN": tech.tfin, "DEVTYPE": 0,
    }

    bake_inst_params(merged, baked, vt.nmos_model, nmos_params)
    bake_inst_params(baked, baked, vt.pmos_model, pmos_params)

    _baked_cache[key] = baked
    return baked


# ---------------------------------------------------------------------------
# NGSPICE netlist generation & runner
# ---------------------------------------------------------------------------
def create_ngspice_netlist(config: TestConfig, work_dir: Path) -> Path:
    """Generate an NGSPICE inverter transient netlist."""
    baked_lib = get_baked_modelcard(config, work_dir)
    netlist_path = work_dir / f"ngspice_{config.label}.cir"
    vt = config.vt

    content = f"""\
* BSIM-CMG CMOS Inverter Transient - NGSPICE ({config.label})
.include "{baked_lib}"
.temp 27
Vdd vdd 0 {config.vdd}
Vin in 0 PULSE({config.pulse_v1} {config.pulse_v2} {config.td} {config.tr} {config.tf} {config.pw} {config.per})
Np out in vdd vdd {vt.pmos_model}
Nn out in 0 0 {vt.nmos_model}
Cload out 0 {config.cload}
.ic V(out)={config.vdd}
.tran {config.tstep} {config.tstop} uic
.end
"""
    netlist_path.write_text(content)
    return netlist_path


def run_ngspice(config: TestConfig, work_dir: Path) -> Dict[str, np.ndarray]:
    """Run NGSPICE transient simulation and parse wrdata output."""
    netlist_path = create_ngspice_netlist(config, work_dir)
    csv_path = work_dir / f"ngspice_{config.label}.csv"
    log_path = work_dir / f"ngspice_{config.label}.log"
    runner_path = work_dir / f"ngspice_{config.label}_runner.cir"

    runner_content = f"""\
* NGSPICE transient runner ({config.label})
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

    lines = run_ngspice_subprocess(runner_path, log_path, csv_path)

    # Parse wrdata: header + data rows (time, v(out), time, v(in))
    data_rows = []
    for line in lines[1:]:
        stripped = line.strip()
        if stripped:
            data_rows.append([float(x) for x in stripped.split()])

    data = np.array(data_rows)
    return {
        "time": data[:, 0],
        "v(out)": data[:, 1],
        "v(in)": data[:, 3],
    }


# ---------------------------------------------------------------------------
# PyCircuitSim netlist generation & runner
# ---------------------------------------------------------------------------
def create_pycircuitsim_netlist(config: TestConfig, work_dir: Path) -> Path:
    """Generate a PyCircuitSim inverter transient netlist."""
    tech, vt = config.tech, config.vt
    l_n_nm = config.l_nmos * 1e9
    l_p_nm = config.l_pmos * 1e9

    netlist_path = work_dir / f"pycircuitsim_{config.label}.sp"
    content = f"""\
* BSIM-CMG Inverter Transient - PyCircuitSim ({config.label})
* Tech={tech.name} VT={vt.vt_name} VDD={config.vdd}V
* L_n={l_n_nm:.0f}nm L_p={l_p_nm:.0f}nm NFIN_N={config.nfin_n} NFIN_P={config.nfin_p}

Vdd 1 0 {config.vdd}
Vin 2 0 PULSE {config.pulse_v1} {config.pulse_v2} {config.td} {config.tr} {config.tf} {config.pw} {config.per}
Mp1 3 2 1 1 {vt.pmos_model} L={l_p_nm:.0f}n NFIN={config.nfin_p} TFIN={tech.tfin*1e9:.1f}n
Mn1 3 2 0 0 {vt.nmos_model} L={l_n_nm:.0f}n NFIN={config.nfin_n} TFIN={tech.tfin*1e9:.1f}n
Cload 3 0 {config.cload}
.ic V(3)={config.vdd}
.model {vt.nmos_model} NMOS (LEVEL=72)
.model {vt.pmos_model} PMOS (LEVEL=72)
.tran {config.tstep} {config.tstop}
.end
"""
    netlist_path.write_text(content)
    return netlist_path


def run_pycircuitsim(config: TestConfig, work_dir: Path) -> Dict[str, np.ndarray]:
    """Run PyCircuitSim transient simulation."""
    from pycircuitsim.parser import Parser
    from pycircuitsim.solver import DCSolver, TransientSolver

    tech, vt = config.tech, config.vt
    netlist_path = create_pycircuitsim_netlist(config, work_dir)
    merged = get_merged_modelcard(config, work_dir)

    logging.disable(logging.CRITICAL)
    try:
        parser = Parser(
            modelcard_path=str(merged),
            model_name_map={"NMOS": vt.nmos_model, "PMOS": vt.pmos_model},
        )
        parser.parse_file(str(netlist_path))
        circuit = parser.circuit
        time_step = parser.analysis_params["tstep"]
        final_time = parser.analysis_params["tstop"]

        initial_guess = circuit.initial_conditions or None
        op_solver = DCSolver(circuit, initial_guess=initial_guess,
                             use_source_stepping=True)
        op_solution = op_solver.solve()

        solver = TransientSolver(
            circuit, t_stop=final_time, dt=time_step,
            initial_guess=op_solution,
            use_gmin_stepping=True, gmin_initial=1e-9,
            gmin_final=1e-12, gmin_steps=5,
            use_pseudo_transient=True, pseudo_transient_steps=5,
            pseudo_transient_cap=1e-12,
            debug=False, nr_tolerance=1e-7,
        )
        results = solver.solve()
    finally:
        logging.disable(logging.NOTSET)

    return {
        "time": results["time"],
        "v(out)": results["3"],
        "v(in)": results["2"],
    }


# ---------------------------------------------------------------------------
# Comparison metrics
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
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_abs_err = float(np.max(np.abs(diff)))
    return {
        "RMSE (mV)": rmse * 1e3,
        "NRMSE (% of Vdd)": rmse / vdd * 100,
        "Max |error| (mV)": max_abs_err * 1e3,
        "Max |error| (% of Vdd)": max_abs_err / vdd * 100,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_single_comparison(
    ng_data: Dict[str, np.ndarray],
    py_data: Dict[str, np.ndarray],
    config: TestConfig,
    metrics_full: Dict[str, float],
    metrics_post: Dict[str, float],
    save_path: Path,
) -> None:
    """Generate a 3-panel comparison plot for one test config."""
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 9),
        gridspec_kw={"height_ratios": [0.8, 1.2, 0.8]},
    )
    ng_t_ns = ng_data["time"] * 1e9
    py_t_ns = py_data["time"] * 1e9

    # Panel 1: V(in)
    axes[0].plot(ng_t_ns, ng_data["v(in)"], "b-", label="NGSPICE", lw=1.5)
    axes[0].plot(py_t_ns, py_data["v(in)"], "r--", label="PyCircuitSim", lw=1.2, alpha=0.8)
    axes[0].set_ylabel("V(in) [V]")
    axes[0].set_title(
        f"BSIM-CMG Inverter Transient: {config.label}\n"
        f"Tech={config.tech.name}  VT={config.vt.vt_name}  VDD={config.vdd:.2f}V  "
        f"L={config.l_nmos*1e9:.0f}/{config.l_pmos*1e9:.0f}nm  "
        f"NFIN={config.nfin_n}/{config.nfin_p}  Cload={config.cload*1e15:.0f}fF"
    )
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.1, config.vdd + 0.1)

    # Panel 2: V(out)
    axes[1].plot(ng_t_ns, ng_data["v(out)"], "b-", label="NGSPICE", lw=1.5)
    axes[1].plot(py_t_ns, py_data["v(out)"], "r--", label="PyCircuitSim", lw=1.2, alpha=0.8)
    axes[1].axvline(x=STARTUP_EXCLUSION * 1e9, color="gray", lw=1, ls=":",
                    label=f"Startup excl ({STARTUP_EXCLUSION*1e9:.1f}ns)")
    axes[1].set_ylabel("V(out) [V]")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.1, config.vdd + 0.15)
    txt = (
        f"Full: NRMSE={metrics_full['NRMSE (% of Vdd)']:.2f}%\n"
        f"Post: NRMSE={metrics_post['NRMSE (% of Vdd)']:.2f}%, "
        f"Max|err|={metrics_post['Max |error| (mV)']:.1f}mV"
    )
    axes[1].text(0.02, 0.05, txt, transform=axes[1].transAxes, fontsize=9,
                 va="bottom", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # Panel 3: Error trace
    t_c, ng_v, py_v, _, _ = interpolate_to_common_time(
        ng_data, py_data, config, t_start=STARTUP_EXCLUSION)
    err_mv = (py_v - ng_v) * 1e3
    axes[2].plot(t_c * 1e9, err_mv, "g-", lw=0.8)
    axes[2].axhline(y=0, color="k", lw=0.5)
    thr_mv = config.vdd * NRMSE_THRESHOLD * 1e3
    axes[2].axhline(y=thr_mv, color="r", lw=0.5, ls="--",
                    label=f"{NRMSE_THRESHOLD*100:.0f}% Vdd = {thr_mv:.0f}mV")
    axes[2].axhline(y=-thr_mv, color="r", lw=0.5, ls="--")
    axes[2].set_ylabel("Error [mV]")
    axes[2].set_xlabel("Time [ns]")
    axes[2].set_title(f"Error (post {STARTUP_EXCLUSION*1e9:.1f}ns settling)", fontsize=10)
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Single-test orchestrator
# ---------------------------------------------------------------------------
def run_single_test(config: TestConfig, work_dir: Path,
                    verbose: bool = True) -> Dict[str, Any]:
    """Run NGSPICE + PyCircuitSim for one config and return metrics."""
    work_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n  [{config.label}]")
        print(f"    {config.summary()}")

    # 1. NGSPICE
    if verbose:
        print(f"    Running NGSPICE ...")
    ng_data = run_ngspice(config, work_dir)

    # 2. PyCircuitSim
    if verbose:
        print(f"    Running PyCircuitSim ...")
    py_data = run_pycircuitsim(config, work_dir)

    # 3. Metrics — full range
    _, ng_full, py_full, _, _ = interpolate_to_common_time(
        ng_data, py_data, config, t_start=0.0)
    metrics_full = compute_metrics(ng_full, py_full, config.vdd)

    # 3b. Metrics — post-settling
    _, ng_post, py_post, _, _ = interpolate_to_common_time(
        ng_data, py_data, config, t_start=STARTUP_EXCLUSION)
    metrics_post = compute_metrics(ng_post, py_post, config.vdd)

    nrmse = metrics_post["NRMSE (% of Vdd)"] / 100.0
    passed = nrmse < NRMSE_THRESHOLD

    # 4. Plot
    plot_path = work_dir / f"comparison_{config.label}.png"
    plot_single_comparison(ng_data, py_data, config, metrics_full, metrics_post, plot_path)

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"    NRMSE={nrmse*100:.2f}%, Max|err|={metrics_post['Max |error| (mV)']:.1f}mV "
              f"-> {status}")

    return {
        "config": config,
        "nrmse_post": nrmse,
        "max_err_mV": metrics_post["Max |error| (mV)"],
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Summary output helpers
# ---------------------------------------------------------------------------
def print_summary_table(results: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    """Print summary table and return (n_pass, n_fail, n_error)."""
    header = (
        f"{'Label':35s} | {'VDD':>5s} | {'L_n':>5s} | {'L_p':>5s} | "
        f"{'NFIN':>7s} | {'Cload':>7s} | {'tr':>5s} | "
        f"{'NRMSE%':>7s} | {'MaxErr':>8s} | {'Status':>6s}"
    )
    print(header)
    print("-" * len(header))

    n_pass = n_fail = n_error = 0
    for r in results:
        cfg: TestConfig = r["config"]
        if "error" in r:
            n_error += 1
            print(
                f"{cfg.label:35s} | {cfg.vdd:5.2f} | "
                f"{cfg.l_nmos*1e9:4.0f}n | {cfg.l_pmos*1e9:4.0f}n | "
                f"{cfg.nfin_n:3d}/{cfg.nfin_p:<3d} | {cfg.cload*1e15:5.0f}fF | "
                f"{cfg.tr*1e12:3.0f}ps | "
                f"{'ERR':>7s} | {'ERR':>8s} | {'ERROR':>6s}"
            )
        else:
            passed = r["passed"]
            if passed:
                n_pass += 1
            else:
                n_fail += 1
            status = "PASS" if passed else "FAIL"
            print(
                f"{cfg.label:35s} | {cfg.vdd:5.2f} | "
                f"{cfg.l_nmos*1e9:4.0f}n | {cfg.l_pmos*1e9:4.0f}n | "
                f"{cfg.nfin_n:3d}/{cfg.nfin_p:<3d} | {cfg.cload*1e15:5.0f}fF | "
                f"{cfg.tr*1e12:3.0f}ps | "
                f"{r['nrmse_post']*100:7.2f} | {r['max_err_mV']:6.1f}mV | "
                f"{status:>6s}"
            )
    return n_pass, n_fail, n_error


def save_summary_csv(results: List[Dict[str, Any]], csv_path: Path) -> None:
    """Save results to CSV file."""
    with csv_path.open("w") as f:
        f.write(
            "tech,vt,config,sweep_type,vdd,l_nmos_nm,l_pmos_nm,"
            "nfin_n,nfin_p,cload_fF,tr_ps,pw_ns,"
            "nrmse_pct,max_err_mV,status\n"
        )
        for r in results:
            cfg: TestConfig = r["config"]
            if "error" in r:
                status = "ERROR"
                nrmse_s = max_s = ""
            else:
                status = "PASS" if r["passed"] else "FAIL"
                nrmse_s = f"{r['nrmse_post']*100:.4f}"
                max_s = f"{r['max_err_mV']:.2f}"
            f.write(
                f"{cfg.tech.name},{cfg.vt.vt_name},{cfg.config_name},"
                f"{cfg.sweep_type},{cfg.vdd:.2f},"
                f"{cfg.l_nmos*1e9:.0f},{cfg.l_pmos*1e9:.0f},"
                f"{cfg.nfin_n},{cfg.nfin_p},"
                f"{cfg.cload*1e15:.1f},{cfg.tr*1e12:.1f},{cfg.pw*1e9:.2f},"
                f"{nrmse_s},{max_s},{status}\n"
            )
    print(f"[CSV] Summary saved: {csv_path}")


def plot_summary_bar(results: List[Dict[str, Any]], save_path: Path,
                     title: str = "Transient Verification Summary") -> None:
    """Generate bar chart of NRMSE across all configs, colored by tech."""
    _plot_summary_bar(results, save_path, title,
                      nrmse_key="nrmse_post", threshold=NRMSE_THRESHOLD,
                      y_label="NRMSE (% of Vdd)")


def run_test_suite(
    configs: List[TestConfig],
    results_dir: Path,
    title: str = "Transient Verification",
) -> int:
    """Run a list of TestConfigs and produce summary output. Returns exit code."""
    return _run_test_suite(
        configs, results_dir, title,
        acceptance_msg=f"NRMSE < {NRMSE_THRESHOLD*100:.0f}% of Vdd (post-settling)",
        run_single_fn=run_single_test,
        print_summary_fn=print_summary_table,
        save_csv_fn=save_summary_csv,
        plot_bar_fn=plot_summary_bar,
    )
