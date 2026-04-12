#!/usr/bin/env python3
"""BSIMAR v4 (LEVEL=74 tech-code) inverter verification against PyCMG (LEVEL=72).

Tests:
  1. NMOS DC sweep (Id-Vgs at Vds=VDD/2)
  2. PMOS DC sweep (Id-Vgs at Vds=-VDD/2)
  3. Inverter VTC (Vout vs Vin via simple NR)
  4. Inverter transient (PyCircuitSim LEVEL=74 vs NGSPICE BSIM-CMG)

Default target: TSMC5 SVT (VDD=0.65V, NMOS L=16nm, PMOS L=20nm, NFIN=10).
"""
from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path bootstrap ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))

from bsimar.config import (
    TECH_CONFIGS, NNTechConfig, CHECKPOINT_DIR, OSDI_PATH as _OSDI_PATH,
    tech_variant_to_code, UNKNOWN_CODE_ID,
)
from pycmg import Model, Instance
from pycmg.tech import TECH_REGISTRY, resolve_modelcard
from helpers import bake_inst_params

from tests.common.nn import nrmse

# ── Constants ───────────────────────────────────────────────────────────────
OSDI_PATH = PROJECT_ROOT / "external_compact_models" / "PyCMG" / "build" / "osdi" / "bsimcmg.osdi"
NGSPICE_BIN = "/usr/local/ngspice-45.2/bin/ngspice"
RESULTS_DIR = PROJECT_ROOT / "tests" / "verify_bsimar_v4_inverter_results"

NFIN = 10
CLOAD = 10e-15          # 10 fF
TR = TF = 100e-12       # 100 ps rise/fall
PW = 0.8e-9             # 800 ps pulse width
TD = 0.5e-9             # 500 ps delay
TSTEP = 10e-12          # 10 ps
TSTOP = 5e-9            # 5 ns
STARTUP_EXCLUSION = 0.1e-9  # 0.1 ns

# Acceptance criteria
DC_NRMSE_THRESHOLD = 10.0     # % for single-device DC sweeps
VTC_NRMSE_THRESHOLD = 10.0    # % for inverter VTC
TRAN_NRMSE_THRESHOLD = 15.0   # % for transient


# ── Tech-specific config ────────────────────────────────────────────────────
@dataclass
class TestTechConfig:
    name: str
    tech_key: str         # key for TECH_CONFIGS and tech_variant_to_code
    variant: str          # e.g. "svt", "lvt"
    vdd: float
    l_nmos: float         # in metres
    l_pmos: float
    nmos_pdk: str         # PyCMG model name (e.g. "nch_svt_mac")
    pmos_pdk: str
    nfin: int = NFIN
    temperature: float = 300.15


TSMC5_SVT = TestTechConfig(
    name="TSMC5", tech_key="tsmc5", variant="svt", vdd=0.65,
    l_nmos=16e-9, l_pmos=20e-9,
    nmos_pdk="nch_svt_mac", pmos_pdk="pch_svt_mac",
)


# ── PyCMG ground-truth helpers ──────────────────────────────────────────────

def _resolve_tsmc_modelcard(
    tech: TestTechConfig, pdk_device: str, l_m: float,
) -> Path:
    """Regenerate a TSMC naive modelcard on-the-fly via PyCMG."""
    tech_config = TECH_REGISTRY[tech.name]
    prefix, rest = pdk_device.split("_", 1)
    vt = rest.replace("_mac", "")
    canonical = ("nmos_" if prefix == "nch" else "pmos_") + vt
    device_config = tech_config.get_device(canonical)
    return Path(resolve_modelcard(
        device_config, tech_config, L=l_m, NFIN=float(tech.nfin),
    ))


def create_pycmg_instance(
    tech: TestTechConfig, device_type: str,
) -> Instance:
    """Create PyCMG Instance for ground-truth evaluation."""
    pdk_device = tech.nmos_pdk if device_type == "nmos" else tech.pmos_pdk
    L = tech.l_nmos if device_type == "nmos" else tech.l_pmos

    modelcard_path = _resolve_tsmc_modelcard(tech, pdk_device, L)
    # Derive model_name from PDK device name
    prefix, rest = pdk_device.split("_", 1)
    vt = rest.replace("_mac", "")
    model_name = ("nmos_" if prefix == "nch" else "pmos_") + vt

    pycmg_model = Model(
        osdi_path=str(OSDI_PATH),
        modelcard_path=str(modelcard_path),
        model_name=model_name,
        model_card_name=model_name,
    )
    return Instance(
        model=pycmg_model,
        params={"L": L, "NFIN": float(tech.nfin)},
        temperature=tech.temperature,
    )


# ── BSIMAR v4 helpers ──────────────────────────────────────────────────────

def create_bsimar_instance(
    tech: TestTechConfig, device_type: str,
) -> Any:
    """Create BSIMAR v4 MOSFET instance for inference."""
    from pycircuitsim.models.mosfet_bsimar import NMOS_BSIMAR, PMOS_BSIMAR

    # Resolve v4 checkpoint (phys-best preferred)
    v4_phys = CHECKPOINT_DIR / f"v4_universal_{device_type}_best.phys.pt"
    v4_plain = CHECKPOINT_DIR / f"v4_universal_{device_type}_best.pt"
    if v4_phys.exists():
        model_path = str(v4_phys)
    elif v4_plain.exists():
        model_path = str(v4_plain)
    else:
        raise FileNotFoundError(
            f"No v4 checkpoint found for {device_type}: "
            f"tried {v4_phys.name}, {v4_plain.name}")

    L = tech.l_nmos if device_type == "nmos" else tech.l_pmos
    tech_code = tech_variant_to_code(tech.tech_key, tech.variant)

    nodes = ["drain", "gate", "source", "bulk"]
    cls = NMOS_BSIMAR if device_type == "nmos" else PMOS_BSIMAR
    return cls(
        name=f"m_{device_type}", nodes=nodes, model_path=model_path,
        L=L, NFIN=float(tech.nfin), tech_code=tech_code,
    )


# ── Test 1 & 2: Single-device DC sweep ─────────────────────────────────────

def test_dc_sweep(
    tech: TestTechConfig,
    device_type: str,
    n_points: int = 71,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compare BSIMAR v4 vs PyCMG for a single-device DC sweep.

    Returns: (nrmse_pct, vgs_arr, id_cmg_arr, id_bsimar_arr)
    """
    vdd = tech.vdd
    cmg = create_pycmg_instance(tech, device_type)
    bsimar = create_bsimar_instance(tech, device_type)

    if device_type == "nmos":
        vgs_sweep = np.linspace(0, vdd, n_points)
        vds = vdd / 2
    else:
        vgs_sweep = np.linspace(0, -vdd, n_points)
        vds = -vdd / 2

    vgs_ok: List[float] = []
    id_cmg: List[float] = []
    id_bsimar: List[float] = []

    for vgs in vgs_sweep:
        try:
            result = cmg.eval_dc({"d": vds, "g": vgs, "s": 0.0, "e": 0.0})
            i_cmg = result["id"]
        except Exception:
            continue

        voltages = {"drain": vds, "gate": vgs, "source": 0.0, "bulk": 0.0}
        bsimar.clear_cache()
        if device_type == "nmos":
            i_bsimar = -bsimar.calculate_current(voltages)
        else:
            i_bsimar = bsimar.calculate_current(voltages)

        vgs_ok.append(vgs)
        id_cmg.append(i_cmg)
        id_bsimar.append(i_bsimar)

    vgs_arr = np.array(vgs_ok)
    cmg_arr = np.array(id_cmg)
    bsimar_arr = np.array(id_bsimar)

    return nrmse(bsimar_arr, cmg_arr), vgs_arr, cmg_arr, bsimar_arr


# ── Test 3: Inverter VTC ───────────────────────────────────────────────────

def _solve_inverter(
    nmos: Any, pmos: Any, vin: float, vdd: float,
    is_pycmg: bool, max_iter: int = 100, tol: float = 1e-9,
) -> Optional[float]:
    """Solve inverter Vout via Newton-Raphson."""
    vout = vdd / 2
    for _ in range(max_iter):
        if is_pycmg:
            try:
                rn = nmos.eval_dc({"d": vout, "g": vin, "s": 0.0, "e": 0.0})
                rp = pmos.eval_dc({"d": vout, "g": vin, "s": vdd, "e": vdd})
            except Exception:
                return None
            f = rn["id"] + rp["id"]
            J = abs(rn["gds"]) + abs(rp["gds"])
        else:
            nv = {"drain": vout, "gate": vin, "source": 0.0, "bulk": 0.0}
            pv = {"drain": vout, "gate": vin, "source": vdd, "bulk": vdd}
            nmos.clear_cache()
            pmos.clear_cache()
            gds_n, _, _ = nmos.get_conductance(nv)
            i_n = nmos.calculate_current(nv)
            gds_p, _, _ = pmos.get_conductance(pv)
            i_p = pmos.calculate_current(pv)
            f = i_n - i_p
            J = gds_n + gds_p

        if J < 1e-15:
            J = 1e-9
        dv = -f / J if not is_pycmg else f / J
        if abs(dv) > 0.1:
            dv = 0.1 * np.sign(dv)
        vout += dv
        vout = max(-0.1, min(vdd + 0.1, vout))
        if abs(f) < tol:
            return vout
    return vout


def test_inverter_vtc(
    tech: TestTechConfig, n_points: int = 71,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compare BSIMAR v4 vs PyCMG inverter VTC.

    Returns: (nrmse_pct, vin_arr, vout_cmg_arr, vout_bsimar_arr)
    """
    vdd = tech.vdd
    vin_sweep = np.linspace(0, vdd, n_points)

    nmos_cmg = create_pycmg_instance(tech, "nmos")
    pmos_cmg = create_pycmg_instance(tech, "pmos")
    nmos_bs = create_bsimar_instance(tech, "nmos")
    pmos_bs = create_bsimar_instance(tech, "pmos")

    vin_ok: List[float] = []
    vout_cmg: List[float] = []
    vout_bs: List[float] = []

    for vin in vin_sweep:
        vc = _solve_inverter(nmos_cmg, pmos_cmg, vin, vdd, is_pycmg=True)
        vb = _solve_inverter(nmos_bs, pmos_bs, vin, vdd, is_pycmg=False)
        if vc is None or vb is None:
            continue
        vin_ok.append(vin)
        vout_cmg.append(vc)
        vout_bs.append(vb)

    vin_arr = np.array(vin_ok)
    cmg_arr = np.array(vout_cmg)
    bs_arr = np.array(vout_bs)

    return nrmse(bs_arr, cmg_arr), vin_arr, cmg_arr, bs_arr


# ── Test 4: Inverter transient ─────────────────────────────────────────────

def _create_baked_modelcard(tech: TestTechConfig) -> Path:
    """Create baked modelcard with instance params for NGSPICE OSDI."""
    nmos_mc = _resolve_tsmc_modelcard(tech, tech.nmos_pdk, tech.l_nmos)
    pmos_mc = _resolve_tsmc_modelcard(tech, tech.pmos_pdk, tech.l_pmos)

    results_dir = RESULTS_DIR / tech.name
    results_dir.mkdir(parents=True, exist_ok=True)

    merged = results_dir / "merged_modelcard.lib"
    merged.write_text(nmos_mc.read_text() + "\n" + pmos_mc.read_text())

    prefix_n, rest_n = tech.nmos_pdk.split("_", 1)
    nmos_model_name = ("nmos_" if prefix_n == "nch" else "pmos_") + rest_n.replace("_mac", "")
    prefix_p, rest_p = tech.pmos_pdk.split("_", 1)
    pmos_model_name = ("nmos_" if prefix_p == "nch" else "pmos_") + rest_p.replace("_mac", "")

    baked = results_dir / "baked_modelcard.lib"
    bake_inst_params(merged, baked, nmos_model_name,
                     {"L": tech.l_nmos, "NFIN": float(tech.nfin), "DEVTYPE": 1})
    bake_inst_params(baked, baked, pmos_model_name,
                     {"L": tech.l_pmos, "NFIN": float(tech.nfin), "DEVTYPE": 0})
    return baked


def run_ngspice_tran(tech: TestTechConfig) -> Dict[str, np.ndarray]:
    """Run NGSPICE transient (ground truth)."""
    results_dir = RESULTS_DIR / tech.name
    results_dir.mkdir(parents=True, exist_ok=True)
    baked = _create_baked_modelcard(tech)

    prefix_n, rest_n = tech.nmos_pdk.split("_", 1)
    nmos_model = ("nmos_" if prefix_n == "nch" else "pmos_") + rest_n.replace("_mac", "")
    prefix_p, rest_p = tech.pmos_pdk.split("_", 1)
    pmos_model = ("nmos_" if prefix_p == "nch" else "pmos_") + rest_p.replace("_mac", "")

    per = TR + PW + TF + max(PW, 1e-9)
    netlist = results_dir / "ngspice_tran.cir"
    netlist.write_text(f"""\
* BSIM-CMG Inverter Transient - NGSPICE ({tech.name})
.include "{baked}"
.temp 27
Vdd vdd 0 {tech.vdd}
Vin in 0 PULSE(0 {tech.vdd} {TD} {TR} {TF} {PW} {per})
Np out in vdd vdd {pmos_model}
Nn out in 0 0 {nmos_model}
Cload out 0 {CLOAD}
.ic V(out)={tech.vdd}
.tran {TSTEP} {TSTOP} uic
.end
""")

    csv_path = results_dir / "ngspice_tran.csv"
    log_path = results_dir / "ngspice_tran.log"
    runner = results_dir / "ngspice_tran_runner.cir"
    runner.write_text(f"""\
.control
osdi {OSDI_PATH}
source {netlist}
set filetype=ascii
set wr_vecnames
run
wrdata {csv_path} v(out) v(in)
.endc
.end
""")

    print(f"  [NGSPICE] Running {tech.name} transient...")
    res = subprocess.run(
        [NGSPICE_BIN, "-b", "-o", str(log_path), str(runner)],
        capture_output=True, text=True,
    )
    if not csv_path.exists():
        log_text = log_path.read_text() if log_path.exists() else "(no log)"
        raise RuntimeError(f"NGSPICE no output: {csv_path}\nRC={res.returncode}\n{log_text[-500:]}")

    lines = csv_path.read_text().strip().split("\n")
    data = np.array([[float(x) for x in line.split()] for line in lines[1:] if line.strip()])
    print(f"  [NGSPICE] {len(data)} pts, V(out) [{data[:,1].min():.4f}, {data[:,1].max():.4f}]V")
    return {"time": data[:, 0], "v(out)": data[:, 1], "v(in)": data[:, 3]}


def run_bsimar_tran(tech: TestTechConfig) -> Dict[str, np.ndarray]:
    """Run PyCircuitSim transient with LEVEL=74 (BSIMAR v4)."""
    import logging
    from pycircuitsim.parser import Parser
    from pycircuitsim.solver import DCSolver, TransientSolver

    results_dir = RESULTS_DIR / tech.name
    results_dir.mkdir(parents=True, exist_ok=True)

    l_nmos_nm = tech.l_nmos * 1e9
    l_pmos_nm = tech.l_pmos * 1e9
    per = TR + PW + TF + max(PW, 1e-9)

    netlist_path = results_dir / "bsimar_v4_tran.sp"
    netlist_path.write_text(f"""\
* BSIMAR v4 Inverter Transient ({tech.name}, LEVEL=74)
Vdd 1 0 {tech.vdd}
Vin 2 0 PULSE 0 {tech.vdd} {TD} {TR} {TF} {PW} {per}
Mp1 3 2 1 1 pmos1 L={l_pmos_nm:.0f}n NFIN={tech.nfin}
Mn1 3 2 0 0 nmos1 L={l_nmos_nm:.0f}n NFIN={tech.nfin}
Cload 3 0 {CLOAD}
.ic V(3)={tech.vdd}
.model nmos1 NMOS (LEVEL=74 TECH={tech.tech_key} VT={tech.variant})
.model pmos1 PMOS (LEVEL=74 TECH={tech.tech_key} VT={tech.variant})
.tran {TSTEP} {TSTOP}
.end
""")

    print(f"  [BSIMAR] Running {tech.name} transient...")
    logging.disable(logging.CRITICAL)
    try:
        parser = Parser()
        parser.parse_file(str(netlist_path))
        circuit = parser.circuit

        op_solver = DCSolver(
            circuit,
            initial_guess=circuit.initial_conditions or None,
            use_source_stepping=True,
        )
        op_solution = op_solver.solve()

        solver = TransientSolver(
            circuit,
            t_stop=parser.analysis_params["tstop"],
            dt=parser.analysis_params["tstep"],
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

    r = {"time": results["time"], "v(out)": results["3"], "v(in)": results["2"]}
    print(f"  [BSIMAR] {len(r['time'])} pts, V(out) [{r['v(out)'].min():.4f}, {r['v(out)'].max():.4f}]V")
    return r


def test_transient(
    tech: TestTechConfig,
) -> Tuple[float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compare BSIMAR v4 vs NGSPICE transient.

    Returns: (nrmse_pct, ng_data, bs_data)
    """
    ng = run_ngspice_tran(tech)
    bs = run_bsimar_tran(tech)

    # Interpolate to common grid (post-settling)
    t_max = min(ng["time"][-1], bs["time"][-1])
    t_common = np.arange(STARTUP_EXCLUSION, t_max, TSTEP)
    ng_vout = np.interp(t_common, ng["time"], ng["v(out)"])
    bs_vout = np.interp(t_common, bs["time"], bs["v(out)"])

    err = nrmse(bs_vout, ng_vout)
    return err, ng, bs


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_dc(
    tech: TestTechConfig, device_type: str,
    vgs: np.ndarray, id_cmg: np.ndarray, id_bs: np.ndarray,
    nrmse_pct: float, save_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(vgs * 1e3, id_cmg * 1e6, "b-", lw=1.5, label="PyCMG (BSIM-CMG)")
    ax1.plot(vgs * 1e3, id_bs * 1e6, "r--", lw=1.2, label="BSIMAR v4")
    ax1.set_ylabel("Id [uA]")
    ax1.set_title(f"{tech.name} {device_type.upper()} DC Sweep | NRMSE={nrmse_pct:.2f}%")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    err = (id_bs - id_cmg) * 1e6
    ax2.plot(vgs * 1e3, err, "g-", lw=0.8)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("Vgs [mV]")
    ax2.set_ylabel("Error [uA]")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_vtc(
    tech: TestTechConfig,
    vin: np.ndarray, vout_cmg: np.ndarray, vout_bs: np.ndarray,
    nrmse_pct: float, save_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(vin, vout_cmg, "b-", lw=1.5, label="PyCMG (BSIM-CMG)")
    ax1.plot(vin, vout_bs, "r--", lw=1.2, label="BSIMAR v4")
    ax1.set_ylabel("Vout [V]")
    ax1.set_title(f"{tech.name} Inverter VTC | NRMSE={nrmse_pct:.2f}%")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    err_mv = (vout_bs - vout_cmg) * 1e3
    ax2.plot(vin, err_mv, "g-", lw=0.8)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("Vin [V]")
    ax2.set_ylabel("Error [mV]")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_transient(
    tech: TestTechConfig,
    ng: Dict[str, np.ndarray], bs: Dict[str, np.ndarray],
    nrmse_pct: float, save_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [0.8, 1.2, 0.8]})

    ng_t = ng["time"] * 1e9
    bs_t = bs["time"] * 1e9

    axes[0].plot(ng_t, ng["v(in)"], "b-", lw=1.5, label="NGSPICE")
    axes[0].plot(bs_t, bs["v(in)"], "r--", lw=1.2, alpha=0.8, label="BSIMAR v4")
    axes[0].set_ylabel("V(in) [V]")
    axes[0].set_title(f"BSIMAR v4 Inverter Transient: {tech.name} | NRMSE={nrmse_pct:.2f}%")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ng_t, ng["v(out)"], "b-", lw=1.5, label="NGSPICE (BSIM-CMG)")
    axes[1].plot(bs_t, bs["v(out)"], "r--", lw=1.2, alpha=0.8, label="BSIMAR v4")
    axes[1].set_ylabel("V(out) [V]")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    t_max = min(ng["time"][-1], bs["time"][-1])
    t_common = np.arange(STARTUP_EXCLUSION, t_max, TSTEP)
    ng_v = np.interp(t_common, ng["time"], ng["v(out)"])
    bs_v = np.interp(t_common, bs["time"], bs["v(out)"])
    err_mv = (bs_v - ng_v) * 1e3
    axes[2].plot(t_common * 1e9, err_mv, "g-", lw=0.8)
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_xlabel("Time [ns]")
    axes[2].set_ylabel("Error [mV]")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    nrmse_pct: float
    threshold: float
    elapsed_s: float

    @property
    def passed(self) -> bool:
        return self.nrmse_pct < self.threshold


def main() -> int:
    tech = TSMC5_SVT
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tech_dir = RESULTS_DIR / tech.name
    tech_dir.mkdir(parents=True, exist_ok=True)

    # Check v4 checkpoints
    for dt in ("nmos", "pmos"):
        v4_phys = CHECKPOINT_DIR / f"v4_universal_{dt}_best.phys.pt"
        v4_plain = CHECKPOINT_DIR / f"v4_universal_{dt}_best.pt"
        if not v4_phys.exists() and not v4_plain.exists():
            print(f"ERROR: No v4 checkpoint for {dt}: {v4_phys.name} / {v4_plain.name}")
            return 1

    print("=" * 72)
    print(f"BSIMAR v4 Inverter Verification: {tech.name} {tech.variant.upper()}")
    print(f"  VDD={tech.vdd}V  L_n={tech.l_nmos*1e9:.0f}nm  L_p={tech.l_pmos*1e9:.0f}nm  NFIN={tech.nfin}")
    tech_code_n = tech_variant_to_code(tech.tech_key, tech.variant)
    print(f"  Tech code: {tech_code_n}")
    print("=" * 72)

    results: List[TestResult] = []

    # Test 1: NMOS DC
    print(f"\n--- Test 1: NMOS DC Sweep ---")
    t0 = time.time()
    nrmse_n, vgs_n, id_cmg_n, id_bs_n = test_dc_sweep(tech, "nmos")
    dt = time.time() - t0
    status = "PASS" if nrmse_n < DC_NRMSE_THRESHOLD else "FAIL"
    print(f"  NRMSE={nrmse_n:.2f}%  [{dt:.1f}s]  {status}")
    results.append(TestResult("NMOS DC", nrmse_n, DC_NRMSE_THRESHOLD, dt))
    plot_dc(tech, "nmos", vgs_n, id_cmg_n, id_bs_n, nrmse_n, tech_dir / "nmos_dc.png")

    # Test 2: PMOS DC
    print(f"\n--- Test 2: PMOS DC Sweep ---")
    t0 = time.time()
    nrmse_p, vgs_p, id_cmg_p, id_bs_p = test_dc_sweep(tech, "pmos")
    dt = time.time() - t0
    status = "PASS" if nrmse_p < DC_NRMSE_THRESHOLD else "FAIL"
    print(f"  NRMSE={nrmse_p:.2f}%  [{dt:.1f}s]  {status}")
    results.append(TestResult("PMOS DC", nrmse_p, DC_NRMSE_THRESHOLD, dt))
    plot_dc(tech, "pmos", vgs_p, id_cmg_p, id_bs_p, nrmse_p, tech_dir / "pmos_dc.png")

    # Test 3: Inverter VTC
    print(f"\n--- Test 3: Inverter VTC ---")
    t0 = time.time()
    nrmse_vtc, vin_vtc, vout_cmg, vout_bs = test_inverter_vtc(tech)
    dt = time.time() - t0
    status = "PASS" if nrmse_vtc < VTC_NRMSE_THRESHOLD else "FAIL"
    print(f"  NRMSE={nrmse_vtc:.2f}%  [{dt:.1f}s]  {status}")
    results.append(TestResult("Inverter VTC", nrmse_vtc, VTC_NRMSE_THRESHOLD, dt))
    plot_vtc(tech, vin_vtc, vout_cmg, vout_bs, nrmse_vtc, tech_dir / "inverter_vtc.png")

    # Test 4: Inverter transient
    print(f"\n--- Test 4: Inverter Transient ---")
    t0 = time.time()
    nrmse_tran, ng_data, bs_data = test_transient(tech)
    dt = time.time() - t0
    status = "PASS" if nrmse_tran < TRAN_NRMSE_THRESHOLD else "FAIL"
    print(f"  NRMSE={nrmse_tran:.2f}%  [{dt:.1f}s]  {status}")
    results.append(TestResult("Transient", nrmse_tran, TRAN_NRMSE_THRESHOLD, dt))
    plot_transient(tech, ng_data, bs_data, nrmse_tran, tech_dir / "transient.png")

    # Summary
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Test':<20s} | {'NRMSE(%)':>10s} | {'Threshold':>10s} | {'Time(s)':>8s} | {'Status':>6s}")
    print(f"  {'-'*20} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*6}")
    n_pass = 0
    for r in results:
        st = "PASS" if r.passed else "FAIL"
        if r.passed:
            n_pass += 1
        print(f"  {r.name:<20s} | {r.nrmse_pct:>10.2f} | {r.threshold:>10.1f} | {r.elapsed_s:>8.1f} | {st:>6s}")

    print(f"\n  Result: {n_pass}/{len(results)} PASS")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
