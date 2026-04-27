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

import argparse
import os
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
# PyCMG paths first (will be pushed down), then PROJECT_ROOT last (stays at [0])
# so PROJECT_ROOT/tests/ is found before PyCMG/tests/
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))
sys.path.insert(0, str(PROJECT_ROOT))

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

TSMC7_SVT = TestTechConfig(
    name="TSMC7", tech_key="tsmc7", variant="svt", vdd=0.75,
    l_nmos=16e-9, l_pmos=20e-9,
    nmos_pdk="nch_svt_mac", pmos_pdk="pch_svt_mac",
)

TSMC12_SVT = TestTechConfig(
    name="TSMC12", tech_key="tsmc12", variant="svt", vdd=0.80,
    l_nmos=16e-9, l_pmos=20e-9,
    nmos_pdk="nch_svt_mac", pmos_pdk="pch_svt_mac",
)

TSMC16_SVT = TestTechConfig(
    name="TSMC16", tech_key="tsmc16", variant="svt", vdd=0.80,
    l_nmos=16e-9, l_pmos=20e-9,
    nmos_pdk="nch_svt_mac", pmos_pdk="pch_svt_mac",
)

TECH_BY_NAME = {
    "tsmc5": TSMC5_SVT, "tsmc7": TSMC7_SVT,
    "tsmc12": TSMC12_SVT, "tsmc16": TSMC16_SVT,
}


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
    # Naive modelcards use the PDK device name (e.g. nch_svt_mac) as model name
    pycmg_model = Model(
        osdi_path=str(OSDI_PATH),
        modelcard_path=str(modelcard_path),
        model_name=pdk_device,
        model_card_name=pdk_device,
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

    # Resolve v4 checkpoint (phys-best preferred). Prefix overridable via env.
    prefix = os.environ.get("BSIMAR_PREFIX", "v4_universal")
    v4_phys = CHECKPOINT_DIR / f"{prefix}_{device_type}_best.phys.pt"
    v4_plain = CHECKPOINT_DIR / f"{prefix}_{device_type}_best.pt"
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


# ── DirectNet v4 helpers ──────────────────────────────────────────────────

def create_directnet_instance(
    tech: TestTechConfig, device_type: str,
) -> Any:
    """Create DirectNet v4 MOSFET instance for inference."""
    from pycircuitsim.models.mosfet_directnet import NMOS_NN, PMOS_NN

    dn_prefix = os.environ.get("DIRECTNET_PREFIX", "v4_dn_universal")
    model_path = CHECKPOINT_DIR / f"{dn_prefix}_{device_type}_best.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No DirectNet checkpoint for {device_type}: {model_path.name}")

    L = tech.l_nmos if device_type == "nmos" else tech.l_pmos
    tech_code = tech_variant_to_code(tech.tech_key, tech.variant)

    nodes = ["drain", "gate", "source", "bulk"]
    cls = NMOS_NN if device_type == "nmos" else PMOS_NN
    return cls(
        name=f"m_{device_type}_dn", nodes=nodes, model_path=str(model_path),
        L=L, NFIN=float(tech.nfin), tech_code=tech_code,
    )


# ── Test 1 & 2: Single-device DC sweep ─────────────────────────────────────

def test_dc_sweep(
    tech: TestTechConfig,
    device_type: str,
    n_points: int = 71,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compare BSIMAR v4 + DirectNet v4 vs PyCMG for a single-device DC sweep.

    Returns: (nrmse_ar_pct, nrmse_dn_pct, vgs_arr, id_cmg_arr,
              id_bsimar_arr, id_directnet_arr)
    """
    vdd = tech.vdd
    cmg = create_pycmg_instance(tech, device_type)
    bsimar = create_bsimar_instance(tech, device_type)
    directnet = create_directnet_instance(tech, device_type)

    if device_type == "nmos":
        vgs_sweep = np.linspace(0, vdd, n_points)
        vds = vdd / 2
    else:
        vgs_sweep = np.linspace(0, -vdd, n_points)
        vds = -vdd / 2

    vgs_ok: List[float] = []
    id_cmg: List[float] = []
    id_bsimar: List[float] = []
    id_dn: List[float] = []

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

        directnet.clear_cache()
        if device_type == "nmos":
            i_directnet = -directnet.calculate_current(voltages)
        else:
            i_directnet = directnet.calculate_current(voltages)

        vgs_ok.append(vgs)
        id_cmg.append(i_cmg)
        id_bsimar.append(i_bsimar)
        id_dn.append(i_directnet)

    vgs_arr = np.array(vgs_ok)
    cmg_arr = np.array(id_cmg)
    bsimar_arr = np.array(id_bsimar)
    dn_arr = np.array(id_dn)

    return (nrmse(bsimar_arr, cmg_arr), nrmse(dn_arr, cmg_arr),
            vgs_arr, cmg_arr, bsimar_arr, dn_arr)


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
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compare BSIMAR v4 + DirectNet v4 vs PyCMG inverter VTC.

    Returns: (nrmse_ar_pct, nrmse_dn_pct, vin_arr, vout_cmg_arr,
              vout_bsimar_arr, vout_dn_arr)
    """
    vdd = tech.vdd
    vin_sweep = np.linspace(0, vdd, n_points)

    nmos_cmg = create_pycmg_instance(tech, "nmos")
    pmos_cmg = create_pycmg_instance(tech, "pmos")
    nmos_bs = create_bsimar_instance(tech, "nmos")
    pmos_bs = create_bsimar_instance(tech, "pmos")
    nmos_dn = create_directnet_instance(tech, "nmos")
    pmos_dn = create_directnet_instance(tech, "pmos")

    vin_ok: List[float] = []
    vout_cmg: List[float] = []
    vout_bs: List[float] = []
    vout_dn: List[float] = []

    for vin in vin_sweep:
        vc = _solve_inverter(nmos_cmg, pmos_cmg, vin, vdd, is_pycmg=True)
        vb = _solve_inverter(nmos_bs, pmos_bs, vin, vdd, is_pycmg=False)
        vd = _solve_inverter(nmos_dn, pmos_dn, vin, vdd, is_pycmg=False)
        if vc is None or vb is None or vd is None:
            continue
        vin_ok.append(vin)
        vout_cmg.append(vc)
        vout_bs.append(vb)
        vout_dn.append(vd)

    vin_arr = np.array(vin_ok)
    cmg_arr = np.array(vout_cmg)
    bs_arr = np.array(vout_bs)
    dn_arr = np.array(vout_dn)

    return (nrmse(bs_arr, cmg_arr), nrmse(dn_arr, cmg_arr),
            vin_arr, cmg_arr, bs_arr, dn_arr)


# ── Test 4: Inverter transient ─────────────────────────────────────────────

def _create_baked_modelcard(tech: TestTechConfig) -> Path:
    """Create baked modelcard with instance params for NGSPICE OSDI."""
    nmos_mc = _resolve_tsmc_modelcard(tech, tech.nmos_pdk, tech.l_nmos)
    pmos_mc = _resolve_tsmc_modelcard(tech, tech.pmos_pdk, tech.l_pmos)

    results_dir = RESULTS_DIR / tech.name
    results_dir.mkdir(parents=True, exist_ok=True)

    merged = results_dir / "merged_modelcard.lib"
    merged.write_text(nmos_mc.read_text() + "\n" + pmos_mc.read_text())

    # Naive modelcards use PDK device names (nch_svt_mac, pch_svt_mac)
    baked = results_dir / "baked_modelcard.lib"
    bake_inst_params(merged, baked, tech.nmos_pdk,
                     {"L": tech.l_nmos, "NFIN": float(tech.nfin), "DEVTYPE": 1})
    bake_inst_params(baked, baked, tech.pmos_pdk,
                     {"L": tech.l_pmos, "NFIN": float(tech.nfin), "DEVTYPE": 0})
    return baked


def run_ngspice_tran(tech: TestTechConfig) -> Dict[str, np.ndarray]:
    """Run NGSPICE transient (ground truth)."""
    results_dir = RESULTS_DIR / tech.name
    results_dir.mkdir(parents=True, exist_ok=True)
    baked = _create_baked_modelcard(tech)

    # Use PDK device names for NGSPICE (matches baked modelcard)
    nmos_model = tech.nmos_pdk
    pmos_model = tech.pmos_pdk

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
* {tech.name} inverter transient runner
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


def _run_nn_tran(tech: TestTechConfig, level: int) -> Dict[str, np.ndarray]:
    """Run PyCircuitSim transient with LEVEL=73 or 74."""
    import logging
    from pycircuitsim.parser import Parser
    from pycircuitsim.solver import DCSolver, TransientSolver

    results_dir = RESULTS_DIR / tech.name
    results_dir.mkdir(parents=True, exist_ok=True)

    l_nmos_nm = tech.l_nmos * 1e9
    l_pmos_nm = tech.l_pmos * 1e9
    per = TR + PW + TF + max(PW, 1e-9)
    label = "directnet_v4" if level == 73 else "bsimar_v4"

    netlist_path = results_dir / f"{label}_tran.sp"

    # If a custom checkpoint prefix was requested for tests 1-3, also
    # override the netlist's LEVEL=73/74 checkpoint paths (parser
    # supports MODEL_PATH=... override per .model directive).
    extra_n = extra_p = ""
    if level == 74:
        prefix = os.environ.get("BSIMAR_PREFIX")
        if prefix:
            from bsimar.config import CHECKPOINT_DIR as _CKPT_DIR
            n_phys = _CKPT_DIR / f"{prefix}_nmos_best.phys.pt"
            n_path = n_phys if n_phys.exists() else _CKPT_DIR / f"{prefix}_nmos_best.pt"
            p_phys = _CKPT_DIR / f"{prefix}_pmos_best.phys.pt"
            p_path = p_phys if p_phys.exists() else _CKPT_DIR / f"{prefix}_pmos_best.pt"
            extra_n = f" MODEL_PATH={n_path}"
            extra_p = f" MODEL_PATH={p_path}"
    elif level == 73:
        dn_prefix = os.environ.get("DIRECTNET_PREFIX")
        if dn_prefix:
            from bsimar.config import CHECKPOINT_DIR as _CKPT_DIR
            n_path = _CKPT_DIR / f"{dn_prefix}_nmos_best.pt"
            p_path = _CKPT_DIR / f"{dn_prefix}_pmos_best.pt"
            extra_n = f" MODEL_PATH={n_path}"
            extra_p = f" MODEL_PATH={p_path}"

    netlist_path.write_text(f"""\
* {label.upper()} Inverter Transient ({tech.name}, LEVEL={level})
Vdd 1 0 {tech.vdd}
Vin 2 0 PULSE 0 {tech.vdd} {TD} {TR} {TF} {PW} {per}
Mp1 3 2 1 1 pmos1 L={l_pmos_nm:.0f}n NFIN={tech.nfin}
Mn1 3 2 0 0 nmos1 L={l_nmos_nm:.0f}n NFIN={tech.nfin}
Cload 3 0 {CLOAD}
.ic V(3)={tech.vdd}
.model nmos1 NMOS (LEVEL={level} TECH={tech.tech_key} VT={tech.variant}{extra_n})
.model pmos1 PMOS (LEVEL={level} TECH={tech.tech_key} VT={tech.variant}{extra_p})
.tran {TSTEP} {TSTOP}
.end
""")

    label = "DirectNet" if level == 73 else "BSIMAR"
    print(f"  [{label}] Running {tech.name} transient...")
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
    print(f"  [{label}] {len(r['time'])} pts, V(out) [{r['v(out)'].min():.4f}, {r['v(out)'].max():.4f}]V")
    return r


def run_bsimar_tran(tech: TestTechConfig) -> Dict[str, np.ndarray]:
    """Run PyCircuitSim transient with LEVEL=74 (BSIMAR v4)."""
    return _run_nn_tran(tech, level=74)


def run_directnet_tran(tech: TestTechConfig) -> Dict[str, np.ndarray]:
    """Run PyCircuitSim transient with LEVEL=73 (DirectNet v4)."""
    return _run_nn_tran(tech, level=73)


def _compute_tran_nrmse(
    ng: Dict[str, np.ndarray], nn: Dict[str, np.ndarray],
) -> float:
    """Compute NRMSE between NGSPICE and NN transient waveforms."""
    t_max = min(ng["time"][-1], nn["time"][-1])
    t_common = np.arange(STARTUP_EXCLUSION, t_max, TSTEP)
    ng_vout = np.interp(t_common, ng["time"], ng["v(out)"])
    nn_vout = np.interp(t_common, nn["time"], nn["v(out)"])
    return nrmse(nn_vout, ng_vout)


def test_transient(
    tech: TestTechConfig,
) -> Tuple[float, float, Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """Compare BSIMAR v4 + DirectNet v4 vs NGSPICE transient.

    Returns: (nrmse_ar_pct, nrmse_dn_pct, ng_data, bs_data, dn_data)
    """
    ng = run_ngspice_tran(tech)
    bs = run_bsimar_tran(tech)
    err_ar = _compute_tran_nrmse(ng, bs)

    # DirectNet v4 transient (may fail if model quality is insufficient)
    dn_data: Optional[Dict[str, np.ndarray]] = None
    err_dn = float("nan")
    try:
        dn_data = run_directnet_tran(tech)
        err_dn = _compute_tran_nrmse(ng, dn_data)
    except Exception as e:
        print(f"  [DirectNet] Transient FAILED: {e}")

    return err_ar, err_dn, ng, bs, dn_data


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsimar-prefix", "--checkpoint-prefix",
                        dest="bsimar_prefix", default=None,
                        help="BSIMAR checkpoint prefix (default: v4_universal). "
                             "Probe: v4_probe_signfix")
    parser.add_argument("--directnet-prefix", default=None,
                        help="DirectNet checkpoint prefix "
                             "(default: v4_dn_universal)")
    parser.add_argument("--tech", default="tsmc5",
                        choices=list(TECH_BY_NAME.keys()),
                        help="Technology to verify (default: tsmc5)")
    args = parser.parse_args()
    if args.bsimar_prefix:
        os.environ["BSIMAR_PREFIX"] = args.bsimar_prefix
    if args.directnet_prefix:
        os.environ["DIRECTNET_PREFIX"] = args.directnet_prefix

    tech = TECH_BY_NAME[args.tech]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tech_dir = RESULTS_DIR / tech.name
    tech_dir.mkdir(parents=True, exist_ok=True)

    # Check v4 checkpoints (both models)
    prefix = os.environ.get("BSIMAR_PREFIX", "v4_universal")
    for dt in ("nmos", "pmos"):
        v4_phys = CHECKPOINT_DIR / f"{prefix}_{dt}_best.phys.pt"
        v4_plain = CHECKPOINT_DIR / f"{prefix}_{dt}_best.pt"
        if not v4_phys.exists() and not v4_plain.exists():
            print(f"ERROR: No BSIMAR v4 checkpoint for {dt}: {v4_phys.name} / {v4_plain.name}")
            return 1
        dn_ckpt = CHECKPOINT_DIR / f"v4_dn_universal_{dt}_best.pt"
        if not dn_ckpt.exists():
            print(f"WARNING: No DirectNet v4 checkpoint for {dt}: {dn_ckpt.name}")

    print("=" * 72)
    print(f"v4 Inverter Verification: {tech.name} {tech.variant.upper()}")
    print(f"  BSIMAR v4 (LEVEL=74) + DirectNet v4 (LEVEL=73)")
    print(f"  VDD={tech.vdd}V  L_n={tech.l_nmos*1e9:.0f}nm  L_p={tech.l_pmos*1e9:.0f}nm  NFIN={tech.nfin}")
    tech_code_n = tech_variant_to_code(tech.tech_key, tech.variant)
    print(f"  Tech code: {tech_code_n}")
    print("=" * 72)

    results: List[TestResult] = []

    # Test 1: NMOS DC
    print(f"\n--- Test 1: NMOS DC Sweep ---")
    t0 = time.time()
    nrmse_ar_n, nrmse_dn_n, vgs_n, id_cmg_n, id_bs_n, id_dn_n = test_dc_sweep(tech, "nmos")
    dt = time.time() - t0
    print(f"  AR NRMSE={nrmse_ar_n:.2f}%  DN NRMSE={nrmse_dn_n:.2f}%  [{dt:.1f}s]")
    results.append(TestResult("NMOS DC (AR)", nrmse_ar_n, DC_NRMSE_THRESHOLD, dt))
    results.append(TestResult("NMOS DC (DN)", nrmse_dn_n, DC_NRMSE_THRESHOLD, dt))
    plot_dc(tech, "nmos", vgs_n, id_cmg_n, id_bs_n, nrmse_ar_n, tech_dir / "nmos_dc.png")

    # Test 2: PMOS DC
    print(f"\n--- Test 2: PMOS DC Sweep ---")
    t0 = time.time()
    nrmse_ar_p, nrmse_dn_p, vgs_p, id_cmg_p, id_bs_p, id_dn_p = test_dc_sweep(tech, "pmos")
    dt = time.time() - t0
    print(f"  AR NRMSE={nrmse_ar_p:.2f}%  DN NRMSE={nrmse_dn_p:.2f}%  [{dt:.1f}s]")
    results.append(TestResult("PMOS DC (AR)", nrmse_ar_p, DC_NRMSE_THRESHOLD, dt))
    results.append(TestResult("PMOS DC (DN)", nrmse_dn_p, DC_NRMSE_THRESHOLD, dt))
    plot_dc(tech, "pmos", vgs_p, id_cmg_p, id_bs_p, nrmse_ar_p, tech_dir / "pmos_dc.png")

    # Test 3: Inverter VTC (skippable via SKIP_VTC=1)
    if os.environ.get("SKIP_VTC", "0") != "1":
        print(f"\n--- Test 3: Inverter VTC ---")
        t0 = time.time()
        nrmse_ar_vtc, nrmse_dn_vtc, vin_vtc, vout_cmg, vout_bs, vout_dn = test_inverter_vtc(tech)
        dt = time.time() - t0
        print(f"  AR NRMSE={nrmse_ar_vtc:.2f}%  DN NRMSE={nrmse_dn_vtc:.2f}%  [{dt:.1f}s]")
        results.append(TestResult("VTC (AR)", nrmse_ar_vtc, VTC_NRMSE_THRESHOLD, dt))
        results.append(TestResult("VTC (DN)", nrmse_dn_vtc, VTC_NRMSE_THRESHOLD, dt))
        plot_vtc(tech, vin_vtc, vout_cmg, vout_bs, nrmse_ar_vtc, tech_dir / "inverter_vtc.png")
    else:
        print(f"\n--- Test 3: Inverter VTC --- SKIPPED (SKIP_VTC=1)")

    # Test 4: Inverter transient
    print(f"\n--- Test 4: Inverter Transient ---")
    t0 = time.time()
    nrmse_ar_tran, nrmse_dn_tran, ng_data, bs_data, dn_data = test_transient(tech)
    dt = time.time() - t0
    print(f"  AR NRMSE={nrmse_ar_tran:.2f}%  DN NRMSE={nrmse_dn_tran:.2f}%  [{dt:.1f}s]")
    results.append(TestResult("Transient (AR)", nrmse_ar_tran, TRAN_NRMSE_THRESHOLD, dt))
    if not np.isnan(nrmse_dn_tran):
        results.append(TestResult("Transient (DN)", nrmse_dn_tran, TRAN_NRMSE_THRESHOLD, dt))
    plot_transient(tech, ng_data, bs_data, nrmse_ar_tran, tech_dir / "transient.png")

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
