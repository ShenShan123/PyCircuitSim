#!/usr/bin/env python3
"""
Comprehensive NN compact model verification: BSIMAR v4 (LEVEL=74) and
DirectNet v4 (LEVEL=73) vs BSIM-CMG (LEVEL=72) ground truth.

.. note:: Run with ``PYTHONUNBUFFERED=1`` to see live progress output.

Tests NMOS DC sweep, PMOS DC sweep, inverter VTC, NMOS pulse-response
transient, and inverter transient across available technologies.
Compares against BSIM-CMG NGSPICE ground truth.

Available checkpoints (v4 tech-code embedding):
- BSIMAR v4 NMOS: v4_universal_nmos_best.phys.pt (+ _norm.npz, _config.npz)
- BSIMAR v4 PMOS: v4_universal_pmos_best.phys.pt (+ _norm.npz, _config.npz)
- DirectNet v4 NMOS: v4_dn_universal_nmos_best.pt (+ _norm.npz)
- DirectNet v4 PMOS: v4_dn_universal_pmos_best.pt (+ _norm.npz)

Strategy:
  1. BSIM-CMG (LEVEL=72) via NGSPICE as ground truth
  2. BSIM-CMG (LEVEL=72) via PyCircuitSim (sanity check)
  3. BSIMAR v4 (LEVEL=74, tech-code embedding) via PyCircuitSim
  4. DirectNet v4 (LEVEL=73, tech-code embedding) via PyCircuitSim

Metrics: NRMSE (%) and MRE (%).

Usage:
    conda run -n pycircuitsim python tests/verify_nn_dc_tran.py
    conda run -n pycircuitsim python tests/verify_nn_dc_tran.py --tech ASAP7
    conda run -n pycircuitsim python tests/verify_nn_dc_tran.py --tech ASAP7,TSMC12
    conda run -n pycircuitsim python tests/verify_nn_dc_tran.py --dc-only
    conda run -n pycircuitsim python tests/verify_nn_dc_tran.py --tran-only
    conda run -n pycircuitsim python tests/verify_nn_dc_tran.py --pmos-only
    conda run -n pycircuitsim python tests/verify_nn_dc_tran.py --inverter-only
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure unbuffered output for live progress
import os
if "PYTHONUNBUFFERED" not in os.environ:
    import functools
    print = functools.partial(print, flush=True)  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))

from helpers import bake_inst_params  # noqa: E402
from bsimar.config import (  # noqa: E402
    CHECKPOINT_DIR, TECH_CONFIGS,
    NNTechConfig, OSDI_PATH,
    tech_variant_to_code, UNKNOWN_CODE_ID,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NGSPICE_BIN = "/usr/local/ngspice-45.2/bin/ngspice"
MODELCARDS_DIR = (
    PROJECT_ROOT / "external_compact_models" / "PyCMG" / "modelcards"
)
RESULTS_BASE = PROJECT_ROOT / "tests" / "verify_nn_dc_tran_results"

# DC thresholds (loose for NN, tight for BSIM-CMG)
DC_NRMSE_THRESHOLD_NN = 0.10   # 10% for NN models
DC_NRMSE_THRESHOLD_CMG = 0.01  # 1% for BSIM-CMG self-consistency

# Transient thresholds
TRAN_NRMSE_THRESHOLD = 0.15    # 15% of Vdd

# Inverter VTC threshold
VTC_NRMSE_THRESHOLD = 0.15     # 15% for inverter VTC

# Transient parameters (NMOS-only pulse response)
TRAN_TSTEP = 10e-12   # 10ps
TRAN_TSTOP = 3e-9     # 3ns
TRAN_TR = 50e-12       # 50ps rise
TRAN_TF = 50e-12       # 50ps fall
TRAN_PW = 1e-9         # 1ns pulse width
TRAN_TD = 0.2e-9       # 200ps delay
TRAN_RLOAD = 5e3       # 5k ohm load resistor
TRAN_STARTUP_EXCL = 0.1e-9  # 0.1ns startup exclusion

# Inverter transient parameters
INV_CLOAD = 1e-15          # 1 fF load capacitance
INV_TRAN_TSTEP = 10e-12    # 10ps
INV_TRAN_TSTOP = 3e-9      # 3ns
INV_TRAN_TR = 50e-12       # 50ps rise
INV_TRAN_TF = 50e-12       # 50ps fall
INV_TRAN_PW = 1e-9         # 1ns pulse width
INV_TRAN_TD = 0.2e-9       # 200ps delay


def _tech_code_in_vocab(tech_key: str, vt_key: str, num_codes: int = 18) -> bool:
    """Check if (tech, vt) maps to a code inside the embedding vocabulary.

    v4 universal models are trained with --num-tech-codes 18.
    ASAP7 codes (18-21) are out-of-range.
    """
    code = tech_variant_to_code(tech_key, vt_key)
    return code < num_codes


# ---------------------------------------------------------------------------
# Technology configurations
# ---------------------------------------------------------------------------
@dataclass
class TestTechConfig:
    """Technology configuration for NMOS + PMOS testing."""
    name: str
    vdd: float
    l_nmos: float          # NMOS channel length [m]
    nfin: int
    tfin: float            # Fin thickness [m]
    nmos_model: str        # NGSPICE NMOS model name
    nn_tech_key: str       # TECH= parameter for NN netlists
    nn_vt: str             # VT= parameter for NN netlists
    single_file: bool      # ASAP7: all models in one file
    modelcard_dir: str     # Subdir under MODELCARDS_DIR
    modelcard_file: str    # For ASAP7: single file; for TSMC: resolved
    l_pmos: float = 0.0    # PMOS channel length [m] (0 = same as l_nmos)
    pmos_model: str = ""   # NGSPICE PMOS model name
    nn_pmos_vt: str = ""   # VT= for PMOS (empty = same as NMOS)

    @property
    def effective_l_pmos(self) -> float:
        """PMOS L, defaulting to l_nmos if not set."""
        return self.l_pmos if self.l_pmos > 0 else self.l_nmos

    @property
    def effective_pmos_vt(self) -> str:
        """PMOS VT, defaulting to NMOS VT if not set."""
        return self.nn_pmos_vt if self.nn_pmos_vt else self.nn_vt


ALL_TEST_TECHS: Dict[str, TestTechConfig] = {
    "ASAP7": TestTechConfig(
        name="ASAP7", vdd=0.7, l_nmos=7e-9, nfin=10, tfin=6.5e-9,
        nmos_model="nmos_rvt", nn_tech_key="asap7", nn_vt="rvt",
        single_file=True, modelcard_dir="ASAP7",
        modelcard_file="7nm_TT_160803.pm",
        l_pmos=7e-9, pmos_model="pmos_rvt",
    ),
    # ASAP7_30nm: alternate geometry for comparison (standard test L)
    "ASAP7_30nm": TestTechConfig(
        name="ASAP7_30nm", vdd=0.7, l_nmos=30e-9, nfin=10, tfin=6.5e-9,
        nmos_model="nmos_rvt", nn_tech_key="asap7", nn_vt="rvt",
        single_file=True, modelcard_dir="ASAP7",
        modelcard_file="7nm_TT_160803.pm",
        l_pmos=30e-9, pmos_model="pmos_rvt",
    ),
    "TSMC5": TestTechConfig(
        name="TSMC5", vdd=0.65, l_nmos=16e-9, nfin=2, tfin=6e-9,
        nmos_model="nch_lvt_mac", nn_tech_key="tsmc5", nn_vt="lvt",
        single_file=False, modelcard_dir="TSMC5",
        modelcard_file="",
        l_pmos=20e-9, pmos_model="pch_lvt_mac",
    ),
    "TSMC7": TestTechConfig(
        name="TSMC7", vdd=0.75, l_nmos=16e-9, nfin=2, tfin=6e-9,
        nmos_model="nch_ulvt_mac", nn_tech_key="tsmc7", nn_vt="ulvt",
        single_file=False, modelcard_dir="TSMC7",
        modelcard_file="",
        l_pmos=20e-9, pmos_model="pch_ulvt_mac",
    ),
    "TSMC12": TestTechConfig(
        name="TSMC12", vdd=0.80, l_nmos=16e-9, nfin=2, tfin=6e-9,
        nmos_model="nch_svt_mac", nn_tech_key="tsmc12", nn_vt="svt",
        single_file=False, modelcard_dir="TSMC12",
        modelcard_file="",
        l_pmos=20e-9, pmos_model="pch_svt_mac",
    ),
    "TSMC16": TestTechConfig(
        name="TSMC16", vdd=0.80, l_nmos=16e-9, nfin=2, tfin=6e-9,
        nmos_model="nch_svt_mac", nn_tech_key="tsmc16", nn_vt="svt",
        single_file=False, modelcard_dir="TSMC16",
        modelcard_file="",
        l_pmos=20e-9, pmos_model="pch_svt_mac",
    ),
}

TECH_ORDER: List[str] = ["ASAP7", "ASAP7_30nm", "TSMC5", "TSMC7", "TSMC12", "TSMC16"]

TECH_COLORS: Dict[str, str] = {
    "ASAP7": "tab:blue",
    "TSMC5": "tab:green",
    "TSMC7": "tab:orange",
    "TSMC12": "tab:purple",
    "TSMC16": "tab:red",
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def nrmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Normalized RMSE as percentage of peak-to-peak range."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    ptp = float(true.max() - true.min())
    if ptp < 1e-30:
        return 0.0
    rmse_val = float(np.sqrt(np.mean((pred - true) ** 2)))
    return rmse_val / ptp * 100.0


def mre(pred: np.ndarray, true: np.ndarray,
        threshold_rel: float = 0.01) -> float:
    """Mean relative error (percent), excluding near-zero samples."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    max_abs = float(np.abs(true).max())
    if max_abs == 0:
        return 0.0
    mask = np.abs(true) > max_abs * threshold_rel
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((true[mask] - pred[mask]) / true[mask]))) * 100.0


# ---------------------------------------------------------------------------
# Modelcard resolution
# ---------------------------------------------------------------------------
def _registry_name(tech: TestTechConfig) -> str:
    """Map test tech name to PyCMG TECH_REGISTRY key (strip suffixes)."""
    # e.g. "ASAP7_30nm" -> "ASAP7"
    base = tech.name.split("_")[0] if "_" in tech.name else tech.name
    return base


def resolve_nmos_modelcard(tech: TestTechConfig) -> Path:
    """Resolve the NMOS modelcard path."""
    if tech.single_file:
        return MODELCARDS_DIR / tech.modelcard_dir / tech.modelcard_file

    # TSMC: resolve via pycmg.tech.resolve_modelcard
    from pycmg.tech import TECH_REGISTRY, resolve_modelcard
    tech_config = TECH_REGISTRY[_registry_name(tech)]
    prefix = tech.nmos_model.split("_", 1)[0]
    vt = tech.nmos_model.split("_", 1)[1].replace("_mac", "")
    canonical = ("nmos_" if prefix == "nch" else "pmos_") + vt
    device_config = tech_config.get_device(canonical)
    return Path(resolve_modelcard(
        device_config, tech_config,
        L=tech.l_nmos, NFIN=float(tech.nfin),
    ))


def resolve_pmos_modelcard(tech: TestTechConfig) -> Path:
    """Resolve the PMOS modelcard path."""
    if tech.single_file:
        # ASAP7: same file has both NMOS and PMOS
        return MODELCARDS_DIR / tech.modelcard_dir / tech.modelcard_file

    # TSMC: resolve via pycmg.tech.resolve_modelcard
    from pycmg.tech import TECH_REGISTRY, resolve_modelcard
    tech_config = TECH_REGISTRY[_registry_name(tech)]
    prefix = tech.pmos_model.split("_", 1)[0]
    vt = tech.pmos_model.split("_", 1)[1].replace("_mac", "")
    canonical = ("pmos_" if prefix == "pch" else "nmos_") + vt
    device_config = tech_config.get_device(canonical)
    return Path(resolve_modelcard(
        device_config, tech_config,
        L=tech.effective_l_pmos, NFIN=float(tech.nfin),
    ))


def create_baked_modelcard(tech: TestTechConfig, work_dir: Path) -> Path:
    """Create baked NMOS modelcard with L/NFIN/TFIN/DEVTYPE for NGSPICE."""
    src = resolve_nmos_modelcard(tech)
    if not src.exists():
        raise FileNotFoundError(f"NMOS modelcard not found: {src}")

    baked = work_dir / f"baked_nmos_{tech.name}.lib"
    baked.write_text(src.read_text())
    bake_inst_params(baked, baked, tech.nmos_model, {
        "L": tech.l_nmos,
        "NFIN": float(tech.nfin),
        "TFIN": tech.tfin,
        "DEVTYPE": 1,
    })
    return baked


def create_baked_pmos_modelcard(tech: TestTechConfig, work_dir: Path) -> Path:
    """Create baked PMOS modelcard with L/NFIN/TFIN/DEVTYPE for NGSPICE."""
    src = resolve_pmos_modelcard(tech)
    if not src.exists():
        raise FileNotFoundError(f"PMOS modelcard not found: {src}")

    baked = work_dir / f"baked_pmos_{tech.name}.lib"
    baked.write_text(src.read_text())
    bake_inst_params(baked, baked, tech.pmos_model, {
        "L": tech.effective_l_pmos,
        "NFIN": float(tech.nfin),
        "TFIN": tech.tfin,
        "DEVTYPE": 0,
    })
    return baked


# ---------------------------------------------------------------------------
# Checkpoint availability
# ---------------------------------------------------------------------------
def get_available_checkpoints() -> Dict[str, Optional[Path]]:
    """Check which NN checkpoints are available.

    Returns dict with keys:
      'bsimar_v4_nmos', 'directnet_v4_nmos',
      'bsimar_v4_pmos', 'directnet_v4_pmos'.
    Value is the checkpoint path if available, None otherwise.
    """
    checkpoints: Dict[str, Optional[Path]] = {}

    for dev in ("nmos", "pmos"):
        suffix = f"_{dev}"
        # BSIMAR v4 (tech-code embedding, phys-best)
        bsimar_phys = CHECKPOINT_DIR / f"v4_universal_{dev}_best.phys.pt"
        bsimar_plain = CHECKPOINT_DIR / f"v4_universal_{dev}_best.pt"
        if bsimar_phys.exists():
            checkpoints[f"bsimar_v4{suffix}"] = bsimar_phys
        elif bsimar_plain.exists():
            checkpoints[f"bsimar_v4{suffix}"] = bsimar_plain
        else:
            checkpoints[f"bsimar_v4{suffix}"] = None

        # DirectNet v4 (tech-code embedding)
        dn_v4 = CHECKPOINT_DIR / f"v4_dn_universal_{dev}_best.pt"
        checkpoints[f"directnet_v4{suffix}"] = dn_v4 if dn_v4.exists() else None

    # Backward-compat aliases (NMOS only, used by existing run_dc_tests/run_tran_tests)
    checkpoints["bsimar_v4"] = checkpoints["bsimar_v4_nmos"]
    checkpoints["directnet_v4"] = checkpoints["directnet_v4_nmos"]

    return checkpoints


# ---------------------------------------------------------------------------
# NGSPICE NMOS DC runner (ground truth)
# ---------------------------------------------------------------------------
def run_ngspice_nmos_dc(
    tech: TestTechConfig, work_dir: Path,
) -> Dict[str, np.ndarray]:
    """Run NGSPICE NMOS Id-Vgs DC sweep. Returns {sweep, id}."""
    baked = create_baked_modelcard(tech, work_dir)
    vds_bias = round(tech.vdd * 0.5, 4)

    # Netlist
    netlist_path = work_dir / f"ngspice_nmos_dc_{tech.name}.cir"
    netlist_content = (
        f"* NMOS Id-Vgs DC (NGSPICE ground truth, {tech.name})\n"
        f'.include "{baked}"\n'
        f".temp 27\n"
        f"Vds d 0 {vds_bias}\n"
        f"Vgs g 0 0.0\n"
        f"N1 d g 0 0 {tech.nmos_model}\n"
        f".dc Vgs 0 {tech.vdd} 0.005\n"
        f".end\n"
    )
    netlist_path.write_text(netlist_content)

    # Runner
    csv_path = work_dir / f"ngspice_nmos_dc_{tech.name}.csv"
    log_path = work_dir / f"ngspice_nmos_dc_{tech.name}.log"
    runner_path = work_dir / f"ngspice_nmos_dc_{tech.name}_runner.cir"
    runner_content = (
        f"* NGSPICE DC runner ({tech.name})\n"
        f".control\n"
        f"osdi {OSDI_PATH}\n"
        f"source {netlist_path}\n"
        f"set filetype=ascii\n"
        f"set wr_vecnames\n"
        f"run\n"
        f"wrdata {csv_path} i(Vds)\n"
        f".endc\n"
        f".end\n"
    )
    runner_path.write_text(runner_content)

    # Run
    res = subprocess.run(
        [NGSPICE_BIN, "-b", "-o", str(log_path), str(runner_path)],
        capture_output=True, text=True,
    )

    if log_path.exists():
        log_text = log_path.read_text()
        if "Fatal:" in log_text:
            raise RuntimeError(f"NGSPICE OSDI fatal error in {tech.name}")

    if not csv_path.exists():
        log_text = log_path.read_text() if log_path.exists() else "(no log)"
        raise RuntimeError(
            f"NGSPICE produced no output for {tech.name}: "
            f"RC={res.returncode}, log tail: ...{log_text[-500:]}"
        )

    # Parse wrdata
    with csv_path.open() as f:
        lines = f.readlines()

    data_rows = []
    for line in lines[1:]:
        stripped = line.strip()
        if stripped:
            data_rows.append([float(x) for x in stripped.split()])
    data = np.array(data_rows)

    if not np.all(np.isfinite(data)):
        raise RuntimeError(f"NGSPICE output contains NaN/Inf for {tech.name}")

    return {"sweep": data[:, 0], "id": np.abs(data[:, 1])}


# ---------------------------------------------------------------------------
# PyCircuitSim BSIM-CMG (LEVEL=72) NMOS DC — sanity check
# ---------------------------------------------------------------------------
def run_pycircuitsim_cmg_nmos_dc(
    tech: TestTechConfig, work_dir: Path,
) -> Dict[str, np.ndarray]:
    """Run PyCircuitSim BSIM-CMG NMOS Id-Vgs. Returns {sweep, id}."""
    from pycircuitsim.parser import Parser
    from pycircuitsim.simulation import run_dc_sweep
    from pycircuitsim.visualizer import Visualizer

    vds_bias = round(tech.vdd * 0.5, 4)
    l_nm = tech.l_nmos * 1e9

    netlist_path = work_dir / f"pycircuitsim_cmg_nmos_dc_{tech.name}.sp"
    content = (
        f"* BSIM-CMG NMOS Id-Vgs ({tech.name})\n"
        f"Vds 1 0 {vds_bias}\n"
        f"Vgs 2 0 0.0\n"
        f"Mn1 1 2 0 0 {tech.nmos_model} L={l_nm:.0f}n"
        f" NFIN={tech.nfin} TFIN={tech.tfin*1e9:.1f}n\n"
        f".model {tech.nmos_model} NMOS (LEVEL=72)\n"
        f".dc Vgs 0 {tech.vdd} 0.005\n"
        f".end\n"
    )
    netlist_path.write_text(content)

    modelcard = resolve_nmos_modelcard(tech)
    name_map = {"NMOS": tech.nmos_model}

    logging.disable(logging.CRITICAL)
    try:
        parser = Parser(
            modelcard_path=str(modelcard),
            model_name_map=name_map,
        )
        parser.parse_file(str(netlist_path))
        circuit = parser.circuit

        vis = Visualizer()
        out_dir = work_dir / f"cmg_dc_{tech.name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        results = run_dc_sweep(
            circuit, parser.analysis_params, vis, out_dir,
            f"cmg_nmos_{tech.name}",
        )
    finally:
        logging.disable(logging.NOTSET)

    sweep = np.array(results["2"])
    signal = np.abs(np.array(results["i(Mn1)"]))
    return {"sweep": sweep, "id": signal}


# ---------------------------------------------------------------------------
# PyCircuitSim NN NMOS DC — BSIMAR (LEVEL=74) or DirectNet (LEVEL=73)
# ---------------------------------------------------------------------------
def run_pycircuitsim_nn_nmos_dc(
    tech: TestTechConfig,
    work_dir: Path,
    level: int,
    model_name: str,
    model_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Run PyCircuitSim NN NMOS Id-Vgs. Returns {sweep, id}.

    Args:
        tech: technology config
        work_dir: output directory
        level: 73 (DirectNet) or 74 (BSIMAR)
        model_name: label for this model variant (e.g. "bsimar_v4")
        model_path: explicit checkpoint path (used for DirectNet v4 MODEL_PATH)
    """
    from pycircuitsim.parser import Parser
    from pycircuitsim.simulation import run_dc_sweep
    from pycircuitsim.visualizer import Visualizer

    vds_bias = round(tech.vdd * 0.5, 4)
    l_nm = tech.l_nmos * 1e9

    netlist_path = work_dir / f"nn_{model_name}_nmos_dc_{tech.name}.sp"

    # Build model params string
    model_params = f"LEVEL={level} TECH={tech.nn_tech_key} VT={tech.nn_vt}"
    if model_path is not None:
        model_params += f" MODEL_PATH={model_path}"

    content = (
        f"* NN NMOS Id-Vgs ({model_name}, {tech.name})\n"
        f"Vds 1 0 {vds_bias}\n"
        f"Vgs 2 0 0.0\n"
        f"Mn1 1 2 0 0 nmos_nn L={l_nm:.0f}n NFIN={tech.nfin}\n"
        f".model nmos_nn NMOS ({model_params})\n"
        f".dc Vgs 0 {tech.vdd} 0.005\n"
        f".end\n"
    )
    netlist_path.write_text(content)

    logging.disable(logging.CRITICAL)
    try:
        parser = Parser()
        parser.parse_file(str(netlist_path))
        circuit = parser.circuit

        vis = Visualizer()
        out_dir = work_dir / f"{model_name}_dc_{tech.name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        results = run_dc_sweep(
            circuit, parser.analysis_params, vis, out_dir,
            f"{model_name}_nmos_{tech.name}",
        )
    finally:
        logging.disable(logging.NOTSET)

    sweep = np.array(results["2"])
    signal = np.abs(np.array(results["i(Mn1)"]))
    return {"sweep": sweep, "id": signal}


# ---------------------------------------------------------------------------
# NGSPICE NMOS pulse response (ground truth, transient)
# ---------------------------------------------------------------------------
def run_ngspice_nmos_tran(
    tech: TestTechConfig, work_dir: Path,
) -> Dict[str, np.ndarray]:
    """Run NGSPICE NMOS pulse response transient.

    Circuit: Vgs pulse -> NMOS drain with Rload to Vdd.
    Returns {time, v(drain), v(gate)}.
    """
    baked = create_baked_modelcard(tech, work_dir)
    per = TRAN_TR + TRAN_PW + TRAN_TF + max(TRAN_PW, 1.0e-9)

    netlist_path = work_dir / f"ngspice_nmos_tran_{tech.name}.cir"
    content = (
        f"* NMOS pulse response (NGSPICE, {tech.name})\n"
        f'.include "{baked}"\n'
        f".temp 27\n"
        f"Vdd vdd 0 {tech.vdd}\n"
        f"Rload vdd drain {TRAN_RLOAD}\n"
        f"Vgs gate 0 PULSE(0 {tech.vdd} {TRAN_TD} {TRAN_TR} {TRAN_TF} {TRAN_PW} {per})\n"
        f"N1 drain gate 0 0 {tech.nmos_model}\n"
        f".ic V(drain)={tech.vdd}\n"
        f".tran {TRAN_TSTEP} {TRAN_TSTOP} uic\n"
        f".end\n"
    )
    netlist_path.write_text(content)

    csv_path = work_dir / f"ngspice_nmos_tran_{tech.name}.csv"
    log_path = work_dir / f"ngspice_nmos_tran_{tech.name}.log"
    runner_path = work_dir / f"ngspice_nmos_tran_{tech.name}_runner.cir"
    runner_content = (
        f"* NGSPICE tran runner ({tech.name})\n"
        f".control\n"
        f"osdi {OSDI_PATH}\n"
        f"source {netlist_path}\n"
        f"set filetype=ascii\n"
        f"set wr_vecnames\n"
        f"run\n"
        f"wrdata {csv_path} v(drain) v(gate)\n"
        f".endc\n"
        f".end\n"
    )
    runner_path.write_text(runner_content)

    res = subprocess.run(
        [NGSPICE_BIN, "-b", "-o", str(log_path), str(runner_path)],
        capture_output=True, text=True,
    )

    if log_path.exists():
        log_text = log_path.read_text()
        if "Fatal:" in log_text:
            raise RuntimeError(f"NGSPICE OSDI fatal error in tran {tech.name}")

    if not csv_path.exists():
        log_text = log_path.read_text() if log_path.exists() else "(no log)"
        raise RuntimeError(
            f"NGSPICE tran produced no output for {tech.name}: "
            f"RC={res.returncode}, log tail: ...{log_text[-500:]}"
        )

    with csv_path.open() as f:
        lines = f.readlines()

    data_rows = []
    for line in lines[1:]:
        stripped = line.strip()
        if stripped:
            data_rows.append([float(x) for x in stripped.split()])
    data = np.array(data_rows)

    return {
        "time": data[:, 0],
        "v(drain)": data[:, 1],
        "v(gate)": data[:, 3],
    }


# ---------------------------------------------------------------------------
# PyCircuitSim NN NMOS pulse response (transient)
# ---------------------------------------------------------------------------
def run_pycircuitsim_nn_nmos_tran(
    tech: TestTechConfig,
    work_dir: Path,
    level: int,
    model_name: str,
    model_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Run PyCircuitSim NN NMOS pulse response transient.

    Circuit: Vgs pulse -> NMOS drain with Rload to Vdd.
    Returns {time, v(drain), v(gate)}.
    """
    from pycircuitsim.parser import Parser
    from pycircuitsim.solver import DCSolver, TransientSolver

    l_nm = tech.l_nmos * 1e9
    per = TRAN_TR + TRAN_PW + TRAN_TF + max(TRAN_PW, 1.0e-9)

    model_params = f"LEVEL={level} TECH={tech.nn_tech_key} VT={tech.nn_vt}"
    if model_path is not None:
        model_params += f" MODEL_PATH={model_path}"

    netlist_path = work_dir / f"nn_{model_name}_nmos_tran_{tech.name}.sp"
    content = (
        f"* NN NMOS pulse response ({model_name}, {tech.name})\n"
        f"Vdd 1 0 {tech.vdd}\n"
        f"Rload 1 3 {TRAN_RLOAD}\n"
        f"Vgs 2 0 PULSE 0 {tech.vdd} {TRAN_TD} {TRAN_TR} {TRAN_TF}"
        f" {TRAN_PW} {per}\n"
        f"Mn1 3 2 0 0 nmos_nn L={l_nm:.0f}n NFIN={tech.nfin}\n"
        f".model nmos_nn NMOS ({model_params})\n"
        f".ic V(3)={tech.vdd}\n"
        f".tran {TRAN_TSTEP} {TRAN_TSTOP}\n"
        f".end\n"
    )
    netlist_path.write_text(content)

    logging.disable(logging.CRITICAL)
    try:
        parser = Parser()
        parser.parse_file(str(netlist_path))
        circuit = parser.circuit

        time_step: float = parser.analysis_params["tstep"]
        final_time: float = parser.analysis_params["tstop"]

        # Stage 1: DC OP
        initial_guess = circuit.initial_conditions if circuit.initial_conditions else None
        op_solver = DCSolver(
            circuit, initial_guess=initial_guess, use_source_stepping=True,
        )
        op_solution = op_solver.solve()

        # Stage 2: Transient
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

    # Node mapping: '1'=Vdd, '2'=gate, '3'=drain
    return {
        "time": results["time"],
        "v(drain)": results["3"],
        "v(gate)": results["2"],
    }


# ---------------------------------------------------------------------------
# NGSPICE PMOS DC runner (ground truth)
# ---------------------------------------------------------------------------
def run_ngspice_pmos_dc(
    tech: TestTechConfig, work_dir: Path,
) -> Dict[str, np.ndarray]:
    """Run NGSPICE PMOS Id-Vgs DC sweep. Returns {sweep, id}.

    Vgs swept from 0 to -VDD, Vds biased at -VDD/2.
    """
    baked = create_baked_pmos_modelcard(tech, work_dir)
    vds_bias = round(-tech.vdd * 0.5, 4)

    netlist_path = work_dir / f"ngspice_pmos_dc_{tech.name}.cir"
    netlist_content = (
        f"* PMOS Id-Vgs DC (NGSPICE ground truth, {tech.name})\n"
        f'.include "{baked}"\n'
        f".temp 27\n"
        f"Vds d 0 {vds_bias}\n"
        f"Vgs g 0 0.0\n"
        f"N1 d g 0 0 {tech.pmos_model}\n"
        f".dc Vgs 0 {-tech.vdd} -0.005\n"
        f".end\n"
    )
    netlist_path.write_text(netlist_content)

    csv_path = work_dir / f"ngspice_pmos_dc_{tech.name}.csv"
    log_path = work_dir / f"ngspice_pmos_dc_{tech.name}.log"
    runner_path = work_dir / f"ngspice_pmos_dc_{tech.name}_runner.cir"
    runner_content = (
        f"* NGSPICE PMOS DC runner ({tech.name})\n"
        f".control\n"
        f"osdi {OSDI_PATH}\n"
        f"source {netlist_path}\n"
        f"set filetype=ascii\n"
        f"set wr_vecnames\n"
        f"run\n"
        f"wrdata {csv_path} i(Vds)\n"
        f".endc\n"
        f".end\n"
    )
    runner_path.write_text(runner_content)

    res = subprocess.run(
        [NGSPICE_BIN, "-b", "-o", str(log_path), str(runner_path)],
        capture_output=True, text=True,
    )

    if log_path.exists():
        log_text = log_path.read_text()
        if "Fatal:" in log_text:
            raise RuntimeError(f"NGSPICE OSDI fatal error in PMOS {tech.name}")

    if not csv_path.exists():
        log_text = log_path.read_text() if log_path.exists() else "(no log)"
        raise RuntimeError(
            f"NGSPICE PMOS produced no output for {tech.name}: "
            f"RC={res.returncode}, log tail: ...{log_text[-500:]}"
        )

    with csv_path.open() as f:
        lines = f.readlines()
    data_rows = []
    for line in lines[1:]:
        stripped = line.strip()
        if stripped:
            data_rows.append([float(x) for x in stripped.split()])
    data = np.array(data_rows)

    if not np.all(np.isfinite(data)):
        raise RuntimeError(f"NGSPICE PMOS output contains NaN/Inf for {tech.name}")

    # Use |Vgs| for sweep (PMOS sweeps negative) and |Id| for current
    sweep = np.abs(data[:, 0])
    # Sort by ascending |Vgs| for consistent interpolation
    sort_idx = np.argsort(sweep)
    return {"sweep": sweep[sort_idx], "id": np.abs(data[sort_idx, 1])}


# ---------------------------------------------------------------------------
# PyCircuitSim NN PMOS DC
# ---------------------------------------------------------------------------
def run_pycircuitsim_nn_pmos_dc(
    tech: TestTechConfig,
    work_dir: Path,
    level: int,
    model_name: str,
    model_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Run PyCircuitSim NN PMOS Id-Vgs. Returns {sweep, id}.

    Vgs swept from 0 to -VDD, Vds biased at -VDD/2.
    """
    from pycircuitsim.parser import Parser
    from pycircuitsim.simulation import run_dc_sweep
    from pycircuitsim.visualizer import Visualizer

    vds_bias = round(-tech.vdd * 0.5, 4)
    l_nm = tech.effective_l_pmos * 1e9

    netlist_path = work_dir / f"nn_{model_name}_pmos_dc_{tech.name}.sp"

    model_params = f"LEVEL={level} TECH={tech.nn_tech_key} VT={tech.effective_pmos_vt}"
    if model_path is not None:
        model_params += f" MODEL_PATH={model_path}"

    content = (
        f"* NN PMOS Id-Vgs ({model_name}, {tech.name})\n"
        f"Vds 1 0 {vds_bias}\n"
        f"Vgs 2 0 0.0\n"
        f"Mp1 1 2 0 0 pmos_nn L={l_nm:.0f}n NFIN={tech.nfin}\n"
        f".model pmos_nn PMOS ({model_params})\n"
        f".dc Vgs 0 {-tech.vdd} -0.005\n"
        f".end\n"
    )
    netlist_path.write_text(content)

    logging.disable(logging.CRITICAL)
    try:
        parser = Parser()
        parser.parse_file(str(netlist_path))
        circuit = parser.circuit

        vis = Visualizer()
        out_dir = work_dir / f"{model_name}_pmos_dc_{tech.name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        results = run_dc_sweep(
            circuit, parser.analysis_params, vis, out_dir,
            f"{model_name}_pmos_{tech.name}",
        )
    finally:
        logging.disable(logging.NOTSET)

    # Use |Vgs| for sweep (PMOS sweeps negative) and |Id| for current
    sweep = np.abs(np.array(results["2"]))
    signal = np.abs(np.array(results["i(Mp1)"]))
    # Sort by ascending |Vgs|
    sort_idx = np.argsort(sweep)
    return {"sweep": sweep[sort_idx], "id": signal[sort_idx]}


# ---------------------------------------------------------------------------
# NGSPICE Inverter VTC (ground truth)
# ---------------------------------------------------------------------------
def run_ngspice_inverter_vtc(
    tech: TestTechConfig, work_dir: Path,
) -> Dict[str, np.ndarray]:
    """Run NGSPICE CMOS inverter VTC DC sweep. Returns {sweep, vout}."""
    baked_nmos = create_baked_modelcard(tech, work_dir)
    baked_pmos = create_baked_pmos_modelcard(tech, work_dir)

    netlist_path = work_dir / f"ngspice_inverter_vtc_{tech.name}.cir"
    netlist_content = (
        f"* CMOS Inverter VTC (NGSPICE, {tech.name})\n"
        f'.include "{baked_nmos}"\n'
        f'.include "{baked_pmos}"\n'
        f".temp 27\n"
        f"Vdd vdd 0 {tech.vdd}\n"
        f"Vin in 0 0.0\n"
        f"Nn out in 0 0 {tech.nmos_model}\n"
        f"Np out in vdd vdd {tech.pmos_model}\n"
        f".dc Vin 0 {tech.vdd} 0.005\n"
        f".end\n"
    )
    netlist_path.write_text(netlist_content)

    csv_path = work_dir / f"ngspice_inverter_vtc_{tech.name}.csv"
    log_path = work_dir / f"ngspice_inverter_vtc_{tech.name}.log"
    runner_path = work_dir / f"ngspice_inverter_vtc_{tech.name}_runner.cir"
    runner_content = (
        f"* NGSPICE inverter VTC runner ({tech.name})\n"
        f".control\n"
        f"osdi {OSDI_PATH}\n"
        f"source {netlist_path}\n"
        f"set filetype=ascii\n"
        f"set wr_vecnames\n"
        f"run\n"
        f"wrdata {csv_path} v(out)\n"
        f".endc\n"
        f".end\n"
    )
    runner_path.write_text(runner_content)

    res = subprocess.run(
        [NGSPICE_BIN, "-b", "-o", str(log_path), str(runner_path)],
        capture_output=True, text=True,
    )

    if log_path.exists():
        log_text = log_path.read_text()
        if "Fatal:" in log_text:
            raise RuntimeError(f"NGSPICE OSDI fatal error in inverter VTC {tech.name}")

    if not csv_path.exists():
        log_text = log_path.read_text() if log_path.exists() else "(no log)"
        raise RuntimeError(
            f"NGSPICE inverter VTC produced no output for {tech.name}: "
            f"RC={res.returncode}, log tail: ...{log_text[-500:]}"
        )

    with csv_path.open() as f:
        lines = f.readlines()
    data_rows = []
    for line in lines[1:]:
        stripped = line.strip()
        if stripped:
            data_rows.append([float(x) for x in stripped.split()])
    data = np.array(data_rows)

    if not np.all(np.isfinite(data)):
        raise RuntimeError(f"NGSPICE inverter VTC contains NaN/Inf for {tech.name}")

    return {"sweep": data[:, 0], "vout": data[:, 1]}


# ---------------------------------------------------------------------------
# PyCircuitSim NN Inverter VTC
# ---------------------------------------------------------------------------
def run_pycircuitsim_nn_inverter_vtc(
    tech: TestTechConfig,
    work_dir: Path,
    level: int,
    model_name: str,
    nmos_model_path: Optional[Path] = None,
    pmos_model_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Run PyCircuitSim NN inverter VTC. Returns {sweep, vout}."""
    from pycircuitsim.parser import Parser
    from pycircuitsim.simulation import run_dc_sweep
    from pycircuitsim.visualizer import Visualizer

    l_nmos_nm = tech.l_nmos * 1e9
    l_pmos_nm = tech.effective_l_pmos * 1e9

    netlist_path = work_dir / f"nn_{model_name}_inverter_vtc_{tech.name}.sp"

    nmos_params = f"LEVEL={level} TECH={tech.nn_tech_key} VT={tech.nn_vt}"
    if nmos_model_path is not None:
        nmos_params += f" MODEL_PATH={nmos_model_path}"

    pmos_params = f"LEVEL={level} TECH={tech.nn_tech_key} VT={tech.effective_pmos_vt}"
    if pmos_model_path is not None:
        pmos_params += f" MODEL_PATH={pmos_model_path}"

    content = (
        f"* NN CMOS Inverter VTC ({model_name}, {tech.name})\n"
        f"Vdd 1 0 {tech.vdd}\n"
        f"Vin 2 0 0.0\n"
        f"Mn1 3 2 0 0 nmos_nn L={l_nmos_nm:.0f}n NFIN={tech.nfin}\n"
        f"Mp1 3 2 1 1 pmos_nn L={l_pmos_nm:.0f}n NFIN={tech.nfin}\n"
        f".model nmos_nn NMOS ({nmos_params})\n"
        f".model pmos_nn PMOS ({pmos_params})\n"
        f".dc Vin 0 {tech.vdd} 0.005\n"
        f".end\n"
    )
    netlist_path.write_text(content)

    logging.disable(logging.CRITICAL)
    try:
        parser = Parser()
        parser.parse_file(str(netlist_path))
        circuit = parser.circuit

        vis = Visualizer()
        out_dir = work_dir / f"{model_name}_inverter_vtc_{tech.name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        results = run_dc_sweep(
            circuit, parser.analysis_params, vis, out_dir,
            f"{model_name}_inverter_{tech.name}",
        )
    finally:
        logging.disable(logging.NOTSET)

    sweep = np.array(results["2"])   # Vin node
    vout = np.array(results["3"])    # Vout node
    return {"sweep": sweep, "vout": vout}


# ---------------------------------------------------------------------------
# Inverter VTC comparison and plotting
# ---------------------------------------------------------------------------
def compare_vtc_curves(
    ref_sweep: np.ndarray,
    ref_vout: np.ndarray,
    test_sweep: np.ndarray,
    test_vout: np.ndarray,
) -> Dict[str, float]:
    """Compare two inverter VTC curves."""
    common_start = max(ref_sweep[0], test_sweep[0])
    common_stop = min(ref_sweep[-1], test_sweep[-1])
    mask = (ref_sweep >= common_start - 1e-10) & (ref_sweep <= common_stop + 1e-10)
    ref_c = ref_sweep[mask]
    ref_v = ref_vout[mask]
    test_interp = np.interp(ref_c, test_sweep, test_vout)

    return {
        "nrmse": nrmse(test_interp, ref_v),
        "max_vout_ref": float(np.max(ref_v)),
        "max_vout_test": float(np.max(test_interp)),
        "n_points": len(ref_c),
    }


def plot_vtc_comparison_multi(
    ref_data: Dict[str, np.ndarray],
    model_results: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, float]]],
    tech: TestTechConfig,
    save_path: Path,
) -> None:
    """Plot inverter VTC overlay for all models vs ground truth."""
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [2, 1]},
    )

    colors = {
        "cmg_pycircuitsim": "green",
        "bsimar_v4": "red",
        "directnet_v4": "purple",
    }
    linestyles = {
        "cmg_pycircuitsim": "-.",
        "bsimar_v4": "--",
        "directnet_v4": ":",
    }

    # Panel 1: VTC
    ax1 = axes[0]
    ax1.plot(ref_data["sweep"], ref_data["vout"], "b-", lw=2,
             label="NGSPICE BSIM-CMG (truth)")
    # Ideal line
    ax1.plot([0, tech.vdd], [tech.vdd, 0], "k--", lw=0.5, alpha=0.3, label="Ideal")
    for mname, (mdata, metrics) in model_results.items():
        label = f"{mname} (NRMSE={metrics['nrmse']:.2f}%)"
        ax1.plot(mdata["sweep"], mdata["vout"],
                 color=colors.get(mname, "gray"),
                 linestyle=linestyles.get(mname, "--"),
                 lw=1.5, label=label)

    ax1.set_xlabel("Vin (V)")
    ax1.set_ylabel("Vout (V)")
    ax1.set_title(
        f"Inverter VTC: {tech.name}  "
        f"L_n={tech.l_nmos*1e9:.0f}nm  L_p={tech.effective_l_pmos*1e9:.0f}nm  "
        f"NFIN={tech.nfin}"
    )
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, tech.vdd)
    ax1.set_ylim(-0.05, tech.vdd + 0.05)

    # Panel 2: Error
    ax2 = axes[1]
    for mname, (mdata, metrics) in model_results.items():
        common_start = max(ref_data["sweep"][0], mdata["sweep"][0])
        common_stop = min(ref_data["sweep"][-1], mdata["sweep"][-1])
        mask = ((ref_data["sweep"] >= common_start - 1e-10) &
                (ref_data["sweep"] <= common_stop + 1e-10))
        ref_c = ref_data["sweep"][mask]
        ref_v = ref_data["vout"][mask]
        test_interp = np.interp(ref_c, mdata["sweep"], mdata["vout"])
        error_mv = (test_interp - ref_v) * 1e3
        ax2.plot(ref_c, error_mv,
                 color=colors.get(mname, "gray"), lw=0.8, label=mname)

    ax2.axhline(y=0, color="k", lw=0.5)
    ax2.set_ylabel("Error [mV]")
    ax2.set_xlabel("Vin (V)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# NGSPICE Inverter transient (ground truth)
# ---------------------------------------------------------------------------
def run_ngspice_inverter_tran(
    tech: TestTechConfig, work_dir: Path,
) -> Dict[str, np.ndarray]:
    """Run NGSPICE CMOS inverter transient with load capacitor.

    Returns {time, v(out), v(in)}.
    """
    baked_nmos = create_baked_modelcard(tech, work_dir)
    baked_pmos = create_baked_pmos_modelcard(tech, work_dir)
    per = INV_TRAN_TR + INV_TRAN_PW + INV_TRAN_TF + max(INV_TRAN_PW, 1.0e-9)

    netlist_path = work_dir / f"ngspice_inverter_tran_{tech.name}.cir"
    content = (
        f"* CMOS Inverter Transient (NGSPICE, {tech.name})\n"
        f'.include "{baked_nmos}"\n'
        f'.include "{baked_pmos}"\n'
        f".temp 27\n"
        f"Vdd vdd 0 {tech.vdd}\n"
        f"Vin in 0 PULSE(0 {tech.vdd} {INV_TRAN_TD} {INV_TRAN_TR}"
        f" {INV_TRAN_TF} {INV_TRAN_PW} {per})\n"
        f"Nn out in 0 0 {tech.nmos_model}\n"
        f"Np out in vdd vdd {tech.pmos_model}\n"
        f"Cload out 0 {INV_CLOAD}\n"
        f".ic V(out)={tech.vdd}\n"
        f".tran {INV_TRAN_TSTEP} {INV_TRAN_TSTOP} uic\n"
        f".end\n"
    )
    netlist_path.write_text(content)

    csv_path = work_dir / f"ngspice_inverter_tran_{tech.name}.csv"
    log_path = work_dir / f"ngspice_inverter_tran_{tech.name}.log"
    runner_path = work_dir / f"ngspice_inverter_tran_{tech.name}_runner.cir"
    runner_content = (
        f"* NGSPICE inverter tran runner ({tech.name})\n"
        f".control\n"
        f"osdi {OSDI_PATH}\n"
        f"source {netlist_path}\n"
        f"set filetype=ascii\n"
        f"set wr_vecnames\n"
        f"run\n"
        f"wrdata {csv_path} v(out) v(in)\n"
        f".endc\n"
        f".end\n"
    )
    runner_path.write_text(runner_content)

    res = subprocess.run(
        [NGSPICE_BIN, "-b", "-o", str(log_path), str(runner_path)],
        capture_output=True, text=True,
    )

    if log_path.exists():
        log_text = log_path.read_text()
        if "Fatal:" in log_text:
            raise RuntimeError(
                f"NGSPICE OSDI fatal error in inverter tran {tech.name}"
            )

    if not csv_path.exists():
        log_text = log_path.read_text() if log_path.exists() else "(no log)"
        raise RuntimeError(
            f"NGSPICE inverter tran produced no output for {tech.name}: "
            f"RC={res.returncode}, log tail: ...{log_text[-500:]}"
        )

    with csv_path.open() as f:
        lines = f.readlines()
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
# PyCircuitSim NN Inverter transient
# ---------------------------------------------------------------------------
def run_pycircuitsim_nn_inverter_tran(
    tech: TestTechConfig,
    work_dir: Path,
    level: int,
    model_name: str,
    nmos_model_path: Optional[Path] = None,
    pmos_model_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Run PyCircuitSim NN inverter transient. Returns {time, v(out), v(in)}."""
    from pycircuitsim.parser import Parser
    from pycircuitsim.solver import DCSolver, TransientSolver

    l_nmos_nm = tech.l_nmos * 1e9
    l_pmos_nm = tech.effective_l_pmos * 1e9
    per = INV_TRAN_TR + INV_TRAN_PW + INV_TRAN_TF + max(INV_TRAN_PW, 1.0e-9)

    nmos_params = f"LEVEL={level} TECH={tech.nn_tech_key} VT={tech.nn_vt}"
    if nmos_model_path is not None:
        nmos_params += f" MODEL_PATH={nmos_model_path}"

    pmos_params = f"LEVEL={level} TECH={tech.nn_tech_key} VT={tech.effective_pmos_vt}"
    if pmos_model_path is not None:
        pmos_params += f" MODEL_PATH={pmos_model_path}"

    netlist_path = work_dir / f"nn_{model_name}_inverter_tran_{tech.name}.sp"
    content = (
        f"* NN Inverter Transient ({model_name}, {tech.name})\n"
        f"Vdd 1 0 {tech.vdd}\n"
        f"Vin 2 0 PULSE 0 {tech.vdd} {INV_TRAN_TD} {INV_TRAN_TR}"
        f" {INV_TRAN_TF} {INV_TRAN_PW} {per}\n"
        f"Mn1 3 2 0 0 nmos_nn L={l_nmos_nm:.0f}n NFIN={tech.nfin}\n"
        f"Mp1 3 2 1 1 pmos_nn L={l_pmos_nm:.0f}n NFIN={tech.nfin}\n"
        f"Cload 3 0 {INV_CLOAD}\n"
        f".model nmos_nn NMOS ({nmos_params})\n"
        f".model pmos_nn PMOS ({pmos_params})\n"
        f".ic V(3)={tech.vdd}\n"
        f".tran {INV_TRAN_TSTEP} {INV_TRAN_TSTOP}\n"
        f".end\n"
    )
    netlist_path.write_text(content)

    logging.disable(logging.CRITICAL)
    try:
        parser = Parser()
        parser.parse_file(str(netlist_path))
        circuit = parser.circuit

        time_step: float = parser.analysis_params["tstep"]
        final_time: float = parser.analysis_params["tstop"]

        # Stage 1: DC OP
        initial_guess = circuit.initial_conditions if circuit.initial_conditions else None
        op_solver = DCSolver(
            circuit, initial_guess=initial_guess, use_source_stepping=True,
        )
        op_solution = op_solver.solve()

        # Stage 2: Transient
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
    return {
        "time": results["time"],
        "v(out)": results["3"],
        "v(in)": results["2"],
    }


# ---------------------------------------------------------------------------
# Inverter transient comparison and plotting
# ---------------------------------------------------------------------------
def compare_inverter_tran_waveforms(
    ref_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    vdd: float,
    t_start: float = 0.0,
) -> Dict[str, float]:
    """Compare inverter transient output waveforms on common time grid."""
    t_max = min(ref_data["time"][-1], test_data["time"][-1])
    t_common = np.arange(max(t_start, ref_data["time"][0]), t_max, INV_TRAN_TSTEP)

    ref_v = np.interp(t_common, ref_data["time"], ref_data["v(out)"])
    test_v = np.interp(t_common, test_data["time"], test_data["v(out)"])

    diff = test_v - ref_v
    rmse_val = float(np.sqrt(np.mean(diff ** 2)))
    nrmse_val = rmse_val / vdd * 100.0
    max_err = float(np.max(np.abs(diff)))

    return {
        "nrmse_vdd": nrmse_val,
        "max_err_v": max_err,
        "max_err_pct": max_err / vdd * 100.0,
        "n_points": len(t_common),
    }


def compute_region_errors(
    ref_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    vdd: float,
    t_start: float = 0.0,
) -> Dict[str, float]:
    """Decompose inverter transient NRMSE into high-rail / low-rail / transition.

    Regions are defined by the input voltage at each time point:
      - high_rail: Vin < 0.1*VDD  (output should be ~VDD)
      - low_rail:  Vin > 0.9*VDD  (output should be ~0)
      - transition: everything else (rising/falling edges)

    Returns dict with keys 'nrmse_high', 'nrmse_low', 'nrmse_trans',
    and 'n_high', 'n_low', 'n_trans' (sample counts).
    """
    t_max = min(ref_data["time"][-1], test_data["time"][-1])
    t_common = np.arange(max(t_start, ref_data["time"][0]), t_max, INV_TRAN_TSTEP)

    ref_v = np.interp(t_common, ref_data["time"], ref_data["v(out)"])
    test_v = np.interp(t_common, test_data["time"], test_data["v(out)"])
    vin = np.interp(t_common, ref_data["time"], ref_data["v(in)"])

    diff = test_v - ref_v

    high_mask = vin < 0.1 * vdd
    low_mask = vin > 0.9 * vdd
    trans_mask = ~high_mask & ~low_mask

    result: Dict[str, float] = {}
    for tag, mask in [("high", high_mask), ("low", low_mask), ("trans", trans_mask)]:
        n = int(mask.sum())
        result[f"n_{tag}"] = float(n)
        if n > 0:
            rmse_val = float(np.sqrt(np.mean(diff[mask] ** 2)))
            result[f"nrmse_{tag}"] = rmse_val / vdd * 100.0
        else:
            result[f"nrmse_{tag}"] = float("nan")

    return result


def plot_inverter_tran_comparison_multi(
    ref_data: Dict[str, np.ndarray],
    model_results: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, float]]],
    tech: TestTechConfig,
    save_path: Path,
) -> None:
    """Plot inverter transient waveform overlay for all models vs ground truth."""
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 10),
        gridspec_kw={"height_ratios": [0.6, 1, 0.6]},
    )

    colors = {
        "bsimar_v4": "red",
        "directnet_v4": "purple",
    }

    # Panel 1: Input pulse
    ax1 = axes[0]
    ng_t_ns = ref_data["time"] * 1e9
    ax1.plot(ng_t_ns, ref_data["v(in)"], "b-", lw=1.5, label="V(in)")
    ax1.set_ylabel("V(in) [V]")
    ax1.set_title(
        f"Inverter Transient: {tech.name}  "
        f"L_n={tech.l_nmos*1e9:.0f}nm  L_p={tech.effective_l_pmos*1e9:.0f}nm  "
        f"NFIN={tech.nfin}  Cload={INV_CLOAD*1e15:.0f}fF"
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, tech.vdd + 0.1)

    # Panel 2: Output voltage
    ax2 = axes[1]
    ax2.plot(ng_t_ns, ref_data["v(out)"], "b-", lw=2,
             label="NGSPICE BSIM-CMG")
    for mname, (mdata, metrics) in model_results.items():
        nn_t_ns = mdata["time"] * 1e9
        label = f"{mname} (NRMSE={metrics['nrmse_vdd']:.2f}%)"
        ax2.plot(nn_t_ns, mdata["v(out)"],
                 color=colors.get(mname, "gray"),
                 linestyle="--", lw=1.5, label=label)

    ax2.set_ylabel("V(out) [V]")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, tech.vdd + 0.15)

    # Panel 3: Error
    ax3 = axes[2]
    for mname, (mdata, metrics) in model_results.items():
        t_max = min(ref_data["time"][-1], mdata["time"][-1])
        t_common = np.arange(TRAN_STARTUP_EXCL, t_max, INV_TRAN_TSTEP)
        ref_v = np.interp(t_common, ref_data["time"], ref_data["v(out)"])
        test_v = np.interp(t_common, mdata["time"], mdata["v(out)"])
        error_mv = (test_v - ref_v) * 1e3
        ax3.plot(t_common * 1e9, error_mv, color=colors.get(mname, "gray"),
                 lw=0.8, label=mname)

    ax3.axhline(y=0, color="k", lw=0.5)
    threshold_mv = tech.vdd * TRAN_NRMSE_THRESHOLD * 1e3
    ax3.axhline(y=threshold_mv, color="r", lw=0.5, ls="--",
                label=f"{TRAN_NRMSE_THRESHOLD*100:.0f}% Vdd")
    ax3.axhline(y=-threshold_mv, color="r", lw=0.5, ls="--")
    ax3.set_ylabel("Error [mV]")
    ax3.set_xlabel("Time [ns]")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# DC comparison and plotting
# ---------------------------------------------------------------------------
def compare_dc_curves(
    ref_sweep: np.ndarray,
    ref_id: np.ndarray,
    test_sweep: np.ndarray,
    test_id: np.ndarray,
) -> Dict[str, float]:
    """Compare two Id-Vgs curves, interpolating test onto ref grid."""
    common_start = max(ref_sweep[0], test_sweep[0])
    common_stop = min(ref_sweep[-1], test_sweep[-1])
    mask = (ref_sweep >= common_start - 1e-10) & (ref_sweep <= common_stop + 1e-10)
    ref_c = ref_sweep[mask]
    ref_v = ref_id[mask]
    test_interp = np.interp(ref_c, test_sweep, test_id)

    return {
        "nrmse": nrmse(test_interp, ref_v),
        "mre": mre(test_interp, ref_v),
        "max_id_ref": float(np.max(ref_v)),
        "max_id_test": float(np.max(test_interp)),
        "n_points": len(ref_c),
    }


def plot_dc_comparison_multi(
    ref_data: Dict[str, np.ndarray],
    model_results: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, float]]],
    tech: TestTechConfig,
    save_path: Path,
) -> None:
    """Plot Id-Vgs overlay for all models vs ground truth, plus log scale."""
    n_models = len(model_results)
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 10),
        gridspec_kw={"height_ratios": [1, 1]},
    )

    colors = {
        "cmg_pycircuitsim": "green",
        "bsimar_v4": "red",
        "directnet_v4": "purple",
    }
    linestyles = {
        "cmg_pycircuitsim": "-.",
        "bsimar_v4": "--",
        "directnet_v4": ":",
    }

    # Linear scale
    ax1 = axes[0]
    ax1.plot(ref_data["sweep"], ref_data["id"], "b-", lw=2,
             label="NGSPICE BSIM-CMG (truth)")
    for mname, (mdata, metrics) in model_results.items():
        label = f"{mname} (NRMSE={metrics['nrmse']:.2f}%)"
        ax1.plot(mdata["sweep"], mdata["id"],
                 color=colors.get(mname, "gray"),
                 linestyle=linestyles.get(mname, "--"),
                 lw=1.5, label=label)

    ax1.set_xlabel("Vgs (V)")
    ax1.set_ylabel("|Id| (A)")
    ax1.set_title(
        f"NMOS Id-Vgs: {tech.name}  "
        f"L={tech.l_nmos*1e9:.0f}nm  NFIN={tech.nfin}  "
        f"Vds={tech.vdd*0.5:.2f}V"
    )
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Log scale
    ax2 = axes[1]
    ref_id_pos = ref_data["id"].copy()
    ref_id_pos[ref_id_pos <= 0] = 1e-15
    ax2.semilogy(ref_data["sweep"], ref_id_pos, "b-", lw=2,
                 label="NGSPICE BSIM-CMG")
    for mname, (mdata, metrics) in model_results.items():
        test_pos = mdata["id"].copy()
        test_pos[test_pos <= 0] = 1e-15
        label = f"{mname} (MRE={metrics['mre']:.2f}%)"
        ax2.semilogy(mdata["sweep"], test_pos,
                     color=colors.get(mname, "gray"),
                     linestyle=linestyles.get(mname, "--"),
                     lw=1.5, label=label)

    ax2.set_xlabel("Vgs (V)")
    ax2.set_ylabel("|Id| (A)")
    ax2.set_title("Log scale")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=1e-12)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Transient comparison and plotting
# ---------------------------------------------------------------------------
def compare_tran_waveforms(
    ref_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    vdd: float,
    t_start: float = 0.0,
) -> Dict[str, float]:
    """Compare transient drain waveforms on common time grid."""
    t_max = min(ref_data["time"][-1], test_data["time"][-1])
    t_common = np.arange(max(t_start, ref_data["time"][0]), t_max, TRAN_TSTEP)

    ref_v = np.interp(t_common, ref_data["time"], ref_data["v(drain)"])
    test_v = np.interp(t_common, test_data["time"], test_data["v(drain)"])

    diff = test_v - ref_v
    rmse_val = float(np.sqrt(np.mean(diff ** 2)))
    nrmse_val = rmse_val / vdd * 100.0
    max_err = float(np.max(np.abs(diff)))

    return {
        "nrmse_vdd": nrmse_val,
        "max_err_v": max_err,
        "max_err_pct": max_err / vdd * 100.0,
        "n_points": len(t_common),
    }


def plot_tran_comparison_multi(
    ref_data: Dict[str, np.ndarray],
    model_results: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, float]]],
    tech: TestTechConfig,
    save_path: Path,
) -> None:
    """Plot transient waveform overlay for all models vs ground truth."""
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 10),
        gridspec_kw={"height_ratios": [0.6, 1, 0.6]},
    )

    colors = {
        "bsimar_v4": "red",
        "directnet_v4": "purple",
    }

    # Panel 1: Gate pulse
    ax1 = axes[0]
    ng_t_ns = ref_data["time"] * 1e9
    ax1.plot(ng_t_ns, ref_data["v(gate)"], "b-", lw=1.5, label="V(gate)")
    ax1.set_ylabel("V(gate) [V]")
    ax1.set_title(
        f"NMOS Pulse Response: {tech.name}  "
        f"L={tech.l_nmos*1e9:.0f}nm  NFIN={tech.nfin}  "
        f"Rload={TRAN_RLOAD/1e3:.0f}k"
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, tech.vdd + 0.1)

    # Panel 2: Drain voltage
    ax2 = axes[1]
    ax2.plot(ng_t_ns, ref_data["v(drain)"], "b-", lw=2,
             label="NGSPICE BSIM-CMG")
    for mname, (mdata, metrics) in model_results.items():
        nn_t_ns = mdata["time"] * 1e9
        label = f"{mname} (NRMSE={metrics['nrmse_vdd']:.2f}%)"
        ax2.plot(nn_t_ns, mdata["v(drain)"],
                 color=colors.get(mname, "gray"),
                 linestyle="--", lw=1.5, label=label)

    ax2.set_ylabel("V(drain) [V]")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, tech.vdd + 0.15)

    # Panel 3: Error (post startup)
    ax3 = axes[2]
    for mname, (mdata, metrics) in model_results.items():
        t_max = min(ref_data["time"][-1], mdata["time"][-1])
        t_common = np.arange(TRAN_STARTUP_EXCL, t_max, TRAN_TSTEP)
        ref_v = np.interp(t_common, ref_data["time"], ref_data["v(drain)"])
        test_v = np.interp(t_common, mdata["time"], mdata["v(drain)"])
        error_mv = (test_v - ref_v) * 1e3
        ax3.plot(t_common * 1e9, error_mv, color=colors.get(mname, "gray"),
                 lw=0.8, label=mname)

    ax3.axhline(y=0, color="k", lw=0.5)
    threshold_mv = tech.vdd * TRAN_NRMSE_THRESHOLD * 1e3
    ax3.axhline(y=threshold_mv, color="r", lw=0.5, ls="--",
                label=f"{TRAN_NRMSE_THRESHOLD*100:.0f}% Vdd")
    ax3.axhline(y=-threshold_mv, color="r", lw=0.5, ls="--")
    ax3.set_ylabel("Error [mV]")
    ax3.set_xlabel("Time [ns]")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    """Container for one test result."""
    tech: str
    model: str
    analysis: str           # "dc" or "tran"
    nrmse_pct: float
    mre_pct: float = float("nan")
    max_id_ref: float = 0.0
    max_id_test: float = 0.0
    passed: bool = False
    error: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DC test runner
# ---------------------------------------------------------------------------
def run_dc_tests(
    tech_names: List[str],
    checkpoints: Dict[str, Optional[Path]],
) -> List[TestResult]:
    """Run DC Id-Vgs tests for all techs and models."""
    results: List[TestResult] = []

    for tech_name in tech_names:
        tech = ALL_TEST_TECHS[tech_name]
        work_dir = RESULTS_BASE / "dc" / tech.name
        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  DC Test: {tech.name}  L={tech.l_nmos*1e9:.0f}nm  NFIN={tech.nfin}  "
              f"VDD={tech.vdd:.2f}V  Vds={tech.vdd*0.5:.2f}V")
        print(f"{'='*70}")

        # 1. NGSPICE ground truth
        print(f"  [1/N] Running NGSPICE BSIM-CMG ground truth...")
        try:
            ng_data = run_ngspice_nmos_dc(tech, work_dir)
            print(f"    Done: {len(ng_data['sweep'])} pts, "
                  f"Id_max={ng_data['id'].max():.4e} A")
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append(TestResult(
                tech=tech.name, model="ngspice", analysis="dc",
                nrmse_pct=float("nan"), error=str(e),
            ))
            continue

        # 2. PyCircuitSim BSIM-CMG sanity check
        cmg_ok = False
        cmg_data: Optional[Dict[str, np.ndarray]] = None
        cmg_metrics: Dict[str, float] = {}
        print(f"  [2/N] Running PyCircuitSim BSIM-CMG (LEVEL=72) sanity...")
        try:
            cmg_data = run_pycircuitsim_cmg_nmos_dc(tech, work_dir)
            cmg_metrics = compare_dc_curves(
                ng_data["sweep"], ng_data["id"],
                cmg_data["sweep"], cmg_data["id"],
            )
            passed = cmg_metrics["nrmse"] < DC_NRMSE_THRESHOLD_CMG * 100
            status = "PASS" if passed else "FAIL"
            print(f"    NRMSE={cmg_metrics['nrmse']:.4f}%  "
                  f"MRE={cmg_metrics['mre']:.2f}%  "
                  f"Id_max={cmg_metrics['max_id_test']:.4e} -> {status}")
            cmg_ok = True
            results.append(TestResult(
                tech=tech.name, model="cmg_pycircuitsim", analysis="dc",
                nrmse_pct=cmg_metrics["nrmse"], mre_pct=cmg_metrics["mre"],
                max_id_ref=cmg_metrics["max_id_ref"],
                max_id_test=cmg_metrics["max_id_test"],
                passed=passed,
            ))
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append(TestResult(
                tech=tech.name, model="cmg_pycircuitsim", analysis="dc",
                nrmse_pct=float("nan"), error=str(e),
            ))

        # Collect model data for multi-model plot
        model_results_for_plot: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, float]]] = {}
        if cmg_ok:
            model_results_for_plot["cmg_pycircuitsim"] = (cmg_data, cmg_metrics)

        # 3. BSIMAR v4 (LEVEL=74, tech-code embedding)
        if checkpoints.get("bsimar_v4") is not None:
            bsimar_ckpt = checkpoints["bsimar_v4"]
            print(f"  [3/N] Running BSIMAR v4 (LEVEL=74)...")
            print(f"    Checkpoint: {bsimar_ckpt.name}")
            try:
                bsimar_data = run_pycircuitsim_nn_nmos_dc(
                    tech, work_dir, level=74,
                    model_name="bsimar_v4",
                    model_path=bsimar_ckpt,
                )
                bsimar_metrics = compare_dc_curves(
                    ng_data["sweep"], ng_data["id"],
                    bsimar_data["sweep"], bsimar_data["id"],
                )
                passed = bsimar_metrics["nrmse"] < DC_NRMSE_THRESHOLD_NN * 100
                status = "PASS" if passed else "FAIL"
                print(f"    NRMSE={bsimar_metrics['nrmse']:.2f}%  "
                      f"MRE={bsimar_metrics['mre']:.2f}%  "
                      f"Id_max={bsimar_metrics['max_id_test']:.4e} -> {status}")
                results.append(TestResult(
                    tech=tech.name, model="bsimar_v4", analysis="dc",
                    nrmse_pct=bsimar_metrics["nrmse"],
                    mre_pct=bsimar_metrics["mre"],
                    max_id_ref=bsimar_metrics["max_id_ref"],
                    max_id_test=bsimar_metrics["max_id_test"],
                    passed=passed,
                ))
                model_results_for_plot["bsimar_v4"] = (bsimar_data, bsimar_metrics)
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append(TestResult(
                    tech=tech.name, model="bsimar_v4", analysis="dc",
                    nrmse_pct=float("nan"), error=str(e),
                ))
        else:
            print(f"  [3/N] BSIMAR v4 -- SKIPPED (no checkpoint)")

        # 4. DirectNet v4 (LEVEL=73, tech-code embedding, explicit MODEL_PATH)
        if checkpoints.get("directnet_v4") is not None:
            dnv4_ckpt = checkpoints["directnet_v4"]
            print(f"  [4/N] Running DirectNet v4 (LEVEL=73, tech-code embedding)...")
            print(f"    Checkpoint: {dnv4_ckpt.name}")
            try:
                dnv4_data = run_pycircuitsim_nn_nmos_dc(
                    tech, work_dir, level=73,
                    model_name="directnet_v4",
                    model_path=dnv4_ckpt,
                )
                dnv4_metrics = compare_dc_curves(
                    ng_data["sweep"], ng_data["id"],
                    dnv4_data["sweep"], dnv4_data["id"],
                )
                # Check for broken model: flat output at ~0.5A
                id_range = float(dnv4_data["id"].max() - dnv4_data["id"].min())
                id_max = float(dnv4_data["id"].max())
                is_broken = (id_range < 1e-6) or (id_max > 0.1)

                if is_broken:
                    print(f"    WARNING: Model appears BROKEN "
                          f"(Id_range={id_range:.2e}, Id_max={id_max:.2e})")
                    results.append(TestResult(
                        tech=tech.name, model="directnet_v4", analysis="dc",
                        nrmse_pct=dnv4_metrics["nrmse"],
                        mre_pct=dnv4_metrics["mre"],
                        max_id_ref=dnv4_metrics["max_id_ref"],
                        max_id_test=dnv4_metrics["max_id_test"],
                        passed=False,
                        error="BROKEN: flat/extreme output",
                    ))
                else:
                    passed = dnv4_metrics["nrmse"] < DC_NRMSE_THRESHOLD_NN * 100
                    status = "PASS" if passed else "FAIL"
                    print(f"    NRMSE={dnv4_metrics['nrmse']:.2f}%  "
                          f"MRE={dnv4_metrics['mre']:.2f}%  "
                          f"Id_max={dnv4_metrics['max_id_test']:.4e} -> {status}")
                    results.append(TestResult(
                        tech=tech.name, model="directnet_v4", analysis="dc",
                        nrmse_pct=dnv4_metrics["nrmse"],
                        mre_pct=dnv4_metrics["mre"],
                        max_id_ref=dnv4_metrics["max_id_ref"],
                        max_id_test=dnv4_metrics["max_id_test"],
                        passed=passed,
                    ))
                model_results_for_plot["directnet_v4"] = (dnv4_data, dnv4_metrics)
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append(TestResult(
                    tech=tech.name, model="directnet_v4", analysis="dc",
                    nrmse_pct=float("nan"), error=str(e),
                ))
        else:
            print(f"  [4/N] DirectNet v4 -- SKIPPED (no checkpoint)")

        # Plot multi-model comparison
        if model_results_for_plot:
            plot_path = work_dir / f"dc_comparison_{tech.name}.png"
            plot_dc_comparison_multi(
                ng_data, model_results_for_plot, tech, plot_path,
            )
            print(f"  [Plot] Saved: {plot_path}")

    return results


# ---------------------------------------------------------------------------
# PMOS DC test runner
# ---------------------------------------------------------------------------
def run_pmos_dc_tests(
    tech_names: List[str],
    checkpoints: Dict[str, Optional[Path]],
) -> List[TestResult]:
    """Run PMOS DC Id-Vgs tests for all techs and NN models."""
    results: List[TestResult] = []

    has_pmos_ckpt = (checkpoints.get("bsimar_v4_pmos") is not None or
                     checkpoints.get("directnet_v4_pmos") is not None)
    if not has_pmos_ckpt:
        print("\n  PMOS DC tests -- SKIPPED (no PMOS checkpoints)")
        return results

    for tech_name in tech_names:
        tech = ALL_TEST_TECHS[tech_name]
        if not tech.pmos_model:
            print(f"\n  PMOS DC {tech.name} -- SKIPPED (no PMOS model configured)")
            continue

        # Skip ASAP7 if tech code is out of vocabulary
        if not _tech_code_in_vocab(tech.nn_tech_key, tech.effective_pmos_vt):
            print(f"\n  PMOS DC {tech.name} -- SKIPPED (tech code out of vocab)")
            continue

        work_dir = RESULTS_BASE / "pmos_dc" / tech.name
        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  PMOS DC Test: {tech.name}  "
              f"L={tech.effective_l_pmos*1e9:.0f}nm  NFIN={tech.nfin}  "
              f"VDD={tech.vdd:.2f}V  Vds={-tech.vdd*0.5:.2f}V")
        print(f"{'='*70}")

        # 1. NGSPICE ground truth
        print(f"  [1/N] Running NGSPICE PMOS BSIM-CMG ground truth...")
        try:
            ng_data = run_ngspice_pmos_dc(tech, work_dir)
            print(f"    Done: {len(ng_data['sweep'])} pts, "
                  f"|Id|_max={ng_data['id'].max():.4e} A")
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append(TestResult(
                tech=tech.name, model="ngspice_pmos", analysis="pmos_dc",
                nrmse_pct=float("nan"), error=str(e),
            ))
            continue

        model_results_for_plot: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, float]]] = {}

        # 2. BSIMAR v4 PMOS
        if checkpoints.get("bsimar_v4_pmos") is not None:
            bsimar_ckpt = checkpoints["bsimar_v4_pmos"]
            print(f"  [2/N] Running BSIMAR v4 PMOS (LEVEL=74)...")
            print(f"    Checkpoint: {bsimar_ckpt.name}")
            try:
                bsimar_data = run_pycircuitsim_nn_pmos_dc(
                    tech, work_dir, level=74,
                    model_name="bsimar_v4",
                    model_path=bsimar_ckpt,
                )
                bsimar_metrics = compare_dc_curves(
                    ng_data["sweep"], ng_data["id"],
                    bsimar_data["sweep"], bsimar_data["id"],
                )
                passed = bsimar_metrics["nrmse"] < DC_NRMSE_THRESHOLD_NN * 100
                status = "PASS" if passed else "FAIL"
                print(f"    NRMSE={bsimar_metrics['nrmse']:.2f}%  "
                      f"MRE={bsimar_metrics['mre']:.2f}%  "
                      f"|Id|_max={bsimar_metrics['max_id_test']:.4e} -> {status}")
                results.append(TestResult(
                    tech=tech.name, model="bsimar_v4_pmos", analysis="pmos_dc",
                    nrmse_pct=bsimar_metrics["nrmse"],
                    mre_pct=bsimar_metrics["mre"],
                    max_id_ref=bsimar_metrics["max_id_ref"],
                    max_id_test=bsimar_metrics["max_id_test"],
                    passed=passed,
                ))
                model_results_for_plot["bsimar_v4"] = (bsimar_data, bsimar_metrics)
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append(TestResult(
                    tech=tech.name, model="bsimar_v4_pmos", analysis="pmos_dc",
                    nrmse_pct=float("nan"), error=str(e),
                ))
        else:
            print(f"  [2/N] BSIMAR v4 PMOS -- SKIPPED (no checkpoint)")

        # 3. DirectNet v4 PMOS
        if checkpoints.get("directnet_v4_pmos") is not None:
            dnv4_ckpt = checkpoints["directnet_v4_pmos"]
            print(f"  [3/N] Running DirectNet v4 PMOS (LEVEL=73)...")
            print(f"    Checkpoint: {dnv4_ckpt.name}")
            try:
                dnv4_data = run_pycircuitsim_nn_pmos_dc(
                    tech, work_dir, level=73,
                    model_name="directnet_v4",
                    model_path=dnv4_ckpt,
                )
                dnv4_metrics = compare_dc_curves(
                    ng_data["sweep"], ng_data["id"],
                    dnv4_data["sweep"], dnv4_data["id"],
                )
                id_range = float(dnv4_data["id"].max() - dnv4_data["id"].min())
                id_max = float(dnv4_data["id"].max())
                is_broken = (id_range < 1e-6) or (id_max > 0.1)

                if is_broken:
                    print(f"    WARNING: PMOS model appears BROKEN "
                          f"(Id_range={id_range:.2e}, Id_max={id_max:.2e})")
                    results.append(TestResult(
                        tech=tech.name, model="directnet_v4_pmos",
                        analysis="pmos_dc",
                        nrmse_pct=dnv4_metrics["nrmse"],
                        mre_pct=dnv4_metrics["mre"],
                        max_id_ref=dnv4_metrics["max_id_ref"],
                        max_id_test=dnv4_metrics["max_id_test"],
                        passed=False,
                        error="BROKEN: flat/extreme output",
                    ))
                else:
                    passed = dnv4_metrics["nrmse"] < DC_NRMSE_THRESHOLD_NN * 100
                    status = "PASS" if passed else "FAIL"
                    print(f"    NRMSE={dnv4_metrics['nrmse']:.2f}%  "
                          f"MRE={dnv4_metrics['mre']:.2f}%  "
                          f"|Id|_max={dnv4_metrics['max_id_test']:.4e} -> {status}")
                    results.append(TestResult(
                        tech=tech.name, model="directnet_v4_pmos",
                        analysis="pmos_dc",
                        nrmse_pct=dnv4_metrics["nrmse"],
                        mre_pct=dnv4_metrics["mre"],
                        max_id_ref=dnv4_metrics["max_id_ref"],
                        max_id_test=dnv4_metrics["max_id_test"],
                        passed=passed,
                    ))
                model_results_for_plot["directnet_v4"] = (dnv4_data, dnv4_metrics)
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append(TestResult(
                    tech=tech.name, model="directnet_v4_pmos", analysis="pmos_dc",
                    nrmse_pct=float("nan"), error=str(e),
                ))
        else:
            print(f"  [3/N] DirectNet v4 PMOS -- SKIPPED (no checkpoint)")

        # Plot PMOS comparison
        if model_results_for_plot:
            plot_path = work_dir / f"pmos_dc_comparison_{tech.name}.png"
            plot_dc_comparison_multi(
                ng_data, model_results_for_plot, tech, plot_path,
            )
            print(f"  [Plot] Saved: {plot_path}")

    return results


# ---------------------------------------------------------------------------
# Inverter VTC test runner
# ---------------------------------------------------------------------------
def run_inverter_vtc_tests(
    tech_names: List[str],
    checkpoints: Dict[str, Optional[Path]],
) -> List[TestResult]:
    """Run inverter VTC tests. Requires both NMOS and PMOS checkpoints."""
    results: List[TestResult] = []

    for tech_name in tech_names:
        tech = ALL_TEST_TECHS[tech_name]
        if not tech.pmos_model:
            continue

        # Skip ASAP7 if tech code is out of vocabulary
        if not _tech_code_in_vocab(tech.nn_tech_key, tech.nn_vt):
            print(f"\n  Inverter VTC {tech.name} -- SKIPPED (tech code out of vocab)")
            continue

        work_dir = RESULTS_BASE / "inverter_vtc" / tech.name
        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  Inverter VTC Test: {tech.name}  "
              f"L_n={tech.l_nmos*1e9:.0f}nm  L_p={tech.effective_l_pmos*1e9:.0f}nm  "
              f"NFIN={tech.nfin}  VDD={tech.vdd:.2f}V")
        print(f"{'='*70}")

        # 1. NGSPICE ground truth
        print(f"  [1/N] Running NGSPICE inverter VTC ground truth...")
        try:
            ng_data = run_ngspice_inverter_vtc(tech, work_dir)
            print(f"    Done: {len(ng_data['sweep'])} pts, "
                  f"Vout range [{ng_data['vout'].min():.3f}, "
                  f"{ng_data['vout'].max():.3f}]V")
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append(TestResult(
                tech=tech.name, model="ngspice_vtc", analysis="vtc",
                nrmse_pct=float("nan"), error=str(e),
            ))
            continue

        model_results_for_plot: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, float]]] = {}

        # Test each NN model type that has both NMOS and PMOS checkpoints
        for model_tag, level in [("bsimar_v4", 74), ("directnet_v4", 73)]:
            nmos_key = f"{model_tag}_nmos"
            pmos_key = f"{model_tag}_pmos"
            nmos_ckpt = checkpoints.get(nmos_key)
            pmos_ckpt = checkpoints.get(pmos_key)

            if nmos_ckpt is None or pmos_ckpt is None:
                missing = []
                if nmos_ckpt is None:
                    missing.append("NMOS")
                if pmos_ckpt is None:
                    missing.append("PMOS")
                print(f"  [N/N] {model_tag} inverter VTC -- SKIPPED "
                      f"(missing {'+'.join(missing)} checkpoint)")
                continue

            print(f"  [N/N] Running {model_tag} inverter VTC (LEVEL={level})...")
            try:
                nn_data = run_pycircuitsim_nn_inverter_vtc(
                    tech, work_dir, level=level,
                    model_name=model_tag,
                    nmos_model_path=nmos_ckpt,
                    pmos_model_path=pmos_ckpt,
                )
                vtc_metrics = compare_vtc_curves(
                    ng_data["sweep"], ng_data["vout"],
                    nn_data["sweep"], nn_data["vout"],
                )
                passed = vtc_metrics["nrmse"] < VTC_NRMSE_THRESHOLD * 100
                status = "PASS" if passed else "FAIL"
                print(f"    NRMSE={vtc_metrics['nrmse']:.2f}% -> {status}")
                results.append(TestResult(
                    tech=tech.name, model=f"{model_tag}_vtc", analysis="vtc",
                    nrmse_pct=vtc_metrics["nrmse"],
                    passed=passed,
                ))
                model_results_for_plot[model_tag] = (nn_data, vtc_metrics)
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append(TestResult(
                    tech=tech.name, model=f"{model_tag}_vtc", analysis="vtc",
                    nrmse_pct=float("nan"), error=str(e),
                ))

        # Plot VTC comparison
        if model_results_for_plot:
            plot_path = work_dir / f"vtc_comparison_{tech.name}.png"
            plot_vtc_comparison_multi(
                ng_data, model_results_for_plot, tech, plot_path,
            )
            print(f"  [Plot] Saved: {plot_path}")

    return results


# ---------------------------------------------------------------------------
# Inverter transient test runner
# ---------------------------------------------------------------------------
def run_inverter_tran_tests(
    tech_names: List[str],
    checkpoints: Dict[str, Optional[Path]],
) -> List[TestResult]:
    """Run inverter transient tests. Requires both NMOS and PMOS checkpoints."""
    results: List[TestResult] = []

    for tech_name in tech_names:
        tech = ALL_TEST_TECHS[tech_name]
        if not tech.pmos_model:
            continue

        # Skip ASAP7 if tech code is out of vocabulary
        if not _tech_code_in_vocab(tech.nn_tech_key, tech.nn_vt):
            print(f"\n  Inverter tran {tech.name} -- SKIPPED (tech code out of vocab)")
            continue

        work_dir = RESULTS_BASE / "inverter_tran" / tech.name
        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  Inverter Transient Test: {tech.name}  "
              f"L_n={tech.l_nmos*1e9:.0f}nm  L_p={tech.effective_l_pmos*1e9:.0f}nm  "
              f"NFIN={tech.nfin}  Cload={INV_CLOAD*1e15:.0f}fF")
        print(f"{'='*70}")

        # 1. NGSPICE ground truth
        print(f"  [1/N] Running NGSPICE inverter transient ground truth...")
        try:
            ng_tran = run_ngspice_inverter_tran(tech, work_dir)
            print(f"    Done: {len(ng_tran['time'])} pts, "
                  f"V(out) [{ng_tran['v(out)'].min():.4f}, "
                  f"{ng_tran['v(out)'].max():.4f}]V")
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append(TestResult(
                tech=tech.name, model="ngspice_inv_tran", analysis="inv_tran",
                nrmse_pct=float("nan"), error=str(e),
            ))
            continue

        model_results_for_plot: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, float]]] = {}

        for model_tag, level in [("bsimar_v4", 74), ("directnet_v4", 73)]:
            nmos_key = f"{model_tag}_nmos"
            pmos_key = f"{model_tag}_pmos"
            nmos_ckpt = checkpoints.get(nmos_key)
            pmos_ckpt = checkpoints.get(pmos_key)

            if nmos_ckpt is None or pmos_ckpt is None:
                missing = []
                if nmos_ckpt is None:
                    missing.append("NMOS")
                if pmos_ckpt is None:
                    missing.append("PMOS")
                print(f"  [N/N] {model_tag} inverter tran -- SKIPPED "
                      f"(missing {'+'.join(missing)} checkpoint)")
                continue

            print(f"  [N/N] Running {model_tag} inverter transient (LEVEL={level})...")
            try:
                nn_tran = run_pycircuitsim_nn_inverter_tran(
                    tech, work_dir, level=level,
                    model_name=model_tag,
                    nmos_model_path=nmos_ckpt,
                    pmos_model_path=pmos_ckpt,
                )
                full_metrics = compare_inverter_tran_waveforms(
                    ng_tran, nn_tran, tech.vdd, t_start=0.0,
                )
                post_metrics = compare_inverter_tran_waveforms(
                    ng_tran, nn_tran, tech.vdd,
                    t_start=TRAN_STARTUP_EXCL,
                )

                v_range = float(nn_tran["v(out)"].max() - nn_tran["v(out)"].min())
                is_broken = v_range < 0.01

                if is_broken:
                    print(f"    WARNING: Flat inverter transient "
                          f"(range={v_range:.4f}V)")
                    results.append(TestResult(
                        tech=tech.name, model=f"{model_tag}_inv_tran",
                        analysis="inv_tran",
                        nrmse_pct=post_metrics["nrmse_vdd"],
                        passed=False,
                        error="BROKEN: flat transient output",
                    ))
                else:
                    passed = post_metrics["nrmse_vdd"] < TRAN_NRMSE_THRESHOLD * 100
                    status = "PASS" if passed else "FAIL"
                    print(f"    Full NRMSE={full_metrics['nrmse_vdd']:.2f}%  "
                          f"Post-startup NRMSE={post_metrics['nrmse_vdd']:.2f}%"
                          f" -> {status}")

                    # Per-region error breakdown
                    region = compute_region_errors(
                        ng_tran, nn_tran, tech.vdd,
                        t_start=TRAN_STARTUP_EXCL,
                    )
                    print(f"    Region breakdown:  "
                          f"High-rail={region['nrmse_high']:.2f}% "
                          f"({int(region['n_high'])}pts)  "
                          f"Low-rail={region['nrmse_low']:.2f}% "
                          f"({int(region['n_low'])}pts)  "
                          f"Transition={region['nrmse_trans']:.2f}% "
                          f"({int(region['n_trans'])}pts)")

                    results.append(TestResult(
                        tech=tech.name, model=f"{model_tag}_inv_tran",
                        analysis="inv_tran",
                        nrmse_pct=post_metrics["nrmse_vdd"],
                        passed=passed,
                        extra={"full_nrmse": full_metrics["nrmse_vdd"],
                               "max_err_mv": post_metrics["max_err_v"] * 1e3,
                               "nrmse_high": region["nrmse_high"],
                               "nrmse_low": region["nrmse_low"],
                               "nrmse_trans": region["nrmse_trans"]},
                    ))
                model_results_for_plot[model_tag] = (nn_tran, post_metrics)
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append(TestResult(
                    tech=tech.name, model=f"{model_tag}_inv_tran",
                    analysis="inv_tran",
                    nrmse_pct=float("nan"), error=str(e),
                ))

        # Plot inverter transient comparison
        if model_results_for_plot:
            plot_path = work_dir / f"inverter_tran_comparison_{tech.name}.png"
            plot_inverter_tran_comparison_multi(
                ng_tran, model_results_for_plot, tech, plot_path,
            )
            print(f"  [Plot] Saved: {plot_path}")

    return results


# ---------------------------------------------------------------------------
# Transient test runner (NMOS pulse response)
# ---------------------------------------------------------------------------
def run_tran_tests(
    tech_names: List[str],
    checkpoints: Dict[str, Optional[Path]],
) -> List[TestResult]:
    """Run transient NMOS pulse response tests for all techs and models."""
    results: List[TestResult] = []

    for tech_name in tech_names:
        tech = ALL_TEST_TECHS[tech_name]
        work_dir = RESULTS_BASE / "tran" / tech.name
        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  Transient Test: {tech.name}  L={tech.l_nmos*1e9:.0f}nm  "
              f"NFIN={tech.nfin}  Rload={TRAN_RLOAD/1e3:.0f}k")
        print(f"{'='*70}")

        # 1. NGSPICE ground truth
        print(f"  [1/N] Running NGSPICE BSIM-CMG transient...")
        try:
            ng_tran = run_ngspice_nmos_tran(tech, work_dir)
            print(f"    Done: {len(ng_tran['time'])} pts, "
                  f"V(drain) [{ng_tran['v(drain)'].min():.4f}, "
                  f"{ng_tran['v(drain)'].max():.4f}]V")
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append(TestResult(
                tech=tech.name, model="ngspice", analysis="tran",
                nrmse_pct=float("nan"), error=str(e),
            ))
            continue

        model_results_for_plot: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, float]]] = {}

        # 2. BSIMAR v4 transient (LEVEL=74, tech-code embedding)
        if checkpoints.get("bsimar_v4") is not None:
            print(f"  [2/N] Running BSIMAR v4 (LEVEL=74) transient...")
            try:
                bsimar_tran = run_pycircuitsim_nn_nmos_tran(
                    tech, work_dir, level=74,
                    model_name="bsimar_v4",
                    model_path=checkpoints["bsimar_v4"],
                )
                # Full comparison
                full_metrics = compare_tran_waveforms(
                    ng_tran, bsimar_tran, tech.vdd, t_start=0.0,
                )
                # Post-startup comparison
                post_metrics = compare_tran_waveforms(
                    ng_tran, bsimar_tran, tech.vdd,
                    t_start=TRAN_STARTUP_EXCL,
                )
                passed = post_metrics["nrmse_vdd"] < TRAN_NRMSE_THRESHOLD * 100
                status = "PASS" if passed else "FAIL"
                print(f"    Full NRMSE={full_metrics['nrmse_vdd']:.2f}%  "
                      f"Post-startup NRMSE={post_metrics['nrmse_vdd']:.2f}% -> {status}")
                results.append(TestResult(
                    tech=tech.name, model="bsimar_v4", analysis="tran",
                    nrmse_pct=post_metrics["nrmse_vdd"],
                    passed=passed,
                    extra={"full_nrmse": full_metrics["nrmse_vdd"],
                           "max_err_mv": post_metrics["max_err_v"] * 1e3},
                ))
                model_results_for_plot["bsimar_v4"] = (bsimar_tran, post_metrics)
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append(TestResult(
                    tech=tech.name, model="bsimar_v4", analysis="tran",
                    nrmse_pct=float("nan"), error=str(e),
                ))
        else:
            print(f"  [2/N] BSIMAR v4 transient -- SKIPPED")

        # 3. DirectNet v4 transient (LEVEL=73)
        if checkpoints.get("directnet_v4") is not None:
            print(f"  [3/N] Running DirectNet v4 (LEVEL=73) transient...")
            try:
                dnv4_tran = run_pycircuitsim_nn_nmos_tran(
                    tech, work_dir, level=73,
                    model_name="directnet_v4",
                    model_path=checkpoints["directnet_v4"],
                )
                full_metrics = compare_tran_waveforms(
                    ng_tran, dnv4_tran, tech.vdd, t_start=0.0,
                )
                post_metrics = compare_tran_waveforms(
                    ng_tran, dnv4_tran, tech.vdd,
                    t_start=TRAN_STARTUP_EXCL,
                )

                # Check for broken output
                v_range = float(dnv4_tran["v(drain)"].max() - dnv4_tran["v(drain)"].min())
                is_broken = v_range < 0.01

                if is_broken:
                    print(f"    WARNING: Flat transient output (range={v_range:.4f}V)")
                    results.append(TestResult(
                        tech=tech.name, model="directnet_v4", analysis="tran",
                        nrmse_pct=post_metrics["nrmse_vdd"],
                        passed=False,
                        error="BROKEN: flat transient output",
                    ))
                else:
                    passed = post_metrics["nrmse_vdd"] < TRAN_NRMSE_THRESHOLD * 100
                    status = "PASS" if passed else "FAIL"
                    print(f"    Full NRMSE={full_metrics['nrmse_vdd']:.2f}%  "
                          f"Post-startup NRMSE={post_metrics['nrmse_vdd']:.2f}% -> {status}")
                    results.append(TestResult(
                        tech=tech.name, model="directnet_v4", analysis="tran",
                        nrmse_pct=post_metrics["nrmse_vdd"],
                        passed=passed,
                        extra={"full_nrmse": full_metrics["nrmse_vdd"],
                               "max_err_mv": post_metrics["max_err_v"] * 1e3},
                    ))
                model_results_for_plot["directnet_v4"] = (dnv4_tran, post_metrics)
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append(TestResult(
                    tech=tech.name, model="directnet_v4", analysis="tran",
                    nrmse_pct=float("nan"), error=str(e),
                ))
        else:
            print(f"  [3/N] DirectNet v4 transient -- SKIPPED")

        # Plot multi-model transient comparison
        if model_results_for_plot:
            plot_path = work_dir / f"tran_comparison_{tech.name}.png"
            plot_tran_comparison_multi(
                ng_tran, model_results_for_plot, tech, plot_path,
            )
            print(f"  [Plot] Saved: {plot_path}")

    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def print_summary(dc_results: List[TestResult], tran_results: List[TestResult]) -> Tuple[int, int, int]:
    """Print formatted summary table. Returns (n_pass, n_fail, n_error)."""
    all_results = dc_results + tran_results
    if not all_results:
        print("  No results.")
        return 0, 0, 0

    print(f"\n{'='*90}")
    print("  SUMMARY TABLE")
    print(f"{'='*90}")
    header = (
        f"  {'Tech':8s} | {'Model':20s} | {'Analysis':6s} | "
        f"{'NRMSE%':>8s} | {'MRE%':>8s} | "
        f"{'Id_max_ref':>12s} | {'Id_max_test':>12s} | {'Status':>8s}"
    )
    print(header)
    print("  " + "-" * 86)

    n_pass = n_fail = n_error = 0

    for r in all_results:
        if r.error:
            n_error += 1
            status = "ERROR"
            nrmse_s = f"{r.nrmse_pct:8.2f}" if np.isfinite(r.nrmse_pct) else "    N/A"
            mre_s = "     N/A"
            ref_s = "         N/A"
            test_s = "         N/A"
            # Show error hint
            error_hint = r.error[:30] if len(r.error) > 30 else r.error
            print(
                f"  {r.tech:8s} | {r.model:20s} | {r.analysis:6s} | "
                f"{nrmse_s:>8s} | {mre_s:>8s} | "
                f"{ref_s:>12s} | {test_s:>12s} | {status:>8s}"
            )
            print(f"  {'':8s}   {'':20s}   {'':6s}   -> {error_hint}")
        elif r.passed:
            n_pass += 1
            status = "PASS"
            nrmse_s = f"{r.nrmse_pct:8.2f}"
            mre_s = f"{r.mre_pct:8.2f}" if np.isfinite(r.mre_pct) else "     N/A"
            ref_s = f"{r.max_id_ref:12.4e}" if r.max_id_ref > 0 else "         N/A"
            test_s = f"{r.max_id_test:12.4e}" if r.max_id_test > 0 else "         N/A"
            print(
                f"  {r.tech:8s} | {r.model:20s} | {r.analysis:6s} | "
                f"{nrmse_s:>8s} | {mre_s:>8s} | "
                f"{ref_s:>12s} | {test_s:>12s} | {status:>8s}"
            )
        else:
            n_fail += 1
            status = "FAIL"
            nrmse_s = f"{r.nrmse_pct:8.2f}" if np.isfinite(r.nrmse_pct) else "    N/A"
            mre_s = f"{r.mre_pct:8.2f}" if np.isfinite(r.mre_pct) else "     N/A"
            ref_s = f"{r.max_id_ref:12.4e}" if r.max_id_ref > 0 else "         N/A"
            test_s = f"{r.max_id_test:12.4e}" if r.max_id_test > 0 else "         N/A"
            print(
                f"  {r.tech:8s} | {r.model:20s} | {r.analysis:6s} | "
                f"{nrmse_s:>8s} | {mre_s:>8s} | "
                f"{ref_s:>12s} | {test_s:>12s} | {status:>8s}"
            )

    total = n_pass + n_fail + n_error
    print(f"\n  Total: {total}  Pass: {n_pass}  Fail: {n_fail}  Error: {n_error}")

    # Per-model summary
    models_seen = sorted(set(r.model for r in all_results if not r.error))
    if models_seen:
        print(f"\n  Per-model DC NRMSE averages (excluding errors):")
        for m in models_seen:
            dc_vals = [r.nrmse_pct for r in dc_results
                       if r.model == m and not r.error and np.isfinite(r.nrmse_pct)]
            if dc_vals:
                avg = np.mean(dc_vals)
                print(f"    {m:20s}: avg NRMSE = {avg:.2f}% "
                      f"(across {len(dc_vals)} techs)")

    return n_pass, n_fail, n_error


# ---------------------------------------------------------------------------
# Save summary CSV
# ---------------------------------------------------------------------------
def save_summary_csv(
    dc_results: List[TestResult],
    tran_results: List[TestResult],
    csv_path: Path,
) -> None:
    """Save results to CSV."""
    import csv
    all_results = dc_results + tran_results
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tech", "model", "analysis", "nrmse_pct", "mre_pct",
            "max_id_ref", "max_id_test", "passed", "error",
        ])
        for r in all_results:
            writer.writerow([
                r.tech, r.model, r.analysis,
                f"{r.nrmse_pct:.4f}" if np.isfinite(r.nrmse_pct) else "",
                f"{r.mre_pct:.4f}" if np.isfinite(r.mre_pct) else "",
                f"{r.max_id_ref:.6e}" if r.max_id_ref > 0 else "",
                f"{r.max_id_test:.6e}" if r.max_id_test > 0 else "",
                "PASS" if r.passed else "FAIL",
                r.error,
            ])
    print(f"  [CSV] Summary saved: {csv_path}")


# ---------------------------------------------------------------------------
# Sign pre-screen diagnostic
# ---------------------------------------------------------------------------
def _eval_nn_single_op(
    tech: TestTechConfig,
    work_dir: Path,
    level: int,
    model_name: str,
    vgs: float,
    vds: float,
    is_pmos: bool = False,
    model_path: Optional[Path] = None,
) -> float:
    """Evaluate a single NN MOSFET at one bias point. Returns raw id (A).

    Uses a 1-point DC sweep (.dc Vds vds vds 1) to extract the current.
    """
    from pycircuitsim.parser import Parser
    from pycircuitsim.simulation import run_dc_sweep
    from pycircuitsim.visualizer import Visualizer

    if is_pmos:
        l_nm = tech.effective_l_pmos * 1e9
        vt_key = tech.effective_pmos_vt
        dev_name = "Mp1"
        dev_type = "PMOS"
        inst_line = f"Mp1 1 2 0 0 nn_model L={l_nm:.0f}n NFIN={tech.nfin}"
    else:
        l_nm = tech.l_nmos * 1e9
        vt_key = tech.nn_vt
        dev_name = "Mn1"
        dev_type = "NMOS"
        inst_line = f"Mn1 1 2 0 0 nn_model L={l_nm:.0f}n NFIN={tech.nfin}"

    params = f"LEVEL={level} TECH={tech.nn_tech_key} VT={vt_key}"
    if model_path is not None:
        params += f" MODEL_PATH={model_path}"

    # Single-point sweep: sweep Vds over [vds, vds] to get one operating point
    netlist_path = work_dir / f"sign_{model_name}_{dev_type}_{vgs:.3f}_{vds:.3f}.sp"
    content = (
        f"* Sign diagnostic ({model_name}, {tech.name})\n"
        f"Vds 1 0 {vds}\n"
        f"Vgs 2 0 {vgs}\n"
        f"{inst_line}\n"
        f".model nn_model {dev_type} ({params})\n"
        f".dc Vds {vds} {vds} 1\n"
        f".end\n"
    )
    netlist_path.write_text(content)

    logging.disable(logging.CRITICAL)
    try:
        parser = Parser()
        parser.parse_file(str(netlist_path))
        circuit = parser.circuit
        vis = Visualizer()
        out_dir = work_dir / f"sign_{model_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        results = run_dc_sweep(
            circuit, parser.analysis_params, vis, out_dir,
            f"sign_{model_name}",
        )
    finally:
        logging.disable(logging.NOTSET)

    current_key = f"i({dev_name})"
    id_val = float(results[current_key][0])
    return id_val


def run_sign_diagnostic(
    tech_names: List[str],
    checkpoints: Dict[str, Optional[Path]],
) -> List[TestResult]:
    """Run sign pre-screen: evaluate NN models at subthreshold bias points.

    Checks that NMOS id <= 0 and PMOS id >= 0 at Vgs=0 for various Vds.
    Fast diagnostic to run before expensive inverter transient tests.
    """
    results: List[TestResult] = []

    print(f"\n{'='*70}")
    print("  SIGN PRE-SCREEN DIAGNOSTIC")
    print(f"{'='*70}")

    for tech_name in tech_names:
        tech = ALL_TEST_TECHS[tech_name]
        if not _tech_code_in_vocab(tech.nn_tech_key, tech.nn_vt):
            print(f"\n  {tech.name} -- SKIPPED (tech code out of vocab)")
            continue

        work_dir = RESULTS_BASE / "sign_diagnostic" / tech.name
        work_dir.mkdir(parents=True, exist_ok=True)

        # Define test points: (vgs, vds, is_pmos, expected_sign_label)
        nmos_points = [
            (0.0, 0.0,            False, "~0"),
            (0.0, 0.02,           False, "<=0"),
            (0.0, 0.05,           False, "<=0"),
            (0.0, tech.vdd * 0.5, False, "<=0"),
        ]
        pmos_points: List[Tuple[float, float, bool, str]] = []
        if tech.pmos_model and _tech_code_in_vocab(
            tech.nn_tech_key, tech.effective_pmos_vt
        ):
            pmos_points = [
                (0.0, 0.0,             True, "~0"),
                (0.0, -0.02,           True, ">=0"),
                (0.0, -tech.vdd * 0.5, True, ">=0"),
            ]

        for model_tag, level in [("bsimar_v4", 74), ("directnet_v4", 73)]:
            nmos_key = f"{model_tag}_nmos" if f"{model_tag}_nmos" in checkpoints else model_tag
            pmos_key = f"{model_tag}_pmos"
            nmos_ckpt = checkpoints.get(nmos_key) or checkpoints.get(model_tag)
            pmos_ckpt = checkpoints.get(pmos_key)

            if nmos_ckpt is None:
                continue

            print(f"\n  {tech.name} / {model_tag} (LEVEL={level}):")
            print(f"    {'Device':6s} {'Vgs':>6s} {'Vds':>8s} {'Id (A)':>12s} "
                  f"{'Expected':>8s} {'Status':>8s}")
            print(f"    {'-'*54}")

            n_sign_fail = 0

            for vgs, vds, is_pmos, expected in nmos_points:
                try:
                    id_val = _eval_nn_single_op(
                        tech, work_dir, level, model_tag,
                        vgs, vds, is_pmos=False, model_path=nmos_ckpt,
                    )
                    # NMOS: id should be <= 0 (current into drain)
                    if expected == "~0":
                        ok = abs(id_val) < 1e-6
                    else:
                        ok = id_val <= 1e-10  # allow tiny positive noise
                    status = "OK" if ok else "FAIL"
                    if not ok:
                        n_sign_fail += 1
                    print(f"    {'NMOS':6s} {vgs:6.3f} {vds:8.4f} {id_val:12.4e} "
                          f"{expected:>8s} {status:>8s}")
                except Exception as e:
                    print(f"    {'NMOS':6s} {vgs:6.3f} {vds:8.4f} {'ERROR':>12s} "
                          f"{expected:>8s} {'ERROR':>8s}  ({e})")
                    n_sign_fail += 1

            if pmos_ckpt is not None:
                for vgs, vds, is_pmos, expected in pmos_points:
                    try:
                        id_val = _eval_nn_single_op(
                            tech, work_dir, level, model_tag,
                            vgs, vds, is_pmos=True, model_path=pmos_ckpt,
                        )
                        # PMOS: id should be >= 0 (current into drain)
                        if expected == "~0":
                            ok = abs(id_val) < 1e-6
                        else:
                            ok = id_val >= -1e-10
                        status = "OK" if ok else "FAIL"
                        if not ok:
                            n_sign_fail += 1
                        print(f"    {'PMOS':6s} {vgs:6.3f} {vds:8.4f} {id_val:12.4e} "
                              f"{expected:>8s} {status:>8s}")
                    except Exception as e:
                        print(f"    {'PMOS':6s} {vgs:6.3f} {vds:8.4f} {'ERROR':>12s} "
                              f"{expected:>8s} {'ERROR':>8s}  ({e})")
                        n_sign_fail += 1

            passed = n_sign_fail == 0
            results.append(TestResult(
                tech=tech.name, model=f"{model_tag}_sign",
                analysis="sign",
                nrmse_pct=0.0 if passed else 100.0,
                passed=passed,
                error="" if passed else f"{n_sign_fail} sign violation(s)",
            ))

    return results


# ---------------------------------------------------------------------------
# Id-Vds curve diagnostic at Vgs=0
# ---------------------------------------------------------------------------
def run_ngspice_nmos_idvds(
    tech: TestTechConfig, work_dir: Path,
) -> Dict[str, np.ndarray]:
    """Run NGSPICE NMOS Id-Vds sweep at Vgs=0. Returns {sweep, id}."""
    baked = create_baked_modelcard(tech, work_dir)

    netlist_path = work_dir / f"ngspice_nmos_idvds_{tech.name}.cir"
    content = (
        f"* NMOS Id-Vds at Vgs=0 (NGSPICE, {tech.name})\n"
        f'.include "{baked}"\n'
        f".temp 27\n"
        f"Vds d 0 0.0\n"
        f"Vgs g 0 0.0\n"
        f"N1 d g 0 0 {tech.nmos_model}\n"
        f".dc Vds -0.1 {tech.vdd} 0.005\n"
        f".end\n"
    )
    netlist_path.write_text(content)

    csv_path = work_dir / f"ngspice_nmos_idvds_{tech.name}.csv"
    log_path = work_dir / f"ngspice_nmos_idvds_{tech.name}.log"
    runner_path = work_dir / f"ngspice_nmos_idvds_{tech.name}_runner.cir"
    runner_content = (
        f"* NGSPICE Id-Vds runner ({tech.name})\n"
        f".control\n"
        f"osdi {OSDI_PATH}\n"
        f"source {netlist_path}\n"
        f"set filetype=ascii\n"
        f"set wr_vecnames\n"
        f"run\n"
        f"wrdata {csv_path} i(Vds)\n"
        f".endc\n"
        f".end\n"
    )
    runner_path.write_text(runner_content)

    res = subprocess.run(
        [NGSPICE_BIN, "-b", "-o", str(log_path), str(runner_path)],
        capture_output=True, text=True,
    )
    if log_path.exists() and "Fatal:" in log_path.read_text():
        raise RuntimeError(f"NGSPICE OSDI fatal error in Id-Vds {tech.name}")
    if not csv_path.exists():
        log_text = log_path.read_text() if log_path.exists() else "(no log)"
        raise RuntimeError(f"NGSPICE Id-Vds no output: {log_text[-300:]}")

    with csv_path.open() as f:
        lines = f.readlines()
    data_rows = []
    for line in lines[1:]:
        s = line.strip()
        if s:
            data_rows.append([float(x) for x in s.split()])
    data = np.array(data_rows)
    return {"sweep": data[:, 0], "id": data[:, 1]}


def run_pycircuitsim_nn_nmos_idvds(
    tech: TestTechConfig,
    work_dir: Path,
    level: int,
    model_name: str,
    model_path: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Run PyCircuitSim NN NMOS Id-Vds at Vgs=0. Returns {sweep, id}."""
    from pycircuitsim.parser import Parser
    from pycircuitsim.simulation import run_dc_sweep
    from pycircuitsim.visualizer import Visualizer

    l_nm = tech.l_nmos * 1e9
    model_params = f"LEVEL={level} TECH={tech.nn_tech_key} VT={tech.nn_vt}"
    if model_path is not None:
        model_params += f" MODEL_PATH={model_path}"

    netlist_path = work_dir / f"nn_{model_name}_nmos_idvds_{tech.name}.sp"
    content = (
        f"* NN NMOS Id-Vds at Vgs=0 ({model_name}, {tech.name})\n"
        f"Vds 1 0 0.0\n"
        f"Vgs 2 0 0.0\n"
        f"Mn1 1 2 0 0 nmos_nn L={l_nm:.0f}n NFIN={tech.nfin}\n"
        f".model nmos_nn NMOS ({model_params})\n"
        f".dc Vds -0.1 {tech.vdd} 0.005\n"
        f".end\n"
    )
    netlist_path.write_text(content)

    logging.disable(logging.CRITICAL)
    try:
        parser = Parser()
        parser.parse_file(str(netlist_path))
        circuit = parser.circuit
        vis = Visualizer()
        out_dir = work_dir / f"{model_name}_idvds_{tech.name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        results = run_dc_sweep(
            circuit, parser.analysis_params, vis, out_dir,
            f"{model_name}_idvds_{tech.name}",
        )
    finally:
        logging.disable(logging.NOTSET)

    sweep = np.array(results["1"])
    signal = np.array(results["i(Mn1)"])
    return {"sweep": sweep, "id": signal}


def plot_idvds_diagnostic(
    ref_data: Dict[str, np.ndarray],
    model_results: Dict[str, Dict[str, np.ndarray]],
    tech: TestTechConfig,
    save_path: Path,
) -> None:
    """Plot Id-Vds at Vgs=0 for BSIM-CMG vs NN models."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    colors = {"bsimar_v4": "red", "directnet_v4": "purple"}

    # Linear scale
    ax1 = axes[0]
    ax1.plot(ref_data["sweep"], ref_data["id"], "b-", lw=2,
             label="NGSPICE BSIM-CMG")
    for mname, mdata in model_results.items():
        ax1.plot(mdata["sweep"], mdata["id"],
                 color=colors.get(mname, "gray"), linestyle="--", lw=1.5,
                 label=mname)
    ax1.axhline(y=0, color="k", lw=0.5, ls=":")
    ax1.axvline(x=0, color="k", lw=0.5, ls=":")
    ax1.set_xlabel("Vds (V)")
    ax1.set_ylabel("Id (A)  [raw, with sign]")
    ax1.set_title(
        f"NMOS Id-Vds at Vgs=0: {tech.name}  "
        f"L={tech.l_nmos*1e9:.0f}nm  NFIN={tech.nfin}"
    )
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Zoomed near Vds=0
    ax2 = axes[1]
    zoom_range = 0.15
    ax2.plot(ref_data["sweep"], ref_data["id"], "b-", lw=2,
             label="NGSPICE BSIM-CMG")
    for mname, mdata in model_results.items():
        ax2.plot(mdata["sweep"], mdata["id"],
                 color=colors.get(mname, "gray"), linestyle="--", lw=1.5,
                 label=mname)
    ax2.axhline(y=0, color="k", lw=0.5, ls=":")
    ax2.axvline(x=0, color="k", lw=0.5, ls=":")
    ax2.set_xlim(-zoom_range, zoom_range)
    ax2.set_xlabel("Vds (V)")
    ax2.set_ylabel("Id (A)  [raw, with sign]")
    ax2.set_title("Zoomed near Vds=0 (boundary behavior)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_idvds_diagnostic(
    tech_names: List[str],
    checkpoints: Dict[str, Optional[Path]],
) -> List[TestResult]:
    """Run Id-Vds sweep at Vgs=0 to visualize Vds=0 boundary and wrong-sign.

    For each tech: NGSPICE ground truth + available NN models.
    Generates comparison plots.
    """
    results: List[TestResult] = []

    print(f"\n{'='*70}")
    print("  Id-Vds DIAGNOSTIC (Vgs=0, subthreshold)")
    print(f"{'='*70}")

    for tech_name in tech_names:
        tech = ALL_TEST_TECHS[tech_name]
        if not _tech_code_in_vocab(tech.nn_tech_key, tech.nn_vt):
            print(f"\n  {tech.name} -- SKIPPED (tech code out of vocab)")
            continue

        work_dir = RESULTS_BASE / "idvds_diagnostic" / tech.name
        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  {tech.name}: NMOS Id-Vds at Vgs=0  "
              f"L={tech.l_nmos*1e9:.0f}nm  NFIN={tech.nfin}")

        # 1. NGSPICE ground truth
        print(f"    Running NGSPICE ground truth...")
        try:
            ng_data = run_ngspice_nmos_idvds(tech, work_dir)
            print(f"    NGSPICE: {len(ng_data['sweep'])} pts, "
                  f"Id range [{ng_data['id'].min():.4e}, "
                  f"{ng_data['id'].max():.4e}]")
        except Exception as e:
            print(f"    NGSPICE ERROR: {e}")
            continue

        nn_results: Dict[str, Dict[str, np.ndarray]] = {}

        for model_tag, level in [("bsimar_v4", 74), ("directnet_v4", 73)]:
            nmos_key = (f"{model_tag}_nmos"
                        if f"{model_tag}_nmos" in checkpoints
                        else model_tag)
            ckpt = checkpoints.get(nmos_key) or checkpoints.get(model_tag)
            if ckpt is None:
                continue

            print(f"    Running {model_tag} (LEVEL={level})...")
            try:
                nn_data = run_pycircuitsim_nn_nmos_idvds(
                    tech, work_dir, level, model_tag, model_path=ckpt,
                )
                nn_results[model_tag] = nn_data

                # Check for wrong-sign: positive Id at positive Vds
                pos_vds_mask = nn_data["sweep"] > 0.01
                if pos_vds_mask.any():
                    wrong_sign_count = int(
                        (nn_data["id"][pos_vds_mask] > 1e-10).sum()
                    )
                    total_pos = int(pos_vds_mask.sum())
                    print(f"    {model_tag}: Id range "
                          f"[{nn_data['id'].min():.4e}, "
                          f"{nn_data['id'].max():.4e}], "
                          f"wrong-sign={wrong_sign_count}/{total_pos}")
                    passed = wrong_sign_count == 0
                else:
                    passed = True

                results.append(TestResult(
                    tech=tech.name, model=f"{model_tag}_idvds",
                    analysis="idvds",
                    nrmse_pct=0.0 if passed else 100.0,
                    passed=passed,
                    error="" if passed else "wrong-sign Id at Vds>0",
                ))
            except Exception as e:
                print(f"    {model_tag} ERROR: {e}")
                results.append(TestResult(
                    tech=tech.name, model=f"{model_tag}_idvds",
                    analysis="idvds",
                    nrmse_pct=float("nan"), error=str(e),
                ))

        # Plot
        if nn_results:
            plot_path = work_dir / f"idvds_diagnostic_{tech.name}.png"
            plot_idvds_diagnostic(ng_data, nn_results, tech, plot_path)
            print(f"    [Plot] Saved: {plot_path}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="NN compact model verification: BSIMAR + DirectNet vs BSIM-CMG",
    )
    parser.add_argument(
        "--tech", type=str, default=",".join(TECH_ORDER),
        help="Comma-separated tech names (default: all)",
    )
    parser.add_argument(
        "--dc-only", action="store_true",
        help="Run NMOS DC tests only",
    )
    parser.add_argument(
        "--tran-only", action="store_true",
        help="Run NMOS transient tests only",
    )
    parser.add_argument(
        "--pmos-only", action="store_true",
        help="Run PMOS DC tests only",
    )
    parser.add_argument(
        "--inverter-only", action="store_true",
        help="Run inverter tests only (VTC + transient)",
    )
    parser.add_argument(
        "--sign-diagnostic", action="store_true",
        help="Run sign pre-screen diagnostic (Vgs=0 bias points)",
    )
    parser.add_argument(
        "--idvds-diagnostic", action="store_true",
        help="Run Id-Vds curve diagnostic at Vgs=0",
    )
    args = parser.parse_args()

    tech_names = [t.strip() for t in args.tech.split(",")]
    for t in tech_names:
        if t not in ALL_TEST_TECHS:
            print(f"ERROR: Unknown tech '{t}'. "
                  f"Available: {list(ALL_TEST_TECHS.keys())}")
            return 1

    # Determine which test suites to run
    explicit_filter = (args.dc_only or args.tran_only or
                       args.pmos_only or args.inverter_only or
                       args.sign_diagnostic or args.idvds_diagnostic)
    run_nmos_dc = args.dc_only or not explicit_filter
    run_nmos_tran = args.tran_only or not explicit_filter
    run_pmos_dc = args.pmos_only or not explicit_filter
    run_inverter_vtc = args.inverter_only or not explicit_filter
    run_inverter_tran = args.inverter_only or not explicit_filter
    run_sign_diag = args.sign_diagnostic
    run_idvds_diag = args.idvds_diagnostic

    # Create results directory
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)

    # Check available checkpoints
    checkpoints = get_available_checkpoints()

    print("=" * 70)
    print("  NN Compact Model Verification")
    print("  BSIMAR (LEVEL=74) + DirectNet (LEVEL=73) vs BSIM-CMG (LEVEL=72)")
    print("=" * 70)
    print(f"\n  Technologies: {', '.join(tech_names)}")
    print(f"  Suites: NMOS_DC={run_nmos_dc}  PMOS_DC={run_pmos_dc}  "
          f"INV_VTC={run_inverter_vtc}  NMOS_TRAN={run_nmos_tran}  "
          f"INV_TRAN={run_inverter_tran}  "
          f"SIGN_DIAG={run_sign_diag}  IDVDS_DIAG={run_idvds_diag}")
    print(f"\n  Checkpoint availability:")
    # Show non-alias keys only
    for name, path in checkpoints.items():
        if name in ("bsimar_v4", "directnet_v4"):
            continue  # skip backward-compat aliases
        if path is not None:
            print(f"    {name:20s}: {path.name}")
        else:
            print(f"    {name:20s}: NOT FOUND")

    if all(v is None for v in checkpoints.values()):
        print("\n  ERROR: No NN checkpoints found. Nothing to test.")
        return 1

    print(f"\n  DC acceptance: NRMSE < {DC_NRMSE_THRESHOLD_NN*100:.0f}% (NN), "
          f"< {DC_NRMSE_THRESHOLD_CMG*100:.0f}% (CMG)")
    print(f"  VTC acceptance: NRMSE < {VTC_NRMSE_THRESHOLD*100:.0f}%")
    print(f"  Tran acceptance: NRMSE < {TRAN_NRMSE_THRESHOLD*100:.0f}% of Vdd")

    # Run tests
    dc_results: List[TestResult] = []
    tran_results: List[TestResult] = []

    if run_nmos_dc:
        dc_results.extend(run_dc_tests(tech_names, checkpoints))

    if run_pmos_dc:
        dc_results.extend(run_pmos_dc_tests(tech_names, checkpoints))

    if run_inverter_vtc:
        dc_results.extend(run_inverter_vtc_tests(tech_names, checkpoints))

    if run_nmos_tran:
        tran_results.extend(run_tran_tests(tech_names, checkpoints))

    if run_inverter_tran:
        tran_results.extend(run_inverter_tran_tests(tech_names, checkpoints))

    # Diagnostics
    diag_results: List[TestResult] = []

    if run_sign_diag:
        diag_results.extend(run_sign_diagnostic(tech_names, checkpoints))

    if run_idvds_diag:
        diag_results.extend(run_idvds_diagnostic(tech_names, checkpoints))

    # Summary (include diagnostics in tran_results for display)
    all_tran_and_diag = tran_results + diag_results
    n_pass, n_fail, n_error = print_summary(dc_results, all_tran_and_diag)

    # Save CSV
    csv_path = RESULTS_BASE / "summary.csv"
    save_summary_csv(dc_results, all_tran_and_diag, csv_path)

    # Final verdict
    total = n_pass + n_fail + n_error
    print(f"\n{'='*70}")

    if n_fail > 0 or n_error > 0:
        print(f"  RESULT: {n_pass} PASS, {n_fail} FAIL, {n_error} ERROR "
              f"out of {total} tests")
        return 1
    else:
        print(f"  RESULT: ALL {n_pass} tests PASSED out of {total}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
