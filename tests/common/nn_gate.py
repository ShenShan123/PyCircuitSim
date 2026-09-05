#!/usr/bin/env python3
"""
Full-terminal NN compact-model verification against LEVEL=72 ground truth.

.. note:: Run with ``PYTHONUNBUFFERED=1`` to see live progress output.

Tests NMOS DC sweep, PMOS DC sweep, inverter VTC, NMOS pulse-response
transient, and inverter transient across available technologies.
Compares against BSIM-CMG NGSPICE ground truth.

Strategy:
  1. BSIM-CMG (LEVEL=72) via NGSPICE as ground truth
  2. BSIM-CMG (LEVEL=72) via PyCircuitSim (sanity check)
  3. BSIM-AR-Full (LEVEL=76) via PyCircuitSim
  4. DirectNet-Full (LEVEL=75) via PyCircuitSim

Metrics: NRMSE (%) and MRE (%).

Usage:
    conda run -n pycircuitsim python tests/single_devices/verify_nn_dc.py
    conda run -n pycircuitsim python tests/single_devices/verify_nn_dc.py --tech TSMC5
    conda run -n pycircuitsim python tests/single_devices/verify_nn_dc.py --pmos-only
    conda run -n pycircuitsim python tests/simple_circuits/verify_nn_inverter.py
    conda run -n pycircuitsim python tests/simple_circuits/verify_nn_inverter.py --vtc-only

This module is the shared body, not an entry point — run one of the two gate
scripts above. `RESULTS_BASE` keeps its generated evidence below `results/`
name so the accuracy reports' raw evidence stays where they pinned it.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
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
# Path bootstrap — script-mode needs PROJECT_ROOT on sys.path so the
# ``tests.common.nn`` import below resolves; ``tests.common.nn`` itself
# appends the bsimar / pycmg directories.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.nn import nrmse, mre, tech_code_in_vocab  # noqa: E402
from tests.common.base import (  # noqa: E402
    DEVICE_DECKS, deck_tokens, template_deck, render_deck_text, render_template,
)
from tests.common.circuit_benchmarks import (  # noqa: E402
    active_model_label, active_model_level, nn_model_parameters,
)
from helpers import bake_inst_params  # noqa: E402
from neural_network.config import CHECKPOINT_DIR, OSDI_PATH  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NGSPICE_BIN = os.environ.get(
    "NGSPICE_BIN", "/usr/local/ngspice-45.2/bin/ngspice")
MODELCARDS_DIR = (
    PROJECT_ROOT / "PDKs"
)
RESULTS_BASE = PROJECT_ROOT / "results" / "tests" / "nn_dc_tran"

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


def _render_mosfet_deck(
    *, model_setup: str, temperature_c: float, drain_bias: str,
    gate_bias: str, source_bias: str, bulk_bias: str, device_name: str,
    nodes: Tuple[str, str, str, str], device: str, load: str, analysis: str,
) -> str:
    """Render one four-terminal characterization deck from the shared source."""
    drain, gate, source, bulk = nodes
    return render_template(DEVICE_DECKS / "mosfet.spice.tmpl", {
        "MODEL_SETUP": model_setup, "TEMP": f"{temperature_c:g}",
        "DRAIN_BIAS": drain_bias, "GATE_BIAS": gate_bias,
        "SOURCE_BIAS": source_bias, "BULK_BIAS": bulk_bias,
        "DEVICE_NAME": device_name, "DRAIN_NODE": drain,
        "GATE_NODE": gate, "SOURCE_NODE": source, "BULK_NODE": bulk,
        "DEVICE": device, "EXTRA_DEVICES": "", "LOAD": load,
        "ANALYSIS": analysis,
    })
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


def _render_inverter_deck(values: Dict[str, str]) -> str:
    """Render the authoritative inverter template with strict tokens."""
    path = template_deck("inverter.spice.tmpl")
    template = path.read_text()
    required = deck_tokens(template)
    missing = [name for name in required if name not in values]
    if missing:
        raise KeyError(f"{path.name}: missing substitutions {missing}")
    return render_deck_text(
        template, {name: values[name] for name in required},
        source_name=path.name,
    )


@dataclass(frozen=True)
class InvCircuitParams:
    """Per-config circuit/timing overrides for the inverter transient runners.

    Every field defaults to the module-level inverter-transient constant, so
    ``circuit=None`` (or a bare ``InvCircuitParams()``) reproduces the legacy
    fixed-point behaviour byte-for-byte. The V6.3.2 parametric harness
    (``tests/common/nn_sweep.py``) populates these for the Cload / input-slew /
    pulse-width sweeps; device geometry, VDD and VT instead ride on
    ``dataclasses.replace(tech, ...)`` of ``TestTechConfig``.
    """
    cload: float = INV_CLOAD
    tr: float = INV_TRAN_TR
    tf: float = INV_TRAN_TF
    pw: float = INV_TRAN_PW
    td: float = INV_TRAN_TD
    tstop: float = INV_TRAN_TSTOP


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
    temperature_c: float = 27.0
    dc_vds_scale: float = 0.5
    dc_vbs: float = 0.0

    # Inverter-specific geometry. 0 = fall back to single-device l_nmos / effective_l_pmos / nfin.
    # Use these to align the inverter test point with the per-tech NN training bins
    # so the verifier doesn't ask the model to extrapolate. Per-tech V4 NMOS L bins:
    #   TSMC5  : {6, 20, 36, 54, 86} nm
    #   TSMC7  : {8, 11, 20, 36, 72, 120} nm
    #   TSMC12 : {16, 20, 36, 72, 120} nm
    #   TSMC16 : {16, 20, 36, 72, 120} nm
    inv_l_nmos: float = 0.0
    inv_l_pmos: float = 0.0
    inv_nfin: int = 0
    # P-side inverter fin count. 0 = symmetric (fall back to inv_nfin). The
    # V6.3.2 P/N-ratio sweep sets this so the PMOS pull-up can differ from the
    # NMOS pull-down without disturbing inv_nfin (the NMOS fin count).
    inv_nfin_p: int = 0

    @property
    def effective_l_pmos(self) -> float:
        """PMOS L, defaulting to l_nmos if not set."""
        return self.l_pmos if self.l_pmos > 0 else self.l_nmos

    @property
    def effective_pmos_vt(self) -> str:
        """PMOS VT, defaulting to NMOS VT if not set."""
        return self.nn_pmos_vt if self.nn_pmos_vt else self.nn_vt

    @property
    def effective_inv_l_nmos(self) -> float:
        return self.inv_l_nmos if self.inv_l_nmos > 0 else self.l_nmos

    @property
    def effective_inv_l_pmos(self) -> float:
        return self.inv_l_pmos if self.inv_l_pmos > 0 else self.effective_l_pmos

    @property
    def effective_inv_nfin(self) -> int:
        return self.inv_nfin if self.inv_nfin > 0 else self.nfin

    @property
    def effective_inv_nfin_p(self) -> int:
        """PMOS inverter fin count, defaulting to the symmetric inverter NFIN."""
        return self.inv_nfin_p if self.inv_nfin_p > 0 else self.effective_inv_nfin


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
        inv_l_nmos=20e-9, inv_l_pmos=20e-9, inv_nfin=2,
    ),
    # TSMC6 (CLN6) — sister node to TSMC7 (V6.9.0 onboarding). Same vdd/geometry
    # family; ulvt is the local-vocab default (all 3 VTs pass, unlike TSMC7).
    "TSMC6": TestTechConfig(
        name="TSMC6", vdd=0.75, l_nmos=16e-9, nfin=2, tfin=6e-9,
        nmos_model="nch_ulvt_mac", nn_tech_key="tsmc6", nn_vt="ulvt",
        single_file=False, modelcard_dir="TSMC6",
        modelcard_file="",
        l_pmos=20e-9, pmos_model="pch_ulvt_mac",
        inv_l_nmos=20e-9, inv_l_pmos=20e-9, inv_nfin=2,
    ),
    "TSMC7": TestTechConfig(
        name="TSMC7", vdd=0.75, l_nmos=16e-9, nfin=2, tfin=6e-9,
        nmos_model="nch_ulvt_mac", nn_tech_key="tsmc7", nn_vt="ulvt",
        single_file=False, modelcard_dir="TSMC7",
        modelcard_file="",
        l_pmos=20e-9, pmos_model="pch_ulvt_mac",
        inv_l_nmos=20e-9, inv_l_pmos=20e-9, inv_nfin=2,
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

TECH_ORDER: List[str] = ["ASAP7", "ASAP7_30nm", "TSMC5", "TSMC6", "TSMC7", "TSMC12", "TSMC16"]

TECH_COLORS: Dict[str, str] = {
    "ASAP7": "tab:blue",
    "TSMC5": "tab:green",
    "TSMC6": "tab:brown",
    "TSMC7": "tab:orange",
    "TSMC12": "tab:purple",
    "TSMC16": "tab:red",
}


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


def resolve_nmos_inv_modelcard(tech: TestTechConfig) -> Path:
    """Resolve the NMOS modelcard path for the inverter geometry."""
    if tech.single_file:
        return MODELCARDS_DIR / tech.modelcard_dir / tech.modelcard_file

    from pycmg.tech import TECH_REGISTRY, resolve_modelcard
    tech_config = TECH_REGISTRY[_registry_name(tech)]
    prefix = tech.nmos_model.split("_", 1)[0]
    vt = tech.nmos_model.split("_", 1)[1].replace("_mac", "")
    canonical = ("nmos_" if prefix == "nch" else "pmos_") + vt
    device_config = tech_config.get_device(canonical)
    return Path(resolve_modelcard(
        device_config, tech_config,
        L=tech.effective_inv_l_nmos, NFIN=float(tech.effective_inv_nfin),
    ))


def resolve_pmos_inv_modelcard(tech: TestTechConfig) -> Path:
    """Resolve the PMOS modelcard path for the inverter geometry."""
    if tech.single_file:
        return MODELCARDS_DIR / tech.modelcard_dir / tech.modelcard_file

    from pycmg.tech import TECH_REGISTRY, resolve_modelcard
    tech_config = TECH_REGISTRY[_registry_name(tech)]
    prefix = tech.pmos_model.split("_", 1)[0]
    vt = tech.pmos_model.split("_", 1)[1].replace("_mac", "")
    canonical = ("pmos_" if prefix == "pch" else "nmos_") + vt
    device_config = tech_config.get_device(canonical)
    return Path(resolve_modelcard(
        device_config, tech_config,
        L=tech.effective_inv_l_pmos, NFIN=float(tech.effective_inv_nfin_p),
    ))


def create_baked_inv_nmos_modelcard(tech: TestTechConfig, work_dir: Path) -> Path:
    """Create baked NMOS modelcard with inverter L/NFIN/TFIN/DEVTYPE for NGSPICE."""
    src = resolve_nmos_inv_modelcard(tech)
    if not src.exists():
        raise FileNotFoundError(f"NMOS modelcard not found: {src}")

    baked = work_dir / f"baked_inv_nmos_{tech.name}.lib"
    baked.write_text(src.read_text())
    bake_inst_params(baked, baked, tech.nmos_model, {
        "L": tech.effective_inv_l_nmos,
        "NFIN": float(tech.effective_inv_nfin),
        "TFIN": tech.tfin,
        "DEVTYPE": 1,
    })
    return baked


def create_baked_inv_pmos_modelcard(tech: TestTechConfig, work_dir: Path) -> Path:
    """Create baked PMOS modelcard with inverter L/NFIN/TFIN/DEVTYPE for NGSPICE."""
    src = resolve_pmos_inv_modelcard(tech)
    if not src.exists():
        raise FileNotFoundError(f"PMOS modelcard not found: {src}")

    baked = work_dir / f"baked_inv_pmos_{tech.name}.lib"
    baked.write_text(src.read_text())
    bake_inst_params(baked, baked, tech.pmos_model, {
        "L": tech.effective_inv_l_pmos,
        "NFIN": float(tech.effective_inv_nfin_p),
        "TFIN": tech.tfin,
        "DEVTYPE": 0,
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
def _env_pin(var_names: Tuple[str, ...]) -> Tuple[Optional[str], str]:
    """First non-empty env var among ``var_names`` -> (stem, var name).

    Mirrors the ``a or b or c`` precedence the callers used before, but keeps
    the *name* of the variable that won so a bad pin can be reported against
    the knob the operator actually turned (audit B5d).
    """
    for name in var_names:
        value = os.environ.get(name)
        if value:
            return value, name
    return None, ""


def _require_pinned_checkpoint(
    stem: str, var_name: str, *, transformer: bool,
) -> Path:
    """Resolve a pin only when its complete runtime bundle exists.

    Returning ``None`` here was
    silently green — every consumer reads ``None`` as "this arm is not
    configured", skips it WITHOUT appending a TestResult row, and the run
    still exits 0 on the other polarity's rows. A pinned stem never falls
    back, and the
    parser never gets a chance to enforce it here because no model is
    instantiated on the skipped path.
    """
    model = CHECKPOINT_DIR / f"{stem}_best.pt"
    missing = _missing_bundle_artifacts(model, transformer=transformer)
    if missing:
        raise FileNotFoundError(
            f"{var_name} pins incomplete checkpoint bundle '{stem}': "
            + ", ".join(str(path) for path in missing)
        )
    return model


def _missing_bundle_artifacts(
    model: Path, *, transformer: bool,
) -> list[Path]:
    """Return missing files for one full-terminal runtime bundle."""
    stem = model.with_name(model.name.removesuffix("_best.pt"))
    required = [
        model,
        stem.with_name(stem.name + "_norm.npz"),
        model.with_name(model.name + ".complete"),
    ]
    if transformer:
        required.append(stem.with_name(stem.name + "_config.npz"))
    return [path for path in required if not path.is_file()]


#: Campaign family -> (result key, env-pin tag, checkpoint stem tag).
_FAMILY_BY_LEVEL: Dict[int, Tuple[str, str, str]] = {
    75: ("directnet_full", "DNF", "dnf"),
    76: ("bsimar_full", "TFF", "tff"),
}


def get_available_checkpoints() -> Dict[str, Optional[Path]]:
    """Return the SELECTED family's checkpoints, honoring explicit pins.

    One gate answers one question, so only the family named by
    ``PYCIRCUITSIM_NN_FORCE_LEVEL`` (default LEVEL=75) is resolved. Resolving
    both would fold an unpinned checkpoint of the other family into this run's
    verdict, its provenance, and — for LEVEL=76 — its ~40x inference cost. That
    used to be harmless only because the second arm's fallback stems named
    retired artifacts; V7.7.0 pointed those fallbacks at the live production
    slots, so on a box carrying both families it silently ran both.

    Keys for the unselected family are absent; every consumer reads through
    ``.get`` and skips a family it cannot resolve.
    """
    level = active_model_level()
    key, tag, stem_tag = _FAMILY_BY_LEVEL[level]
    transformer = level == 76
    checkpoints: Dict[str, Optional[Path]] = {}

    for dev in ("nmos", "pmos"):
        override, variable = _env_pin((
            f"PYCIRCUITSIM_NN_CHECKPOINT_{tag}_{dev.upper()}",
            f"PYCIRCUITSIM_NN_CHECKPOINT_{dev.upper()}",
            "PYCIRCUITSIM_NN_CHECKPOINT_OVERRIDE",
        ))
        if override:
            checkpoints[f"{key}_{dev}"] = _require_pinned_checkpoint(
                override, variable, transformer=transformer,
            )
            continue
        candidates = [
            CHECKPOINT_DIR / f"{tech}_{stem_tag}_{size}_{dev}_best.pt"
            for size in ("large", "medium", "small", "xl")
            for tech in ("tsmc5", "tsmc6", "tsmc7", "tsmc12", "tsmc16")
        ] + [
            CHECKPOINT_DIR / f"refac_{stem_tag}_{size}_{dev}_best.pt"
            for size in ("large", "medium", "small", "xl")
        ]
        checkpoints[f"{key}_{dev}"] = next(
            (
                path for path in candidates
                if not _missing_bundle_artifacts(
                    path, transformer=transformer,
                )
            ),
            None,
        )

    checkpoints[key] = checkpoints[f"{key}_nmos"]

    return checkpoints


# ---------------------------------------------------------------------------
# NGSPICE NMOS DC runner (ground truth)
# ---------------------------------------------------------------------------
def run_ngspice_nmos_dc(
    tech: TestTechConfig, work_dir: Path,
) -> Dict[str, np.ndarray]:
    """Run NGSPICE NMOS Id-Vgs DC sweep. Returns {sweep, id}."""
    baked = create_baked_modelcard(tech, work_dir)
    vds_bias = round(tech.vdd * tech.dc_vds_scale, 4)

    # Netlist
    netlist_path = work_dir / f"ngspice_nmos_dc_{tech.name}.cir"
    netlist_content = _render_mosfet_deck(
        model_setup=f'.include "{baked}"',
        temperature_c=tech.temperature_c,
        drain_bias=f"Vds d 0 {vds_bias}", gate_bias="Vgs g 0 0",
        source_bias="", bulk_bias=f"Vbs b 0 {tech.dc_vbs:g}",
        device_name="N1", nodes=("d", "g", "0", "b"),
        device=tech.nmos_model, load="",
        analysis=f".dc Vgs 0 {tech.vdd:g} 0.005",
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

    # NGSPICE reports current entering Vds; PyCircuitSim reports current
    # leaving the drain. Convert conventions without discarding sign.
    return {"sweep": data[:, 0], "id": -data[:, 1]}


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

    vds_bias = round(tech.vdd * tech.dc_vds_scale, 4)
    l_nm = tech.l_nmos * 1e9

    netlist_path = work_dir / f"pycircuitsim_cmg_nmos_dc_{tech.name}.sp"
    content = _render_mosfet_deck(
        model_setup=f".model {tech.nmos_model} NMOS (LEVEL=72)",
        temperature_c=27.0, drain_bias=f"Vds 1 0 {vds_bias}",
        gate_bias="Vgs 2 0 0", source_bias="", bulk_bias="",
        device_name="Mn1", nodes=("1", "2", "0", "0"),
        device=(
            f"{tech.nmos_model} L={l_nm:.0f}n NFIN={tech.nfin} "
            f"TFIN={tech.tfin * 1e9:.1f}n"
        ),
        load="", analysis=f".dc Vgs 0 {tech.vdd:g} 0.005",
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
            require_convergence=True,
        )
    finally:
        logging.disable(logging.NOTSET)

    sweep = np.array(results["2"])
    signal = np.abs(np.array(results["i(Mn1)"]))
    return {"sweep": sweep, "id": signal}


# ---------------------------------------------------------------------------
# PyCircuitSim NN NMOS DC — BSIMAR (LEVEL=76) or DirectNet (LEVEL=75)
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
        level: 75 (DirectNet-Full) or 76 (BSIM-AR-Full)
        model_name: label for this model variant (e.g. "bsimar_full")
        model_path: explicit checkpoint path (used for DirectNet-Full MODEL_PATH)
    """
    from pycircuitsim.parser import Parser
    from pycircuitsim.simulation import run_dc_sweep
    from pycircuitsim.visualizer import Visualizer

    vds_bias = round(tech.vdd * tech.dc_vds_scale, 4)
    l_nm = tech.l_nmos * 1e9

    netlist_path = work_dir / f"nn_{model_name}_nmos_dc_{tech.name}.sp"

    # Build model params string
    model_params = nn_model_parameters(
        level, tech.nn_tech_key, tech.nn_vt,
    )
    # Omit MODEL_PATH for stems the parser's per-tech preempt cascade handles
    # (tsmc{5,7,12,16}_dnf_* / refac_dnf_*) so each tech resolves its own
    # checkpoint from TECH= instead of being pinned to whichever single tech the
    # `directnet_full` alias happened to resolve to (bug report B1). A genuine
    # env pin still wins (the parser reads it before the preempt).
    if model_path is not None and not _cascade_handles_stem(model_path):
        model_params += f" MODEL_PATH={model_path}"

    content = _render_mosfet_deck(
        model_setup=f".model nmos_nn NMOS ({model_params})",
        temperature_c=tech.temperature_c,
        drain_bias=f"Vds 1 0 {vds_bias}", gate_bias="Vgs 2 0 0",
        source_bias="", bulk_bias=f"Vbs 3 0 {tech.dc_vbs:g}",
        device_name="Mn1", nodes=("1", "2", "0", "3"),
        device=f"nmos_nn L={l_nm:.0f}n NFIN={tech.nfin}", load="",
        analysis=f".dc Vgs 0 {tech.vdd:g} 0.005",
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
            require_convergence=True,
        )
    finally:
        logging.disable(logging.NOTSET)

    sweep = np.array(results["2"])
    signal = np.array(results["i(Mn1)"])
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
    content = _render_mosfet_deck(
        model_setup=f'.include "{baked}"', temperature_c=27.0,
        drain_bias=f"Vdd vdd 0 {tech.vdd:g}",
        gate_bias=(
            f"Vgs gate 0 PULSE(0 {tech.vdd:g} {TRAN_TD:g} {TRAN_TR:g} "
            f"{TRAN_TF:g} {TRAN_PW:g} {per:g})"
        ),
        source_bias="", bulk_bias="", device_name="N1",
        nodes=("drain", "gate", "0", "0"), device=tech.nmos_model,
        load=(
            f"Rload vdd drain {TRAN_RLOAD:g}\n"
            f".ic V(drain)={tech.vdd:g}"
        ),
        analysis=f".tran {TRAN_TSTEP:g} {TRAN_TSTOP:g} uic",
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

    model_params = nn_model_parameters(
        level, tech.nn_tech_key, tech.nn_vt,
    )
    # Omit MODEL_PATH for stems the parser's per-tech preempt cascade handles
    # (tsmc{5,7,12,16}_dnf_* / refac_dnf_*) so each tech resolves its own
    # checkpoint from TECH= instead of being pinned to whichever single tech the
    # `directnet_full` alias happened to resolve to (bug report B1). A genuine
    # env pin still wins (the parser reads it before the preempt).
    if model_path is not None and not _cascade_handles_stem(model_path):
        model_params += f" MODEL_PATH={model_path}"

    netlist_path = work_dir / f"nn_{model_name}_nmos_tran_{tech.name}.sp"
    content = _render_mosfet_deck(
        model_setup=f".model nmos_nn NMOS ({model_params})",
        temperature_c=27.0, drain_bias=f"Vdd 1 0 {tech.vdd:g}",
        gate_bias=(
            f"Vgs 2 0 PULSE 0 {tech.vdd:g} {TRAN_TD:g} {TRAN_TR:g} "
            f"{TRAN_TF:g} {TRAN_PW:g} {per:g}"
        ),
        source_bias="", bulk_bias="", device_name="Mn1",
        nodes=("3", "2", "0", "0"),
        device=f"nmos_nn L={l_nm:.0f}n NFIN={tech.nfin}",
        load=f"Rload 1 3 {TRAN_RLOAD:g}\n.ic V(3)={tech.vdd:g}",
        analysis=f".tran {TRAN_TSTEP:g} {TRAN_TSTOP:g}",
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
    vds_bias = round(-tech.vdd * tech.dc_vds_scale, 4)

    netlist_path = work_dir / f"ngspice_pmos_dc_{tech.name}.cir"
    netlist_content = _render_mosfet_deck(
        model_setup=f'.include "{baked}"',
        temperature_c=tech.temperature_c,
        drain_bias=f"Vds d 0 {vds_bias}", gate_bias="Vgs g 0 0",
        source_bias="", bulk_bias=f"Vbs b 0 {tech.dc_vbs:g}",
        device_name="N1", nodes=("d", "g", "0", "b"),
        device=tech.pmos_model, load="",
        analysis=f".dc Vgs 0 {-tech.vdd:g} -0.005",
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

    # Use |Vgs| for an ascending interpolation axis; retain signed current.
    sweep = np.abs(data[:, 0])
    # Sort by ascending |Vgs| for consistent interpolation
    sort_idx = np.argsort(sweep)
    return {"sweep": sweep[sort_idx], "id": data[sort_idx, 1]}


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

    vds_bias = round(-tech.vdd * tech.dc_vds_scale, 4)
    l_nm = tech.effective_l_pmos * 1e9

    netlist_path = work_dir / f"nn_{model_name}_pmos_dc_{tech.name}.sp"

    model_params = nn_model_parameters(
        level, tech.nn_tech_key, tech.effective_pmos_vt,
    )
    # Omit MODEL_PATH for stems the parser's per-tech preempt cascade handles
    # (tsmc{5,7,12,16}_dnf_* / refac_dnf_*) so each tech resolves its own
    # checkpoint from TECH= instead of being pinned to whichever single tech the
    # `directnet_full` alias happened to resolve to (bug report B1). A genuine
    # env pin still wins (the parser reads it before the preempt).
    if model_path is not None and not _cascade_handles_stem(model_path):
        model_params += f" MODEL_PATH={model_path}"

    content = _render_mosfet_deck(
        model_setup=f".model pmos_nn PMOS ({model_params})",
        temperature_c=tech.temperature_c,
        drain_bias=f"Vds 1 0 {vds_bias}", gate_bias="Vgs 2 0 0",
        source_bias="", bulk_bias=f"Vbs 3 0 {tech.dc_vbs:g}",
        device_name="Mp1", nodes=("1", "2", "0", "3"),
        device=f"pmos_nn L={l_nm:.0f}n NFIN={tech.nfin}", load="",
        analysis=f".dc Vgs 0 {-tech.vdd:g} -0.005",
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
            require_convergence=True,
        )
    finally:
        logging.disable(logging.NOTSET)

    # Use |Vgs| for an ascending interpolation axis; retain signed current.
    sweep = np.abs(np.array(results["2"]))
    signal = np.array(results["i(Mp1)"])
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
    baked_nmos = create_baked_inv_nmos_modelcard(tech, work_dir)
    baked_pmos = create_baked_inv_pmos_modelcard(tech, work_dir)

    netlist_path = work_dir / f"ngspice_inverter_vtc_{tech.name}.cir"
    netlist_content = _render_inverter_deck(
        {
            "MODEL_SETUP": (
                f'.include "{baked_nmos}"\n.include "{baked_pmos}"'
            ),
            "VDD": f"{tech.vdd:g}",
            "TEMP": f"{tech.temperature_c:g}",
            "INPUT_SPEC": "0",
            "N_PREFIX": "N", "P_PREFIX": "N",
            "N_DEVICE": tech.nmos_model,
            "P_DEVICE": tech.pmos_model,
            "OUTPUT_LOAD": "", "INITIAL_CONDITION": "",
            "ANALYSIS": f".dc Vin 0 {tech.vdd:g} 0.005",
        },
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
def _cascade_handles_stem(path: Optional[Path]) -> bool:
    """True if the path stem is one the parser's per-tech preempt cascade
    can route to the right checkpoint based on the netlist's TECH=.

    For these stems we deliberately omit MODEL_PATH in the netlist so a
    single test invocation can pick TSMC5 large for TSMC5 netlists, TSMC7
    large for TSMC7 netlists, etc. — instead of pinning ONE tech's net for
    every tech. All five per-tech ``tsmc{5,6,7,12,16}_dnf_*`` stems route
    through the parser's preempt cascade; ``refac_dnf_*`` is the
    universal-refactor preset the cascade also recognises. An explicit env
    pin (``PYCIRCUITSIM_NN_CHECKPOINT_DNF_{NMOS,PMOS}``) still wins because the
    parser reads it before the preempt, so omitting MODEL_PATH never defeats
    the benchmark's capacity-tier pinning.
    """
    if path is None:
        return False
    stem = path.name
    return any(stem.startswith(p) for p in (
        "tsmc5_dnf_", "tsmc6_dnf_", "tsmc7_dnf_", "tsmc12_dnf_",
        "tsmc16_dnf_", "refac_dnf_", "tsmc5_tff_", "tsmc6_tff_",
        "tsmc7_tff_", "tsmc12_tff_", "tsmc16_tff_", "refac_tff_",
    ))


def _ckpt_label(path: Path) -> str:
    """Human-readable label for a checkpoint path in the progress log.

    audit B5e: for a cascade-handled stem the path is only an *existence
    sentinel* — MODEL_PATH is deliberately omitted from the netlist (see
    ``_cascade_handles_stem``) and the parser resolves the real per-tech
    checkpoint from ``TECH=``. Printing the sentinel's filename as
    "Checkpoint:" contradicts the ``[NN-resolver]`` line the parser emits two
    lines later, so name the mechanism instead of the file.
    """
    if _cascade_handles_stem(path):
        return (f"resolved per-tech by the parser cascade from TECH= "
                f"(sentinel {path.name} is NOT loaded) — see [NN-resolver] below")
    return path.name


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

    l_nmos_nm = tech.effective_inv_l_nmos * 1e9
    l_pmos_nm = tech.effective_inv_l_pmos * 1e9
    nfin = tech.effective_inv_nfin
    nfin_p = tech.effective_inv_nfin_p

    netlist_path = work_dir / f"nn_{model_name}_inverter_vtc_{tech.name}.sp"

    nmos_params = nn_model_parameters(
        level, tech.nn_tech_key, tech.nn_vt,
    )
    if nmos_model_path is not None and not _cascade_handles_stem(nmos_model_path):
        nmos_params += f" MODEL_PATH={nmos_model_path}"

    pmos_params = nn_model_parameters(
        level, tech.nn_tech_key, tech.effective_pmos_vt,
    )
    if pmos_model_path is not None and not _cascade_handles_stem(pmos_model_path):
        pmos_params += f" MODEL_PATH={pmos_model_path}"

    content = _render_inverter_deck(
        {
            "MODEL_SETUP": (
                f".model nmos_nn NMOS ({nmos_params})\n"
                f".model pmos_nn PMOS ({pmos_params})"
            ),
            "VDD": f"{tech.vdd:g}",
            "INPUT_SPEC": "0",
            "N_PREFIX": "M", "P_PREFIX": "M",
            "N_DEVICE": f"nmos_nn L={l_nmos_nm:g}n NFIN={nfin}",
            "P_DEVICE": f"pmos_nn L={l_pmos_nm:g}n NFIN={nfin_p}",
            "OUTPUT_LOAD": "", "INITIAL_CONDITION": "",
            "TEMP": f"{tech.temperature_c:g}",
            "ANALYSIS": f".dc Vin 0 {tech.vdd:g} 0.005",
        },
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
            require_convergence=True,
        )
    finally:
        logging.disable(logging.NOTSET)

    sweep = np.array(results["in"])
    vout = np.array(results["out"])
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
        "bsimar_full": "red",
        "directnet_full": "purple",
    }
    linestyles = {
        "cmg_pycircuitsim": "-.",
        "bsimar_full": "--",
        "directnet_full": ":",
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
        f"L_n={tech.effective_inv_l_nmos*1e9:.0f}nm  "
        f"L_p={tech.effective_inv_l_pmos*1e9:.0f}nm  "
        f"NFIN={tech.effective_inv_nfin}"
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
    circuit: Optional[InvCircuitParams] = None,
) -> Dict[str, np.ndarray]:
    """Run NGSPICE CMOS inverter transient with load capacitor.

    ``circuit`` overrides Cload / input slew / pulse width / delay / tstop;
    ``circuit=None`` uses the legacy module-global fixed point.

    Returns {time, v(out), v(in)}.
    """
    cp = circuit or InvCircuitParams()
    baked_nmos = create_baked_inv_nmos_modelcard(tech, work_dir)
    baked_pmos = create_baked_inv_pmos_modelcard(tech, work_dir)
    per = cp.tr + cp.pw + cp.tf + max(cp.pw, 1.0e-9)

    netlist_path = work_dir / f"ngspice_inverter_tran_{tech.name}.cir"
    content = _render_inverter_deck(
        {
            "MODEL_SETUP": (
                f'.include "{baked_nmos}"\n.include "{baked_pmos}"'
            ),
            "VDD": f"{tech.vdd:g}", "TEMP": f"{tech.temperature_c:g}",
            "INPUT_SPEC": (
                f"PULSE(0 {tech.vdd:g} {cp.td:g} {cp.tr:g} {cp.tf:g} "
                f"{cp.pw:g} {per:g})"
            ),
            "N_PREFIX": "N", "P_PREFIX": "N",
            "N_DEVICE": tech.nmos_model,
            "P_DEVICE": tech.pmos_model,
            "OUTPUT_LOAD": f"Cload out 0 {cp.cload:g}",
            "INITIAL_CONDITION": f".ic V(out)={tech.vdd:g}",
            "ANALYSIS": f".tran {INV_TRAN_TSTEP:g} {cp.tstop:g} uic",
        },
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
    circuit: Optional[InvCircuitParams] = None,
) -> Dict[str, np.ndarray]:
    """Run PyCircuitSim NN inverter transient. Returns {time, v(out), v(in)}.

    ``circuit`` overrides Cload / input slew / pulse width / delay / tstop;
    ``circuit=None`` uses the legacy module-global fixed point.
    """
    from tests.common.circuit_benchmarks import run_directnet_transient

    cp = circuit or InvCircuitParams()
    l_nmos_nm = tech.effective_inv_l_nmos * 1e9
    l_pmos_nm = tech.effective_inv_l_pmos * 1e9
    nfin = tech.effective_inv_nfin
    nfin_p = tech.effective_inv_nfin_p
    per = cp.tr + cp.pw + cp.tf + max(cp.pw, 1.0e-9)

    nmos_params = nn_model_parameters(
        level, tech.nn_tech_key, tech.nn_vt,
    )
    if nmos_model_path is not None and not _cascade_handles_stem(nmos_model_path):
        nmos_params += f" MODEL_PATH={nmos_model_path}"

    pmos_params = nn_model_parameters(
        level, tech.nn_tech_key, tech.effective_pmos_vt,
    )
    if pmos_model_path is not None and not _cascade_handles_stem(pmos_model_path):
        pmos_params += f" MODEL_PATH={pmos_model_path}"

    netlist_path = work_dir / f"nn_{model_name}_inverter_tran_{tech.name}.sp"
    content = _render_inverter_deck(
        {
            "MODEL_SETUP": (
                f".model nmos_nn NMOS ({nmos_params})\n"
                f".model pmos_nn PMOS ({pmos_params})"
            ),
            "VDD": f"{tech.vdd:g}",
            "INPUT_SPEC": (
                f"PULSE 0 {tech.vdd:g} {cp.td:g} {cp.tr:g} {cp.tf:g} "
                f"{cp.pw:g} {per:g}"
            ),
            "N_PREFIX": "M", "P_PREFIX": "M",
            "N_DEVICE": f"nmos_nn L={l_nmos_nm:g}n NFIN={nfin}",
            "P_DEVICE": f"pmos_nn L={l_pmos_nm:g}n NFIN={nfin_p}",
            "OUTPUT_LOAD": f"Cload out 0 {cp.cload:g}",
            "INITIAL_CONDITION": f".ic V(out)={tech.vdd:g}",
            "TEMP": f"{tech.temperature_c:g}",
            "ANALYSIS": f".tran {INV_TRAN_TSTEP:g} {cp.tstop:g} uic",
        },
    )
    netlist_path.write_text(content)

    results, nr_failed, nr_error_msg = run_directnet_transient(netlist_path)

    out = {
        "time": results["time"],
        "v(out)": results["out"],
        "v(in)": results["in"],
    }
    if nr_failed:
        out["_nr_partial"] = True
        out["_nr_error_msg"] = nr_error_msg
    return out


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
        "bsimar_full": "red",
        "directnet_full": "purple",
    }

    # Panel 1: Input pulse
    ax1 = axes[0]
    ng_t_ns = ref_data["time"] * 1e9
    ax1.plot(ng_t_ns, ref_data["v(in)"], "b-", lw=1.5, label="V(in)")
    ax1.set_ylabel("V(in) [V]")
    ax1.set_title(
        f"Inverter Transient: {tech.name}  "
        f"L_n={tech.effective_inv_l_nmos*1e9:.0f}nm  "
        f"L_p={tech.effective_inv_l_pmos*1e9:.0f}nm  "
        f"NFIN={tech.effective_inv_nfin}  Cload={INV_CLOAD*1e15:.0f}fF"
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
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 10),
        gridspec_kw={"height_ratios": [1, 1]},
    )

    colors = {
        "cmg_pycircuitsim": "green",
        "bsimar_full": "red",
        "directnet_full": "purple",
    }
    linestyles = {
        "cmg_pycircuitsim": "-.",
        "bsimar_full": "--",
        "directnet_full": ":",
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
        "bsimar_full": "red",
        "directnet_full": "purple",
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
        # Skip techs whose code is out of the NN vocabulary (ASAP7 — out of
        # scope, Rule 14). Matches the skip already present in the PMOS-DC /
        # inverter runners; without it a bare `--tech` (which defaults to the
        # full TECH_ORDER incl. ASAP7) ran ASAP7 NMOS DC and produced garbage
        # rows + a nonzero exit (bug report B6).
        if not tech_code_in_vocab(tech.nn_tech_key, tech.nn_vt):
            print(f"\n  DC {tech.name} -- SKIPPED (tech code out of vocab)")
            continue
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

        # 3. BSIM-AR-Full (LEVEL=76, tech-code embedding)
        if checkpoints.get("bsimar_full") is not None:
            bsimar_ckpt = checkpoints["bsimar_full"]
            print(f"  [3/N] Running BSIM-AR-Full (LEVEL=76)...")
            print(f"    Checkpoint: {_ckpt_label(bsimar_ckpt)}")
            try:
                bsimar_data = run_pycircuitsim_nn_nmos_dc(
                    tech, work_dir, level=76,
                    model_name="bsimar_full",
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
                    tech=tech.name, model="bsimar_full", analysis="dc",
                    nrmse_pct=bsimar_metrics["nrmse"],
                    mre_pct=bsimar_metrics["mre"],
                    max_id_ref=bsimar_metrics["max_id_ref"],
                    max_id_test=bsimar_metrics["max_id_test"],
                    passed=passed,
                ))
                model_results_for_plot["bsimar_full"] = (bsimar_data, bsimar_metrics)
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append(TestResult(
                    tech=tech.name, model="bsimar_full", analysis="dc",
                    nrmse_pct=float("nan"), error=str(e),
                ))
        else:
            print(f"  [3/N] BSIM-AR-Full -- SKIPPED (no checkpoint)")

        # 4. DirectNet-Full (LEVEL=75, tech-code embedding, explicit MODEL_PATH)
        if checkpoints.get("directnet_full") is not None:
            dnf_ckpt = checkpoints["directnet_full"]
            print(f"  [4/N] Running DirectNet-Full (LEVEL=75, tech-code embedding)...")
            print(f"    Checkpoint: {_ckpt_label(dnf_ckpt)}")
            try:
                dnf_data = run_pycircuitsim_nn_nmos_dc(
                    tech, work_dir, level=75,
                    model_name="directnet_full",
                    model_path=dnf_ckpt,
                )
                dnf_metrics = compare_dc_curves(
                    ng_data["sweep"], ng_data["id"],
                    dnf_data["sweep"], dnf_data["id"],
                )
                # Check for broken model: flat output at ~0.5A
                id_range = float(dnf_data["id"].max() - dnf_data["id"].min())
                id_max = float(dnf_data["id"].max())
                is_broken = (id_range < 1e-6) or (id_max > 0.1)

                if is_broken:
                    print(f"    WARNING: Model appears BROKEN "
                          f"(Id_range={id_range:.2e}, Id_max={id_max:.2e})")
                    results.append(TestResult(
                        tech=tech.name, model="directnet_full", analysis="dc",
                        nrmse_pct=dnf_metrics["nrmse"],
                        mre_pct=dnf_metrics["mre"],
                        max_id_ref=dnf_metrics["max_id_ref"],
                        max_id_test=dnf_metrics["max_id_test"],
                        passed=False,
                        error="BROKEN: flat/extreme output",
                    ))
                else:
                    passed = dnf_metrics["nrmse"] < DC_NRMSE_THRESHOLD_NN * 100
                    status = "PASS" if passed else "FAIL"
                    print(f"    NRMSE={dnf_metrics['nrmse']:.2f}%  "
                          f"MRE={dnf_metrics['mre']:.2f}%  "
                          f"Id_max={dnf_metrics['max_id_test']:.4e} -> {status}")
                    results.append(TestResult(
                        tech=tech.name, model="directnet_full", analysis="dc",
                        nrmse_pct=dnf_metrics["nrmse"],
                        mre_pct=dnf_metrics["mre"],
                        max_id_ref=dnf_metrics["max_id_ref"],
                        max_id_test=dnf_metrics["max_id_test"],
                        passed=passed,
                    ))
                model_results_for_plot["directnet_full"] = (dnf_data, dnf_metrics)
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append(TestResult(
                    tech=tech.name, model="directnet_full", analysis="dc",
                    nrmse_pct=float("nan"), error=str(e),
                ))
        else:
            print(f"  [4/N] DirectNet-Full -- SKIPPED (no checkpoint)")

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

    has_pmos_ckpt = (checkpoints.get("bsimar_full_pmos") is not None or
                     checkpoints.get("directnet_full_pmos") is not None)
    if not has_pmos_ckpt:
        print("\n  PMOS DC tests -- SKIPPED (no PMOS checkpoints)")
        return results

    for tech_name in tech_names:
        tech = ALL_TEST_TECHS[tech_name]
        if not tech.pmos_model:
            print(f"\n  PMOS DC {tech.name} -- SKIPPED (no PMOS model configured)")
            continue

        # Skip ASAP7 if tech code is out of vocabulary
        if not tech_code_in_vocab(tech.nn_tech_key, tech.effective_pmos_vt):
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

        # 2. BSIM-AR-Full PMOS
        if checkpoints.get("bsimar_full_pmos") is not None:
            bsimar_ckpt = checkpoints["bsimar_full_pmos"]
            print(f"  [2/N] Running BSIM-AR-Full PMOS (LEVEL=76)...")
            print(f"    Checkpoint: {_ckpt_label(bsimar_ckpt)}")
            try:
                bsimar_data = run_pycircuitsim_nn_pmos_dc(
                    tech, work_dir, level=76,
                    model_name="bsimar_full",
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
                    tech=tech.name, model="bsimar_full_pmos", analysis="pmos_dc",
                    nrmse_pct=bsimar_metrics["nrmse"],
                    mre_pct=bsimar_metrics["mre"],
                    max_id_ref=bsimar_metrics["max_id_ref"],
                    max_id_test=bsimar_metrics["max_id_test"],
                    passed=passed,
                ))
                model_results_for_plot["bsimar_full"] = (bsimar_data, bsimar_metrics)
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append(TestResult(
                    tech=tech.name, model="bsimar_full_pmos", analysis="pmos_dc",
                    nrmse_pct=float("nan"), error=str(e),
                ))
        else:
            print(f"  [2/N] BSIM-AR-Full PMOS -- SKIPPED (no checkpoint)")

        # 3. DirectNet-Full PMOS
        if checkpoints.get("directnet_full_pmos") is not None:
            dnf_ckpt = checkpoints["directnet_full_pmos"]
            print(f"  [3/N] Running DirectNet-Full PMOS (LEVEL=75)...")
            print(f"    Checkpoint: {_ckpt_label(dnf_ckpt)}")
            try:
                dnf_data = run_pycircuitsim_nn_pmos_dc(
                    tech, work_dir, level=75,
                    model_name="directnet_full",
                    model_path=dnf_ckpt,
                )
                dnf_metrics = compare_dc_curves(
                    ng_data["sweep"], ng_data["id"],
                    dnf_data["sweep"], dnf_data["id"],
                )
                id_range = float(dnf_data["id"].max() - dnf_data["id"].min())
                id_max = float(dnf_data["id"].max())
                is_broken = (id_range < 1e-6) or (id_max > 0.1)

                if is_broken:
                    print(f"    WARNING: PMOS model appears BROKEN "
                          f"(Id_range={id_range:.2e}, Id_max={id_max:.2e})")
                    results.append(TestResult(
                        tech=tech.name, model="directnet_full_pmos",
                        analysis="pmos_dc",
                        nrmse_pct=dnf_metrics["nrmse"],
                        mre_pct=dnf_metrics["mre"],
                        max_id_ref=dnf_metrics["max_id_ref"],
                        max_id_test=dnf_metrics["max_id_test"],
                        passed=False,
                        error="BROKEN: flat/extreme output",
                    ))
                else:
                    passed = dnf_metrics["nrmse"] < DC_NRMSE_THRESHOLD_NN * 100
                    status = "PASS" if passed else "FAIL"
                    print(f"    NRMSE={dnf_metrics['nrmse']:.2f}%  "
                          f"MRE={dnf_metrics['mre']:.2f}%  "
                          f"|Id|_max={dnf_metrics['max_id_test']:.4e} -> {status}")
                    results.append(TestResult(
                        tech=tech.name, model="directnet_full_pmos",
                        analysis="pmos_dc",
                        nrmse_pct=dnf_metrics["nrmse"],
                        mre_pct=dnf_metrics["mre"],
                        max_id_ref=dnf_metrics["max_id_ref"],
                        max_id_test=dnf_metrics["max_id_test"],
                        passed=passed,
                    ))
                model_results_for_plot["directnet_full"] = (dnf_data, dnf_metrics)
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append(TestResult(
                    tech=tech.name, model="directnet_full_pmos", analysis="pmos_dc",
                    nrmse_pct=float("nan"), error=str(e),
                ))
        else:
            print(f"  [3/N] DirectNet-Full PMOS -- SKIPPED (no checkpoint)")

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
        if not tech_code_in_vocab(tech.nn_tech_key, tech.nn_vt):
            print(f"\n  Inverter VTC {tech.name} -- SKIPPED (tech code out of vocab)")
            continue

        work_dir = RESULTS_BASE / "inverter_vtc" / tech.name
        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  Inverter VTC Test: {tech.name}  "
              f"L_n={tech.effective_inv_l_nmos*1e9:.0f}nm  "
              f"L_p={tech.effective_inv_l_pmos*1e9:.0f}nm  "
              f"NFIN={tech.effective_inv_nfin}  VDD={tech.vdd:.2f}V")
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
        for model_tag, level in [("bsimar_full", 76), ("directnet_full", 75)]:
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
        if not tech_code_in_vocab(tech.nn_tech_key, tech.nn_vt):
            print(f"\n  Inverter tran {tech.name} -- SKIPPED (tech code out of vocab)")
            continue

        work_dir = RESULTS_BASE / "inverter_tran" / tech.name
        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  Inverter Transient Test: {tech.name}  "
              f"L_n={tech.effective_inv_l_nmos*1e9:.0f}nm  "
              f"L_p={tech.effective_inv_l_pmos*1e9:.0f}nm  "
              f"NFIN={tech.effective_inv_nfin}  Cload={INV_CLOAD*1e15:.0f}fF")
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

        for model_tag, level in [("bsimar_full", 76), ("directnet_full", 75)]:
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
                # B4: a transient that diverged mid-run (NR exhausted) is
                # recovered as a truncated waveform; compare_inverter_tran_*
                # only scores the matching prefix, so a divergence after a
                # railed prefix yields nrmse~0 -> spurious PASS. The
                # `_nr_partial` flag (set by run_pycircuitsim_nn_inverter_tran)
                # is an automatic FAIL — the very "fail loud" intent the flag
                # was added for.
                nr_partial = bool(nn_tran.get("_nr_partial"))
                # also catch UNFLAGGED truncation: the scorers only see the
                # common time prefix, so a waveform that ends early or is
                # near-empty must fail here whether or not the flag was set
                if (len(nn_tran["time"]) < 3
                        or nn_tran["time"][-1] < 0.98 * ng_tran["time"][-1]):
                    nr_partial = True

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
                elif nr_partial:
                    print(f"    FAIL: NR diverged mid-transient — partial "
                          f"waveform ({nn_tran.get('_nr_error_msg', '')[:80]})")
                    results.append(TestResult(
                        tech=tech.name, model=f"{model_tag}_inv_tran",
                        analysis="inv_tran",
                        nrmse_pct=post_metrics["nrmse_vdd"],
                        passed=False,
                        error="NR diverged mid-transient (partial waveform)",
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
        # Skip out-of-vocab techs (ASAP7 — Rule 14), as the PMOS-DC / inverter
        # runners do; a bare `--tech` defaults to the full TECH_ORDER incl.
        # ASAP7 and otherwise runs ASAP7 NMOS transient -> garbage/ERROR rows
        # and a nonzero exit (bug report B6).
        if not tech_code_in_vocab(tech.nn_tech_key, tech.nn_vt):
            print(f"\n  Transient {tech.name} -- SKIPPED (tech code out of vocab)")
            continue
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

        # 2. BSIM-AR-Full transient (LEVEL=76, tech-code embedding)
        if checkpoints.get("bsimar_full") is not None:
            print(f"  [2/N] Running BSIM-AR-Full (LEVEL=76) transient...")
            try:
                bsimar_tran = run_pycircuitsim_nn_nmos_tran(
                    tech, work_dir, level=76,
                    model_name="bsimar_full",
                    model_path=checkpoints["bsimar_full"],
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
                    tech=tech.name, model="bsimar_full", analysis="tran",
                    nrmse_pct=post_metrics["nrmse_vdd"],
                    passed=passed,
                    extra={"full_nrmse": full_metrics["nrmse_vdd"],
                           "max_err_mv": post_metrics["max_err_v"] * 1e3},
                ))
                model_results_for_plot["bsimar_full"] = (bsimar_tran, post_metrics)
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append(TestResult(
                    tech=tech.name, model="bsimar_full", analysis="tran",
                    nrmse_pct=float("nan"), error=str(e),
                ))
        else:
            print(f"  [2/N] BSIM-AR-Full transient -- SKIPPED")

        # 3. DirectNet-Full transient (LEVEL=75)
        if checkpoints.get("directnet_full") is not None:
            print(f"  [3/N] Running DirectNet-Full (LEVEL=75) transient...")
            try:
                dnf_tran = run_pycircuitsim_nn_nmos_tran(
                    tech, work_dir, level=75,
                    model_name="directnet_full",
                    model_path=checkpoints["directnet_full"],
                )
                full_metrics = compare_tran_waveforms(
                    ng_tran, dnf_tran, tech.vdd, t_start=0.0,
                )
                post_metrics = compare_tran_waveforms(
                    ng_tran, dnf_tran, tech.vdd,
                    t_start=TRAN_STARTUP_EXCL,
                )

                # Check for broken output
                v_range = float(dnf_tran["v(drain)"].max() - dnf_tran["v(drain)"].min())
                is_broken = v_range < 0.01

                if is_broken:
                    print(f"    WARNING: Flat transient output (range={v_range:.4f}V)")
                    results.append(TestResult(
                        tech=tech.name, model="directnet_full", analysis="tran",
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
                        tech=tech.name, model="directnet_full", analysis="tran",
                        nrmse_pct=post_metrics["nrmse_vdd"],
                        passed=passed,
                        extra={"full_nrmse": full_metrics["nrmse_vdd"],
                               "max_err_mv": post_metrics["max_err_v"] * 1e3},
                    ))
                model_results_for_plot["directnet_full"] = (dnf_tran, post_metrics)
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append(TestResult(
                    tech=tech.name, model="directnet_full", analysis="tran",
                    nrmse_pct=float("nan"), error=str(e),
                ))
        else:
            print(f"  [3/N] DirectNet-Full transient -- SKIPPED")

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
    else:
        l_nm = tech.l_nmos * 1e9
        vt_key = tech.nn_vt
        dev_name = "Mn1"
        dev_type = "NMOS"

    params = nn_model_parameters(level, tech.nn_tech_key, vt_key)
    # See B1 note above: let the parser resolve the per-tech checkpoint for
    # cascade-handled stems rather than pinning one tech's net for all techs.
    if model_path is not None and not _cascade_handles_stem(model_path):
        params += f" MODEL_PATH={model_path}"

    # Single-point sweep: sweep Vds over [vds, vds] to get one operating point
    netlist_path = work_dir / f"sign_{model_name}_{dev_type}_{vgs:.3f}_{vds:.3f}.sp"
    content = _render_mosfet_deck(
        model_setup=f".model nn_model {dev_type} ({params})",
        temperature_c=tech.temperature_c,
        drain_bias=f"Vds 1 0 {vds:g}", gate_bias=f"Vgs 2 0 {vgs:g}",
        source_bias="", bulk_bias="", device_name=dev_name,
        nodes=("1", "2", "0", "0"),
        device=f"nn_model L={l_nm:.0f}n NFIN={tech.nfin}", load="",
        analysis=f".dc Vds {vds:g} {vds:g} 1",
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
            require_convergence=True,
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
        if not tech_code_in_vocab(tech.nn_tech_key, tech.nn_vt):
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
        if tech.pmos_model and tech_code_in_vocab(
            tech.nn_tech_key, tech.effective_pmos_vt
        ):
            pmos_points = [
                (0.0, 0.0,             True, "~0"),
                (0.0, -0.02,           True, ">=0"),
                (0.0, -tech.vdd * 0.5, True, ">=0"),
            ]

        for model_tag, level in [("bsimar_full", 76), ("directnet_full", 75)]:
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
            # audit B5i: `nrmse_pct` here is NOT a metric — this screen
            # evaluates isolated bias points, so there is no curve to score.
            # 0.0/100.0 is a pass/fail flag squeezed into the shared
            # TestResult column; read the `sign` rows' Status, never their
            # NRMSE (unlike the `idvds` rows, which carry a real NRMSE).
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
    content = _render_mosfet_deck(
        model_setup=f'.include "{baked}"', temperature_c=27.0,
        drain_bias="Vds d 0 0", gate_bias="Vgs g 0 0",
        source_bias="", bulk_bias="", device_name="N1",
        nodes=("d", "g", "0", "0"), device=tech.nmos_model, load="",
        analysis=f".dc Vds -0.1 {tech.vdd:g} 0.005",
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

    subprocess.run(
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
    model_params = nn_model_parameters(
        level, tech.nn_tech_key, tech.nn_vt,
    )
    # Omit MODEL_PATH for stems the parser's per-tech preempt cascade handles
    # (tsmc{5,7,12,16}_dnf_* / refac_dnf_*) so each tech resolves its own
    # checkpoint from TECH= instead of being pinned to whichever single tech the
    # `directnet_full` alias happened to resolve to (bug report B1). A genuine
    # env pin still wins (the parser reads it before the preempt).
    if model_path is not None and not _cascade_handles_stem(model_path):
        model_params += f" MODEL_PATH={model_path}"

    netlist_path = work_dir / f"nn_{model_name}_nmos_idvds_{tech.name}.sp"
    content = _render_mosfet_deck(
        model_setup=f".model nmos_nn NMOS ({model_params})",
        temperature_c=tech.temperature_c, drain_bias="Vds 1 0 0",
        gate_bias="Vgs 2 0 0", source_bias="", bulk_bias="",
        device_name="Mn1", nodes=("1", "2", "0", "0"),
        device=f"nmos_nn L={l_nm:.0f}n NFIN={tech.nfin}", load="",
        analysis=f".dc Vds -0.1 {tech.vdd:g} 0.005",
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
            require_convergence=True,
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

    colors = {"bsimar_full": "red", "directnet_full": "purple"}

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

    Pass criterion is the wrong-sign count alone — that is what this
    diagnostic exists for, and a legitimately inaccurate but correctly-signed
    subthreshold curve must not turn red here for a different reason. The
    NRMSE/MRE carried on each row are the real curve errors against the
    NGSPICE reference (audit B5i: they used to be a fabricated 0.0/100.0 that
    printed as a perfect score); they are **reported, not gated**.
    """
    results: List[TestResult] = []

    print(f"\n{'='*70}")
    print("  Id-Vds DIAGNOSTIC (Vgs=0, subthreshold)")
    print(f"{'='*70}")

    for tech_name in tech_names:
        tech = ALL_TEST_TECHS[tech_name]
        if not tech_code_in_vocab(tech.nn_tech_key, tech.nn_vt):
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

        for model_tag, level in [("bsimar_full", 76), ("directnet_full", 75)]:
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
                reason = ""
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
                    if not passed:
                        reason = "wrong-sign Id at Vds>0"
                else:
                    # audit B5i: an empty screen means the sweep returned no
                    # usable points — nothing was actually checked, so this
                    # is a failure, not a free pass.
                    passed = False
                    reason = "no Vds>0.01 samples in the NN sweep"
                    print(f"    {model_tag}: {reason}")

                # audit B5i: score the curve against the NGSPICE reference
                # already in hand instead of fabricating 0.0/100.0. Reported
                # only — the verdict above stays the wrong-sign count.
                m = compare_dc_curves(
                    ng_data["sweep"], ng_data["id"],
                    nn_data["sweep"], nn_data["id"],
                )
                print(f"    {model_tag}: NRMSE={m['nrmse']:.2f}%  "
                      f"MRE={m['mre']:.2f}%  (reported, not gated)")

                results.append(TestResult(
                    tech=tech.name, model=f"{model_tag}_idvds",
                    analysis="idvds",
                    nrmse_pct=m["nrmse"],
                    mre_pct=m["mre"],
                    passed=passed,
                    error=reason,
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
def run_gate(tech_names: List[str], *,
             nmos_dc: bool = False, pmos_dc: bool = False,
             nmos_tran: bool = False, inverter_vtc: bool = False,
             inverter_tran: bool = False, sign_diag: bool = False,
             idvds_diag: bool = False, title: str = "NN Compact Model Verification") -> int:
    """Run a selection of the NN suites and return a process exit code.

    Factored out of the former ``main()`` in V7.5.8 so the device suites and
    the inverter suites can be invoked from their own tier packages
    (``tests/single_devices/verify_nn_dc.py`` and
    ``tests/simple_circuits/verify_nn_inverter.py``) without duplicating any
    of the setup, scoring or reporting. The suite bodies, thresholds and
    verdict logic below are unchanged from when this was one flat gate.
    """
    for t in tech_names:
        if t not in ALL_TEST_TECHS:
            print(f"ERROR: Unknown tech '{t}'. "
                  f"Available: {list(ALL_TEST_TECHS.keys())}")
            return 1

    RESULTS_BASE.mkdir(parents=True, exist_ok=True)

    # A pinned-but-absent stem is fatal (audit B5d) — report it as a clean
    # red instead of a traceback.
    try:
        checkpoints = get_available_checkpoints()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("=" * 70)
    print(f"  {title}")
    print(f"  {active_model_label()} vs BSIM-CMG (LEVEL=72)")
    print("=" * 70)
    print(f"\n  Technologies: {', '.join(tech_names)}")
    print(f"  Suites: NMOS_DC={nmos_dc}  PMOS_DC={pmos_dc}  "
          f"INV_VTC={inverter_vtc}  NMOS_TRAN={nmos_tran}  "
          f"INV_TRAN={inverter_tran}  "
          f"SIGN_DIAG={sign_diag}  IDVDS_DIAG={idvds_diag}")
    print(f"\n  Checkpoint availability:")
    for name, path in checkpoints.items():
        if name in ("bsimar_full", "directnet_full"):
            continue  # skip backward-compat aliases
        print(f"    {name:20s}: {path.name if path is not None else 'NOT FOUND'}")

    if all(v is None for v in checkpoints.values()):
        print("\n  ERROR: No NN checkpoints found. Nothing to test.")
        return 1

    print(f"\n  DC acceptance: NRMSE < {DC_NRMSE_THRESHOLD_NN*100:.0f}% (NN), "
          f"< {DC_NRMSE_THRESHOLD_CMG*100:.0f}% (CMG)")
    print(f"  VTC acceptance: NRMSE < {VTC_NRMSE_THRESHOLD*100:.0f}%")
    print(f"  Tran acceptance: NRMSE < {TRAN_NRMSE_THRESHOLD*100:.0f}% of Vdd")

    dc_results: List[TestResult] = []
    tran_results: List[TestResult] = []

    if nmos_dc:
        dc_results.extend(run_dc_tests(tech_names, checkpoints))
    if pmos_dc:
        dc_results.extend(run_pmos_dc_tests(tech_names, checkpoints))
    if inverter_vtc:
        dc_results.extend(run_inverter_vtc_tests(tech_names, checkpoints))
    if nmos_tran:
        tran_results.extend(run_tran_tests(tech_names, checkpoints))
    if inverter_tran:
        tran_results.extend(run_inverter_tran_tests(tech_names, checkpoints))

    diag_results: List[TestResult] = []
    if sign_diag:
        diag_results.extend(run_sign_diagnostic(tech_names, checkpoints))
    if idvds_diag:
        diag_results.extend(run_idvds_diagnostic(tech_names, checkpoints))

    all_tran_and_diag = tran_results + diag_results
    n_pass, n_fail, n_error = print_summary(dc_results, all_tran_and_diag)

    save_summary_csv(dc_results, all_tran_and_diag, RESULTS_BASE / "summary.csv")

    # A run that scored no NN rows (checkpoints missing / polarity skipped)
    # must not exit green on the CMG/NGSPICE reference rows alone.
    nn_scored = [r for r in dc_results + all_tran_and_diag
                 if "directnet" in r.model or "bsimar" in r.model]
    if not nn_scored:
        print(f"\n{'='*70}")
        print("  ERROR: no NN model was scored — passing rows are "
              "reference-only; nothing under test")
        return 1

    total = n_pass + n_fail + n_error
    print(f"\n{'='*70}")
    if n_fail > 0 or n_error > 0:
        print(f"  RESULT: {n_pass} PASS, {n_fail} FAIL, {n_error} ERROR "
              f"out of {total} tests")
        return 1
    print(f"  RESULT: ALL {n_pass} tests PASSED out of {total}")
    return 0


def tech_arg_parser(description: str) -> argparse.ArgumentParser:
    """Argument parser carrying the --tech selector both gate scripts share."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--tech", type=str, default=",".join(TECH_ORDER),
                    help="Comma-separated tech names (default: all)")
    return ap


def parse_techs(args: argparse.Namespace) -> List[str]:
    return [t.strip() for t in args.tech.split(",")]
