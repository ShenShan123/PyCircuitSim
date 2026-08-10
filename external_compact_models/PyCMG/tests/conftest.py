"""
Pytest configuration and technology registry for PyCMG verification tests.

The registry provides deterministic modelcard selection:
- ASAP7: Explicit TT corner + rvt variant (no glob ambiguity)
- TSMC: on-the-fly generation via ``pycmg.tech.resolve_modelcard``

Tiered registry:
- TECHNOLOGIES / TECH_NAMES: Original 5 entries + TSMC6 (backward-compatible)
- CORE_VT_VARIANTS / CORE_VT_NAMES: Additional core-voltage Vt flavors
- ALL_TECHNOLOGIES / ALL_TECH_NAMES: Union of all

This module imports from ``pycmg.tech.TECH_REGISTRY`` as the source of truth
for technology metadata (vdd, tfin). TSMC modelcards are generated on-the-fly
from the raw PDK files via ``resolve_modelcard``; ASAP7 still uses the
pre-committed static modelcards because its PDK ships pre-baked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

from pycmg.tech import TECH_REGISTRY, TechConfig, resolve_modelcard
from tests.helpers import ROOT, OSDI_PATH

# ---------------------------------------------------------------------------
# Test-specific constants: L and NFIN values used by existing tests.
# These are NOT stored in pycmg.tech (which treats L and NFIN as swept params).
# ---------------------------------------------------------------------------
_ASAP7_TEST_L = 7e-9
_ASAP7_TEST_NFIN = 1.0

_TSMC_NMOS_TEST_L = 16e-9
_TSMC_PMOS_TEST_L = 20e-9   # PMOS uses L=20nm to avoid L=16nm convergence
_TSMC_TEST_NFIN = 2.0

# ---------------------------------------------------------------------------
# Helper: build old-format inst_params dicts from tech registry + test values
# ---------------------------------------------------------------------------

def _make_asap7_params(is_pmos: bool) -> Dict[str, float]:
    """Build ASAP7 inst_params in the old test format (includes L, NFIN)."""
    tfin = TECH_REGISTRY["ASAP7"].tfin
    return {
        "L": _ASAP7_TEST_L,
        "TFIN": tfin,
        "NFIN": _ASAP7_TEST_NFIN,
        "DEVTYPE": 0 if is_pmos else 1,
    }


def _make_tsmc_params(is_pmos: bool, tech_name: str = "TSMC7") -> Dict[str, float]:
    """Build TSMC inst_params in the old test format (includes L, NFIN)."""
    # All TSMC nodes share TFIN=6e-9 — use the registry value for consistency
    tfin = TECH_REGISTRY[tech_name].tfin
    return {
        "L": _TSMC_PMOS_TEST_L if is_pmos else _TSMC_NMOS_TEST_L,
        "TFIN": tfin,
        "NFIN": _TSMC_TEST_NFIN,
        "DEVTYPE": 0 if is_pmos else 1,
    }


# ---------------------------------------------------------------------------
# Helper: ASAP7 / TSMC entry builders (old dict format for test consumption)
# ---------------------------------------------------------------------------

def _asap7_entry(nmos_model: str, pmos_model: str) -> Dict[str, Any]:
    """Build an ASAP7 registry entry (all variants share the same TT file)."""
    return {
        "tech": "ASAP7",
        "vdd": TECH_REGISTRY["ASAP7"].vdd, "corner": "TT",
        "asap7_file": "7nm_TT_160803.pm",  # ASAP7 is still file-based
        "nmos_model": nmos_model, "pmos_model": pmos_model,
        "nmos_params": _make_asap7_params(is_pmos=False),
        "pmos_params": _make_asap7_params(is_pmos=True),
    }


def _tsmc_entry(tech_name: str, vt: str) -> Dict[str, Any]:
    """Build a TSMC registry entry for a core-voltage Vt variant.

    Follows the naming convention: nch_{vt}_mac / pch_{vt}_mac
    NMOS uses L=16nm, PMOS uses L=20nm (avoids L=16nm convergence).
    Modelcards are resolved on-the-fly via ``resolve_modelcard`` at lookup time.
    """
    return {
        "tech": tech_name,
        "vdd": TECH_REGISTRY[tech_name].vdd,
        "nmos_model": f"nch_{vt}",
        "pmos_model": f"pch_{vt}",
        # Canonical device keys for TECH_REGISTRY[tech_name].devices[...]
        "nmos_device": f"nmos_{vt.replace('_mac', '')}",
        "pmos_device": f"pmos_{vt.replace('_mac', '')}",
        "nmos_params": _make_tsmc_params(is_pmos=False, tech_name=tech_name),
        "pmos_params": _make_tsmc_params(is_pmos=True, tech_name=tech_name),
    }


# ---------------------------------------------------------------------------
# Tier 1: Original technologies (backward-compatible, used by existing tests)
# ---------------------------------------------------------------------------
#
# Each entry specifies:
#   tech:         technology name (key into pycmg.tech.TECH_REGISTRY)
#   vdd:          core supply voltage (V)
#   nmos_model:   .model name inside the NMOS modelcard
#   pmos_model:   .model name inside the PMOS modelcard
#   nmos_device:  canonical device key in TECH_REGISTRY[tech].devices (TSMC only)
#   pmos_device:  canonical device key in TECH_REGISTRY[tech].devices (TSMC only)
#   nmos_params:  instance params for NMOS (baked into modelcard for NGSPICE)
#   pmos_params:  instance params for PMOS
#   asap7_file:   static ASAP7 filename (ASAP7 only)
#
# TSMC modelcards are generated on-the-fly from the raw PDK via
# ``pycmg.tech.resolve_modelcard`` at lookup time (see get_tech_modelcard()).
#
TECHNOLOGIES: Dict[str, Dict[str, Any]] = {
    "ASAP7":  _asap7_entry("nmos_rvt", "pmos_rvt"),
    # NOTE: Original TSMC entries use NMOS=svt + PMOS=lvt (historical choice).
    # New Vt variant entries in CORE_VT_VARIANTS use matched Vt for both.
    "TSMC5": {
        "tech": "TSMC5", "vdd": TECH_REGISTRY["TSMC5"].vdd,
        "nmos_model": "nch_svt_mac", "pmos_model": "pch_lvt_mac",
        "nmos_device": "nmos_svt", "pmos_device": "pmos_lvt",
        "nmos_params": _make_tsmc_params(is_pmos=False, tech_name="TSMC5"),
        "pmos_params": _make_tsmc_params(is_pmos=True, tech_name="TSMC5"),
    },
    "TSMC6": {
        "tech": "TSMC6", "vdd": TECH_REGISTRY["TSMC6"].vdd,
        "nmos_model": "nch_svt_mac", "pmos_model": "pch_lvt_mac",
        "nmos_device": "nmos_svt", "pmos_device": "pmos_lvt",
        "nmos_params": _make_tsmc_params(is_pmos=False, tech_name="TSMC6"),
        "pmos_params": _make_tsmc_params(is_pmos=True, tech_name="TSMC6"),
    },
    "TSMC7": {
        "tech": "TSMC7", "vdd": TECH_REGISTRY["TSMC7"].vdd,
        "nmos_model": "nch_svt_mac", "pmos_model": "pch_lvt_mac",
        "nmos_device": "nmos_svt", "pmos_device": "pmos_lvt",
        "nmos_params": _make_tsmc_params(is_pmos=False, tech_name="TSMC7"),
        "pmos_params": _make_tsmc_params(is_pmos=True, tech_name="TSMC7"),
    },
    "TSMC12": {
        "tech": "TSMC12", "vdd": TECH_REGISTRY["TSMC12"].vdd,
        "nmos_model": "nch_svt_mac", "pmos_model": "pch_lvt_mac",
        "nmos_device": "nmos_svt", "pmos_device": "pmos_lvt",
        "nmos_params": _make_tsmc_params(is_pmos=False, tech_name="TSMC12"),
        "pmos_params": _make_tsmc_params(is_pmos=True, tech_name="TSMC12"),
    },
    "TSMC16": {
        "tech": "TSMC16", "vdd": TECH_REGISTRY["TSMC16"].vdd,
        "nmos_model": "nch_svt_mac", "pmos_model": "pch_lvt_mac",
        "nmos_device": "nmos_svt", "pmos_device": "pmos_lvt",
        "nmos_params": _make_tsmc_params(is_pmos=False, tech_name="TSMC16"),
        "pmos_params": _make_tsmc_params(is_pmos=True, tech_name="TSMC16"),
    },
}

TECH_NAMES = list(TECHNOLOGIES.keys())

# ---------------------------------------------------------------------------
# Tier 2: Core-voltage Vt variants (same Vdd & geometry, different threshold)
# ---------------------------------------------------------------------------
CORE_VT_VARIANTS: Dict[str, Dict[str, Any]] = {
    # ASAP7 — lvt, slvt, sram (rvt is already in TECHNOLOGIES)
    "ASAP7_lvt":  _asap7_entry("nmos_lvt", "pmos_lvt"),
    "ASAP7_slvt": _asap7_entry("nmos_slvt", "pmos_slvt"),
    "ASAP7_sram": _asap7_entry("nmos_sram", "pmos_sram"),

    # TSMC5 — svt already tested via TECHNOLOGIES; add lvt, ulvt, elvt
    "TSMC5_lvt":  _tsmc_entry("TSMC5", "lvt_mac"),
    "TSMC5_ulvt": _tsmc_entry("TSMC5", "ulvt_mac"),
    "TSMC5_elvt": _tsmc_entry("TSMC5", "elvt_mac"),

    # TSMC6 — svt already tested; add lvt, ulvt (N7-derived node, same flavors)
    "TSMC6_lvt":  _tsmc_entry("TSMC6", "lvt_mac"),
    "TSMC6_ulvt": _tsmc_entry("TSMC6", "ulvt_mac"),

    # TSMC7 — svt already tested; add lvt, ulvt
    "TSMC7_lvt":  _tsmc_entry("TSMC7", "lvt_mac"),
    "TSMC7_ulvt": _tsmc_entry("TSMC7", "ulvt_mac"),

    # TSMC12 — svt already tested; add lvt, hvt, ulvt, lnvt
    "TSMC12_lvt":  _tsmc_entry("TSMC12", "lvt_mac"),
    "TSMC12_hvt":  _tsmc_entry("TSMC12", "hvt_mac"),
    "TSMC12_ulvt": _tsmc_entry("TSMC12", "ulvt_mac"),
    "TSMC12_lnvt": _tsmc_entry("TSMC12", "lnvt_mac"),

    # TSMC16 — svt already tested; add lvt, hvt, ulvt, lnvt
    "TSMC16_lvt":  _tsmc_entry("TSMC16", "lvt_mac"),
    "TSMC16_hvt":  _tsmc_entry("TSMC16", "hvt_mac"),
    "TSMC16_ulvt": _tsmc_entry("TSMC16", "ulvt_mac"),
    "TSMC16_lnvt": _tsmc_entry("TSMC16", "lnvt_mac"),
}

CORE_VT_NAMES = list(CORE_VT_VARIANTS.keys())

# ---------------------------------------------------------------------------
# Union: all technologies (Tier 1 + Tier 2)
# ---------------------------------------------------------------------------
ALL_TECHNOLOGIES: Dict[str, Dict[str, Any]] = {**TECHNOLOGIES, **CORE_VT_VARIANTS}
ALL_TECH_NAMES = list(ALL_TECHNOLOGIES.keys())


def get_tech_modelcard(tech_name: str, device_type: str = "nmos") -> Tuple[Path, str, Dict[str, float]]:
    """Get modelcard path, model name, and instance params for a technology.

    Searches ALL_TECHNOLOGIES (Tier 1 + Tier 2). For ASAP7 entries this
    returns the static pre-committed modelcard; for TSMC entries this
    regenerates the naive modelcard on-the-fly via ``resolve_modelcard`` and
    returns the cached path under ``build/modelcards/``.

    Args:
        tech_name: Key from ALL_TECHNOLOGIES registry
        device_type: "nmos" or "pmos"

    Returns:
        Tuple of (modelcard_path, model_name, inst_params)
    """
    tech = ALL_TECHNOLOGIES[tech_name]
    model_key = f"{device_type}_model"
    params_key = f"{device_type}_params"
    inst_params = tech[params_key]

    # ASAP7: static pre-committed modelcard
    if tech["tech"] == "ASAP7":
        modelcard = ROOT / "modelcards" / "ASAP7" / tech["asap7_file"]
        if not modelcard.exists():
            raise FileNotFoundError(f"Modelcard not found: {modelcard}")
        return modelcard, tech[model_key], inst_params

    # TSMC: regenerate on-the-fly from the raw PDK. Pass the test's actual
    # NFIN so resolve_modelcard selects the correct NFIN-group variant.
    tech_config = TECH_REGISTRY[tech["tech"]]
    device_key = f"{device_type}_device"
    device_config = tech_config.get_device(tech[device_key])
    modelcard = Path(
        resolve_modelcard(
            device_config, tech_config,
            L=inst_params["L"], NFIN=inst_params["NFIN"],
        )
    )
    if not modelcard.exists():
        raise FileNotFoundError(
            f"Modelcard not generated by resolve_modelcard: {modelcard}"
        )
    return modelcard, tech[model_key], inst_params


# -- pytest hooks (keep existing) --

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Add test report attribute to node for result tracking."""
    outcome = yield
    report = outcome.get_result()
    setattr(item, "rep_" + report.when, report)


# ---------------------------------------------------------------------------
# Shared test utilities
# ---------------------------------------------------------------------------

# Shared skip marker — replaces per-file @pytest.mark.skipif(not OSDI_PATH.exists(), ...)
requires_osdi = pytest.mark.skipif(
    not OSDI_PATH.exists(), reason="missing OSDI build artifact"
)


def standard_bias_points(
    vdd: float, device_type: str, regions: str = "all",
) -> dict[str, dict[str, float]]:
    """Canonical bias points for DC verification tests.

    Args:
        vdd: Supply voltage
        device_type: "nmos" or "pmos"
        regions: "all" for 3 regions, or a specific region name

    Returns:
        Dict mapping region name -> terminal voltage dict {d, g, s, e}
    """
    if device_type == "nmos":
        points = {
            "off":        {"d": vdd,       "g": 0.0,       "s": 0.0, "e": 0.0},
            "linear":     {"d": 0.3 * vdd, "g": vdd,       "s": 0.0, "e": 0.0},
            "saturation": {"d": vdd,       "g": 0.8 * vdd, "s": 0.0, "e": 0.0},
        }
    else:
        # ve=0 exercises deep reverse body bias (Vbs = -Vdd)
        points = {
            "off":        {"d": 0.0,       "g": vdd,       "s": vdd, "e": 0.0},
            "linear":     {"d": 0.7 * vdd, "g": 0.0,       "s": vdd, "e": 0.0},
            "saturation": {"d": 0.0,       "g": 0.2 * vdd, "s": vdd, "e": 0.0},
        }
    if regions == "all":
        return points
    return {regions: points[regions]}


REGION_NAMES = ["off", "linear", "saturation"]
