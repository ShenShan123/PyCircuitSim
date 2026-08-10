"""Technology registry for PyCMG training data generation.

Provides DeviceConfig, TechConfig, and TECH_REGISTRY covering all supported
FinFET technology nodes (ASAP7, TSMC5, TSMC7, TSMC12, TSMC16) with their
device variants (rvt, lvt, slvt, svt, hvt, ulvt, elvt, lnvt, sram).

Usage::

    from pycmg.tech import TECH_REGISTRY, get_tech_config, list_techs

    tech = get_tech_config("TSMC7")
    dev = tech.get_device("nmos_svt")
    print(dev.pdk_device)  # "nch_svt_mac"
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pycmg.parser import (
    _ASSIGN_RE,
    _find_length_variant,
    _scan_all_variants,
    parse_number_with_suffix,
    parse_tsmc_pdk,
    scan_pdk_geometry_combos,
)

# Project root for resolving relative modelcard/pdk paths
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(rel_path: str) -> Path:
    """Resolve a path that may be relative to project root or absolute."""
    p = Path(rel_path)
    if p.is_absolute():
        return p
    return _PROJECT_ROOT / p


def _parse_modelcard_l(modelcard_path: str) -> float:
    """Extract L from the first .model block in a modelcard file.

    Used for ASAP7-style modelcards where L is a parameter inside the
    model block (on continuation lines starting with '+').

    Args:
        modelcard_path: Path to the modelcard file (relative or absolute).

    Returns:
        Gate length L in meters.

    Raises:
        RuntimeError: If no L parameter is found in any .model block.
    """
    path = _resolve_path(modelcard_path)
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    in_model = False
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("*"):
            continue
        if stripped.lower().startswith(".model"):
            in_model = True
            continue
        if in_model and stripped.startswith("+"):
            content = stripped[1:]
            for m in _ASSIGN_RE.finditer(content):
                key = m.group(1).lower()
                if key == "l":
                    return parse_number_with_suffix(m.group(2))
        elif in_model and not stripped.startswith("+"):
            # End of model block continuation — only check the first block
            break

    raise RuntimeError(
        f"No L parameter found in modelcard: {modelcard_path}"
    )


def _scan_pdk_device_min_l(pdk_path: str, device_name: str) -> float:
    """Scan a TSMC PDK file for the minimum lmin across all numbered variants.

    Args:
        pdk_path: Path to the TSMC PDK file (relative or absolute).
        device_name: PDK device name, e.g. ``"nch_svt_mac"``.

    Returns:
        Minimum gate length in meters.

    Raises:
        RuntimeError: If no numbered variants are found for the device.
    """
    path = _resolve_path(pdk_path)
    variants = _scan_all_variants(str(path), device_name)
    if not variants:
        raise RuntimeError(
            f"No numbered variants found for device '{device_name}' "
            f"in PDK file: {pdk_path}"
        )
    return min(v.lmin for v in variants)


@dataclass
class DeviceConfig:
    """Configuration for a single device within a technology node.

    Attributes:
        model_name: SPICE model name, e.g. "nmos_rvt", "nch_svt_mac".
        inst_params: Static instance parameters (TFIN, DEVTYPE only).
            L and NFIN are excluded because they are swept during data generation.
        modelcard: Path to a static modelcard file relative to project root.
            Used for ASAP7 which has pre-built modelcard files.
            None for TSMC nodes (modelcards are generated from PDK on the fly).
        pdk_device: Device name within the TSMC PDK file, e.g. "nch_svt_mac".
            None for ASAP7 (no PDK parsing needed).
        _min_l: Per-device minimum channel length, auto-detected and cached.
            Populated by Task 3 (get_min_l); defaults to None.
    """

    model_name: str
    inst_params: Dict[str, float]
    modelcard: Optional[str] = None
    pdk_device: Optional[str] = None
    _min_l: Optional[float] = None

    def get_min_l(self, pdk_path: str | None = None) -> float:
        """Return minimum gate length for this device, auto-detecting if needed.

        For TSMC devices (pdk_device is set), scans the PDK file for the
        smallest ``lmin`` across all numbered length variants.  For ASAP7
        devices (modelcard is set), parses L from the first ``.model`` block.

        The result is cached in ``_min_l`` after the first call.

        Args:
            pdk_path: Path to TSMC PDK file, required for TSMC devices.
                Ignored for ASAP7 devices (which use modelcard instead).

        Returns:
            Minimum gate length in meters.

        Raises:
            RuntimeError: If min_l cannot be determined (no modelcard,
                no pdk_device, or missing pdk_path for TSMC device).
        """
        if self._min_l is not None:
            return self._min_l
        if self.pdk_device is not None and pdk_path is not None:
            self._min_l = _scan_pdk_device_min_l(pdk_path, self.pdk_device)
        elif self.modelcard is not None:
            self._min_l = _parse_modelcard_l(self.modelcard)
        else:
            raise RuntimeError(
                f"Cannot detect min_l for {self.model_name}: "
                f"no modelcard or pdk_device+pdk_path available"
            )
        return self._min_l

    def get_geometry_combos(
        self,
        pdk_path: str | None = None,
        ref_pdk_path: str | None = None,
        ref_device_name: str | None = None,
    ) -> list[tuple[float, float]]:
        """Return PDK-defined (L, NFIN) combinations for sweep.

        For TSMC devices, returns all ``(lmin, nfin)`` points where
        ``nfin`` is each of ``{nfinmin, nfinmax}`` per variant.

        For ASAP7 devices (no PDK binning), uses a reference TSMC device's
        NFIN boundaries combined with ASAP7's single L value.

        Args:
            pdk_path: Path to TSMC PDK file (for TSMC devices).
            ref_pdk_path: Reference TSMC PDK path for ASAP7 NFIN boundaries.
            ref_device_name: Reference TSMC device name for ASAP7 NFIN
                boundaries (e.g., ``"nch_svt_mac"`` for NMOS).

        Returns:
            Sorted list of ``(L, NFIN)`` tuples.
        """
        if self.pdk_device is not None and pdk_path is not None:
            resolved = str(_resolve_path(pdk_path))
            return scan_pdk_geometry_combos(resolved, self.pdk_device)

        # ASAP7: use reference TSMC device NFIN boundaries with ASAP7 L
        min_l = self.get_min_l(pdk_path)
        if ref_pdk_path is not None and ref_device_name is not None:
            resolved_ref = str(_resolve_path(ref_pdk_path))
            ref_combos = scan_pdk_geometry_combos(resolved_ref, ref_device_name)
            # Extract unique NFIN values from reference device
            nfin_values = sorted({nfin for _, nfin in ref_combos})
            return [(min_l, nfin) for nfin in nfin_values]

        # Fallback: single combo
        return [(min_l, 1.0)]


@dataclass
class TechConfig:
    """Configuration for a technology node.

    Attributes:
        name: Technology identifier, e.g. "ASAP7", "TSMC7".
        vdd: Core supply voltage in volts.
        tfin: Fin height/thickness in meters (e.g. 6.5e-9 for ASAP7).
        devices: Mapping from canonical device name to DeviceConfig.
            Canonical names follow the pattern: {nmos|pmos}_{vt_flavor}.
        pdk_path: Path to the TSMC PDK file relative to project root.
            None for ASAP7 (uses static modelcards instead).
    """

    name: str
    vdd: float
    tfin: float
    devices: Dict[str, DeviceConfig] = field(default_factory=dict)
    pdk_path: Optional[str] = None

    def list_devices(self) -> List[str]:
        """Return sorted list of canonical device names."""
        return sorted(self.devices.keys())

    def get_device(self, name: str) -> DeviceConfig:
        """Look up a device by canonical name.

        Args:
            name: Canonical device name, e.g. "nmos_rvt", "pmos_svt".

        Returns:
            The corresponding DeviceConfig.

        Raises:
            KeyError: If the device name is not found.
        """
        if name not in self.devices:
            available = ", ".join(sorted(self.devices.keys()))
            raise KeyError(
                f"Device '{name}' not found in {self.name}. "
                f"Available: {available}"
            )
        return self.devices[name]


# ---------------------------------------------------------------------------
# Helper functions to reduce repetition in registry construction
# ---------------------------------------------------------------------------

_ASAP7_MODELCARD = "modelcards/ASAP7/7nm_TT_160803.pm"


def _asap7_device(model_name: str, *, is_pmos: bool) -> DeviceConfig:
    """Create a DeviceConfig for an ASAP7 device."""
    return DeviceConfig(
        model_name=model_name,
        inst_params={"TFIN": 6.5e-9, "DEVTYPE": 0 if is_pmos else 1},
        modelcard=_ASAP7_MODELCARD,
        pdk_device=None,
    )


def _tsmc_device(
    pdk_device: str, *, tfin: float, is_pmos: bool
) -> DeviceConfig:
    """Create a DeviceConfig for a TSMC device."""
    return DeviceConfig(
        model_name=pdk_device,
        inst_params={"TFIN": tfin, "DEVTYPE": 0 if is_pmos else 1},
        modelcard=None,
        pdk_device=pdk_device,
    )


def _asap7_devices(vt_flavors: List[str]) -> Dict[str, DeviceConfig]:
    """Build device dict for ASAP7 with given Vt flavors."""
    devices: Dict[str, DeviceConfig] = {}
    for vt in vt_flavors:
        devices[f"nmos_{vt}"] = _asap7_device(f"nmos_{vt}", is_pmos=False)
        devices[f"pmos_{vt}"] = _asap7_device(f"pmos_{vt}", is_pmos=True)
    return devices


def _tsmc_devices(
    vt_flavors: List[str], *, tfin: float
) -> Dict[str, DeviceConfig]:
    """Build device dict for a TSMC node with given Vt flavors.

    Uses the naming convention: nch_{vt}_mac / pch_{vt}_mac for PDK device
    names, and nmos_{vt} / pmos_{vt} for canonical device names.
    """
    devices: Dict[str, DeviceConfig] = {}
    for vt in vt_flavors:
        pdk_n = f"nch_{vt}_mac"
        pdk_p = f"pch_{vt}_mac"
        devices[f"nmos_{vt}"] = _tsmc_device(pdk_n, tfin=tfin, is_pmos=False)
        devices[f"pmos_{vt}"] = _tsmc_device(pdk_p, tfin=tfin, is_pmos=True)
    return devices


# ---------------------------------------------------------------------------
# TECH_REGISTRY: master technology registry
# ---------------------------------------------------------------------------

TECH_REGISTRY: Dict[str, TechConfig] = {
    "ASAP7": TechConfig(
        name="ASAP7",
        vdd=0.9,
        tfin=6.5e-9,
        devices=_asap7_devices(["rvt", "lvt", "slvt", "sram"]),
        pdk_path=None,
    ),
    "TSMC5": TechConfig(
        name="TSMC5",
        vdd=0.65,
        tfin=6e-9,
        devices=_tsmc_devices(["svt", "lvt", "ulvt", "elvt"], tfin=6e-9),
        pdk_path="modelcards/TSMC5/cln5_1d2_sp_v1d2_2p2.l",
    ),
    "TSMC6": TechConfig(
        name="TSMC6",
        vdd=0.75,
        tfin=6e-9,
        devices=_tsmc_devices(["svt", "lvt", "ulvt"], tfin=6e-9),
        pdk_path="modelcards/TSMC6/cln6_1d8_sp_v1d0_2p2.l",
    ),
    "TSMC7": TechConfig(
        name="TSMC7",
        vdd=0.75,
        tfin=6e-9,
        devices=_tsmc_devices(["svt", "lvt", "ulvt"], tfin=6e-9),
        pdk_path="modelcards/TSMC7/cln7_1d8_sp_v1d2_2p2.l",
    ),
    "TSMC12": TechConfig(
        name="TSMC12",
        vdd=0.80,
        tfin=6e-9,
        devices=_tsmc_devices(
            ["svt", "lvt", "hvt", "ulvt", "lnvt"], tfin=6e-9
        ),
        pdk_path="modelcards/TSMC12/cln12ffcll_1d8_sp_v1d0_2p4.l",
    ),
    "TSMC16": TechConfig(
        name="TSMC16",
        vdd=0.80,
        tfin=6e-9,
        devices=_tsmc_devices(
            ["svt", "lvt", "hvt", "ulvt", "lnvt"], tfin=6e-9
        ),
        pdk_path="modelcards/TSMC16/crn16ffcll_1d8_sp_v1d0_2p1.l",
    ),
}


# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------


def list_techs() -> List[str]:
    """Return sorted list of all registered technology names."""
    return sorted(TECH_REGISTRY.keys())


def get_tech_config(name: str) -> TechConfig:
    """Look up a technology by name.

    Args:
        name: Technology name, e.g. "ASAP7", "TSMC7".

    Returns:
        The corresponding TechConfig.

    Raises:
        KeyError: If the technology name is not found.
    """
    if name not in TECH_REGISTRY:
        available = ", ".join(sorted(TECH_REGISTRY.keys()))
        raise KeyError(
            f"Technology '{name}' not found. Available: {available}"
        )
    return TECH_REGISTRY[name]


# ---------------------------------------------------------------------------
# Naive TSMC modelcard generation (extracted from scripts/generate_naive_tsmc.py)
# ---------------------------------------------------------------------------

# Parameters that should NOT be included in naive modelcards (instance parameters)
_INSTANCE_PARAMS = {
    'l', 'lmin', 'lmax',  # Length parameters
    'w', 'wmin', 'wmax',  # Width parameters
    'tfin', 'tfinmin', 'tfinmax',  # Fin thickness
    'nfin', 'nfinmin', 'nfinmax',  # Fin count
    'nf', 'nfmulti',  # Number of fingers
    'multi',  # Multiplier
}


def generate_naive_tsmc_modelcard(
    pdk_path: str,
    model_type: str,
    device_type: str,
    L: float,
    output_path: str,
    tech: str,
    NFIN: Optional[float] = None,
) -> None:
    """Generate naive TSMC modelcard by merging global + variant parameters.

    Extracts parameters from the full TSMC PDK (which has .global + numbered
    variant structure) and writes a single ``.model`` block that can be used
    directly with both PyCMG and NGSPICE+OSDI.

    Instance parameters (L, W, TFIN, NFIN, NF, MULTI) are excluded from the
    output; they should be supplied in the netlist when instantiating the
    device.

    Args:
        pdk_path: Path to full TSMC PDK file (e.g., cln7_1d8_sp_v1d2_2p2.l).
        model_type: ``"nch"`` for NMOS or ``"pch"`` for PMOS.
        device_type: Device type suffix (e.g., ``"svt_mac"``, ``"lvt_mac"``).
        L: Target gate length in meters (e.g., 16e-9).
        output_path: Output file path.
        tech: Technology name for header comment (e.g., ``"TSMC7"``).
        NFIN: Fin count for NFIN-group-specific variant selection.

    Raises:
        RuntimeError: If no variant matches *L* (and *NFIN*) in the PDK file.
    """
    base_name = f"{model_type}_{device_type}"  # e.g., "nch_svt_mac"

    # Delegate extraction + merge to parse_tsmc_pdk (single source of truth)
    parsed = parse_tsmc_pdk(pdk_path, model_type, device_type, L, NFIN)
    merged_params = parsed.params

    # Determine which variant was selected (needed for header comment only)
    variant_num = _find_length_variant(pdk_path, base_name, L, NFIN)

    # Write naive modelcard (PROCESS PARAMETERS ONLY).
    #
    # ATOMIC WRITE (V6.4.7 S9b fix): the on-the-fly cache file for a given
    # (pdk_device, L, NFIN) is shared by all three temperature bins (cards are
    # T-independent), so parallel data-generation workers race on it. A plain
    # truncate+write let a reader — another worker's Model() parse — see a
    # partial card, producing a DEGENERATE modelcard (only PHIG/TOXP non-zero,
    # the rest defaulted to 0) and physically-wrong training rows. Write to a
    # per-process temp file then os.replace() (atomic rename on the same
    # filesystem), so a reader only ever observes a complete card.
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.with_name(f"{output_file.name}.tmp.{os.getpid()}")

    nfin_str = f" NFIN={NFIN:.0f}" if NFIN is not None else ""
    with open(tmp_file, "w") as f:
        # Header
        f.write(f"* Naive {tech} {device_type} modelcard for L={L*1e9:.1f}nm{nfin_str}\n")
        f.write(f"* Generated from: {pdk_path}\n")
        f.write(f"* Device: {base_name}, Variant: .{variant_num}\n")
        f.write(f"* Process corner: TT (typical)\n")
        f.write(f"* Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("*\n")
        f.write("* This is a NAIVE modelcard - single .model definition without subcircuits.\n")
        f.write("* Instance parameters (L, W, TFIN, NFIN, NF, MULTI) should be provided\n")
        f.write("* in the netlist when instantiating the device.\n")
        f.write("*\n")

        # Model definition
        f.write(f".model {base_name} bsimcmg (\n")

        # Write all PROCESS parameters (not instance parameters!)
        param_count = 0
        skipped_sentinels = []
        for key, val in merged_params.items():
            # Skip instance parameters
            if key.lower() in _INSTANCE_PARAMS:
                continue

            # Skip sentinel values (TSMC PDKs use -999*10^n as "use default" markers)
            try:
                fval = float(val)
                if abs(fval) > 1e6:
                    # Check if value is a TSMC sentinel: ±999 * 10^n
                    exp = int(math.log10(abs(fval)))
                    mantissa = abs(fval) / (10.0 ** (exp - 2))  # normalize to 3-digit mantissa
                    if abs(mantissa - 999.0) < 0.5:
                        skipped_sentinels.append(f"{key}={val}")
                        continue
            except (ValueError, TypeError, OverflowError):
                pass

            # Format parameter line
            f.write(f"  + {key} = {val}\n")
            param_count += 1

        f.write(")\n")
        f.write(f"* Total parameters: {param_count}\n")
        if skipped_sentinels:
            f.write(f"* Skipped sentinel values: {', '.join(skipped_sentinels)}\n")

    # Atomic publish: a reader either sees no file (and generates its own) or
    # the fully-written card — never a truncated one.
    os.replace(tmp_file, output_file)


# ---------------------------------------------------------------------------
# resolve_modelcard() — on-the-fly generation with caching
# ---------------------------------------------------------------------------


def resolve_modelcard(
    device: DeviceConfig,
    tech: TechConfig,
    L: float,
    NFIN: Optional[float] = None,
    cache_dir: str = "build/modelcards",
) -> str:
    """Resolve a modelcard path for a device at a given gate length and NFIN.

    For ASAP7 devices (with a static ``modelcard`` attribute), returns the
    resolved path directly — the same file is used regardless of *L* or *NFIN*.

    For TSMC devices (with a ``pdk_device`` attribute), generates a naive
    modelcard on-the-fly from the full PDK and caches it under *cache_dir*.
    When *NFIN* is provided, the correct NFIN group variant is selected and
    the cache filename includes the NFIN value.

    Args:
        device: Device configuration from the technology registry.
        tech: Technology configuration (provides ``pdk_path`` and ``name``).
        L: Target gate length in meters (e.g., 16e-9).
        NFIN: Fin count for NFIN-group-specific variant selection.
        cache_dir: Directory for cached generated modelcards, resolved
            relative to the project root.  Defaults to ``"build/modelcards"``.

    Returns:
        Absolute path to the modelcard file (as a string).

    Raises:
        ValueError: If *device* has neither ``modelcard`` nor ``pdk_device``.
        RuntimeError: If *L* (and *NFIN*) fall outside all PDK variant ranges.
    """
    # ASAP7: static modelcard, L/NFIN-independent
    if device.modelcard is not None:
        return str(_resolve_path(device.modelcard))

    # TSMC: generate from PDK on-the-fly
    if device.pdk_device is not None:
        if tech.pdk_path is None:
            raise RuntimeError(
                f"Technology '{tech.name}' has no pdk_path configured "
                f"but device '{device.model_name}' requires PDK generation"
            )

        # Resolve cache directory relative to project root
        cache_root = _resolve_path(cache_dir)
        tech_cache = cache_root / tech.name

        # Build output filename, including NFIN when provided
        L_nm = int(L * 1e9)
        if NFIN is not None:
            filename = f"{device.pdk_device}_l{L_nm}nm_nfin{NFIN:.0f}.l"
        else:
            filename = f"{device.pdk_device}_l{L_nm}nm.l"
        output_path = tech_cache / filename

        # Return cached file if it already exists
        if output_path.exists():
            return str(output_path)

        # Generate: split pdk_device into model_type and device_type
        # e.g., "nch_svt_mac" -> model_type="nch", device_type="svt_mac"
        parts = device.pdk_device.split("_", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid pdk_device format: '{device.pdk_device}' "
                f"(expected 'nch_xxx' or 'pch_xxx')"
            )
        model_type, device_type = parts[0], parts[1]

        pdk_resolved = str(_resolve_path(tech.pdk_path))
        generate_naive_tsmc_modelcard(
            pdk_path=pdk_resolved,
            model_type=model_type,
            device_type=device_type,
            L=L,
            output_path=str(output_path),
            tech=tech.name,
            NFIN=NFIN,
        )

        return str(output_path)

    raise ValueError(
        f"Device '{device.model_name}' has neither modelcard nor pdk_device "
        f"configured — cannot resolve modelcard"
    )
