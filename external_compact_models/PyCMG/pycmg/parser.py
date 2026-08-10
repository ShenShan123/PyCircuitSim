"""
Modelcard parsing and parameter extraction for SPICE/TSMC PDK files.

This module provides:
- parse_number_with_suffix: SPICE number parsing (e.g., "1n" -> 1e-9)
- parse_modelcard: General SPICE modelcard parser
- parse_tsmc_pdk: TSMC PDK parameter extraction
- ParsedModel: Dataclass for parsed model results

Dependencies: osdi_types only (no core.py imports).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# Module-level regex for SPICE parameter assignment: NAME = VALUE[suffix]
_ASSIGN_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"([+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?[a-zA-Z]*)"
)


_SUFFIX_SCALE: Dict[str, float] = {
    "t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3,
    "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12,
    "f": 1e-15, "a": 1e-18, "z": 1e-21, "y": 1e-24,
}


def parse_number_with_suffix(token: str) -> float:
    s = token.strip()
    scale = 1.0
    pos = None
    for i, ch in enumerate(s):
        if ch not in "+-0123456789.eE":
            pos = i
            break
    if pos is not None:
        suffix = s[pos:].lower()
        s = s[:pos]
        scale = _SUFFIX_SCALE.get(suffix, 1.0)
    if not s or s in {"+", "-"}:
        return 0.0
    return float(s) * scale


@dataclass
class ParsedModel:
    name: str
    params: Dict[str, float]


def parse_modelcard(path: str, target_model_name: Optional[str] = None) -> ParsedModel:
    assign_re = _ASSIGN_RE
    target_lower = target_model_name.lower() if target_model_name else None

    def _parse_params(lines: List[str]) -> Dict[str, float]:
        parsed_params: Dict[str, float] = {}
        for line in lines:
            for match in assign_re.finditer(line):
                key = match.group(1)
                val = match.group(2)
                key_lower = key.lower()
                parsed = parse_number_with_suffix(val)
                if key_lower == "eotacc" and parsed <= 1.0e-10:
                    parsed = 1.1e-10
                if key_lower == "nf":
                    parsed = 1.0  # Single-fin default
                if key_lower == "nfin":
                    parsed = 1.0  # Single-fin default
                parsed_params[key_lower] = parsed
        return parsed_params

    def _is_valid_model(model_type: str, params: Dict[str, float]) -> bool:
        mtype = model_type.lower()
        if mtype == "bsimcmg":
            return True
        if mtype in {"nmos", "pmos"}:
            level = None
            for key, val in params.items():
                if key.lower() == "level":
                    level = val
                    break
            return level == 72
        return False

    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    idx = 0
    while idx < len(lines):
        raw = lines[idx]
        trimmed = raw
        pos_comment = trimmed.find("*")
        if pos_comment != -1:
            trimmed = trimmed[:pos_comment]
        trimmed = trimmed.strip()
        if not trimmed:
            idx += 1
            continue
        if trimmed.lower().startswith(".model"):
            block_lines = [trimmed]
            idx += 1
            while idx < len(lines):
                cont_raw = lines[idx]
                cont = cont_raw
                pos_comment = cont.find("*")
                if pos_comment != -1:
                    cont = cont[:pos_comment]
                cont = cont.strip()
                if not cont:
                    idx += 1
                    continue
                if cont.startswith("+"):
                    block_lines.append(cont[1:].strip())
                    idx += 1
                    continue
                break

            parts = block_lines[0].split()
            if len(parts) >= 3:
                model_name = parts[1]
                model_type = parts[2]
                if target_lower is None or model_name.lower() == target_lower:
                    params = _parse_params(block_lines)
                    if _is_valid_model(model_type, params):
                        # Inject DEVTYPE parameter for ASAP7 compatibility
                        # BSIM-CMG v107 uses DEVTYPE to distinguish NMOS (1) vs PMOS (0)
                        # ASAP7 modelcards often omit this, causing PMOS to behave incorrectly
                        model_type_lower = model_type.lower()
                        if "devtype" not in params:
                            if model_type_lower == "pmos":
                                params["devtype"] = 0.0  # PMOS
                            elif model_type_lower == "nmos":
                                params["devtype"] = 1.0  # NMOS
                        return ParsedModel(name=model_name, params=params)
            continue
        idx += 1

    expected = target_model_name if target_model_name else "bsimcmg or level=72 nmos/pmos"
    raise RuntimeError(f"no {expected} model found in modelcard: {path}")


def parse_tsmc_pdk(
    path: str,
    model_type: str,
    device_type: str,
    L: float,
    NFIN: Optional[float] = None,
) -> ParsedModel:
    """Extract and merge model parameters from full TSMC PDK.

    This function works with all TSMC FinFET PDKs (TSMC5, TSMC7, TSMC12, TSMC16)
    which share the same structure:
    - .global model: base parameters for all variants
    - .1 through .N variants: length-binned models with lmin/lmax and nfinmin/nfinmax
    - Subcircuit wrappers: not needed for OSDI (we use model directly)

    Args:
        path: Path to TSMC PDK file (e.g., cln7_1d8_sp_v1d2_2p2.l)
        model_type: ``"nch"`` for NMOS, ``"pch"`` for PMOS.
        device_type: Device type - ``"svt_mac"``, ``"lvt_mac"``, etc.
        L: Gate length in meters (used for automatic variant selection).
        NFIN: Fin count.  When given, selects the correct NFIN group.

    Returns:
        ParsedModel with merged global + variant parameters.
    """
    base_name = f"{model_type}_{device_type}"  # e.g., "nch_svt_mac"
    expected_type = "nmos" if model_type == "nch" else "pmos"

    # Extract global model parameters (base)
    try:
        global_params = _extract_model_params(path, f"{base_name}.global", expected_type)
    except RuntimeError as e:
        raise RuntimeError(
            f"TSMC PDK file '{path}' does not contain the expected .global model "
            f"'{base_name}.global'. This usually means:\n"
            f"  1. The file is not a valid TSMC PDK file\n"
            f"  2. The model_type '{model_type}' and device_type '{device_type}' combination "
            f"does not exist in this PDK\n"
            f"  3. The PDK file format has changed\n\n"
            f"Expected model: .model {base_name}.global {expected_type} (...)\n"
            f"Original error: {e}"
        ) from e

    # Find which variant matches the L (and NFIN) value
    variant_num = _find_length_variant(path, base_name, L, NFIN)

    # Extract variant model parameters
    variant_params = _extract_model_params(path, f"{base_name}.{variant_num}", expected_type)

    # Merge: variant overrides global
    merged_params = {**global_params, **variant_params}

    return ParsedModel(name=base_name, params=merged_params)


@dataclass
class VariantInfo:
    """Parsed information about a single numbered PDK variant."""

    variant_num: int
    lmin: float
    lmax: float
    nfinmin: Optional[float] = None
    nfinmax: Optional[float] = None


def _scan_all_variants(path: str, base_name: str) -> List[VariantInfo]:
    """Scan a TSMC PDK file and return all numbered variant records for a device.

    Parses every ``.model {base_name}.N`` block (where N is a digit) and
    extracts lmin, lmax, nfinmin, nfinmax from each.

    Args:
        path: Path to TSMC PDK file.
        base_name: Base model name (e.g., ``"nch_svt_mac"``).

    Returns:
        List of :class:`VariantInfo` in file order.
    """
    assign_re = _ASSIGN_RE
    prefix = f"{base_name.lower()}."
    variants: List[VariantInfo] = []

    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    idx = 0
    while idx < len(lines):
        raw = lines[idx]
        trimmed = raw.strip()

        if not trimmed or trimmed.startswith("*"):
            idx += 1
            continue

        if trimmed.lower().startswith(".model"):
            parts = trimmed.split()
            if len(parts) >= 3:
                model_name = parts[1]
                if model_name.lower().startswith(prefix):
                    suffix = model_name[len(base_name) + 1:]

                    if suffix.lower() == "global":
                        idx += 1
                        continue

                    if suffix.isdigit():
                        block_lines = [trimmed]
                        idx += 1
                        while idx < len(lines):
                            cont = lines[idx].strip()
                            if not cont or cont.startswith("*"):
                                idx += 1
                                continue
                            if cont.startswith("+"):
                                block_lines.append(cont[1:].strip())
                                idx += 1
                                continue
                            break

                        lmin = lmax = nfinmin = nfinmax = None
                        for line in block_lines:
                            for match in assign_re.finditer(line):
                                key = match.group(1).lower()
                                val = parse_number_with_suffix(match.group(2))
                                if key == "lmin":
                                    lmin = val
                                elif key == "lmax":
                                    lmax = val
                                elif key == "nfinmin":
                                    nfinmin = val
                                elif key == "nfinmax":
                                    nfinmax = val

                        if lmin is not None and lmax is not None:
                            variants.append(VariantInfo(
                                variant_num=int(suffix),
                                lmin=lmin,
                                lmax=lmax,
                                nfinmin=nfinmin,
                                nfinmax=nfinmax,
                            ))
                        continue  # idx already advanced
                    else:
                        sys.stderr.write(
                            f"Warning: Skipping unexpected variant '{model_name}' "
                            f"(suffix '{suffix}' is not numeric or 'global')\n"
                        )

        idx += 1

    return variants


def _find_length_variant(
    path: str, base_name: str, L: float, NFIN: Optional[float] = None,
) -> int:
    """Find which numbered variant matches L (and optionally NFIN).

    TSMC PDKs organise variants as a 2-D grid of (L_bin x NFIN_group).
    When *NFIN* is provided, both ``lmin <= L <= lmax`` **and**
    ``nfinmin <= NFIN <= nfinmax`` must be satisfied.  When *NFIN* is
    ``None``, the first variant whose L range contains *L* is returned
    (legacy behaviour).

    Args:
        path: Path to TSMC PDK file.
        base_name: Base model name (e.g., ``"nch_svt_mac"``).
        L: Gate length in meters.
        NFIN: Fin count.  When given, selects the correct NFIN group.

    Returns:
        Variant number (integer).

    Raises:
        RuntimeError: If no variant matches.
    """
    variants = _scan_all_variants(path, base_name)

    for v in variants:
        if v.lmin <= L <= v.lmax:
            if NFIN is None:
                return v.variant_num
            if (v.nfinmin is not None and v.nfinmax is not None
                    and v.nfinmin <= NFIN <= v.nfinmax):
                return v.variant_num

    nfin_str = f" NFIN={NFIN}" if NFIN is not None else ""
    raise RuntimeError(
        f"No variant found for {base_name} with L={L:.3e}{nfin_str} "
        f"in file: {path}"
    )


def scan_pdk_geometry_combos(
    path: str, base_name: str,
) -> List[Tuple[float, float]]:
    """Return all PDK-defined (L, NFIN) sweep points for a device.

    For each numbered variant, generates two points using the bin
    boundaries: ``(lmin, nfinmin)`` and ``(lmin, nfinmax)``.  Both points
    share the same modelcard (same variant binning coefficients); only the
    NFIN instance parameter differs.

    The result is deduplicated (adjacent NFIN groups share boundary
    values) and sorted ascending by ``(L, NFIN)``.

    Args:
        path: Path to TSMC PDK file.
        base_name: Base model name (e.g., ``"pch_lvt_mac"``).

    Returns:
        Sorted list of unique ``(L, NFIN)`` tuples.
    """
    variants = _scan_all_variants(path, base_name)

    combos: set[Tuple[float, float]] = set()
    for v in variants:
        if v.nfinmin is not None:
            combos.add((v.lmin, v.nfinmin))
        if v.nfinmax is not None:
            combos.add((v.lmin, v.nfinmax))

    return sorted(combos)


def _extract_model_params(path: str, model_name: str, expected_type: str) -> Dict[str, float]:
    """
    Extract parameters from a single .model block in TSMC PDK.

    Works with all TSMC FinFET PDKs (TSMC5, TSMC7, TSMC12, TSMC16).
    Reads from the model name match to the next non-continuation line.
    Parses all key=value pairs with SPICE number suffix support.

    Args:
        path: Path to TSMC PDK file
        model_name: Full model name including suffix (e.g., "nch_svt_mac.global" or "nch_svt_mac.4")
        expected_type: Expected model type ("nmos" or "pmos")

    Returns:
        Dictionary of parameter names to float values

    Raises:
        RuntimeError: If model not found
    """
    assign_re = _ASSIGN_RE

    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    idx = 0
    while idx < len(lines):
        raw = lines[idx]
        trimmed = raw.strip()

        # Skip comments and empty lines
        if not trimmed or trimmed.startswith("*"):
            idx += 1
            continue

        # Look for the target model
        if trimmed.lower().startswith(".model"):
            # Check if this matches our target using exact word match to avoid
            # substring false positives (e.g., ".1" matching ".10", ".14", etc.)
            parts = trimmed.split()
            if len(parts) >= 3 and parts[1] == model_name and expected_type in trimmed.lower():
                # Found it - parse the block
                block_lines = [trimmed]
                idx += 1
                while idx < len(lines):
                    cont_raw = lines[idx]
                    cont = cont_raw.strip()
                    if not cont or cont.startswith("*"):
                        idx += 1
                        continue
                    if cont.startswith("+"):
                        block_lines.append(cont[1:].strip())
                        idx += 1
                        continue
                    break

                # Parse parameters from the block
                params: Dict[str, float] = {}
                for line in block_lines:
                    for match in assign_re.finditer(line):
                        key = match.group(1)
                        val = match.group(2)
                        key_lower = key.lower()
                        parsed = parse_number_with_suffix(val)

                        # Apply EOTACC clamping for OSDI compatibility
                        if key_lower == "eotacc" and parsed <= 1.0e-10:
                            parsed = 1.1e-10

                        params[key_lower] = parsed

                # Inject DEVTYPE if not present (ASAP7 compatibility)
                # TSMC7 typically has this, but provides safety net
                if "devtype" not in params:
                    expected_type_lower = expected_type.lower()
                    if expected_type_lower == "pmos":
                        params["devtype"] = 0.0  # PMOS
                    elif expected_type_lower == "nmos":
                        params["devtype"] = 1.0  # NMOS

                return params

        idx += 1

    raise RuntimeError(f"Model {model_name} (type={expected_type}) not found in file: {path}")
