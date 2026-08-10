"""
PyCMG testing utilities — NGSPICE comparison helpers.

Consolidates NGSPICE runner, modelcard baking, and assertion functions
previously duplicated across test_integration.py, test_asap7.py,
test_tsmc{5,7,12,16}.py.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from pycmg import Model, Instance

ROOT = Path(__file__).resolve().parents[1]
OSDI_PATH = ROOT / "build" / "osdi" / "bsimcmg.osdi"
BUILD = ROOT / "build"
NGSPICE_BIN = os.environ.get("NGSPICE_BIN", "/usr/local/ngspice-45.2/bin/ngspice")

# Tolerances — single source of truth
ABS_TOL_I = 1e-9      # Current (A)
ABS_TOL_Q = 1e-18     # Charge (C)
ABS_TOL_G = 1e-6      # Conductance (S) — Jacobian floor
ABS_TOL_C = 1e-18     # Capacitance (F)
REL_TOL = 5e-3        # DC/transient relative (0.5%)
REL_TOL_JAC = 1e-2    # Jacobian relative (1%) — central finite-difference
REL_TOL_CAP = 1e-2    # Capacitance relative (1%) — condensation approximation


def bake_inst_params(src: Path, dst: Path, model_name: str,
                     inst_params: Dict[str, Any]) -> None:
    """Bake instance parameters into modelcard for NGSPICE OSDI compatibility.

    NGSPICE's OSDI interface cannot accept instance parameters on the device
    line (unlike HSPICE/Spectre). All geometric parameters (L, TFIN, NFIN)
    must be injected into the .model block.

    Args:
        src: Source modelcard file path
        dst: Destination path for baked modelcard
        model_name: Target .model name to modify
        inst_params: Dict of params to bake (e.g. {"L": 16e-9, "TFIN": 6e-9})
    """
    text = src.read_text()

    # Clamp EOTACC (standardized threshold: <= 1.0e-10 → 1.1e-10)
    def fix_eotacc(m: re.Match) -> str:
        val = float(m.group(1))
        if val <= 1.0e-10:
            return "EOTACC = 1.1e-10"
        return m.group(0)
    text = re.sub(r"EOTACC\s*=\s*([0-9eE+\-\.]+)", fix_eotacc, text, flags=re.IGNORECASE)

    lines: List[str] = []
    in_target = False
    found_keys: set = set()
    target_lower = model_name.lower()

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()

        if stripped.lower().startswith(".model"):
            # Close previous target block if open
            if in_target:
                for key, val in inst_params.items():
                    if key.upper() not in found_keys:
                        lines.append(f"+ {key.upper()} = {val}")
                in_target = False
                found_keys.clear()

            parts = stripped.split()
            if len(parts) >= 3 and parts[1].lower() == target_lower:
                parts[2] = "bsimcmg"
                prefix = line[:line.lower().find(".model")]
                line = f"{prefix}{' '.join(parts)}"
                in_target = True

        elif in_target:
            if stripped == ')':
                # Insert params BEFORE closing bracket
                for key, val in inst_params.items():
                    if key.upper() not in found_keys:
                        lines.append(f"+ {key.upper()} = {val}")
                in_target = False
                found_keys.clear()
            else:
                # Replace existing param values in-place
                for key, val in inst_params.items():
                    pattern = rf"(?i)\b{re.escape(key)}\s*=\s*([0-9eE+\-\.]+[a-zA-Z]*)"
                    def repl(m: re.Match, k: str = key.upper(), v: Any = val) -> str:
                        found_keys.add(k)
                        return f"{k} = {v}"
                    line, _ = re.subn(pattern, repl, line)

        lines.append(line)

    # Handle case where .model block has no closing bracket
    if in_target:
        for key, val in inst_params.items():
            if key.upper() not in found_keys:
                lines.append(f"+ {key.upper()} = {val}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + "\n")


def _run_ngspice_op_query(
    modelcard: Path, model_name: str, inst_params: Dict[str, Any],
    vd: float, vg: float, vs: float, ve: float, temp_c: float,
    wrdata_vars: str, output_map: Dict[str, str], tag: str,
) -> Dict[str, float]:
    """Shared helper: bake params, run NGSPICE .op, parse CSV output.

    Args:
        wrdata_vars: Space-separated NGSPICE variables for wrdata command.
        output_map: Maps result key -> NGSPICE header name.
    """
    out_dir = BUILD / "ngspice_eval" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    ng_modelcard = out_dir / f"ng_{model_name}.lib"
    bake_inst_params(modelcard, ng_modelcard, model_name, inst_params)

    net = [
        "* OP point query",
        f'.include "{ng_modelcard}"',
        f".temp {temp_c}",
        f"Vd d 0 {vd}",
        f"Vg g 0 {vg}",
        f"Vs s 0 {vs}",
        f"Ve e 0 {ve}",
        f"N1 d g s e {model_name}",
        ".op",
        ".end",
    ]

    out_csv = out_dir / "ng_op.csv"
    net_path = out_dir / "netlist.cir"
    log_path = out_dir / "ng_op.log"
    runner_path = out_dir / "runner.cir"

    runner_path.write_text(
        "* ngspice runner\n"
        ".control\n"
        f"osdi {OSDI_PATH}\n"
        f"source {net_path}\n"
        "set filetype=ascii\n"
        "set wr_vecnames\n"
        ".options saveinternals\n"
        "run\n"
        f"wrdata {out_csv} {wrdata_vars}\n"
        ".endc\n"
        ".end\n"
    )
    net_path.write_text("\n".join(net))

    try:
        res = subprocess.run(
            [NGSPICE_BIN, "-b", "-o", str(log_path), str(runner_path)],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"NGSPICE timed out after 120s (tag={tag}). See log: {log_path}"
        )
    if res.returncode != 0:
        raise RuntimeError(
            f"NGSPICE failed (tag={tag}):\n{res.stdout}\n{res.stderr}\n"
            f"See log: {log_path}"
        )

    if not out_csv.exists():
        raise RuntimeError(
            f"NGSPICE produced no output (tag={tag}). See log: {log_path}"
        )

    with out_csv.open() as f:
        csv_lines = f.readlines()
        if not csv_lines:
            raise RuntimeError(f"Empty NGSPICE output: {out_csv}")
        headers = csv_lines[0].split()
        values = [float(x) for x in csv_lines[1].split()]
        idx_map = {name: i for i, name in enumerate(headers)}
        return {key: values[idx_map[ng_name]] for key, ng_name in output_map.items()}


# NGSPICE variable sets and output mappings for OP / AC queries
_OP_WRDATA = (
    "v(g) v(d) v(s) v(e) i(vg) i(vd) i(vs) i(ve) "
    "@n1[qg] @n1[qd] @n1[qs] @n1[qb] @n1[gm] @n1[gds] @n1[gmbs]"
)
_OP_MAP: Dict[str, str] = {
    "id": "i(vd)", "ig": "i(vg)", "is": "i(vs)", "ie": "i(ve)",
    "qg": "@n1[qg]", "qd": "@n1[qd]", "qs": "@n1[qs]", "qb": "@n1[qb]",
    "gm": "@n1[gm]", "gds": "@n1[gds]", "gmb": "@n1[gmbs]",
}
_AC_WRDATA = "@n1[cgg] @n1[cgd] @n1[cgs] @n1[cdg] @n1[cdd]"
_AC_MAP: Dict[str, str] = {
    "cgg": "@n1[cgg]", "cgd": "@n1[cgd]", "cgs": "@n1[cgs]",
    "cdg": "@n1[cdg]", "cdd": "@n1[cdd]",
}


def run_ngspice_op(modelcard: Path, model_name: str, inst_params: Dict[str, Any],
                   vd: float, vg: float, vs: float = 0.0, ve: float = 0.0,
                   temp_c: float = 27.0,
                   tag: str = "op") -> Dict[str, float]:
    """Run NGSPICE operating point analysis and return currents/charges/derivatives."""
    return _run_ngspice_op_query(
        modelcard, model_name, inst_params, vd, vg, vs, ve, temp_c,
        _OP_WRDATA, _OP_MAP, tag,
    )


def run_ngspice_ac(modelcard: Path, model_name: str, inst_params: Dict[str, Any],
                   vd: float, vg: float, vs: float = 0.0, ve: float = 0.0,
                   temp_c: float = 27.0,
                   tag: str = "ac") -> Dict[str, float]:
    """Run NGSPICE AC capacitance extraction (cgg, cgd, cgs, cdg, cdd)."""
    return _run_ngspice_op_query(
        modelcard, model_name, inst_params, vd, vg, vs, ve, temp_c,
        _AC_WRDATA, _AC_MAP, tag,
    )


def assert_close(label: str, py_val: float, ng_val: float,
                 abs_tol: Optional[float] = None,
                 rel_tol: float = REL_TOL) -> None:
    """Assert PyCMG and NGSPICE values are within tolerance.

    Auto-selects abs_tol based on label if not provided:
    - Labels starting with 'q' → ABS_TOL_Q (1e-18)
    - Labels starting with 'c' → ABS_TOL_C (1e-18)
    - Labels starting with 'g' or 'd(' (conductance) → ABS_TOL_G (1e-6)
    - Default → ABS_TOL_I (1e-9)

    Args:
        label: Descriptive label for error messages
        py_val: PyCMG computed value
        ng_val: NGSPICE reference value
        abs_tol: Absolute tolerance override (auto-selected if None)
        rel_tol: Relative tolerance
    """
    if abs_tol is None:
        lbl_lower = label.lower().split("/")[-1]
        if lbl_lower.startswith("q"):
            abs_tol = ABS_TOL_Q
        elif lbl_lower.startswith("c"):
            abs_tol = ABS_TOL_C
        elif lbl_lower.startswith("g") or lbl_lower.startswith("d("):
            abs_tol = ABS_TOL_G
        else:
            abs_tol = ABS_TOL_I

    diff = abs(py_val - ng_val)
    if diff <= abs_tol:
        return
    denom = max(abs(ng_val), abs_tol)
    if diff / denom <= rel_tol:
        return
    pytest.fail(
        f"{label} mismatch:\n"
        f"  PyCMG:   {py_val:.6e}\n"
        f"  NGSPICE: {ng_val:.6e}\n"
        f"  Diff:    {diff:.6e} (abs_tol={abs_tol:.3e}, rel_tol={rel_tol:.3e})"
    )


def get_wave(ng_wave: Dict[str, np.ndarray], key_lower: str,
             n_points: int) -> np.ndarray:
    """Look up a waveform key case-insensitively in NGSPICE output."""
    for k in ng_wave:
        if k.lower() == key_lower:
            return ng_wave[k]
    return np.zeros(n_points)


def run_ngspice_transient(modelcard: Path, model_name: str,
                          inst_params: Dict[str, Any], vdd: float,
                          t_step: float = 10e-12,
                          t_stop: float = 5e-9,
                          temp_c: float = 27.0,
                          tag: str = "tran",
                          device_type: str = "nmos") -> Dict[str, np.ndarray]:
    """Run NGSPICE transient simulation with pulse stimulus on gate.

    Args:
        modelcard: Path to modelcard file
        model_name: Model name in the modelcard
        inst_params: Instance parameters to bake
        vdd: Supply voltage
        t_step: Transient time step
        t_stop: Transient stop time
        temp_c: Temperature in Celsius
        tag: Unique tag for output files
        device_type: "nmos" or "pmos" — controls bias and pulse polarity

    NMOS: Vd=Vdd, Vs=0, gate pulse 0→Vdd (turns on then off)
    PMOS: Vd=0, Vs=Vdd, gate pulse Vdd→0 (turns on then off)

    Returns dict mapping variable names to numpy arrays:
        'time', 'v(d)', 'v(g)', 'v(s)', 'v(e)',
        'i(vd)', 'i(vg)', 'i(vs)', 'i(ve)'
    """
    if device_type not in ("nmos", "pmos"):
        raise ValueError(f"device_type must be 'nmos' or 'pmos', got {device_type!r}")

    out_dir = BUILD / "ngspice_eval" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    ng_modelcard = out_dir / f"ng_{model_name}.lib"
    bake_inst_params(modelcard, ng_modelcard, model_name, inst_params)

    rise_time = 100e-12  # 100 ps
    width = 2e-9         # 2 ns
    period = 4e-9        # 4 ns

    if device_type == "pmos":
        # PMOS: source at Vdd, drain at 0, gate pulse from Vdd→0→Vdd
        net = [
            "* Transient test (PMOS)",
            f'.include "{ng_modelcard}"',
            f".temp {temp_c}",
            "Vd d 0 0",
            f"Vg g 0 PULSE({vdd} 0 500p {rise_time:.0e} {rise_time:.0e} {width:.0e} {period:.0e})",
            f"Vs s 0 {vdd}",
            "Ve e 0 0",
            f"N1 d g s e {model_name}",
            f".tran {t_step:.0e} {t_stop:.0e}",
            ".end",
        ]
    else:
        # NMOS: drain at Vdd, source at 0, gate pulse from 0→Vdd→0
        net = [
            "* Transient test",
            f'.include "{ng_modelcard}"',
            f".temp {temp_c}",
            f"Vd d 0 {vdd}",
            f"Vg g 0 PULSE(0 {vdd} 500p {rise_time:.0e} {rise_time:.0e} {width:.0e} {period:.0e})",
            "Vs s 0 0",
            "Ve e 0 0",
            f"N1 d g s e {model_name}",
            f".tran {t_step:.0e} {t_stop:.0e}",
            ".end",
        ]

    raw_path = out_dir / "tran.raw"
    net_path = out_dir / "tran.cir"
    log_path = out_dir / "tran.log"
    runner_path = out_dir / "runner.cir"

    runner_path.write_text(
        "* ngspice runner\n"
        ".control\n"
        f"osdi {OSDI_PATH}\n"
        f"source {net_path}\n"
        "set filetype=ascii\n"
        "run\n"
        f"write {raw_path} v(d) v(g) v(s) v(e) i(vd) i(vg) i(vs) i(ve)\n"
        ".endc\n"
        ".end\n"
    )
    net_path.write_text("\n".join(net))

    try:
        res = subprocess.run(
            [NGSPICE_BIN, "-b", "-o", str(log_path), str(runner_path)],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"NGSPICE timed out after 120s (tag={tag}). See log: {log_path}"
        )
    if res.returncode != 0:
        raise RuntimeError(
            f"NGSPICE transient failed (tag={tag}):\n{res.stdout}\n{res.stderr}\n"
            f"See log: {log_path}"
        )

    return parse_ngspice_raw(raw_path)


def parse_ngspice_raw(raw_path: Path) -> Dict[str, np.ndarray]:
    """Parse NGSPICE ASCII raw file into dict of numpy arrays.

    Handles the standard NGSPICE raw format:
        Variables: (index, name, type)
        Values: (index followed by variable values, one per line)
    """
    with raw_path.open() as f:
        content = f.read()

    lines = content.splitlines()

    # Parse header: extract variable names
    headers: List[str] = []
    n_points = 0
    data_start = 0

    for i, line in enumerate(lines):
        if line.startswith("No. Variables:"):
            n_vars = int(line.split(":")[1].strip())
        elif line.startswith("No. Points:"):
            n_points = int(line.split(":")[1].strip())
        elif line.startswith("Variables:"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("Values:"):
                parts = lines[j].split()
                if len(parts) >= 2:
                    headers.append(parts[1])
                j += 1
            data_start = j + 1
            break

    if not headers or n_points == 0:
        raise RuntimeError(f"Could not parse NGSPICE raw file: {raw_path}")

    # Parse values — collect into list of rows, then convert to numpy
    # NGSPICE raw format per data point:
    #   " <index>\t<first_value>"   ← first line has integer index + tab + value
    #   "\t<value_2>"              ← subsequent values one per line, tab-indented
    #   "\t<value_3>"
    #   ...
    # The integer index is NOT a data variable and must be skipped.
    n_vars = len(headers)
    all_rows: List[List[float]] = []
    current_row: List[float] = []

    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        # Detect "index\tvalue" lines: they start with an integer followed by tab
        parts = stripped.split('\t')
        if len(parts) == 2:
            # Could be "<int_index>\t<float_value>" — skip the index
            try:
                int(parts[0])          # Verify first part is integer index
                val = float(parts[1])  # Take only the float value
                current_row.append(val)
                if len(current_row) == n_vars:
                    all_rows.append(current_row)
                    current_row = []
                continue
            except (ValueError, IndexError):
                pass
        # Regular line: just a single float value (tab-indented)
        try:
            val = float(stripped)
            current_row.append(val)
            if len(current_row) == n_vars:
                all_rows.append(current_row)
                current_row = []
        except ValueError:
            pass

    if not all_rows:
        raise RuntimeError(f"No data points parsed from {raw_path}")

    data = np.array(all_rows)  # shape: (n_points, n_vars)

    result: Dict[str, np.ndarray] = {}
    for i, h in enumerate(headers):
        result[h] = data[:, i]

    return result


def run_dc_comparison(
    tech_name: str,
    device_type: str,
    bias: Dict[str, float],
    tag: str,
    *,
    outputs: Optional[List[str]] = None,
    temp_c: float = 27.0,
    temp_k: Optional[float] = None,
    check_off_state: bool = False,
    check_ids: bool = False,
    rel_tol: float = REL_TOL,
    abs_tol_override: Optional[Dict[str, float]] = None,
) -> None:
    """Run PyCMG eval_dc vs NGSPICE OP comparison for a single bias point.

    Replaces the repeated pattern: get_tech_modelcard -> run_ngspice_op ->
    Model -> Instance -> eval_dc -> assert_close for each output.

    Args:
        tech_name: Key from conftest ALL_TECHNOLOGIES
        device_type: "nmos" or "pmos"
        bias: Terminal voltages {"d": ..., "g": ..., "s": ..., "e": ...}
        tag: Unique tag for NGSPICE output directory
        outputs: List of output keys to compare (default: id, gm, gds, gmb, qg, qd, qs, qb)
        temp_c: Temperature in Celsius (for NGSPICE)
        temp_k: Temperature in Kelvin (for PyCMG). If None, derived from temp_c.
        check_off_state: If True, use relaxed leakage-ratio comparison for id
        check_ids: If True, also verify ids = id - is consistency
        rel_tol: Relative tolerance override
        abs_tol_override: Per-output absolute tolerance overrides {key: tol}
    """
    # Avoid circular import: conftest imports from helpers, so import lazily
    from tests.conftest import ALL_TECHNOLOGIES, get_tech_modelcard

    tech = ALL_TECHNOLOGIES[tech_name]
    modelcard, model_name, inst_params = get_tech_modelcard(tech_name, device_type)
    vdd = tech["vdd"]

    if temp_k is None:
        temp_k = temp_c + 273.15

    if outputs is None:
        outputs = ["id", "gm", "gds", "gmb", "qg", "qd", "qs", "qb"]

    # NGSPICE reference
    ng = run_ngspice_op(
        modelcard, model_name, inst_params,
        bias["d"], bias["g"], bias["s"], bias["e"],
        temp_c=temp_c, tag=tag,
    )

    # PyCMG
    model = Model(str(OSDI_PATH), str(modelcard), model_name)
    kwargs: Dict[str, Any] = {"params": inst_params}
    if abs(temp_k - 300.15) > 0.01:
        kwargs["temperature"] = temp_k
    inst = Instance(model, **kwargs)
    py = inst.eval_dc(bias)

    prefix = f"{tech_name}/{device_type}/{tag}"

    for key in outputs:
        kw: Dict[str, Any] = {"rel_tol": rel_tol}
        if abs_tol_override and key in abs_tol_override:
            kw["abs_tol"] = abs_tol_override[key]
        assert_close(f"{prefix}/{key}", py[key], ng[key], **kw)

    if check_off_state:
        _leakage_floor = 1e-12
        if abs(ng["id"]) > _leakage_floor and abs(py["id"]) > _leakage_floor:
            ratio = abs(py["id"] / ng["id"])
            assert 0.1 < ratio < 10.0, (
                f"{prefix}: PyCMG/NGSPICE off-state id ratio {ratio:.2f} outside [0.1, 10.0]"
            )

    if check_ids:
        ids_from_components = py["id"] - py["is"]
        assert abs(py["ids"] - ids_from_components) < 1e-15, (
            f"{prefix}: ids ({py['ids']:.3e}) != id - is ({ids_from_components:.3e})"
        )
        ng_ids = ng["id"] - ng["is"]
        assert_close(f"{prefix}/ids", py["ids"], ng_ids, rel_tol=rel_tol)
