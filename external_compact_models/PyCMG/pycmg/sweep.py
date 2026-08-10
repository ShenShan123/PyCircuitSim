"""Sweep building blocks for PyCMG training data generation.

Provides voltage grid construction, threshold detection, device resolution,
and configuration dataclasses for systematic parameter sweeps across
technology nodes and device variants.
"""

from __future__ import annotations

import csv
import fnmatch
import itertools
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from pycmg.model import Instance
    from pycmg.tech import TechConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_KEYS: List[str] = [
    "id", "ig", "is", "ie", "ids",
    "qg", "qd", "qs", "qb",
    "gm", "gds", "gmb",
    "cgg", "cgd", "cgs", "cdg", "cdd",
]

# NN training target columns — subset of OUTPUT_KEYS (13 of 17)
# Excludes ig, is, ie, ids which are not needed for circuit simulation.
NN_OUTPUT_COLUMNS: List[str] = [
    "id", "gm", "gds", "gmb",
    "qg", "qd", "qs", "qb",
    "cgg", "cgd", "cgs", "cdg", "cdd",
]

GEOM_COLUMNS: List[str] = ["tech", "device", "L", "NFIN", "TFIN", "temp_K"]

VOLTAGE_COLUMNS: List[str] = ["Vg", "Vd", "Vs", "Ve", "Vth"]


def build_all_columns(process_keys: List[str]) -> List[str]:
    """Build the full ordered column list for sweep output.

    Column order: geometry -> sorted process vars -> voltages -> outputs.

    Args:
        process_keys: Process variation parameter names (e.g., ["eot", "toxp"]).

    Returns:
        Ordered list of column names.
    """
    return GEOM_COLUMNS + sorted(process_keys) + VOLTAGE_COLUMNS + OUTPUT_KEYS


# ---------------------------------------------------------------------------
# Voltage helpers
# ---------------------------------------------------------------------------


def build_nodes(
    vg_mag: float,
    vd_mag: float,
    ve_mag: float,
    vdd: float,
    device_type: str,
) -> Dict[str, float]:
    """Convert magnitude-space voltages to actual terminal voltages.

    In magnitude space, all voltages are positive and represent the
    *magnitude* of the bias relative to the source terminal. This function
    maps them to actual node voltages depending on device polarity.

    Args:
        vg_mag: Gate voltage magnitude (0 to Vdd*voltage_scale).
        vd_mag: Drain voltage magnitude (0 to Vdd*voltage_scale).
        ve_mag: Bulk/body voltage magnitude (typically 0).
        vdd: Supply voltage.
        device_type: ``"nmos"`` or ``"pmos"``.

    Returns:
        Dict with keys ``"g"``, ``"d"``, ``"s"``, ``"e"`` mapped to voltages.
    """
    if device_type == "nmos":
        return {"g": vg_mag, "d": vd_mag, "s": 0.0, "e": ve_mag}
    else:
        # PMOS: source at Vdd, drain/gate/bulk reflected
        return {
            "g": vdd - vg_mag,
            "d": vdd - vd_mag,
            "s": vdd,
            "e": vdd - ve_mag,
        }


def build_voltage_grid(
    vdd: float,
    vth_mag: float,
    vg_points: int = 50,
    vd_points: int = 50,
    dense_ratio: float = 0.6,
    voltage_scale: float = 1.0,
    v_min: float = 0.0,
    n_dense_mid: int = 0,
    vth_center: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a non-uniform Vg grid (dense near threshold) and uniform Vd grid.

    Extends the original with three new parameters:
      v_min: Lower voltage bound (default 0.0 = existing behavior). Use -vdd
          for NN source-relative training data that covers NR overshoot.
      n_dense_mid: Extra dense points near the mid-supply crossing
          ((v_min + v_max) / 2). 0 = disabled (default).
      vth_center: Explicit signed threshold center for dense region. When None
          (default), uses ``vth_mag`` (positive, backward-compatible). Pass a
          negative value for PMOS source-relative frame (e.g., -0.2).

    Args:
        vdd: Nominal supply voltage (used for dense region width calculation).
        vth_mag: Threshold voltage magnitude (backward-compat; used when
            vth_center is None).
        vg_points: Total target Vg points (dense + sparse).
        vd_points: Number of Vd points (uniform).
        dense_ratio: Fraction of vg_points in dense threshold region.
        voltage_scale: Upper bound multiplier: v_max = vdd * voltage_scale.
        v_min: Lower voltage bound (signed). Default 0.0.
        n_dense_mid: Extra dense points near the mid-supply crossing.
        vth_center: Signed threshold center. None → use vth_mag.

    Returns:
        Tuple of (vg_array, vd_array), both sorted, no duplicates.
    """
    v_max = vdd * voltage_scale
    _vth = vth_center if vth_center is not None else vth_mag

    # Dense region: ±0.15·Vdd around threshold, clipped to [v_min, v_max]
    dense_lo = max(v_min, _vth - 0.15 * vdd)
    dense_hi = min(v_max, _vth + 0.15 * vdd)

    n_dense = int(vg_points * dense_ratio)
    n_sparse = vg_points - n_dense

    vg_dense = np.linspace(dense_lo, dense_hi, n_dense)
    vg_sparse = np.linspace(v_min, v_max, n_sparse)

    parts = [vg_dense, vg_sparse]
    if n_dense_mid > 0:
        mid = (v_min + v_max) / 2.0
        mid_lo = mid - 0.15 * vdd
        mid_hi = mid + 0.15 * vdd
        parts.append(np.linspace(mid_lo, mid_hi, n_dense_mid))

    vg_all = np.unique(np.concatenate(parts))
    vd_all = np.linspace(v_min, v_max, vd_points)

    return vg_all, vd_all


# ---------------------------------------------------------------------------
# Threshold detection
# ---------------------------------------------------------------------------


def find_threshold(
    inst: Instance,
    vdd: float,
    device_type: str = "nmos",
    n_coarse: int = 30,
) -> float:
    """Find threshold voltage via peak-gm method (coarse sweep).

    Sweeps Vg_mag from 0 to Vdd at Vd = Vdd/2 in magnitude space,
    evaluates DC at each point, and returns the Vg_mag where |gm| is
    maximum. This is a standard approximation of Vth used in compact
    model characterization.

    Args:
        inst: Initialized PyCMG Instance.
        vdd: Supply voltage.
        device_type: ``"nmos"`` or ``"pmos"``.
        n_coarse: Number of uniform sweep points.

    Returns:
        Threshold voltage magnitude (positive float).
    """
    vg_sweep = np.linspace(0.0, vdd, n_coarse)
    vd_mag = vdd / 2.0

    best_vg = 0.0
    best_gm = 0.0

    for vg_mag in vg_sweep:
        nodes = build_nodes(vg_mag, vd_mag, 0.0, vdd, device_type)
        result = inst.eval_dc(nodes)
        gm_abs = abs(result["gm"])
        if gm_abs > best_gm:
            best_gm = gm_abs
            best_vg = float(vg_mag)

    return best_vg


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def resolve_devices(
    device_filter: Dict[str, List[str]] | None,
    tech_name: str,
    tech_config: TechConfig,
) -> List[str]:
    """Resolve which devices to sweep for a given technology.

    Supports explicit device names and glob patterns (e.g., ``"nmos_*"``).
    Missing devices are skipped with a warning rather than raising an error.

    Args:
        device_filter: Optional mapping from tech name to device name patterns.
            ``None`` means all devices in the technology.
        tech_name: Technology identifier (e.g., ``"ASAP7"``).
        tech_config: Technology configuration with device registry.

    Returns:
        Sorted list of resolved canonical device names.
    """
    available = sorted(tech_config.devices.keys())

    if device_filter is None:
        return available

    patterns = device_filter.get(tech_name)
    if patterns is None:
        return available

    resolved: list[str] = []
    for pattern in patterns:
        # Check if it's a glob pattern
        if any(c in pattern for c in ("*", "?", "[", "]")):
            matches = [d for d in available if fnmatch.fnmatch(d, pattern)]
            if not matches:
                warnings.warn(
                    f"Pattern '{pattern}' matched no devices in {tech_name}. "
                    f"Available: {', '.join(available)}",
                    stacklevel=2,
                )
            resolved.extend(matches)
        else:
            # Exact name
            if pattern in tech_config.devices:
                resolved.append(pattern)
            else:
                warnings.warn(
                    f"Device '{pattern}' not found in {tech_name}. "
                    f"Available: {', '.join(available)}",
                    stacklevel=2,
                )

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for d in resolved:
        if d not in seen:
            seen.add(d)
            unique.append(d)

    return unique


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SweepConfig:
    """Configuration for a full training-data sweep.

    Attributes:
        techs: Technology names to sweep (e.g., ``["ASAP7", "TSMC7"]``).
        devices: Optional per-tech device filter (supports glob patterns).
            ``None`` means all devices in each technology.
        sweep_geometry: When ``True``, use PDK-defined (L, NFIN) combinations
            from the technology modelcard bin boundaries.  Each variant
            contributes ``(lmin, nfinmin)`` and ``(lmin, nfinmax)`` points.
            When ``False``, use a single default ``(min_l, 1.0)`` per device.
        temperatures: Operating temperatures in Kelvin.
        vg_points: Number of gate voltage grid points.
        vd_points: Number of drain voltage grid points.
        ve_values: Body bias voltage magnitudes to sweep.
        process_vars: Optional process variation parameters to sweep.
            Maps parameter name to list of values.
        dense_ratio: Fraction of vg_points in the threshold-dense region.
        n_coarse: Number of points for coarse threshold detection.
        voltage_scale: Voltage range multiplier. Use 2.0 to extend sweeps
            to 2*VDD for NN simulator convergence training.
    """

    techs: List[str]
    devices: Dict[str, List[str]] | None = None
    sweep_geometry: bool = True
    temperatures: List[float] = field(
        default_factory=lambda: [233.15, 273.15, 300.15, 358.15, 398.15]
    )
    vg_points: int = 50
    vd_points: int = 50
    ve_values: List[float] = field(default_factory=lambda: [0.0])
    process_vars: Dict[str, List[float]] | None = None
    dense_ratio: float = 0.6
    n_coarse: int = 30
    voltage_scale: float = 1.0


@dataclass
class SweepResult:
    """Container for sweep output data.

    Attributes:
        columns: Ordered column names matching ``build_all_columns()`` output.
        data: List of rows, each row is a list of ``str | float`` values.
        metadata: Arbitrary metadata dict (e.g., sweep config, timing info).
    """

    columns: List[str]
    data: List[list]  # str | float
    metadata: Dict[str, object]


# ---------------------------------------------------------------------------
# Core sweep loop
# ---------------------------------------------------------------------------


def sweep_dc(osdi_path: str, config: SweepConfig, verbose: int = 2) -> SweepResult:
    """Run a full DC sweep across technologies, devices, and operating conditions.

    Iterates over the Cartesian product of technologies, devices, gate lengths,
    fin counts, temperatures, process variation combos, body biases, and
    voltage grid points. At each bias point, ``Instance.eval_dc()`` is called
    and the result is appended to the output table.

    Args:
        osdi_path: Path to the compiled OSDI binary (``bsimcmg.osdi``).
        config: Sweep configuration specifying all parameter ranges.
        verbose: Verbosity level.
            0 = silent, 1 = summary only, 2 = per-configuration progress.

    Returns:
        A :class:`SweepResult` containing the full data table, column names,
        and metadata about the sweep.
    """
    from pycmg.model import Instance, Model
    from pycmg.tech import TECH_REGISTRY, get_tech_config, resolve_modelcard

    resolved_techs: List[str] = (
        list(TECH_REGISTRY.keys()) if "all" in config.techs else config.techs
    )

    # Expand process grid (Cartesian product)
    if config.process_vars:
        pkeys = sorted(config.process_vars.keys())
        process_combos: List[Dict[str, float]] = [
            dict(zip(pkeys, c))
            for c in itertools.product(*(config.process_vars[k] for k in pkeys))
        ]
    else:
        pkeys: List[str] = []
        process_combos = [{}]

    # ASAP7 reference PDK for NFIN boundaries (TSMC7)
    _REF_PDK = "modelcards/TSMC7/cln7_1d8_sp_v1d2_2p2.l"

    data: List[list] = []
    columns = build_all_columns(pkeys)
    count = 0

    for tech_name in resolved_techs:
        tech = get_tech_config(tech_name)
        device_list = resolve_devices(config.devices, tech_name, tech)

        for device_name in device_list:
            device = tech.get_device(device_name)
            device_type = "nmos" if device.inst_params.get("DEVTYPE", 1.0) != 0 else "pmos"

            # Get PDK-defined (L, NFIN) geometry combinations
            if config.sweep_geometry:
                if device.pdk_device is not None:
                    geometry_combos = device.get_geometry_combos(tech.pdk_path)
                else:
                    # ASAP7: use TSMC7 NFIN boundaries matched per device type
                    ref_dev = "nch_svt_mac" if device_type == "nmos" else "pch_svt_mac"
                    geometry_combos = device.get_geometry_combos(
                        ref_pdk_path=_REF_PDK, ref_device_name=ref_dev,
                    )
            else:
                min_l = device.get_min_l(tech.pdk_path)
                geometry_combos = [(min_l, 1.0)]

            for L, nfin in geometry_combos:
                try:
                    modelcard_path = resolve_modelcard(
                        device, tech, L, NFIN=nfin,
                    )
                except (RuntimeError, ValueError) as e:
                    if verbose >= 2:
                        print(
                            f"  Warning: {tech_name} {device_name} "
                            f"L={L * 1e9:.0f}nm NFIN={nfin:.0f}: {e}, skipping"
                        )
                    continue

                model = Model(osdi_path, modelcard_path, device.model_name)
                params = {**device.inst_params, "L": L, "NFIN": nfin}

                for proc_combo in process_combos:
                    for temp in config.temperatures:
                        inst = Instance(
                            model,
                            params=params,
                            temperature=temp,
                            model_overrides=proc_combo if proc_combo else None,
                        )

                        t0 = time.time()
                        vth_mag = find_threshold(
                            inst, tech.vdd, device_type, config.n_coarse
                        )
                        vg_arr, vd_arr = build_voltage_grid(
                            tech.vdd,
                            vth_mag,
                            config.vg_points,
                            config.vd_points,
                            config.dense_ratio,
                            config.voltage_scale,
                        )

                        n_points = 0
                        for ve in config.ve_values:
                            for vg_mag in vg_arr:
                                for vd_mag in vd_arr:
                                    nodes = build_nodes(
                                        vg_mag, vd_mag, ve, tech.vdd, device_type
                                    )
                                    result = inst.eval_dc(nodes)
                                    row: list = [
                                        tech_name,
                                        device_name,
                                        L,
                                        nfin,
                                        tech.tfin,
                                        temp,
                                        *[proc_combo.get(k, 0.0) for k in pkeys],
                                        nodes["g"],
                                        nodes["d"],
                                        nodes["s"],
                                        nodes["e"],
                                        vth_mag,
                                        *[result[k] for k in OUTPUT_KEYS],
                                    ]
                                    data.append(row)
                                    n_points += 1

                        count += 1
                        if verbose >= 2:
                            elapsed = time.time() - t0
                            proc_str = " ".join(
                                f"{k}={v:.2e}" for k, v in proc_combo.items()
                            )
                            print(
                                f"[{count}] {tech_name} {device_name} "
                                f"L={L * 1e9:.0f}nm NFIN={nfin:.0f} "
                                f"T={temp - 273.15:.0f}C "
                                f"{proc_str} Vth={vth_mag:.3f}V "
                                f"{n_points}pts ({elapsed:.1f}s)"
                            )

    if verbose >= 1:
        print(f"Done. {len(data)} rows across {count} configurations.")

    return SweepResult(
        columns=columns,
        data=data,
        metadata={
            "techs": resolved_techs,
            "total_configs": count,
            "total_rows": len(data),
        },
    )


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def _format_cell(value: object) -> str:
    """Format a single cell value for CSV output.

    Strings are written as-is; floats use ``%.6e`` scientific notation.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return f"{value:.6e}"
    return str(value)


def to_csv(
    results: SweepResult,
    output_dir: str,
    split_by: str = "tech",
) -> List[str]:
    """Write SweepResult to CSV files.

    Args:
        results: Sweep output from :func:`sweep_dc`.
        output_dir: Directory where CSV files will be written. Created if it
            does not exist.
        split_by: How to split data into files:
            - ``"tech"``: one file per technology (e.g., ``ASAP7_dc.csv``).
            - ``"device"``: one file per tech+device
              (e.g., ``ASAP7_nmos_rvt_dc.csv``).
            - ``"none"``: single file (``training_data_dc.csv``).

    Returns:
        List of absolute paths to the written CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Group rows by split key
    groups: Dict[str, List[list]] = {}

    if split_by == "none":
        groups["training_data_dc"] = results.data
    else:
        for row in results.data:
            if split_by == "device":
                key = f"{row[0]}_{row[1]}_dc"
            else:
                # Default: split by tech
                key = f"{row[0]}_dc"
            groups.setdefault(key, []).append(row)

    written: List[str] = []
    for name, rows in sorted(groups.items()):
        filepath = os.path.join(output_dir, f"{name}.csv")
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(results.columns)
            for row in rows:
                writer.writerow([_format_cell(v) for v in row])
        written.append(os.path.abspath(filepath))

    return written


def save_npz(
    inputs: np.ndarray,
    geometry: np.ndarray,
    outputs: np.ndarray,
    output_path: "str | Path",
    metadata: "Optional[Dict[str, object]]" = None,
    sample_class: "Optional[np.ndarray]" = None,
) -> None:
    """Save NN training arrays to a .npz file.

    The .npz layout is the contract between pycmg data generation and the
    nn_model training pipeline:
      inputs       (N, 4)  — source-relative terminal voltages [Vd, Vg, Vs, Vb]
      geometry     (N, 15) — [NFIN, L, T, PHIG, U0, VSAT, EOT, ETA0, CIT, RDSW,
                              CFS, TOXP, CGSL, UA, EU]
      outputs      (N, 13) — NN_OUTPUT_COLUMNS order
      sample_class (N,)    — int8 codes (B1; only present when supplied)

    Optional metadata keys are saved as ``meta_<key>`` arrays.

    Args:
        inputs: (N, 4) float64 array.
        geometry: (N, 15) float64 array.
        outputs: (N, 13) float64 array.
        output_path: Destination file path.
        metadata: Optional dict of scalar/array metadata.
        sample_class: Optional (N,) int8 array tagging each row's origin
            (anchor / vds_zero / subthresh / small_vds / grid / hot /
            lhs). Saved as a top-level array so consumers can subset
            rows for B2 slope loss or B6 universal overlay without
            reading metadata.
    """
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_dict: Dict[str, object] = {
        "inputs": inputs,
        "geometry": geometry,
        "outputs": outputs,
    }
    if sample_class is not None:
        sc = np.asarray(sample_class)
        if sc.shape[0] != inputs.shape[0]:
            raise ValueError(
                f"sample_class length {sc.shape[0]} != inputs length "
                f"{inputs.shape[0]}"
            )
        save_dict["sample_class"] = sc.astype(np.int8, copy=False)
    if metadata:
        for k, v in metadata.items():
            save_dict[f"meta_{k}"] = np.array(v)
    np.savez(output_path, **save_dict)
    final_path = output_path if output_path.endswith(".npz") else output_path + ".npz"
    size_kb = os.path.getsize(final_path) / 1024
    print(f"  Saved {inputs.shape[0]} samples → {output_path} ({size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def generate_dataset(
    osdi_path: str,
    techs: Optional[List[str]] = None,
    devices: Optional[Dict[str, List[str]]] = None,
    output_dir: str = "./training_data",
    sweep_geometry: bool = True,
    temperatures: Optional[List[float]] = None,
    vg_points: int = 50,
    vd_points: int = 50,
    ve_values: Optional[List[float]] = None,
    process_vars: Optional[Dict[str, List[float]]] = None,
    dense_ratio: float = 0.6,
    split_by: str = "tech",
    verbose: int = 2,
    voltage_scale: float = 1.0,
) -> List[str]:
    """Generate training data CSVs from a DC sweep.

    Convenience wrapper that builds a :class:`SweepConfig`, calls
    :func:`sweep_dc`, and writes the results with :func:`to_csv`.

    Args:
        osdi_path: Path to the compiled OSDI binary.
        techs: Technology names to sweep. Defaults to ``["all"]``.
        devices: Optional per-tech device filter (supports glob patterns).
        output_dir: Output directory for CSV files.
        sweep_geometry: When ``True`` (default), use PDK-defined (L, NFIN)
            combinations.  When ``False``, use single default per device.
        temperatures: Temperatures in Kelvin. ``None`` uses SweepConfig defaults.
        vg_points: Number of gate voltage grid points.
        vd_points: Number of drain voltage grid points.
        ve_values: Body bias voltage magnitudes. ``None`` uses SweepConfig defaults.
        process_vars: Process variation parameters to sweep.
        dense_ratio: Fraction of vg_points in the threshold-dense region.
        split_by: CSV split strategy (``"tech"``, ``"device"``, or ``"none"``).
        verbose: Verbosity level (0=silent, 1=summary, 2=per-config).
        voltage_scale: Voltage range multiplier (default 1.0). Use 2.0 to
            extend sweeps to 2*VDD for NN simulator convergence training.

    Returns:
        List of absolute paths to the written CSV files.
    """
    if techs is None:
        techs = ["all"]

    kwargs: Dict[str, object] = {
        "techs": techs,
        "devices": devices,
        "sweep_geometry": sweep_geometry,
        "vg_points": vg_points,
        "vd_points": vd_points,
        "process_vars": process_vars,
        "dense_ratio": dense_ratio,
        "voltage_scale": voltage_scale,
    }
    if temperatures is not None:
        kwargs["temperatures"] = temperatures
    if ve_values is not None:
        kwargs["ve_values"] = ve_values

    config = SweepConfig(**kwargs)  # type: ignore[arg-type]
    result = sweep_dc(osdi_path, config, verbose=verbose)
    return to_csv(result, output_dir, split_by=split_by)
