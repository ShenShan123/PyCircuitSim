"""One-at-a-time (OAT) sensitivity analysis for BSIM-CMG process parameters.

Identifies the most influential model parameters on I-V, Q-V, and C-V
characteristics by perturbing each parameter independently and measuring
the normalized change in device outputs.  This is the first step toward
modeling process variation vs. device characteristics for NN compact models.
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from typing import Dict, List

from .osdi_types import (
    ACCESS_FLAG_SET,
    PARA_KIND_MASK,
    PARA_KIND_MODEL,
    PARA_TY_MASK,
    PARA_TY_REAL,
    OsdiDescriptor,
)
from .core import OsdiModel
from .sweep import OUTPUT_KEYS, build_nodes

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_CATEGORIES: Dict[str, List[str]] = {
    "iv": ["ids", "gm", "gds", "gmb"],
    "qv": ["qg", "qd", "qs", "qb"],
    "cv": ["cgg", "cgd", "cgs", "cdg", "cdd"],
}

# Instance/geometry parameters that should never be perturbed.
_SKIP_PARAMS: set[str] = {
    "l", "w", "tfin", "nfin", "devtype", "nf", "multi",
    "lmin", "lmax", "wmin", "wmax",
    "tfinmin", "tfinmax", "nfinmin", "nfinmax",
    "nfmulti", "geomod", "rgeomod", "minr",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ParamInfo:
    """Metadata for a single OSDI model parameter."""

    index: int    # Position in desc.param_opvar array
    name: str     # Lowercase parameter name
    value: float  # Baseline value read from the model buffer


@dataclass
class SensitivityResult:
    """Results from OAT sensitivity analysis.

    Attributes:
        param_names: All analyzed parameter names.
        sensitivities: ``{param_name: {output_key: avg_abs_normalized_sensitivity}}``.
        rankings: ``{category: [param_names ordered by descending score]}``.
        bias_points: Representative bias points used for evaluation.
        delta_fraction: Perturbation fraction that was applied.
    """

    param_names: List[str]
    sensitivities: Dict[str, Dict[str, float]]
    rankings: Dict[str, List[str]]
    bias_points: List[Dict[str, float]]
    delta_fraction: float


# ---------------------------------------------------------------------------
# Parameter enumeration
# ---------------------------------------------------------------------------


def enumerate_model_params(
    desc: OsdiDescriptor,
    model: OsdiModel,
) -> List[ParamInfo]:
    """Discover all real-valued model-level parameters and their current values.

    Filters out instance/geometry parameters and parameters with near-zero
    baseline values (for which relative perturbation is meaningless).

    Args:
        desc: OSDI descriptor with parameter metadata.
        model: OsdiModel whose buffer holds current parameter values.

    Returns:
        List of :class:`ParamInfo` for candidate parameters.
    """
    params: List[ParamInfo] = []

    for i in range(desc.num_params):
        param = desc.param_opvar[i]

        # Must be a model-level real parameter
        if (param.flags & PARA_KIND_MASK) != PARA_KIND_MODEL:
            continue
        if (param.flags & PARA_TY_MASK) != PARA_TY_REAL:
            continue

        name_raw = param.name[0].decode("utf-8", errors="replace") if param.name else ""
        if not name_raw:
            continue
        name = name_raw.lower()

        if name in _SKIP_PARAMS:
            continue

        # Read current value from model buffer (use ACCESS_FLAG_SET which is
        # known to return a valid pointer for model params — same as apply_param)
        ptr = desc.access(None, model.data(), i, ACCESS_FLAG_SET)
        if not ptr:
            continue
        value = float(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))[0])

        # Skip near-zero values (relative perturbation undefined)
        if abs(value) < 1e-30:
            continue

        params.append(ParamInfo(index=i, name=name, value=value))

    return params


# ---------------------------------------------------------------------------
# Bias point construction
# ---------------------------------------------------------------------------


def _build_bias_points(
    vdd: float,
    vth_mag: float,
    device_type: str,
) -> List[Dict[str, float]]:
    """Build representative bias points covering major operating regions.

    Returns 4 bias points: subthreshold, linear, saturation, strong inversion.
    """
    return [
        # Subthreshold: Vg < Vth, moderate Vd
        build_nodes(vth_mag * 0.5, vdd * 0.5, 0.0, vdd, device_type),
        # Linear: Vg > Vth, small Vd
        build_nodes(vdd * 0.8, vdd * 0.1, 0.0, vdd, device_type),
        # Saturation: Vg above Vth, Vd = Vdd
        build_nodes(vdd * 0.8, vdd, 0.0, vdd, device_type),
        # Strong inversion: Vg = Vdd, Vd = Vdd/2
        build_nodes(vdd, vdd * 0.5, 0.0, vdd, device_type),
    ]


# ---------------------------------------------------------------------------
# Core sensitivity computation
# ---------------------------------------------------------------------------


def compute_sensitivity(
    osdi_path: str,
    modelcard_path: str,
    model_name: str,
    inst_params: Dict[str, float],
    vdd: float,
    device_type: str,
    temperature: float = 300.15,
    delta_fraction: float = 0.05,
    top_n: int = 9,
    verbose: int = 1,
) -> SensitivityResult:
    """Run OAT sensitivity analysis on all real-valued model parameters.

    For each candidate parameter, applies a central-difference perturbation
    of ``+/- delta_fraction * value`` and measures the normalized output
    change at several representative bias points.

    A **fresh Model** is created for every perturbation direction to avoid
    shared-buffer contamination between perturbations.

    Args:
        osdi_path: Path to the compiled OSDI binary.
        modelcard_path: Path to the model card file.
        model_name: SPICE model name (e.g., ``"nmos_rvt"``).
        inst_params: Instance parameters (L, TFIN, NFIN, etc.).
        vdd: Nominal supply voltage.
        device_type: ``"nmos"`` or ``"pmos"``.
        temperature: Operating temperature in Kelvin (default 300.15 K = 27 C).
        delta_fraction: Relative perturbation size (default 0.05 = 5%).
        top_n: Number of top parameters to report per category.
        verbose: Verbosity (0=silent, 1=progress, 2=detail).

    Returns:
        :class:`SensitivityResult` with sensitivities and rankings.
    """
    from .model import Instance, Model
    from .sweep import find_threshold

    # ------------------------------------------------------------------
    # 1. Baseline evaluation
    # ------------------------------------------------------------------
    baseline_model = Model(osdi_path, modelcard_path, model_name)
    baseline_inst = Instance(
        baseline_model, params=inst_params, temperature=temperature,
    )
    vth_mag = find_threshold(baseline_inst, vdd, device_type)
    bias_points = _build_bias_points(vdd, vth_mag, device_type)

    baseline_results: List[Dict[str, float]] = []
    for bp in bias_points:
        baseline_results.append(baseline_inst.eval_dc(bp))

    # Sanity check: if baseline currents are NaN, the model is broken at
    # this configuration (e.g., invalid binning params at this L).
    nan_count = sum(
        1 for r in baseline_results if math.isnan(r.get("ids", 0.0))
    )
    if nan_count == len(baseline_results):
        import warnings  # rare path: only on broken model
        warnings.warn(
            f"Baseline eval_dc returns NaN for all bias points. "
            f"The model may be invalid at this geometry. "
            f"Try a different L multiplier or device.",
            stacklevel=2,
        )

    if verbose >= 1:
        print(f"Baseline Vth={vth_mag:.4f}V, {len(bias_points)} bias points")

    # ------------------------------------------------------------------
    # 2. Enumerate candidate parameters
    # ------------------------------------------------------------------
    candidate_params = enumerate_model_params(
        baseline_model.descriptor, baseline_model.model,
    )
    if verbose >= 1:
        print(f"Found {len(candidate_params)} candidate model parameters")

    # ------------------------------------------------------------------
    # 3. Perturb each parameter and compute sensitivity
    # ------------------------------------------------------------------
    sensitivities: Dict[str, Dict[str, float]] = {}
    n_bp = len(bias_points)

    for p_idx, p in enumerate(candidate_params):
        delta = p.value * delta_fraction
        if abs(delta) < 1e-40:
            continue

        # Forward perturbation: fresh Model to avoid buffer contamination
        try:
            model_fwd = Model(osdi_path, modelcard_path, model_name)
            inst_fwd = Instance(
                model_fwd, params=inst_params, temperature=temperature,
                model_overrides={p.name: p.value + delta},
            )
            fwd_results = [inst_fwd.eval_dc(bp) for bp in bias_points]
        except Exception:
            if verbose >= 2:
                print(f"  Skip {p.name}: forward perturbation failed")
            continue

        # Backward perturbation: fresh Model
        try:
            model_bwd = Model(osdi_path, modelcard_path, model_name)
            inst_bwd = Instance(
                model_bwd, params=inst_params, temperature=temperature,
                model_overrides={p.name: p.value - delta},
            )
            bwd_results = [inst_bwd.eval_dc(bp) for bp in bias_points]
        except Exception:
            if verbose >= 2:
                print(f"  Skip {p.name}: backward perturbation failed")
            continue

        # Central-difference normalized sensitivity per output
        param_sens: Dict[str, float] = {}
        for key in OUTPUT_KEYS:
            accum = 0.0
            for bp_i in range(n_bp):
                y_fwd = fwd_results[bp_i][key]
                y_bwd = bwd_results[bp_i][key]
                y_base = baseline_results[bp_i][key]
                dy = y_fwd - y_bwd
                dp = 2.0 * delta
                # Normalized sensitivity: (dy/dp) * (p/y)
                # Skip NaN/Inf values (broken model at this bias)
                if (
                    abs(y_base) > 1e-30
                    and abs(p.value) > 1e-30
                    and math.isfinite(y_fwd)
                    and math.isfinite(y_bwd)
                    and math.isfinite(y_base)
                ):
                    s = (dy / dp) * (p.value / y_base)
                else:
                    s = 0.0
                accum += abs(s)
            param_sens[key] = accum / n_bp

        sensitivities[p.name] = param_sens

        if verbose >= 2:
            top_key = max(param_sens, key=lambda k: param_sens[k])
            print(
                f"  [{p_idx + 1}/{len(candidate_params)}] {p.name} "
                f"(val={p.value:.3e}): peak output={top_key} "
                f"S={param_sens[top_key]:.3e}"
            )

    if verbose >= 1:
        print(f"Analyzed {len(sensitivities)} parameters successfully")

    # ------------------------------------------------------------------
    # 4. Rank parameters per category
    # ------------------------------------------------------------------
    rankings = rank_parameters(sensitivities, OUTPUT_CATEGORIES, top_n)

    return SensitivityResult(
        param_names=list(sensitivities.keys()),
        sensitivities=sensitivities,
        rankings=rankings,
        bias_points=bias_points,
        delta_fraction=delta_fraction,
    )


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def rank_parameters(
    sensitivities: Dict[str, Dict[str, float]],
    categories: Dict[str, List[str]],
    top_n: int = 9,
) -> Dict[str, List[str]]:
    """Rank parameters by aggregate sensitivity within each output category.

    Args:
        sensitivities: ``{param_name: {output_key: sensitivity}}``.
        categories: ``{category_name: [output_keys]}``.
        top_n: Number of top parameters to return per category.

    Returns:
        ``{category_name: [param_names sorted by descending score]}``.
    """
    rankings: Dict[str, List[str]] = {}
    for cat_name, output_keys in categories.items():
        scores: Dict[str, float] = {}
        for param_name, sens_dict in sensitivities.items():
            score = sum(abs(sens_dict.get(k, 0.0)) for k in output_keys)
            scores[param_name] = score
        ranked = sorted(scores, key=lambda p: scores[p], reverse=True)
        rankings[cat_name] = ranked[:top_n]
    return rankings


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_sensitivity_table(
    result: SensitivityResult,
    category: str,
) -> str:
    """Format a ranked sensitivity table for terminal output.

    Args:
        result: Sensitivity analysis result.
        category: One of ``"iv"``, ``"qv"``, ``"cv"``.

    Returns:
        Multi-line formatted table string.
    """
    output_keys = OUTPUT_CATEGORIES[category]
    ranked = result.rankings[category]

    # Header
    cat_label = {"iv": "I-V", "qv": "Q-V", "cv": "C-V"}.get(category, category)
    col_width = 10
    header = f"{'Rank':<6}{'Parameter':<20}"
    for k in output_keys:
        header += f"{k:>{col_width}}"
    header += f"{'Score':>{col_width}}"

    lines = [
        f"=== {cat_label} Sensitivity (top {len(ranked)}) ===",
        header,
        "-" * len(header),
    ]

    for rank, param_name in enumerate(ranked, 1):
        sens = result.sensitivities[param_name]
        row = f"{rank:<6}{param_name:<20}"
        score = 0.0
        for k in output_keys:
            val = sens.get(k, 0.0)
            score += abs(val)
            row += f"{val:>{col_width}.3e}"
        row += f"{score:>{col_width}.3e}"
        lines.append(row)

    return "\n".join(lines)
