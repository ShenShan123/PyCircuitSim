"""Deep harness module for paired NN/LEVEL=72 simple-circuit experiments.

Callers choose a catalog case, technology, and corner.  This module hides deck
rendering, topology parity, both simulator adapters, trace alignment, domain
metrics, and accepted-reference support diagnostics behind that small seam.
"""
from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from tests.common.base import (
    OSDI_PATH, bake_inst_params, deck_tokens, template_deck, render_deck_text,
    run_ngspice_subprocess,
)
from tests.common.circuit_benchmarks import (
    BenchTech, active_model_label, active_model_level, active_model_name,
    bench_variant, full_metrics, usable_vts,
    get_baked_modelcard, parse_netlist, run_directnet_dc_sweep,
    run_directnet_transient, run_ngspice_wrdata,
)
from tests.common.gate_result import GateResult
from tests.common.simple_circuit_catalog import (
    AnalysisSpec, CircuitCase, DeviceRoleSpec, DIAGNOSTIC,
)


@dataclass(frozen=True)
class Corner:
    """Technology-independent stress applied before a deck is rendered."""

    name: str
    vdd_scale: float = 1.0
    temperature_c: Optional[float] = None
    body_reverse_frac: float = 0.0
    nfin: Optional[int] = None
    nfin_p: Optional[int] = None
    l_nmos: Optional[float] = None
    l_pmos: Optional[float] = None
    vt_mode: str = ""


CORNERS: Dict[str, Corner] = {
    "nominal": Corner("nominal"),
    "temp_cold": Corner("temp_cold", temperature_c=-25.0),
    "temp_hot": Corner("temp_hot", temperature_c=125.0),
    "vdd_low": Corner("vdd_low", vdd_scale=0.85),
    "vdd_high": Corner("vdd_high", vdd_scale=1.10),
    "body_reverse": Corner("body_reverse", body_reverse_frac=0.10),
    "pn_n3p2": Corner("pn_n3p2", nfin=3, nfin_p=2),
    "pn_n2p3": Corner("pn_n2p3", nfin=2, nfin_p=3),
    "joint_hot_lowvdd": Corner(
        "joint_hot_lowvdd", vdd_scale=0.90, temperature_c=125.0,
        nfin=3, nfin_p=2, l_nmos=20e-9, l_pmos=20e-9,
    ),
    "vt_alternate": Corner("vt_alternate", vt_mode="alternate"),
    "vt_asymmetric": Corner("vt_asymmetric", vt_mode="asymmetric"),
    "ln_20": Corner("ln_20", l_nmos=20e-9),
    "lp_16": Corner("lp_16", l_pmos=16e-9),
    "nfin_high": Corner("nfin_high", nfin=5, nfin_p=5),
}


@dataclass(frozen=True)
class RunSpec:
    """Explicit NN family and provenance for one experiment invocation."""

    model_level: int
    model_family: str
    checkpoint_pins: Tuple[Tuple[str, str], ...] = ()
    campaign_manifest_sha256: str = ""
    omp_threads: int = 1
    mkl_threads: int = 1
    torch_threads: int = 1

    def __post_init__(self) -> None:
        expected = {
            73: "DirectNet", 74: "BSIM-AR",
            75: "DirectNet-Full", 76: "BSIM-AR-Full",
        }
        if self.model_level not in expected:
            raise ValueError(f"unsupported NN model level {self.model_level}")
        if self.model_family != expected[self.model_level]:
            raise ValueError(
                f"model family/level mismatch: {self.model_family!r} / "
                f"{self.model_level}"
            )
        if self.campaign_manifest_sha256 and not re.fullmatch(
            r"[0-9a-f]{64}", self.campaign_manifest_sha256,
        ):
            raise ValueError("campaign manifest digest must be 64 lowercase hex chars")
        if min(self.omp_threads, self.mkl_threads, self.torch_threads) < 1:
            raise ValueError("thread counts must be positive")

    def result_fields(self) -> Dict[str, Any]:
        """Return the provenance fields copied onto every result row."""
        return {
            "model_family": self.model_family,
            "model_level": self.model_level,
            "checkpoint_pins": dict(self.checkpoint_pins),
            "campaign_manifest_sha256": self.campaign_manifest_sha256,
            "thread_settings": {
                "omp": self.omp_threads,
                "mkl": self.mkl_threads,
                "torch": self.torch_threads,
            },
        }

    def validate_checkpoint_pins(self, checkpoint_dir: Path) -> None:
        """Fail before reference work when an explicit NN bundle is incomplete."""
        for polarity, raw_pin in self.checkpoint_pins:
            raw = Path(raw_pin)
            base = raw if raw.is_absolute() else checkpoint_dir / raw
            if base.name.endswith("_best.pt"):
                model = base
                stem = base.with_name(base.name[:-len("_best.pt")])
            else:
                model = base.with_name(base.name + "_best.pt")
                stem = base
            required = [
                model,
                stem.with_name(stem.name + "_norm.npz"),
                model.with_name(model.name + ".complete"),
            ]
            if self.model_level in {74, 76}:
                required.append(stem.with_name(stem.name + "_config.npz"))
            missing = [path for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"explicit {polarity} checkpoint bundle is incomplete: "
                    + ", ".join(str(path) for path in missing)
                )

    @classmethod
    def from_environment(cls) -> "RunSpec":
        """Resolve the effective model and execution settings once per run."""
        level = active_model_level()
        tag = {73: "DN", 74: "TF", 75: "DNF", 76: "TFF"}[level]
        pins = []
        for polarity in ("nmos", "pmos"):
            suffix = polarity.upper()
            value = os.environ.get(
                f"PYCIRCUITSIM_NN_CHECKPOINT_{tag}_{suffix}",
                os.environ.get(f"PYCIRCUITSIM_NN_CHECKPOINT_{suffix}", ""),
            )
            if value:
                pins.append((polarity, value))

        def _threads(name: str) -> int:
            raw = os.environ.get(name, "1")
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError(f"{name}={raw!r} is not an integer") from exc

        return cls(
            model_level=level,
            model_family=active_model_name(),
            checkpoint_pins=tuple(pins),
            campaign_manifest_sha256=os.environ.get(
                "PYCIRCUITSIM_CAMPAIGN_MANIFEST_SHA256", "",
            ),
            omp_threads=_threads("OMP_NUM_THREADS"),
            mkl_threads=_threads("MKL_NUM_THREADS"),
            torch_threads=_threads("PYCIRCUITSIM_TORCH_THREADS"),
        )


@dataclass
class Trace:
    """Engine-neutral accepted analysis trace."""

    axis_name: str
    axis: np.ndarray
    signals: Dict[str, np.ndarray]
    converged: bool = True
    partial: bool = False
    reference: bool = False
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(
        self,
        *,
        expected_start: Optional[float] = None,
        expected_stop: Optional[float] = None,
        endpoint_tolerance: float = 0.0,
        max_step: Optional[float] = None,
        minimum_points: Optional[int] = None,
    ) -> None:
        """Fail loud on incomplete, ragged, or non-finite evidence."""
        self.axis = np.asarray(self.axis, dtype=float)
        if self.axis.ndim != 1 or self.axis.size == 0:
            raise ValueError("trace axis must be a non-empty vector")
        if minimum_points is not None:
            if minimum_points < 1:
                raise ValueError("trace minimum point count must be positive")
            if self.axis.size < minimum_points:
                raise ValueError(
                    f"trace has {self.axis.size} points; expected at least "
                    f"{minimum_points}"
                )
        if not np.all(np.isfinite(self.axis)):
            raise ValueError("trace axis contains NaN/Inf")
        if self.axis.size > 1:
            delta = np.diff(self.axis)
            if not (np.all(delta > 0.0) or np.all(delta < 0.0)):
                raise ValueError("trace axis must be strictly monotonic")
            if max_step is not None:
                if not math.isfinite(max_step) or max_step <= 0.0:
                    raise ValueError("trace maximum step must be finite and positive")
                largest = float(np.max(np.abs(delta)))
                if largest > max_step * (1.0 + 1e-9) + 1e-18:
                    raise ValueError(
                        f"trace axis gap {largest:g} exceeds declared step "
                        f"{max_step:g}"
                    )
        span = abs(float(self.axis[-1] - self.axis[0]))
        atol = max(1e-18, span * 1e-9)
        if expected_start is not None and not np.isclose(
            self.axis[0], expected_start, rtol=1e-9,
            atol=max(atol, endpoint_tolerance),
        ):
            raise ValueError(
                f"trace starts at {self.axis[0]:g}, not requested "
                f"{expected_start:g}"
            )
        if expected_stop is not None and not np.isclose(
            self.axis[-1], expected_stop, rtol=1e-9,
            atol=max(atol, endpoint_tolerance),
        ):
            raise ValueError(
                f"trace does not reach requested stop {expected_stop:g}; "
                f"last point is {self.axis[-1]:g}"
            )
        for name, values in self.signals.items():
            array = np.asarray(values)
            if array.ndim != 1 or array.size != self.axis.size:
                raise ValueError(
                    f"trace {name} has {array.size} values for "
                    f"{self.axis.size} axis points")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"trace {name} contains NaN/Inf")
            self.signals[name] = array


_ANALYSIS_SUFFIXES: Dict[str, float] = {
    "t": 1e12, "g": 1e9, "m": 1e6, "k": 1e3,
    "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15,
}


def _analysis_number(raw: str) -> float:
    """Parse the numeric subset accepted in analysis cards."""
    token = raw.strip()
    suffix = token[-1].lower() if len(token) > 1 else ""
    if suffix in _ANALYSIS_SUFFIXES:
        return float(token[:-1]) * _ANALYSIS_SUFFIXES[suffix]
    return float(token)


def analysis_axis_limits(analysis: AnalysisSpec) -> Tuple[float, float]:
    """Return the requested first and last independent-variable values."""
    parts = analysis.card.lstrip(".").split()
    if not parts or parts[0].lower() != analysis.kind:
        raise ValueError(
            f"analysis {analysis.name!r} kind/card mismatch: "
            f"{analysis.kind!r} vs {analysis.card!r}"
        )
    if analysis.kind == "op":
        if len(parts) != 1:
            raise ValueError(f"invalid OP card {analysis.card!r}")
        return 0.0, 0.0
    if analysis.kind == "dc" and len(parts) == 5:
        return _analysis_number(parts[2]), _analysis_number(parts[3])
    if analysis.kind == "tran" and len(parts) in {3, 4}:
        return 0.0, _analysis_number(parts[2])
    if analysis.kind == "ac" and len(parts) == 5:
        from pycircuitsim.simulation import build_ac_frequencies

        axis = build_ac_frequencies({
            "sweep_type": parts[1],
            "num_points": int(parts[2]),
            "fstart": _analysis_number(parts[3]),
            "fstop": _analysis_number(parts[4]),
        })
        return float(axis[0]), float(axis[-1])
    raise ValueError(f"unsupported analysis card {analysis.card!r}")


def analysis_endpoint_tolerance(analysis: AnalysisSpec) -> float:
    """Allow only one declared DC increment for floating-loop roundoff."""
    parts = analysis.card.lstrip(".").split()
    if analysis.kind == "dc" and len(parts) == 5:
        return abs(_analysis_number(parts[4])) * (1.0 + 1e-9)
    if analysis.kind == "tran" and len(parts) in {3, 4}:
        return abs(_analysis_number(parts[1])) * (1.0 + 1e-9)
    return 0.0


def analysis_max_step(analysis: AnalysisSpec) -> Optional[float]:
    """Return the largest legal linear DC/transient output-axis increment."""
    parts = analysis.card.lstrip(".").split()
    if analysis.kind == "dc" and len(parts) == 5:
        return abs(_analysis_number(parts[4]))
    if analysis.kind == "tran" and len(parts) in {3, 4}:
        return abs(_analysis_number(parts[1]))
    return None


def analysis_minimum_points(analysis: AnalysisSpec) -> Optional[int]:
    """Minimum complete trace length for fixed-grid DC, AC, and OP cards."""
    parts = analysis.card.lstrip(".").split()
    if analysis.kind == "op":
        return 1
    if analysis.kind == "dc" and len(parts) == 5:
        start, stop = analysis_axis_limits(analysis)
        step = abs(_analysis_number(parts[4]))
        expected = int(round(abs(stop - start) / step)) + 1
        return max(expected - 1, 1)
    if analysis.kind == "ac" and len(parts) == 5:
        from pycircuitsim.simulation import build_ac_frequencies

        return int(build_ac_frequencies({
            "sweep_type": parts[1],
            "num_points": int(parts[2]),
            "fstart": _analysis_number(parts[3]),
            "fstop": _analysis_number(parts[4]),
        }).size)
    return None


def apply_corner(bt: BenchTech, corner: Corner) -> BenchTech:
    """Return the concrete benchmark technology for one declared corner."""
    values: Dict[str, Any] = {"vdd": round(bt.vdd * corner.vdd_scale, 6)}
    if corner.temperature_c is not None:
        values["temperature_c"] = corner.temperature_c
    if corner.nfin is not None:
        values["nfin"] = corner.nfin
    if corner.nfin_p is not None:
        values["nfin_p"] = corner.nfin_p
    if corner.l_nmos is not None:
        if corner.l_nmos not in bt.profile.l_values:
            raise ValueError(
                f"{bt.name} has no NMOS L={corner.l_nmos * 1e9:g} nm corner"
            )
        values["l_nmos"] = corner.l_nmos
    if corner.l_pmos is not None:
        if corner.l_pmos not in bt.profile.l_values:
            raise ValueError(
                f"{bt.name} has no PMOS L={corner.l_pmos * 1e9:g} nm corner"
            )
        values["l_pmos"] = corner.l_pmos
    if corner.vt_mode:
        available = usable_vts(bt.name)
        alternates = [
            pair.vt_name for pair in bt.profile.vt_pairs
            if pair.vt_name in available and pair.vt_name != bt.vt
        ]
        if not alternates:
            raise ValueError(f"{bt.name} has no alternate trained VT")
        alternate = alternates[0]
        if corner.vt_mode == "alternate":
            values.update(nmos_vt=alternate, pmos_vt=alternate)
        elif corner.vt_mode == "asymmetric":
            values.update(nmos_vt=bt.effective_nmos_vt, pmos_vt=alternate)
        else:
            raise ValueError(f"unknown VT corner mode {corner.vt_mode!r}")
    return bench_variant(bt, **values)


def analysis_applies_to_corner(
    case: CircuitCase,
    analysis: AnalysisSpec,
    base_bt: BenchTech,
    corner: Corner,
) -> bool:
    """Whether a corner changes a physical field observed by this analysis."""
    if corner.name == "nominal":
        return True
    kinds = set(analysis.device_kinds)
    if corner.body_reverse_frac:
        tokens = set(deck_tokens(
            template_deck(case.template, tier=case.tier).read_text()
        ))
        body_tokens = {
            "nmos": {"BODY_N", "BODY_N_NODE"},
            "pmos": {"BODY_P", "BODY_P_NODE"},
        }
        return any(tokens & body_tokens[kind] for kind in kinds)

    try:
        stressed = apply_corner(base_bt, corner)
    except ValueError:
        return False
    if stressed.vdd != base_bt.vdd \
            or stressed.temperature_c != base_bt.temperature_c:
        return True
    if "nmos" in kinds and (
        stressed.l_nmos != base_bt.l_nmos
        or stressed.nfin != base_bt.nfin
        or stressed.effective_nmos_vt != base_bt.effective_nmos_vt
    ):
        return True
    if "pmos" in kinds and (
        stressed.l_pmos != base_bt.l_pmos
        or stressed.effective_nfin_p != base_bt.effective_nfin_p
        or stressed.effective_pmos_vt != base_bt.effective_pmos_vt
    ):
        return True
    return False


def applicable_analyses(
    case: CircuitCase,
    base_bt: BenchTech,
    corner: Corner,
) -> Tuple[AnalysisSpec, ...]:
    """Return only analyses whose observed physics changes at this corner."""
    return tuple(
        analysis for analysis in case.analyses
        if analysis_applies_to_corner(case, analysis, base_bt, corner)
    )


def _number(value: float) -> str:
    return f"{value:.12g}"


def _spice_length(value: float) -> str:
    return f"{value * 1e9:g}n"


def _spice_cap(value: float) -> str:
    return f"{value * 1e15:g}f"


def _expand(value: str, available: Mapping[str, str]) -> str:
    """Resolve placeholders inside an analysis-specific substitution."""
    result = value
    for _ in range(8):
        names = deck_tokens(result)
        if not names:
            return result
        missing = [name for name in names if name not in available]
        if missing:
            raise KeyError(f"nested substitutions missing {missing}")
        result = re.sub(
            r"<([A-Z][A-Z0-9_]*)>",
            lambda match: available[match.group(1)], result,
        )
    raise ValueError(f"recursive deck substitution did not terminate: {value}")


def _common_substitutions(
    bt: BenchTech,
    corner: Corner,
    *,
    reference: bool,
    baked_lib: Path,
    control: bool = False,
    model_level: Optional[int] = None,
    ring_n_stages: int = 5,
    ring_cload: float = 0.5e-15,
) -> Dict[str, str]:
    vdd = bt.vdd
    reverse = corner.body_reverse_frac * vdd
    level = model_level or {
        "DirectNet": 73, "BSIM-AR": 74,
        "DirectNet-Full": 75, "BSIM-AR-Full": 76,
    }.get(active_model_label().split(" (")[0], 73)
    vcm = 0.55 * vdd
    if reference:
        model_setup = f'.include "{baked_lib}"'
        n_prefix = p_prefix = "N"
        n_device = bt.nmos_model
        p_device = bt.pmos_model
    elif control:
        model_setup = (
            f".model {bt.nmos_model} NMOS (LEVEL=72)\n"
            f".model {bt.pmos_model} PMOS (LEVEL=72)"
        )
        n_prefix = p_prefix = "M"
        n_device = (
            f"{bt.nmos_model} L={_spice_length(bt.l_nmos)} "
            f"NFIN={bt.nfin} TFIN={_spice_length(bt.tfin)}"
        )
        p_device = (
            f"{bt.pmos_model} L={_spice_length(bt.l_pmos)} "
            f"NFIN={bt.effective_nfin_p} TFIN={_spice_length(bt.tfin)}"
        )
    else:
        family = {
            75: " FAMILY=directnet-full",
            76: " FAMILY=bsimar-full",
        }.get(level, "")
        model_setup = (
            f".model nmos_nn NMOS (LEVEL={level}{family} TECH={bt.nn_tech} "
            f"VT={bt.effective_nmos_vt})\n"
            f".model pmos_nn PMOS (LEVEL={level}{family} TECH={bt.nn_tech} "
            f"VT={bt.effective_pmos_vt})"
        )
        n_prefix = p_prefix = "M"
        n_device = (
            f"nmos_nn L={_spice_length(bt.l_nmos)} NFIN={bt.nfin}"
        )
        p_device = (
            f"pmos_nn L={_spice_length(bt.l_pmos)} "
            f"NFIN={bt.effective_nfin_p}"
        )
    if ring_n_stages not in {3, 5, 7, 9}:
        raise ValueError(
            f"ring stage count {ring_n_stages} has no authoritative template"
        )
    values = {
        "LEVEL": str(level),
        "TECH": bt.nn_tech,
        "NVT": bt.effective_nmos_vt,
        "PVT": bt.effective_pmos_vt,
        "LN": _spice_length(bt.l_nmos),
        "LP": _spice_length(bt.l_pmos),
        "NFN": str(bt.nfin),
        "NFP": str(bt.effective_nfin_p),
        "TEMP": _number(bt.temperature_c),
        "VDD": _number(vdd),
        "HALF_VDD": _number(0.5 * vdd),
        "BODY_N": _number(-reverse),
        "BODY_P": _number(vdd + reverse),
        "BODY_NETWORK": (
            f"Vbn bn 0 {_number(-reverse)}\n"
            f"Vbp bp 0 {_number(vdd + reverse)}"
        ),
        "BODY_N_NODE": "bn",
        "BODY_P_NODE": "bp",
        "GATE_N": _number(0.65 * vdd),
        "GATE_P": _number(0.35 * vdd),
        "FOLLOW_N_IC": _number(0.25 * vdd),
        "FOLLOW_P_IC": _number(0.75 * vdd),
        "IBIAS": _number(5e-6 * bt.nfin / 2.0),
        "TAIL_CURRENT": _number(10e-6 * bt.nfin / 2.0),
        "TAIL_GATE": _number(0.45 * vdd),
        "BIAS_N": _number(0.45 * vdd),
        "CAS_N": _number(0.65 * vdd),
        "BIAS_P": _number(0.55 * vdd),
        "CAS_P": _number(0.35 * vdd),
        "VCM": _number(vcm),
        "DIFF_LO": _number(vcm - 0.10 * vdd),
        "DIFF_HI": _number(vcm + 0.10 * vdd),
        "P_VCM": _number(0.45 * vdd),
        "P_DIFF_LO": _number(0.45 * vdd - 0.06 * vdd),
        "P_DIFF_HI": _number(0.45 * vdd + 0.06 * vdd),
        "N_STEER_INP": _number(vcm + 0.02 * vdd),
        "P_STEER_INP": _number(0.45 * vdd - 0.02 * vdd),
        "VBN": _number(0.45 * vdd),
        "VBP": _number(0.55 * vdd),
        "OPAMP_LO": _number(vcm - 0.15),
        "OPAMP_HI": _number(vcm + 0.15),
        # Source specs are tokens so one topology can serve the DC transfer
        # and the small-signal rejection experiments.  The defaults render
        # byte-identically to the pre-token deck, which is what keeps the
        # frozen simple-v1 opamp cell unchanged.
        "VDD_SPEC": _number(vdd),
        "VINP_SPEC": _number(vcm),
        "VINN_SPEC": _number(vcm),
        "MIRROR_OUT_N": "0",
        "MIRROR_OUT_P": _number(vdd),
        "MID_RAIL": _number(0.5 * vdd),
        # Mirror reference-current window: from deep subthreshold to several
        # times the nominal bias, so the mirror ratio is scored across the
        # inversion transition rather than at one operating current.
        "IREF_LO": _number(0.05e-6 * bt.nfin / 2.0),
        "IREF_HI": _number(20e-6 * bt.nfin / 2.0),
        "IREF_STEP": _number(0.5e-6 * bt.nfin / 2.0),
        # --- Tier A: self-biased cells ----------------------------------
        # Rs sets the constant-gm operating current through the K=2 width
        # ratio; 20k lands the cell near the subthreshold/strong-inversion
        # boundary, which is where a fitted model is least constrained.
        "BETA_RS": "20k",
        "BETA_RSTART": "10e6",
        "REF_RBIAS": "500k",
        "DIODE_RLOAD": "200k",
        "CASCODE_OUT": _number(0.6 * vdd),
        # --- Tier B: closed-loop systems --------------------------------
        "BUFFER_IN": _number(0.5 * vdd),
        "BUF_LO": _number(0.30 * vdd),
        "BUF_HI": _number(0.70 * vdd),
        "BUF_TSTEP": "5p",
        "BUF_TSTOP": "6n",
        "BUFFER_STEP": (
            f"{'PULSE(' if reference else 'PULSE'} "
            f"{_number(0.45 * vdd)} {_number(0.55 * vdd)} 1n 50p 50p 2n 6n"
            f"{')' if reference else ''}"
        ),
        "IBIAS_CELL": _number(2e-6 * bt.nfin / 2.0),
        "LDO_VREF": _number(0.35 * vdd),
        "LDO_VDD_SPEC": _number(vdd),
        "LDO_RFB1": "200k",
        "LDO_RFB2": "200k",
        "LDO_COUT": "500f",
        "LDO_VDD_LO": _number(0.80 * vdd),
        "LDO_TSTEP": "10p",
        "LDO_TSTOP": "20n",
        "LDO_LOAD_DC": _number(1e-6 * bt.nfin / 2.0),
        "LDO_LOAD_STEP": (
            f"{'PULSE(' if reference else 'PULSE'} "
            f"{_number(1e-6 * bt.nfin / 2.0)} {_number(8e-6 * bt.nfin / 2.0)} "
            f"5n 100p 100p 8n 20n{')' if reference else ''}"
        ),
        "CC": "20f",
        "CL": "50f",
        "WL": _number(vdd),
        "VIN": _number(0.6 * vdd),
        "TD": "0.5n",
        "SLEW": "0.1n",
        "PW": "1.9n",
        "PER": "4n",
        "INPUT_DELAY": "0.5n",
        "INPUT_RISE": "20p",
        "INPUT_FALL": "20p",
        "INPUT_WIDTH": "1n",
        "INPUT_PERIOD": "2n",
        "SRAM_WL_WIDTH": "1.5n",
        "SRAM_WL_PERIOD": "3n",
        "CLOCK_DELAY": "1n",
        "CLOCK_RISE": "20p",
        "CLOCK_FALL": "20p",
        "CLOCK_WIDTH": "2n",
        "CLOCK_PERIOD": "4n",
        "CSAMPLE": "100f",
        "FOLLOWER_LOAD": "20k",
        "COMMON_GATE_LOAD": "12k",
        "DIFFPAIR_LOAD": "18k",
        "CASCODE_LOAD": "20k",
        "CHAIN_LOAD": "2f",
        "STORAGE_LOADS": "Cq q 0 2f\nCqb qb 0 2f",
        "LOGIC_LOAD": "5f",
        "HOLD_LOAD": "100f",
        "CS_LOAD": "50k",
        "BULK_LOAD": "100k",
        "INPUT_SPEC": "0",
        "LOAD_NETWORK": "",
        "BULK_NETWORK": "",
        "DEVICE_PREFIX": f"{n_prefix}n",
        "SOURCE_NODE": "0",
        "BULK_NODE": "0",
        "DEVICE": n_device,
        "INITIAL_CONDITION": "",
        "OUTPUT_LOAD": "",
        "AC_INP": "1",
        "AC_INN": "0",
        "DIFFPAIR_CAP": "5f",
        "ACTIVE_TAIL_CURRENT": _number(2e-6 * bt.nfin / 2.0),
        "VA_SPEC": "0",
        "VB_SPEC": "0",
        "WL_SPEC": "0",
        "BL_SPEC": _number(vdd),
        "BLB_SPEC": _number(vdd),
        "Q_IC": _number(vdd),
        "QB_IC": "0",
        "BAKED_LIB": str(baked_lib),
        "NMOS": bt.nmos_model,
        "PMOS": bt.pmos_model,
        "MODEL_SETUP": model_setup,
        "N_PREFIX": n_prefix,
        "P_PREFIX": p_prefix,
        "N_DEVICE": n_device,
        "P_DEVICE": p_device,
        "PULSE_OPEN": "PULSE(" if reference else "PULSE",
        "PULSE_CLOSE": ")" if reference else "",
        "RING_CLOAD": _spice_cap(ring_cload),
        "BIAS_CURRENT_SPEC": (
            f"{'PULSE(' if reference else 'PULSE'} "
            f"{_number(0.2e-6 * bt.nfin / 2.0)} "
            f"{_number(5e-6 * bt.nfin / 2.0)} 0.5n 50p 50p 3n 6n"
            f"{')' if reference else ''}"
        ),
    }
    return values


#: Source specs whose default is another token's value.  These must be
#: resolved AFTER case-level overrides are applied, not while the base
#: substitution table is built: a caller that overrides ``VCM`` (the
#: parametric opamp sweep does) would otherwise have its override silently
#: dropped, because the spec had already been frozen to the unoverridden bias.
_SPEC_DEFAULTS: Dict[str, str] = {
    "VDD_SPEC": "<VDD>",
    "VINP_SPEC": "<VCM>",
    "VINN_SPEC": "<VCM>",
    "MIRROR_OUT_P": "<VDD>",
}


def _resolved_role(
    case: CircuitCase,
    role: DeviceRoleSpec,
    bt: BenchTech,
) -> Dict[str, Any]:
    is_pmos = role.polarity == "pmos"
    length = role.length_m or (bt.l_pmos if is_pmos else bt.l_nmos)
    base_nfin = bt.effective_nfin_p if is_pmos else bt.nfin
    nfin = base_nfin + role.nfin_delta
    if nfin < 1:
        raise ValueError(f"{case.case_id}/{role.name}: NFIN must be positive")
    vt = role.vt or (
        bt.effective_pmos_vt if is_pmos else bt.effective_nmos_vt
    )
    pair = bt.profile.get_vt_pair(vt)
    source_model = pair.pmos_model if is_pmos else pair.nmos_model
    return {
        "polarity": role.polarity,
        "length": length,
        "nfin": nfin,
        "vt": vt,
        "source_model": source_model,
        "candidate_model": f"{role.name}_nn",
        "reference_model": f"v768_{case.case_id}_{role.name}",
    }


def resolved_device_geometries(
    case: CircuitCase,
    analyses: Sequence[AnalysisSpec],
    bt: BenchTech,
) -> Tuple[Tuple[str, str, float, int], ...]:
    """Return the distinct polarity/VT/L/NFIN points a case instantiates."""
    if case.device_roles:
        geometries = [
            (
                role.polarity,
                str(resolved["vt"]),
                float(resolved["length"]),
                int(resolved["nfin"]),
            )
            for role in case.device_roles
            for resolved in (_resolved_role(case, role, bt),)
        ]
    else:
        kinds = sorted({kind for analysis in analyses
                        for kind in analysis.device_kinds})
        geometries = [
            (
                kind,
                bt.effective_pmos_vt if kind == "pmos"
                else bt.effective_nmos_vt,
                bt.l_pmos if kind == "pmos" else bt.l_nmos,
                bt.effective_nfin_p if kind == "pmos" else bt.nfin,
            )
            for kind in kinds
        ]
    return tuple(dict.fromkeys(geometries))


def _role_substitutions(
    case: CircuitCase,
    bt: BenchTech,
    *,
    reference: bool,
    baked_lib: Path,
    level: int,
    control: bool = False,
) -> Dict[str, str]:
    if not case.device_roles:
        return {}
    values: Dict[str, str] = {}
    declarations: List[str] = []
    for role in case.device_roles:
        resolved = _resolved_role(case, role, bt)
        if reference:
            values[role.token] = str(resolved["reference_model"])
        else:
            kind = "PMOS" if role.polarity == "pmos" else "NMOS"
            model = (
                str(resolved["reference_model"])
                if control else str(resolved["candidate_model"])
            )
            family = {
                75: " FAMILY=directnet-full",
                76: " FAMILY=bsimar-full",
            }.get(level, "")
            declarations.append(
                f".model {model} {kind} (LEVEL={72 if control else level}"
                + ("" if control else
                   f"{family} TECH={bt.nn_tech} VT={resolved['vt']}")
                + ")"
            )
            values[role.token] = (
                f"{model} L={float(resolved['length']) * 1e9:g}n "
                f"NFIN={int(resolved['nfin'])}"
                + (f" TFIN={bt.tfin * 1e9:g}n" if control else "")
            )
    values["MODEL_SETUP"] = (
        f'.include "{baked_lib}"' if reference else "\n".join(declarations)
    )
    return values


def _role_source_modelcard(
    bt: BenchTech,
    role: DeviceRoleSpec,
    resolved: Mapping[str, Any],
) -> Path:
    if bt.profile.single_file:
        return bt.profile.get_nmos_modelcard(
            bt.profile.get_vt_pair(str(resolved["vt"])),
            float(resolved["length"]),
        )
    from pycmg.tech import TECH_REGISTRY, resolve_modelcard

    config = TECH_REGISTRY[bt.name]
    device = config.get_device(str(resolved["source_model"]).replace(
        "nch_", "nmos_",
    ).replace("pch_", "pmos_").replace("_mac", ""))
    return Path(resolve_modelcard(
        device,
        config,
        L=float(resolved["length"]),
        NFIN=float(resolved["nfin"]),
    ))


def get_case_baked_modelcard(
    case: CircuitCase,
    bt: BenchTech,
    work_dir: Path,
) -> Path:
    """Build the ordinary pair or a role-specific OSDI model library."""
    if not case.device_roles:
        return get_baked_modelcard(
            bt, bt.nfin, work_dir, nfin_p=bt.effective_nfin_p,
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    fragments: List[str] = []
    for role in case.device_roles:
        resolved = _resolved_role(case, role, bt)
        source = _role_source_modelcard(bt, role, resolved)
        if not source.is_file():
            raise FileNotFoundError(f"role modelcard not found: {source}")
        alias = str(resolved["reference_model"])
        original = str(resolved["source_model"])
        source_text = source.read_text()
        renamed = re.sub(
            rf"(?im)^(\s*\.model\s+){re.escape(original)}(\s+)",
            rf"\g<1>{alias}\g<2>",
            source_text,
            count=1,
        )
        if renamed == source_text:
            raise ValueError(
                f"{case.case_id}/{role.name}: model {original!r} not found"
            )
        fragment = work_dir / f"role_{role.name}.lib"
        fragment.write_text(renamed)
        bake_inst_params(
            fragment,
            fragment,
            alias,
            {
                "L": float(resolved["length"]),
                "NFIN": float(resolved["nfin"]),
                "TFIN": bt.tfin,
                "DEVTYPE": 0 if role.polarity == "pmos" else 1,
            },
        )
        fragments.append(fragment.read_text())
    library = work_dir / f"baked_roles_{case.case_id}_{bt.name}.lib"
    library.write_text("\n".join(fragments))
    return library


def _render_one(
    case: CircuitCase,
    analysis: AnalysisSpec,
    bt: BenchTech,
    corner: Corner,
    *,
    reference: bool,
    baked_lib: Path,
    control: bool = False,
    model_level: Optional[int] = None,
    substitutions: Optional[Mapping[str, str]] = None,
    ring_n_stages: int = 5,
    ring_cload: float = 0.5e-15,
) -> str:
    relative = case.template
    if relative == "ring_oscillator.spice.tmpl":
        relative = {
            3: "ring_oscillator_3stage.spice.tmpl",
            5: "ring_oscillator.spice.tmpl",
            7: "ring_oscillator_7stage.spice.tmpl",
            9: "ring_oscillator_9stage.spice.tmpl",
        }.get(ring_n_stages, "")
        if not relative:
            raise ValueError(
                f"ring stage count {ring_n_stages} has no authoritative template"
            )
    path = template_deck(relative, tier=case.tier)
    template = path.read_text()
    available = _common_substitutions(
        bt, corner, reference=reference, baked_lib=baked_lib,
        control=control,
        model_level=model_level,
        ring_n_stages=ring_n_stages, ring_cload=ring_cload,
    )
    available.update(_role_substitutions(
        case,
        bt,
        reference=reference,
        baked_lib=baked_lib,
        level=model_level or active_model_level(),
        control=control,
    ))
    overrides = dict(substitutions or {})
    available.update(overrides)
    for name, default in _SPEC_DEFAULTS.items():
        if name not in overrides:
            available[name] = _expand(default, available)
    analysis_overrides = analysis.substitutions()
    available.update(analysis_overrides)
    for name in analysis_overrides:
        available[name] = _expand(available[name], available)
    # Reference analysis is executed explicitly in the NGSPICE control block;
    # keeping the template slot empty prevents accidental `run` semantics.
    available["ANALYSIS"] = (
        "" if reference else "." + _expand(analysis.card, available)
    )
    required = deck_tokens(template)
    missing = [name for name in required if name not in available]
    if missing:
        raise KeyError(f"{relative}: no values for {missing}")
    substitutions = {name: available[name] for name in required}
    return render_deck_text(
        template, substitutions, source_name=relative, body_only=False,
    )


def _resolved_analysis(
    case: CircuitCase,
    analysis: AnalysisSpec,
    bt: BenchTech,
    corner: Corner,
) -> AnalysisSpec:
    """Resolve technology placeholders in an experiment's analysis card."""
    candidate_values = _common_substitutions(
        bt, corner, reference=False, baked_lib=Path("<unused>"),
    )
    reference_values = _common_substitutions(
        bt, corner, reference=True, baked_lib=Path("<unused>"),
    )
    candidate_values.update(_role_substitutions(
        case,
        bt,
        reference=False,
        baked_lib=Path("<unused>"),
        level=active_model_level(),
    ))
    reference_values.update(_role_substitutions(
        case,
        bt,
        reference=True,
        baked_lib=Path("<unused>"),
        level=active_model_level(),
    ))
    for values in (candidate_values, reference_values):
        for name, default in _SPEC_DEFAULTS.items():
            values[name] = _expand(default, values)
    analysis_overrides = analysis.substitutions()
    candidate_values.update(analysis_overrides)
    reference_values.update(analysis_overrides)
    for name in analysis_overrides:
        candidate_values[name] = _expand(
            candidate_values[name], candidate_values,
        )
        reference_values[name] = _expand(
            reference_values[name], reference_values,
        )
    candidate_card = _expand(analysis.card, candidate_values)
    reference_card = _expand(analysis.card, reference_values)
    if candidate_card != reference_card:
        raise ValueError(
            f"analysis card differs by adapter: {candidate_card!r} != "
            f"{reference_card!r}"
        )
    return replace(analysis, card=candidate_card)


def render_case_decks(
    case: CircuitCase,
    analysis: AnalysisSpec,
    base_bt: BenchTech,
    corner: Corner,
    *,
    baked_lib: Path,
    substitutions: Optional[Mapping[str, str]] = None,
    ring_n_stages: int = 5,
    ring_cload: float = 0.5e-15,
    model_level: Optional[int] = None,
) -> Tuple[str, str]:
    """Render the candidate and LEVEL=72 decks for one identical experiment."""
    bt = apply_corner(base_bt, corner)
    candidate = _render_one(
        case, analysis, bt, corner, reference=False, baked_lib=baked_lib,
        substitutions=substitutions or {}, ring_n_stages=ring_n_stages,
        ring_cload=ring_cload, model_level=model_level,
    )
    reference = _render_one(
        case, analysis, bt, corner, reference=True, baked_lib=baked_lib,
        substitutions=substitutions or {}, ring_n_stages=ring_n_stages,
        ring_cload=ring_cload, model_level=model_level,
    )
    return candidate, reference


def render_case_control_deck(
    case: CircuitCase,
    analysis: AnalysisSpec,
    base_bt: BenchTech,
    corner: Corner,
    *,
    baked_lib: Path,
    substitutions: Optional[Mapping[str, str]] = None,
    ring_n_stages: int = 5,
    ring_cload: float = 0.5e-15,
) -> str:
    """Render the same experiment for PyCircuitSim's LEVEL=72 adapter."""
    bt = apply_corner(base_bt, corner)
    return _render_one(
        case,
        analysis,
        bt,
        corner,
        reference=False,
        control=True,
        baked_lib=baked_lib,
        substitutions=substitutions or {},
        ring_n_stages=ring_n_stages,
        ring_cload=ring_cload,
    )


def _source_kind(parts: Sequence[str]) -> str:
    tail = " ".join(parts[3:]).lower()
    if "pulse" in tail:
        return "pulse"
    if "ac" in tail:
        return "ac"
    return "dc"


def _normalized_spec(parts: Sequence[str]) -> str:
    """Normalize equivalent parenthesized and bare SPICE value syntax."""
    text = " ".join(parts).lower().replace("(", " ").replace(")", " ")
    return " ".join(text.split())


def topology_signature(text: str) -> Counter[Tuple[str, ...]]:
    """Canonical physical multiset for a flat rendered deck.

    Engine-specific model names and M/N device prefixes are ignored. Element
    values, source waveforms, temperature, options, IC values, MOS polarity,
    and terminal order are retained. An unresolved subcircuit instance is
    rejected because silently ignoring it would turn the parity check into a
    false assurance.
    """
    signature: Counter[Tuple[str, ...]] = Counter()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("*"):
            continue
        parts = line.split()
        head = parts[0]
        low = head.lower()
        if low in {".include", ".model", ".end", ".dc", ".tran", ".ac", ".op"}:
            continue
        if low.startswith(".temp"):
            signature[("temp", _normalized_spec(parts[1:]))] += 1
            continue
        if low in {".option", ".options"}:
            signature[("options", _normalized_spec(parts[1:]))] += 1
            continue
        if low == ".ic":
            assignments = tuple(sorted(
                (node.lower(), value.lower())
                for node, value in re.findall(
                    r"v\(([^)]+)\)\s*=\s*([^\s]+)",
                    line,
                    flags=re.IGNORECASE,
                )
            ))
            signature[("ic", *[f"{node}={value}"
                                for node, value in assignments])] += 1
            continue
        prefix = head[0].upper()
        if prefix == "X":
            raise ValueError(f"topology parity requires flat decks: {line}")
        if prefix in ("M", "N"):
            if len(parts) < 6:
                raise ValueError(f"malformed MOS line: {line}")
            name = head.lower()
            polarity = "p" if len(name) > 1 and name[1] == "p" else "n"
            signature[("mos", polarity, *[node.lower()
                                           for node in parts[1:5]])] += 1
        elif prefix in ("R", "C", "L"):
            signature[(prefix.lower(), parts[1].lower(),
                       parts[2].lower(), _normalized_spec(parts[3:]))] += 1
        elif prefix in ("V", "I"):
            signature[(prefix.lower(), head.lower(), parts[1].lower(),
                       parts[2].lower(), _source_kind(parts),
                       _normalized_spec(parts[3:]))] += 1
        else:
            raise ValueError(f"unsupported topology card in parity check: {line}")
    return signature


def connectivity_signature(text: str) -> Counter[Tuple[str, ...]]:
    """Return topology alone, for detecting duplicate template ownership."""
    physical = topology_signature(text)
    signature: Counter[Tuple[str, ...]] = Counter()
    for item, count in physical.items():
        kind = item[0]
        if kind in {"temp", "options"}:
            continue
        if kind == "ic":
            projected = ("ic", *(value.split("=", 1)[0] for value in item[1:]))
        elif kind in {"r", "c", "l"}:
            projected = item[:3]
        elif kind in {"v", "i"}:
            projected = (item[0], item[2], item[3], item[4])
        else:
            projected = item
        signature[projected] += count
    return signature


def topology_mismatch(candidate: str, reference: str) -> str:
    """Return a readable candidate/reference connectivity difference."""
    candidate_sig = topology_signature(candidate)
    reference_sig = topology_signature(reference)
    if candidate_sig == reference_sig:
        return ""
    candidate_only = list((candidate_sig - reference_sig).elements())
    reference_only = list((reference_sig - candidate_sig).elements())
    return (f"physical deck mismatch: candidate-only={candidate_only}; "
            f"reference-only={reference_only}")


def _model_declarations(deck: str) -> Dict[str, Tuple[str, Dict[str, str]]]:
    declarations: Dict[str, Tuple[str, Dict[str, str]]] = {}
    for raw in deck.splitlines():
        line = raw.strip()
        if not line.lower().startswith(".model "):
            continue
        parts = line.replace("(", " ").replace(")", " ").split()
        if len(parts) < 3:
            raise ValueError(f"malformed model declaration: {line}")
        params: Dict[str, str] = {}
        for token in parts[3:]:
            if "=" in token:
                name, value = token.split("=", 1)
                params[name.upper()] = value
        declarations[parts[1].lower()] = (parts[2].lower(), params)
    return declarations


def physical_deck_mismatch(
    candidate: str,
    reference: str,
    analysis: AnalysisSpec,
    bt: BenchTech,
    *,
    baked_lib: Path,
    case: Optional[CircuitCase] = None,
    model_level: Optional[int] = None,
    device_kinds: Optional[Sequence[str]] = None,
    control: bool = False,
) -> str:
    """Return any physical or compact-model binding drift between adapters."""
    mismatch = topology_mismatch(candidate, reference)
    if mismatch:
        return mismatch

    expected_card = "." + analysis.card.lstrip(".").lower()
    candidate_cards = [
        line.strip().lower()
        for line in candidate.splitlines()
        if line.strip().lower().startswith((".op", ".dc", ".tran", ".ac"))
    ]
    if candidate_cards != [expected_card]:
        return (
            f"candidate analysis mismatch: expected {[expected_card]}, "
            f"got {candidate_cards}"
        )
    reference_cards = [
        line.strip().lower()
        for line in reference.splitlines()
        if line.strip().lower().startswith((".op", ".dc", ".tran", ".ac"))
    ]
    if reference_cards:
        return f"reference deck unexpectedly embeds analysis cards: {reference_cards}"

    includes = []
    for raw in reference.splitlines():
        parts = raw.strip().split(maxsplit=1)
        if parts and parts[0].lower() == ".include" and len(parts) == 2:
            includes.append(parts[1].strip().strip('"'))
    if includes != [str(baked_lib)]:
        return (
            f"reference model include mismatch: expected {[str(baked_lib)]}, "
            f"got {includes}"
        )

    declarations = _model_declarations(candidate)
    expected_level = "72" if control else str(model_level or active_model_level())
    if case is not None and case.device_roles:
        expected_models = {
            str(_resolved_role(case, role, bt)[
                "reference_model" if control else "candidate_model"
            ]): (
                role.polarity, str(_resolved_role(case, role, bt)["vt"]),
            )
            for role in case.device_roles
        }
    else:
        all_models = {
            (bt.nmos_model if control else "nmos_nn"): (
                "nmos", bt.effective_nmos_vt,
            ),
            (bt.pmos_model if control else "pmos_nn"): (
                "pmos", bt.effective_pmos_vt,
            ),
        }
        selected_kinds = set(device_kinds or ("nmos", "pmos"))
        if not selected_kinds <= {"nmos", "pmos"}:
            return f"unknown device kinds in parity contract: {sorted(selected_kinds)}"
        expected_models = {
            name: values for name, values in all_models.items()
            if values[0] in selected_kinds
        }
    for name, (kind, vt) in expected_models.items():
        declaration = declarations.get(name)
        if declaration is None:
            return f"candidate model declaration missing: {name}"
        actual_kind, params = declaration
        expected_params = {
            "LEVEL": expected_level,
        }
        if not control:
            expected_params.update({"TECH": bt.nn_tech, "VT": vt})
        if not control and expected_level in {"75", "76"}:
            expected_params["FAMILY"] = {
                "75": "directnet-full",
                "76": "bsimar-full",
            }[expected_level]
        if actual_kind != kind or any(
            params.get(key, "").lower() != value.lower()
            for key, value in expected_params.items()
        ):
            return (
                f"candidate model binding mismatch for {name}: "
                f"kind={actual_kind}, params={params}, expected={expected_params}"
            )

    role_geometry: Dict[str, Tuple[float, int, str, str]] = {}
    if case is not None:
        for role in case.device_roles:
            resolved = _resolved_role(case, role, bt)
            for instance in role.instances:
                role_geometry[instance.lower()] = (
                    float(resolved["length"]),
                    int(resolved["nfin"]),
                    str(resolved["reference_model"]).lower(),
                    str(resolved["candidate_model"]).lower(),
                )
    default_geometry = {
        "n": (bt.l_nmos, bt.nfin, bt.nmos_model.lower(), "nmos_nn"),
        "p": (
            bt.l_pmos, bt.effective_nfin_p, bt.pmos_model.lower(), "pmos_nn",
        ),
    }
    for deck, is_reference in ((candidate, False), (reference, True)):
        for raw in deck.splitlines():
            parts = raw.strip().split()
            if not parts or parts[0][0].upper() not in {"M", "N"}:
                continue
            polarity = "p" if len(parts[0]) > 1 and parts[0][1].lower() == "p" else "n"
            instance = parts[0][1:].lower()
            length, nfin, reference_model, candidate_model = role_geometry.get(
                instance, default_geometry[polarity],
            )
            if is_reference:
                if len(parts) < 6 or parts[5].lower() != reference_model:
                    return f"reference {polarity} model binding mismatch: {raw.strip()}"
                continue
            expected_model = reference_model if control else candidate_model
            if len(parts) < 6 or parts[5].lower() != expected_model:
                return f"candidate {polarity} model binding mismatch: {raw.strip()}"
            params = {
                key.upper(): value
                for token in parts[6:] if "=" in token
                for key, value in (token.split("=", 1),)
            }
            try:
                actual_l = _analysis_number(params["L"])
                actual_nfin = int(float(params["NFIN"]))
            except (KeyError, ValueError):
                return f"candidate geometry is incomplete: {raw.strip()}"
            if not math.isclose(actual_l, length, rel_tol=1e-12, abs_tol=1e-18) \
                    or actual_nfin != nfin:
                return (
                    f"candidate {polarity} geometry mismatch: "
                    f"L={actual_l:g}, NFIN={actual_nfin}; "
                    f"expected L={length:g}, NFIN={nfin}"
                )
            if control:
                try:
                    actual_tfin = _analysis_number(params["TFIN"])
                except (KeyError, ValueError):
                    return f"control geometry is missing TFIN: {raw.strip()}"
                if not math.isclose(
                    actual_tfin, bt.tfin, rel_tol=1e-12, abs_tol=1e-18,
                ):
                    return (
                        f"control TFIN mismatch: {actual_tfin:g}; "
                        f"expected {bt.tfin:g}"
                    )
    return ""


def _body_only(deck: str) -> str:
    lines = [line for line in deck.splitlines()
             if line.strip().lower() != ".end"
             and not line.lstrip().startswith("*")]
    return "\n".join(lines)


def _support_voltage_signals(candidate_deck: str) -> Tuple[str, ...]:
    nodes: set[str] = set()
    for raw in candidate_deck.splitlines():
        parts = raw.split()
        if parts and parts[0][0].upper() == "M" and len(parts) >= 5:
            nodes.update(node for node in parts[1:5]
                         if node.lower() not in ("0", "gnd"))
    return tuple(f"v({node})" for node in sorted(nodes))


def _parse_real_wrdata(
    data: np.ndarray,
    signals: Sequence[str],
    *,
    axis_name: str,
) -> Trace:
    expected = 2 * len(signals)
    if data.ndim != 2 or data.shape[1] != expected:
        raise RuntimeError(
            f"NGSPICE wrdata width {data.shape if data.ndim == 2 else data.ndim} "
            f"does not match {len(signals)} real vectors ({expected} columns)")
    axis = data[:, 0]
    values = {signal: data[:, 2 * index + 1]
              for index, signal in enumerate(signals)}
    trace = Trace(axis_name, axis, values, reference=True)
    trace.validate()
    return trace


def _run_ngspice_ac_trace(
    deck: str,
    signals: Sequence[str],
    analysis_card: str,
    work_dir: Path,
    tag: str,
) -> Trace:
    work_dir.mkdir(parents=True, exist_ok=True)
    deck_path = work_dir / f"ngspice_{tag}.cir"
    csv_path = work_dir / f"ngspice_{tag}.csv"
    log_path = work_dir / f"ngspice_{tag}.log"
    runner_path = work_dir / f"ngspice_{tag}_runner.cir"
    deck_path.write_text(deck)
    runner_path.write_text(
        f"* NGSPICE simple-circuit AC runner ({tag})\n"
        ".control\n"
        f"osdi {OSDI_PATH}\n"
        f"source {deck_path}\n"
        "set filetype=ascii\n"
        "set wr_vecnames\n"
        f"{analysis_card}\n"
        f"wrdata {csv_path} {' '.join(signals)}\n"
        ".endc\n.end\n"
    )
    lines = run_ngspice_subprocess(runner_path, log_path, csv_path)
    rows = [[float(value) for value in line.split()]
            for line in lines[1:] if line.strip()]
    data = np.asarray(rows, dtype=float)
    if data.ndim != 2:
        raise RuntimeError("NGSPICE AC produced a malformed matrix")
    values: Dict[str, np.ndarray] = {}
    if data.shape[1] == 3 * len(signals):
        axis = data[:, 0]
        for index, signal in enumerate(signals):
            offset = 3 * index
            values[signal] = data[:, offset + 1] + 1j * data[:, offset + 2]
    elif data.shape[1] == 1 + 2 * len(signals):
        axis = data[:, 0]
        for index, signal in enumerate(signals):
            offset = 1 + 2 * index
            values[signal] = data[:, offset] + 1j * data[:, offset + 1]
    else:
        raise RuntimeError(
            f"NGSPICE AC wrdata width {data.shape[1]} is incompatible with "
            f"{len(signals)} vectors")
    trace = Trace("frequency", axis, values, reference=True)
    trace.validate()
    return trace


def run_reference_trace(
    deck: str,
    analysis: AnalysisSpec,
    work_dir: Path,
    tag: str,
    *,
    support_signals: Sequence[str] = (),
) -> Trace:
    """Run the accepted LEVEL=72 trajectory for one rendered experiment."""
    signals = list(dict.fromkeys((*analysis.signals, *support_signals)))
    card = analysis.card
    if analysis.kind == "ac":
        trace = _run_ngspice_ac_trace(deck, signals, card, work_dir, tag)
    else:
        data = run_ngspice_wrdata(
            _body_only(deck), " ".join(signals), work_dir, tag, card,
        )
        trace = _parse_real_wrdata(
            data,
            signals,
            axis_name="time" if analysis.kind == "tran" else "sweep",
        )
        if analysis.kind == "op":
            trace.axis = np.asarray([0.0])
    start, stop = analysis_axis_limits(analysis)
    trace.validate(
        expected_start=start,
        expected_stop=stop,
        endpoint_tolerance=analysis_endpoint_tolerance(analysis),
        max_step=analysis_max_step(analysis),
        minimum_points=analysis_minimum_points(analysis),
    )
    return trace


def _lookup_signal(results: Mapping[str, Any], signal: str) -> np.ndarray:
    match = re.fullmatch(r"([vi])\(([^)]+)\)", signal, re.IGNORECASE)
    if not match:
        raise ValueError(f"unsupported signal syntax: {signal}")
    kind, name = match.groups()
    wanted = name if kind.lower() == "v" else f"i({name})"
    for key, values in results.items():
        if key.lower() == wanted.lower():
            return np.asarray(values)
    raise KeyError(f"candidate results carry no {signal}; keys={list(results)}")


def _dc_axis(params: Mapping[str, Any], n_points: int) -> np.ndarray:
    start = float(params["start"])
    step = float(params["step"])
    return start + step * np.arange(n_points, dtype=float)


def _run_candidate_ac_trace(
    path: Path,
    signals: Sequence[str],
) -> Trace:
    from pycircuitsim.simulation import (
        _circuit_has_nn,
        _solve_dc_with_retry,
        build_ac_frequencies,
    )
    from pycircuitsim.solver import ACSolver, DCSolver

    parser = parse_netlist(path)
    circuit = parser.circuit
    has_nn = _circuit_has_nn(circuit)

    def _solve(use_gmin: bool) -> Tuple[DCSolver, Dict[str, float]]:
        solver = DCSolver(
            circuit, initial_guess=circuit.initial_conditions or None,
            use_source_stepping=True, use_gmin_stepping=use_gmin,
        )
        return solver, solver.solve()

    solver, dc_solution = _solve_dc_with_retry(circuit, has_nn, _solve)
    converged = bool(getattr(solver, "_last_solve_converged", True))
    if not converged:
        raise RuntimeError("candidate AC operating point did not converge")
    frequencies = build_ac_frequencies(parser.analysis_params)
    raw = ACSolver(circuit, dc_solution=dc_solution).solve(frequencies)
    values = {signal: _lookup_signal(raw, signal) for signal in signals}
    trace = Trace("frequency", frequencies, values, converged=converged)
    trace.validate()
    return trace


def _run_candidate_op_trace(
    path: Path,
    signals: Sequence[str],
) -> Trace:
    """Solve one NN operating point and expose nodes plus source currents."""
    from pycircuitsim.simulation import _circuit_has_nn, _solve_dc_with_retry
    from pycircuitsim.solver import DCSolver

    parser = parse_netlist(path)
    circuit = parser.circuit

    def _solve(use_gmin: bool) -> Tuple[DCSolver, Dict[str, float]]:
        solver = DCSolver(
            circuit,
            initial_guess=circuit.initial_conditions or None,
            use_source_stepping=True,
            use_gmin_stepping=use_gmin,
        )
        return solver, solver.solve()

    solver, solution = _solve_dc_with_retry(
        circuit, _circuit_has_nn(circuit), _solve,
    )
    converged = bool(getattr(solver, "_last_solve_converged", True))
    if not converged:
        raise RuntimeError("candidate operating point did not converge")
    raw: Dict[str, Any] = dict(solution)
    for component in circuit.components:
        try:
            raw[f"i({component.name})"] = np.asarray([
                component.calculate_current(solution)
            ])
        except (AttributeError, NotImplementedError):
            continue
    values = {
        signal: np.atleast_1d(_lookup_signal(raw, signal)) for signal in signals
    }
    trace = Trace("op", np.asarray([0.0]), values, converged=True)
    trace.validate(expected_start=0.0, expected_stop=0.0)
    return trace


def run_candidate_trace(
    deck: str,
    analysis: AnalysisSpec,
    work_dir: Path,
    tag: str,
    *,
    require_convergence: bool = True,
) -> Tuple[Trace, Path]:
    """Run the NN adapter and return a structured trace plus rendered path."""
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / f"candidate_{tag}.sp"
    path.write_text(deck)
    logging.disable(logging.CRITICAL)
    try:
        if analysis.kind == "dc":
            parser = parse_netlist(path)
            results = run_directnet_dc_sweep(
                path, work_dir, tag, require_convergence=require_convergence,
            )
            first = _lookup_signal(results, analysis.signals[0])
            axis = _dc_axis(parser.analysis_params, first.size)
            values = {signal: _lookup_signal(results, signal)
                      for signal in analysis.signals}
            trace = Trace("sweep", axis, values,
                          converged=require_convergence)
        elif analysis.kind == "tran":
            results, partial, error = run_directnet_transient(path)
            values = {signal: _lookup_signal(results, signal)
                      for signal in analysis.signals}
            trace = Trace(
                "time", np.asarray(results["time"]), values,
                converged=not partial, partial=partial, error=error,
            )
        elif analysis.kind == "ac":
            trace = _run_candidate_ac_trace(path, analysis.signals)
        elif analysis.kind == "op":
            trace = _run_candidate_op_trace(path, analysis.signals)
        else:
            raise ValueError(f"unsupported analysis kind {analysis.kind!r}")
    finally:
        logging.disable(logging.NOTSET)
    if trace.partial:
        trace.validate(max_step=analysis_max_step(analysis))
    else:
        start, stop = analysis_axis_limits(analysis)
        trace.validate(
            expected_start=start,
            expected_stop=stop,
            endpoint_tolerance=analysis_endpoint_tolerance(analysis),
            max_step=analysis_max_step(analysis),
            minimum_points=analysis_minimum_points(analysis),
        )
    return trace, path


def run_level72_control_trace(
    deck: str,
    analysis: AnalysisSpec,
    work_dir: Path,
    tag: str,
    *,
    modelcard: Path,
) -> Trace:
    """Run the same deck through PyCircuitSim's LEVEL=72 solver adapter."""
    from pycircuitsim.parser import Parser
    from pycircuitsim.simulation import (
        _solve_dc_with_retry,
        build_ac_frequencies,
    )
    from pycircuitsim.solver import ACSolver, DCSolver

    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / f"control_{tag}.sp"
    path.write_text(deck)
    parser = Parser(modelcard_path=str(modelcard))
    parser.parse_file(str(path))
    circuit = parser.circuit
    if analysis.kind == "dc":
        results = run_directnet_dc_sweep(
            path,
            work_dir,
            f"control_{tag}",
            require_convergence=True,
            parsed=parser,
        )
        first = _lookup_signal(results, analysis.signals[0])
        trace = Trace(
            "sweep",
            _dc_axis(parser.analysis_params, first.size),
            {signal: _lookup_signal(results, signal)
             for signal in analysis.signals},
        )
    elif analysis.kind == "tran":
        results, partial, error = run_directnet_transient(path, parsed=parser)
        trace = Trace(
            "time",
            np.asarray(results["time"]),
            {signal: _lookup_signal(results, signal)
             for signal in analysis.signals},
            converged=not partial,
            partial=partial,
            error=error,
        )
        if partial:
            raise RuntimeError(error or "LEVEL=72 control transient ended early")
    else:
        def _solve(use_gmin: bool) -> Tuple[DCSolver, Dict[str, float]]:
            solver = DCSolver(
                circuit,
                initial_guess=circuit.initial_conditions or None,
                use_source_stepping=True,
                use_gmin_stepping=use_gmin,
            )
            return solver, solver.solve()

        solver, solution = _solve_dc_with_retry(circuit, False, _solve)
        if not solver._last_solve_converged:
            raise RuntimeError("LEVEL=72 control operating point did not converge")
        if analysis.kind == "op":
            raw: Dict[str, Any] = dict(solution)
            for component in circuit.components:
                try:
                    raw[f"i({component.name})"] = np.asarray([
                        component.calculate_current(solution)
                    ])
                except (AttributeError, NotImplementedError):
                    continue
            trace = Trace(
                "op",
                np.asarray([0.0]),
                {signal: np.atleast_1d(_lookup_signal(raw, signal))
                 for signal in analysis.signals},
            )
        elif analysis.kind == "ac":
            axis = build_ac_frequencies(parser.analysis_params)
            results = ACSolver(circuit, dc_solution=solution).solve(axis)
            trace = Trace(
                "frequency",
                axis,
                {signal: _lookup_signal(results, signal)
                 for signal in analysis.signals},
            )
        else:
            raise ValueError(f"unsupported control analysis {analysis.kind!r}")
    start, stop = analysis_axis_limits(analysis)
    trace.validate(
        expected_start=start,
        expected_stop=stop,
        endpoint_tolerance=analysis_endpoint_tolerance(analysis),
        max_step=analysis_max_step(analysis),
        minimum_points=analysis_minimum_points(analysis),
    )
    return trace


def _ascending(axis: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(axis)
    return axis[order], values[order]


def _interpolate(
    target: np.ndarray,
    source_axis: np.ndarray,
    source: np.ndarray,
) -> np.ndarray:
    axis, values = _ascending(source_axis, source)
    if np.iscomplexobj(values):
        log_target = np.log10(target)
        log_axis = np.log10(axis)
        mag = np.interp(log_target, log_axis, np.abs(values))
        phase = np.interp(log_target, log_axis, np.unwrap(np.angle(values)))
        return mag * np.exp(1j * phase)
    return np.interp(target, axis, values)


def _common_grid(candidate: Trace, reference: Trace) -> np.ndarray:
    if candidate.axis.size == reference.axis.size == 1:
        if not np.isclose(candidate.axis[0], reference.axis[0]):
            raise ValueError("candidate/reference scalar axes differ")
        return np.asarray([float(candidate.axis[0])])
    lo = max(float(np.min(candidate.axis)), float(np.min(reference.axis)))
    hi = min(float(np.max(candidate.axis)), float(np.max(reference.axis)))
    if not hi > lo:
        raise ValueError("candidate/reference axes do not overlap")
    count = min(max(min(candidate.axis.size, reference.axis.size), 64), 600)
    if candidate.axis_name == "frequency":
        return np.logspace(math.log10(lo), math.log10(hi), count)
    return np.linspace(lo, hi, count)


def _metric_key(signal: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", signal.lower()).strip("_")


def _phase_aligned_nrmse(test: np.ndarray, reference: np.ndarray) -> float:
    if test.size < 8 or reference.size != test.size:
        return float("nan")
    a = np.asarray(test, dtype=float) - float(np.mean(test))
    b = np.asarray(reference, dtype=float) - float(np.mean(reference))
    max_lag = max(1, min(test.size // 10, 80))
    correlation = np.correlate(a, b, mode="full")
    center = test.size - 1
    window = correlation[center - max_lag:center + max_lag + 1]
    lag = int(np.argmax(window)) - max_lag
    shifted = np.roll(test, -lag)
    if lag > 0:
        shifted = shifted[:-lag]
        ref = reference[:-lag]
    elif lag < 0:
        shifted = shifted[-lag:]
        ref = reference[-lag:]
    else:
        ref = reference
    return full_metrics(shifted, ref)["nrmse_pct"]


def _crossing(axis: np.ndarray, values: np.ndarray, level: float) -> float:
    shifted = values - level
    indexes = np.flatnonzero(shifted[:-1] * shifted[1:] <= 0)
    if indexes.size == 0:
        return float("nan")
    index = int(indexes[0])
    y0, y1 = values[index], values[index + 1]
    fraction = ((level - y0) / (y1 - y0)) if y1 != y0 else 0.0
    return float(axis[index] + fraction * (axis[index + 1] - axis[index]))


def _first_edge_duration(
    axis: np.ndarray,
    values: np.ndarray,
    low: float,
    high: float,
) -> float:
    """First 10–90 or 90–10 edge duration in a transient trace."""
    for index in range(values.size - 1):
        rising = values[index] <= low < values[index + 1]
        falling = values[index] >= high > values[index + 1]
        if not rising and not falling:
            continue
        start_level, stop_level = ((low, high) if rising else (high, low))
        start = _crossing(axis[index:], values[index:], start_level)
        for stop_index in range(index, values.size - 1):
            y0, y1 = values[stop_index], values[stop_index + 1]
            crossed = ((y0 <= stop_level < y1) if rising
                       else (y0 >= stop_level > y1))
            if crossed:
                stop = _crossing(
                    axis[stop_index:], values[stop_index:], stop_level,
                )
                return abs(stop - start)
        return float("nan")
    return float("nan")


def _clock_hold_windows(clock: np.ndarray, level: float) -> List[slice]:
    falling = np.flatnonzero((clock[:-1] > level) & (clock[1:] <= level))
    rising = np.flatnonzero((clock[:-1] <= level) & (clock[1:] > level))
    windows: List[slice] = []
    for edge in falling:
        start = int(edge + 1)
        later = rising[rising >= start]
        stop = int(later[0] + 1) if later.size else int(clock.size)
        if stop - start >= 2:
            windows.append(slice(start, stop))
    return windows


def clock_hold_samples(
    axis: np.ndarray,
    storage: np.ndarray,
    clock: np.ndarray,
    *,
    level: float,
) -> np.ndarray:
    """Sample storage at the center of every clock-defined hold interval."""
    time = np.asarray(axis, dtype=float)
    values = np.asarray(storage, dtype=float)
    clock_values = np.asarray(clock, dtype=float)
    if time.ndim != 1 or values.shape != time.shape or clock_values.shape != time.shape:
        raise ValueError("clock hold sampling requires aligned one-dimensional traces")
    windows = _clock_hold_windows(clock_values, level)
    return np.asarray([
        float(np.median(values[window])) for window in windows
    ], dtype=float)


def _relative_error(test: float, reference: float) -> float:
    if not np.isfinite(test) or not np.isfinite(reference) or abs(reference) < 1e-30:
        return float("nan")
    return abs(test - reference) / abs(reference) * 100.0


def _gradient_gain(axis: np.ndarray, values: np.ndarray) -> float:
    return float(np.max(np.abs(np.gradient(values, axis))))


def _bandwidth(axis: np.ndarray, values: np.ndarray) -> float:
    """First reference-independent -3 dB crossing of an AC response."""
    magnitude = np.abs(values)
    if magnitude.size < 2 or magnitude[0] <= 0.0:
        return float("nan")
    target = float(magnitude[0] / math.sqrt(2.0))
    indexes = np.flatnonzero(magnitude <= target)
    if indexes.size == 0 or int(indexes[0]) == 0:
        return float("nan")
    index = int(indexes[0])
    x0, x1 = math.log10(float(axis[index - 1])), math.log10(float(axis[index]))
    y0, y1 = float(magnitude[index - 1]), float(magnitude[index])
    fraction = (target - y0) / (y1 - y0) if y1 != y0 else 0.0
    return float(10.0 ** (x0 + fraction * (x1 - x0)))


def _period(axis: np.ndarray, values: np.ndarray, level: float) -> float:
    shifted = values - level
    indexes = np.flatnonzero((shifted[:-1] < 0) & (shifted[1:] >= 0))
    if indexes.size < 3:
        return float("nan")
    crossings = []
    for index in indexes:
        y0, y1 = values[index], values[index + 1]
        fraction = ((level - y0) / (y1 - y0)) if y1 != y0 else 0.0
        crossings.append(axis[index] + fraction * (axis[index + 1] - axis[index]))
    return float(np.mean(np.diff(crossings)))


#: Required domain outputs for every implemented metric profile.  This is both
#: the catalog vocabulary and the runtime evidence contract: an event metric
#: that cannot be measured is an error, not a present-but-NaN result.
METRIC_CONTRACTS: Dict[str, Tuple[str, ...]] = {
    "trace": (), "transient": (), "ac": (), "cascode_ac": (),
    "common_source_ac": (
        "gain_test", "gain_ref", "gain_error_pct", "bandwidth_test_hz",
        "bandwidth_ref_hz", "bandwidth_error_pct",
    ),
    "common_source_floating_ac": (
        "gain_test", "gain_ref", "gain_error_pct", "bandwidth_test_hz",
        "bandwidth_ref_hz", "bandwidth_error_pct",
        "bulk_response_max_error_v",
    ),
    "inverter_vtc": ("trip_shift_v", "leakage_error_a"),
    "inverter_energy": (
        "delay_error_pct", "energy_test_j", "energy_ref_j",
        "energy_error_pct", "leakage_error_a",
    ),
    "active_load_op": ("output_error_v", "mirror_error_v"),
    "active_load_ac": (
        "gain_test", "gain_ref", "gain_error_pct", "bandwidth_test_hz",
        "bandwidth_ref_hz", "bandwidth_error_pct",
    ),
    "source_follower": ("gain_test", "gain_ref", "gain_error_pct"),
    "gain": ("gain_test", "gain_ref", "gain_error_pct"),
    "opamp": ("gain_test", "gain_ref", "gain_error_pct"),
    "ring_osc": ("period_test_s", "period_ref_s", "period_error_pct"),
    "current_mirror": (
        "ratio_test", "ratio_ref", "ratio_error_pct",
        "output_resistance_error_pct",
    ),
    "cascode": ("output_resistance_error_pct",),
    "inverter_chain": (
        "delay_error_pct", "amplitude_error_pct", "rise_fall_error_pct",
    ),
    "logic_tran": (
        "delay_error_pct", "amplitude_error_pct", "rise_fall_error_pct",
    ),
    "logic_vtc": ("trip_shift_v",),
    "transmission_gate": ("ron_error_pct",),
    "hold_droop": (
        "droop_test_v", "droop_ref_v", "droop_error_v",
        "feedthrough_error_v",
    ),
    "switchcap": (
        "droop_test_v", "droop_ref_v", "droop_error_v",
        "feedthrough_error_v", "charge_error_vdd_pct",
    ),
    "diffpair": ("diff_gain_error_pct",),
    "diffpair_diff_ac": (
        "diff_gain_test", "diff_gain_ref", "diff_gain_error_pct",
    ),
    "diffpair_cm_ac": (
        "cm_gain_test", "cm_gain_ref", "cm_gain_error_pct",
    ),
    "opamp_diff_ac": (
        "diff_gain_test", "diff_gain_ref", "diff_gain_error_pct",
    ),
    "opamp_cm_ac": ("cm_gain_test", "cm_gain_ref", "cm_gain_error_pct"),
    "opamp_supply_ac": (
        "supply_gain_test", "supply_gain_ref", "supply_gain_error_pct",
    ),
    "mirror_iref": (
        "iref_points", "iref_ratio_test", "iref_ratio_ref",
        "iref_ratio_error_pct", "iref_worst_ratio_error_pct",
    ),
    "switchcap_multicycle": (
        "cycle_drift_test_v", "cycle_drift_ref_v", "cycle_drift_error_v",
        "final_sample_error_v",
    ),
    "ring_supply": (
        "period_test_s", "period_ref_s", "period_error_pct",
        "supply_current_test_a", "supply_current_ref_a",
        "supply_current_error_pct",
    ),
    "diode_load": ("diode_drop_error_v", "diode_drop_worst_error_v"),
    "bias_op": ("bias_current_error_pct", "bias_node_error_v"),
    "bias_fanout_op": ("bias_node_error_v", "supply_current_error_pct"),
    "self_bias_cell": (
        "bias_current_test_a", "bias_current_ref_a",
        "bias_current_error_pct", "startup_vdd_test_v",
        "startup_vdd_ref_v", "startup_vdd_error_v", "bias_node_error_v",
    ),
    "self_bias_cascode": (
        "output_resistance_error_pct", "bias_node_error_v",
    ),
    "mos_reference": ("vref_error_v", "line_sensitivity_error_pct"),
    "unity_gain": (
        "follow_error_test_v", "follow_error_ref_v", "follow_error_v",
        "closed_loop_gain_error_pct",
    ),
    "settling": (
        "settling_test_s", "settling_ref_s", "settling_error_pct",
        "overshoot_error_v",
    ),
    "line_regulation": (
        "vout_error_v", "line_regulation_test", "line_regulation_ref",
        "line_regulation_error_pct",
    ),
    "load_regulation": (
        "load_droop_test_v", "load_droop_ref_v", "load_droop_error_v",
        "recovery_error_v",
    ),
    "closed_loop_ac": (
        "gain_test", "gain_ref", "gain_error_pct", "bandwidth_test_hz",
        "bandwidth_ref_hz", "bandwidth_error_pct", "peaking_test_db",
        "peaking_ref_db", "peaking_error_db",
    ),
    "ldo_psrr_ac": ("psrr_test_db", "psrr_ref_db", "psrr_error_db"),
    "ldo_output_impedance_ac": (
        "output_impedance_test_ohm", "output_impedance_ref_ohm",
        "output_impedance_error_pct",
    ),
    "sram_write_margin": (
        "write_trip_test_v", "write_trip_ref_v", "write_trip_error_v",
    ),
    "sram_hold": ("hold_margin_error_v", "retention"),
    "sram_read": ("read_disturb_error_v",),
    "sram_write": ("write_time_error_pct", "write_final_error_v"),
    "sram_snm": ("hold_margin_error_v", "retention", "positive"),
}

METRIC_PROFILES: frozenset = frozenset(METRIC_CONTRACTS)
_AGGREGATE_METRICS: Tuple[str, ...] = (
    "mre_pct", "r2", "nrmse_pct", "max_err",
)


def validate_analysis_metrics(
    analysis: AnalysisSpec,
    metrics: Mapping[str, Any],
    domain: Mapping[str, Any],
) -> None:
    """Require every promised metric to exist and carry a finite value."""
    if analysis.metric_profile not in METRIC_CONTRACTS:
        raise ValueError(f"unknown metric profile {analysis.metric_profile!r}")
    payload = {**metrics, **domain}
    required = (*_AGGREGATE_METRICS, *METRIC_CONTRACTS[analysis.metric_profile])
    missing = sorted(name for name in required if name not in payload)
    if missing:
        raise ValueError(f"missing required metrics: {missing}")
    boolean_metrics = {"positive", "retention"}
    nonnumeric = sorted(
        name for name in required
        if (
            name in boolean_metrics
            and not isinstance(payload[name], (bool, np.bool_))
        ) or (
            name not in boolean_metrics
            and (
                isinstance(payload[name], (bool, np.bool_))
                or not isinstance(
                    payload[name], (int, float, np.integer, np.floating),
                )
            )
        )
    )
    if nonnumeric:
        raise ValueError(f"required metrics must be numeric: {nonnumeric}")
    nonfinite = sorted(
        name for name in required
        if not isinstance(payload[name], (bool, np.bool_))
        and not np.isfinite(payload[name])
    )
    if nonfinite:
        raise ValueError(f"non-finite required metrics: {nonfinite}")


def analysis_metric_vocabulary(analysis: AnalysisSpec) -> frozenset[str]:
    """Return every stable aggregate/domain key this profile can emit."""
    vocabulary = set(_AGGREGATE_METRICS)
    vocabulary.update(METRIC_CONTRACTS.get(analysis.metric_profile, ()))
    if analysis.phase_align:
        vocabulary.add("phase_aligned_nrmse_pct")
    if analysis.metric_profile in {"logic_vtc", "logic_tran", "active_load_op"}:
        vocabulary.add("internal_node_nrmse_pct")
    return frozenset(vocabulary)


def _domain_metrics(
    profile: str,
    grid: np.ndarray,
    candidate: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    vdd: float,
) -> Dict[str, Any]:
    domain: Dict[str, Any] = {}
    names = list(candidate)
    if not names:
        return domain
    if profile in (
        "common_source_ac", "common_source_floating_ac", "active_load_ac",
    ):
        output = names[0]
        gain_test = float(np.abs(candidate[output][0]))
        gain_ref = float(np.abs(reference[output][0]))
        bandwidth_test = _bandwidth(grid, candidate[output])
        bandwidth_ref = _bandwidth(grid, reference[output])
        domain.update(
            gain_test=gain_test,
            gain_ref=gain_ref,
            gain_error_pct=_relative_error(gain_test, gain_ref),
            bandwidth_test_hz=bandwidth_test,
            bandwidth_ref_hz=bandwidth_ref,
            bandwidth_error_pct=_relative_error(bandwidth_test, bandwidth_ref),
        )
        if profile == "common_source_floating_ac" and len(names) >= 2:
            bulk = names[1]
            domain["bulk_response_max_error_v"] = float(np.max(np.abs(
                candidate[bulk] - reference[bulk]
            )))
    if profile == "active_load_op" and len(names) >= 2:
        domain.update(
            output_error_v=abs(float(np.real(
                candidate[names[0]][0] - reference[names[0]][0]
            ))),
            mirror_error_v=abs(float(np.real(
                candidate[names[1]][0] - reference[names[1]][0]
            ))),
        )
    if profile == "closed_loop_ac":
        output = names[0]
        test = np.abs(candidate[output])
        ref = np.abs(reference[output])
        gain_test, gain_ref = float(test[0]), float(ref[0])
        bandwidth_test = _bandwidth(grid, candidate[output])
        bandwidth_ref = _bandwidth(grid, reference[output])
        peaking_test = 20.0 * math.log10(
            max(float(np.max(test)) / max(gain_test, 1e-30), 1e-30)
        )
        peaking_ref = 20.0 * math.log10(
            max(float(np.max(ref)) / max(gain_ref, 1e-30), 1e-30)
        )
        domain.update(
            gain_test=gain_test, gain_ref=gain_ref,
            gain_error_pct=_relative_error(gain_test, gain_ref),
            bandwidth_test_hz=bandwidth_test,
            bandwidth_ref_hz=bandwidth_ref,
            bandwidth_error_pct=_relative_error(bandwidth_test, bandwidth_ref),
            peaking_test_db=peaking_test, peaking_ref_db=peaking_ref,
            peaking_error_db=abs(peaking_test - peaking_ref),
        )
    if profile == "ldo_psrr_ac":
        gain_test = float(np.abs(candidate[names[0]][0]))
        gain_ref = float(np.abs(reference[names[0]][0]))
        psrr_test = -20.0 * math.log10(max(gain_test, 1e-30))
        psrr_ref = -20.0 * math.log10(max(gain_ref, 1e-30))
        domain.update(
            psrr_test_db=psrr_test,
            psrr_ref_db=psrr_ref,
            psrr_error_db=abs(psrr_test - psrr_ref),
        )
    if profile == "ldo_output_impedance_ac":
        impedance_test = float(np.abs(candidate[names[0]][0]))
        impedance_ref = float(np.abs(reference[names[0]][0]))
        domain.update(
            output_impedance_test_ohm=impedance_test,
            output_impedance_ref_ohm=impedance_ref,
            output_impedance_error_pct=_relative_error(
                impedance_test, impedance_ref,
            ),
        )
    if profile == "inverter_vtc" and len(names) >= 2:
        output, current = names[0], names[1]
        trip_test = _crossing(grid, np.real(candidate[output]), vdd / 2.0)
        trip_ref = _crossing(grid, np.real(reference[output]), vdd / 2.0)
        domain.update(
            trip_shift_v=abs(trip_test - trip_ref),
            leakage_error_a=float(np.max(np.abs(
                np.real(candidate[current][[0, -1]])
                - np.real(reference[current][[0, -1]])
            ))),
        )
    if profile == "inverter_energy" and len(names) >= 3:
        input_name, output, current = names[0], names[1], names[2]
        input_test = _crossing(
            grid, np.real(candidate[input_name]), vdd / 2.0,
        )
        input_ref = _crossing(
            grid, np.real(reference[input_name]), vdd / 2.0,
        )
        output_test = _crossing(grid, np.real(candidate[output]), vdd / 2.0)
        output_ref = _crossing(grid, np.real(reference[output]), vdd / 2.0)
        delay_test = abs(output_test - input_test)
        delay_ref = abs(output_ref - input_ref)
        energy_test = float(np.trapezoid(
            np.abs(np.real(candidate[current])), grid,
        ) * vdd)
        energy_ref = float(np.trapezoid(
            np.abs(np.real(reference[current])), grid,
        ) * vdd)
        pre = max(grid.size // 8, 1)
        leakage_test = float(np.mean(np.abs(np.real(candidate[current][:pre]))))
        leakage_ref = float(np.mean(np.abs(np.real(reference[current][:pre]))))
        domain.update(
            delay_error_pct=_relative_error(delay_test, delay_ref),
            energy_test_j=energy_test,
            energy_ref_j=energy_ref,
            energy_error_pct=_relative_error(energy_test, energy_ref),
            leakage_error_a=abs(leakage_test - leakage_ref),
        )
    if profile in ("source_follower", "gain", "opamp"):
        name = names[0]
        gain_test = _gradient_gain(grid, np.real(candidate[name]))
        gain_ref = _gradient_gain(grid, np.real(reference[name]))
        domain.update(gain_test=gain_test, gain_ref=gain_ref,
                      gain_error_pct=_relative_error(gain_test, gain_ref))
    if profile == "ring_osc":
        name = names[0]
        test = _period(grid, np.real(candidate[name]), vdd / 2.0)
        ref = _period(grid, np.real(reference[name]), vdd / 2.0)
        domain.update(period_test_s=test, period_ref_s=ref,
                      period_error_pct=_relative_error(test, ref))
    if profile == "current_mirror":
        # Iref is an ideal source whose value is identical in both decks;
        # NGSPICE does not expose i(I*) as a wrdata vector.  Comparing the
        # median output currents therefore gives the same mirror-ratio error
        # without asking either engine for a synthetic reference-current row.
        current_name = names[0]
        # Score the saturation/compliance half of each sweep: high Vout for
        # the NMOS sink and low Vout for the PMOS source.  Including the
        # triode knee would turn this into an on-resistance measurement.
        compliance = (grid <= 0.5 * vdd if "outp" in current_name.lower()
                      else grid >= 0.5 * vdd)
        test_ratio = np.median(np.abs(candidate[current_name][compliance]))
        ref_ratio = np.median(np.abs(reference[current_name][compliance]))
        gt = np.gradient(np.real(candidate[current_name]), grid)
        gr = np.gradient(np.real(reference[current_name]), grid)
        r_test = 1.0 / (np.median(np.abs(gt[compliance])) + 1e-30)
        r_ref = 1.0 / (np.median(np.abs(gr[compliance])) + 1e-30)
        domain.update(
            ratio_test=float(test_ratio), ratio_ref=float(ref_ratio),
            ratio_error_pct=_relative_error(float(test_ratio), float(ref_ratio)),
            output_resistance_error_pct=_relative_error(r_test, r_ref),
        )
    if profile in ("cascode",):
        current_name = next((name for name in names if name.lower().startswith("i(")),
                            names[0])
        compliance = (grid <= 0.5 * vdd if "outp" in current_name.lower()
                      else grid >= 0.5 * vdd)
        gt = np.gradient(np.real(candidate[current_name]), grid)
        gr = np.gradient(np.real(reference[current_name]), grid)
        domain["output_resistance_error_pct"] = _relative_error(
            1.0 / (np.median(np.abs(gt[compliance])) + 1e-30),
            1.0 / (np.median(np.abs(gr[compliance])) + 1e-30),
        )
    if profile in ("inverter_chain", "logic_tran") and len(names) >= 2:
        input_name = names[0]
        # For the FO4 case, names[1] is the loaded driver output; the final
        # receiver output remains a separately scored trace but is not the
        # propagation-delay endpoint.
        output_name = names[1]
        tin_t = _crossing(grid, np.real(candidate[input_name]), vdd / 2.0)
        tin_r = _crossing(grid, np.real(reference[input_name]), vdd / 2.0)
        tout_t = _crossing(grid, np.real(candidate[output_name]), vdd / 2.0)
        tout_r = _crossing(grid, np.real(reference[output_name]), vdd / 2.0)
        delay_t, delay_r = abs(tout_t - tin_t), abs(tout_r - tin_r)
        amp_t = float(np.ptp(np.real(candidate[output_name])))
        amp_r = float(np.ptp(np.real(reference[output_name])))
        edge_t = _first_edge_duration(
            grid, np.real(candidate[output_name]), 0.1 * vdd, 0.9 * vdd,
        )
        edge_r = _first_edge_duration(
            grid, np.real(reference[output_name]), 0.1 * vdd, 0.9 * vdd,
        )
        domain.update(
            delay_error_pct=_relative_error(delay_t, delay_r),
            amplitude_error_pct=_relative_error(amp_t, amp_r),
            rise_fall_error_pct=_relative_error(edge_t, edge_r),
        )
    if profile == "transmission_gate":
        voltage = next((name for name in names if name.lower().startswith("v(")), None)
        current = next((name for name in names if name.lower().startswith("i(")), None)
        if voltage and current:
            rt = np.median(np.abs((grid - np.real(candidate[voltage])) /
                                  (np.real(candidate[current]) + 1e-30)))
            rr = np.median(np.abs((grid - np.real(reference[voltage])) /
                                  (np.real(reference[current]) + 1e-30)))
            domain["ron_error_pct"] = _relative_error(float(rt), float(rr))
    if profile in ("hold_droop", "switchcap"):
        name = names[0]
        clock_name = next((item for item in names[1:]
                           if "phi" in item.lower()), "")
        windows = (
            _clock_hold_windows(np.real(reference[clock_name]), vdd / 2.0)
            if clock_name else []
        )
        if windows:
            window = windows[-1]
            test_tail = np.real(candidate[name][window])
            ref_tail = np.real(reference[name][window])
            edge = window.start
            feedthrough_t = abs(float(np.real(candidate[name][edge]
                                              - candidate[name][edge - 1])))
            feedthrough_r = abs(float(np.real(reference[name][edge]
                                              - reference[name][edge - 1])))
        else:
            half = grid.size // 2
            test_tail = np.real(candidate[name][half:])
            ref_tail = np.real(reference[name][half:])
            feedthrough_t = float(np.max(np.abs(np.diff(test_tail))))
            feedthrough_r = float(np.max(np.abs(np.diff(ref_tail))))
        droop_t = float(np.ptp(test_tail))
        droop_r = float(np.ptp(ref_tail))
        domain.update(
            droop_test_v=droop_t, droop_ref_v=droop_r,
            droop_error_v=abs(droop_t - droop_r),
            feedthrough_error_v=abs(feedthrough_t - feedthrough_r),
        )
        if profile == "switchcap":
            domain["charge_error_vdd_pct"] = (
                abs(float(test_tail[0] - ref_tail[0])) / vdd * 100.0)
    if profile == "diffpair" and len(names) >= 2:
        test_diff = np.real(candidate[names[1]] - candidate[names[0]])
        ref_diff = np.real(reference[names[1]] - reference[names[0]])
        gain_t = _gradient_gain(grid, test_diff)
        gain_r = _gradient_gain(grid, ref_diff)
        domain["diff_gain_error_pct"] = _relative_error(gain_t, gain_r)
    if profile == "diffpair_diff_ac" and len(names) >= 2:
        gain_t = float(np.abs(candidate[names[1]][0] - candidate[names[0]][0]))
        gain_r = float(np.abs(reference[names[1]][0] - reference[names[0]][0]))
        domain.update(diff_gain_test=gain_t, diff_gain_ref=gain_r,
                      diff_gain_error_pct=_relative_error(gain_t, gain_r))
    if profile == "diffpair_cm_ac" and len(names) >= 2:
        gain_t = float(np.abs((candidate[names[1]][0]
                              + candidate[names[0]][0]) / 2.0))
        gain_r = float(np.abs((reference[names[1]][0]
                              + reference[names[0]][0]) / 2.0))
        domain.update(cm_gain_test=gain_t, cm_gain_ref=gain_r,
                      cm_gain_error_pct=_relative_error(gain_t, gain_r))
    if profile in ("opamp_diff_ac", "opamp_cm_ac", "opamp_supply_ac"):
        # Low-frequency magnitude of the single output.  The rejection ratios
        # are formed from these gains by `_derive_case_metrics`; no analysis
        # can produce a ratio on its own because each drives one stimulus.
        key = {"opamp_diff_ac": "diff", "opamp_cm_ac": "cm",
               "opamp_supply_ac": "supply"}[profile]
        gain_t = float(np.abs(candidate[names[0]][0]))
        gain_r = float(np.abs(reference[names[0]][0]))
        domain.update(**{
            f"{key}_gain_test": gain_t, f"{key}_gain_ref": gain_r,
            f"{key}_gain_error_pct": _relative_error(gain_t, gain_r),
        })
    if profile == "mirror_iref":
        # The sweep axis *is* the reference current, so the mirror ratio is
        # the measured output current divided by the axis.  Points at a
        # negligible reference current carry no ratio and are dropped from the
        # aggregate rather than being clamped into it.
        name = names[0]
        usable = np.abs(grid) > 1e-12
        if int(np.count_nonzero(usable)) >= 3:
            ratio_t = np.abs(np.real(candidate[name][usable])) / np.abs(grid[usable])
            ratio_r = np.abs(np.real(reference[name][usable])) / np.abs(grid[usable])
            errors = np.abs(ratio_t - ratio_r) / np.maximum(np.abs(ratio_r), 1e-30)
            domain.update(
                iref_points=int(np.count_nonzero(usable)),
                iref_ratio_test=float(np.median(ratio_t)),
                iref_ratio_ref=float(np.median(ratio_r)),
                iref_ratio_error_pct=float(np.median(errors) * 100.0),
                iref_worst_ratio_error_pct=float(np.max(errors) * 100.0),
            )
    if profile == "switchcap_multicycle":
        name = names[0]
        clock_name = next((item for item in names[1:]
                           if "phi" in item.lower()), "")
        if clock_name:
            test_samples = clock_hold_samples(
                grid, np.real(candidate[name]), np.real(candidate[clock_name]),
                level=vdd / 2.0,
            )
            ref_samples = clock_hold_samples(
                grid, np.real(reference[name]), np.real(reference[clock_name]),
                level=vdd / 2.0,
            )
            count = min(test_samples.size, ref_samples.size)
            if count >= 2:
                test_samples = test_samples[:count]
                ref_samples = ref_samples[:count]
                drift_t = float(test_samples[-1] - test_samples[0])
                drift_r = float(ref_samples[-1] - ref_samples[0])
                domain.update(
                    hold_samples=count,
                    cycle_drift_test_v=drift_t,
                    cycle_drift_ref_v=drift_r,
                    cycle_drift_error_v=abs(drift_t - drift_r),
                    final_sample_error_v=abs(float(
                        test_samples[-1] - ref_samples[-1]
                    )),
                )
    if profile == "ring_supply":
        voltage = next((name for name in names if name.lower().startswith("v(")),
                       names[0])
        test = _period(grid, np.real(candidate[voltage]), vdd / 2.0)
        ref = _period(grid, np.real(reference[voltage]), vdd / 2.0)
        domain.update(period_test_s=test, period_ref_s=ref,
                      period_error_pct=_relative_error(test, ref))
        current = next((name for name in names
                        if name.lower().startswith("i(")), "")
        if current:
            # Mean supply draw over the settled window: an oscillator's
            # dynamic current is set by the charge model, which no other
            # simple-circuit metric observes.
            settled = slice(max(grid.size // 4, 1), None)
            draw_t = float(np.mean(np.abs(np.real(candidate[current][settled]))))
            draw_r = float(np.mean(np.abs(np.real(reference[current][settled]))))
            domain.update(
                supply_current_test_a=draw_t, supply_current_ref_a=draw_r,
                supply_current_error_pct=_relative_error(draw_t, draw_r),
            )
    if profile == "diode_load":
        # With the load line fixed by the resistor, the node voltage IS the
        # model's answer: there is no source imposing the current, so the
        # diode drop is the whole result.
        domain["diode_drop_error_v"] = max(
            abs(float(np.real(candidate[name][-1])
                      - np.real(reference[name][-1])))
            for name in names
        )
        domain["diode_drop_worst_error_v"] = max(
            float(np.max(np.abs(np.real(candidate[name])
                                - np.real(reference[name]))))
            for name in names
        )
    if profile == "bias_op":
        current = next((name for name in names
                        if name.lower().startswith("i(")), "")
        voltages = [name for name in names if name.lower().startswith("v(")]
        if current:
            current_test = float(abs(np.real(candidate[current][0])))
            current_ref = float(abs(np.real(reference[current][0])))
            domain["bias_current_error_pct"] = _relative_error(
                current_test, current_ref,
            )
        if voltages:
            domain["bias_node_error_v"] = max(
                abs(float(np.real(candidate[name][0]
                                  - reference[name][0])))
                for name in voltages
            )
    if profile == "bias_fanout_op" and len(names) >= 2:
        bias_name, current_name = names[0], names[1]
        test_bias = np.real(candidate[bias_name])
        ref_bias = np.real(reference[bias_name])
        draw_test = float(abs(np.real(candidate[current_name][0])))
        draw_ref = float(abs(np.real(reference[current_name][0])))
        domain.update(
            bias_node_error_v=abs(float(test_bias[-1] - ref_bias[-1])),
            supply_current_error_pct=_relative_error(draw_test, draw_ref),
        )
    if profile == "self_bias_cell":
        # Supply ramp of a self-biased cell.  Both engines are asked the same
        # two questions: how much current the loop settles at, and at which
        # supply it leaves the degenerate zero-current solution.
        current = next((name for name in names
                        if name.lower().startswith("i(")), "")
        voltages = [name for name in names if name.lower().startswith("v(")]
        if current:
            draw_t = float(abs(np.real(candidate[current][-1])))
            draw_r = float(abs(np.real(reference[current][-1])))
            domain.update(
                bias_current_test_a=draw_t, bias_current_ref_a=draw_r,
                bias_current_error_pct=_relative_error(draw_t, draw_r),
            )
            starts = []
            for values in (candidate[current], reference[current]):
                magnitude = np.abs(np.real(values))
                target = 0.5 * float(magnitude[-1])
                reached = np.flatnonzero(magnitude >= target)
                starts.append(float(grid[reached[0]]) if reached.size
                              else float("nan"))
            domain.update(startup_vdd_test_v=starts[0],
                          startup_vdd_ref_v=starts[1],
                          startup_vdd_error_v=abs(starts[0] - starts[1]))
        if voltages:
            domain["bias_node_error_v"] = max(
                abs(float(np.real(candidate[name][-1])
                          - np.real(reference[name][-1])))
                for name in voltages
            )
    if profile == "self_bias_cascode":
        current = next((name for name in names
                        if name.lower().startswith("i(")), names[0])
        compliance = grid >= 0.5 * vdd
        if int(np.count_nonzero(compliance)) >= 3:
            gt = np.gradient(np.real(candidate[current]), grid)
            gr = np.gradient(np.real(reference[current]), grid)
            domain["output_resistance_error_pct"] = _relative_error(
                1.0 / (np.median(np.abs(gt[compliance])) + 1e-30),
                1.0 / (np.median(np.abs(gr[compliance])) + 1e-30),
            )
        # The internally generated rails are the point of this case: report
        # them so a compliance error can be attributed to bias generation
        # rather than to the cascode devices.
        rails = [name for name in names if name.lower().startswith("v(")]
        if rails:
            domain["bias_node_error_v"] = max(
                abs(float(np.median(np.real(candidate[name]))
                          - np.median(np.real(reference[name]))))
                for name in rails
            )
    if profile == "mos_reference":
        name = names[0]
        test = np.real(candidate[name])
        ref = np.real(reference[name])
        domain["vref_error_v"] = abs(float(test[-1] - ref[-1]))
        # Line sensitivity over the top of the supply ramp: dVref/dVdd is what
        # a reference is actually specified on, and it is a derivative the
        # pointwise device metrics never constrain.
        top = grid >= 0.7 * float(np.max(grid))
        if int(np.count_nonzero(top)) >= 3:
            slope_t = float(np.polyfit(grid[top], test[top], 1)[0])
            slope_r = float(np.polyfit(grid[top], ref[top], 1)[0])
            domain.update(
                line_sensitivity_test=slope_t, line_sensitivity_ref=slope_r,
                line_sensitivity_error_pct=_relative_error(slope_t, slope_r),
            )
    if profile == "unity_gain" and len(names) >= 2:
        out_name, in_name = names[0], names[1]
        test_out = np.real(candidate[out_name])
        ref_out = np.real(reference[out_name])
        drive = np.real(reference[in_name])
        # A unity-gain buffer should reproduce its input.  Report the closed
        # loop's own follower error for each engine, then their difference.
        domain.update(
            follow_error_test_v=float(np.max(np.abs(test_out - drive))),
            follow_error_ref_v=float(np.max(np.abs(ref_out - drive))),
            follow_error_v=abs(float(np.max(np.abs(test_out - drive)))
                               - float(np.max(np.abs(ref_out - drive)))),
        )
        gain_t = _gradient_gain(grid, test_out)
        gain_r = _gradient_gain(grid, ref_out)
        domain["closed_loop_gain_error_pct"] = _relative_error(gain_t, gain_r)
    if profile == "settling" and len(names) >= 2:
        out_name = names[0]
        test = np.real(candidate[out_name])
        ref = np.real(reference[out_name])
        settles = []
        overshoots = []
        for values in (test, ref):
            final = float(values[-1])
            span = float(np.ptp(values))
            band = max(0.02 * span, 1e-6)
            outside = np.flatnonzero(np.abs(values - final) > band)
            settles.append(float(grid[outside[-1]]) if outside.size
                           else float(grid[0]))
            overshoots.append(float(np.max(values) - final))
        domain.update(
            settling_test_s=settles[0], settling_ref_s=settles[1],
            settling_error_pct=_relative_error(settles[0], settles[1]),
            overshoot_error_v=abs(overshoots[0] - overshoots[1]),
        )
    if profile == "line_regulation":
        name = names[0]
        test = np.real(candidate[name])
        ref = np.real(reference[name])
        domain["vout_error_v"] = abs(float(test[-1] - ref[-1]))
        slope_t = float(np.polyfit(grid, test, 1)[0])
        slope_r = float(np.polyfit(grid, ref, 1)[0])
        domain.update(
            line_regulation_test=slope_t, line_regulation_ref=slope_r,
            line_regulation_error_pct=_relative_error(slope_t, slope_r),
        )
    if profile == "load_regulation":
        name = names[0]
        test = np.real(candidate[name])
        ref = np.real(reference[name])
        # Droop is measured from each engine's own pre-step level, so a static
        # output offset does not leak into the transient load-step metric.
        pre = max(grid.size // 5, 1)
        droops = []
        recoveries = []
        for values in (test, ref):
            level = float(np.median(values[:pre]))
            droops.append(level - float(np.min(values[pre:])))
            recoveries.append(float(values[-1]) - level)
        domain.update(
            load_droop_test_v=droops[0], load_droop_ref_v=droops[1],
            load_droop_error_v=abs(droops[0] - droops[1]),
            recovery_error_v=abs(recoveries[0] - recoveries[1]),
        )
    if profile == "sram_write_margin":
        name = names[0]
        trip_t = _crossing(grid, np.real(candidate[name]), vdd / 2.0)
        trip_r = _crossing(grid, np.real(reference[name]), vdd / 2.0)
        domain.update(write_trip_test_v=trip_t, write_trip_ref_v=trip_r,
                      write_trip_error_v=abs(trip_t - trip_r))
    if profile == "logic_vtc":
        name = names[0]
        trip_t = _crossing(grid, np.real(candidate[name]), vdd / 2.0)
        trip_r = _crossing(grid, np.real(reference[name]), vdd / 2.0)
        domain["trip_shift_v"] = abs(trip_t - trip_r)
    if profile in ("sram_hold", "sram_read", "sram_write", "sram_snm"):
        first = names[0]
        test = np.real(candidate[first])
        ref = np.real(reference[first])
        if profile in ("sram_hold", "sram_snm"):
            domain.update(
                hold_margin_error_v=abs(float(test[-1] - ref[-1])),
                retention=all(
                    abs(float(np.real(candidate[name][-1]
                                      - candidate[name][0]))) < 0.2 * vdd
                    for name in names
                ),
            )
        if profile == "sram_read":
            domain["read_disturb_error_v"] = abs(
                float(np.max(np.abs(test - test[0])))
                - float(np.max(np.abs(ref - ref[0])))
            )
        if profile == "sram_write":
            crossing_t = _crossing(grid, test, vdd / 2.0)
            crossing_r = _crossing(grid, ref, vdd / 2.0)
            domain["write_time_error_pct"] = _relative_error(
                crossing_t, crossing_r,
            )
            domain["write_final_error_v"] = abs(float(test[-1] - ref[-1]))
        if profile == "sram_snm":
            domain["positive"] = bool(float(np.min(test)) >= -1e-3)
    return domain


def compare_traces(
    candidate: Trace,
    reference: Trace,
    analysis: AnalysisSpec,
    *,
    vdd: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Align two traces and return required aggregate plus domain metrics."""
    grid = _common_grid(candidate, reference)
    candidate_values: Dict[str, np.ndarray] = {}
    reference_values: Dict[str, np.ndarray] = {}
    metrics: Dict[str, float] = {}
    per_signal: List[Tuple[str, Dict[str, float]]] = []
    for signal in analysis.signals:
        test = _interpolate(grid, candidate.axis, candidate.signals[signal])
        truth = _interpolate(grid, reference.axis, reference.signals[signal])
        candidate_values[signal] = test
        reference_values[signal] = truth
        basic = full_metrics(np.abs(test) if np.iscomplexobj(test) else test,
                             np.abs(truth) if np.iscomplexobj(truth) else truth)
        if grid.size == 1:
            difference = abs(float(np.real(test[0] - truth[0])))
            basic["nrmse_pct"] = difference / max(vdd, 1e-30) * 100.0
            basic["r2"] = 1.0 if difference <= 1e-12 else 0.0
        per_signal.append((signal, basic))
        prefix = _metric_key(signal)
        for name, value in basic.items():
            metrics[f"{prefix}_{name}"] = value
        if np.iscomplexobj(test):
            ratio = test / (truth + 1e-30)
            metrics[f"{prefix}_phase_maxerr_deg"] = float(
                np.max(np.abs(np.rad2deg(np.angle(ratio)))))
        if analysis.phase_align and signal.lower().startswith("v("):
            aligned = _phase_aligned_nrmse(np.real(test), np.real(truth))
            metrics[f"{prefix}_phase_aligned_nrmse_pct"] = aligned

    aggregate = per_signal or [("", {"mre_pct": float("nan"), "r2": float("nan"),
                                      "nrmse_pct": float("nan"),
                                      "max_err": float("nan")})]
    voltage = [(name, item) for name, item in aggregate
               if name.lower().startswith("v(")]
    scored = voltage or aggregate
    # NRMSE and R2 are normalized by the reference's dynamic range, so a
    # signal that barely moves — an internally generated bias rail reported
    # for attribution, say — yields an enormous NRMSE from a sub-millivolt
    # error. Those signals keep their own per-signal metrics and their domain
    # metric; they are only held out of the case aggregate, and which ones
    # were held out is recorded rather than left implicit.
    flat = [name for name, _ in scored
            if float(np.ptp(np.real(reference_values[name])))
            < max(1e-6, 1e-4 * vdd)]
    if flat and len(flat) < len(scored):
        metrics["flat_signals_excluded"] = float(len(flat))
        scored = [(name, item) for name, item in scored if name not in flat]
    metrics.update({
        "mre_pct": max(item["mre_pct"] for _, item in scored),
        "r2": min(item["r2"] for _, item in scored),
        "nrmse_pct": max(item["nrmse_pct"] for _, item in scored),
        "max_err": max(item["max_err"] for _, item in scored),
    })
    aligned_values = [value for key, value in metrics.items()
                      if key.endswith("phase_aligned_nrmse_pct")]
    if aligned_values:
        metrics["phase_aligned_nrmse_pct"] = max(aligned_values)
    if analysis.metric_profile in ("logic_vtc", "logic_tran") \
            and len(analysis.signals) >= 2:
        internal_key = _metric_key(analysis.signals[-1])
        metrics["internal_node_nrmse_pct"] = metrics[
            f"{internal_key}_nrmse_pct"
        ]
    if analysis.metric_profile == "active_load_op" \
            and len(analysis.signals) >= 2:
        internal_key = _metric_key(analysis.signals[1])
        metrics["internal_node_nrmse_pct"] = metrics[
            f"{internal_key}_nrmse_pct"
        ]
    domain = _domain_metrics(
        analysis.metric_profile, grid, candidate_values, reference_values, vdd,
    )
    return metrics, domain


def _qualification_pass(
    case: CircuitCase,
    metrics: Mapping[str, float],
    domain: Mapping[str, Any],
    *,
    vdd: float,
    partial: bool,
) -> bool:
    if partial:
        return False
    if case.case_id == "ring_osc":
        return bool(domain.get("period_error_pct", float("inf")) <= 5.0)
    if case.case_id == "opamp":
        return bool(domain.get("gain_ref", 0.0) >= 5.0
                    and domain.get("gain_error_pct", float("inf")) <= 10.0)
    if case.case_id == "sram_snm":
        return bool(metrics.get("nrmse_pct", float("inf")) <= 10.0
                    and domain.get("positive", False))
    if case.case_id == "switchcap":
        droop_ref = float(domain.get("droop_ref_v", 0.0))
        return bool(domain.get("charge_error_vdd_pct", float("inf")) <= 5.0
                    and domain.get("droop_error_v", float("inf"))
                    <= max(0.10 * droop_ref, 0.001 * vdd))
    return False


def support_diagnostic(
    candidate_path: Path,
    reference: Trace,
) -> Dict[str, Any]:
    """Check accepted LEVEL=72 terminal trajectories against NN support.

    Only accepted reference points are used.  Candidate Newton trial states
    never enter this diagnostic, so solver overshoot cannot be mislabeled as a
    compact-model coverage hole.
    """
    parser = parse_netlist(candidate_path)
    rows: Dict[str, Any] = {}
    total = outside = 0
    for component in parser.circuit.components:
        stats = getattr(component, "_norm_stats", None)
        nodes = getattr(component, "nodes", ())
        if stats is None or len(nodes) != 4:
            continue
        node_values: List[np.ndarray] = []
        for node in nodes:
            if str(node).lower() in ("0", "gnd"):
                node_values.append(np.zeros_like(reference.axis))
            else:
                node_values.append(reference.signals[f"v({node})"])
        vd, vg, vs, vb = node_values
        raw = np.column_stack((
            vd - vs, vg - vs, np.zeros_like(vs), vb - vs,
            np.full_like(vs, math.log2(max(float(component.NFIN), 1.0))),
            np.full_like(vs, float(component.L)),
            np.full_like(vs, float(component.temperature)),
        ))
        lower = np.asarray(stats.input_min, dtype=float)
        upper = np.asarray(stats.input_max, dtype=float)
        mask = (raw < lower) | (raw > upper)
        total += int(mask.size)
        outside += int(np.count_nonzero(mask))
        rows[component.name] = {
            "outside_values": int(np.count_nonzero(mask)),
            "points": int(raw.shape[0]),
            "min": raw.min(axis=0).tolist(),
            "max": raw.max(axis=0).tolist(),
        }
    return {
        "outside_values": outside,
        "checked_values": total,
        "inside": bool(total > 0 and outside == 0),
        "devices": rows,
    }


def _reference_stability(
    traces: Sequence[Trace],
    analysis: AnalysisSpec,
    vdd: float,
) -> Dict[str, Any]:
    if len(traces) < 2:
        return {"reference_repeats": len(traces)}
    worst = 0.0
    headline_worst = 0.0
    for trace in traces[1:]:
        metrics, domain = compare_traces(trace, traces[0], analysis, vdd=vdd)
        validate_analysis_metrics(analysis, metrics, domain)
        worst = max(worst, float(metrics["nrmse_pct"]))
        headline = {**metrics, **domain}.get(analysis.headline_metric)
        if isinstance(headline, (float, np.floating)):
            headline_worst = max(headline_worst, abs(float(headline)))
    return {"reference_repeats": len(traces),
            "reference_repeat_nrmse_pct": worst,
            "reference_repeat_headline_max": headline_worst,
            "reference_repeat_headline_metric": analysis.headline_metric}


def _unconverged_diagnostic(
    case: CircuitCase,
    analysis: AnalysisSpec,
    base_bt: BenchTech,
    corner: Corner,
    work_dir: Path,
    failure: Exception,
    *,
    reference_converged: bool,
    run_spec: RunSpec,
) -> Dict[str, Any]:
    """Recover the numbers behind a DC convergence failure, for reading only.

    A row that did not converge stays an ``error``: it never becomes a pass,
    never enters a numeric aggregate, and keeps its slot in the denominator.
    But "did not converge" and "was 200 mV out" are different facts about a
    model, and collapsing them is how a 0/10 AC score came to mean an
    unconverged DC operating point.  So the sweep is repeated once without the
    convergence requirement and its metrics are filed under a separate key
    that no scoring path reads.

    Only DC is recovered.  A transient already returns its committed prefix
    with ``partial=True``, and an AC response linearised about a state that is
    not a fixed point is not a measurement of anything.
    """
    if analysis.kind != "dc" or not reference_converged:
        return {}
    if "did not converge" not in str(failure):
        return {}
    bt = apply_corner(base_bt, corner)
    try:
        resolved = _resolved_analysis(case, analysis, bt, corner)
        baked = get_case_baked_modelcard(case, bt, work_dir)
        candidate_deck, reference_deck = render_case_decks(
            case,
            resolved,
            base_bt,
            corner,
            baked_lib=baked,
            model_level=run_spec.model_level,
        )
        reference = run_reference_trace(
            reference_deck, resolved, work_dir,
            f"{case.case_id}_{analysis.name}_unconv_ref",
        )
        candidate, _ = run_candidate_trace(
            candidate_deck, resolved, work_dir,
            f"{case.case_id}_{analysis.name}_unconv",
            require_convergence=False,
        )
        metrics, domain = compare_traces(
            candidate, reference, resolved, vdd=bt.vdd,
        )
        return {"unconverged_diagnostic": {**metrics, **domain}}
    except Exception:  # noqa: BLE001 — a diagnostic must never mask the failure
        return {}


def run_case_analysis(
    case: CircuitCase,
    analysis: AnalysisSpec,
    base_bt: BenchTech,
    corner: Corner,
    work_dir: Path,
    *,
    reference_repeats: int = 1,
    diagnose_support: bool = True,
    run_spec: Optional[RunSpec] = None,
    run_level72_control: bool = False,
) -> GateResult:
    """Run one complete paired experiment and return a structured result."""
    resolved_run_spec = run_spec or RunSpec.from_environment()
    provenance = resolved_run_spec.result_fields()
    bt = apply_corner(base_bt, corner)
    reference_converged = False
    candidate_converged = False
    partial = False
    control_converged: Optional[bool] = None
    control_domain: Dict[str, Any] = {}
    stage = "setup"
    try:
        resolved_analysis = _resolved_analysis(case, analysis, bt, corner)
        baked = get_case_baked_modelcard(case, bt, work_dir)
        candidate_deck, reference_deck = render_case_decks(
            case,
            resolved_analysis,
            base_bt,
            corner,
            baked_lib=baked,
            model_level=resolved_run_spec.model_level,
        )
        mismatch = physical_deck_mismatch(
            candidate_deck,
            reference_deck,
            resolved_analysis,
            bt,
            baked_lib=baked,
            case=case,
            model_level=resolved_run_spec.model_level,
        )
        if mismatch:
            raise ValueError(mismatch)
        support_signals = (_support_voltage_signals(candidate_deck)
                           if diagnose_support and analysis.kind != "ac" else ())
        stage = "reference"
        references = [
            run_reference_trace(
                reference_deck, resolved_analysis, work_dir,
                # Cards have technology placeholders; the deck renderer and
                # control runner must execute the same resolved limits.
                f"{case.case_id}_{analysis.name}_ref{index}",
                support_signals=support_signals,
            )
            for index in range(reference_repeats)
        ]
        reference_converged = True
        if run_level72_control:
            stage = "control"
            control_deck = render_case_control_deck(
                case,
                resolved_analysis,
                base_bt,
                corner,
                baked_lib=baked,
            )
            control_mismatch = physical_deck_mismatch(
                control_deck,
                reference_deck,
                resolved_analysis,
                bt,
                baked_lib=baked,
                case=case,
                model_level=72,
                control=True,
            )
            if control_mismatch:
                raise ValueError(f"LEVEL=72 control {control_mismatch}")
            control = run_level72_control_trace(
                control_deck,
                resolved_analysis,
                work_dir,
                f"{case.case_id}_{analysis.name}",
                modelcard=baked,
            )
            control_converged = True
            control_metrics, control_specific = compare_traces(
                control, references[0], resolved_analysis, vdd=bt.vdd,
            )
            control_nrmse = float(control_metrics["nrmse_pct"])
            if not np.isfinite(control_nrmse) or \
                    control_nrmse > case.control_nrmse_limit_pct:
                raise RuntimeError(
                    f"LEVEL=72 control NRMSE {control_nrmse:.6g}% exceeds "
                    f"{case.control_nrmse_limit_pct:g}%"
                )
            control_domain = {
                "level72_control": {**control_metrics, **control_specific}
            }
        stage = "candidate"
        candidate, candidate_path = run_candidate_trace(
            candidate_deck, resolved_analysis, work_dir,
            f"{case.case_id}_{analysis.name}",
        )
        candidate_converged = candidate.converged
        partial = candidate.partial
        metrics, domain = compare_traces(
            candidate, references[0], resolved_analysis, vdd=bt.vdd,
        )
        if candidate.partial:
            return GateResult(
                case_id=case.case_id,
                tech=bt.name,
                corner=corner.name,
                analysis=analysis.name,
                role=case.role,
                status="error",
                error=candidate.error or "candidate transient ended early",
                domain={"partial_diagnostic": {**metrics, **domain}},
                reference_converged=True,
                candidate_converged=False,
                control_converged=control_converged,
                partial=True,
                execution_state="partial",
                error_kind="candidate",
                **provenance,
            )
        stage = "metrics"
        validate_analysis_metrics(resolved_analysis, metrics, domain)
        domain.update(_reference_stability(
            references, resolved_analysis, bt.vdd))
        domain.update(control_domain)
        if diagnose_support and analysis.kind != "ac":
            domain["reference_support"] = support_diagnostic(
                candidate_path, references[0],
            )
        if case.role == DIAGNOSTIC:
            status = "diagnostic"
        else:
            status = ("pass" if _qualification_pass(
                case, metrics, domain, vdd=bt.vdd, partial=candidate.partial,
            ) else "fail")
        return GateResult(
            case_id=case.case_id, tech=bt.name, corner=corner.name,
            analysis=analysis.name, role=case.role, status=status,
            metrics=metrics, domain=domain,
            reference_converged=True,
            candidate_converged=candidate.converged,
            control_converged=control_converged,
            partial=candidate.partial,
            **provenance,
        )
    except Exception as exc:  # noqa: BLE001 - evidence rows retain all errors
        domain = _unconverged_diagnostic(
            case, analysis, base_bt, corner, work_dir, exc,
            reference_converged=reference_converged,
            run_spec=resolved_run_spec,
        )
        if stage in {"reference", "control"}:
            execution_state, error_kind = "reference_error", "reference"
        elif stage == "candidate":
            execution_state, error_kind = (
                ("nonconverged", "candidate")
                if "converg" in str(exc).lower()
                else ("error", "candidate")
            )
        elif stage == "metrics" or "mismatch" in str(exc).lower():
            execution_state, error_kind = "error", "result_schema"
        else:
            execution_state, error_kind = "infrastructure_error", "infrastructure"
        return GateResult(
            case_id=case.case_id, tech=bt.name, corner=corner.name,
            analysis=analysis.name, role=case.role, status="error",
            error=f"{type(exc).__name__}: {exc}",
            domain=domain,
            reference_converged=reference_converged,
            candidate_converged=candidate_converged,
            control_converged=False if stage == "control" else control_converged,
            partial=partial,
            execution_state=execution_state,
            error_kind=error_kind,
            **provenance,
        )


#: Derived quantity -> (numerator gain, denominator gain, reported stem).
#: A rejection ratio is a property of a *pair* of experiments, so it cannot be
#: produced by the per-analysis metric path and is formed here instead.
_DERIVED_RATIOS: Dict[str, Tuple[str, str]] = {
    "cmrr": ("diff_gain", "cm_gain"),
    "psrr": ("diff_gain", "supply_gain"),
}


def _gain(domains: Mapping[str, Mapping[str, Any]], key: str) -> float:
    for domain in domains.values():
        if key in domain:
            return float(domain[key])
    return float("nan")


def _derive_case_metrics(
    case: CircuitCase, results: Sequence[GateResult],
) -> List[GateResult]:
    """Form this case's cross-analysis quantities as their own result rows.

    A missing input is an ``error`` row rather than a silent omission: a
    rejection ratio that quietly disappears because one AC sweep failed would
    shrink the denominator exactly where the failure is most interesting.
    """
    if not case.derived_metrics:
        return []
    provenance = {
        "model_family": results[0].model_family,
        "model_level": results[0].model_level,
        "checkpoint_pins": results[0].checkpoint_pins,
        "campaign_manifest_sha256": results[0].campaign_manifest_sha256,
        "thread_settings": results[0].thread_settings,
    }
    domains = {result.analysis: result.domain for result in results
               if result.status != "error"}
    metrics: Dict[str, float] = {}
    domain: Dict[str, Any] = {}
    missing: List[str] = []
    for name in case.derived_metrics:
        numerator, denominator = _DERIVED_RATIOS[name]
        for role in ("test", "ref"):
            top = _gain(domains, f"{numerator}_{role}")
            bottom = _gain(domains, f"{denominator}_{role}")
            if not np.isfinite(top) or not np.isfinite(bottom)                     or top <= 0.0 or bottom <= 0.0:
                missing.append(f"{name}:{role}")
                domain[f"{name}_db_{role}"] = float("nan")
                continue
            domain[f"{name}_db_{role}"] = 20.0 * math.log10(top / bottom)
        error = abs(domain.get(f"{name}_db_test", float("nan"))
                    - domain.get(f"{name}_db_ref", float("nan")))
        domain[f"{name}_db_error"] = error
    if missing:
        return [GateResult(
            case_id=case.case_id, tech=results[0].tech,
            corner=results[0].corner, analysis="derived", role=case.role,
            status="error",
            error=f"derived metrics lack finite inputs: {sorted(set(missing))}",
            reference_converged=all(r.reference_converged for r in results),
            candidate_converged=all(r.candidate_converged for r in results),
            execution_state="error",
            error_kind="result_schema",
            **provenance,
        )]
    return [GateResult(
        case_id=case.case_id, tech=results[0].tech, corner=results[0].corner,
        analysis="derived", role=case.role, status="diagnostic",
        metrics=metrics, domain=domain,
        reference_converged=all(r.reference_converged for r in results),
        candidate_converged=all(r.candidate_converged for r in results),
        **provenance,
    )]


def run_case(
    case: CircuitCase,
    base_bt: BenchTech,
    corner: Corner,
    work_dir: Path,
    *,
    reference_repeats: int = 1,
    diagnose_support: bool = True,
    run_spec: Optional[RunSpec] = None,
    run_level72_control: bool = False,
) -> List[GateResult]:
    """Run every declared analysis and enforce the case-level result schema."""
    if reference_repeats < 1:
        raise ValueError("reference_repeats must be >= 1")
    resolved_run_spec = run_spec or RunSpec.from_environment()
    selected_analyses = applicable_analyses(case, base_bt, corner)
    if not selected_analyses:
        raise ValueError(
            f"{case.case_id}/{corner.name} is a no-op: no analysis observes "
            "a changed physical field"
        )
    results = [
        run_case_analysis(
            case, analysis, base_bt, corner,
            work_dir / analysis.name,
            reference_repeats=reference_repeats,
            diagnose_support=diagnose_support,
            run_spec=resolved_run_spec,
            run_level72_control=run_level72_control,
        )
        for analysis in selected_analyses
    ]
    results.extend(_derive_case_metrics(case, results))
    if not any(result.status == "error" for result in results):
        produced = {
            key
            for result in results
            for payload in (result.metrics, result.domain)
            for key in payload
        }
        missing = sorted(set(case.required_metrics) - produced)
        if missing:
            results.append(GateResult(
                case_id=case.case_id,
                tech=base_bt.name,
                corner=corner.name,
                analysis="result_schema",
                role=case.role,
                status="error",
                error=f"required result metrics were not emitted: {missing}",
                reference_converged=all(
                    result.reference_converged for result in results
                ),
                candidate_converged=all(
                    result.candidate_converged for result in results
                ),
                partial=any(result.partial for result in results),
                execution_state="error",
                error_kind="result_schema",
                **resolved_run_spec.result_fields(),
            ))
    return results
