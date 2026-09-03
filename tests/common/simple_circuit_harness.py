"""Deep harness module for paired NN/LEVEL=72 simple-circuit experiments.

Callers choose a catalog case, technology, and corner.  This module hides deck
rendering, topology parity, both simulator adapters, trace alignment, domain
metrics, and accepted-reference support diagnostics behind that small seam.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from tests.common.base import (
    OSDI_PATH, deck_tokens, template_deck, render_deck_text,
    run_ngspice_subprocess,
)
from tests.common.circuit_benchmarks import (
    BenchTech, active_model_label, bench_variant, full_metrics,
    get_baked_modelcard, parse_netlist, run_directnet_dc_sweep,
    run_directnet_transient, run_ngspice_wrdata,
)
from tests.common.gate_result import GateResult
from tests.common.simple_circuit_catalog import (
    AnalysisSpec, CircuitCase, DIAGNOSTIC,
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
}


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

    def validate(self) -> None:
        """Fail loud on empty, ragged, or non-finite numerical evidence."""
        self.axis = np.asarray(self.axis, dtype=float)
        if self.axis.ndim != 1 or self.axis.size == 0:
            raise ValueError("trace axis must be a non-empty vector")
        if not np.all(np.isfinite(self.axis)):
            raise ValueError("trace axis contains NaN/Inf")
        for name, values in self.signals.items():
            array = np.asarray(values)
            if array.ndim != 1 or array.size != self.axis.size:
                raise ValueError(
                    f"trace {name} has {array.size} values for "
                    f"{self.axis.size} axis points")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"trace {name} contains NaN/Inf")
            self.signals[name] = array


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
        values["l_nmos"] = corner.l_nmos
    if corner.l_pmos is not None:
        values["l_pmos"] = corner.l_pmos
    return bench_variant(bt, **values)


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


def render_ring_stages(
    *,
    n_stages: int,
    p_prefix: str,
    n_prefix: str,
    p_device: str,
    n_device: str,
    cload: float,
    vdd: float,
) -> Tuple[str, str]:
    """Build the variable odd-stage block used by the canonical ring template."""
    if n_stages < 3 or n_stages % 2 == 0:
        raise ValueError("ring oscillator needs an odd stage count >= 3")
    lines: List[str] = []
    for index in range(1, n_stages + 1):
        inp = f"n{index - 1}" if index > 1 else f"n{n_stages}"
        out = f"n{index}"
        lines.extend((
            f"{p_prefix}p{index} {out} {inp} vdd vdd {p_device}",
            f"{n_prefix}n{index} {out} {inp} 0 0 {n_device}",
        ))
        lines.append(f"Cl{index} {out} 0 {_spice_cap(cload)}")
    ic = " ".join(
        f"V(n{index})={'0' if index % 2 else _number(vdd)}"
        for index in range(1, n_stages + 1)
    )
    return "\n".join(lines), ic


def _common_substitutions(
    bt: BenchTech,
    corner: Corner,
    *,
    reference: bool,
    baked_lib: Path,
    ring_n_stages: int = 5,
    ring_cload: float = 0.5e-15,
) -> Dict[str, str]:
    vdd = bt.vdd
    reverse = corner.body_reverse_frac * vdd
    level = {"DirectNet": 73, "BSIM-AR": 74,
             "DirectNet-Full": 75, "BSIM-AR-Full": 76}.get(
                 active_model_label().split(" (")[0], 73)
    vcm = 0.55 * vdd
    if reference:
        model_setup = f'.include "{baked_lib}"'
        n_prefix = p_prefix = "N"
        n_device = bt.nmos_model
        p_device = bt.pmos_model
    else:
        model_setup = (
            f".model nmos_nn NMOS (LEVEL={level} TECH={bt.nn_tech} "
            f"VT={bt.effective_nmos_vt})\n"
            f".model pmos_nn PMOS (LEVEL={level} TECH={bt.nn_tech} "
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
    ring_stages, ring_ic = render_ring_stages(
        n_stages=ring_n_stages,
        p_prefix=p_prefix,
        n_prefix=n_prefix,
        p_device=p_device,
        n_device=n_device,
        cload=ring_cload,
        vdd=vdd,
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
        "OUTPUT_LOAD": "",
        "AC_INP": "1",
        "AC_INN": "0",
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
        "RING_STAGES": ring_stages,
        "RING_IC": ring_ic,
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


def _render_one(
    case: CircuitCase,
    analysis: AnalysisSpec,
    bt: BenchTech,
    corner: Corner,
    *,
    reference: bool,
    baked_lib: Path,
    substitutions: Optional[Mapping[str, str]] = None,
    ring_n_stages: int = 5,
    ring_cload: float = 0.5e-15,
) -> str:
    relative = case.template
    path = template_deck(relative, tier=case.tier)
    template = path.read_text()
    available = _common_substitutions(
        bt, corner, reference=reference, baked_lib=baked_lib,
        ring_n_stages=ring_n_stages, ring_cload=ring_cload,
    )
    overrides = dict(substitutions or {})
    available.update(overrides)
    for name, default in _SPEC_DEFAULTS.items():
        if name not in overrides:
            available[name] = _expand(default, available)
    for name, raw in analysis.substitutions().items():
        available[name] = _expand(raw, available)
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
    for values in (candidate_values, reference_values):
        for name, default in _SPEC_DEFAULTS.items():
            values[name] = _expand(default, values)
    for name, raw in analysis.substitutions().items():
        candidate_values[name] = _expand(raw, candidate_values)
        reference_values[name] = _expand(raw, reference_values)
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
) -> Tuple[str, str]:
    """Render the candidate and LEVEL=72 decks for one identical experiment."""
    bt = apply_corner(base_bt, corner)
    candidate = _render_one(
        case, analysis, bt, corner, reference=False, baked_lib=baked_lib,
        substitutions=substitutions or {}, ring_n_stages=ring_n_stages,
        ring_cload=ring_cload,
    )
    reference = _render_one(
        case, analysis, bt, corner, reference=True, baked_lib=baked_lib,
        substitutions=substitutions or {}, ring_n_stages=ring_n_stages,
        ring_cload=ring_cload,
    )
    return candidate, reference


def _source_kind(parts: Sequence[str]) -> str:
    tail = " ".join(parts[3:]).lower()
    if "pulse" in tail:
        return "pulse"
    if "ac" in tail:
        return "ac"
    return "dc"


def topology_signature(text: str) -> Counter[Tuple[str, ...]]:
    """Canonical connectivity multiset for a flat rendered deck.

    Values, model names, and engine-specific M/N device prefixes are ignored;
    element kind, MOS polarity, terminal order, source kind, and IC nodes are
    retained.  An unresolved subcircuit instance is rejected because silently
    ignoring it would turn the parity check into a false assurance.
    """
    signature: Counter[Tuple[str, ...]] = Counter()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("*"):
            continue
        parts = line.split()
        head = parts[0]
        low = head.lower()
        if low.startswith((".include", ".model", ".temp", ".end",
                           ".dc", ".tran", ".ac", ".options")):
            continue
        if low == ".ic":
            nodes = tuple(sorted(
                match.lower() for match in
                re.findall(r"v\(([^)]+)\)", line, flags=re.IGNORECASE)
            ))
            signature[("ic", *nodes)] += 1
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
                       parts[2].lower())] += 1
        elif prefix in ("V", "I"):
            signature[(prefix.lower(), parts[1].lower(), parts[2].lower(),
                       _source_kind(parts))] += 1
        else:
            raise ValueError(f"unsupported topology card in parity check: {line}")
    return signature


def topology_mismatch(candidate: str, reference: str) -> str:
    """Return a readable candidate/reference connectivity difference."""
    candidate_sig = topology_signature(candidate)
    reference_sig = topology_signature(reference)
    if candidate_sig == reference_sig:
        return ""
    candidate_only = list((candidate_sig - reference_sig).elements())
    reference_only = list((reference_sig - candidate_sig).elements())
    return (f"topology mismatch: candidate-only={candidate_only}; "
            f"reference-only={reference_only}")


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
        return _run_ngspice_ac_trace(deck, signals, card, work_dir, tag)
    data = run_ngspice_wrdata(
        _body_only(deck), " ".join(signals), work_dir, tag, card,
    )
    return _parse_real_wrdata(
        data, signals, axis_name="time" if analysis.kind == "tran" else "sweep",
    )


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
    from pycircuitsim.simulation import _circuit_has_nn, _solve_dc_with_retry
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
    start = float(parser.analysis_params["fstart"])
    stop = float(parser.analysis_params["fstop"])
    points = int(parser.analysis_params["num_points"])
    sweep_type = str(parser.analysis_params["sweep_type"])
    if sweep_type == "dec":
        count = int(round(math.log10(stop / start) * points)) + 1
        frequencies = np.logspace(math.log10(start), math.log10(stop), count)
    elif sweep_type == "oct":
        count = int(round(math.log2(stop / start) * points)) + 1
        frequencies = np.logspace(math.log10(start), math.log10(stop), count)
    else:
        frequencies = np.linspace(start, stop, points)
    raw = ACSolver(circuit, dc_solution=dc_solution).solve(frequencies)
    values = {signal: _lookup_signal(raw, signal) for signal in signals}
    trace = Trace("frequency", frequencies, values, converged=converged)
    trace.validate()
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
        else:
            raise ValueError(f"unsupported analysis kind {analysis.kind!r}")
    finally:
        logging.disable(logging.NOTSET)
    trace.validate()
    return trace, path


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


def _relative_error(test: float, reference: float) -> float:
    if not np.isfinite(test) or not np.isfinite(reference) or abs(reference) < 1e-30:
        return float("nan")
    return abs(test - reference) / abs(reference) * 100.0


def _gradient_gain(axis: np.ndarray, values: np.ndarray) -> float:
    return float(np.max(np.abs(np.gradient(values, axis))))


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


#: Every metric profile ``_domain_metrics`` implements.  A case naming a
#: profile that is not here emits no domain metrics at all, which shows up
#: only as a missing required metric on a run where nothing errored — far too
#: late.  The catalog contract check verifies membership instead.
METRIC_PROFILES: frozenset = frozenset({
    "trace", "transient", "ac",
    "source_follower", "gain", "opamp", "ring_osc", "current_mirror",
    "cascode", "cascode_ac", "inverter_chain", "logic_tran", "logic_vtc",
    "transmission_gate", "hold_droop", "switchcap", "diffpair",
    "diffpair_diff_ac", "diffpair_cm_ac", "sram_hold", "sram_read",
    "sram_write", "sram_snm",
    # V7.6.7 additions
    "mirror_iref", "opamp_diff_ac", "opamp_cm_ac", "opamp_supply_ac",
    "switchcap_multicycle", "ring_supply", "sram_write_margin",
    "diode_load", "self_bias_cell", "self_bias_cascode", "mos_reference",
    "unity_gain", "settling", "line_regulation", "load_regulation",
})


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
        half = grid.size // 2
        test_tail = np.real(candidate[name][half:])
        ref_tail = np.real(reference[name][half:])
        droop_t = float(np.ptp(test_tail))
        droop_r = float(np.ptp(ref_tail))
        domain.update(
            droop_test_v=droop_t, droop_ref_v=droop_r,
            droop_error_v=abs(droop_t - droop_r),
            feedthrough_error_v=abs(float(np.max(np.abs(np.diff(test_tail))))
                                    - float(np.max(np.abs(np.diff(ref_tail))))),
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
        test = np.real(candidate[name])
        ref = np.real(reference[name])
        # Compare drift accumulated after the first quarter of the window, so
        # a constant offset in the first sample does not masquerade as drift
        # and vice versa.
        start = max(grid.size // 4, 1)
        drift_t = float(test[-1] - test[start])
        drift_r = float(ref[-1] - ref[start])
        domain.update(
            cycle_drift_test_v=drift_t, cycle_drift_ref_v=drift_r,
            cycle_drift_error_v=abs(drift_t - drift_r),
            final_sample_error_v=abs(float(test[-1] - ref[-1])),
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
    for trace in traces[1:]:
        metrics, _ = compare_traces(trace, traces[0], analysis, vdd=vdd)
        worst = max(worst, float(metrics["nrmse_pct"]))
    return {"reference_repeats": len(traces),
            "reference_repeat_nrmse_pct": worst}


def _unconverged_diagnostic(
    case: CircuitCase,
    analysis: AnalysisSpec,
    base_bt: BenchTech,
    corner: Corner,
    work_dir: Path,
    failure: Exception,
    *,
    reference_converged: bool,
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
        resolved = _resolved_analysis(analysis, bt, corner)
        baked = get_baked_modelcard(
            bt, bt.nfin, work_dir, nfin_p=bt.effective_nfin_p,
        )
        candidate_deck, reference_deck = render_case_decks(
            case, resolved, base_bt, corner, baked_lib=baked,
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
) -> GateResult:
    """Run one complete paired experiment and return a structured result."""
    bt = apply_corner(base_bt, corner)
    reference_converged = False
    candidate_converged = False
    partial = False
    try:
        resolved_analysis = _resolved_analysis(analysis, bt, corner)
        baked = get_baked_modelcard(
            bt, bt.nfin, work_dir, nfin_p=bt.effective_nfin_p,
        )
        candidate_deck, reference_deck = render_case_decks(
            case, resolved_analysis, base_bt, corner, baked_lib=baked,
        )
        mismatch = topology_mismatch(candidate_deck, reference_deck)
        if mismatch:
            raise ValueError(mismatch)
        support_signals = (_support_voltage_signals(candidate_deck)
                           if diagnose_support and analysis.kind != "ac" else ())
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
        candidate, candidate_path = run_candidate_trace(
            candidate_deck, resolved_analysis, work_dir,
            f"{case.case_id}_{analysis.name}",
        )
        candidate_converged = candidate.converged
        partial = candidate.partial
        metrics, domain = compare_traces(
            candidate, references[0], resolved_analysis, vdd=bt.vdd,
        )
        domain.update(_reference_stability(
            references, resolved_analysis, bt.vdd))
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
            partial=candidate.partial,
        )
    except Exception as exc:  # noqa: BLE001 - evidence rows retain all errors
        domain = _unconverged_diagnostic(
            case, analysis, base_bt, corner, work_dir, exc,
            reference_converged=reference_converged,
        )
        return GateResult(
            case_id=case.case_id, tech=bt.name, corner=corner.name,
            analysis=analysis.name, role=case.role, status="error",
            error=f"{type(exc).__name__}: {exc}",
            domain=domain,
            reference_converged=reference_converged,
            candidate_converged=candidate_converged,
            partial=partial,
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
        )]
    return [GateResult(
        case_id=case.case_id, tech=results[0].tech, corner=results[0].corner,
        analysis="derived", role=case.role, status="diagnostic",
        metrics=metrics, domain=domain,
        reference_converged=all(r.reference_converged for r in results),
        candidate_converged=all(r.candidate_converged for r in results),
    )]


def run_case(
    case: CircuitCase,
    base_bt: BenchTech,
    corner: Corner,
    work_dir: Path,
    *,
    reference_repeats: int = 1,
    diagnose_support: bool = True,
) -> List[GateResult]:
    """Run every declared analysis and enforce the case-level result schema."""
    if reference_repeats < 1:
        raise ValueError("reference_repeats must be >= 1")
    results = [
        run_case_analysis(
            case, analysis, base_bt, corner,
            work_dir / analysis.name,
            reference_repeats=reference_repeats,
            diagnose_support=diagnose_support,
        )
        for analysis in case.analyses
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
            ))
    return results
