"""Behavioral contracts for every catalog metric profile.

Deck parity proves that both simulators receive the same experiment; it does
not prove that the harness extracts the intended quantity from their traces.
These tests keep that second oracle hermetic by feeding known, physically
shaped traces through the same ``compare_traces`` interface used by campaigns.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.common.simple_circuit_catalog import AnalysisSpec, cases
from tests.common.simple_circuit_harness import (
    METRIC_PROFILES,
    Trace,
    compare_traces,
    validate_analysis_metrics,
)

VDD = 0.8


def _catalog_analyses_by_profile() -> dict[str, AnalysisSpec]:
    """Return one real catalog analysis for every metric profile."""
    result: dict[str, AnalysisSpec] = {}
    for case in cases():
        for analysis in case.analyses:
            result.setdefault(analysis.metric_profile, analysis)
    return result


CATALOG_ANALYSES_BY_PROFILE = _catalog_analyses_by_profile()
CATALOG_ANALYSES = tuple(
    (case.case_id, analysis)
    for case in cases()
    for analysis in case.analyses
)


def _axis(kind: str) -> np.ndarray:
    if kind == "op":
        return np.asarray([0.0])
    if kind == "ac":
        return np.logspace(3.0, 9.0, 121)
    if kind == "tran":
        return np.linspace(0.0, 8e-9, 801)
    return np.linspace(0.0, VDD, 161)


def _lowpass(
    frequency: np.ndarray,
    gain: float = 10.0,
    corner_hz: float = 1e6,
) -> np.ndarray:
    return gain / (1.0 + 1j * frequency / corner_hz)


def _ac_signals(
    analysis: AnalysisSpec,
    frequency: np.ndarray,
) -> dict[str, np.ndarray]:
    profile = analysis.metric_profile
    names = analysis.signals
    base = _lowpass(frequency)
    if profile == "diffpair_diff_ac":
        return {names[0]: -0.5 * base, names[1]: 0.5 * base}
    if profile == "diffpair_cm_ac":
        return {names[0]: 0.1 * base, names[1]: 0.1 * base}
    if profile == "cascode_ac":
        return {
            names[0]: _lowpass(frequency, 20.0),
            names[1]: _lowpass(frequency, 10.0),
        }
    if profile == "common_source_floating_ac":
        return {names[0]: base, names[1]: 0.01 * base}
    return {name: base * (1.0 - 0.1 * index)
            for index, name in enumerate(names)}


def _op_signals(analysis: AnalysisSpec) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for index, name in enumerate(analysis.signals):
        value = 5e-6 if name.lower().startswith("i(") else 0.25 + 0.1 * index
        values[name] = np.asarray([value])
    return values


def _dc_signals(
    analysis: AnalysisSpec,
    sweep: np.ndarray,
) -> dict[str, np.ndarray]:
    profile = analysis.metric_profile
    names = analysis.signals
    rising = 0.1 + 0.6 * sweep / VDD
    falling = VDD - sweep
    current = 1e-6 + 2e-6 * sweep / VDD

    if profile in {"inverter_vtc", "logic_vtc", "sram_write_margin"}:
        result = {names[0]: falling}
        for name in names[1:]:
            result[name] = (
                current if name.lower().startswith("i(") else 0.9 * falling
            )
        return result
    if profile == "sram_snm":
        return {names[0]: 0.05 + 0.7 * sweep / VDD}
    if profile in {"current_mirror", "cascode", "self_bias_cascode"}:
        return {
            name: current if name.lower().startswith("i(")
            else 0.2 + 0.1 * sweep / VDD
            for name in names
        }
    if profile == "mirror_iref":
        return {names[0]: 2.0 * sweep + 1e-9}
    if profile == "transmission_gate":
        output = 0.9 * sweep
        return {names[0]: output, names[1]: (sweep - output) / 1e3 + 1e-12}
    if profile == "self_bias_cell":
        return {
            name: 1e-9 + 5e-6 * (sweep / VDD) ** 2
            if name.lower().startswith("i(") else 0.1 + 0.3 * sweep / VDD
            for name in names
        }
    if profile == "unity_gain":
        return {names[0]: 0.95 * sweep + 0.01, names[1]: sweep}
    return {
        name: current if name.lower().startswith("i(")
        else rising * (1.0 - 0.05 * index)
        for index, name in enumerate(names)
    }


def _clock(time: np.ndarray) -> np.ndarray:
    return np.where(np.mod(time, 2e-9) < 1e-9, VDD, 0.0)


def _transition(
    time: np.ndarray,
    center: float,
    *,
    falling: bool = False,
) -> np.ndarray:
    rising = VDD / (1.0 + np.exp(-(time - center) / 0.15e-9))
    return VDD - rising if falling else rising


def _tran_signals(
    analysis: AnalysisSpec,
    time: np.ndarray,
) -> dict[str, np.ndarray]:
    profile = analysis.metric_profile
    names = analysis.signals
    if profile in {"ring_osc", "ring_supply"}:
        voltage = VDD / 2.0 + 0.35 * VDD * np.sin(2.0 * np.pi * time / 2e-9)
        result = {names[0]: voltage}
        for name in names[1:]:
            result[name] = 5e-6 * (1.1 + 0.1 * np.sin(2.0 * np.pi * time / 2e-9))
        return result
    if profile == "inverter_energy":
        return {
            names[0]: _transition(time, 2.0e-9),
            names[1]: _transition(time, 2.5e-9, falling=True),
            names[2]: 1e-7 + 5e-6 * np.exp(-((time - 2.3e-9) / 0.4e-9) ** 2),
        }
    if profile in {"inverter_chain", "logic_tran"}:
        result = {
            names[0]: _transition(time, 2.0e-9),
            names[1]: _transition(time, 2.5e-9, falling=True),
        }
        for name in names[2:]:
            result[name] = _transition(time, 3.0e-9)
        return result
    if profile in {"hold_droop", "switchcap", "switchcap_multicycle"}:
        result = {names[0]: 0.55 + 0.03 * time / time[-1]}
        if len(names) > 1:
            result[names[1]] = _clock(time)
        return result
    if profile == "sram_hold":
        drift = 0.005 * time / time[-1]
        return {names[0]: 0.7 + drift, names[1]: 0.1 - drift}
    if profile == "sram_read":
        disturb = 0.05 * np.exp(-((time - 3e-9) / 0.6e-9) ** 2)
        return {names[0]: 0.7 - disturb, names[1]: 0.1 + disturb}
    if profile == "sram_write":
        return {names[0]: _transition(time, 3e-9, falling=True),
                names[1]: _transition(time, 3e-9)}
    if profile == "settling":
        output = 0.2 + 0.4 * (1.0 - np.exp(-time / 1e-9))
        return {names[0]: output, names[1]: np.full_like(time, 0.6)}
    if profile == "load_regulation":
        after = np.maximum(time - 2e-9, 0.0)
        output = 0.7 - np.where(time >= 2e-9, 0.08 * np.exp(-after / 2e-9), 0.0)
        return {names[0]: output}
    return {name: _transition(time, 2e-9 + index * 0.3e-9)
            for index, name in enumerate(names)}


def _reference_trace(analysis: AnalysisSpec) -> Trace:
    axis = _axis(analysis.kind)
    if analysis.kind == "ac":
        signals = _ac_signals(analysis, axis)
        axis_name = "frequency"
    elif analysis.kind == "op":
        signals = _op_signals(analysis)
        axis_name = "point"
    elif analysis.kind == "tran":
        signals = _tran_signals(analysis, axis)
        axis_name = "time"
    else:
        signals = _dc_signals(analysis, axis)
        axis_name = "sweep"
    return Trace(axis_name, axis, signals, reference=True)


def _mutated_trace(analysis: AnalysisSpec, reference: Trace) -> Trace:
    """Change the behavior named by one profile's headline metric."""
    profile = analysis.metric_profile
    names = analysis.signals
    axis = reference.axis
    signals = {name: values.copy() for name, values in reference.signals.items()}

    if profile in {"common_source_ac", "active_load_ac", "closed_loop_ac"}:
        signals[names[0]] = _lowpass(axis, corner_hz=3e6)
    elif profile == "common_source_floating_ac":
        signals[names[1]] = 2.0 * signals[names[1]]
    elif profile in {"ring_osc", "ring_supply"}:
        signals[names[0]] = (
            VDD / 2.0
            + 0.35 * VDD * np.sin(2.0 * np.pi * axis / 1.6e-9)
        )
    elif profile == "inverter_energy":
        signals[names[2]] *= 1.2
    elif profile in {"inverter_chain", "logic_tran"}:
        signals[names[1]] = _transition(axis, 3.0e-9, falling=True)
    elif profile in {"inverter_vtc", "logic_vtc", "sram_write_margin"}:
        signals[names[0]] = np.clip(signals[names[0]] + 0.05, 0.0, VDD)
    elif profile == "transmission_gate":
        signals[names[0]] = 0.85 * axis
    elif profile in {"hold_droop", "switchcap", "switchcap_multicycle"}:
        signals[names[0]] = 0.55 + 0.06 * axis / axis[-1]
    elif profile == "self_bias_cell":
        current = next(name for name in names if name.lower().startswith("i("))
        signals[current] *= 1.2
    elif profile == "settling":
        signals[names[0]] = 0.2 + 0.4 * (1.0 - np.exp(-axis / 2e-9))
    elif profile == "load_regulation":
        after = np.maximum(axis - 2e-9, 0.0)
        signals[names[0]] = 0.7 - np.where(
            axis >= 2e-9,
            0.16 * np.exp(-after / 2e-9),
            0.0,
        )
    elif profile == "sram_read":
        disturb = 0.10 * np.exp(-((axis - 3e-9) / 0.6e-9) ** 2)
        signals[names[0]] = 0.7 - disturb
    elif profile == "sram_write":
        signals[names[0]] = _transition(axis, 3.5e-9, falling=True)
    elif profile == "sram_hold":
        signals[names[0]] += 0.05
    else:
        signals[names[0]] *= 1.2

    return Trace(reference.axis_name, axis.copy(), signals)


def test_metric_profile_registry_contains_only_catalog_profiles() -> None:
    """A retired profile must not remain as an apparently supported contract."""
    assert METRIC_PROFILES == frozenset(CATALOG_ANALYSES_BY_PROFILE)


@pytest.mark.parametrize(
    ("case_id", "analysis"),
    CATALOG_ANALYSES,
    ids=[f"{case_id}-{analysis.name}" for case_id, analysis in CATALOG_ANALYSES],
)
def test_every_catalog_analysis_accepts_an_identical_known_trace(
    case_id: str,
    analysis: AnalysisSpec,
) -> None:
    """Every promised event metric must be reachable and finite."""
    del case_id
    reference = _reference_trace(analysis)
    candidate = Trace(
        reference.axis_name,
        reference.axis.copy(),
        {name: values.copy() for name, values in reference.signals.items()},
    )

    metrics, domain = compare_traces(candidate, reference, analysis, vdd=VDD)

    validate_analysis_metrics(analysis, metrics, domain)
    headline = {**metrics, **domain}[analysis.headline_metric]
    assert float(headline) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("case_id", "analysis"),
    CATALOG_ANALYSES,
    ids=[f"{case_id}-{analysis.name}" for case_id, analysis in CATALOG_ANALYSES],
)
def test_every_catalog_analysis_detects_its_headline_behavior(
    case_id: str,
    analysis: AnalysisSpec,
) -> None:
    """A profile must react when the physical behavior it names is changed."""
    del case_id
    reference = _reference_trace(analysis)
    candidate = _mutated_trace(analysis, reference)

    metrics, domain = compare_traces(candidate, reference, analysis, vdd=VDD)

    validate_analysis_metrics(analysis, metrics, domain)
    headline = {**metrics, **domain}[analysis.headline_metric]
    assert float(headline) > 1e-12


def test_pmos_self_biased_cascode_scores_the_low_voltage_compliance_region() -> None:
    """PMOS output resistance is set near ground, opposite to the NMOS case."""
    sweep = np.linspace(0.0, VDD, 161)
    low = sweep <= VDD / 2.0
    reference_current = np.where(
        low,
        1e-6 + 1e-7 * sweep,
        1e-6 + 1e-7 * VDD / 2.0 + 1e-5 * (sweep - VDD / 2.0),
    )
    candidate_current = np.where(
        low,
        1e-6 + 2e-7 * sweep,
        1e-6 + 2e-7 * VDD / 2.0 + 1e-5 * (sweep - VDD / 2.0),
    )
    analysis = AnalysisSpec(
        "pmos_compliance",
        "dc",
        "dc Voutp 0 0.8 0.005",
        ("i(load)", "v(px)"),
        metric_profile="self_bias_cascode",
        device_kinds=("pmos",),
    )
    reference = Trace(
        "sweep",
        sweep,
        {"i(load)": reference_current, "v(px)": 0.6 - 0.1 * sweep},
        reference=True,
    )
    candidate = Trace(
        "sweep",
        sweep,
        {"i(load)": candidate_current, "v(px)": 0.6 - 0.1 * sweep},
    )

    _metrics, domain = compare_traces(candidate, reference, analysis, vdd=VDD)

    assert domain["output_resistance_error_pct"] == pytest.approx(50.0)
