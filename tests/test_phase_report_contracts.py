"""AC phase errors remain visible from measured traces through the report."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.v710_regate_collect import collect, render
from tests.common.gate_result import GateResult
from tests.common.simple_circuit_catalog import get_case
from tests.common.simple_circuit_harness import Trace, compare_traces


@pytest.mark.parametrize("phase_degrees", [0.0, 30.0, 180.0])
def test_terminal_capacitance_reports_phase_from_all_admittance_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase_degrees: float,
) -> None:
    """Completed terminal AC solves must survive the shared metric contract.

    A rotation of one admittance entry must be visible even when that entry
    contributes little to the capacitance-matrix aggregate error.
    """
    from tests.common import terminal_integrity as terminal
    from tests.common.circuit_benchmarks import BENCH
    from tests.common.simple_circuit_harness import RunSpec

    reference = 1e-6 * (1.0 + 1j) * (4.0 * np.eye(4) - np.ones((4, 4)))
    candidate = reference.copy()
    candidate[3, 2] *= np.exp(1j * np.deg2rad(phase_degrees))
    reference_columns = iter(reference.T)
    candidate_columns = iter(candidate.T)
    monkeypatch.setattr(terminal, "get_baked_modelcard", lambda *a, **kw: tmp_path / "model.lib")
    monkeypatch.setattr(terminal, "_reference_ac_currents", lambda *a, **kw: next(reference_columns))
    monkeypatch.setattr(terminal, "_candidate_ac_currents", lambda *a, **kw: next(candidate_columns))
    result = terminal.run_terminal_capacitance_bias(
        BENCH["TSMC12"], "nmos", terminal.TerminalBias("known", 0.4, 0.7, 0.0, 0.0),
        tmp_path, RunSpec(75, "DirectNet-Full"),
    )
    assert result.status == "diagnostic", result.error
    assert result.reference_converged and result.candidate_converged
    assert result.metrics["phase_maxerr_deg"] == pytest.approx(phase_degrees)
    expected_c = candidate.imag / (2 * np.pi * terminal.AC_FREQUENCY_HZ)
    np.testing.assert_allclose(result.domain["candidate_capacitance_f"], expected_c)


@pytest.mark.parametrize("legacy_row", [False, True], ids=["current", "legacy"])
def test_magnitude_identical_phase_error_reaches_human_report(
    tmp_path: Path, legacy_row: bool,
) -> None:
    """A reader must see the 30-degree error even with zero magnitude error.

    The topology screen has no frozen phase threshold, so its result stays
    diagnostic. Older rows have only per-signal errors; collection must
    derive and display the same aggregate from that recorded evidence.
    """
    case = get_case("common_source_nn")
    frequency = np.logspace(3, 9, 121)
    lowpass = 10.0 / (1.0 + 1j * frequency / 1e6)
    rotation = np.exp(1j * np.deg2rad(30.0))
    rows: list[GateResult] = []
    for analysis in case.analyses:
        reference = Trace("frequency", frequency, {
            signal: lowpass * (0.01 if signal == "v(vb)" else 1.0)
            for signal in analysis.signals
        }, reference=True)
        candidate = Trace("frequency", frequency, {
            signal: values * rotation
            for signal, values in reference.signals.items()
        })
        metrics, domain = compare_traces(candidate, reference, analysis, vdd=0.8)
        assert metrics["nrmse_pct"] == pytest.approx(0.0, abs=1e-10)
        if legacy_row:
            del metrics["phase_maxerr_deg"]
        rows.append(GateResult(
            case_id=case.case_id, tech="TSMC12", corner="nominal",
            analysis=analysis.name, role="diagnostic", status="diagnostic",
            metrics=metrics, domain=domain,
            model_family="DirectNet-Full", model_level=75,
            checkpoint_pins={
                "nmos": "tsmc12_dnf_large_nmos",
                "pmos": "tsmc12_dnf_large_pmos",
            },
            thread_settings={"omp": 1, "mkl": 1, "torch": 1},
        ))
    log = (tmp_path / "dnf" / "large" / "tsmc12"
           / f"{case.campaign_suite}.omp1.log")
    log.parent.mkdir(parents=True)
    log.write_text("\n".join(row.marker() for row in rows)
                   + "\n===V710_DONE rc=0===\n")

    data = collect(tmp_path)
    entry = data["dnf"]["large"][case.campaign_suite]["TSMC12"]["omp1"]
    assert entry["result_complete"] is True
    assert entry["status"] == "DIAGNOSTIC"
    report = render(data)
    for analysis in case.analyses:
        assert f"{analysis.name}:phase_maxerr_deg=30" in report
