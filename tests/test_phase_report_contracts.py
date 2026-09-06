"""AC phase errors remain visible from measured traces through the report."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.v710_regate_collect import collect, render
from tests.common.gate_result import GateResult
from tests.common.simple_circuit_catalog import get_case
from tests.common.simple_circuit_harness import Trace, compare_traces


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
