"""The hierarchy summary must retain the metrics its shared comparator promises."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.common.circuit_benchmarks import BENCH
from tests.common.simple_circuit_harness import RunSpec, Trace
from tests.simple_circuits import verify_nn_subckt as subckt


@pytest.mark.parametrize(("kind", "phase"), (("dc", 0.0), ("tran", 0.0), ("ac", 0.0), ("ac", 30.0)))
def test_hierarchy_summary_retains_complete_comparison_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, phase: float,
) -> None:
    """Equal flat/nested responses remain characterizable, including AC phase."""
    analysis = next(a for a in subckt.SUBCKT_ANALYSES if a.kind == kind)
    if kind == "ac":
        axis = np.logspace(3, 11, 81)
        response = 1.0 / (1.0 + 1j * axis / 1e7)
        axis_name = "frequency"
    else:
        axis = np.linspace(0.0, 0.8 if kind == "dc" else 4e-9, 161)
        response = np.linspace(0.1, 0.7, axis.size)
        axis_name = "sweep" if kind == "dc" else "time"
    reference = Trace(axis_name, axis, {"v(out)": response, "v(mid)": response * 0.5}, reference=True)
    rotation = np.exp(1j * np.deg2rad(phase)) if kind == "ac" else 1.0

    monkeypatch.setattr(subckt, "get_baked_modelcard", lambda *_a, **_k: tmp_path / "baked.lib")
    monkeypatch.setattr(subckt, "parse_netlist", lambda *_a: object())
    monkeypatch.setattr(subckt, "flattened_candidate_mismatch", lambda *_a: "")
    monkeypatch.setattr(subckt, "run_reference_trace", lambda *_a, **_k: reference)

    def candidate(
        deck: str, spec: object, path: Path, tag: str, **kwargs: object,
    ) -> tuple[Trace, Path]:
        internal = "v(Xbuf.m)" if tag.startswith("hierarchical") else "v(mid)"
        return Trace(axis_name, axis, {"v(out)": response * rotation,
                                       internal: response * rotation * 0.5}), path / "candidate.sp"

    monkeypatch.setattr(subckt, "run_candidate_trace", candidate)
    row = subckt.run_nn_subckt_analysis(BENCH["TSMC12"], analysis, tmp_path, RunSpec(75, "DirectNet-Full"))
    assert row.status == "diagnostic", row.error
    assert row.reference_converged and row.candidate_converged
    assert row.metrics["nrmse_pct"] == pytest.approx(0.0, abs=1e-10)
    assert row.domain["flat_hierarchical_max_error_v"] == pytest.approx(0.0)
    if kind == "ac":
        assert row.metrics["phase_maxerr_deg"] == pytest.approx(phase, abs=1e-10)
    else:
        assert "phase_maxerr_deg" not in row.metrics
