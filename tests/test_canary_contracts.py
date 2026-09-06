"""Fail-closed execution and polarity contracts for the source-frame canary."""
from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from tests.common import base
from tests.common.gate_result import parse_result_markers
from tests.common.simple_circuit_harness import RunSpec
from tests.single_devices import verify_nn_lifted_source_dc as canary


@pytest.fixture
def work_dir() -> Iterator[Path]:
    root = base.PROJECT_ROOT / "results" / "tests"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="canary_contract_", dir=root) as path:
        yield Path(path)


def test_failed_reference_cannot_reuse_a_previous_csv(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tech = canary.ALL_TEST_TECHS["TSMC12"]
    csv_path = work_dir / "ngspice_nmos_lifted_TSMC12_vs0mV.csv"
    csv_path.write_text("sweep current\n0 0\n0.005 -1e-6\n0.01 -2e-6\n")
    monkeypatch.setattr(canary, "create_baked_modelcard", lambda *_args: work_dir / "model.lib")
    monkeypatch.setattr(base.subprocess, "run", lambda *_args, **_kwargs: subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="reference failed",
    ))

    with pytest.raises(RuntimeError, match="NGSPICE failed"):
        canary.run_ngspice_dc_lifted(tech, work_dir, 0.0, "nmos")
    assert not csv_path.exists()


def _mock_main(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> dict[str, np.ndarray]:
    axis = np.arange(161, dtype=float) * 0.005
    trace = {"sweep": axis, "id": axis * 1e-4}
    monkeypatch.setattr(canary, "RESULTS_DIR", work_dir)
    monkeypatch.setattr(canary.RunSpec, "from_environment", classmethod(
        lambda cls: RunSpec(75, "DirectNet-Full"),
    ))
    monkeypatch.setattr(canary, "run_ngspice_dc_lifted", lambda *_args: trace)
    monkeypatch.setattr(canary, "run_nn_dc_lifted", lambda *_args: trace)
    return trace


@pytest.mark.parametrize("runner", ["reference", "candidate"])
def test_partial_sweep_keeps_its_error_slot(
    runner: str, work_dir: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace = _mock_main(work_dir, monkeypatch)
    partial = {key: value[:50] for key, value in trace.items()}
    function = "run_ngspice_dc_lifted" if runner == "reference" else "run_nn_dc_lifted"
    monkeypatch.setattr(canary, function, lambda *_args: partial)

    assert canary.main(["--tech", "TSMC12"]) == (2 if runner == "reference" else 1)
    rows = parse_result_markers(capsys.readouterr().out)
    assert len(rows) == len(canary.LIFTED_DEVICES) * len(canary.VS0_FRACTIONS)
    assert all(row["status"] == "error" and row["metrics"] == {} for row in rows)


def test_diagnostic_mode_does_not_hide_failed_execution(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _mock_main(work_dir, monkeypatch)

    def failed_reference(*_args: Any) -> dict[str, np.ndarray]:
        raise RuntimeError("reference failed")

    monkeypatch.setattr(canary, "run_ngspice_dc_lifted", failed_reference)
    assert canary.main(["--tech", "TSMC12", "--no-gate"]) == 2
    rows = parse_result_markers(capsys.readouterr().out)
    assert rows and all(row["role"] == "diagnostic" for row in rows)


def test_banner_names_the_selected_family(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _mock_main(work_dir, monkeypatch)
    assert canary.main(["--tech", "TSMC12"]) == 0
    banner = capsys.readouterr().out.split("[CSV]", 1)[0]
    assert "DirectNet-Full" in banner and "LEVEL=75" in banner


@pytest.mark.parametrize("device", canary.LIFTED_DEVICES)
def test_reference_and_candidate_share_biases_and_preserve_current_sign(
    device: str, work_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PMOS must use its own model and mirrored sources; abs(Id) hides a defect."""
    from pycircuitsim import parser, simulation

    tech = canary.ALL_TEST_TECHS["TSMC12"]
    axis = np.arange(161, dtype=float) * 0.005
    gate = axis if device == "nmos" else tech.vdd - axis
    # Deliberately negative oriented drain current so an abs() regression fails.
    current = -axis * 1e-4
    source_current = -current if device == "nmos" else current
    device_current = current if device == "nmos" else -current
    lines = ["sweep current\n", *[
        f"{voltage:g} {value:g}\n" for voltage, value in zip(gate, source_current)
    ]]
    monkeypatch.setattr(canary, "run_ngspice_subprocess", lambda *_args: lines)
    monkeypatch.setattr(canary, "create_baked_modelcard", lambda *_args: work_dir / "n.lib")
    monkeypatch.setattr(canary, "create_baked_pmos_modelcard", lambda *_args: work_dir / "p.lib")
    parse = Mock()
    monkeypatch.setattr(parser.Parser, "parse_file", parse)
    solve = Mock(return_value={"g": gate, "i(Mdut)": device_current})
    monkeypatch.setattr(simulation, "run_dc_sweep", solve)

    reference = canary.run_ngspice_dc_lifted(tech, work_dir, 0.08, device)
    candidate = canary.run_nn_dc_lifted(tech, work_dir, 0.08, device)

    np.testing.assert_allclose(reference["sweep"], axis, atol=1e-15)
    np.testing.assert_allclose(candidate["sweep"], axis, atol=1e-15)
    np.testing.assert_allclose(reference["id"], current)
    np.testing.assert_allclose(candidate["id"], current)
    assert solve.call_args.kwargs["require_convergence"] is True
    tag = f"{device}_lifted_TSMC12_vs80mV"
    reference_deck = (work_dir / f"ngspice_{tag}.cir").read_text()
    candidate_deck = (work_dir / f"nn_{tag}.sp").read_text()
    expected = (["Vd d 0 0.8", "Vg g 0 0", "Vs s 0 0.08", "Vb b 0 0",
                 ".dc Vg 0 0.8 0.005"] if device == "nmos" else
                ["Vd d 0 0", "Vg g 0 0.8", "Vs s 0 0.72", "Vb b 0 0.8",
                 ".dc Vg 0.8 0 -0.005"])
    assert all(line in reference_deck and line in candidate_deck for line in expected)
    assert f".model dut_nn {device.upper()}" in candidate_deck
    assert (tech.nmos_model if device == "nmos" else tech.pmos_model) in reference_deck
    assert f"L={16 if device == 'nmos' else 20}n NFIN=2" in candidate_deck


def test_pmos_failure_cannot_be_hidden_by_passing_nmos_lifts(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace = _mock_main(work_dir, monkeypatch)

    def candidate(
        tech: Any, path: Path, vs0: float, device: str,
    ) -> dict[str, np.ndarray]:
        return trace if device == "nmos" else {**trace, "id": -trace["id"]}

    monkeypatch.setattr(canary, "run_nn_dc_lifted", candidate)
    assert canary.main(["--tech", "TSMC12"]) == 1
    rows = parse_result_markers(capsys.readouterr().out)
    assert len(rows) == 6
    assert {row["analysis"] for row in rows} == {
        "vs0_0pct", "vs0_10pct", "vs0_20pct",
        "pmos_vs0_0pct", "pmos_vs0_10pct", "pmos_vs0_20pct",
    }
    assert [row["status"] for row in rows] == ["pass"] * 3 + ["fail"] * 3


def test_diagnostic_mode_reports_large_error_without_a_gate_verdict(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace = _mock_main(work_dir, monkeypatch)
    monkeypatch.setattr(canary, "run_nn_dc_lifted", lambda *_args: {
        **trace, "id": -trace["id"],
    })
    assert canary.main(["--tech", "TSMC12", "--no-gate"]) == 0
    output = capsys.readouterr().out
    rows = parse_result_markers(output)
    assert len(rows) == 6
    assert all(row["role"] == row["status"] == "diagnostic" for row in rows)
    assert all(row["metrics"]["nrmse_pct"] > canary.DC_NRMSE_PASS for row in rows)
    assert "PASSED" not in output


@pytest.mark.parametrize("invalid", ["nonfinite", "duplicate", "missing_point"])
def test_malformed_reference_never_enters_metrics(
    invalid: str, work_dir: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace = {key: value.copy() for key, value in _mock_main(work_dir, monkeypatch).items()}
    if invalid == "nonfinite":
        trace["id"][20] = np.nan
    elif invalid == "duplicate":
        trace["sweep"][20] = trace["sweep"][19]
    else:
        trace = {key: np.delete(value, 20) for key, value in trace.items()}
    monkeypatch.setattr(canary, "run_ngspice_dc_lifted", lambda *_args: trace)
    metrics = Mock(side_effect=AssertionError("invalid data was scored"))
    monkeypatch.setattr(canary, "curve_metrics", metrics)

    assert canary.main(["--tech", "TSMC12"]) == 2
    rows = parse_result_markers(capsys.readouterr().out)
    assert len(rows) == 6 and all(row["status"] == "error" for row in rows)
    metrics.assert_not_called()
