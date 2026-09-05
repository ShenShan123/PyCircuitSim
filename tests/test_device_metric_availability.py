"""Unavailable physical metrics must not hide convergence or impersonate crashes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.common import device_integrity as device
from tests.common.circuit_benchmarks import BENCH
from tests.common.gate_result import GateResult, result_exit_code
from tests.common.simple_circuit_harness import CORNERS, RunSpec


def run_subthreshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, polarity: str,
    *, flat_reference: bool = False, flat_candidate: bool = True,
) -> GateResult:
    """Synthetic curves test metric classification through the real runner."""
    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "75")
    spec = next(spec for spec in device.build_sweeps(BENCH["TSMC5"], polarity)
                if spec.suite == "subthreshold")
    grid = np.linspace(min(spec.start, spec.stop), max(spec.start, spec.stop), 111)
    exponential = 1e-12 * 10.0 ** (np.abs(grid) / 0.06)
    reference = np.full_like(grid, 1e-8) if flat_reference else exponential
    candidate = np.full_like(grid, 1e-8) if flat_candidate else exponential
    monkeypatch.setattr(device, "get_baked_modelcard", lambda *a, **k: tmp_path / "baked.lib")
    monkeypatch.setattr(device, "physical_deck_mismatch", lambda *a, **k: "")
    monkeypatch.setattr(device, "run_reference_sweep", lambda *a, **k: device.DeviceTrace(grid, reference))
    monkeypatch.setattr(device, "run_candidate_sweep", lambda *a, **k: device.DeviceTrace(grid, candidate))
    return device.run_sweep(spec, BENCH["TSMC5"], CORNERS["nominal"], tmp_path, level=75)


@pytest.mark.parametrize("polarity", ("nmos", "pmos"))
def test_flat_candidate_retains_a_scientific_error_and_convergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, polarity: str,
) -> None:
    result = run_subthreshold(tmp_path, monkeypatch, polarity)
    assert result.status == "error"
    assert result.error_kind == "candidate"
    assert result.execution_state == "error"
    assert result.candidate_converged and result.reference_converged
    assert result_exit_code([result]) == 1
    assert result.metrics == {}
    recovered = result.payload()["domain"]["uncharacterized_diagnostic"]
    assert recovered["ss_test_mv_dec"] is None
    assert recovered["ss_test_decades_spanned"] == 0.0
    assert recovered["nrmse_pct"] > 0.0


@pytest.mark.parametrize("polarity", ("nmos", "pmos"))
def test_characterizable_candidate_still_emits_complete_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, polarity: str,
) -> None:
    result = run_subthreshold(tmp_path, monkeypatch, polarity, flat_candidate=False)
    assert result.status == "diagnostic"
    assert result_exit_code([result]) == 0
    assert result.domain["ss_test_mv_dec"] == pytest.approx(60.0)


def test_flat_reference_is_not_excused_as_a_candidate_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = run_subthreshold(tmp_path, monkeypatch, "nmos", flat_reference=True)
    assert result.status == "error"
    assert result.error_kind != "candidate"
    assert result_exit_code([result]) == 2


@pytest.mark.parametrize("defect", ("missing", "none", "infinite", "unexpected_nan"))
def test_malformed_metric_payload_still_fails_as_infrastructure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str,
) -> None:
    original = device.suite_metrics

    def broken_metrics(*args: object, **kwargs: object) -> tuple[dict, dict]:
        metrics, domain = original(*args, **kwargs)
        if defect == "missing":
            domain.pop("ss_test_mv_dec")
        elif defect == "none":
            domain["ss_test_mv_dec"] = None
        elif defect == "infinite":
            domain["ss_test_mv_dec"] = float("inf")
        else:
            domain["ioff_error_pct"] = float("nan")
        return metrics, domain

    monkeypatch.setattr(device, "suite_metrics", broken_metrics)
    result = run_subthreshold(tmp_path, monkeypatch, "nmos")
    assert result.status == "error"
    assert result.error_kind == "infrastructure"
    assert result_exit_code([result]) == 2


def test_cli_convergence_count_includes_uncharacterizable_converged_curves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.single_devices import verify_device_integrity as cli

    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "75")
    result = GateResult(
        case_id="device_subthreshold", tech="TSMC5", corner="nominal",
        analysis="nmos_idvg_log", role="diagnostic", status="error",
        error="candidate slope is unavailable", execution_state="error", error_kind="candidate",
    )
    monkeypatch.setattr(RunSpec, "validate_checkpoint_pins", lambda *a, **k: None)
    monkeypatch.setattr(cli, "RESULTS_BASE", tmp_path)
    monkeypatch.setattr(cli, "run_device_suites", lambda *a, **k: [result])
    assert cli.main(["--tech", "TSMC5", "--suite", "subthreshold", "--device", "nmos"]) == 1
    output = capsys.readouterr().out
    assert "CHARACTERIZED : 0/1" in output
    assert "CONVERGED     : 1/1" in output
