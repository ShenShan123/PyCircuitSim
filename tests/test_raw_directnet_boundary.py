"""Focused V7.6.0 checks for the raw DirectNet diagnostic boundary."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, NoReturn
from unittest.mock import patch

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from neural_network.data.normalize import NormStats, OUTPUT_COLUMN_ORDER
from neural_network.models.direct_net import DirectNet
from examples.complex_circuits.pycircuitsim_bench import run_compare
from pycircuitsim.models.mosfet_directnet import NMOS_NN
from pycircuitsim.models.mosfet_nn import _MOSFETNNBase
from pycircuitsim import solver


class _TraceProbe:
    """Minimal device exposing the run-comparison trace protocol."""

    def __init__(self, trace: Dict[str, object]) -> None:
        self.trace = trace

    def configure_evaluator(
        self, boundary: str, correction_trace: bool,
    ) -> None:
        del boundary, correction_trace

    def evaluator_trace(self) -> Dict[str, object]:
        return self.trace


def _raise_trace_failure(*args: object, **kwargs: object) -> NoReturn:
    """Raise the harness failure type while retaining its shared metadata."""
    del kwargs
    meta = args[-1]
    assert isinstance(meta, dict)
    raise run_compare._fail(meta, "trace probe failure")


def _write_checkpoint(root: Path) -> Path:
    """Write a deterministic DirectNet whose raw current is always negative."""
    model = DirectNet(
        input_dim=7,
        hidden_dim=8,
        n_layers=1,
        output_dim=13,
        num_tech_codes=2,
        tech_embed_dim=2,
        tech_embed_dropout=0.0,
        unknown_code_id=1,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.net[-1].bias[0] = -10.0
    checkpoint = root / "raw_probe_best.pt"
    torch.save(model.state_dict(), checkpoint)
    stats = NormStats(
        mode="zscore",
        input_mean=np.zeros(7, dtype=np.float64),
        input_std=np.ones(7, dtype=np.float64),
        input_min=np.asarray(
            [-0.8, -0.8, -0.8, -0.8, 0.0, 0.0, 200.0],
            dtype=np.float64,
        ),
        input_max=np.asarray(
            [0.8, 0.8, 0.8, 0.8, 8.0, 1e-6, 500.0],
            dtype=np.float64,
        ),
        output_mean=np.zeros(13, dtype=np.float64),
        output_std=np.asarray(
            [1e-4] * 4 + [1e-15] * 9,
            dtype=np.float64,
        ),
        output_columns=list(OUTPUT_COLUMN_ORDER),
    )
    stats.save(str(root / "raw_probe_norm.npz"))
    return checkpoint


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_checkpoint(tmp_path_factory.mktemp("raw_directnet"))


def _device(checkpoint: Path, multiplier: float = 1.0) -> NMOS_NN:
    return NMOS_NN(
        "Mprobe", ["d", "g", "s", "b"], str(checkpoint),
        L=16e-9, NFIN=2.0, tech_code=0, multiplier=multiplier,
    )


def test_raw_directnet_bypasses_input_and_output_corrections(
    checkpoint: Path,
) -> None:
    voltages = {"d": 1.0, "g": 1.1, "s": 0.0, "b": 0.0}
    native = _device(checkpoint)
    raw = _device(checkpoint, multiplier=4.0)
    raw.configure_evaluator("raw-directnet", correction_trace=False)

    native_result = native._eval(voltages)
    raw_result = raw._eval(voltages)
    native_x, _, _ = native._prep_voltages(voltages)
    raw_x, _, _ = raw._prep_voltages(voltages)

    assert raw_x[0, :4].tolist() == pytest.approx([1.0, 1.1, 0.0, 0.0])
    assert native_x[0, :4].tolist() != raw_x[0, :4].tolist()
    assert raw_result["id"] == pytest.approx(-1e-3)
    assert native_result["id"] < raw_result["id"]  # rail extrapolation
    assert raw_result["gds"] == 0.0
    assert native_result["gds"] > 0.0  # negative/zero-gds guard
    assert raw.calculate_current(voltages) == -4.0 * raw_result["id"]

    reverse = {"d": -0.1, "g": 0.5, "s": 0.0, "b": 0.0}
    native.clear_cache()
    raw.clear_cache()
    assert native._eval(reverse)["id"] == 0.0  # taper + sign clamp
    assert raw._eval(reverse)["id"] == pytest.approx(-1e-3)


def test_raw_directnet_is_never_prewarmed_with_production_values(
    checkpoint: Path,
) -> None:
    raw = _device(checkpoint)
    raw.configure_evaluator("raw-directnet", correction_trace=False)

    _MOSFETNNBase.batch_eval(
        [raw], {"d": 0.2, "g": 0.5, "s": 0.0, "b": 0.0},
    )

    assert raw._eval_cache is None


def test_raw_directnet_keeps_the_classic_level73_solver_stamp(
    checkpoint: Path,
) -> None:
    raw = _device(checkpoint)
    raw.configure_evaluator("raw-directnet", correction_trace=False)
    matrix = np.zeros((4, 4), dtype=np.float64)
    rhs = np.zeros(4, dtype=np.float64)

    solver._stamp_mosfet_dc(
        raw, matrix, rhs, {"d": 0, "g": 1, "s": 2, "b": 3},
        {"d": 0.2, "g": 0.5, "s": 0.0, "b": 0.0}, 1e-12,
    )

    assert np.isfinite(matrix).all()
    assert np.isfinite(rhs).all()
    assert raw._eval_cache is not None


def test_correction_trace_is_observational_and_records_first_activations(
    checkpoint: Path,
) -> None:
    outside = {"d": 1.0, "g": 1.1, "s": 0.0, "b": 0.0}
    reverse = {"d": -0.1, "g": 0.5, "s": 0.0, "b": 0.0}
    untraced = _device(checkpoint)
    traced = _device(checkpoint)
    traced.configure_evaluator("native", correction_trace=True)

    assert traced._eval(outside) == untraced._eval(outside)
    traced.clear_cache()
    traced._eval(reverse)
    trace = traced.evaluator_trace()

    assert trace["evaluations"] == 2
    assert trace["max_normalized_support_distance"] == pytest.approx(0.3)
    first: Dict[str, Dict[str, float]] = trace["first_activation"]
    assert first["input_clamp"]["eval_index"] == 1
    assert first["rail_extrapolation"]["eval_index"] == 1
    assert first["negative_gds_guard"]["eval_index"] == 1
    assert first["reverse_taper"]["eval_index"] == 2
    assert first["sign_clamp"]["eval_index"] == 2
    assert first["input_clamp"]["vds"] == 1.0
    assert first["reverse_taper"]["vds"] == -0.1

    raw_untraced = _device(checkpoint)
    raw_untraced.configure_evaluator("raw-directnet", correction_trace=False)
    raw_traced = _device(checkpoint)
    raw_traced.configure_evaluator("raw-directnet", correction_trace=True)
    assert raw_traced._eval(outside) == raw_untraced._eval(outside)
    raw_traced.clear_cache()
    raw_traced._eval(reverse)
    assert set(raw_traced.evaluator_trace()["first_activation"]) == {
        "input_clamp", "rail_extrapolation", "negative_gds_guard",
        "reverse_taper", "sign_clamp",
    }


def test_native_batch_trace_does_not_change_cached_values(
    checkpoint: Path,
) -> None:
    voltages = {"d": 1.0, "g": 1.1, "s": 0.0, "b": 0.0}
    devices = [_device(checkpoint), _device(checkpoint)]
    _MOSFETNNBase.batch_eval(devices, voltages)
    baseline = [dict(device._eval_cache or {}) for device in devices]

    for device in devices:
        device.configure_evaluator("native", correction_trace=True)
    _MOSFETNNBase.batch_eval(devices, voltages)

    assert [device._eval_cache for device in devices] == baseline
    assert [device.evaluator_trace()["evaluations"] for device in devices] == [1, 1]


def test_run_compare_validates_and_emits_raw_boundary_configuration() -> None:
    raw = run_compare.SimOptions(
        evaluator_boundary="raw-directnet", correction_trace=True,
    )
    run_compare._validate_evaluator_boundary(73, raw)
    provenance = run_compare._model_provenance(
        SimpleNamespace(model_level=73, tech="tsmc5"), raw,
    )
    assert provenance["evaluator_boundary"] == "raw-directnet"
    assert provenance["correction_trace"] is True

    with pytest.raises(ValueError, match="LEVEL=73"):
        run_compare._validate_evaluator_boundary(72, raw)
    with pytest.raises(ValueError, match="LEVEL=73"):
        run_compare._validate_evaluator_boundary(
            72, run_compare.SimOptions(correction_trace=True),
        )
    with pytest.raises(ValueError, match="LEVEL=72"):
        run_compare._validate_evaluator_boundary(
            73, run_compare.SimOptions(evaluator_boundary="reduced-osdi"),
        )
    with pytest.raises(SystemExit) as exc:
        run_compare.main([
            "--category", "amplifier",
            "--model-level", "72",
            "--evaluator-boundary", "raw-directnet",
        ])
    assert exc.value.code == 2


def test_simulation_metadata_contains_opt_in_correction_trace() -> None:
    trace = {
        "device": "M1",
        "evaluations": 1,
        "max_normalized_support_distance": 0.25,
        "first_activation": {},
    }
    device = _TraceProbe(trace)
    circuit = SimpleNamespace(components=[device])
    td = SimpleNamespace(model_level=73, temp_c=None)
    plan = SimpleNamespace(kind="dc_source")
    result = run_compare.SweepResult(
        kind="dc", x_name="x", x=np.asarray([]), v={}, i={}, ok=None,
        source="pycircuitsim", meta={},
    )
    opts = run_compare.SimOptions(
        evaluator_boundary="raw-directnet", correction_trace=True,
    )
    with (
        patch.object(run_compare, "build_circuit", return_value=(circuit, Path("x"))),
        patch.object(run_compare, "_mosfets", return_value=[device]),
        patch.object(run_compare, "_supply_rail", return_value=1.0),
        patch.object(run_compare, "_node_table", return_value={}),
        patch.object(run_compare, "_branch_sources", return_value={}),
        patch.object(run_compare, "_simulate_dc", return_value=result),
    ):
        observed = run_compare.simulate(td, plan, Path("."), opts)

    assert observed.meta["correction_trace"] == [trace]


def test_failed_simulation_metadata_contains_opt_in_correction_trace() -> None:
    trace = {
        "device": "M1",
        "evaluations": 1,
        "max_normalized_support_distance": 0.25,
        "first_activation": {"input_clamp": {"eval_index": 1}},
    }
    device = _TraceProbe(trace)
    circuit = SimpleNamespace(components=[device])
    td = SimpleNamespace(model_level=73, temp_c=None)
    plan = SimpleNamespace(kind="dc_source")
    opts = run_compare.SimOptions(correction_trace=True)

    with (
        patch.object(run_compare, "build_circuit", return_value=(circuit, Path("x"))),
        patch.object(run_compare, "_mosfets", return_value=[device]),
        patch.object(run_compare, "_supply_rail", return_value=1.0),
        patch.object(run_compare, "_node_table", return_value={}),
        patch.object(run_compare, "_branch_sources", return_value={}),
        patch.object(run_compare, "_simulate_dc", side_effect=_raise_trace_failure),
        pytest.raises(run_compare.SimFailure) as exc,
    ):
        run_compare.simulate(td, plan, Path("."), opts)

    assert exc.value.partial_meta["correction_trace"] == [trace]
