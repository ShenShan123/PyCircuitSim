"""V7.6.8 differentiable full-terminal circuit-loss contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pycircuitsim.models.mosfet_directnet_full import _load_artifacts
from scripts import finetune_active_hermite as hermite
from scripts import finetune_dnf_unrolled_ldo as unrolled


class _ConstantModel(torch.nn.Module):
    def __init__(self, outputs: list[float]) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(outputs, dtype=torch.float64))

    def forward(
        self,
        x: torch.Tensor,
        tech_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert tech_codes is not None
        return self.bias.to(x).expand(len(x), -1)


def _stats(*, mode: str = "zscore") -> SimpleNamespace:
    lower = np.full(7, -10.0)
    upper = np.full(7, 10.0)
    lower[6], upper[6] = 200.0, 500.0
    return SimpleNamespace(
        mode=mode,
        input_mean=np.zeros(7),
        input_std=np.ones(7),
        input_min=lower,
        input_max=upper,
        output_mean=np.zeros(6),
        output_std=np.ones(6),
        asinh_scale=(np.arange(1.0, 7.0) if mode == "asinh" else None),
    )


def _one_device_artifact(*, is_pmos: bool = False) -> unrolled.CircuitArtifact:
    return unrolled.CircuitArtifact.from_mapping({
        "schema_version": np.asarray(1),
        "schema_name": np.asarray("v768-dnf-unrolled-ldo"),
        "free_nodes": np.asarray(["d", "g", "b"]),
        "fixed_nodes": np.asarray(["s"]),
        "output_node": np.asarray("d"),
        "vout_index": np.asarray(0),
        "vout_target": np.asarray([1.0]),
        "state_group": np.asarray([0]),
        "group_segment_offsets": np.asarray([0, 1]),
        "group_sweep": np.asarray([0.0]),
        "free_voltage_l72": np.asarray([[1.0, 0.25, 0.5]]),
        "fixed_voltage_l72": np.asarray([[0.0]]),
        "arm_scale": np.ones((1, 3)),
        "mos_is_pmos": np.asarray([is_pmos]),
        "mos_term_free": np.asarray([[0, 1, -1, 2]]),
        "mos_term_fixed": np.asarray([[-1, -1, 0, -1]]),
        "mos_nfin": np.asarray([2.0]),
        "mos_length": np.asarray([20e-9]),
        "mos_temperature": np.asarray([300.15]),
        "mos_multiplier": np.asarray([2.0]),
        "mos_code": np.asarray([0]),
        "mos_support_min": np.asarray([_stats().input_min]),
        "mos_support_max": np.asarray([_stats().input_max]),
        "mos_support_min_fractional_margin": np.asarray([0.0]),
        "resistor_term_free": np.asarray([[0, -1]]),
        "resistor_term_fixed": np.asarray([[-1, 0]]),
        "resistor_conductance": np.asarray([0.5]),
        "current_term_free": np.asarray([[2, -1]]),
        "current_term_fixed": np.asarray([[-1, 0]]),
        "current_value": np.asarray([0.25]),
        "gmin": np.asarray(1e-12),
        "gmin_terminal_pairs": np.asarray([[0, 2], [0, 3], [2, 3]]),
        "gmin_scales_with_multiplier": np.asarray(False),
        "terminal_order": np.asarray(["d", "g", "s", "b"]),
    }, device=torch.device("cpu"), dtype=torch.float64)


def _fixed_contract_artifact() -> unrolled.CircuitArtifact:
    """Expand the small residual fixture to the fixed 28-state/9-device gate."""
    base = _one_device_artifact()
    states = unrolled.EXPECTED_STATES
    devices = unrolled.EXPECTED_DEVICES
    return replace(
        base,
        state_group=torch.arange(states),
        group_segment_offsets=torch.arange(states + 1),
        group_sweep=torch.as_tensor(
            [0.65 + 0.005 * index for index in range(14)]
            + [0.65 - 0.005 * index for index in range(14)],
            dtype=torch.float64,
        ),
        free_voltage_l72=base.free_voltage_l72.repeat(states, 1),
        fixed_voltage_l72=base.fixed_voltage_l72.repeat(states, 1),
        arm_scale=base.arm_scale.repeat(states, 1),
        mos_is_pmos=base.mos_is_pmos.repeat(devices),
        mos_term_free=base.mos_term_free.repeat(devices, 1),
        mos_term_fixed=base.mos_term_fixed.repeat(devices, 1),
        mos_nfin=base.mos_nfin.repeat(devices),
        mos_length=base.mos_length.repeat(devices),
        mos_temperature=base.mos_temperature.repeat(devices),
        mos_multiplier=base.mos_multiplier.repeat(devices),
        mos_code=base.mos_code.repeat(devices),
        mos_support_min=base.mos_support_min.repeat(devices, 1),
        mos_support_max=base.mos_support_max.repeat(devices, 1),
        mos_support_min_fractional_margin=(
            base.mos_support_min_fractional_margin.repeat(devices)
        ),
    )


def _write_provenance_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, Path], dict[str, object]]:
    artifact_path = tmp_path / "topology.npz"
    artifact_path.write_bytes(b"fixed topology")
    checkpoints: dict[str, Path] = {}
    parents: dict[str, dict[str, str]] = {}
    for polarity in ("nmos", "pmos"):
        checkpoint = tmp_path / f"parent_{polarity}_best.pt"
        normalization = hermite._normalization_path(checkpoint)
        completion = Path(f"{checkpoint}.complete")
        checkpoint.write_bytes(f"{polarity} checkpoint".encode())
        normalization.write_bytes(f"{polarity} normalization".encode())
        completion.write_bytes(f"{polarity} completion".encode())
        checkpoints[polarity] = checkpoint
        parents[polarity] = {
            "checkpoint_sha256": hermite.sha256_file(checkpoint),
            "normalization_sha256": hermite.sha256_file(normalization),
            "completion_sha256": hermite.sha256_file(completion),
        }
    marker: dict[str, object] = {
        "schema_version": unrolled.SCHEMA_VERSION,
        "schema_name": unrolled.SCHEMA_NAME,
        "artifact": artifact_path.name,
        "artifact_sha256": hermite.sha256_file(artifact_path),
        "states": unrolled.EXPECTED_STATES,
        "groups": unrolled.EXPECTED_GROUPS,
        "devices": unrolled.EXPECTED_DEVICES,
        "parent_checkpoints": parents,
    }
    Path(f"{artifact_path}.complete").write_text(json.dumps(marker))
    return artifact_path, checkpoints, marker


def _metrics(jacobian: float = 1.0) -> dict[str, object]:
    values = {
        "normalized_mae": [1.0] * 6,
        "physical_max_abs": [1.0] * 6,
    }
    return {
        "replay": dict(values),
        "hermite": dict(values),
        "current_jacobian_mae": jacobian,
        "current_jacobian_mae_by_head": [jacobian] * 3,
    }


def test_asinh_denormalization_is_exact_and_differentiable() -> None:
    normalized = torch.tensor(
        [[-0.5, 0.0, 0.5, 1.0, -1.0, 0.25]],
        dtype=torch.float64,
        requires_grad=True,
    )
    stats = _stats(mode="asinh")

    physical = unrolled.denormalize_outputs(normalized, stats)
    expected = torch.as_tensor(stats.asinh_scale) * torch.sinh(normalized)

    torch.testing.assert_close(physical, expected)
    physical.sum().backward()
    assert normalized.grad is not None
    assert torch.all(torch.isfinite(normalized.grad))


def test_terminal_closure_applies_multiplier_and_preserves_kcl() -> None:
    independent = torch.tensor([[[1.0, 2.0, 3.0]]])

    currents = unrolled.close_terminal_currents(
        independent, torch.tensor([2.0]),
    )

    expected = torch.tensor([[[2.0, 4.0, -12.0, 6.0]]])
    torch.testing.assert_close(currents, expected)
    torch.testing.assert_close(currents.sum(dim=-1), torch.zeros((1, 1)))


def test_residual_stamps_all_terminals_passives_sources_and_gmin() -> None:
    artifact = _one_device_artifact(is_pmos=True)
    model = _ConstantModel([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    residual = unrolled.FullTerminalResidual(
        artifact,
        models={"nmos": model, "pmos": model},
        stats={"nmos": _stats(), "pmos": _stats()},
    )

    result = residual(artifact.free_voltage_l72)

    # MOS free-terminal KCL is [2, 4, 6]. The 2-S multiplier is already
    # applied. R(d,s) adds 0.5 A at d, I(b,s) adds 0.25 A at b, and the
    # physical GMIN branches add their exact d-s/d-b/s-b currents.
    expected = torch.tensor(
        [[2.5 + 1.5e-12, 4.0, 6.25]], dtype=torch.float64,
    )
    torch.testing.assert_close(result, expected, rtol=0.0, atol=1e-14)


def test_source_relative_inputs_are_checked_without_clamping() -> None:
    artifact = _one_device_artifact()
    stats = _stats()
    artifact.mos_support_max[0, 0] = 0.75
    residual = unrolled.FullTerminalResidual(
        artifact,
        models={
            "nmos": _ConstantModel([0.0] * 6),
            "pmos": _ConstantModel([0.0] * 6),
        },
        stats={"nmos": stats, "pmos": stats},
    )

    with pytest.raises(unrolled.SupportError, match="input 0"):
        residual(artifact.free_voltage_l72)


def test_true_lm_step_uses_normal_equations_and_trust_clip() -> None:
    initial = torch.zeros((1, 1), dtype=torch.float64)

    result = unrolled.unroll_lm(
        lambda value: value - 1.0,
        initial,
        torch.ones_like(initial),
        steps=1,
        lm_lambda=1.0,
        step=1.0,
        trust_clip=0.1,
        create_graph=True,
    )

    # Unclipped true-LM delta is -0.5; the declared trust region makes the
    # accepted voltage update +0.1 V.
    torch.testing.assert_close(
        result.voltages, torch.tensor([[0.1]], dtype=torch.float64))
    torch.testing.assert_close(
        result.max_abs_update, torch.tensor(0.1, dtype=torch.float64))


def test_lm_normal_equations_are_solved_in_float64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, torch.dtype] = {}
    original_solve = torch.linalg.solve

    def recording_solve(
        matrix: torch.Tensor, right: torch.Tensor,
    ) -> torch.Tensor:
        observed["matrix"] = matrix.dtype
        observed["right"] = right.dtype
        return original_solve(matrix, right)

    monkeypatch.setattr(torch.linalg, "solve", recording_solve)
    result = unrolled.unroll_lm(
        lambda value: value - 1.0,
        torch.zeros((1, 1), dtype=torch.float32),
        torch.ones((1, 1), dtype=torch.float32),
        steps=1,
        lm_lambda=1.0,
        step=1.0,
        trust_clip=0.1,
        create_graph=True,
    )

    assert observed == {"matrix": torch.float64, "right": torch.float64}
    assert result.voltages.dtype == torch.float32


def test_lm_unroll_preserves_second_order_model_gradient() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    initial = torch.full((1, 1), 0.25, dtype=torch.float64)

    result = unrolled.unroll_lm(
        lambda value: parameter * value - 1.0,
        initial,
        torch.ones_like(initial),
        steps=2,
        lm_lambda=0.05,
        step=1.0,
        trust_clip=0.1,
        create_graph=True,
    )
    result.voltages.square().sum().backward()

    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad)


def test_candidate_gate_requires_both_polarities_jacobian_and_half_vout() -> None:
    baselines = {"nmos": _metrics(), "pmos": _metrics()}
    candidates = {"nmos": _metrics(1.049), "pmos": _metrics(1.0)}

    passed = unrolled.evaluate_candidate_gate(
        baselines, candidates,
        baseline_vout_mae=0.1, candidate_vout_mae=0.05,
    )
    assert passed["feasible"] is True

    candidates["pmos"] = _metrics(1.051)
    failed_j = unrolled.evaluate_candidate_gate(
        baselines, candidates,
        baseline_vout_mae=0.1, candidate_vout_mae=0.05,
    )
    assert failed_j["feasible"] is False
    assert "pmos.current_jacobian_mae" in failed_j["failures"]

    failed_curve = unrolled.evaluate_candidate_gate(
        baselines, {"nmos": _metrics(), "pmos": _metrics()},
        baseline_vout_mae=0.1, candidate_vout_mae=0.0500001,
    )
    assert failed_curve["feasible"] is False
    assert "local_vout_mae" in failed_curve["failures"]


def test_only_predeclared_control_and_treatment_arms_exist() -> None:
    assert unrolled.circuit_weight("control") == 0.0
    assert unrolled.circuit_weight("treatment") == 1.0
    with pytest.raises(ValueError, match="arm"):
        unrolled.circuit_weight("weight-search")


def test_epoch_schedule_runs_both_replay_polarities() -> None:
    assert unrolled.replay_polarities("treatment") == ("nmos", "pmos")
    assert unrolled.replay_polarities("control") == ("nmos", "pmos")


def test_control_never_emits_a_candidate_checkpoint() -> None:
    assert unrolled.should_save_candidate(
        "treatment", {"feasible": True},
    ) is True
    assert unrolled.should_save_candidate(
        "control", {"feasible": True},
    ) is False
    assert unrolled.should_save_candidate(
        "treatment", {"feasible": False},
    ) is False


def test_candidate_stems_end_in_the_parser_polarity_suffix() -> None:
    assert unrolled._stem(
        Path("tsmc5_dnf_medium_nmos_best.pt"), "nmos", None, "treatment",
    ) == "tsmc5_dnf_medium_unrolled_treatment_nmos"
    assert unrolled._stem(
        Path("tsmc5_dnf_medium_pmos_best.pt"), "pmos", None, "treatment",
    ) == "tsmc5_dnf_medium_unrolled_treatment_pmos"
    with pytest.raises(ValueError, match="must end in _nmos"):
        unrolled._stem(
            Path("tsmc5_dnf_medium_nmos_best.pt"), "nmos", "wrong_pmos",
            "treatment",
        )
    with pytest.raises(ValueError, match="nmos checkpoint"):
        unrolled._stem(
            Path("tsmc5_dnf_medium_pmos_best.pt"), "nmos", None, "treatment",
        )


def test_topology_provenance_rejects_corrupt_hashes_and_counts(
    tmp_path: Path,
) -> None:
    artifact_path, checkpoints, marker = _write_provenance_fixture(tmp_path)
    artifact = _fixed_contract_artifact()
    marker_path = Path(f"{artifact_path}.complete")
    unrolled._verify_topology_provenance(
        artifact_path, checkpoints, artifact)

    corruptions = (
        ("artifact_sha256", None),
        ("states", None),
        ("checkpoint_sha256", "nmos"),
        ("normalization_sha256", "pmos"),
        ("completion_sha256", "pmos"),
    )
    for field, polarity in corruptions:
        corrupted = copy.deepcopy(marker)
        if polarity is None:
            corrupted[field] = "wrong"
        else:
            corrupted["parent_checkpoints"][polarity][field] = "wrong"
        marker_path.write_text(json.dumps(corrupted))
        with pytest.raises(ValueError):
            unrolled._verify_topology_provenance(
                artifact_path, checkpoints, artifact)
    marker_path.write_text(json.dumps(marker))


def test_topology_provenance_cross_checks_npz_denominator(tmp_path: Path) -> None:
    artifact_path, checkpoints, _marker = _write_provenance_fixture(tmp_path)

    with pytest.raises(ValueError, match="artifact states must be 28"):
        unrolled._verify_topology_provenance(
            artifact_path, checkpoints, _one_device_artifact())


@pytest.mark.skipif(
    os.environ.get("PYCIRCUITSIM_RUN_PRIVATE_INTEGRATION") != "1",
    reason="requires the preserved V7.6.4 parent bundle and harvested LDO artifact",
)
def test_real_parent_residual_matches_frozen_ldo_baseline() -> None:
    """Catch polarity, device-order, code, and input-frame integration drift."""
    root = Path(__file__).resolve().parents[1]
    artifact_path = (
        root / "results/v768_dnf_unrolled_ldo"
        / "tsmc5_ldo_1_tb_line_max_unrolled.npz"
    )
    checkpoint_dir = root / "results/v764_terminal_l_matched_checkpoints"
    checkpoints = {
        polarity: checkpoint_dir / f"tsmc5_dnf_medium_{polarity}_best.pt"
        for polarity in ("nmos", "pmos")
    }
    artifact = unrolled.CircuitArtifact.load(
        artifact_path, device=torch.device("cpu"), dtype=torch.float32)
    marker = unrolled._verify_topology_provenance(
        artifact_path, checkpoints, artifact)
    assert marker["artifact_sha256"] == (
        "0ff0399a9b33c03d3f9840482c2613d68ada722120dcbdf026d4a2dff09422da"
    )

    models: dict[str, torch.nn.Module] = {}
    stats: dict[str, object] = {}
    for polarity, checkpoint in checkpoints.items():
        model, normalization, _codes, outputs = _load_artifacts(checkpoint)
        assert tuple(outputs) == tuple(hermite.OUTPUT_COLUMNS)
        models[polarity] = copy.deepcopy(model).cpu().eval()
        stats[polarity] = normalization
    residual = unrolled.FullTerminalResidual(
        artifact, models=models, stats=stats)
    parent_kcl = residual(artifact.free_voltage_l72).detach()
    assert hashlib.sha256(parent_kcl.numpy().tobytes()).hexdigest() == (
        "03f700813335dfe4e4c739d4fb2954deb72b58dbd482e107c8c4b669cd9dc96a"
    )
    scaled = parent_kcl / artifact.arm_scale
    assert float(torch.sqrt(torch.mean(scaled.square()))) == pytest.approx(
        0.379681795835495, rel=1e-7)

    result, metrics = unrolled._circuit_metrics(
        artifact, models, stats, create_graph=False)
    assert metrics["vout_mae"] == pytest.approx(0.07031194865703583, rel=1e-6)
    assert metrics["residual_rms"] == pytest.approx(
        0.00011036972136935219, rel=2e-5)
    assert float(torch.max(torch.abs(
        result.voltages[:, artifact.output_index]
        - artifact.free_voltage_l72[:, artifact.output_index]
    )).detach()) == pytest.approx(0.1160021424293518, rel=1e-6)
