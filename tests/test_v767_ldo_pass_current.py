"""Focused contracts for the V7.6.7 LDO pass-current correction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.fit_ldo_pass_current import (
    aligned_sweep_state,
    apply_output_row_delta,
    candidate_gate,
    checkpoint_override_prefix,
    minimum_local_scale,
    solve_output_row_delta,
    validate_line_sweep,
    validate_sweep_grid,
)


class _Circuit:
    def get_nodes(self) -> list[str]:
        return ["VDD", "out", "x1.n1"]


def test_checkpoint_override_uses_the_parser_save_prefix_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tsmc5_dnf_medium_pmos_best.pt"
    assert checkpoint_override_prefix(path, "pmos").endswith(
        "tsmc5_dnf_medium_pmos")
    with pytest.raises(ValueError, match="_nmos_best"):
        checkpoint_override_prefix(path, "nmos")


def test_line_sweep_denominator_is_not_caller_adjustable() -> None:
    plan = SimpleNamespace(
        label="up", source="V1", start=0.65, stop=0.715, step=0.005,
        kind="dc_source",
    )

    assert validate_line_sweep(plan) == 14
    plan.stop = 0.71
    with pytest.raises(ValueError, match="plan 'up' changed"):
        validate_line_sweep(plan)


def test_ngspice_grid_must_match_the_fixed_plan() -> None:
    plan = SimpleNamespace(
        label="up", source="V1", start=0.65, stop=0.715, step=0.005,
        kind="dc_source",
    )
    expected = np.arange(0.65, 0.72, 0.005)
    sweep = SimpleNamespace(kind="dc", x=expected.copy())

    np.testing.assert_array_equal(validate_sweep_grid(plan, sweep), expected)
    sweep.x[-1] = 0.714
    with pytest.raises(ValueError, match="abscissa"):
        validate_sweep_grid(plan, sweep)


def test_ngspice_grid_rejects_non_dc_sweeps() -> None:
    plan = SimpleNamespace(
        label="up", source="V1", start=0.65, stop=0.715, step=0.005,
        kind="dc_source",
    )
    sweep = SimpleNamespace(
        kind="tran", x=np.arange(0.65, 0.72, 0.005),
    )

    with pytest.raises(ValueError, match="not a DC sweep"):
        validate_sweep_grid(plan, sweep)


def test_aligned_sweep_state_requires_the_complete_physical_state() -> None:
    sweep = SimpleNamespace(v={
        "vdd": np.asarray([0.65]),
        "out": np.asarray([0.48]),
        "x1.n1": np.asarray([0.22]),
        "x1.mm1#di": np.asarray([0.21]),
    })
    state = aligned_sweep_state(_Circuit(), sweep, 0)
    assert state == {"VDD": 0.65, "out": 0.48, "x1.n1": 0.22,
                     "0": 0.0, "GND": 0.0}

    sweep.v.pop("out")
    with pytest.raises(ValueError, match="missing=.*out"):
        aligned_sweep_state(_Circuit(), sweep, 0)


def test_disabled_local_residual_is_an_exact_parent_control() -> None:
    local = np.asarray([[1.0, 2.0], [1.0, -1.0]], dtype=np.float64)
    replay = np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    delta, diagnostics = solve_output_row_delta(
        local, np.asarray([0.4, -0.2]), replay,
        local_weight=0.0, anchor_weight=1.0, ridge_ratio=1e-6,
    )
    assert np.array_equal(delta, np.zeros(2, dtype=np.float64))
    assert diagnostics["control_exact"] is True


def test_ridge_solution_reduces_local_error_with_replay_anchor() -> None:
    local = np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    residual = np.asarray([1.0, 1.0], dtype=np.float64)
    replay = np.asarray([[1.0, -1.0], [1.0, 2.0]], dtype=np.float64)
    delta, diagnostics = solve_output_row_delta(
        local, residual, replay,
        local_weight=1.0, anchor_weight=1.0, ridge_ratio=1e-6,
    )
    assert np.mean(np.abs(residual - local @ delta)) < np.mean(
        np.abs(residual))
    assert diagnostics["ridge"] > 0.0


def test_minimum_local_scale_uses_only_the_declared_training_error() -> None:
    parent = np.asarray([0.0, 0.0], dtype=np.float64)
    target = np.asarray([1.0, 1.0], dtype=np.float64)
    full_effect = np.asarray([1.0, 1.0], dtype=np.float64)

    scale = minimum_local_scale(
        parent, target, full_effect, target_ratio=0.5,
    )

    assert scale == pytest.approx(0.5, abs=1e-12)


def test_only_the_selected_output_row_changes() -> None:
    state = {
        "net.12.weight": torch.arange(18, dtype=torch.float32).reshape(3, 6),
        "net.12.bias": torch.arange(3, dtype=torch.float32),
        "net.10.weight": torch.ones((6, 6), dtype=torch.float32),
    }
    original = {name: value.clone() for name, value in state.items()}
    apply_output_row_delta(
        state, "net.12.weight", "net.12.bias", row=0,
        delta=np.arange(7, dtype=np.float64),
    )
    assert torch.equal(state["net.12.weight"][1:],
                       original["net.12.weight"][1:])
    assert torch.equal(state["net.12.bias"][1:],
                       original["net.12.bias"][1:])
    assert torch.equal(state["net.10.weight"], original["net.10.weight"])
    assert not torch.equal(state["net.12.weight"][0],
                           original["net.12.weight"][0])


def test_candidate_gate_keeps_local_and_global_requirements_independent() -> None:
    baseline = {
        "replay": {"normalized_mae": [1.0] * 6,
                   "physical_max_abs": [1.0] * 6},
        "hermite": {"normalized_mae": [1.0] * 6,
                    "physical_max_abs": [1.0] * 6},
        "current_jacobian_mae": 2.0,
    }
    candidate = {
        "replay": {"normalized_mae": [1.01] * 6,
                   "physical_max_abs": [1.04] * 6},
        "hermite": {"normalized_mae": [1.01] * 6,
                    "physical_max_abs": [1.04] * 6},
        "current_jacobian_mae": 2.08,
    }
    passed = candidate_gate(
        baseline, candidate, local_parent_mae=1.0, local_candidate_mae=0.49,
    )
    assert passed["eligible"] is True

    failed = candidate_gate(
        baseline, candidate, local_parent_mae=1.0, local_candidate_mae=0.51,
    )
    assert failed["eligible"] is False
    assert "local_i_d_mae" in failed["failures"]
