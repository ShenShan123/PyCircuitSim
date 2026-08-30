"""V7.6.6 active-Hermite fine-tuning and feasibility contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts import finetune_active_hermite as hermite


def _metrics(
    mae: list[float],
    maximum: list[float],
    jacobian: float,
) -> dict[str, object]:
    return {
        "replay": {
            "normalized_mae": mae,
            "physical_max_abs": maximum,
        },
        "hermite": {
            "normalized_mae": mae,
            "physical_max_abs": maximum,
        },
        "current_jacobian_mae": jacobian,
        "current_jacobian_mae_by_head": [jacobian] * 3,
    }


def test_feasibility_requires_every_value_head_and_both_value_sets() -> None:
    baseline = _metrics([1.0] * 6, [2.0] * 6, 4.0)
    candidate = _metrics([1.019] * 6, [2.099] * 6, 2.9)

    passed = hermite.evaluate_feasibility(
        baseline, candidate, require_jacobian_improvement=True,
    )
    assert passed["feasible"] is True

    candidate["replay"]["normalized_mae"][4] = 1.021
    failed = hermite.evaluate_feasibility(
        baseline, candidate, require_jacobian_improvement=True,
    )
    assert failed["feasible"] is False
    assert "replay.normalized_mae[4]" in failed["failures"]


def test_lambda_zero_control_does_not_require_jacobian_improvement() -> None:
    baseline = _metrics([1.0] * 6, [1.0] * 6, 4.0)
    candidate = _metrics([1.0] * 6, [1.0] * 6, 5.0)

    control = hermite.evaluate_feasibility(
        baseline, candidate, require_jacobian_improvement=False,
    )
    treatment = hermite.evaluate_feasibility(
        baseline, candidate, require_jacobian_improvement=True,
    )

    assert control["feasible"] is True
    assert treatment["feasible"] is False
    assert "current_jacobian_mae" in treatment["failures"]


def test_feasibility_boundary_is_inclusive() -> None:
    baseline = _metrics([2.0] * 6, [4.0] * 6, 8.0)
    candidate = _metrics([2.04] * 6, [4.2] * 6, 6.0)

    result = hermite.evaluate_feasibility(
        baseline, candidate, require_jacobian_improvement=True,
    )

    assert result["feasible"] is True


def test_conflicting_jacobian_gradient_is_projected_off_value_gradient() -> None:
    value = [torch.tensor([1.0, 0.0])]
    jacobian = [torch.tensor([-1.0, 1.0])]

    projected, conflicted = hermite.project_jacobian_gradients(value, jacobian)

    assert conflicted is True
    assert torch.dot(value[0], projected[0]).item() == pytest.approx(0.0)
    torch.testing.assert_close(projected[0], torch.tensor([0.0, 1.0]))


def test_nonconflicting_jacobian_gradient_is_unchanged() -> None:
    value = [torch.tensor([1.0, 0.0])]
    jacobian = [torch.tensor([1.0, 1.0])]

    projected, conflicted = hermite.project_jacobian_gradients(value, jacobian)

    assert conflicted is False
    torch.testing.assert_close(projected[0], jacobian[0])


def test_combined_gradient_with_zero_lambda_equals_value_gradient() -> None:
    value = [torch.tensor([2.0, -3.0])]
    jacobian = [torch.tensor([100.0, 100.0])]

    combined, _conflicted = hermite.combine_gradients(
        value, jacobian, lambda_jacobian=0.0,
    )

    torch.testing.assert_close(combined[0], value[0])


def test_epoch_order_sweeps_every_replay_row_once() -> None:
    first = hermite.epoch_order(262_144, seed=766, epoch=3)
    second = hermite.epoch_order(262_144, seed=766, epoch=3)

    assert len(first) == 262_144
    assert len(np.unique(first)) == 262_144
    assert first.min() == 0
    assert first.max() == 262_143
    np.testing.assert_array_equal(first, second)


def test_pairing_uses_only_synchronized_feasible_epochs() -> None:
    assert hermite.common_feasible_epochs([1, 2, 4], [2, 3, 4]) == [2, 4]
