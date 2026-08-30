"""V7.6.6 active-Hermite overlay selection and provenance contracts."""

from __future__ import annotations

import numpy as np

from scripts import generate_active_hermite_overlay as hermite


def test_bin_seed_is_coordinate_stable_and_domain_separated() -> None:
    coordinates = ("tsmc5", "nmos", "svt", 12.001, 135.01e-9, 300.15)

    first = hermite.coordinate_seed(766, "candidates", *coordinates)
    reordered = [
        hermite.coordinate_seed(766, "candidates", *item)
        for item in [
            ("tsmc5", "nmos", "lvt", 2.0, 6e-9, 248.15),
            coordinates,
        ]
    ]

    assert first == reordered[1]
    assert first != hermite.coordinate_seed(766, "validation", *coordinates)
    assert first != hermite.coordinate_seed(
        766, "candidates", "tsmc5", "nmos", "svt", 12.0,
        135.01e-9, 300.15,
    )


def test_balanced_indices_are_exact_unique_and_deterministic() -> None:
    indices = np.arange(90, dtype=np.int64)
    sample_class = np.repeat(np.arange(9, dtype=np.int8), 10)

    selected = hermite.balanced_indices(indices, sample_class, 64, seed=17)

    assert len(selected) == 64
    assert len(np.unique(selected)) == 64
    counts = np.bincount(sample_class[selected], minlength=9)
    assert counts.max() - counts.min() <= 1
    np.testing.assert_array_equal(
        selected,
        hermite.balanced_indices(indices, sample_class, 64, seed=17),
    )


def test_role_assignment_reserves_validation_before_active_top_k() -> None:
    source_rows = np.arange(64, dtype=np.int64)
    validation_order = np.roll(source_rows, 7)
    scores = np.arange(64, dtype=np.float64)
    scores[20:23] = 100.0

    roles, ranks = hermite.assign_roles(
        source_rows, validation_order, scores,
        validation_count=16, active_count=16,
    )

    validation = source_rows[roles == hermite.ROLE_VALIDATION]
    active = source_rows[roles == hermite.ROLE_ACTIVE]
    assert len(validation) == 16
    assert len(active) == 16
    assert set(validation).isdisjoint(active)
    remaining = np.setdiff1d(source_rows, validation)
    expected = remaining[np.lexsort((remaining, -scores[remaining]))[:16]]
    np.testing.assert_array_equal(np.sort(active), np.sort(expected))
    assert set(ranks[roles == hermite.ROLE_ACTIVE]) == set(range(1, 17))


def test_role_assignment_breaks_score_ties_by_source_row() -> None:
    source_rows = np.arange(64, dtype=np.int64)[::-1]
    scores = np.ones(64, dtype=np.float64)

    roles, _ranks = hermite.assign_roles(
        source_rows, source_rows, scores,
        validation_count=16, active_count=16,
    )

    active = source_rows[roles == hermite.ROLE_ACTIVE]
    np.testing.assert_array_equal(np.sort(active), np.arange(16))


def test_fixed_replay_is_exact_balanced_and_excludes_candidates() -> None:
    bin_ids = np.repeat(np.arange(4, dtype=np.int32), 100)
    sample_class = np.tile(np.repeat(np.arange(5, dtype=np.int8), 20), 4)
    excluded = np.arange(0, 400, 10, dtype=np.int64)

    replay = hermite.fixed_replay_indices(
        bin_ids, sample_class, excluded, count=128, seed=766,
    )

    assert len(replay) == 128
    assert len(np.unique(replay)) == 128
    assert set(replay).isdisjoint(excluded)
    bin_counts = np.bincount(bin_ids[replay], minlength=4)
    assert bin_counts.max() - bin_counts.min() <= 1


def test_expected_plan_counts_are_fixed() -> None:
    plan = hermite.expected_plan_counts(840, 64, 16, 16)

    assert plan == {
        "bins": 840,
        "queried_rows": 53_760,
        "active_rows": 13_440,
        "validation_rows": 13_440,
        "unused_rows": 26_880,
    }


def test_active_score_is_invariant_to_per_head_error_scale() -> None:
    per_head = np.asarray([
        [1.0, 10.0, 0.25],
        [3.0, 20.0, 1.00],
        [5.0, 40.0, 0.50],
    ])
    rescaled = per_head * np.asarray([1e-3, 1e6, 7.0])

    score, scale = hermite.head_balanced_scores(per_head)
    rescaled_score, _rescaled_scale = hermite.head_balanced_scores(rescaled)

    np.testing.assert_allclose(score, rescaled_score)
    np.testing.assert_array_equal(scale, per_head.mean(axis=0))


class _FakeInstance:
    def condense_last_jacobian(self) -> np.ndarray:
        return np.arange(16, dtype=np.float64).reshape(4, 4)

    def condense_last_react(self) -> np.ndarray:
        return np.arange(100, 116, dtype=np.float64).reshape(4, 4)


def test_terminal_jacobian_uses_solver_positive_current_sign() -> None:
    jacobian = hermite.terminal_jacobian(_FakeInstance())

    assert jacobian.shape == (6, 3)
    expected_current = -np.arange(16, dtype=np.float64).reshape(4, 4)[
        np.ix_((0, 1, 3), (0, 1, 3))
    ]
    expected_charge = np.arange(100, 116, dtype=np.float64).reshape(4, 4)[
        np.ix_((0, 1, 3), (0, 1, 3))
    ]
    np.testing.assert_array_equal(jacobian[:3], expected_current)
    np.testing.assert_array_equal(jacobian[3:], expected_charge)
