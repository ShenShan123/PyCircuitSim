"""Full-terminal trajectory-overlay contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pytest

from scripts import v764_append_full_terminal_corridor as corridor


def test_evaluate_terminal_rows_uses_six_surface_contract_order() -> None:
    inputs = np.asarray([
        [0.2, 0.3, 0.0, -0.1],
        [-0.4, -0.2, 0.0, 0.1],
    ])

    def evaluate(
        vd: float, vg: float, vs: float, vb: float,
    ) -> Tuple[Optional[Dict[str, float]], str]:
        return {
            "qb": vb,
            "qg": vg,
            "qd": vd,
            "i_b": 10.0 + vb,
            "i_g": 10.0 + vg,
            "i_d": 10.0 + vd,
        }, ""

    actual = corridor.evaluate_terminal_rows(inputs, evaluate)

    np.testing.assert_allclose(actual, np.asarray([
        [10.2, 10.3, 9.9, 0.2, 0.3, -0.1],
        [9.6, 9.8, 10.1, -0.4, -0.2, 0.1],
    ]))


def test_evaluate_terminal_rows_fails_loud_on_rejected_truth_point() -> None:
    inputs = np.asarray([[0.2, 0.3, 0.0, -0.1]])

    def reject(
        vd: float, vg: float, vs: float, vb: float,
    ) -> Tuple[Optional[Dict[str, float]], str]:
        del vd, vg, vs, vb
        return None, "internal_node_solve_failed"

    with pytest.raises(RuntimeError, match="row 0.*internal_node_solve_failed"):
        corridor.evaluate_terminal_rows(inputs, reject)


def test_validate_fragment_requires_source_relative_uniform_geometry() -> None:
    inputs = np.asarray([[0.2, 0.3, 0.0, -0.1]])
    geometry = np.zeros((1, 15), dtype=np.float64)
    geometry[0, :3] = [2.0, 16e-9, 300.15]
    corridor.validate_fragment_arrays(
        inputs, geometry, nfin=2.0, length=16e-9, temperature=300.15,
    )

    bad_inputs = inputs.copy()
    bad_inputs[0, 2] = 0.01
    with pytest.raises(ValueError, match="source-relative"):
        corridor.validate_fragment_arrays(
            bad_inputs, geometry, nfin=2.0, length=16e-9,
            temperature=300.15,
        )

    bad_geometry = np.vstack([geometry, geometry])
    bad_geometry[1, 1] = 20e-9
    with pytest.raises(ValueError, match="uniform geometry"):
        corridor.validate_fragment_arrays(
            np.vstack([inputs, inputs]), bad_geometry, nfin=2.0,
            length=16e-9, temperature=300.15,
        )


def test_appended_metadata_preserves_parent_audit_and_records_overlay(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent.npz"
    fragment = tmp_path / "fragment.npz"
    parent.write_bytes(b"parent")
    fragment.write_bytes(b"fragment")
    metadata: Dict[str, object] = {
        "dataset_variant": "v760_full_terminal_core_plus_tg",
        "generator_release": "V7.6.3",
        "requested_rows": 12,
        "kept_rows": 10,
        "rejected_rows": 2,
        "manifest_json": json.dumps([{
            "status": "partial",
            "requested": 12,
            "kept": 10,
            "rejected": 2,
            "failure_reason_counts": {"terminal_current_over_1A": 2},
        }]),
        "externally_appended_sample_class_names": np.asarray([], dtype=str),
    }

    actual = corridor.appended_metadata(
        metadata,
        parent_path=parent,
        fragment_path=fragment,
        corridor_rows=3,
        tech="tsmc5",
        device="nmos",
        variant="lvt",
        length=16e-9,
        nfin=2.0,
        temperature=300.15,
        source_commit="a" * 40,
        generator_command="python append.py",
    )

    assert int(actual["requested_rows"]) == 15
    assert int(actual["kept_rows"]) == 13
    assert int(actual["rejected_rows"]) == 2
    assert actual["source_commit"] == "a" * 40
    assert actual["source_dirty"] is False
    assert actual["generator_release"] == corridor.GENERATOR_RELEASE
    assert actual["parent_dataset"] == parent.name
    assert actual["corridor_fragment"] == fragment.name
    assert actual["externally_appended_sample_class_names"].tolist() == [
        "traj_corridor",
    ]
    manifest = json.loads(str(actual["manifest_json"]))
    assert manifest[-1]["sample_class"] == "traj_corridor"
    assert manifest[-1]["kept"] == 3
    assert sum(int(row.get("rejected", 0)) for row in manifest) == 2
