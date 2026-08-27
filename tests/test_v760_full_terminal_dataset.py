"""V7.6.0 full-terminal generation and dataset-schema contracts."""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "bsim_cmg"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from neural_network.data import dataset as dataset_module
from neural_network.cli import train as train_cli
from neural_network.config import DirectNetConfig
from neural_network.data.contracts import (
    FULL_TERMINAL_OUTPUT_CONTRACT,
    REDUCED_OUTPUT_CONTRACT,
    dataset_filename,
)
from neural_network.training import trainer
from pycmg import nn_generate


FULL_COLUMNS = ["i_d", "i_g", "i_b", "qd", "qg", "qb"]


class _Instance:
    def eval_dc(self, _nodes: dict[str, float]) -> dict[str, float]:
        return {
            "id": -0.2,
            "ig": -0.3,
            "is": 1.0,
            "ie": -0.5,
            "qd": 2e-15,
            "qg": 3e-15,
            "qs": -10e-15,
            "qb": 5e-15,
        }


def test_output_contracts_use_isolated_default_artifact_names() -> None:
    assert dataset_filename(
        "tsmc5", "nmos", REDUCED_OUTPUT_CONTRACT,
    ) == "tsmc5_nmos.npz"
    assert dataset_filename(
        "tsmc5", "nmos", FULL_TERMINAL_OUTPUT_CONTRACT,
    ) == "tsmc5_dnf_nmos.npz"
    assert dataset_filename(
        "universal", "pmos", FULL_TERMINAL_OUTPUT_CONTRACT, "v760",
    ) == "universal_v760_dnf_pmos.npz"


def test_full_terminal_cli_defaults_match_level75_artifacts() -> None:
    per_tech = train_cli.argparse.Namespace(
        data=None, tech_scope="tsmc5", device_type="nmos",
        output_contract=FULL_TERMINAL_OUTPUT_CONTRACT,
        exp_name=None, model="direct", size="large",
        loss_preset="default",
    )
    universal = train_cli.argparse.Namespace(
        **{**vars(per_tech), "tech_scope": "universal"},
    )

    assert train_cli._resolve_data_path(per_tech).name == "tsmc5_dnf_nmos.npz"
    assert train_cli._make_save_prefix(per_tech) == "tsmc5_dnf_large_nmos"
    assert train_cli._resolve_data_path(universal).name == (
        "universal_dnf_nmos.npz"
    )
    assert train_cli._make_save_prefix(universal) == "refac_dnf_large_nmos"


def test_full_terminal_assembly_declares_v760_provenance() -> None:
    result = {
        "inputs": np.zeros((1, 4)),
        "geometry": np.zeros((1, 15)),
        "outputs": np.zeros((1, len(FULL_COLUMNS))),
        "sample_class": np.asarray([
            nn_generate.SAMPLE_CLASS_CODES["anchor"],
        ], dtype=np.int8),
        "n_kept": 1,
        "n_failed": 0,
        "n_requested": 1,
        "failure_reasons": [],
    }
    assembled = nn_generate._assemble(
        [result], {"output_contract": "full-terminal"}, verbose=False,
    )
    assert assembled["metadata"]["dataset_variant"] == (
        "v760_full_terminal_core_plus_tg"
    )
    assert assembled["metadata"][
        "externally_appended_sample_class_names"
    ].size == 0


def test_full_terminal_vds_zero_respects_declared_voltage_envelope() -> None:
    points = nn_generate._vds_zero_line_points(
        0.65, is_pmos=True, voltage_box_factor=1.0,
    )
    assert max(abs(vg) for vg, _vd, _vbs in points) == pytest.approx(0.65)
    assert all(vd == 0.0 for _vg, vd, _vbs in points)


def test_generator_emits_solver_positive_independent_surfaces() -> None:
    result, reason = nn_generate._eval_single_point_with_reason(
        _Instance(), 0.2, 0.3, output_contract="full-terminal",
    )
    assert reason == ""
    assert result == {
        "i_d": 0.2,
        "i_g": 0.3,
        "i_b": 0.5,
        "qd": 2e-15,
        "qg": 3e-15,
        "qb": 5e-15,
    }


def test_full_terminal_rejection_checks_every_terminal_current() -> None:
    instance = _Instance()
    instance.eval_dc = lambda _nodes: {
        **_Instance().eval_dc({}), "ig": 2.0,
    }
    result, reason = nn_generate._eval_single_point_with_reason(
        instance, 0.2, 0.3, output_contract="full-terminal",
    )
    assert result is None
    assert reason == "terminal_current_over_1A"


def test_full_terminal_rejects_nonfinite_dependent_terminal() -> None:
    instance = _Instance()
    instance.eval_dc = lambda _nodes: {
        **_Instance().eval_dc({}), "qs": float("nan"),
    }
    result, reason = nn_generate._eval_single_point_with_reason(
        instance, 0.2, 0.3, output_contract="full-terminal",
    )
    assert result is None
    assert reason == "non_finite_output"


def test_loader_uses_dataset_declared_output_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "full.npz"
    outputs = np.arange(48, dtype=float).reshape(8, 6)
    np.savez(
        path,
        inputs=np.zeros((8, 4)),
        geometry=np.column_stack([
            np.repeat([2.0, 3.0, 4.0, 5.0], 2),
            np.repeat([16e-9, 18e-9, 20e-9, 22e-9], 2),
            np.full(8, 300.15),
            np.zeros((8, 12)),
        ]),
        outputs=outputs,
        meta_output_columns=np.asarray(FULL_COLUMNS),
        sample_class=np.zeros(8, dtype=np.int8),
    )
    monkeypatch.setattr(
        "neural_network.eval.loo_labels.get_or_build_tech_variant_labels",
        lambda *_args, **_kwargs: np.zeros(8, dtype=int),
    )

    train, validation, test, normalizer = dataset_module.load_and_split_bsimar(
        str(path), FULL_COLUMNS, "nmos", apply_filter=False,
        norm_mode="zscore", split_mode="random", seed=3,
    )
    assert sum(map(len, (train, validation, test))) == 8
    assert train.outputs.shape[1] == 6
    assert normalizer.stats.output_columns == FULL_COLUMNS

    with pytest.raises(ValueError, match="declared output columns"):
        dataset_module.load_and_split_bsimar(
            str(path), list(reversed(FULL_COLUMNS)), "nmos",
            apply_filter=False, norm_mode="zscore", split_mode="random",
        )


def test_training_cli_routes_full_terminal_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "full.npz"
    data_path.touch()
    observed: dict[str, object] = {}
    monkeypatch.setattr(train_cli, "validate_canonical_dataset", lambda _p: None)
    monkeypatch.setattr(
        train_cli,
        "train_directnet",
        lambda *_args, **kwargs: observed.update(kwargs),
    )
    monkeypatch.setattr(sys, "argv", [
        "train", "--model", "direct", "--size", "small",
        "--device-type", "nmos", "--data", str(data_path),
        "--output-contract", "full-terminal", "--apply-filter", "off",
        "--exp-name", "v760_full_smoke",
    ])

    train_cli.main()

    assert observed["output_columns"] == FULL_COLUMNS
    assert observed["apply_filter"] is False


def test_training_cli_rejects_full_terminal_transformer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "full.npz"
    data_path.touch()
    monkeypatch.setattr(train_cli, "validate_canonical_dataset", lambda _p: None)
    monkeypatch.setattr(sys, "argv", [
        "train", "--model", "transformer", "--data", str(data_path),
        "--output-contract", "full-terminal",
    ])
    with pytest.raises(SystemExit) as exc_info:
        train_cli.main()
    assert exc_info.value.code == 2


def test_full_terminal_training_writes_runtime_complete_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(760)
    data_path = tmp_path / "training.npz"
    n_rows = 40
    np.savez(
        data_path,
        inputs=rng.uniform(-0.5, 0.5, size=(n_rows, 4)),
        geometry=np.column_stack([
            np.full(n_rows, 2.0), np.full(n_rows, 16e-9),
            np.full(n_rows, 300.15), np.zeros((n_rows, 12)),
        ]),
        outputs=np.column_stack([
            rng.normal(scale=1e-4, size=(n_rows, 3)),
            rng.normal(scale=1e-15, size=(n_rows, 3)),
        ]),
        meta_output_columns=np.asarray(FULL_COLUMNS),
        sample_class=np.zeros(n_rows, dtype=np.int8),
    )
    monkeypatch.setattr(
        "neural_network.eval.loo_labels.get_or_build_tech_variant_labels",
        lambda *_args, **_kwargs: np.zeros(n_rows, dtype=int),
    )
    monkeypatch.setattr(trainer, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(trainer, "_NUM_WORKERS", 1)
    torch.manual_seed(760)

    trainer.train_directnet(
        str(data_path),
        config=DirectNetConfig(
            batch_size=16, trunk_hidden=8, trunk_layers=1,
            max_epochs=1, patience=1,
        ),
        save_prefix="dnf_smoke", device_str="cpu", overwrite=True,
        num_tech_codes=2, p_unknown=0.0, apply_filter=False,
        split_mode="random", output_columns=FULL_COLUMNS,
    )

    marker_path = tmp_path / "dnf_smoke_best.pt.complete"
    marker = json.loads(marker_path.read_text())
    assert marker["family"] == "directnet-full"
    assert marker["output_columns"] == FULL_COLUMNS
    assert len(marker["checkpoint_sha256"]) == 64
    assert len(marker["normalization_sha256"]) == 64


def test_full_terminal_training_rejects_legacy_head_options() -> None:
    with pytest.raises(ValueError, match="separate six-surface"):
        trainer.train_directnet(
            "unused.npz",
            output_columns=FULL_COLUMNS,
            output_subset=["i_d"],
            apply_filter=False,
        )
