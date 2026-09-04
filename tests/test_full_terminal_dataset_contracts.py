"""Full-terminal generation, training, and dataset-schema contracts."""

from __future__ import annotations

import sys
import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "bsim_cmg"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from neural_network.data import dataset as dataset_module
from neural_network.cli import train as train_cli
from neural_network.config import DirectNetConfig, TransformerConfig
from neural_network.data.contracts import dataset_filename
from neural_network.losses.bni_mae import SubthresholdIdLoss
from neural_network.training import trainer
from pycmg import nn_generate


FULL_COLUMNS = ["i_d", "i_g", "i_b", "qd", "qg", "qb"]


def _write_dataset_marker(data_path: Path) -> Path:
    """Bind a smoke dataset to the same marker contract as scored data."""
    marker_path = data_path.with_suffix(data_path.suffix + ".complete")
    marker_path.write_text(json.dumps({
        "dataset": data_path.name,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "source_commit": "a" * 40,
        "source_dirty": False,
    }))
    return marker_path


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


def test_full_terminal_datasets_use_the_canonical_artifact_name() -> None:
    assert dataset_filename("tsmc5", "nmos") == "tsmc5_dnf_nmos.npz"
    assert dataset_filename(
        "universal", "pmos", "v760",
    ) == "universal_v760_dnf_pmos.npz"


def test_full_terminal_cli_defaults_match_level75_artifacts() -> None:
    per_tech = train_cli.argparse.Namespace(
        data=None, tech_scope="tsmc5", device_type="nmos",
        exp_name=None, model="direct", size="large",
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


@pytest.mark.parametrize(
    ("tech_name", "is_pmos", "expected_guard"),
    [
        ("tsmc5", False, 1.50),
        ("tsmc16", False, 1.50),
        ("tsmc7", False, 1.20),
        ("tsmc5", True, 1.20),
    ],
)
def test_transmission_gate_corridor_certifies_rail_overshoot_guard(
    tech_name: str,
    is_pmos: bool,
    expected_guard: float,
) -> None:
    """Pass-device data must cover rails plus the transient source guard."""
    vdd = 0.75
    points = nn_generate._tg_corridor_points(
        vdd,
        is_pmos,
        tech_name=tech_name,
    )
    magnitudes = [abs(value) for point in points for value in point]

    assert max(magnitudes) == pytest.approx(expected_guard * vdd)
    assert any(abs(vds) == pytest.approx(vdd) for _vgs, vds, _vbs in points)


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
        _Instance(), 0.2, 0.3,
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
        instance, 0.2, 0.3,
    )
    assert result is None
    assert reason == "terminal_current_over_1A"


def test_full_terminal_rejects_nonfinite_dependent_terminal() -> None:
    instance = _Instance()
    instance.eval_dc = lambda _nodes: {
        **_Instance().eval_dc({}), "qs": float("nan"),
    }
    result, reason = nn_generate._eval_single_point_with_reason(
        instance, 0.2, 0.3,
    )
    assert result is None
    assert reason == "non_finite_output"


def test_internal_node_failure_has_auditable_safety_reason() -> None:
    instance = _Instance()
    instance.eval_dc = lambda _nodes: (_ for _ in ()).throw(
        RuntimeError("Internal node NR failed to converge at d=0.2000")
    )
    result, reason = nn_generate._eval_single_point_with_reason(
        instance, 0.2, 0.3, _silent=True,
    )
    assert result is None
    assert reason == "internal_node_solve_failed"


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
        str(path), "nmos",
        norm_mode="zscore", split_mode="random", seed=3,
    )
    assert sum(map(len, (train, validation, test))) == 8
    assert train.outputs.shape[1] == 6
    assert normalizer.stats.output_columns == FULL_COLUMNS

    bad_path = tmp_path / "bad-columns.npz"
    np.savez(
        bad_path,
        inputs=np.zeros((8, 4)),
        geometry=np.zeros((8, 15)),
        outputs=outputs,
        meta_output_columns=np.asarray(list(reversed(FULL_COLUMNS))),
    )
    with pytest.raises(ValueError, match="declared output columns"):
        dataset_module.load_and_split_bsimar(
            str(bad_path), "nmos", norm_mode="zscore", split_mode="random",
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
        "--exp-name", "v760_full_smoke",
    ])

    train_cli.main()

    assert "output_columns" not in observed
    assert "apply_filter" not in observed


def test_training_cli_routes_full_terminal_transformer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "full.npz"
    data_path.touch()
    observed: dict[str, object] = {}
    monkeypatch.setattr(train_cli, "validate_canonical_dataset", lambda _p: None)
    monkeypatch.setattr(
        train_cli,
        "train_transformer",
        lambda *_args, **kwargs: observed.update(kwargs),
    )
    monkeypatch.setattr(sys, "argv", [
        "train", "--model", "transformer", "--size", "small",
        "--device-type", "nmos", "--data", str(data_path),
        "--full-terminal-ar-targets", "3",
        "--autoregressive-training",
        "--training-overlay-classes", "traj_corridor,hot",
        "--subthresh", "--amp",
        "--exp-name", "v761_full_smoke",
    ])

    train_cli.main()

    assert "output_columns" not in observed
    assert "apply_filter" not in observed
    assert observed["full_terminal_ar_target_dim"] == 3
    assert observed["autoregressive_training"] is True
    assert observed["subthresh"] is True
    assert observed["amp"] is True
    assert observed["training_overlay_classes"] == {
        "traj_corridor", "hot",
    }


@pytest.mark.parametrize(
    "retired_option",
    (
        "--output-contract",
        "--apply-filter",
        "--loss-preset",
        "--sobolev",
        "--charge-sobolev",
        "--monotonic",
        "--ekv-core",
    ),
)
def test_training_cli_rejects_retired_reduced_options(
    retired_option: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["train", retired_option])
    with pytest.raises(SystemExit) as exc_info:
        train_cli.main()
    assert exc_info.value.code == 2


def test_subthreshold_loss_accepts_full_terminal_drain_column() -> None:
    columns = ["qg", "qb", "qd", "i_d", "i_g", "i_b"]
    criterion = SubthresholdIdLoss(column_order=columns)
    x_norm = torch.zeros((2, 7), dtype=torch.float32)
    x_norm[:, 4] = 1.0
    prediction = torch.zeros((2, 6), dtype=torch.float32, requires_grad=True)
    truth = torch.zeros((2, 6), dtype=torch.float32)
    truth[:, 3] = torch.asinh(torch.tensor(1e-8))
    stats = torch.ones(6, dtype=torch.float32)

    loss = criterion(
        x_norm=x_norm,
        y_pred_norm=prediction,
        y_true_norm=truth,
        in_mean=torch.zeros(7),
        in_std=torch.ones(7),
        out_std=stats,
        out_mean=torch.zeros(6),
        asinh_scale=stats,
    )
    loss.backward()

    assert loss > 0.0
    assert prediction.grad is not None
    assert torch.all(torch.isfinite(prediction.grad))


def test_transformer_validation_can_use_autoregressive_runtime_path() -> None:
    """Checkpoint selection must be able to match deployed AR inference."""
    class _ValidationProbe(torch.nn.Module):
        def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor | None = None,
            *,
            tech_codes: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del tech_codes
            return torch.zeros_like(x) if y is None else torch.ones_like(x)

    values = torch.zeros((4, 2), dtype=torch.float32)
    codes = torch.zeros(4, dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(values, values, codes),
        batch_size=2,
    )
    criterion = torch.nn.L1Loss()

    runtime_loss = trainer._epoch_eval(
        _ValidationProbe(), loader, criterion, torch.device("cpu"),
        is_transformer=True, autoregressive=True,
    )
    teacher_forced_loss = trainer._epoch_eval(
        _ValidationProbe(), loader, criterion, torch.device("cpu"),
        is_transformer=True, autoregressive=False,
    )

    assert runtime_loss == 0.0
    assert teacher_forced_loss == 1.0


def test_transformer_training_can_use_autoregressive_runtime_path() -> None:
    """Fine-tuning can condition every head on deployed predictions."""
    class _TrainingProbe(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.25))
            self.seen_autoregressive: list[bool] = []

        def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor | None = None,
            *,
            tech_codes: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del tech_codes
            self.seen_autoregressive.append(y is None)
            return self.bias * torch.ones_like(x)

    values = torch.zeros((4, 2), dtype=torch.float32)
    codes = torch.zeros(4, dtype=torch.long)
    weights = torch.ones_like(values)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(values, values, codes, weights),
        batch_size=2,
    )

    runtime_model = _TrainingProbe()
    trainer._epoch_train(
        runtime_model, loader, trainer.MAELoss(),
        torch.optim.SGD(runtime_model.parameters(), lr=0.01),
        torch.device("cpu"), is_transformer=True,
        autoregressive_training=True,
    )
    teacher_model = _TrainingProbe()
    trainer._epoch_train(
        teacher_model, loader, trainer.MAELoss(),
        torch.optim.SGD(teacher_model.parameters(), lr=0.01),
        torch.device("cpu"), is_transformer=True,
        autoregressive_training=False,
    )

    assert runtime_model.seen_autoregressive == [True, True]
    assert teacher_model.seen_autoregressive == [False, False]


def test_full_terminal_transformer_uses_distinct_checkpoint_stem() -> None:
    args = train_cli.argparse.Namespace(
        data=None, tech_scope="tsmc5", device_type="nmos",
        exp_name=None, model="transformer", size="large",
    )
    assert train_cli._resolve_data_path(args).name == "tsmc5_dnf_nmos.npz"
    assert train_cli._make_save_prefix(args) == "tsmc5_tff_large_nmos"


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
    dataset_marker = _write_dataset_marker(data_path)
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
        num_tech_codes=2, p_unknown=0.0, split_mode="random",
    )

    marker_path = tmp_path / "dnf_smoke_best.pt.complete"
    marker = json.loads(marker_path.read_text())
    assert marker["family"] == "directnet-full"
    assert marker["output_columns"] == FULL_COLUMNS
    assert len(marker["checkpoint_sha256"]) == 64
    assert len(marker["normalization_sha256"]) == 64
    assert marker["dataset"] == data_path.name
    assert marker["dataset_sha256"] == hashlib.sha256(
        data_path.read_bytes()).hexdigest()
    assert marker["dataset_completion_marker"] == dataset_marker.name
    assert marker["dataset_completion_marker_sha256"] == hashlib.sha256(
        dataset_marker.read_bytes()).hexdigest()
    assert marker["dataset_source_commit"] == "a" * 40


def test_full_terminal_transformer_writes_verified_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(761)
    data_path = tmp_path / "training_tff.npz"
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
    _write_dataset_marker(data_path)
    monkeypatch.setattr(
        "neural_network.eval.loo_labels.get_or_build_tech_variant_labels",
        lambda *_args, **_kwargs: np.zeros(n_rows, dtype=int),
    )
    monkeypatch.setattr(trainer, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(trainer, "_NUM_WORKERS", 1)
    torch.manual_seed(761)

    trainer.train_transformer(
        str(data_path),
        config=TransformerConfig(
            batch_size=16, d_model=8, nhead=2, num_layers=1,
            dim_feedforward=16, dropout=0.0, max_epochs=1, patience=1,
        ),
        save_prefix="tff_smoke", device_str="cpu", overwrite=True,
        num_tech_codes=2, p_unknown=0.0, split_mode="random",
        full_terminal_ar_target_dim=3, subthresh=True,
        autoregressive_training=True,
    )

    marker = json.loads(
        (tmp_path / "tff_smoke_best.pt.complete").read_text())
    assert marker["family"] == "bsimar-full"
    assert marker["output_columns"] == FULL_COLUMNS
    assert marker["target_columns"] == [
        "qg", "qb", "qd", "i_d", "i_g", "i_b",
    ]
    assert marker["ar_target_dim"] == 3
    assert marker["training_mode"] == "autoregressive"
    assert len(marker["configuration_sha256"]) == 64
    with np.load(tmp_path / "tff_smoke_config.npz") as config:
        assert config["output_contract"].item() == "full-terminal"
        assert config["ar_target_dim"].item() == 3
        assert config["validation_mode"].item() == "autoregressive"
        assert config["training_mode"].item() == "autoregressive"

    from pycircuitsim.models.mosfet_bsimar_full import NMOS_TFF

    device = NMOS_TFF(
        "Mtrained", ["d", "g", "s", "b"],
        str(tmp_path / "tff_smoke_best.pt"),
        L=16e-9, NFIN=2.0, tech_code=0,
    )
    currents, current_jacobian = device.get_terminal_stamp({
        "d": 0.2, "g": 0.3, "s": 0.05, "b": 0.1,
    })
    assert np.all(np.isfinite(currents))
    assert np.all(np.isfinite(current_jacobian))
    assert np.sum(currents) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(current_jacobian.sum(axis=0), 0.0, atol=1e-12)
