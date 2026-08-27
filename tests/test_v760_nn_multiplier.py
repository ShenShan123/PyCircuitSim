"""Focused regressions for the V7.6.0 NN instance-multiplier contract."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Type

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from neural_network.data.normalize import NormStats, OUTPUT_COLUMN_ORDER
from neural_network.models.direct_net import DirectNet
from neural_network.models.transformer import TransformerEncoderModel
from pycircuitsim.models.mosfet_bsimar import NMOS_BSIMAR, PMOS_BSIMAR
from pycircuitsim.models.mosfet_directnet import NMOS_NN, PMOS_NN
from pycircuitsim.models.mosfet_nn import _MOSFETNNBase
from pycircuitsim.parser import Parser


FAMILIES = (
    (73, NMOS_NN, "dn_nmos"),
    (74, NMOS_BSIMAR, "tf_nmos"),
)
PMOS_FAMILIES = (
    (PMOS_NN, "dn_pmos"),
    (PMOS_BSIMAR, "tf_pmos"),
)
VOLTAGES = {"d": 0.4, "g": 0.5, "s": 0.0, "b": 0.0}
PMOS_VOLTAGES = {"d": 0.0, "g": 0.1, "s": 0.4, "b": 0.4}


def _write_norm(path: Path, mode: str) -> None:
    physical_scales = np.asarray(
        [1e-4] * 4 + [1e-15] * 4 + [1e-15] * 5,
        dtype=np.float64,
    )
    stats = NormStats(
        mode=mode,
        input_mean=np.asarray(
            [0.0, 0.0, 0.0, 0.0, 1.0, 16e-9, 300.15],
            dtype=np.float64,
        ),
        input_std=np.asarray(
            [1.0, 1.0, 1.0, 1.0, 1.0, 10e-9, 100.0],
            dtype=np.float64,
        ),
        input_min=np.asarray(
            [-0.8, -0.8, -0.8, -0.8, 0.0, 1e-9, 200.0],
            dtype=np.float64,
        ),
        input_max=np.asarray(
            [0.8, 0.8, 0.8, 0.8, 8.0, 1e-6, 500.0],
            dtype=np.float64,
        ),
        output_mean=np.zeros(13, dtype=np.float64),
        output_std=(
            physical_scales if mode == "zscore"
            else np.ones(13, dtype=np.float64)
        ),
        asinh_scale=physical_scales if mode == "asinh" else None,
        output_columns=list(OUTPUT_COLUMN_ORDER),
    )
    stats.save(str(path))


def _write_directnet(root: Path, stem: str, current_sign: float) -> Path:
    torch.manual_seed(760)
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
        model.net[-1].bias[0] = current_sign * 10.0
    checkpoint = root / f"{stem}_best.pt"
    torch.save(model.state_dict(), checkpoint)
    _write_norm(root / f"{stem}_norm.npz", "zscore")
    return checkpoint


def _write_bsimar(root: Path, stem: str, current_sign: float) -> Path:
    config = {
        "input_dim": 7,
        "target_dim": 13,
        "d_model": 8,
        "nhead": 2,
        "num_layers": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "num_tech_codes": 2,
    }
    torch.manual_seed(760)
    model = TransformerEncoderModel(**config)
    with torch.no_grad():
        model.output_heads[4].bias.fill_(current_sign * 10.0)
    checkpoint = root / f"{stem}_best.pt"
    torch.save(model.state_dict(), checkpoint)
    np.savez(
        root / f"{stem}_config.npz",
        **{key: np.asarray(value) for key, value in config.items()},
    )
    _write_norm(root / f"{stem}_norm.npz", "asinh")
    return checkpoint


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Path]:
    root = tmp_path_factory.mktemp("v760_nn_multiplier")
    return {
        "dn_nmos": _write_directnet(root, "dn_nmos", -1.0),
        "dn_pmos": _write_directnet(root, "dn_pmos", 1.0),
        "tf_nmos": _write_bsimar(root, "tf_nmos", -1.0),
        "tf_pmos": _write_bsimar(root, "tf_pmos", 1.0),
    }


@pytest.mark.parametrize(("level", "device_type", "checkpoint_key"), FAMILIES)
def test_parser_preserves_nn_instance_multiplier(
    level: int,
    device_type: Type[_MOSFETNNBase],
    checkpoint_key: str,
    checkpoints: Dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = checkpoints[checkpoint_key]
    monkeypatch.setattr(
        "pycircuitsim.parser._resolve_nn_checkpoint",
        lambda **_kwargs: (str(checkpoint), 0),
    )
    parser = Parser()
    parser.parse_line(
        f".model nmos_nn NMOS (LEVEL={level} TECH=tsmc5 VT=svt)"
    )
    parser.parse_line("M1 d g s b nmos_nn L=16n NFIN=2 m=4")

    device = parser.circuit.components[-1]
    assert isinstance(device, device_type)
    assert device.m == 4.0


@pytest.mark.parametrize(("_level", "device_type", "checkpoint_key"), FAMILIES)
def test_nn_multiplier_scales_only_solver_facing_outputs(
    _level: int,
    device_type: Type[_MOSFETNNBase],
    checkpoint_key: str,
    checkpoints: Dict[str, Path],
) -> None:
    checkpoint = checkpoints[checkpoint_key]
    reference = device_type(
        "Mref", ["d", "g", "s", "b"], str(checkpoint),
        L=16e-9, NFIN=2.0, tech_code=0,
    )
    multiplied = device_type(
        "M4", ["d", "g", "s", "b"], str(checkpoint),
        L=16e-9, NFIN=2.0, tech_code=0, multiplier=4.0,
    )

    reference.get_capacitances(VOLTAGES)
    multiplied_caps = multiplied.get_capacitances(VOLTAGES)
    raw_reference = reference._eval(VOLTAGES).copy()
    raw_multiplied = multiplied._eval(VOLTAGES).copy()

    assert raw_multiplied == raw_reference
    assert multiplied.calculate_current(VOLTAGES) == -4.0 * raw_multiplied["id"]
    assert multiplied.get_conductance(VOLTAGES) == tuple(
        4.0 * raw_multiplied[key] for key in ("gds", "gm", "gmb")
    )
    assert multiplied.get_charges(VOLTAGES) == {
        key: 4.0 * raw_multiplied[key] for key in ("qg", "qd", "qs", "qb")
    }
    assert multiplied_caps == {
        key: 4.0 * raw_multiplied[key]
        for key in ("cgg", "cgd", "cgs", "cdg", "cdd")
    }
    assert multiplied._eval_cache == raw_multiplied


@pytest.mark.parametrize(("device_type", "checkpoint_key"), PMOS_FAMILIES)
def test_pmos_current_obeys_nn_multiplier(
    device_type: Type[_MOSFETNNBase],
    checkpoint_key: str,
    checkpoints: Dict[str, Path],
) -> None:
    checkpoint = checkpoints[checkpoint_key]
    device = device_type(
        "M4", ["d", "g", "s", "b"], str(checkpoint),
        L=16e-9, NFIN=2.0, tech_code=0, multiplier=4.0,
    )
    raw = device._eval(PMOS_VOLTAGES)["id"]

    assert raw != 0.0
    assert device.calculate_current(PMOS_VOLTAGES) == 4.0 * raw


@pytest.mark.parametrize(("_level", "device_type", "checkpoint_key"), FAMILIES)
@pytest.mark.parametrize("multiplier", (0.0, -1.0))
def test_nn_multiplier_must_be_positive(
    _level: int,
    device_type: Type[_MOSFETNNBase],
    checkpoint_key: str,
    checkpoints: Dict[str, Path],
    multiplier: float,
) -> None:
    checkpoint = checkpoints[checkpoint_key]
    with pytest.raises(ValueError, match="multiplier m must be positive"):
        device_type(
            "Mbad", ["d", "g", "s", "b"], str(checkpoint),
            L=16e-9, NFIN=2.0, tech_code=0, multiplier=multiplier,
        )
