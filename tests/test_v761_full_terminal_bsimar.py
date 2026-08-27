"""V7.6.1 contracts for the full-terminal BSIM-AR family."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from neural_network.data.normalize import NormStats
from neural_network.models.transformer import TransformerEncoderModel
from pycircuitsim.parser import Parser
from pycircuitsim.solver import _full_charge_stamp, _full_current_stamp, _is_mosfet


FULL_COLUMNS = ["i_d", "i_g", "i_b", "qd", "qg", "qb"]
TFF_COLUMNS = ["qg", "qb", "qd", "i_d", "i_g", "i_b"]
VOLTAGES = {"d": 0.20, "g": 0.30, "s": 0.05, "b": 0.10}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checkpoint(root: Path, stem: str = "full") -> Path:
    model = TransformerEncoderModel(
        input_dim=7,
        target_dim=len(TFF_COLUMNS),
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        num_tech_codes=2,
        tech_embed_dropout=0.0,
        unknown_code_id=1,
        ar_target_dim=len(TFF_COLUMNS),
    )
    checkpoint = root / f"{stem}_best.pt"
    torch.save(model.state_dict(), checkpoint)
    norm_path = root / f"{stem}_norm.npz"
    NormStats(
        mode="asinh",
        input_mean=np.zeros(7),
        input_std=np.ones(7),
        input_min=np.asarray([-1.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0]),
        input_max=np.asarray([1.0, 1.0, 0.0, 1.0, 8.0, 1e-6, 500.0]),
        output_mean=np.zeros(len(FULL_COLUMNS)),
        output_std=np.ones(len(FULL_COLUMNS)),
        asinh_scale=np.ones(len(FULL_COLUMNS)),
        output_columns=FULL_COLUMNS,
    ).save(str(norm_path))
    config_path = root / f"{stem}_config.npz"
    np.savez(
        config_path,
        input_dim=7,
        target_dim=len(TFF_COLUMNS),
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        num_tech_codes=2,
        ar_target_dim=len(TFF_COLUMNS),
        output_contract="full-terminal",
        target_columns=np.asarray(TFF_COLUMNS),
    )
    (root / f"{stem}_best.pt.complete").write_text(json.dumps({
        "family": "bsimar-full",
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint),
        "normalization": norm_path.name,
        "normalization_sha256": _sha256(norm_path),
        "configuration": config_path.name,
        "configuration_sha256": _sha256(config_path),
        "output_columns": FULL_COLUMNS,
        "target_columns": TFF_COLUMNS,
    }))
    return checkpoint


@pytest.fixture()
def checkpoint(tmp_path: Path) -> Path:
    return _write_checkpoint(tmp_path)


def _linear_surfaces(self: object, x: torch.Tensor) -> torch.Tensor:
    vd, vg, _vs, vb = (x[:, i] for i in range(4))
    values = {
        "i_d": 2.0 * vd + 3.0 * vg + 5.0 * vb + 0.1,
        "i_g": -vd + 4.0 * vg + 2.0 * vb + 0.2,
        "i_b": 0.5 * vd - 2.0 * vg + 6.0 * vb - 0.3,
        "qd": 7e-15 * vd + 2e-15 * vg + 1e-15 * vb,
        "qg": -3e-15 * vd + 8e-15 * vg + 4e-15 * vb,
        "qb": 1e-15 * vd - 5e-15 * vg + 9e-15 * vb,
    }
    return torch.stack([
        torch.asinh(values[name]) for name in TFF_COLUMNS
    ], dim=1)


def test_six_target_transformer_preserves_legacy_default() -> None:
    legacy = TransformerEncoderModel(
        d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.0,
    )
    full = TransformerEncoderModel(
        target_dim=6, ar_target_dim=6,
        d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.0,
    )
    x = torch.zeros((2, 7))
    codes = torch.zeros(2, dtype=torch.long)
    assert legacy(x, tech_codes=codes).shape == (2, 13)
    assert full(x, tech_codes=codes).shape == (2, 6)
    assert full(x, torch.zeros((2, 6)), tech_codes=codes).shape == (2, 6)


def test_bsimar_full_reconstructs_closed_terminal_stamps(
    checkpoint: Path,
) -> None:
    from pycircuitsim.models.mosfet_bsimar_full import NMOS_TFF

    device = NMOS_TFF(
        "M1", ["d", "g", "s", "b"], str(checkpoint),
        L=16e-9, NFIN=2.0, tech_code=0, multiplier=2.0,
    )
    device._forward_model = MethodType(_linear_surfaces, device)
    currents, current_jacobian = device.get_terminal_stamp(VOLTAGES)
    charges, charge_jacobian = device.get_charge_stamp(VOLTAGES)

    assert np.sum(currents) == pytest.approx(0.0, abs=1e-14)
    assert np.sum(charges) == pytest.approx(0.0, abs=1e-28)
    np.testing.assert_allclose(current_jacobian.sum(axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(current_jacobian.sum(axis=1), 0.0, atol=1e-14)
    np.testing.assert_allclose(charge_jacobian.sum(axis=0), 0.0, atol=1e-28)
    np.testing.assert_allclose(charge_jacobian.sum(axis=1), 0.0, atol=1e-28)
    np.testing.assert_allclose(
        current_jacobian[0], [4.0, 6.0, -20.0, 10.0], atol=1e-6,
    )
    np.testing.assert_allclose(
        charge_jacobian[1], [-6e-15, 16e-15, -18e-15, 8e-15], atol=1e-20,
    )
    assert _is_mosfet(device)
    assert _full_current_stamp(device) is not None
    assert _full_charge_stamp(device) is not None


def test_level76_requires_explicit_family_and_force_retargets(
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pycircuitsim.models.mosfet_bsimar_full import NMOS_TFF

    monkeypatch.setattr(
        "pycircuitsim.parser._resolve_nn_checkpoint",
        lambda **_kwargs: (str(checkpoint), 0),
    )
    parser = Parser()
    parser.parse_line(
        ".model full NMOS (LEVEL=76 FAMILY=bsimar-full TECH=tsmc5 VT=svt)"
    )
    parser.parse_line("M1 d g s b full L=16n NFIN=2")
    assert isinstance(parser.circuit.components[-1], NMOS_TFF)

    missing_family = Parser()
    missing_family.parse_line(".model bad NMOS (LEVEL=76 TECH=tsmc5 VT=svt)")
    with pytest.raises(ValueError, match="FAMILY=bsimar-full"):
        missing_family.parse_line("M1 d g s b bad L=16n NFIN=2")

    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "76")
    forced = Parser()
    forced.parse_line(".model old NMOS (LEVEL=73 TECH=tsmc5 VT=svt)")
    forced.parse_line("M1 d g s b old L=16n NFIN=2")
    assert isinstance(forced.circuit.components[-1], NMOS_TFF)


@pytest.mark.parametrize("artifact", ["best.pt", "norm.npz", "config.npz"])
def test_bsimar_full_rejects_artifact_checksum_mutation(
    tmp_path: Path,
    artifact: str,
) -> None:
    from pycircuitsim.models.mosfet_bsimar_full import NMOS_TFF

    checkpoint = _write_checkpoint(tmp_path, "bad")
    artifact_path = tmp_path / f"bad_{artifact}"
    artifact_path.write_bytes(artifact_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        NMOS_TFF(
            "Mbad", ["d", "g", "s", "b"], str(checkpoint),
            L=16e-9, NFIN=2.0, tech_code=0,
        )
