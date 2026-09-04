"""Shared contracts for the DirectNet-Full and BSIM-AR-Full families."""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from neural_network.data.normalize import NormStats
from neural_network.models.direct_net import DirectNet
from neural_network.models.transformer import (
    TransformerEncoderModel,
)

from pycircuitsim.circuit import Circuit
from pycircuitsim.models.mosfet_bsimar_full import (
    NMOS_TFF,
    PMOS_TFF,
)
from pycircuitsim.models.mosfet_directnet_full import (
    NMOS_DNF,
    PMOS_DNF,
)
from pycircuitsim.parser import Parser
from pycircuitsim.solver import (
    _full_charge_stamp,
    _full_current_stamp,
    _is_mosfet,
    _require_nn_caps,
)

FULL_COLUMNS = ["i_d", "i_g", "i_b", "qd", "qg", "qb"]
TFF_COLUMNS = ["qg", "qb", "qd", "i_d", "i_g", "i_b"]
VOLTAGES = {"d": 0.20, "g": 0.30, "s": 0.05, "b": 0.10}


@dataclass(frozen=True)
class FullTerminalFamily:
    """The facts that vary across the two full-terminal adapters."""

    slug: str
    level: int
    family_parameter: str
    nmos_type: type[Any]
    pmos_type: type[Any]
    checksum_artifacts: tuple[str, ...]


DIRECTNET_FULL = FullTerminalFamily(
    "directnet-full",
    75,
    "directnet-full",
    NMOS_DNF,
    PMOS_DNF,
    ("best.pt", "norm.npz"),
)
BSIMAR_FULL = FullTerminalFamily(
    "bsimar-full",
    76,
    "bsimar-full",
    NMOS_TFF,
    PMOS_TFF,
    ("best.pt", "norm.npz", "config.npz"),
)
FAMILIES = (DIRECTNET_FULL, BSIMAR_FULL)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checkpoint(
    root: Path,
    family: FullTerminalFamily,
    stem: str = "full",
) -> Path:
    if family is DIRECTNET_FULL:
        model: torch.nn.Module = DirectNet(
            input_dim=7,
            hidden_dim=8,
            n_layers=1,
            output_dim=len(FULL_COLUMNS),
            num_tech_codes=2,
            tech_embed_dim=2,
            tech_embed_dropout=0.0,
            unknown_code_id=1,
        )
        norm_mode = "zscore"
    else:
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
        norm_mode = "asinh"

    checkpoint = root / f"{stem}_best.pt"
    torch.save(model.state_dict(), checkpoint)
    norm_path = root / f"{stem}_norm.npz"
    norm_kwargs: dict[str, Any] = {}
    if family is BSIMAR_FULL:
        norm_kwargs["asinh_scale"] = np.ones(len(FULL_COLUMNS))
    NormStats(
        mode=norm_mode,
        input_mean=np.zeros(7),
        input_std=np.ones(7),
        input_min=np.asarray([-1.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0]),
        input_max=np.asarray([1.0, 1.0, 0.0, 1.0, 8.0, 1e-6, 500.0]),
        output_mean=np.zeros(len(FULL_COLUMNS)),
        output_std=np.ones(len(FULL_COLUMNS)),
        output_columns=FULL_COLUMNS,
        **norm_kwargs,
    ).save(str(norm_path))

    marker: dict[str, Any] = {
        "family": family.family_parameter,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint),
        "normalization": norm_path.name,
        "normalization_sha256": _sha256(norm_path),
        "output_columns": FULL_COLUMNS,
    }
    if family is BSIMAR_FULL:
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
        marker.update({
            "configuration": config_path.name,
            "configuration_sha256": _sha256(config_path),
            "target_columns": TFF_COLUMNS,
            "ar_target_dim": len(TFF_COLUMNS),
        })
    (root / f"{stem}_best.pt.complete").write_text(json.dumps(marker))
    return checkpoint


@pytest.fixture(params=FAMILIES, ids=lambda family: family.slug)
def family(request: pytest.FixtureRequest) -> FullTerminalFamily:
    return request.param


@pytest.fixture()
def checkpoint(tmp_path: Path, family: FullTerminalFamily) -> Path:
    return _write_checkpoint(tmp_path, family)


def _surface_values(x: torch.Tensor) -> dict[str, torch.Tensor]:
    vd, vg, _vs, vb = (x[:, index] for index in range(4))
    return {
        "i_d": 2.0 * vd + 3.0 * vg + 5.0 * vb + 0.1,
        "i_g": -vd + 4.0 * vg + 2.0 * vb + 0.2,
        "i_b": 0.5 * vd - 2.0 * vg + 6.0 * vb - 0.3,
        "qd": 7e-15 * vd + 2e-15 * vg + 1e-15 * vb,
        "qg": -3e-15 * vd + 8e-15 * vg + 4e-15 * vb,
        "qb": 1e-15 * vd - 5e-15 * vg + 9e-15 * vb,
    }


def _directnet_surfaces(self: object, x: torch.Tensor) -> torch.Tensor:
    del self
    values = _surface_values(x)
    return torch.stack([values[name] for name in FULL_COLUMNS], dim=1)


def _bsimar_surfaces(self: object, x: torch.Tensor) -> torch.Tensor:
    del self
    values = _surface_values(x)
    return torch.stack([torch.asinh(values[name]) for name in TFF_COLUMNS], dim=1)


def _bind_linear(device: Any, family: FullTerminalFamily) -> None:
    implementation = (
        _directnet_surfaces if family is DIRECTNET_FULL else _bsimar_surfaces
    )
    device._forward_model = MethodType(implementation, device)


def test_full_terminal_families_reconstruct_closed_terminal_stamps(
    family: FullTerminalFamily,
    checkpoint: Path,
) -> None:
    device = family.nmos_type(
        "M1", ["d", "g", "s", "b"], str(checkpoint),
        L=16e-9, NFIN=2.0, tech_code=0, multiplier=2.0,
    )
    _bind_linear(device, family)

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


@pytest.mark.parametrize("polarity", ("nmos", "pmos"))
def test_scalar_current_sign_does_not_mutate_full_terminal_stamp(
    family: FullTerminalFamily,
    checkpoint: Path,
    polarity: str,
) -> None:
    device_type = family.nmos_type if polarity == "nmos" else family.pmos_type
    device = device_type(
        "M1", ["d", "g", "s", "b"], str(checkpoint),
        L=16e-9, NFIN=2.0, tech_code=0, multiplier=2.0,
    )
    _bind_linear(device, family)
    currents_before, jacobian_before = device.get_terminal_stamp(VOLTAGES)
    expected = currents_before[0] if polarity == "nmos" else -currents_before[0]

    assert device.calculate_current(VOLTAGES) == pytest.approx(expected)
    currents_after, jacobian_after = device.get_terminal_stamp(VOLTAGES)
    np.testing.assert_array_equal(currents_after, currents_before)
    np.testing.assert_array_equal(jacobian_after, jacobian_before)


def test_full_terminal_families_reject_outside_certified_support(
    family: FullTerminalFamily,
    checkpoint: Path,
) -> None:
    device = family.nmos_type(
        "M1", ["d", "g", "s", "b"], str(checkpoint),
        L=16e-9, NFIN=2.0, tech_code=0,
    )
    with pytest.raises(ValueError, match="certified support"):
        device.get_terminal_stamp({**VOLTAGES, "d": 2.0})


def test_full_terminal_devices_share_one_verified_artifact_load(
    family: FullTerminalFamily,
    checkpoint: Path,
) -> None:
    first = family.nmos_type(
        "M1", ["d1", "g", "s", "b"], str(checkpoint),
        L=16e-9, NFIN=2.0, tech_code=0,
    )
    second = family.nmos_type(
        "M2", ["d2", "g", "s", "b"], str(checkpoint),
        L=16e-9, NFIN=4.0, tech_code=0,
    )
    assert first._nn_model is second._nn_model
    assert first._norm_stats is second._norm_stats


CHECKSUM_CASES = tuple(
    (family, artifact)
    for family in FAMILIES
    for artifact in family.checksum_artifacts
)


@pytest.mark.parametrize(
    ("family", "artifact"),
    CHECKSUM_CASES,
    ids=lambda value: value.slug if isinstance(value, FullTerminalFamily) else value,
)
def test_full_terminal_families_reject_artifact_checksum_mutation(
    tmp_path: Path,
    family: FullTerminalFamily,
    artifact: str,
) -> None:
    checkpoint = _write_checkpoint(tmp_path, family, "bad")
    target = tmp_path / f"bad_{artifact}"
    target.write_bytes(target.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        family.nmos_type(
            "Mbad", ["d", "g", "s", "b"], str(checkpoint),
            L=16e-9, NFIN=2.0, tech_code=0,
        )


def test_full_terminal_level_requires_explicit_family_and_force_retargets(
    family: FullTerminalFamily,
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "pycircuitsim.parser._resolve_nn_checkpoint",
        lambda **_kwargs: (str(checkpoint), 0),
    )
    parser = Parser()
    parser.parse_line(
        f".model full NMOS (LEVEL={family.level} "
        f"FAMILY={family.family_parameter} TECH=tsmc5 VT=svt)"
    )
    parser.parse_line("M1 d g s b full L=16n NFIN=2 m=3")
    device = parser.circuit.components[-1]
    assert isinstance(device, family.nmos_type)
    assert device.m == 3.0

    missing_family = Parser()
    missing_family.parse_line(
        f".model bad NMOS (LEVEL={family.level} TECH=tsmc5 VT=svt)"
    )
    with pytest.raises(ValueError, match=f"FAMILY={family.family_parameter}"):
        missing_family.parse_line("M1 d g s b bad L=16n NFIN=2")

    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", str(family.level))
    forced = Parser()
    forced.parse_line(".model old NMOS (LEVEL=73 TECH=tsmc5 VT=svt)")
    forced.parse_line("M1 d g s b old L=16n NFIN=2")
    assert isinstance(forced.circuit.components[-1], family.nmos_type)


def test_full_terminal_family_requires_completion_marker(
    family: FullTerminalFamily,
    checkpoint: Path,
) -> None:
    checkpoint.with_suffix(".pt.complete").unlink()
    with pytest.raises(FileNotFoundError, match="completion marker"):
        family.nmos_type(
            "Mbad", ["d", "g", "s", "b"], str(checkpoint),
            L=16e-9, NFIN=2.0, tech_code=0,
        )


def test_netlist_temperature_rebinds_both_full_terminal_families(
    family: FullTerminalFamily,
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "pycircuitsim.parser._resolve_nn_checkpoint",
        lambda **_kwargs: (str(checkpoint), 0),
    )
    parser = Parser()
    parser.parse_line(".temp 125")
    parser.parse_line(
        f".model full NMOS (LEVEL={family.level} "
        f"FAMILY={family.family_parameter} TECH=tsmc5 VT=svt)"
    )
    parser.parse_line("M1 d g s b full L=16n NFIN=2")
    device = parser.circuit.components[-1]
    assert device.temperature == 398.15

    device._cache_voltages = (0.1, 0.2, 0.0, 0.0)
    device._eval_cache = (np.ones(4), np.ones((4, 4)))
    parser.parse_line(".temp -25")
    assert device.temperature == 248.15
    assert device._cache_voltages is None
    assert device._eval_cache is None


def test_directnet_full_charge_jacobians_are_lazy_and_self_healing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _write_checkpoint(tmp_path, DIRECTNET_FULL)
    device = NMOS_DNF(
        "M1", ["d", "g", "s", "b"], str(checkpoint),
        L=16e-9, NFIN=2.0, tech_code=0,
    )
    _bind_linear(device, DIRECTNET_FULL)
    original_grad = torch.autograd.grad
    grad_calls = 0

    def _counted_grad(*args: object, **kwargs: object) -> object:
        nonlocal grad_calls
        grad_calls += 1
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", _counted_grad)
    device.get_terminal_stamp(VOLTAGES)
    assert grad_calls == 3
    device.get_charge_stamp(VOLTAGES)
    assert grad_calls == 9
    device.get_charge_stamp(VOLTAGES)
    assert grad_calls == 9

    second = NMOS_DNF(
        "M2", ["d", "g", "s", "b"], str(checkpoint),
        L=16e-9, NFIN=2.0, tech_code=0,
    )
    _bind_linear(second, DIRECTNET_FULL)
    circuit = Circuit()
    circuit.add_component(second)
    _require_nn_caps(circuit)
    second.get_terminal_stamp(VOLTAGES)
    assert grad_calls == 15


def test_six_target_transformer_preserves_reduced_family_default() -> None:
    legacy = TransformerEncoderModel(
        d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.0,
    )
    full = TransformerEncoderModel(
        target_dim=6,
        ar_target_dim=6,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        dropout=0.0,
    )
    inputs = torch.zeros((2, 7))
    codes = torch.zeros(2, dtype=torch.long)
    assert legacy(inputs, tech_codes=codes).shape == (2, 13)
    assert full(inputs, tech_codes=codes).shape == (2, 6)
    assert full(inputs, torch.zeros((2, 6)), tech_codes=codes).shape == (2, 6)
