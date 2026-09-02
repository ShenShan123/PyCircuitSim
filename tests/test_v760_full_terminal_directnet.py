"""V7.6.0 contracts for the separate full-terminal DirectNet family."""

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
from neural_network.models.direct_net import DirectNet
from pycircuitsim.parser import Parser
from pycircuitsim.circuit import Circuit
from pycircuitsim.solver import (
    _full_charge_stamp,
    _full_current_stamp,
    _is_mosfet,
    _require_nn_caps,
)


FULL_COLUMNS = ["i_d", "i_g", "i_b", "qd", "qg", "qb"]
VOLTAGES = {"d": 0.20, "g": 0.30, "s": 0.05, "b": 0.10}


def _write_checkpoint(root: Path, stem: str = "full") -> Path:
    model = DirectNet(
        input_dim=7,
        hidden_dim=8,
        n_layers=1,
        output_dim=len(FULL_COLUMNS),
        num_tech_codes=2,
        tech_embed_dim=2,
        tech_embed_dropout=0.0,
        unknown_code_id=1,
    )
    checkpoint = root / f"{stem}_best.pt"
    torch.save(model.state_dict(), checkpoint)
    norm_path = root / f"{stem}_norm.npz"
    NormStats(
        mode="zscore",
        input_mean=np.zeros(7),
        input_std=np.ones(7),
        input_min=np.asarray([-1.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0]),
        input_max=np.asarray([1.0, 1.0, 0.0, 1.0, 8.0, 1e-6, 500.0]),
        output_mean=np.zeros(len(FULL_COLUMNS)),
        output_std=np.ones(len(FULL_COLUMNS)),
        output_columns=FULL_COLUMNS,
    ).save(str(norm_path))
    sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (root / f"{stem}_best.pt.complete").write_text(json.dumps({
        "family": "directnet-full",
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "normalization": norm_path.name,
        "normalization_sha256": sha256(norm_path),
        "output_columns": FULL_COLUMNS,
    }))
    return checkpoint


@pytest.fixture()
def checkpoint(tmp_path: Path) -> Path:
    return _write_checkpoint(tmp_path)


def _linear_surfaces(self: object, x: torch.Tensor) -> torch.Tensor:
    """Three independent currents and charges in normalized coordinates."""
    vd, vg, _vs, vb = (x[:, i] for i in range(4))
    return torch.stack([
        2.0 * vd + 3.0 * vg + 5.0 * vb + 0.1,
        -vd + 4.0 * vg + 2.0 * vb + 0.2,
        0.5 * vd - 2.0 * vg + 6.0 * vb - 0.3,
        7e-15 * vd + 2e-15 * vg + 1e-15 * vb,
        -3e-15 * vd + 8e-15 * vg + 4e-15 * vb,
        1e-15 * vd - 5e-15 * vg + 9e-15 * vb,
    ], dim=1)


def test_full_terminal_surfaces_reconstruct_exact_closure(
    checkpoint: Path,
) -> None:
    from pycircuitsim.models.mosfet_directnet_full import NMOS_DNF

    device = NMOS_DNF(
        "M2", ["d", "g", "s", "b"], str(checkpoint),
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


def test_full_terminal_scalar_current_adapts_only_pmos_boundary(
    checkpoint: Path,
) -> None:
    """Scalar comparison signs must not alter full-terminal solver stamps."""
    from pycircuitsim.models.mosfet_directnet_full import NMOS_DNF, PMOS_DNF

    nmos = NMOS_DNF(
        "MN", ["d", "g", "s", "b"], str(checkpoint),
        L=16e-9, NFIN=2.0, tech_code=0, multiplier=2.0,
    )
    pmos = PMOS_DNF(
        "MP", ["d", "g", "s", "b"], str(checkpoint),
        L=16e-9, NFIN=2.0, tech_code=0, multiplier=2.0,
    )
    nmos._forward_model = MethodType(_linear_surfaces, nmos)
    pmos._forward_model = MethodType(_linear_surfaces, pmos)

    nmos_stamp_before = nmos.get_terminal_stamp(VOLTAGES)
    pmos_stamp_before = pmos.get_terminal_stamp(VOLTAGES)

    assert nmos.calculate_current(VOLTAGES) == pytest.approx(
        nmos_stamp_before[0][0])
    assert pmos.calculate_current(VOLTAGES) == pytest.approx(
        -pmos_stamp_before[0][0])

    nmos_stamp_after = nmos.get_terminal_stamp(VOLTAGES)
    pmos_stamp_after = pmos.get_terminal_stamp(VOLTAGES)
    np.testing.assert_array_equal(nmos_stamp_after[0], nmos_stamp_before[0])
    np.testing.assert_array_equal(nmos_stamp_after[1], nmos_stamp_before[1])
    np.testing.assert_array_equal(pmos_stamp_after[0], pmos_stamp_before[0])
    np.testing.assert_array_equal(pmos_stamp_after[1], pmos_stamp_before[1])


def test_full_terminal_family_rejects_outside_certified_support(
    checkpoint: Path,
) -> None:
    from pycircuitsim.models.mosfet_directnet_full import NMOS_DNF

    device = NMOS_DNF(
        "M1", ["d", "g", "s", "b"], str(checkpoint),
        L=16e-9, NFIN=2.0, tech_code=0,
    )
    with pytest.raises(ValueError, match="certified support"):
        device.get_terminal_stamp({**VOLTAGES, "d": 2.0})


def test_full_terminal_charge_jacobians_are_lazy_and_self_healing(
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pycircuitsim.models.mosfet_directnet_full import NMOS_DNF

    device = NMOS_DNF(
        "M1", ["d", "g", "s", "b"], str(checkpoint),
        L=16e-9, NFIN=2.0, tech_code=0,
    )
    device._forward_model = MethodType(_linear_surfaces, device)
    original_grad = torch.autograd.grad
    grad_calls = 0

    def _counted_grad(*args: object, **kwargs: object) -> object:
        nonlocal grad_calls
        grad_calls += 1
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", _counted_grad)
    device.get_terminal_stamp(VOLTAGES)
    assert grad_calls == 3

    # A direct charge consumer self-heals a missed solver declaration and
    # replaces the current-only cache with one carrying all six derivatives.
    device.get_charge_stamp(VOLTAGES)
    assert grad_calls == 9
    device.get_charge_stamp(VOLTAGES)
    assert grad_calls == 9

    second = NMOS_DNF(
        "M2", ["d", "g", "s", "b"], str(checkpoint),
        L=16e-9, NFIN=2.0, tech_code=0,
    )
    second._forward_model = MethodType(_linear_surfaces, second)
    circuit = Circuit()
    circuit.add_component(second)
    _require_nn_caps(circuit)
    second.get_terminal_stamp(VOLTAGES)
    assert grad_calls == 15


def test_full_terminal_devices_share_one_verified_artifact_load(
    checkpoint: Path,
) -> None:
    """Large circuits must not deserialize identical weights per instance."""
    from pycircuitsim.models.mosfet_directnet_full import NMOS_DNF

    first = NMOS_DNF(
        "M1", ["d1", "g", "s", "b"], str(checkpoint),
        L=16e-9, NFIN=2.0, tech_code=0,
    )
    second = NMOS_DNF(
        "M2", ["d2", "g", "s", "b"], str(checkpoint),
        L=16e-9, NFIN=4.0, tech_code=0,
    )

    assert first._nn_model is second._nn_model
    assert first._norm_stats is second._norm_stats


@pytest.mark.parametrize("artifact", ["checkpoint", "normalization"])
def test_full_terminal_rejects_completed_artifact_checksum_mutation(
    tmp_path: Path,
    artifact: str,
) -> None:
    from pycircuitsim.models.mosfet_directnet_full import NMOS_DNF

    checkpoint = _write_checkpoint(tmp_path, artifact)
    target = (
        checkpoint if artifact == "checkpoint"
        else tmp_path / f"{artifact}_norm.npz")
    target.write_bytes(target.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        NMOS_DNF(
            "Mbad", ["d", "g", "s", "b"], str(checkpoint),
            L=16e-9, NFIN=2.0, tech_code=0,
        )


def test_level75_requires_explicit_new_family_and_complete_artifacts(
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pycircuitsim.models.mosfet_directnet_full import NMOS_DNF

    monkeypatch.setattr(
        "pycircuitsim.parser._resolve_nn_checkpoint",
        lambda **_kwargs: (str(checkpoint), 0),
    )
    parser = Parser()
    parser.parse_line(
        ".model full NMOS (LEVEL=75 FAMILY=directnet-full "
        "TECH=tsmc5 VT=svt)"
    )
    parser.parse_line("M1 d g s b full L=16n NFIN=2 m=3")
    device = parser.circuit.components[-1]
    assert isinstance(device, NMOS_DNF)
    assert device.m == 3.0

    legacy = Parser()
    legacy.parse_line(".model retired NMOS (LEVEL=75 TECH=tsmc5 VT=svt)")
    with pytest.raises(ValueError, match="FAMILY=directnet-full"):
        legacy.parse_line("M1 d g s b retired L=16n NFIN=2")

    checkpoint.with_suffix(".pt.complete").unlink()
    with pytest.raises(FileNotFoundError, match="completion marker"):
        NMOS_DNF(
            "Mbad", ["d", "g", "s", "b"], str(checkpoint),
            L=16e-9, NFIN=2.0, tech_code=0,
        )


def test_force_level75_retargets_existing_directnet_decks(
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Campaign retargeting must select the new family without deck copies."""
    from pycircuitsim.models.mosfet_directnet_full import NMOS_DNF

    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "75")
    monkeypatch.setattr(
        "pycircuitsim.parser._resolve_nn_checkpoint",
        lambda **_kwargs: (str(checkpoint), 0),
    )
    parser = Parser()
    parser.parse_line(
        ".model original NMOS (LEVEL=73 TECH=tsmc5 VT=svt)"
    )
    parser.parse_line("M1 d g s b original L=16n NFIN=2")
    assert isinstance(parser.circuit.components[-1], NMOS_DNF)
