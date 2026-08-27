"""V7.6.0 integration contracts for full-terminal AnalogGym campaigns."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from examples.complex_circuits.pycircuitsim_bench import campaign
from examples.complex_circuits.pycircuitsim_bench import run_compare
from examples.complex_circuits.pycircuitsim_bench import translate
from tests.common import complex as complex_common


def test_level75_model_stub_is_explicit() -> None:
    assert translate._model_stub(
        "nsvt_l32_f7", "NMOS", tech="tsmc5", model_level=75,
    ) == (
        ".model nsvt_l32_f7 NMOS "
        "(LEVEL=75 FAMILY=directnet-full TECH=tsmc5 VT=svt)"
    )


def test_level76_model_stub_is_explicit() -> None:
    assert translate._model_stub(
        "nsvt_l32_f7", "NMOS", tech="tsmc5", model_level=76,
    ) == (
        ".model nsvt_l32_f7 NMOS "
        "(LEVEL=76 FAMILY=bsimar-full TECH=tsmc5 VT=svt)"
    )


def test_level75_provenance_uses_family_specific_pins(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    from neural_network import config as nn_config

    for device in ("nmos", "pmos"):
        stem = f"tsmc5_dnf_large_{device}"
        checkpoint = tmp_path / f"{stem}_best.pt"
        norm = tmp_path / f"{stem}_norm.npz"
        marker = checkpoint.with_suffix(".pt.complete")
        checkpoint.write_bytes(f"weights-{device}".encode())
        norm.write_bytes(f"norm-{device}".encode())
        marker.write_text(json.dumps({"family": "directnet-full"}))
    monkeypatch.setattr(nn_config, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setenv(
        "PYCIRCUITSIM_NN_CHECKPOINT_DNF_NMOS", "tsmc5_dnf_large_nmos",
    )
    monkeypatch.setenv(
        "PYCIRCUITSIM_NN_CHECKPOINT_DNF_PMOS", "tsmc5_dnf_large_pmos",
    )

    record = run_compare._model_provenance(
        SimpleNamespace(model_level=75, tech="tsmc5"),
    )

    assert record["family"] == "directnet_full"
    assert record["checkpoints"]["nmos"]["stem"] == "tsmc5_dnf_large_nmos"
    assert record["checkpoints"]["pmos"]["complete"] is True
    assert "completion_sha256" in record["checkpoints"]["nmos"]


def test_level75_campaign_uses_distinct_stems_and_environment(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured: dict[str, str] = {}

    def fake_run(*_args: object, **kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs["env"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    assert campaign._checkpoint_stems("tsmc5", "large", 75) == {
        "nmos": "tsmc5_dnf_large_nmos",
        "pmos": "tsmc5_dnf_large_pmos",
    }
    campaign.run_deck(
        "tsmc5", "amplifier", "example", "tb_gain.cir",
        tmp_path / "out", tmp_path / "work", False, 10.0, 75, "large",
        "commit",
    )
    assert captured["PYCIRCUITSIM_NN_CHECKPOINT_DNF_NMOS"] == (
        "tsmc5_dnf_large_nmos"
    )
    assert captured["PYCIRCUITSIM_NN_CHECKPOINT_DNF_PMOS"] == (
        "tsmc5_dnf_large_pmos"
    )


def test_level75_campaign_banner_names_selected_family(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "75")
    assert complex_common.active_model_name() == "DirectNet-Full"
    assert complex_common.active_model_label() == "DirectNet-Full (LEVEL=75)"


def test_level76_campaign_uses_tff_stems_and_banner(
    monkeypatch: Any,
) -> None:
    assert campaign._checkpoint_stems("tsmc5", "large", 76) == {
        "nmos": "tsmc5_tff_large_nmos",
        "pmos": "tsmc5_tff_large_pmos",
    }
    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "76")
    assert complex_common.active_model_name() == "BSIM-AR-Full"
    assert complex_common.active_model_label() == "BSIM-AR-Full (LEVEL=76)"


def test_scored_campaign_rejects_diagnostic_directnet_rows(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(campaign, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(campaign, "artifact_record_is_current", lambda _r: True)
    monkeypatch.setattr(campaign, "executable_record_is_current", lambda _r: True)
    checkpoints: dict[str, dict[str, Any]] = {}
    for device in ("nmos", "pmos"):
        stem = f"tsmc5_dn_large_{device}"
        checkpoint = tmp_path / f"{stem}_best.pt"
        norm = tmp_path / f"{stem}_norm.npz"
        marker = checkpoint.with_suffix(".pt.complete")
        checkpoint.write_bytes(device.encode())
        norm.write_bytes(f"norm-{device}".encode())
        marker.write_text("complete")
        checkpoints[device] = {
            "stem": stem,
            "complete": True,
            "checkpoint_sha256": campaign.file_sha256(checkpoint),
            "norm_sha256": campaign.file_sha256(norm),
        }
    base_model: dict[str, Any] = {
        "family": "directnet", "level": 73, "checkpoints": checkpoints,
    }
    row = {
        "ground_truth": {
            "family": "bsim_cmg", "level": 72, "modelcard": {},
            "osdi": {}, "ngspice": {},
        },
        "py_model": base_model,
    }
    monkeypatch.setattr(
        campaign,
        "file_sha256",
        lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    assert campaign._model_matches_row(row, "tsmc5", 73, "large")

    row["py_model"] = {**base_model, "evaluator_boundary": "raw-directnet"}
    assert not campaign._model_matches_row(row, "tsmc5", 73, "large")
    row["py_model"] = {**base_model, "correction_trace": True}
    assert not campaign._model_matches_row(row, "tsmc5", 73, "large")

    row["py_model"] = base_model
    checkpoint = tmp_path / "tsmc5_dn_large_nmos_best.pt"
    original_stat = checkpoint.stat()
    checkpoint.write_bytes(b"swap")
    os.utime(
        checkpoint,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    assert checkpoint.stat().st_size == original_stat.st_size
    assert not campaign._model_matches_row(row, "tsmc5", 73, "large")
