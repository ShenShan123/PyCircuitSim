"""V7.6.0 integration contracts for full-terminal AnalogGym campaigns."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from examples.complex_circuits.pycircuitsim_bench import campaign
from examples.complex_circuits.pycircuitsim_bench import run_compare
from examples.complex_circuits.pycircuitsim_bench import translate
from scripts import v762_directnet_full_analoggym_report as dnf_report
from tests.common import circuit_benchmarks as circuit_common


def _run_compare_main_for_row(
    row: dict[str, Any],
    tmp_path: Path,
    monkeypatch: Any,
    *,
    strict: bool,
) -> int:
    monkeypatch.setattr(run_compare, "_pin_design_tree", lambda *_args: None)
    monkeypatch.setattr(
        run_compare,
        "compare_deck",
        lambda *_args, **_kwargs: row,
    )
    monkeypatch.setattr(run_compare, "_print_row", lambda _row: None)
    monkeypatch.setattr(run_compare, "write_result", lambda *_args: None)
    argv = [
        "--root", str(tmp_path),
        "--tech", "tsmc5",
        "--category", "ldo",
        "--design", "ldo_1",
        "--deck", "tb_line_max.cir",
    ]
    if strict:
        argv.append("--require-full-agreement")
    return run_compare.main(argv)


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
    assert circuit_common.active_model_name() == "DirectNet-Full"
    assert circuit_common.active_model_label() == "DirectNet-Full (LEVEL=75)"


def test_campaign_summary_keeps_missing_metrics_in_denominator(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A partial quarantine must not hide required missing measurements."""
    verdict = {
        "ran": True,
        "ng_ran": True,
        "measured": 0,
        "agree": 0,
        "missing_py": 3,
        "not_comparable": 2,
        "engine_ok": True,
        "op_worst_abs": None,
        "py_seconds": 0.1,
        "ng_seconds": 0.1,
    }
    row = tmp_path / "tsmc5_ldo_partial_tb_load.json"
    row.write_text(json.dumps({"verdict": verdict}))
    monkeypatch.setattr(
        campaign,
        "corpus",
        lambda _tech, _families: [("ldo", "partial", "tb_load.cir")],
    )
    monkeypatch.setattr(campaign, "_model_matches_row", lambda *_args: True)

    summary = campaign.summarize("tsmc5", tmp_path, ["dc_source"])

    assert "**dc_source: 0/1 decks fully agree**" in summary
    assert "(1 quarantined as invalid test examples" not in summary


def test_run_compare_strict_gate_rejects_metric_failure(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A completed subprocess is not a deck pass when a metric disagrees."""
    verdict = {
        "ran": True,
        "ng_ran": True,
        "engine_ok": True,
        "measured": 1,
        "agree": 0,
        "missing_py": 0,
    }
    row = {
        "verdict": verdict,
        "pycircuitsim": {"sweeps": [{
            "kind": "dc_source", "finite": True,
            "points": 28, "solved": 28, "failed": 0,
            "flag_ok": 28, "truncated_at": None,
        }]},
    }

    assert _run_compare_main_for_row(
        row, tmp_path, monkeypatch, strict=True,
    ) == 1


def test_run_compare_strict_gate_rejects_partial_sweep(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Matching subset metrics cannot pass a physically incomplete deck."""
    row = {
        "verdict": {
            "ran": True, "ng_ran": True, "engine_ok": True,
            "measured": 1, "agree": 1, "missing_py": 0,
        },
        "pycircuitsim": {"sweeps": [{
            "kind": "dc_source", "finite": True,
            "points": 28, "solved": 27, "failed": 0,
            "flag_ok": 27, "truncated_at": None,
        }]},
    }

    assert _run_compare_main_for_row(
        row, tmp_path, monkeypatch, strict=True,
    ) == 1


def test_run_compare_strict_gate_accepts_complete_agreement(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    row = {
        "verdict": {
            "ran": True, "ng_ran": True, "engine_ok": True,
            "measured": 1, "agree": 1, "missing_py": 0,
        },
        "pycircuitsim": {"sweeps": [{
            "kind": "dc_source", "finite": True,
            "points": 28, "solved": 28, "failed": 0,
            "flag_ok": 28, "truncated_at": None,
        }]},
    }

    assert _run_compare_main_for_row(
        row, tmp_path, monkeypatch, strict=True,
    ) == 0


def test_run_compare_strict_gate_rejects_nonfinite_unmeasured_node(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Agreement on selected metrics cannot hide NaN elsewhere in a sweep."""
    row = {
        "verdict": {
            "ran": True, "ng_ran": True, "engine_ok": True,
            "measured": 1, "agree": 1, "missing_py": 0,
        },
        "pycircuitsim": {"sweeps": [{
            "kind": "ac", "finite": False,
            "points": 28, "solved": 28, "failed": 0,
            "flag_ok": None, "dc_converged": True, "truncated_at": None,
        }]},
    }

    assert _run_compare_main_for_row(
        row, tmp_path, monkeypatch, strict=True,
    ) == 1


def test_sweep_summary_checks_every_result_array_for_finiteness() -> None:
    plan = SimpleNamespace(kind="ac", label="ac")
    sweep = SimpleNamespace(
        x=np.asarray([1.0, 2.0]),
        v={
            "measured": np.asarray([1.0 + 0.0j, 2.0 + 0.0j]),
            "unmeasured": np.asarray([0.0 + 0.0j, np.nan + 0.0j]),
        },
        i={},
        meta={"points": 2, "solved": 2, "failed": 0},
    )

    assert run_compare._sweep_summary(plan, sweep)["finite"] is False


def test_run_compare_non_strict_keeps_campaign_aggregation_contract(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    row = {
        "verdict": {
            "ran": True, "ng_ran": True, "engine_ok": True,
            "measured": 1, "agree": 0, "missing_py": 0,
        },
    }

    assert _run_compare_main_for_row(
        row, tmp_path, monkeypatch, strict=False,
    ) == 0


def test_deck_agreement_requires_explicit_reference_success() -> None:
    verdict = {
        "ran": True, "engine_ok": True,
        "measured": 1, "agree": 1, "missing_py": 0,
    }

    assert not run_compare.deck_fully_agrees(verdict)


def test_campaign_summary_rejects_failed_reference(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Matching metrics cannot pass when the ground-truth run failed."""
    verdict = {
        "ran": True,
        "ng_ran": False,
        "measured": 1,
        "agree": 1,
        "missing_py": 0,
        "not_comparable": 0,
        "engine_ok": True,
        "op_worst_abs": 0.0,
        "py_seconds": 0.1,
        "ng_seconds": 0.1,
    }
    row = tmp_path / "tsmc16_amplifier_failed_tb_dc.json"
    row.write_text(json.dumps({"verdict": verdict}))
    monkeypatch.setattr(
        campaign,
        "corpus",
        lambda _tech, _families: [("amplifier", "failed", "tb_dc.cir")],
    )
    monkeypatch.setattr(campaign, "_model_matches_row", lambda *_args: True)

    summary = campaign.summarize("tsmc16", tmp_path, ["dc_temp"])

    assert "**dc_temp: 0/1 decks fully agree**" in summary
    assert "| NG FAIL |" in summary


def test_analoggym_aggregate_does_not_quarantine_missing_py_values() -> None:
    """Partial measurements stay in the scored denominator as failures."""
    partial = dnf_report._row_counts({
        "ran": True,
        "ng_ran": True,
        "engine_ok": True,
        "measured": 0,
        "agree": 0,
        "missing_py": 3,
        "not_comparable": 2,
    })
    whole_quarantine = dnf_report._row_counts({
        "ran": True,
        "ng_ran": True,
        "engine_ok": True,
        "measured": 0,
        "agree": 0,
        "missing_py": 0,
        "not_comparable": 2,
    })

    assert partial["scored"] == 1
    assert partial["quarantined"] == 0
    assert whole_quarantine["scored"] == 0
    assert whole_quarantine["quarantined"] == 1


def test_level76_campaign_uses_tff_stems_and_banner(
    monkeypatch: Any,
) -> None:
    assert campaign._checkpoint_stems("tsmc5", "large", 76) == {
        "nmos": "tsmc5_tff_large_nmos",
        "pmos": "tsmc5_tff_large_pmos",
    }
    monkeypatch.setenv("PYCIRCUITSIM_NN_FORCE_LEVEL", "76")
    assert circuit_common.active_model_name() == "BSIM-AR-Full"
    assert circuit_common.active_model_label() == "BSIM-AR-Full (LEVEL=76)"


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
