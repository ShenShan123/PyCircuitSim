"""Multi-day work must resume safely and keep artifact failures visible."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

from scripts import v771_campaign as campaign


def test_training_inventory_covers_every_supported_model() -> None:
    jobs = campaign.training_jobs()
    expected = {(model, size, tech, device)
                for model in ("direct", "transformer")
                for size in ("small", "medium", "large", "xl")
                for tech in campaign.TECHS for device in campaign.DEVICES}
    assert len(jobs) == len(set(jobs)) == 80
    assert set(jobs) == expected
    assert {job[0] for job in jobs[:4]} == {"direct", "transformer"}


@pytest.fixture
def runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(campaign, "STATE", tmp_path)
    monkeypatch.setattr(campaign, "assert_source", lambda commit: None)
    return tmp_path


def test_resume_preserves_completed_work_when_gpu_assignment_changes(runner: Path) -> None:
    target = runner / "calls"
    command = [sys.executable, "-c",
               "import sys; open(sys.argv[1], 'a').write('call\\n')", str(target)]
    campaign.run_job("training", command, {"CUDA_VISIBLE_DEVICES": "0"}, "a" * 40)
    campaign.run_job("training", command, {"CUDA_VISIBLE_DEVICES": "3"}, "a" * 40)
    assert target.read_text() == "call\n"
    assert len(list((runner / "logs").glob("*.log"))) == 1


def test_successful_child_with_invalid_artifacts_is_failed(runner: Path) -> None:
    def validate() -> None:
        raise ValueError("normalization checksum mismatch")

    with pytest.raises(ValueError, match="checksum"):
        campaign.run_job("bad-bundle", [sys.executable, "-c", "pass"], {},
                         "a" * 40, validate)
    record = json.loads((runner / "jobs/bad-bundle.json").read_text())
    assert record["returncode"] == 0
    assert record["status"] == "failed"
    assert "checksum" in record["error"]


def test_resume_rejects_a_live_worker(runner: Path) -> None:
    command = [sys.executable, "-c", "raise AssertionError('duplicate worker')"]
    campaign.write_json(runner / "jobs/active.json", {
        "command": command, "source_commit": "a" * 40, "environment": {},
        "status": "running", "pid": os.getpid(),
    })
    with pytest.raises(RuntimeError, match="still exists"):
        campaign.run_job("active", command, {}, "a" * 40)
    assert not (runner / "logs").exists()


@pytest.mark.parametrize("tag", ("dnf", "tff"))
def test_bundle_validation_binds_configuration_and_source_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tag: str,
) -> None:
    monkeypatch.setattr(campaign, "DATA", tmp_path)
    monkeypatch.setattr(campaign, "CHECKPOINTS", tmp_path)
    stem = f"tsmc5_{tag}_small_nmos"
    dataset = "tsmc5_dnf_nmos.npz"
    commit = "a" * 40
    completion = tmp_path / f"{dataset}.complete"
    campaign.write_json(completion, {"source_commit": commit, "source_dirty": False,
                                     "dataset_sha256": "b" * 64})
    marker = {"dataset": dataset, "dataset_source_commit": commit,
              "dataset_sha256": "b" * 64,
              "dataset_completion_marker_sha256": campaign.sha256(completion)}
    artifacts = [("checkpoint", "_best.pt"), ("normalization", "_norm.npz")]
    if tag == "tff":
        artifacts.append(("configuration", "_config.npz"))
    for field, suffix in artifacts:
        path = tmp_path / f"{stem}{suffix}"
        path.write_bytes(field.encode())
        marker[field] = path.name
        marker[f"{field}_sha256"] = campaign.sha256(path)
    campaign.write_json(tmp_path / f"{stem}_best.pt.complete", marker)
    campaign.validate_bundle(stem, commit)
    with pytest.raises(ValueError, match="provenance"):
        campaign.validate_bundle(stem, "c" * 40)
    # TFF configuration is part of readiness, even if weights are unchanged.
    field, suffix = artifacts[-1]
    (tmp_path / f"{stem}{suffix}").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        campaign.validate_bundle(stem, commit)
