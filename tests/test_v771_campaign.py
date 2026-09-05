"""Multi-day work must resume safely and keep artifact failures visible."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

from scripts import v771_campaign as campaign


@pytest.mark.parametrize("name", ("v771", "v772"))
def test_campaign_outputs_are_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    for attribute in ("CAMPAIGN", "STATE", "DATA", "CHECKPOINTS"):
        monkeypatch.setattr(campaign, attribute, getattr(campaign, attribute))
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.delenv("BSIMAR_DATA_DIR", raising=False)
    monkeypatch.delenv("BSIMAR_CHECKPOINT_DIR", raising=False)
    campaign.configure_campaign(name)
    assert campaign.STATE == tmp_path / f"results/{name}_campaign"
    assert campaign.DATA == tmp_path / f"results/{name}_full_data"
    assert campaign.CHECKPOINTS == tmp_path / f"results/{name}_full_checkpoints"


@pytest.mark.parametrize(("stage", "status", "ready"), (
    ("data", "complete", False), ("train", "running", False),
    ("train", "failed", False), ("train", "waiting", False),
    ("train", "complete", True), ("evaluate", "running", True),
    ("evaluate", "failed", True), ("evaluate", "complete", True),
))
def test_successor_waits_for_completed_training(
    tmp_path: Path, stage: str, status: str, ready: bool,
) -> None:
    path = tmp_path / "state.json"
    campaign.write_json(path, {"stage": stage, "status": status})
    assert campaign.predecessor_training_complete(path) is ready


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


def test_training_uses_physical_gpu_identity_despite_cuda_ordinal_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An idle physical GPU 0 must not route onto CUDA's fastest busy GPU."""
    physical_uuid = "GPU-11111111-2222-3333-4444-555555555555"
    captured: list[str] = []

    def nvidia_query(command: list[str], **kwargs: object) -> str:
        assert "--id=0" in command
        return physical_uuid + "\n" if "--query-gpu=uuid" in command else ""

    def run_job(
        name: str, command: list[str], env: dict[str, str], commit: str,
        validate: object,
    ) -> None:
        captured.append(env["CUDA_VISIBLE_DEVICES"])

    monkeypatch.setattr(campaign.subprocess, "check_output", nvidia_query)
    monkeypatch.setattr(campaign, "run_job", run_job)
    monkeypatch.setattr(campaign, "training_jobs", lambda: [("direct", "small", "tsmc5", "nmos")])
    campaign.train(sys.executable, "a" * 40, {}, ["0"])
    assert captured == [physical_uuid]


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
