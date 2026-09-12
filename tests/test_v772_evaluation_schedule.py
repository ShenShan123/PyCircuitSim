"""The complete campaign must progress from device evidence to circuit difficulty."""
from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import pytest

from scripts import v772_evaluate as campaign
from scripts.v710_regate_jobs import build_pools
from tests.common.simple_circuit_catalog import cases


def test_schedule_keeps_every_pool_cell_and_every_gate_entry_point() -> None:
    jobs = campaign.inventory()
    assert len({job["id"] for job in jobs}) == len(jobs)
    for pool, expected in build_pools().items():
        assert sorted(job["line"] for job in jobs if job.get("pool") == pool) == sorted(expected)
    catalog = {case.campaign_suite: case for case in cases()}
    for job in jobs:
        if job.get("suite") in catalog:
            assert job["stage"] == int(catalog[job["suite"]].tier[1]) + 1
    # These simulator-free entry points execute inside the root pytest job.
    covered = {"verify_simple_circuit_catalog", "verify_circuit_sweep_canaries",
               "verify_accuracy_campaign_tools"}
    for job in jobs:
        if job["kind"] == "pool":
            covered.add(job["suite"].split("__")[0])
        else:
            covered.update(Path(arg).stem for arg in job["command"] if arg.endswith(".py"))
    expected = {p.stem for p in (campaign.ROOT / "tests").rglob("verify_*.py")}
    assert covered == expected
    assert {job["cwd"] for job in jobs if job["id"] in ("root_contracts", "pycmg_reference")} == {
        ".", "external_compact_models/bsim_cmg"}
    for suite, applies in (("verify_device_integrity", campaign.device_corner_applies),
                           ("verify_terminal_integrity", campaign.terminal_corner_applies)):
        expected = {(tag, size, tech, corner) for tag in ("dnf", "tff")
                    for size in campaign.CLEAN_VARIANTS for tech in campaign.TECHS
                    for corner, config in campaign.CORNERS.items()
                    if corner != "nominal" and any(applies(campaign.BENCH[tech], device, config)
                                                   for device in ("nmos", "pmos"))}
        selected = [job for job in jobs if job["role"] == "device-corner-diagnostic"
                    and job["command"][0].endswith(f"/{suite}.py")]
        assert {(job["tag"], job["size"], job["tech"], job["command"][-1])
                for job in selected} == expected
        assert all(job["stage"] == 1 for job in selected)


def test_infrastructure_failure_prevents_admission_of_later_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[str] = []

    def run(job: dict[str, Any], source: str) -> dict[str, Any]:
        executed.append(job["id"])
        return {"status": "infrastructure_error"}

    monkeypatch.setattr(campaign, "STATE", tmp_path)
    monkeypatch.setattr(campaign, "run_one", run)
    jobs = [dict(id=name, stage=1, role="clean") for name in ("failed", "not_started")]
    assert not campaign.run_stage(jobs, "source", 1)
    assert executed == ["failed"]


@pytest.mark.parametrize("kind,role,rc,log,expected", [
    ("pool", "clean", 0, "worker FAIL retained", "complete"),
    ("pool", "clean", 3, "worker infrastructure failure", "infrastructure_error"),
    ("standalone", "supplemental", 1, "scientific FAIL", "complete"),
    ("standalone", "supplemental", 1, "Traceback (most recent call last):", "infrastructure_error"),
    ("standalone", "parametric-diagnostic", 2, "RESULT exit-code = 2", "complete"),
    ("standalone", "parametric-diagnostic", 2, "invalid command", "infrastructure_error"),
    ("standalone", "supplemental", -9, "killed", "infrastructure_error"),
])
def test_scientific_verdicts_do_not_mask_infrastructure_failures(
    kind: str, role: str, rc: int, log: str, expected: str,
) -> None:
    assert campaign.classify(dict(kind=kind, role=role), rc, log) == expected


def test_retry_preserves_the_failed_record_and_simulator_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = tmp_path / "state"
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(campaign, "STATE", state)
    manifest = tmp_path / "results" / f"{campaign.CAMPAIGN}_full_clean/campaign_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    job = dict(id="retry", kind="standalone", role="supplemental", stage=1,
               command=["-c", "print('recovered')"], cwd=".")
    original = state / "artifacts/retry/trace.csv"
    original.parent.mkdir(parents=True)
    original.write_text("retained failed trace")
    campaign.write_json(state / "jobs/retry.json", {"status": "infrastructure_error"})
    result = campaign.run_one(job, "source")
    assert result["status"] == "complete"
    archived = list((state / "attempts/retry").glob("*/artifacts/trace.csv"))
    assert len(archived) == 1 and archived[0].read_text() == "retained failed trace"
    assert json.loads(archived[0].parents[1].joinpath("job.json").read_text())["status"] == "infrastructure_error"


@pytest.mark.parametrize("role", ["contract", "supplemental"])
def test_fixture_contracts_do_not_inherit_the_real_campaign_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str,
) -> None:
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(campaign, "STATE", tmp_path / "state")
    (tmp_path / "state").mkdir()
    manifest = tmp_path / "results" / f"{campaign.CAMPAIGN}_full_clean/campaign_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    command = ("import os; assert ('BSIMAR_DATA_DIR' in os.environ) == "
               f"{role != 'contract'}; assert ('BSIMAR_CHECKPOINT_DIR' in os.environ) == "
               f"{role != 'contract'}")
    result = campaign.run_one(dict(id=role, kind="standalone", role=role, stage=0,
                                  command=["-c", command], cwd="."), "source")
    assert result["returncode"] == 0
