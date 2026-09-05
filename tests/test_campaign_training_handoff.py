"""Consolidation must preserve active workers and gate only after all training."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import v772_consolidate as runner


def test_live_training_finishes_before_stop_and_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = tmp_path / "training"
    training.mkdir()
    source, gate = "a" * 40, "b" * 40
    calls: list[str] = []
    for key, value in tuple(runner.os.environ.items()):
        if key.startswith(("PYCIRCUITSIM_", "BSIMAR_", "V710_")):
            monkeypatch.setenv(key, value)
    stages = [
        {"complete": ["one"], "running": ["two"], "failed": [], "total": 2},
        {"complete": ["one", "two"], "running": [], "failed": [], "total": 2},
    ]
    monkeypatch.setattr(runner.campaign, "ROOT", tmp_path)
    for name in ("CAMPAIGN", "STATE", "DATA", "CHECKPOINTS"):
        monkeypatch.setattr(runner.campaign, name, getattr(runner.campaign, name))
    monkeypatch.setattr(runner.campaign, "git", lambda *args: gate)
    monkeypatch.setattr(runner.campaign, "assert_source", lambda commit: None)
    monkeypatch.setattr(runner, "validate_dataset_source", lambda *args: {})
    monkeypatch.setattr(runner, "training_progress", lambda *args: stages.pop(0))
    monkeypatch.setattr(runner.subprocess, "check_output",
                        lambda args, **kwargs: source if args[1] == "rev-parse" else "")
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: calls.append("training continues"))

    def systemctl(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        if "stop" in command:
            assert not stages
            calls.append("stop old evaluation tail")
        assert "start" not in command
        return subprocess.CompletedProcess(command, 0)

    def evaluate(python: str, commit: str, env: dict[str, str], parallel: int) -> None:
        assert not stages
        assert calls == ["training continues", "stop old evaluation tail"]
        assert env["V710_DATASET_SOURCE_COMMIT"] == source
        assert env["BSIMAR_CHECKPOINT_DIR"] == str(training / "results/v771_r2_checkpoints")
        calls.append("evaluate V7.7.2")

    monkeypatch.setattr(runner.subprocess, "run", systemctl)
    monkeypatch.setattr(runner.campaign, "evaluate", evaluate)
    monkeypatch.setattr(sys, "argv", ["handoff", "--training-root", str(training),
                                     "--training-source", source])
    assert runner.main() == 0
    state = json.loads((tmp_path / "results/v772_campaign/state.json").read_text())
    assert state["stage"] == "evaluate" and state["status"] == "complete"
    assert calls[-1] == "evaluate V7.7.2"


@pytest.mark.parametrize("returncode", (None, 1))
def test_a_completion_label_cannot_hide_an_unsuccessful_training_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, returncode: int | None,
) -> None:
    monkeypatch.setattr(runner.campaign, "training_jobs",
                        lambda: [("direct", "small", "tsmc5", "nmos")])
    path = tmp_path / "results/v771_campaign/jobs/train-tsmc5_dnf_small_nmos.json"
    runner.campaign.write_json(path, {"source_commit": "a" * 40,
                                     "status": "complete", "returncode": returncode})
    with pytest.raises(ValueError, match="did not exit successfully"):
        runner.training_progress(tmp_path, "a" * 40)
