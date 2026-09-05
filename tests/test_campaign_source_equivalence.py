"""Harness fixes may reuse training only with exact numerical source identity."""
from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from scripts.v710_regate_manifest import validate_dataset_source


def commit(root: Path, message: str) -> str:
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "commit", "-qm", message], cwd=root, check=True, capture_output=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


@pytest.fixture
def source_repo(tmp_path: Path) -> tuple[Path, str]:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    for name in ("pycircuitsim/solver.py", "external_compact_models/neural_network/config.py",
                 "external_compact_models/bsim_cmg/pycmg/nn_generate.py",
                 "circuit_templates/L0_devices/device.spice.tmpl", "PDKs/ASAP7/card.pm",
                 "environment.yml"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original\n")
    return tmp_path, commit(tmp_path, "test: original numerical source")


def test_harness_change_requires_explicit_original_source(source_repo: tuple[Path, str]) -> None:
    root, source = source_repo
    (root / "tests").mkdir()
    (root / "tests/metric.py").write_text("fixed metric classification\n")
    (root / "external_compact_models/neural_network/README.md").write_text("campaign update\n")
    gate = commit(root, "test: harness and documentation correction")
    with pytest.raises(ValueError, match="does not match"):
        validate_dataset_source(root, {source}, gate)
    identity = validate_dataset_source(root, {source}, gate, source)
    assert identity["dataset_source_commit"] == source
    assert len(identity["model_source_sha256"]) == 64
    with pytest.raises(ValueError, match="does not match"):
        validate_dataset_source(root, {source, gate}, gate, source)


@pytest.mark.parametrize("changed", (
    "pycircuitsim/solver.py", "external_compact_models/neural_network/config.py",
    "external_compact_models/bsim_cmg/pycmg/nn_generate.py",
    "circuit_templates/L0_devices/device.spice.tmpl", "PDKs/ASAP7/card.pm", "environment.yml",
))
def test_changed_numerical_inputs_cannot_reuse_the_old_source(
    source_repo: tuple[Path, str], changed: str,
) -> None:
    root, source = source_repo
    (root / changed).write_text("changed numerical input\n")
    gate = commit(root, "test: mutate numerical source")
    with pytest.raises(ValueError, match="sources differ"):
        validate_dataset_source(root, {source}, gate, source)


def test_same_source_rejects_a_misleading_explicit_pin(source_repo: tuple[Path, str]) -> None:
    root, source = source_repo
    assert validate_dataset_source(root, {source}, source) == {}
    with pytest.raises(ValueError, match="declared dataset source"):
        validate_dataset_source(root, {source}, source, "0" * 40)
