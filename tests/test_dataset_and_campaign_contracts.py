"""Dataset provenance, split, and accuracy-campaign contracts."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "bsim_cmg"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from pycmg.nn_generate import _assemble  # noqa: E402
from external_compact_models.bsim_cmg.scripts import (  # noqa: E402
    generate_nn_data,
)
from neural_network.data.sampling import (  # noqa: E402
    grouped_split_indices,
    stratified_sample_indices,
)
from neural_network.data.dataset import validate_canonical_dataset  # noqa: E402
from neural_network.data.contracts import (  # noqa: E402
    FULL_TERMINAL_OUTPUT_COLUMN_ORDER,
)
from neural_network.data import dataset as dataset_module  # noqa: E402
from scripts.v710_regate_collect import collect  # noqa: E402
from scripts import v710_regate_manifest as manifest_module  # noqa: E402
from scripts.v710_regate_manifest import (  # noqa: E402
    _artifact_hashes,
    _dataset_provenance,
    _verify_group,
)
from tests.common import nn_sweep  # noqa: E402
from tests.common.gate_result import GateResult  # noqa: E402


FULL_COLUMNS = list(FULL_TERMINAL_OUTPUT_COLUMN_ORDER)


@pytest.fixture
def report_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """A complete, checksum-bound report matrix from the new campaign."""
    from scripts import v730_docs_build as docs
    from tests.simple_circuits.verify_accuracy_campaign_tools import (
        _report_payload,
    )

    root = tmp_path / "results" / "v770_full_clean"
    root.mkdir(parents=True)
    data = {
        tag: {
            variant: {
                suite: {
                    tech: {omp: _report_payload(suite) for omp in required}
                    for tech in docs.TECHS
                }
                for suite, required in docs.REPORT_SUITES.items()
            }
            for variant in docs.TIERS
        }
        for tag in ("dnf", "tff")
    }
    data_path = root / "data.json"
    data_path.write_text(json.dumps(data))
    manifest_path = root / "campaign_manifest.json"
    manifest = {
        "job_count": 600,
        "source_commit": "a" * 40,
        "source_dirty": False,
        "checkpoint_sha256": {str(i): "b" * 64 for i in range(280)},
        "pdk_sha256": {"model.l": "c" * 64},
    }
    manifest_path.write_text(json.dumps(manifest))
    (root / "collection_provenance.json").write_text(json.dumps({
        "campaign_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes(),
        ).hexdigest(),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "source_commit": manifest["source_commit"],
    }))
    rendered = tmp_path / "docs"
    rendered.mkdir()
    monkeypatch.setattr(docs, "ROOT", tmp_path)
    monkeypatch.setattr(docs, "DOCS", rendered)
    monkeypatch.setattr(docs, "PASS_DATA", dict(docs.PASS_DATA))
    monkeypatch.setattr(docs, "REPORT_PASS", dict(docs.REPORT_PASS))
    return root


def test_report_cli_publishes_the_selected_campaign(
    report_campaign: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import v730_docs_build as docs

    monkeypatch.setattr(
        sys, "argv", ["docs", "--campaign", "v770_full_clean"],
    )
    assert docs.main() == 0
    for stem in ("DirectNet-L75", "BSIM-AR-L76"):
        report = (docs.DOCS / f"{stem}-clean.md").read_text()
        assert "V7.7.0" in report
        assert "600 jobs" in report
        assert "a" * 40 in report
        assert "results/v770_full_clean/" in report
        assert "v766_full_clean" not in report
    assert "V7.7.0" in (docs.DOCS / "README.md").read_text()
    monkeypatch.setattr(
        sys, "argv", ["docs", "--campaign", "v770_full_clean", "--check"],
    )
    assert docs.main() == 0


@pytest.mark.parametrize("check_only", (False, True))
def test_v771_report_uses_only_its_registered_complete_campaign(
    report_campaign: Path, monkeypatch: pytest.MonkeyPatch, check_only: bool,
) -> None:
    from scripts import v730_docs_build as docs

    report_campaign.rename(report_campaign.with_name("v771_full_clean"))
    arguments = ["docs", "--campaign", "v771_full_clean"]
    monkeypatch.setattr(sys, "argv", arguments)
    assert docs.main() == 0
    report = (docs.DOCS / "DirectNet-L75-clean.md").read_text()
    assert "V7.7.1" in report
    assert "results/v771_full_clean/" in report
    assert "600 jobs" in report
    assert "v770_full_clean" not in report
    if check_only:
        monkeypatch.setattr(sys, "argv", arguments + ["--check"])
        assert docs.main() == 0


@pytest.mark.parametrize("check_only", (False, True))
@pytest.mark.parametrize(
    "defect",
    ("missing_data", "data_drift", "manifest_drift", "missing_metric",
     "missing_family", "dirty_source", "wrong_job_count"),
)
def test_explicit_report_campaign_never_falls_back_to_preserved_reports(
    report_campaign: Path,
    monkeypatch: pytest.MonkeyPatch,
    check_only: bool,
    defect: str,
) -> None:
    from scripts import v730_docs_build as docs

    preserved = "preserved historical report\n"
    names = ("DirectNet-L75-clean.md", "BSIM-AR-L76-clean.md", "README.md")
    for name in names:
        (docs.DOCS / name).write_text(preserved)
    digest = hashlib.sha256(preserved.encode()).hexdigest()
    monkeypatch.setattr(
        docs, "PRESERVED_REPORT_SHA256",
        {("dnf", False): digest, ("tff", False): digest},
    )
    monkeypatch.setattr(docs, "PRESERVED_README_SHA256", digest)

    data_path = report_campaign / "data.json"
    manifest_path = report_campaign / "campaign_manifest.json"
    if defect == "missing_data":
        data_path.unlink()
    elif defect == "data_drift":
        data_path.write_text(data_path.read_text() + "\n")
    elif defect == "manifest_drift":
        manifest_path.write_text(manifest_path.read_text() + "\n")
    else:
        data = json.loads(data_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        if defect == "missing_metric":
            del data["dnf"]["small"]["verify_circuit_ring_osc"]["TSMC5"]["omp1"]["metric"]
        elif defect == "missing_family":
            del data["tff"]
        elif defect == "dirty_source":
            manifest["source_dirty"] = True
        elif defect == "wrong_job_count":
            manifest["job_count"] = 480
        data_path.write_text(json.dumps(data))
        manifest_path.write_text(json.dumps(manifest))
        provenance_path = report_campaign / "collection_provenance.json"
        provenance = json.loads(provenance_path.read_text())
        provenance.update(
            campaign_manifest_sha256=hashlib.sha256(
                manifest_path.read_bytes(),
            ).hexdigest(),
            data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        )
        provenance_path.write_text(json.dumps(provenance))

    monkeypatch.setattr(
        sys, "argv",
        ["docs", "--campaign", "v770_full_clean"]
        + (["--check"] if check_only else []),
    )
    assert docs.main() == 1
    assert all((docs.DOCS / name).read_text() == preserved for name in names)


def test_stratified_cap_keeps_every_stratum() -> None:
    strata = np.asarray([[0], [0], [0], [1], [1], [2], [2], [2], [2]])
    selected = stratified_sample_indices(strata, n_samples=5, seed=7)
    assert set(strata[selected, 0]) == {0, 1, 2}
    assert len(np.unique(selected)) == 5


def test_combo_split_never_leaks_a_group() -> None:
    strata = np.repeat(np.arange(12)[:, None], repeats=5, axis=0)
    splits = grouped_split_indices(strata, train_ratio=0.8, val_ratio=0.1, seed=3)
    groups = [{int(value) for value in strata[idx, 0]} for idx in splits]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert sum(len(index) for index in splits) == len(strata)


def test_combo_split_supports_an_empty_validation_partition() -> None:
    strata = np.repeat(np.arange(4)[:, None], repeats=3, axis=0)
    train, validation, test = grouped_split_indices(
        strata, train_ratio=0.75, val_ratio=0.0, seed=7,
    )
    assert len(validation) == 0
    assert len(train) + len(test) == len(strata)
    assert set(train).isdisjoint(test)


def test_loader_passes_full_combo_strata_to_grouped_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "loader.npz"
    geometry = np.zeros((6, 15), dtype=np.float64)
    geometry[:, :3] = np.asarray([
        [2, 16e-9, 300.15], [2, 16e-9, 300.15],
        [3, 20e-9, 300.15], [3, 20e-9, 300.15],
        [4, 20e-9, 398.15], [4, 20e-9, 398.15],
    ])
    np.savez(
        path, inputs=np.arange(24, dtype=float).reshape(6, 4),
        geometry=geometry,
        outputs=np.repeat(
            np.arange(6, dtype=float)[:, None] + 1.0, 6, axis=1,
        ),
        meta_output_columns=np.asarray(FULL_COLUMNS),
    )
    tech_codes = np.asarray([0, 0, 1, 1, 1, 1])
    monkeypatch.setattr(
        "neural_network.eval.loo_labels.get_or_build_tech_variant_labels",
        lambda *_args, **_kwargs: tech_codes,
    )
    observed: List[np.ndarray] = []

    def _capture(
        strata: np.ndarray, _train: float, _val: float, _seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        observed.append(strata.copy())
        return np.arange(4), np.asarray([4]), np.asarray([5])

    monkeypatch.setattr(dataset_module, "grouped_split_indices", _capture)
    dataset_module.load_and_split_bsimar(
        str(path), "nmos",
    )
    np.testing.assert_array_equal(
        observed[0], np.column_stack([tech_codes, geometry[:, :3]]),
    )


def test_training_overlay_moves_its_whole_combo_stratum_to_train(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Circuit-derived rows and same-geometry peers cannot remain held out."""
    path = tmp_path / "overlay.npz"
    geometry = np.zeros((6, 15), dtype=np.float64)
    geometry[:, :3] = np.asarray([
        [2, 16e-9, 300.15], [2, 16e-9, 300.15],
        [3, 20e-9, 300.15], [3, 20e-9, 300.15],
        [2, 16e-9, 398.15], [2, 16e-9, 398.15],
    ])
    np.savez(
        path,
        inputs=np.arange(24, dtype=float).reshape(6, 4),
        geometry=geometry,
        outputs=np.repeat(
            np.arange(6, dtype=float)[:, None] + 1.0, 6, axis=1,
        ),
        meta_output_columns=np.asarray(FULL_COLUMNS),
        sample_class=np.asarray([0, 0, 0, 0, 1, 0], dtype=np.int8),
        meta_sample_class_names=np.asarray(["grid", "traj_corridor"]),
    )
    monkeypatch.setattr(
        "neural_network.eval.loo_labels.get_or_build_tech_variant_labels",
        lambda *_args, **_kwargs: np.zeros(6, dtype=int),
    )
    monkeypatch.setattr(
        dataset_module,
        "grouped_split_indices",
        lambda *_args, **_kwargs: (
            np.asarray([0, 1]), np.asarray([2, 3]), np.asarray([4, 5]),
        ),
    )

    train, validation, test, _normalizer = (
        dataset_module.load_and_split_bsimar(
            str(path), "nmos",
            training_overlay_classes={"traj_corridor"},
        )
    )

    assert (len(train), len(validation), len(test)) == (4, 2, 0)
    assert train.sample_class is not None
    assert train.sample_class.tolist().count(1) == 1


def test_training_overlay_rejects_unknown_class_and_random_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "overlay.npz"
    np.savez(
        path,
        inputs=np.zeros((3, 4)), geometry=np.zeros((3, 15)),
        outputs=np.ones((3, 6)),
        meta_output_columns=np.asarray(FULL_COLUMNS),
        sample_class=np.asarray([0, 1, 0]),
        meta_sample_class_names=np.asarray(["grid", "traj_corridor"]),
    )
    monkeypatch.setattr(
        "neural_network.eval.loo_labels.get_or_build_tech_variant_labels",
        lambda *_args, **_kwargs: np.zeros(3, dtype=int),
    )

    with pytest.raises(ValueError, match="not in dataset"):
        dataset_module.load_and_split_bsimar(
            str(path), "nmos",
            training_overlay_classes={"missing"},
        )
    with pytest.raises(ValueError, match="require split_mode='combo'"):
        dataset_module.load_and_split_bsimar(
            str(path), "nmos",
            split_mode="random",
            training_overlay_classes={"traj_corridor"},
        )


def _bin_result(n_failed: int) -> Dict[str, np.ndarray]:
    return {
        "inputs": np.zeros((1, 4)),
        "geometry": np.zeros((1, 15)),
        "outputs": np.zeros((1, 6)),
        "sample_class": np.zeros(1, dtype=np.int8),
        "n_kept": np.asarray(1),
        "n_failed": np.asarray(n_failed),
        "n_requested": np.asarray(1 + n_failed),
        "failed_inputs": np.ones((n_failed, 4)),
        "failed_class": np.zeros(n_failed, dtype=np.int8),
        "failure_reasons": np.asarray(
            ["non_finite_output"] * n_failed, dtype=str,
        ),
    }


def test_dataset_assembly_fails_loud_on_rejected_point() -> None:
    with pytest.raises(RuntimeError, match="rejected 1 points"):
        _assemble([_bin_result(1)], {}, verbose=False)


def test_dataset_assembly_fails_loud_on_dropped_bin() -> None:
    with pytest.raises(RuntimeError, match="dropped 1 bins"):
        _assemble([None], {}, verbose=False)


def test_canonical_dataset_allows_only_declared_safety_rejections() -> None:
    data = _assemble(
        [_bin_result(1)], {}, verbose=False, allow_safety_rejections=True,
    )
    assert data["metadata"]["rejected_rows"] == 1
    assert data["metadata"]["allow_safety_rejections"]

    result = _bin_result(1)
    result["failure_reasons"] = np.asarray(["eval_exception:RuntimeError"])
    with pytest.raises(RuntimeError, match="unapproved rejection reasons"):
        _assemble(
            [result], {}, verbose=False, allow_safety_rejections=True,
        )


def test_diagnostic_dataset_records_rejection_manifest() -> None:
    data = _assemble(
        [_bin_result(1), None], {}, verbose=False,
        allow_rejected_points=True,
    )
    assert data["metadata"]["rejected_rows"] == 1
    assert data["metadata"]["dropped_bins"] == 1
    manifest = json.loads(str(data["metadata"]["manifest_json"]))
    assert manifest[0]["status"] == "partial"
    assert manifest[0]["failed_coordinates"] == [[1.0, 1.0, 1.0, 1.0]]
    assert manifest[1]["status"] == "dropped"


def _write_canonical_dataset(tmp_path: Path, **overrides: object) -> Path:
    path = tmp_path / "training.npz"
    metadata: Dict[str, object] = {
        "allow_rejected_points": False,
        "allow_safety_rejections": False,
        "dataset_variant": "v7517_generated_core_plus_tg",
        "dropped_bins": 0,
        "generator_release": "V7.5.17",
        "kept_rows": 4,
        "manifest_json": '[{"status": "complete"}]',
        "modelcard_sha256_json": json.dumps({"modelcard.l": "c" * 64}),
        "osdi_sha256": "a" * 64,
        "rejected_rows": 0,
        "requested_rows": 4,
        "source_commit": "b" * 40,
        "source_dirty": False,
    }
    metadata.update(overrides)
    arrays: Dict[str, np.ndarray] = {
        "inputs": np.zeros((4, 4)),
        "geometry": np.zeros((4, 15)),
        "outputs": np.zeros((4, 6)),
        "meta_output_columns": np.asarray(FULL_COLUMNS),
    }
    arrays.update({f"meta_{key}": np.asarray(value)
                   for key, value in metadata.items()})
    np.savez(path, **arrays)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".npz.complete").write_text(json.dumps({
        "dataset": path.name,
        "dataset_sha256": digest,
        "rows": 4,
        "source_commit": metadata["source_commit"],
        "generator_release": metadata["generator_release"],
    }))
    return path


def test_canonical_dataset_validator_rejects_diagnostic_data(
    tmp_path: Path,
) -> None:
    canonical = _write_canonical_dataset(tmp_path)
    validate_canonical_dataset(canonical)
    diagnostic = _write_canonical_dataset(
        tmp_path, allow_rejected_points=True, rejected_rows=1,
        requested_rows=5,
    )
    with pytest.raises(ValueError, match="diagnostic dataset"):
        validate_canonical_dataset(diagnostic)


def test_canonical_dataset_validator_accepts_audited_safety_rejections(
    tmp_path: Path,
) -> None:
    manifest = json.dumps([{
        "status": "partial",
        "requested": 5,
        "kept": 4,
        "rejected": 1,
        "failure_reason_counts": {"terminal_current_over_1A": 1},
    }])
    path = _write_canonical_dataset(
        tmp_path,
        allow_safety_rejections=True,
        rejected_rows=1,
        requested_rows=5,
        manifest_json=manifest,
    )
    validate_canonical_dataset(path)


def test_canonical_dataset_validator_rejects_bad_checksum(
    tmp_path: Path,
) -> None:
    path = _write_canonical_dataset(tmp_path)
    marker_path = path.with_suffix(".npz.complete")
    marker = json.loads(marker_path.read_text())
    marker["dataset_sha256"] = "0" * 64
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="checksum"):
        validate_canonical_dataset(path)


def test_canonical_dataset_validator_rejects_empty_model_provenance(
    tmp_path: Path,
) -> None:
    path = _write_canonical_dataset(tmp_path, modelcard_sha256_json="{}")
    with pytest.raises(ValueError, match="modelcard SHA-256"):
        validate_canonical_dataset(path)


def test_collector_preserves_temperature_rows_and_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    log = root / "dnf" / "large" / "tsmc5" / "verify_nn_multi_tech_dc.omp1.log"
    log.parent.mkdir(parents=True)
    manifest = root / "campaign_manifest.json"
    manifest.write_text("{}\n")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    log.write_text(
        f"===V710_PROVENANCE sha256={digest}===\n"
        " TSMC5_nmos_temp_-25c NRMSE= 1.0% MRE= 2.0% R2= 0.9 "
        "MaxErr=0.1 PASS\n"
        " TSMC5_nmos_temp_+125c NRMSE= 2.0% MRE= 3.0% R2= 0.8 "
        "MaxErr=0.2 FAIL\n"
        "===V710_DONE rc=1===\n"
    )
    data = collect(root, require_manifest=True)
    entry = data["dnf"]["large"]["verify_nn_multi_tech_dc"]["TSMC5"]["omp1"]
    assert entry["n"] == 2
    assert set(entry["rows"]) == {
        "TSMC5_nmos_temp_-25c", "TSMC5_nmos_temp_+125c",
    }
    log.write_text(log.read_text().replace(digest, "0" * 64))
    with pytest.raises(ValueError, match="provenance mismatch"):
        collect(root, require_manifest=True)


def test_campaign_manifest_detects_checkpoint_mutation(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    group = ("dnf", "small", "tsmc5")
    for device in ("nmos", "pmos"):
        stem = f"tsmc5_dnf_small_{device}"
        (checkpoints / f"{stem}_best.pt").write_bytes(b"original")
        (checkpoints / f"{stem}_norm.npz").write_bytes(b"original")
        (checkpoints / f"{stem}_best.pt.complete").write_text(json.dumps({
            "dataset": f"tsmc5_dnf_{device}.npz",
            "dataset_sha256": "a" * 64,
            "dataset_completion_marker": f"tsmc5_dnf_{device}.npz.complete",
            "dataset_completion_marker_sha256": "b" * 64,
            "dataset_source_commit": "c" * 40,
        }))
    manifest = tmp_path / "campaign_manifest.json"
    manifest.write_text(json.dumps({
        "checkpoint_sha256": _artifact_hashes(checkpoints, {group}),
        "dataset_provenance": _dataset_provenance(checkpoints, {group}),
    }))
    _verify_group(manifest, checkpoints, group)
    (checkpoints / "tsmc5_dnf_small_nmos_best.pt").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="checkpoint artifacts drifted"):
        _verify_group(manifest, checkpoints, group)


def test_dataset_generator_marks_untracked_templates_dirty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        command: List[str], **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        if command[1:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, "a" * 40 + "\n", "")
        if command[1:] == ["status", "--porcelain"]:
            return subprocess.CompletedProcess(
                command, 0, "?? circuit_templates/new.spice.tmpl\n", "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(generate_nn_data.subprocess, "run", fake_run)
    data: Dict[str, Dict[str, object]] = {"metadata": {}}
    generate_nn_data._add_run_provenance(data)

    assert data["metadata"]["source_dirty"] is True


def test_campaign_manifest_rejects_untracked_templates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for device in ("nmos", "pmos"):
        stem = f"tsmc5_dnf_small_{device}"
        for suffix in ("_best.pt", "_norm.npz", "_best.pt.complete"):
            (checkpoints / f"{stem}{suffix}").write_bytes(b"artifact")
    jobs = tmp_path / "jobs.txt"
    jobs.write_text("dnf small TSMC5 verify_nn_ac 1\n")
    ngspice = tmp_path / "ngspice"
    ngspice.write_bytes(b"ngspice")
    osdi = tmp_path / "bsimcmg.osdi"
    osdi.write_bytes(b"osdi")
    pdk_root = tmp_path / "PDKs"
    pdk_root.mkdir()

    def fake_git(_root: Path, *args: str) -> str:
        if args == ("status", "--porcelain"):
            return "?? circuit_templates/new.spice.tmpl"
        if args == ("status", "--porcelain", "--untracked-files=no"):
            return ""
        if args == ("ls-files", "--others", "--exclude-standard"):
            return "circuit_templates/new.spice.tmpl"
        if args == ("rev-parse", "HEAD"):
            return "a" * 40
        raise AssertionError(f"unexpected git invocation: {args}")

    monkeypatch.setattr(manifest_module, "_git", fake_git)
    with pytest.raises(SystemExit, match="clean worktree"):
        manifest_module.main([
            "--output", str(tmp_path / "campaign_manifest.json"),
            "--jobs", str(jobs),
            "--checkpoints", str(checkpoints),
            "--ngspice", str(ngspice),
            "--osdi", str(osdi),
            "--pdk-root", str(pdk_root),
        ])


@pytest.mark.parametrize("tag", ["dnf", "tff"])
def test_full_terminal_campaign_requires_dataset_provenance(
    tmp_path: Path,
    tag: str,
) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    group = (tag, "small", "tsmc5")
    for device in ("nmos", "pmos"):
        stem = f"tsmc5_{tag}_small_{device}"
        (checkpoints / f"{stem}_best.pt").touch()
        (checkpoints / f"{stem}_norm.npz").touch()
        (checkpoints / f"{stem}_best.pt.complete").write_text(json.dumps({
            "family": "full-terminal",
        }))

    with pytest.raises(ValueError, match="dataset provenance"):
        _dataset_provenance(checkpoints, {group})

    for device in ("nmos", "pmos"):
        marker = checkpoints / f"tsmc5_{tag}_small_{device}_best.pt.complete"
        payload = json.loads(marker.read_text())
        payload.update({
            "dataset": f"tsmc5_dnf_{device}.npz",
            "dataset_sha256": "a" * 64,
            "dataset_completion_marker": f"tsmc5_dnf_{device}.npz.complete",
            "dataset_completion_marker_sha256": "b" * 64,
            "dataset_source_commit": "c" * 40,
        })
        marker.write_text(json.dumps(payload))

    provenance = _dataset_provenance(checkpoints, {group})
    assert len(provenance) == 2
    assert {entry["dataset_source_commit"] for entry in provenance.values()} == {
        "c" * 40,
    }


def test_all_three_pn_ratios_are_declared_for_every_tech() -> None:
    for tech in nn_sweep.NN_TECHS:
        configs = nn_sweep.build_inv_parametric(tech, "vtc")
        ratios = {
            round(float(config.swept["pn_ratio"]), 1)
            for config in configs if config.sweep_type == "pn_ratio"
        }
        assert ratios == {0.5, 1.5, 2.0}


@dataclass(frozen=True)
class _Config:
    label: str
    config_name: str


def test_failed_baseline_does_not_shrink_denominator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def make_baseline(_tech: str, _dimension: str) -> _Config:
        return _Config("baseline", "baseline")

    def make_params(_tech: str, _dimension: str) -> List[_Config]:
        return [_Config("corner-a", "a"), _Config("corner-b", "b")]

    def run_one(config: _Config, _work: Path, _checkpoints: Dict) -> Dict:
        return {"config": config, "passed": False, "error": "failed"}

    results = nn_sweep.run_nn_multi_tech(
        ["TSMC5"], "nmos", tmp_path,
        make_baseline, make_params, run_one,
    )
    assert [result["config"].label for result in results] == [
        "baseline", "corner-a", "corner-b",
    ]


def _ac_campaign_row(device: str, *, complete_metrics: bool = True) -> GateResult:
    metrics = {
        "gain0_db_error": 0.1,
        "bandwidth_ratio": 1.0,
        "nrmse_pct": 1.0,
        "phase_error_deg": 2.0,
        "mag_nrmse_pct": 1.0,
        "mre_pct": 1.0,
        "r2": 0.99,
        "max_err": 0.01,
    }
    if not complete_metrics:
        del metrics["bandwidth_ratio"]
    return GateResult(
        case_id="nn_ac",
        tech="TSMC12",
        corner="nominal",
        analysis=device,
        role="qualification",
        status="pass",
        metrics=metrics,
        model_family="DirectNet-Full",
        model_level=75,
        checkpoint_pins={
            "nmos": "tsmc12_dnf_small_nmos",
            "pmos": "tsmc12_dnf_small_pmos",
        },
        thread_settings={"omp": 1, "mkl": 1, "torch": 1},
    )


def test_structured_nn_ac_rows_drive_report_and_completeness(tmp_path: Path) -> None:
    from scripts.v710_regate_collect import render

    root = tmp_path / "campaign"
    log = root / "dnf" / "small" / "tsmc12" / "verify_nn_ac.omp1.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "\n".join([
            _ac_campaign_row("nmos").marker(),
            _ac_campaign_row("pmos").marker(),
            "===V710_DONE rc=0===",
        ]) + "\n"
    )

    data = collect(root)
    entry = data["dnf"]["small"]["verify_nn_ac"]["TSMC12"]["omp1"]
    assert entry["result_complete"] is True
    assert entry["nmos"]["status"] == "PASS"
    assert entry["pmos"]["f3db_ratio"] == 1.0
    assert "**Device CS-amp AC: 2/2**" in render(data)


def test_malformed_structured_nn_row_is_invalid_without_crashing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    log = root / "dnf" / "small" / "tsmc12" / "verify_nn_ac.omp1.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "\n".join([
            _ac_campaign_row("nmos", complete_metrics=False).marker(),
            _ac_campaign_row("pmos").marker(),
            "===V710_DONE rc=0===",
        ]) + "\n"
    )

    entry = collect(root)["dnf"]["small"]["verify_nn_ac"]["TSMC12"]["omp1"]
    assert entry["result_complete"] is False
    assert entry["status"] == "ERROR"
    assert "missing required metrics" in entry["error"]


def test_unstructured_nn_logs_are_display_only_not_coverage(tmp_path: Path) -> None:
    from scripts import v730_coverage

    log = tmp_path / "dnf" / "small" / "tsmc12" / "verify_nn_ac.omp1.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "  AC | tsmc12_nmos | 0.1 | 1.0 | 1.0 | 2.0 | PASS\n"
        "===V710_DONE rc=0===\n"
    )

    scanned = v730_coverage.scan_logs(tmp_path)
    assert list(scanned.values()) == [None]


def test_legacy_json_cannot_satisfy_a_structured_coverage_cell(
    tmp_path: Path,
) -> None:
    from scripts import v730_coverage

    (tmp_path / "data.json").write_text(json.dumps({
        "dnf": {
            "small": {
                "verify_nn_ac": {
                    "TSMC12": {"omp1": {"rc": "0"}},
                },
            },
        },
    }))
    previous = v730_coverage.PASSES
    v730_coverage.PASSES = [("test", tmp_path)]
    try:
        index = v730_coverage.build_index(["test"])
    finally:
        v730_coverage.PASSES = previous

    assert index == {}


def test_structured_nn_rows_bind_the_declared_corner(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    log = root / "dnf" / "small" / "tsmc12" / "verify_nn_ac.omp1.log"
    log.parent.mkdir(parents=True)
    wrong = _ac_campaign_row("nmos").payload()
    wrong["corner"] = "hot"
    log.write_text(
        "\n".join([
            GateResult(**wrong).marker(),
            _ac_campaign_row("pmos").marker(),
            "===V710_DONE rc=0===",
        ]) + "\n"
    )

    entry = collect(root)["dnf"]["small"]["verify_nn_ac"]["TSMC12"]["omp1"]
    assert entry["result_complete"] is False
    assert "corner mismatch" in entry["error"]


def test_all_error_parametric_denominator_renders_without_numeric_aggregate(
    tmp_path: Path,
) -> None:
    from scripts.v710_regate_collect import render

    root = tmp_path / "campaign"
    log = (
        root / "dnf" / "small" / "tsmc12"
        / "verify_nn_multi_tech_dc.omp1.log"
    )
    log.parent.mkdir(parents=True)
    configs = [
        config
        for device in ("nmos", "pmos")
        for config in (
            nn_sweep.make_dc_baseline("TSMC12", device),
            *nn_sweep.build_dc_parametric("TSMC12", device),
        )
    ]
    rows = [
        GateResult(
            case_id="nn_parametric_dc",
            tech="TSMC12",
            corner=config.sweep_type,
            analysis=config.label,
            role="qualification",
            status="error",
            error="candidate did not converge",
            candidate_converged=False,
            execution_state="nonconverged",
            error_kind="candidate",
            model_family="DirectNet-Full",
            model_level=75,
            checkpoint_pins={
                "nmos": "tsmc12_dnf_small_nmos",
                "pmos": "tsmc12_dnf_small_pmos",
            },
            thread_settings={"omp": 1, "mkl": 1, "torch": 1},
        )
        for config in configs
    ]
    log.write_text(
        "\n".join([*(row.marker() for row in rows), "===V710_DONE rc=1==="])
        + "\n"
    )

    data = collect(root)
    entry = data["dnf"]["small"]["verify_nn_multi_tech_dc"]["TSMC12"]["omp1"]
    assert entry["result_complete"] is True
    assert entry["n_pass"] == 0
    assert "max_error" not in entry
    assert f"| TSMC12 | 0/{len(configs)} | — | — | — | — | — |" in render(data)
