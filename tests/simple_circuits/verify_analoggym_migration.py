#!/usr/bin/env python3
"""Regression checks for the AnalogGym comparison harness and deck corpus."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional
from unittest.mock import patch

import numpy as np


REPO_ROOT: Path = Path(__file__).resolve().parents[2]
BENCH_ROOT: Path = REPO_ROOT / "examples" / "complex_circuits"
TOOLS_ROOT: Path = BENCH_ROOT / "tools"

if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

# The shared AnalogGym tools require an explicit technology tree at import.
os.environ.setdefault("AG_TREE", str(BENCH_ROOT / "designs_tsmc5"))

from tests.common import nn_gate  # noqa: E402
from tests.simple_circuits import verify_nn_ac  # noqa: E402
from examples.complex_circuits.pycircuitsim_bench import (  # noqa: E402
    AnalysisPlan,
    DeckOptions,
    SweepResult,
    TranslatedDeck,
)
from examples.complex_circuits.pycircuitsim_bench import (  # noqa: E402
    campaign,
    run_compare,
    translate,
)

import build_amp  # noqa: E402


def _translated_deck(kind: str, *, has_ic: bool = False) -> TranslatedDeck:
    """Return the smallest translated deck accepted by ``_op_delta_for``."""
    return TranslatedDeck(
        tech="tsmc5",
        category="charge_pump" if has_ic else "amplifier",
        design="regression",
        deck="tb_tran.cir" if kind == "tran" else "tb_gain.cir",
        design_dir=BENCH_ROOT,
        netlist_text="",
        modelcard_path=BENCH_ROOT / "unused-modelcard.spice",
        plans=[AnalysisPlan(kind=kind, control=kind)],
        meas=[],
        nodesets={},
        ic={"out": 0.25} if has_ic else {},
        params={},
        options=DeckOptions(),
        devices=0,
        multipliers={},
        stability=None,
        temp_c=None,
        warnings=[],
    )


def _sweep(kind: str, source: str, values: np.ndarray,
           *, operating_point: Optional[Dict[str, float]] = None) -> SweepResult:
    """Build a one-node sweep with an optional PyCircuitSim operating point."""
    return SweepResult(
        kind=kind,
        x_name="time" if kind == "tran" else "frequency",
        x=np.arange(values.size, dtype=float),
        v={"out": values},
        i={},
        ok=None,
        source=source,
        meta=({"operating_point": operating_point}
              if operating_point is not None else {}),
    )


def verify_transient_op_uses_reference_time_zero() -> None:
    """A transient startup must be compared with NGSPICE's actual first point."""
    td = _translated_deck("tran", has_ic=True)
    py = _sweep("tran", "pycircuitsim", np.array([0.25, 0.30]),
                operating_point={"out": 0.25})
    ng = _sweep("tran", "ngspice", np.array([0.25, 0.31]))

    with patch.object(
        run_compare,
        "ngspice_operating_point",
        side_effect=AssertionError("transient diagnostics must not run a new .op"),
    ):
        report = run_compare._op_delta_for(
            td, [py], [ng], BENCH_ROOT, run_compare.SimOptions()
        )

    assert report is not None
    assert report["source"] == "transient-time-zero"
    assert report["worst_abs"] == 0.0


def verify_failed_ic_transient_does_not_compare_unconstrained_op() -> None:
    """Without an NG transient point, an ``.ic`` startup has no fair OP peer."""
    td = _translated_deck("tran", has_ic=True)
    py = _sweep("tran", "pycircuitsim", np.array([0.25]),
                operating_point={"out": 0.25})

    with patch.object(
        run_compare,
        "ngspice_operating_point",
        side_effect=AssertionError("an unconstrained .op is not the .ic startup"),
    ):
        report = run_compare._op_delta_for(
            td, [py], [], BENCH_ROOT, run_compare.SimOptions()
        )

    assert report is None


def verify_both_failed_ic_transient_does_not_compare_unconstrained_op() -> None:
    """The deck plan still identifies a transient when neither engine swept."""
    td = _translated_deck("tran", has_ic=True)

    with patch.object(
        run_compare,
        "ngspice_operating_point",
        side_effect=AssertionError("an unconstrained .op is not the .ic startup"),
    ):
        report = run_compare._op_delta_for(
            td,
            [],
            [],
            BENCH_ROOT,
            run_compare.SimOptions(),
            partial_op={"out": 0.25},
        )

    assert report is None


def verify_failed_unconstrained_transient_uses_operating_point() -> None:
    """A failed transient without ``.ic`` still has a comparable startup OP."""
    td = _translated_deck("tran")
    py = _sweep("tran", "pycircuitsim", np.array([0.25]),
                operating_point={"out": 0.25})

    with patch.object(
        run_compare, "ngspice_operating_point", return_value={"out": 0.25}
    ) as mocked_op:
        report = run_compare._op_delta_for(
            td, [py], [], BENCH_ROOT, run_compare.SimOptions()
        )

    mocked_op.assert_called_once()
    assert report is not None
    assert report["source"] == "ngspice-op-run (failed transient's operating point)"
    assert report["worst_abs"] == 0.0


def verify_ac_still_uses_dedicated_operating_point() -> None:
    """AC continues to compare the operating point it linearized around."""
    td = _translated_deck("ac")
    py = _sweep("ac", "pycircuitsim", np.array([0.25]),
                operating_point={"out": 0.25})
    ng = _sweep("ac", "ngspice", np.array([99.0]))

    with patch.object(
        run_compare, "ngspice_operating_point", return_value={"out": 0.25}
    ) as mocked_op:
        report = run_compare._op_delta_for(
            td, [py], [ng], BENCH_ROOT, run_compare.SimOptions()
        )

    mocked_op.assert_called_once()
    assert report is not None
    assert report["source"] == "ngspice-op-run"
    assert report["worst_abs"] == 0.0


def verify_voltage_error_stats_are_aggregatable() -> None:
    """Rows must retain sufficient statistics for a per-tech NN error report."""
    report = run_compare.op_delta(
        {"a": 1.0, "b": 3.0}, {"a": 1.0, "b": 2.0}
    )
    stats = report["error_stats"]
    assert stats["n"] == 2
    assert stats["sum_squared_error"] == 1.0
    assert stats["truth_sum"] == 3.0
    assert stats["truth_sum_squared"] == 5.0
    assert np.isclose(stats["mre"], 1.0 / 6.0)
    assert np.isclose(stats["r2"], -1.0)
    assert np.isclose(stats["nrmse"], np.sqrt(0.5))
    assert stats["max_error"] == 1.0


def verify_multi_segment_voltage_stats_are_complete() -> None:
    """Every comparable segment must contribute to campaign voltage error."""
    td = _translated_deck("dc")
    py = [
        _sweep("dc", "pycircuitsim", np.array([1.0, 2.0])),
        _sweep("dc", "pycircuitsim", np.array([3.0, 4.0])),
    ]
    ng = [
        _sweep("dc", "ngspice", np.array([1.0, 1.0])),
        _sweep("dc", "ngspice", np.array([1.0, 1.0])),
    ]
    report = run_compare._op_delta_for(
        td, py, ng, BENCH_ROOT, run_compare.SimOptions()
    )
    assert report is not None
    assert report["segments"] == 2
    assert report["error_stats"]["n"] == 4
    assert report["error_stats"]["sum_squared_error"] == 14.0
    assert report["worst_abs"] == 3.0


def verify_directnet_model_stub_is_explicit() -> None:
    """An NN deck must retain its technology and model-card VT flavor."""
    assert translate._model_stub(
        "nsvt_l32_f7", "NMOS", tech="tsmc5", model_level=73
    ) == ".model nsvt_l32_f7 NMOS (LEVEL=73 TECH=tsmc5 VT=svt)"
    assert translate._model_stub(
        "phvt_l90_f4", "PMOS", tech="tsmc16", model_level=73
    ) == ".model phvt_l90_f4 PMOS (LEVEL=73 TECH=tsmc16 VT=hvt)"


def verify_nn_temperature_sweep_rebinds_geometry() -> None:
    """A temperature sweep must update the NN input and invalidate its cache."""
    import torch

    from pycircuitsim.models.mosfet_directnet import NMOS_NN

    device = object.__new__(NMOS_NN)
    device.NFIN = 4.0
    device.L = 32e-9
    device.temperature = 300.15
    device._norm_key = ("norm", 0, 0)
    device._device = torch.device("cpu")
    device._norm_stats = SimpleNamespace(
        input_mean=np.zeros(7), input_std=np.ones(7)
    )
    device._eval_cache = {"id": 1.0}
    device._cache_voltages = (0.1, 0.2, 0.0, 0.0)
    device._cache_has_caps = True
    device._q_prev = {"qg": 1.0}
    device._q_prev2 = {"qg": 1.0}
    device._v_prev_tran = {"d": 0.1}

    device.set_temperature(398.15)

    assert device.temperature == 398.15
    assert np.isclose(float(device._geo_norm_t[2]), 398.15)
    assert device._eval_cache is None
    assert device._q_prev is None
    assert device._q_prev2 is None
    assert device._v_prev_tran is None

    circuit = SimpleNamespace(components=[device])
    assert run_compare._mosfets(circuit) == [device]


def verify_campaign_propagates_deck_failures() -> None:
    """A failed campaign child must make the campaign itself fail loudly."""
    with TemporaryDirectory() as temp_dir, patch.object(
        campaign,
        "corpus",
        return_value=[("amplifier", "broken", "tb_gain.cir")],
    ), patch.object(
        campaign,
        "_current_commit",
        return_value="0123456789abcdef",
    ), patch.object(
        campaign,
        "run_deck",
        return_value={
            "category": "amplifier",
            "design": "broken",
            "deck": "tb_gain.cir",
            "rc": 2,
            "log": "broken.log",
        },
    ):
        rc = campaign.main(
            [
                "--tech",
                "tsmc5",
                "--families",
                "ac",
                "--jobs",
                "1",
                "--out",
                temp_dir,
            ]
        )
        summary = (Path(temp_dir) / "summary.md").read_text()
    assert rc != 0, "a failed child process was reported as campaign success"
    assert "(1 quarantined)" not in summary, (
        "a missing failed row was mislabeled as an invalid test example"
    )


def verify_modelcard_materializer_reads_generated_geometry() -> None:
    """Private model libraries must be reproducible without rewriting decks."""
    from examples.complex_circuits.tools.materialize_modelcards import (
        model_specs,
    )

    text = """
Nm1 d g s b nsvt_l32_f7 m=2
Nm2 d g s b phvt_l90_f4 m=1
Nm3 d g s b nsvt_l32_f7 m=8
"""
    assert model_specs(text) == {
        "nsvt_l32_f7": ("n", "svt", 32 * 1e-9, 7),
        "phvt_l90_f4": ("p", "hvt", 90 * 1e-9, 4),
    }


def verify_modelcard_materializer_writes_and_rejects_alias_drift() -> None:
    """Tree discovery, alias round-trip, and writes are one tested contract."""
    from examples.complex_circuits.tools.materialize_modelcards import (
        materialize_tree,
    )

    with TemporaryDirectory() as temp_dir:
        tree = Path(temp_dir)
        design = tree / "amplifier" / "example"
        design.mkdir(parents=True)
        (design / "netlist.spice").write_text(
            "Nm1 d g s b nsvt_l32_f7 m=1\n")

        class FakeLibrary:
            def model_name(self, kind: str, vt: str, length_m: float,
                           nfin: int) -> str:
                return f"{kind}{vt}_l{round(length_m * 1e9)}_f{nfin}"

            def write(self, path: Path) -> None:
                path.write_text("materialized\n")

        module = ModuleType("pycmg_lib")
        module.MODELS_FILE = "private_models.spice"  # type: ignore[attr-defined]
        module.ModelLibrary = FakeLibrary  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"pycmg_lib": module}), patch.dict(
            os.environ, {}, clear=False
        ):
            assert materialize_tree(tree) == 1
        output = design / "private_models.spice"
        assert output.read_text() == "materialized\n"

        class DriftedLibrary(FakeLibrary):
            def model_name(self, kind: str, vt: str, length_m: float,
                           nfin: int) -> str:
                return "wrong_alias"

        output.unlink()
        module.ModelLibrary = DriftedLibrary  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"pycmg_lib": module}):
            try:
                materialize_tree(tree)
            except RuntimeError as exc:
                assert "round-trip changed" in str(exc)
            else:
                raise AssertionError("alias drift was accepted")
        assert not output.exists()


def verify_campaign_resume_is_checkpoint_exact() -> None:
    """A same-stem retrain must not reuse rows from the previous weights."""
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        checkpoints = {}
        for device in ("nmos", "pmos"):
            stem = f"tsmc5_dn_large_{device}"
            checkpoint = root / f"{stem}_best.pt"
            norm = root / f"{stem}_norm.npz"
            checkpoint.write_bytes(f"weights-{device}".encode())
            norm.write_bytes(f"norm-{device}".encode())
            checkpoint.with_suffix(".pt.complete").touch()
            checkpoints[device] = {
                "selection": "explicit",
                "stem": stem,
                "checkpoint_sha256": run_compare.file_sha256(checkpoint),
                "norm_sha256": run_compare.file_sha256(norm),
                "complete": True,
            }
        row_path = root / "row.json"
        row_path.write_text(json.dumps({
            "code_commit": "old-code",
            "ground_truth": {"family": "bsim_cmg", "level": 72},
            "py_model": {
                "family": "directnet",
                "level": 73,
                "tech": "tsmc5",
                "checkpoints": checkpoints,
            }
        }))

        with patch.object(campaign, "CHECKPOINT_DIR", root), patch.object(
            campaign, "artifact_record_is_current", return_value=True
        ), patch.object(
            campaign, "executable_record_is_current", return_value=True
        ):
            assert campaign._row_matches_model(row_path, "tsmc5", 73, "large")
            assert campaign._row_matches_model(
                row_path, "tsmc5", 73, "large", "old-code"
            )
            assert not campaign._row_matches_model(
                row_path, "tsmc5", 73, "large", "new-code"
            )
            (root / "tsmc5_dn_large_nmos_best.pt").write_bytes(
                b"retrained weights"
            )
            assert not campaign._row_matches_model(
                row_path, "tsmc5", 73, "large"
            )


def verify_directnet_provenance_resolves_every_pin_branch() -> None:
    """Scored rows must identify exact polarity-specific model artifacts."""
    from neural_network import config as nn_config

    td = replace(_translated_deck("ac"), model_level=73)
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        for device in ("nmos", "pmos"):
            stem = f"tsmc5_dn_large_{device}"
            (root / f"{stem}_best.pt").write_bytes(device.encode())
            (root / f"{stem}_norm.npz").write_bytes(f"norm-{device}".encode())
            (root / f"{stem}_best.pt.complete").touch()
        pins = {
            "PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS": "tsmc5_dn_large_nmos",
            "PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS": "tsmc5_dn_large_pmos",
        }
        with patch.object(nn_config, "CHECKPOINT_DIR", root), patch.dict(
            os.environ, pins, clear=True
        ):
            record = run_compare._model_provenance(td)
        assert record["checkpoints"]["nmos"]["selection"] == "explicit"
        assert record["checkpoints"]["pmos"]["complete"] is True
        assert record["checkpoints"]["nmos"]["checkpoint_sha256"] == (
            run_compare.file_sha256(
                root / "tsmc5_dn_large_nmos_best.pt")
        )

        with patch.object(nn_config, "CHECKPOINT_DIR", root), patch.dict(
            os.environ, {}, clear=True
        ):
            automatic = run_compare._model_provenance(td)
        assert automatic["checkpoints"]["nmos"] == {"selection": "automatic"}

        with patch.object(nn_config, "CHECKPOINT_DIR", root), patch.dict(
            os.environ,
            {"PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS":
             "tsmc5_dn_large_pmos"},
            clear=True,
        ):
            try:
                run_compare._model_provenance(td)
            except run_compare.SimFailure as exc:
                assert "opposite polarity" in str(exc)
            else:
                raise AssertionError("opposite-polarity pin was accepted")

        with patch.object(nn_config, "CHECKPOINT_DIR", root), patch.dict(
            os.environ,
            {"PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS": "missing_nmos"},
            clear=True,
        ):
            try:
                run_compare._model_provenance(td)
            except run_compare.SimFailure as exc:
                assert "cannot resolve" in str(exc)
            else:
                raise AssertionError("missing checkpoint pin was accepted")


def verify_reference_provenance_detects_artifact_changes() -> None:
    """Resume eligibility must include ignored ground-truth artifacts."""
    from examples.complex_circuits.pycircuitsim_bench.provenance import (
        artifact_record,
        artifact_record_is_current,
    )

    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "modelcard.spice"
        path.write_text("version one\n")
        record = artifact_record(path)
        assert artifact_record_is_current(record)
        path.write_text("version two changed\n")
        assert not artifact_record_is_current(record)


def verify_partial_campaign_summary_counts_missing_rows() -> None:
    """A valid row followed by a missing row must retain an integer count."""
    entries = [
        ("amplifier", "complete", "tb_gain.cir"),
        ("amplifier", "missing", "tb_gain.cir"),
    ]
    verdict = {
        "ran": True,
        "measured": 1,
        "agree": 1,
        "missing_py": 0,
        "not_comparable": 0,
        "engine_ok": True,
        "op_worst_abs": 0.0,
        "py_seconds": 0.1,
        "ng_seconds": 0.1,
    }
    with TemporaryDirectory() as temp_dir, patch.object(
        campaign, "corpus", return_value=entries
    ):
        out = Path(temp_dir)
        (out / "tsmc5_amplifier_complete_tb_gain.json").write_text(
            json.dumps({"verdict": verdict})
        )
        with patch.object(campaign, "_model_matches_row", return_value=True):
            summary = campaign.summarize("tsmc5", out, ["ac"])
    assert "**ac: 1/1 decks fully agree** (1 missing)" in summary


def verify_campaign_summary_surfaces_simulator_failures() -> None:
    """Measurement-control success must not hide a failed simulator."""
    verdict = {
        "ran": False,
        "ng_ran": True,
        "measured": 0,
        "agree": 0,
        "missing_py": 8,
        "not_comparable": 0,
        "engine_ok": True,
        "op_worst_abs": None,
        "py_seconds": 3.0,
        "ng_seconds": 0.1,
    }
    with TemporaryDirectory() as temp_dir, patch.object(
        campaign,
        "corpus",
        return_value=[("amplifier", "failed", "tb_gain.cir")],
    ):
        out = Path(temp_dir)
        (out / "tsmc5_amplifier_failed_tb_gain.json").write_text(
            json.dumps({"verdict": verdict})
        )
        with patch.object(campaign, "_model_matches_row", return_value=True):
            summary = campaign.summarize("tsmc5", out, ["ac"])
    assert "| PY FAIL |" in summary


def verify_campaign_pins_transient_policy_and_clears_diagnostics() -> None:
    """Parent diagnostic variables must not mutate a scored child policy."""
    captured: Dict[str, str] = {}

    def fake_run(*args: Any, **kwargs: Any) -> SimpleNamespace:
        del args
        captured.update(kwargs["env"])
        return SimpleNamespace(returncode=0)

    polluted = {
        "PYCIRCUITSIM_BENCH_TRAN_METHOD": "gear2",
        "PYCIRCUITSIM_BENCH_TRAN_SUBSTEPS": "99",
        "PYCIRCUITSIM_BENCH_TRAN_REFINE": "1",
        "PYCIRCUITSIM_BENCH_TRAN_TMAX": "1",
    }
    with TemporaryDirectory() as temp_dir, patch.dict(
        os.environ, polluted, clear=False
    ), patch.object(campaign.subprocess, "run", side_effect=fake_run):
        root = Path(temp_dir)
        campaign.run_deck(
            "tsmc5", "amplifier", "example", "tb_tran.cir",
            root / "out", root / "work", False, 10.0, 72, "large",
            "commit",
        )
    for name in campaign._TRANSIENT_DIAGNOSTIC_ENV:
        assert name not in captured
    policy = json.loads(captured["PYCIRCUITSIM_BENCH_CAMPAIGN_POLICY"])
    assert policy == {
        "refine_requested": False,
        "stride": 1,
        "transient_method_override": None,
        "transient_refine": False,
    }


def verify_campaign_manifest_rejects_changed_refine_policy() -> None:
    """A resume cannot mix refined and unrefined transient semantics."""
    with TemporaryDirectory() as temp_dir, patch.object(
        campaign, "_current_commit", return_value="commit"
    ):
        out = Path(temp_dir)
        campaign._campaign_provenance(
            out, "tsmc5", ["tran"], 73, "large", True,
            summarize_only=False,
        )
        try:
            campaign._campaign_provenance(
                out, "tsmc5", ["tran"], 73, "large", False,
                summarize_only=True,
            )
        except SystemExit as exc:
            assert "refine" in str(exc)
        else:
            raise AssertionError("changed transient policy was accepted")


def verify_campaign_requeues_failed_rows_and_invalidates_force() -> None:
    """Persisted failures and stale forced rows cannot produce success."""
    entry = ("amplifier", "broken", "tb_gain.cir")
    verdict = {"ran": False, "ng_ran": True}
    with TemporaryDirectory() as temp_dir:
        out = Path(temp_dir)
        row_path = out / "tsmc5_amplifier_broken_tb_gain.json"
        row_path.write_text(json.dumps({"verdict": verdict}))
        observed_absent: list[bool] = []

        def failed_run(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            del args, kwargs
            observed_absent.append(not row_path.exists())
            return {
                "category": "amplifier", "design": "broken",
                "deck": "tb_gain.cir", "rc": 2, "log": "broken.log",
            }

        with patch.object(campaign, "corpus", return_value=[entry]), \
                patch.object(campaign, "_current_commit", return_value="commit"), \
                patch.object(campaign, "_row_matches_model", return_value=True), \
                patch.object(campaign, "run_deck", side_effect=failed_run), \
                patch.object(campaign, "_model_matches_row", return_value=True):
            rc = campaign.main([
                "--tech", "tsmc5", "--families", "ac", "--jobs", "1",
                "--out", temp_dir,
            ])
        assert rc != 0
        assert observed_absent == [True]

        row_path.write_text(json.dumps({"verdict": {"ran": True,
                                                     "ng_ran": True}}))
        observed_absent.clear()
        with patch.object(campaign, "corpus", return_value=[entry]), \
                patch.object(campaign, "_current_commit", return_value="commit"), \
                patch.object(campaign, "run_deck", side_effect=failed_run), \
                patch.object(campaign, "_model_matches_row", return_value=True):
            rc = campaign.main([
                "--tech", "tsmc5", "--families", "ac", "--jobs", "1",
                "--out", temp_dir, "--force",
            ])
        assert rc != 0
        assert observed_absent == [True]


def verify_summarize_only_fails_on_incomplete_evidence() -> None:
    """A human-readable MISSING row must also produce a nonzero command."""
    entry = ("amplifier", "missing", "tb_gain.cir")
    with TemporaryDirectory() as temp_dir:
        out = Path(temp_dir)
        (out / "campaign_provenance.json").write_text(json.dumps({
            "code_commit": "commit", "tech": "tsmc5", "families": ["ac"],
            "model_level": 72, "checkpoint_size": "large", "refine": False,
        }))
        with patch.object(campaign, "corpus", return_value=[entry]):
            rc = campaign.main([
                "--tech", "tsmc5", "--families", "ac",
                "--out", temp_dir, "--summarize-only",
            ])
        assert rc != 0


def verify_failed_simulation_retains_elapsed_time() -> None:
    """A failed long-running solve must not be reported as zero seconds."""
    td = _translated_deck("ac")
    with TemporaryDirectory() as temp_dir, patch.object(
        run_compare,
        "ngspice_sweep",
        side_effect=run_compare.SimFailure("reference unavailable"),
    ), patch.object(
        run_compare,
        "simulate",
        side_effect=run_compare.SimFailure("solver exhausted"),
    ), patch.object(
        run_compare.time,
        "perf_counter",
        side_effect=(10.0, 13.0),
    ), patch.object(
        run_compare,
        "_reference_provenance",
        return_value={"family": "bsim_cmg", "level": 72},
    ):
        row = run_compare.compare_translated(td, Path(temp_dir))
    assert row["pycircuitsim"]["seconds"] == 3.0


def verify_large_directnet_checkpoints_enable_shared_gates() -> None:
    """Production-large files must satisfy the shared gate's sentinel."""
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        for device in ("nmos", "pmos"):
            (root / f"tsmc12_dn_large_{device}_best.pt").touch()
        with patch.object(nn_gate, "CHECKPOINT_DIR", root), patch.object(
            nn_gate, "_env_pin", return_value=(None, None)
        ):
            checkpoints = nn_gate.get_available_checkpoints()
    assert checkpoints["directnet_v4_nmos"].name == (
        "tsmc12_dn_large_nmos_best.pt"
    )
    assert checkpoints["directnet_v4_pmos"].name == (
        "tsmc12_dn_large_pmos_best.pt"
    )


def verify_ac_gate_rejects_nonconverged_operating_point() -> None:
    """Numerical AC agreement cannot validate a non-fixed linearization."""
    metrics = {
        "gain0_db_err": 0.0,
        "f3db_ratio": 1.0,
        "mag_nrmse": 0.0,
    }
    assert verify_nn_ac.ac_gate_passes(True, True, metrics)
    assert not verify_nn_ac.ac_gate_passes(False, True, metrics)


def verify_analoggym_rejects_nonconverged_ac_operating_point() -> None:
    """The campaign must not score an AC solve about a non-fixed state."""
    td = _translated_deck("ac")
    plan = td.plans[0]
    meta: Dict[str, object] = {}
    with patch.object(
        run_compare,
        "_dc_solve",
        return_value=({"out": 0.25}, False, "NR exhausted"),
    ):
        try:
            run_compare._simulate_ac(
                td,
                plan,
                SimpleNamespace(),
                {"out": "out"},
                run_compare.SimOptions(),
                0.65,
                meta,
            )
        except run_compare.SimFailure as exc:
            assert "refusing AC linearization" in str(exc)
        else:
            raise AssertionError("nonconverged AC operating point was scored")


def verify_only_validated_alternate_seed_is_emitted() -> None:
    """Only the fallback that the current corpus actually exercises is shipped."""
    assert build_amp.ALT_NODESET_DECKS == {("TSMC5", "qu2017_azc_pin_3")}
    design = build_amp.AmpDesign(
        vdd=0.65,
        vcm=0.1625,
        cload=500e-12,
        gbw_ideal=1e6,
        roles={},
        passives={},
    )
    fan = build_amp.Topology(subckt="Fan_SMC_Pin_3", ports=[])
    qu2017 = build_amp.Topology(subckt="Qu2017_AZC_Pin_3", ports=[])

    with patch.object(build_amp, "TECH", "TSMC5"):
        fan_decks = build_amp.emit_testbenches(fan, design)
        qu2017_decks = build_amp.emit_testbenches(qu2017, design)

    assert "tb_tran_altns.cir" not in fan_decks
    assert "tb_tran_altns.cir" in qu2017_decks

    actual = {
        path.relative_to(BENCH_ROOT).as_posix()
        for path in BENCH_ROOT.glob(
            "designs_tsmc*/amplifier/*/tb_tran_altns.cir"
        )
    }
    expected = {
        "designs_tsmc5/amplifier/Qu2017_AZC_Pin_3/tb_tran_altns.cir"
    }
    assert actual == expected, (
        f"unqualified alternate-seed decks: {actual - expected}"
    )


def verify_write_design_reconciles_alternate_seed_inventory() -> None:
    """A regeneration removes stale helpers and retains the allowlisted one."""
    design = build_amp.AmpDesign(
        vdd=0.65,
        vcm=0.1625,
        cload=500e-12,
        gbw_ideal=1e6,
        roles={},
        passives={},
    )
    fan = build_amp.Topology(subckt="Fan_SMC_Pin_3", ports=[])
    qu2017 = build_amp.Topology(subckt="Qu2017_AZC_Pin_3", ports=[])

    with TemporaryDirectory() as temp_dir, patch.object(
        build_amp, "TECH", "TSMC5"
    ):
        output_root = Path(temp_dir)
        fan_dir = output_root / "fan"
        fan_dir.mkdir()
        stale_helper = fan_dir / "tb_tran_altns.cir"
        stale_helper.write_text("obsolete\n")
        build_amp.write_design(fan_dir, fan, design)
        assert not stale_helper.exists()

        qu2017_dir = output_root / "qu2017"
        build_amp.write_design(qu2017_dir, qu2017, design)
        assert (qu2017_dir / "tb_tran_altns.cir").is_file()


def verify_regeneration_fails_loudly_without_source_corpus() -> None:
    """A missing untracked AnalogGym source tree must not report success."""
    with TemporaryDirectory() as temp_dir:
        isolated_tree = Path(temp_dir) / "designs_tsmc5"
        isolated_tree.mkdir()
        env = dict(os.environ, AG_TREE=str(isolated_tree))
        result = subprocess.run(
            [
                sys.executable,
                str(TOOLS_ROOT / "regen_decks.py"),
                "amplifier",
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    output = result.stdout + result.stderr
    assert result.returncode != 0, "regeneration hid a completely failed run"
    assert "source corpus" in output.lower()


def verify_regeneration_preflights_incomplete_source_corpus() -> None:
    """Missing inputs must abort before an existing generated deck is touched."""
    with TemporaryDirectory() as temp_dir:
        isolated_root = Path(temp_dir)
        isolated_tree = isolated_root / "designs_tsmc5"
        design_dir = isolated_tree / "amplifier" / "Fan_SMC_Pin_3"
        design_dir.mkdir(parents=True)
        (design_dir / "design.json").write_text("{}\n")
        generated_netlist = design_dir / "netlist.spice"
        generated_netlist.write_text("unchanged generated deck\n")
        (isolated_root / "designs").mkdir()

        env = dict(os.environ, AG_TREE=str(isolated_tree))
        result = subprocess.run(
            [
                sys.executable,
                str(TOOLS_ROOT / "regen_decks.py"),
                "amplifier",
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        output = result.stdout + result.stderr
        assert result.returncode != 0, "incomplete sources reported success"
        assert "missing source inputs" in output.lower()
        assert generated_netlist.read_text() == "unchanged generated deck\n"


def main() -> None:
    """Run the focused checks without requiring pytest discovery."""
    checks = (
        verify_transient_op_uses_reference_time_zero,
        verify_failed_ic_transient_does_not_compare_unconstrained_op,
        verify_both_failed_ic_transient_does_not_compare_unconstrained_op,
        verify_failed_unconstrained_transient_uses_operating_point,
        verify_ac_still_uses_dedicated_operating_point,
        verify_voltage_error_stats_are_aggregatable,
        verify_multi_segment_voltage_stats_are_complete,
        verify_directnet_model_stub_is_explicit,
        verify_nn_temperature_sweep_rebinds_geometry,
        verify_campaign_propagates_deck_failures,
        verify_modelcard_materializer_reads_generated_geometry,
        verify_modelcard_materializer_writes_and_rejects_alias_drift,
        verify_campaign_resume_is_checkpoint_exact,
        verify_directnet_provenance_resolves_every_pin_branch,
        verify_reference_provenance_detects_artifact_changes,
        verify_partial_campaign_summary_counts_missing_rows,
        verify_campaign_summary_surfaces_simulator_failures,
        verify_campaign_pins_transient_policy_and_clears_diagnostics,
        verify_campaign_manifest_rejects_changed_refine_policy,
        verify_campaign_requeues_failed_rows_and_invalidates_force,
        verify_summarize_only_fails_on_incomplete_evidence,
        verify_failed_simulation_retains_elapsed_time,
        verify_large_directnet_checkpoints_enable_shared_gates,
        verify_ac_gate_rejects_nonconverged_operating_point,
        verify_analoggym_rejects_nonconverged_ac_operating_point,
        verify_only_validated_alternate_seed_is_emitted,
        verify_write_design_reconciles_alternate_seed_inventory,
        verify_regeneration_fails_loudly_without_source_corpus,
        verify_regeneration_preflights_incomplete_source_corpus,
    )
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print(f"\nAnalogGym migration regressions: {len(checks)}/{len(checks)} PASS")


if __name__ == "__main__":
    main()
