#!/usr/bin/env python3
"""Regression checks for the AnalogGym comparison harness and deck corpus."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from typing import Dict, Optional
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
                "checkpoint_sha256": run_compare._sha256(checkpoint),
                "norm_sha256": run_compare._sha256(norm),
                "complete": True,
            }
        row_path = root / "row.json"
        row_path.write_text(json.dumps({
            "py_model": {
                "family": "directnet",
                "level": 73,
                "tech": "tsmc5",
                "checkpoints": checkpoints,
            }
        }))

        with patch.object(campaign, "CHECKPOINT_DIR", root):
            assert campaign._row_matches_model(row_path, "tsmc5", 73, "large")
            (root / "tsmc5_dn_large_nmos_best.pt").write_bytes(
                b"retrained weights"
            )
            assert not campaign._row_matches_model(
                row_path, "tsmc5", 73, "large"
            )


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
        summary = campaign.summarize("tsmc5", out, ["ac"])
    assert "**ac: 1/1 decks fully agree** (1 missing)" in summary


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
        verify_directnet_model_stub_is_explicit,
        verify_nn_temperature_sweep_rebinds_geometry,
        verify_campaign_propagates_deck_failures,
        verify_modelcard_materializer_reads_generated_geometry,
        verify_campaign_resume_is_checkpoint_exact,
        verify_partial_campaign_summary_counts_missing_rows,
        verify_large_directnet_checkpoints_enable_shared_gates,
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
