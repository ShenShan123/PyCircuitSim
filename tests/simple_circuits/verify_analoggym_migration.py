#!/usr/bin/env python3
"""Regression checks for the AnalogGym comparison harness and deck corpus."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional
from unittest.mock import patch

import numpy as np


REPO_ROOT: Path = Path(__file__).resolve().parents[2]
BENCH_ROOT: Path = REPO_ROOT / "examples" / "complex_circuits"
TOOLS_ROOT: Path = BENCH_ROOT / "tools"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

# The shared AnalogGym tools require an explicit technology tree at import.
os.environ.setdefault("AG_TREE", str(BENCH_ROOT / "designs_tsmc5"))

from examples.complex_circuits.pycircuitsim_bench import (  # noqa: E402
    AnalysisPlan,
    DeckOptions,
    SweepResult,
    TranslatedDeck,
)
from examples.complex_circuits.pycircuitsim_bench import run_compare  # noqa: E402

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
