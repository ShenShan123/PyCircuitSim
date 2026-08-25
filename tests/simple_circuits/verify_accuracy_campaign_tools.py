#!/usr/bin/env python3
"""Regression checks for accuracy-campaign completeness and verdict hygiene."""
from __future__ import annotations

from contextlib import redirect_stdout
import fcntl
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import v710_regate_collect as collect
from scripts import v710_regate_jobs as jobs
from scripts import v730_coverage as coverage
from scripts import v730_docs_build as docs
from pycircuitsim.solver import (
    _mna_residual_at_node_voltages,
    _mna_residual_inf,
)
from tests.simple_circuits import verify_complex_opamp_ac as opamp_ac
from tests.simple_circuits import verify_nn_ac


def _check_residual_completes_voltage_source_currents() -> None:
    """KCL residuals must solve the MNA branch-current tail, not assume zero."""
    mna = np.array([[2.0, 1.0], [1.0, 0.0]])
    rhs = np.array([0.0, 1.0])
    zero_tail = np.array([1.0, 0.0])
    assert _mna_residual_inf(mna, rhs, zero_tail) == 2.0

    residual, current_scale = _mna_residual_at_node_voltages(
        mna, rhs, np.array([1.0]), num_nodes=1,
    )
    assert residual < 1e-12
    assert current_scale == 0.0


def _check_nn_ac_banner_tracks_forced_family() -> None:
    """Gate output must identify the NN family actually selected by the parser."""
    with patch.dict(os.environ, {}, clear=True):
        assert verify_nn_ac.active_model_label() == "DirectNet (LEVEL=73)"
    with patch.dict(os.environ, {"PYCIRCUITSIM_NN_FORCE_LEVEL": "74"}, clear=True):
        assert verify_nn_ac.active_model_label() == "BSIM-AR (LEVEL=74)"
    with patch.dict(os.environ, {"PYCIRCUITSIM_NN_FORCE_LEVEL": "75"}, clear=True):
        assert verify_nn_ac.active_model_label() == "PFN (LEVEL=75)"


def _check_opamp_ac_refines_bias_and_requires_fixed_point() -> None:
    """A narrow transition needs a fine sweep and a converged AC fixed point."""
    center = 0.501

    def transition(lo: float, hi: float, step: float) -> tuple[np.ndarray, np.ndarray]:
        vin = np.arange(lo, hi + 0.5 * step, step)
        vout = 1.0 / (1.0 + np.exp((vin - center) / 0.00015))
        return vin, vout

    coarse_vin, coarse_vout = transition(0.45, 0.55, 0.002)
    _coarse_bias, coarse_output = opamp_ac._peak_gain_bias(
        coarse_vin, coarse_vout,
    )
    assert not 0.15 < coarse_output < 0.85

    fine_bias, fine_output = opamp_ac._refined_peak_gain_bias(
        coarse_vin, coarse_vout, transition,
    )
    assert abs(fine_bias - center) <= 0.0001
    assert 0.15 < fine_output < 0.85

    metrics = {"gain0_db_err": 0.0, "gbw_ratio": 1.0, "pm_err": 0.0}
    assert opamp_ac.opamp_ac_gate_passes(True, True, metrics)
    assert not opamp_ac.opamp_ac_gate_passes(False, True, metrics)
    assert not opamp_ac.opamp_ac_gate_passes(True, False, metrics)


def _check_verdict_predicate(predicate: Callable[[object], bool]) -> None:
    """Require scientific exits while rejecting infrastructure outcomes."""
    assert predicate({"rc": "0"})
    assert predicate({"rc": "1"})
    for rc in ("RACED", "no-ckpt", "-1", "124", "126", None):
        assert not predicate({"rc": rc}), rc


def _check_log_scan() -> None:
    """Raw logs must override stale JSON, including unfinished attempts."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        cell = root / "dn" / "small" / "tsmc5"
        cell.mkdir(parents=True)
        (cell / "pass.omp1.log").write_text("===V710_DONE rc=0===\n")
        (cell / "fail.omp1.log").write_text("===V710_DONE rc=1===\n")
        (cell / "race.omp1.log").write_text(
            "===V710_DONE rc=0===\n===V710_DONE rc=1===\n"
        )
        (cell / "malformed.omp1.log").write_text(
            "===V710_DONE malformed===\n===V710_DONE rc=0===\n"
        )
        (cell / "timeout.omp1.log").write_text("===V710_DONE rc=124===\n")
        (cell / "killed.omp1.log").write_text("===V710_DONE rc=137===\n")
        (cell / "unfinished.omp1.log").write_text("suite still running\n")
        scanned = coverage.scan_logs(root)
        assert {key[3] for key in scanned} == {
            "pass", "fail", "race", "malformed", "timeout", "killed",
            "unfinished",
        }
        assert {key[3] for key, rc in scanned.items() if rc is not None} == {
            "pass", "fail",
        }

        stale_suites = {
            suite: {"TSMC5": {"omp1": {"rc": "0"}}}
            for suite in ("race", "malformed", "timeout", "killed", "unfinished")
        }
        stale_suites.update({
            "pass": {"TSMC5": {"omp1": {"rc": "0"}}},
            "fail": {"TSMC5": {"omp1": {"rc": "0"}}},
        })
        (root / "data.json").write_text(json.dumps({
            "dn": {"small": stale_suites},
        }))
        old_passes = coverage.PASSES
        coverage.PASSES = [("test", root)]
        try:
            index = coverage.build_index(["test"])
        finally:
            coverage.PASSES = old_passes
        assert {key[3] for key in index} == {"pass", "fail"}


def _check_lock_contention() -> None:
    """A contended dispatcher must not report a completed campaign cell."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        out = root / "results"
        log = out / "dn" / "small" / "tsmc5" / "verify_nn_ac.omp1.log"
        log.parent.mkdir(parents=True)
        job_file = root / "jobs.txt"
        job_file.write_text("dn small TSMC5 verify_nn_ac 1\n")
        with Path(f"{log}.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            env = os.environ.copy()
            env.update({
                "JOBS": str(job_file),
                "PAR": "1",
                "V710_OUT": str(out),
            })
            result = subprocess.run(
                ["bash", str(ROOT / "scripts" / "v710_regate.sh")],
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        assert result.returncode == 123, result.stdout + result.stderr
        assert "ACTIVE dn/small/tsmc5/verify_nn_ac/omp1" in result.stdout
        assert not log.exists()


def _check_invalid_rendering() -> None:
    """An invalid row without parsed metrics must remain visibly invalid."""
    report = collect.render({
        "dn": {
            "small": {
                "verify_nn_multi_tech_dc": {
                    "TSMC5": {"omp1": {"rc": "RACED"}},
                },
            },
        },
    })
    assert "| TSMC5 | INVALID | | | |" in report
    assert "| TSMC5 | — | | | |" not in report


def _check_device_metrics_survive_collection() -> None:
    """Per-tech reports must retain Rule 13's R² and maximum-error evidence."""
    with tempfile.TemporaryDirectory() as raw:
        log = (Path(raw) / "dn" / "small" / "tsmc5"
               / "verify_nn_multi_tech_dc.omp1.log")
        log.parent.mkdir(parents=True)
        log.write_text(
            "  TSMC5_nmos_base  NRMSE= 2.00%  MRE= 3.00%  "
            "R2= 0.95000  MaxErr=4.000e-06  PASS\n"
            "===V710_DONE rc=0===\n"
        )
        entry = collect.collect(Path(raw))["dn"]["small"][
            "verify_nn_multi_tech_dc"
        ]["TSMC5"]["omp1"]
        assert entry["min_r2"] == 0.95
        assert entry["max_error"] == 4e-6
        assert entry["rows"]["TSMC5_nmos_base"]["max_error"] == 4e-6


def _check_clean_pool() -> None:
    """The clean campaign must cover the exact requested Cartesian product."""
    clean = jobs.build_pools()["clean"]
    parsed = {tuple(line.split()) for line in clean}
    expected = {
        (tag, variant, tech, suite, str(omp))
        for tag in ("dn", "tf", "pfn")
        for variant in ("small", "medium", "large", "xl")
        for tech in ("TSMC5", "TSMC6", "TSMC7", "TSMC12", "TSMC16")
        for suite, omps in {
            **{suite: (1,) for suite in jobs.DEVICE_SUITES},
            **{suite: (1,) for suite in jobs.DETERMINISTIC},
            **{suite: (1, 2, 4) for suite in jobs.MULTISTABLE},
        }.items()
        for omp in omps
    }
    assert parsed == expected
    assert len(clean) == len(parsed)


def _check_job_generator_help_is_read_only() -> None:
    """A standard ``--help`` request must not become an output directory."""
    with tempfile.TemporaryDirectory() as raw:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "v710_regate_jobs.py"),
             "--help"],
            cwd=raw,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "usage:" in result.stdout
        assert not (Path(raw) / "--help").exists()


def _check_training_archive_is_selectable_and_completion_gated() -> None:
    """Training must preserve prior checkpoints and never accept partial runs."""
    source = (ROOT / "scripts" / "recipe_train.sh").read_text()
    assert 'CKPT="${BSIMAR_CHECKPOINT_DIR:-' in source
    assert '[ -f "$ckpt" ] && [ -f "$ckpt.complete" ]' in source
    regate_source = (ROOT / "scripts" / "v710_regate.sh").read_text()
    assert 'NG="${NGSPICE_BIN:-' in regate_source
    for script in ("recipe_train.sh", "v710_regate.sh"):
        result = subprocess.run(
            ["bash", str(ROOT / "scripts" / script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Usage:" in result.stdout


def _coverage_data() -> dict[str, dict]:
    """Return one complete TSMC5 DirectNet clean coverage snapshot."""
    return {
        "dn": {
            variant: {
                suite: {
                    "TSMC5": {
                        f"omp{omp}": {"rc": "0"}
                        for omp in omps
                    },
                }
                for suite, omps in coverage.SUITES.items()
            }
            for variant in ("small", "medium", "large", "xl")
        },
    }


def _check_fail_on_gaps() -> None:
    """The optional coverage gate must fail gaps and unavailable groups."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        checkpoints = root / "checkpoints"
        checkpoints.mkdir()
        for variant in ("small", "medium", "large", "xl"):
            for device in ("nmos", "pmos"):
                (checkpoints / f"tsmc5_dn_{variant}_{device}_best.pt.complete").touch()

        data = _coverage_data()
        del data["dn"]["small"]["verify_complex_ring_osc"]["TSMC5"]["omp4"]
        (root / "data.json").write_text(json.dumps(data))
        args = [
            "v730_coverage.py", "--tag", "dn", "--set", "clean",
            "--techs", "TSMC5", "--passes", "test", "--require-complete",
        ]
        old_argv, old_ckpt, old_passes = sys.argv, coverage.CKPT, coverage.PASSES
        coverage.CKPT = checkpoints
        coverage.PASSES = [("test", root)]
        try:
            sys.argv = args
            with redirect_stdout(io.StringIO()):
                assert coverage.main() == 0
            sys.argv = [*args, "--fail-on-gaps"]
            with redirect_stdout(io.StringIO()):
                assert coverage.main() == 1

            (root / "data.json").write_text(json.dumps(_coverage_data()))
            with redirect_stdout(io.StringIO()):
                assert coverage.main() == 0

            coverage.CKPT = root / "no-checkpoints"
            with redirect_stdout(io.StringIO()):
                assert coverage.main() == 1
        finally:
            sys.argv = old_argv
            coverage.CKPT = old_ckpt
            coverage.PASSES = old_passes


def _report_payload(suite: str, rc: str = "1") -> dict[str, object]:
    """Minimal parsed payload consumed by one generated report suite."""
    result: dict[str, object] = {"rc": rc}
    if suite == "verify_nn_ac":
        device = {
            "gain0_err_db": "2.0", "f3db_ratio": "0.5",
            "mag_nrmse_pct": "12.0", "status": "FAIL",
        }
        result.update(nmos=dict(device), pmos=dict(device))
    elif suite.startswith("verify_nn_multi_tech"):
        result.update(
            n=1, n_pass=0, mean_nrmse=12.0, max_nrmse=15.0, mean_mre=20.0,
            min_r2=0.5, max_error=0.001,
        )
    elif suite == "verify_complex_opamp_ac":
        result.update(
            dc_gain_err_db="4.0", gbw_ratio="0.5", pm_err_deg="20.0",
            mag_nrmse_pct="12.0", status="FAIL",
        )
    else:
        result["metric"] = 12.0
    return result


def _check_report_payload_completeness() -> None:
    """Valid FAILs count only when their suite-specific payload was parsed."""
    for suite in docs.REPORT_SUITES:
        payload = _report_payload(suite)
        assert docs._report_result_complete(suite, payload), suite
        assert not docs._report_result_complete(suite, {**payload, "rc": "124"})
        assert not docs._report_result_complete(suite, {"rc": "1"})
        if suite.startswith("verify_nn_multi_tech"):
            for missing in ("max_nrmse", "mean_mre", "min_r2", "max_error"):
                incomplete = dict(payload)
                del incomplete[missing]
                assert not docs._report_result_complete(suite, incomplete)

    data = {
        "dn": {
            variant: {
                suite: {
                    tech: {
                        omp: _report_payload(suite)
                        for omp in required
                    }
                    for tech in docs.TECHS
                }
                for suite, required in docs.REPORT_SUITES.items()
            }
            for variant in docs.TIERS
        },
    }
    old_data = docs.PASS_DATA
    docs.PASS_DATA = {"test": data}
    try:
        assert docs._matrix_complete_in_pass("dn", False, "test")
        cell = data["dn"]["small"]["verify_complex_ring_osc"]["TSMC5"]["omp1"]
        del cell["metric"]
        assert not docs._matrix_complete_in_pass("dn", False, "test")
        cell["metric"] = 12.0
        cell["rc"] = "124"
        assert not docs._matrix_complete_in_pass("dn", False, "test")
    finally:
        docs.PASS_DATA = old_data

    old_data, old_report_pass = docs.PASS_DATA, docs.REPORT_PASS
    docs.PASS_DATA = {"V7.5.16": data}
    docs.REPORT_PASS = {
        **old_report_pass,
        ("dn", False): "V7.5.16",
    }
    try:
        assert "V7.5.16 `large`" in docs.scoreboard()
    finally:
        docs.PASS_DATA = old_data
        docs.REPORT_PASS = old_report_pass


def main() -> int:
    _check_residual_completes_voltage_source_currents()
    _check_nn_ac_banner_tracks_forced_family()
    _check_opamp_ac_refines_bias_and_requires_fixed_point()
    _check_verdict_predicate(collect.is_verdict)
    assert coverage.is_verdict is collect.is_verdict
    assert docs.is_verdict is collect.is_verdict

    assert collect.rc_of(
        "===V710_DONE rc=0===\n===V710_DONE rc=1===\n"
    ) == "RACED"
    assert collect.rc_of(
        "===V710_DONE malformed===\n===V710_DONE rc=0===\n"
    ) == "RACED"
    assert collect._verdict({"omp1": {"rc": "RACED"}}) == "INVALID"
    assert collect._strict({
        "omp1": {"rc": "0"},
        "omp2": {"rc": "RACED"},
        "omp4": {"rc": "0"},
    }) == "INVALID"
    _check_log_scan()
    _check_lock_contention()
    _check_invalid_rendering()
    _check_device_metrics_survive_collection()
    _check_clean_pool()
    _check_job_generator_help_is_read_only()
    _check_training_archive_is_selectable_and_completion_gated()
    _check_fail_on_gaps()
    _check_report_payload_completeness()
    clean_jobs = len(jobs.build_pools()["clean"])
    print(
        f"Accuracy campaign tools: {clean_jobs} unique clean jobs; "
        "invalid outcomes excluded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
