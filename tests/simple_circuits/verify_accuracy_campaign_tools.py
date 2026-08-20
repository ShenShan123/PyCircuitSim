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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import v710_regate_collect as collect
from scripts import v710_regate_jobs as jobs
from scripts import v730_coverage as coverage
from scripts import v730_docs_build as docs


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


def _check_clean_pool() -> None:
    """The clean campaign must cover the exact requested Cartesian product."""
    clean = jobs.build_pools()["clean"]
    parsed = {tuple(line.split()) for line in clean}
    expected = {
        (tag, variant, tech, suite, str(omp))
        for tag in ("dn", "tf")
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
            for missing in ("max_nrmse", "mean_mre"):
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


def main() -> int:
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
    _check_clean_pool()
    _check_fail_on_gaps()
    _check_report_payload_completeness()
    print("Accuracy campaign tools: 480 unique jobs; invalid outcomes excluded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
