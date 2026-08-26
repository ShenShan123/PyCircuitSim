#!/usr/bin/env python3
"""Regression checks for accuracy-campaign completeness and verdict hygiene."""
from __future__ import annotations

from contextlib import redirect_stdout
import fcntl
import hashlib
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
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import benchmark_collect
from scripts import v710_regate_collect as collect
from scripts import v710_regate_jobs as jobs
from scripts import v730_coverage as coverage
from scripts import v730_docs_build as docs
from pycircuitsim.solver import (
    _mna_residual_at_node_voltages,
    _mna_residual_inf,
    _voltage_source_tail_projector,
)
from pycircuitsim.circuit import Circuit
from pycircuitsim.models.passive import VoltageSource
from tests.simple_circuits import verify_complex_opamp_ac as opamp_ac
from tests.simple_circuits import verify_nn_ac


def _write_executable(path: Path, source: str) -> None:
    """Write an executable local test stub without invoking project tooling."""
    path.write_text(source)
    path.chmod(0o755)


def _recipe_worker(
    root: Path, checkpoint_dir: Path, mode: str,
) -> subprocess.CompletedProcess[str]:
    """Run one recipe worker through a deterministic fake conda command."""
    fake_bin = root / "bin"
    fake_bin.mkdir(exist_ok=True)
    _write_executable(
        fake_bin / "conda",
        """#!/usr/bin/env bash
set -u
printf 'invoked\\n' >> "$FAKE_CONDA_LOG"
case "$FAKE_TRAIN_MODE" in
  success|marker-fail)
    stem="$BSIMAR_CHECKPOINT_DIR/tsmc5_dn_small_nmos"
    : > "${stem}_best.pt"
    : > "${stem}_norm.npz"
    ;;
  fail) exit 1 ;;
  *) exit 99 ;;
esac
    """,
    )
    if mode == "marker-fail":
        _write_executable(
            fake_bin / "touch",
            "#!/usr/bin/env bash\necho marker-write-failed >&2\nexit 1\n",
        )
    env = os.environ.copy()
    env.update({
        "PATH": f"{fake_bin}:{env['PATH']}",
        "BSIMAR_CHECKPOINT_DIR": str(checkpoint_dir),
        "RECIPE_TRAIN_LOG_DIR": str(root / "logs"),
        "FAKE_CONDA_LOG": str(root / "conda.log"),
        "FAKE_TRAIN_MODE": mode,
        "MODEL": "direct",
    })
    return subprocess.run(
        ["bash", str(ROOT / "scripts" / "recipe_train.sh"), "_one",
         "clean", "tsmc5", "small", "nmos", "0", "noforce"],
        env=env, capture_output=True, text=True, check=False, timeout=10,
    )


def _ready_checkpoint(
    checkpoint_dir: Path, tag: str, *, config: bool = True,
    complete: bool = True,
) -> None:
    """Create the two-polarity artifacts required by a ready checkpoint."""
    for device in ("nmos", "pmos"):
        stem = checkpoint_dir / f"tsmc5_{tag}_small_{device}"
        stem.with_name(stem.name + "_best.pt").touch()
        stem.with_name(stem.name + "_norm.npz").touch()
        if complete:
            stem.with_name(stem.name + "_best.pt.complete").touch()
        if tag == "tf" and config:
            stem.with_name(stem.name + "_config.npz").touch()


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


def _check_rank_deficient_tail_projector() -> None:
    """Cached source-current fitting must match direct least squares exactly."""
    circuit = Circuit()
    circuit.add_component(VoltageSource("v1", ["n", "0"], 0.4))
    circuit.add_component(VoltageSource("v2", ["n", "0"], 0.4))
    node_map = circuit.get_node_map()
    dense = np.array(
        [[2.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=float,
    )
    rhs = np.array([0.3, 0.4, 0.4])
    voltages = np.array([0.25])
    projector = _voltage_source_tail_projector(circuit, node_map, 1, 2)
    assert projector is _voltage_source_tail_projector(circuit, node_map, 1, 2)

    for matrix in (dense, csr_matrix(dense)):
        cached, cached_scale = _mna_residual_at_node_voltages(
            matrix, rhs, voltages, 1, tail_projector=projector,
        )
        direct, direct_scale = _mna_residual_at_node_voltages(
            matrix, rhs, voltages, 1,
        )
        balance = rhs[:1] - dense[:1, :1].dot(voltages)
        tail = np.linalg.lstsq(dense[:1, 1:], balance, rcond=None)[0]
        expected = _mna_residual_inf(matrix, rhs, np.r_[voltages, tail])
        assert abs(cached - direct) < 1e-14
        assert abs(cached - expected) < 1e-14
        assert cached_scale == direct_scale == abs(rhs[0])


def _check_nn_ac_banner_tracks_forced_family() -> None:
    """Gate output must identify the NN family actually selected by the parser."""
    with patch.dict(os.environ, {}, clear=True):
        assert verify_nn_ac.active_model_label() == "DirectNet (LEVEL=73)"
        assert verify_nn_ac.active_model_name() == "DirectNet"
    with patch.dict(os.environ, {"PYCIRCUITSIM_NN_FORCE_LEVEL": "74"}, clear=True):
        assert verify_nn_ac.active_model_label() == "BSIM-AR (LEVEL=74)"
        assert verify_nn_ac.active_model_name() == "BSIM-AR"
    with patch.dict(os.environ, {"PYCIRCUITSIM_NN_FORCE_LEVEL": "75"}, clear=True):
        try:
            verify_nn_ac.active_model_label()
        except ValueError as exc:
            assert "unsupported NN model level 75" in str(exc)
        else:
            raise AssertionError("retired LEVEL=75 was accepted")

    parsed = benchmark_collect.parse_complex_log(
        "verify_complex_sram_snm",
        "NG SNM=120.0mV  BSIM-AR SNM=118.0mV\nall-positive: yes\n",
    )
    assert parsed["headline"] == "NG_SNM=120.0mV DN_SNM=118.0mV"
    assert parsed["gate"] == "PASS"

    result: dict[str, object] = {
        "tech": "TSMC5", "passed": True, "op_ok": True,
        "dc_op": {"vout": 0.5, "vo1i": 0.4, "vtail": 0.2},
        "m": {
            "gain0_db": 1.0, "gain0_db_ref": 1.0, "gain0_db_err": 0.0,
            "gbw_test": 1.0, "gbw_ref": 1.0, "gbw_ratio": 1.0,
            "pm_test": 60.0, "pm_ref": 60.0, "pm_err": 0.0,
            "mag_nrmse": 0.0,
        },
    }
    printed = io.StringIO()
    with patch.dict(os.environ, {"PYCIRCUITSIM_NN_FORCE_LEVEL": "74"},
                    clear=True), redirect_stdout(printed):
        opamp_ac._print_result(result)
    assert "BSIM-AR DC OP" in printed.getvalue()
    assert "NN DC OP" not in printed.getvalue()


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
                "V710_CAMPAIGN_DIGEST": "0" * 64,
                "V710_TEST_BYPASS_MANIFEST": "1",
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


def _check_error_rows_count_in_device_denominators() -> None:
    """A parsed ERROR row remains a failed configuration, not a dropped row."""
    with tempfile.TemporaryDirectory() as raw:
        log = (Path(raw) / "dn" / "small" / "tsmc5"
               / "verify_nn_multi_tech_dc.omp1.log")
        log.parent.mkdir(parents=True)
        log.write_text(
            "  TSMC5_nmos_base  NRMSE= 2.00%  MRE= 3.00%  "
            "R2= 0.95000  MaxErr=4.000e-06  PASS\n"
            "  TSMC5_pmos_base ERROR: reference solver failed\n"
            "===V710_DONE rc=1===\n"
        )
        entry = collect.collect(Path(raw))["dn"]["small"][
            "verify_nn_multi_tech_dc"
        ]["TSMC5"]["omp1"]
        assert entry["n"] == 2
        assert entry["n_pass"] == 1
        assert entry["rows"]["TSMC5_pmos_base"]["status"] == "ERROR"


def _check_explicit_opamp_errors_are_complete_failures() -> None:
    """Caught convergence errors stay in report denominators without metrics."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        logs = {
            "verify_complex_opamp": (
                "TSMC5    | ERROR — RuntimeError('DC operating point did not converge')\n"
            ),
            "verify_complex_opamp_ac": (
                "  TSMC5    | ERROR — DC operating point did not converge\n"
            ),
        }
        for suite, summary in logs.items():
            log = root / "dn" / "small" / "tsmc5" / f"{suite}.omp1.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(summary + "===V710_DONE rc=1===\n")

        data = collect.collect(root)
        for suite in logs:
            entry = data["dn"]["small"][suite]["TSMC5"]["omp1"]
            assert entry["status"] == "ERROR"
            assert "DC operating point did not converge" in entry["error"]
            assert docs._report_result_complete(suite, entry)
            assert "metric" not in entry
            assert not docs._report_result_complete(
                suite, {"rc": "1", "status": "ERROR"},
            )
            assert not docs._report_result_complete(
                suite, {"rc": "0", "status": "ERROR", "error": "bad"},
            )


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


def _check_training_lifecycle_subprocesses() -> None:
    """Workers skip only complete artifacts and fail closed around markers."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        checkpoint_dir = root / "checkpoints"
        checkpoint_dir.mkdir()
        stem = checkpoint_dir / "tsmc5_dn_small_nmos"

        for suffix in ("_best.pt", "_norm.npz", "_best.pt.complete"):
            stem.with_name(stem.name + suffix).touch()
        skipped = _recipe_worker(root, checkpoint_dir, "success")
        assert skipped.returncode == 0, skipped.stdout + skipped.stderr
        assert "SKIP existing" in skipped.stdout
        assert not (root / "conda.log").exists()

        stem.with_name(stem.name + "_best.pt.complete").unlink()
        retrained = _recipe_worker(root, checkpoint_dir, "success")
        assert retrained.returncode == 0, retrained.stdout + retrained.stderr
        assert "RETRAIN incomplete" in retrained.stdout
        assert stem.with_name(stem.name + "_best.pt.complete").exists()

        stem.with_name(stem.name + "_norm.npz").unlink()
        failed = _recipe_worker(root, checkpoint_dir, "fail")
        assert failed.returncode == 3, failed.stdout + failed.stderr
        assert not stem.with_name(stem.name + "_best.pt.complete").exists()

        stem.with_name(stem.name + "_norm.npz").touch()
        marker_failure = _recipe_worker(root, checkpoint_dir, "marker-fail")
        assert marker_failure.returncode == 3
        assert "LIFECYCLE ERROR" in marker_failure.stderr
        assert not stem.with_name(stem.name + "_best.pt.complete").exists()


def _check_training_cli_contract() -> None:
    """Only the documented public force argument is accepted."""
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "recipe_train.sh"), "--force", "--typo"],
        capture_output=True, text=True, check=False, timeout=10,
    )
    assert result.returncode == 2
    assert "Usage:" in result.stderr


def _check_regate_readiness_and_family_ownership() -> None:
    """Re-gate rejects partial artifacts and owns its selected NN family."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        checkpoint_dir = root / "checkpoints"
        checkpoint_dir.mkdir()
        runner = root / "runner"
        observed = root / "force-level.txt"
        _write_executable(
            runner,
            "#!/usr/bin/env bash\nprintf '%s\\n' \"$PYCIRCUITSIM_NN_FORCE_LEVEL\" > \"$FAKE_FORCE_LOG\"\necho stub-suite-ran\n",
        )
        env = os.environ.copy()
        env.update({
            "NGSPICE_BIN": "/bin/true",
            "NN_PY": str(runner),
            "BSIMAR_CHECKPOINT_DIR": str(checkpoint_dir),
            "V710_OUT": str(root / "out"),
            "V710_SCRATCH": str(root / "scratch"),
            "FAKE_FORCE_LOG": str(observed),
            "PYCIRCUITSIM_NN_FORCE_LEVEL": "75",
            "V710_CAMPAIGN_DIGEST": "0" * 64,
            "V710_TEST_BYPASS_MANIFEST": "1",
        })
        command = ["bash", str(ROOT / "scripts" / "v710_regate.sh"), "_one",
                   "dn", "small", "TSMC5", "verify_nn_ac", "1"]

        _ready_checkpoint(checkpoint_dir, "dn", complete=False)
        blocked = subprocess.run(command, env=env, capture_output=True, text=True,
                                check=False, timeout=10)
        assert blocked.returncode == 3
        assert "NO-CKPT" in blocked.stdout
        assert not observed.exists()

        (checkpoint_dir / "tsmc5_dn_small_nmos_norm.npz").unlink()
        partial = subprocess.run(command, env=env, capture_output=True, text=True,
                                 check=False, timeout=10)
        assert partial.returncode == 3
        assert "still no verdict" in partial.stdout
        assert not observed.exists()

        _ready_checkpoint(checkpoint_dir, "dn")
        ran = subprocess.run(command, env=env, capture_output=True, text=True,
                             check=False, timeout=10)
        assert ran.returncode == 0, ran.stdout + ran.stderr
        assert observed.read_text().strip() == "73"

        _ready_checkpoint(checkpoint_dir, "tf", config=False)
        with patch.object(coverage, "CKPT", checkpoint_dir):
            assert not coverage.ckpt_exists("tf", "small", "TSMC5", True)
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


def _check_regate_interpreter_failures_are_infrastructure() -> None:
    """A broken Python environment must not become a scientific FAIL row."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        jobs_file = root / "jobs.txt"
        jobs_file.write_text("")
        env = os.environ.copy()
        env.update({
            "JOBS": str(jobs_file),
            "NGSPICE_BIN": "/bin/true",
            "NN_PY": str(root / "missing-python"),
        })
        env.pop("V710_PY_PREFLIGHTED", None)
        missing = subprocess.run(
            ["bash", str(ROOT / "scripts" / "v710_regate.sh")],
            env=env, capture_output=True, text=True, check=False, timeout=10,
        )
        assert missing.returncode == 2
        assert "NN_PY executable not found" in missing.stderr

        checkpoint_dir = root / "checkpoints"
        checkpoint_dir.mkdir()
        _ready_checkpoint(checkpoint_dir, "dn")
        runner = root / "traceback-python"
        _write_executable(
            runner,
            """#!/usr/bin/env bash
if [ "${1:-}" = "-c" ]; then exit 0; fi
printf '%s\n' 'Traceback (most recent call last):' >&2
printf '%s\n' 'ModuleNotFoundError: No module named numpy' >&2
exit 1
""",
        )
        env.update({
            "NN_PY": str(runner),
            "BSIMAR_CHECKPOINT_DIR": str(checkpoint_dir),
            "V710_OUT": str(root / "out"),
            "V710_SCRATCH": str(root / "scratch"),
            "V710_CAMPAIGN_DIGEST": "0" * 64,
            "V710_TEST_BYPASS_MANIFEST": "1",
        })
        gate = subprocess.run(
            ["bash", str(ROOT / "scripts" / "v710_regate.sh"), "_one",
             "dn", "small", "TSMC5", "verify_nn_ac", "1"],
            env=env, capture_output=True, text=True, check=False, timeout=10,
        )
        assert gate.returncode == 3
        log = (root / "out" / "dn" / "small" / "tsmc5"
               / "verify_nn_ac.omp1.log")
        assert "===V710_DONE rc=infra===" in log.read_text()
        assert "===V710_DONE rc=1===" not in log.read_text()


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
                stem = checkpoints / f"tsmc5_dn_{variant}_{device}"
                for suffix in ("_best.pt", "_norm.npz", "_best.pt.complete"):
                    stem.with_name(stem.name + suffix).touch()

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


def _check_readme_uses_only_current_families() -> None:
    """The generated scoreboard contains only supported model families."""
    data = {
        tag: {
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
        }
        for tag in docs.CURRENT_CLEAN_TAGS
    }
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        templates = root / "templates"
        rendered = root / "rendered"
        templates.mkdir()
        rendered.mkdir()
        (templates / "README.md.in").write_text("<!--SCOREBOARD-->\n")

        old_root, old_tpl, old_docs = docs.ROOT, docs.TPL, docs.DOCS
        old_data, old_pass = docs.PASS_DATA, docs.REPORT_PASS
        old_provenance = docs._v7517_provenance_complete
        docs.ROOT = root
        docs.TPL = templates
        docs.DOCS = rendered
        docs.PASS_DATA = {"V7.5.17": data}
        docs.REPORT_PASS = {
            **old_pass,
            ("dn", False): "V7.5.17",
            ("tf", False): "V7.5.17",
        }
        docs._v7517_provenance_complete = lambda: True
        try:
            with redirect_stdout(io.StringIO()):
                assert docs.build_readme(check=False)
            result = (rendered / "README.md").read_text()
            assert "V7.5.17 `small` **0/20**" in result
            assert "| 75 |" not in result
        finally:
            docs.ROOT = old_root
            docs.TPL = old_tpl
            docs.DOCS = old_docs
            docs.PASS_DATA = old_data
            docs.REPORT_PASS = old_pass
            docs._v7517_provenance_complete = old_provenance


def _check_incomplete_reports_preserve_verified_output() -> None:
    """Missing raw evidence preserves only a checksum-verified report."""
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        templates = root / "templates"
        rendered = root / "rendered"
        templates.mkdir()
        rendered.mkdir()
        (templates / "DirectNet-L73-clean.md.in").write_text("<!--SCOREBOARD-->\n")
        (templates / "README.md.in").write_text("<!--SCOREBOARD-->\n")
        destination = rendered / "DirectNet-L73-clean.md"
        original = b"preserved rendered evidence\n"
        destination.write_bytes(original)
        readme_destination = rendered / "README.md"
        readme_original = b"preserved scoreboard\n"
        readme_destination.write_bytes(readme_original)

        old_tpl, old_docs = docs.TPL, docs.DOCS
        old_data, old_pass, old_hashes, old_readme_hash = (
            docs.PASS_DATA, docs.REPORT_PASS, docs.PRESERVED_REPORT_SHA256,
            docs.PRESERVED_README_SHA256,
        )
        docs.TPL = templates
        docs.DOCS = rendered
        docs.PASS_DATA = {"V7.5.16": {}}
        docs.REPORT_PASS = {**old_pass, ("dn", False): "V7.5.16"}
        docs.PRESERVED_REPORT_SHA256 = {
            **old_hashes,
            ("dn", False): hashlib.sha256(original).hexdigest(),
        }
        docs.PRESERVED_README_SHA256 = hashlib.sha256(readme_original).hexdigest()
        try:
            with redirect_stdout(io.StringIO()):
                assert docs.build("dn", False, check=False)
            assert destination.read_bytes() == original
            destination.write_bytes(b"drifted rendered evidence\n")
            with redirect_stdout(io.StringIO()):
                assert not docs.build("dn", False, check=False)

            with redirect_stdout(io.StringIO()):
                assert docs.build_readme(check=False)
            assert readme_destination.read_bytes() == readme_original
            readme_destination.write_bytes(b"drifted scoreboard\n")
            with redirect_stdout(io.StringIO()):
                assert not docs.build_readme(check=False)
            scoreboard = docs.scoreboard()
            assert "2026-08-19 `large`" in scoreboard
            assert "V7.5.16 `large`" not in scoreboard
        finally:
            docs.TPL = old_tpl
            docs.DOCS = old_docs
            docs.PASS_DATA = old_data
            docs.REPORT_PASS = old_pass
            docs.PRESERVED_REPORT_SHA256 = old_hashes
            docs.PRESERVED_README_SHA256 = old_readme_hash


def main() -> int:
    _check_residual_completes_voltage_source_currents()
    _check_rank_deficient_tail_projector()
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
    _check_error_rows_count_in_device_denominators()
    _check_explicit_opamp_errors_are_complete_failures()
    _check_clean_pool()
    _check_job_generator_help_is_read_only()
    _check_training_lifecycle_subprocesses()
    _check_training_cli_contract()
    _check_regate_readiness_and_family_ownership()
    _check_regate_interpreter_failures_are_infrastructure()
    _check_fail_on_gaps()
    _check_report_payload_completeness()
    _check_readme_uses_only_current_families()
    _check_incomplete_reports_preserve_verified_output()
    clean_pool = jobs.build_pools()["clean"]
    clean_jobs = len(clean_pool)
    current_jobs = sum(
        line.split(maxsplit=1)[0] in docs.CURRENT_CLEAN_TAGS
        for line in clean_pool
    )
    assert current_jobs == 480
    print(
        f"Accuracy campaign tools: {clean_jobs} unique clean jobs; "
        f"{current_jobs} current DirectNet/BSIM-AR jobs; invalid outcomes excluded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
