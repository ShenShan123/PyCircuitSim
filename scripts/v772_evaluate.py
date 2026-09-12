#!/usr/bin/env python3
"""Resume the complete V7.7.2-weight evaluation in increasing circuit difficulty."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v710_regate_jobs import build_pools, TECHS, CLEAN_VARIANTS  # noqa: E402
from scripts.v710_regate_manifest import _verify_group  # noqa: E402
from tests.common.simple_circuit_catalog import cases, SIMPLE_V2  # noqa: E402
from tests.common.simple_circuit_harness import CORNERS, applicable_analyses  # noqa: E402
from tests.common.circuit_benchmarks import BENCH  # noqa: E402
from tests.common.circuit_sweep import CIRCUITS, all_dimensions  # noqa: E402

CAMPAIGN = "v772_eval_20260912"
STATE = ROOT / "results" / CAMPAIGN
TRAINING = Path("/data2/home/shenshan/PyCircuitSim-v771")
CHECKPOINTS = TRAINING / "results/v771_r2_checkpoints"
DATA = TRAINING / "results/v771_r2_data"
TRAINING_SOURCE = "6be83348c1f5db6720d7504ed6dcea874a3a7418"
SAVED = Path("/data2/home/shenshan/PyCircuitSim-v772/results/v772_campaign")
NGSPICE = Path("/usr/local/ngspice-45.2/bin/ngspice")
OSDI = ROOT / "external_compact_models/bsim_cmg/build/osdi/bsimcmg.osdi"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def environment() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PYCIRCUITSIM_", "BSIMAR_", "V710_"))}
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYCIRCUITSIM_TORCH_THREADS="1",
               OMP_WAIT_POLICY="passive", KMP_BLOCKTIME="0", PYTHONHASHSEED="0",
               BSIMAR_DATA_DIR=str(DATA), BSIMAR_CHECKPOINT_DIR=str(CHECKPOINTS),
               NGSPICE_BIN=str(NGSPICE), NN_PY=sys.executable)
    return env


def inventory() -> list[dict[str, Any]]:
    """Preserve complete pool membership; stages only control execution order."""
    jobs: list[dict[str, Any]] = []
    catalog = {case.campaign_suite: case for case in cases()}
    device_suites = {"verify_device_integrity", "verify_terminal_integrity",
                     "verify_nn_multi_tech_dc", "verify_nn_lifted_source_dc"}
    for pool, lines in build_pools().items():
        for line in lines:
            tag, size, tech, suite, omp = line.split()
            stage = (int(catalog[suite].tier[1]) + 1 if suite in catalog else
                     1 if suite in device_suites else
                     4 if suite == "verify_circuit_opamp_ac" else 3)
            jobs.append(dict(id=f"{pool}__{tag}__{size}__{tech}__{suite}__{omp}",
                             stage=stage, pool=pool, line=line, tag=tag, size=size,
                             tech=tech, suite=suite, kind="pool", role=pool))

    def add(name: str, stage: int, command: list[str], *,
            group: tuple[str, str, str] | None = None,
            cwd: str = ".", role: str = "supplemental") -> None:
        job: dict[str, Any] = dict(id=name, stage=stage, command=command,
                                  cwd=cwd, kind="standalone", role=role)
        if group:
            job.update(zip(("tag", "size", "tech"), group))
        jobs.append(job)

    add("root_contracts", 0, ["-m", "pytest", "-q", "tests", "-ra"], role="contract")
    add("pycmg_reference", 0, ["-m", "pytest", "-q", "tests", "-ra"],
        cwd="external_compact_models/bsim_cmg", role="reference")
    add("geometry", 0, ["tests/single_devices/verify_data_geometry_coverage.py",
                        "--data-dir", str(DATA)], role="preflight")
    fixed = {
        1: ["single_devices/verify_bsimcmg_op.py", "single_devices/verify_cmg_multiplier.py",
            "single_devices/verify_bsimcmg_dc_comprehensive.py"],
        3: ["simple_circuits/verify_ac.py", "simple_circuits/verify_bsimcmg_inverter_op.py",
            "simple_circuits/verify_bsimcmg_tran_comprehensive.py",
            "simple_circuits/verify_multi_tech.py", "simple_circuits/verify_subckt.py"],
    }
    for stage, scripts in fixed.items():
        for script in scripts:
            add(Path(script).stem, stage, [f"tests/{script}"], role="reference")
    add("ar_cache", 1, ["tests/perf/verify_ar_cache.py"], role="optimization-diagnostic")
    for tag in ("dnf", "tff"):
        for size in CLEAN_VARIANTS:
            for tech in TECHS:
                group = tag, size, tech
                prefix = "__".join(group)
                for name, stage, script, options in (
                    ("device", 1, "single_devices/verify_nn_dc.py", []),
                    ("sign", 1, "single_devices/verify_nn_dc.py", ["--sign-diagnostic"]),
                    ("idvds", 1, "single_devices/verify_nn_dc.py", ["--idvds-diagnostic"]),
                    ("inverter", 3, "simple_circuits/verify_nn_inverter.py", []),
                ):
                    add(f"{prefix}__{name}", stage,
                        [f"tests/{script}", "--tech", tech, *options], group=group)
                for case in cases(score_version=SIMPLE_V2):
                    # Every non-nominal applicable corner is an independent
                    # resumable job. The nominal cell is already in its pool.
                    for corner, config in CORNERS.items():
                        if corner == "nominal" or not applicable_analyses(case, BENCH[tech], config):
                            continue
                        add(f"{prefix}__{case.case_id}__{corner}", 1 + int(case.tier[1]),
                            ["tests/simple_circuits/verify_circuit_topologies.py",
                             "--tech", tech, "--case", case.case_id, "--corner", corner],
                            group=group, role="corner-diagnostic")
                for circuit in CIRCUITS:
                    for dimension in all_dimensions(circuit):
                        add(f"{prefix}__sweep__{circuit}__{dimension}", 3 if circuit == "sram" else 4,
                            ["tests/simple_circuits/verify_circuit_sweep.py", circuit,
                             "--tech", tech, "--dimension", dimension, "--pin-strict"],
                            group=group, role="parametric-diagnostic")
    # Three independent reference repeats plus L72 attribution, outside scores.
    for case in cases(score_version=SIMPLE_V2):
        for tech in TECHS:
            add(f"reference_repeats__{case.case_id}__{tech}", 1 + int(case.tier[1]),
                ["tests/simple_circuits/verify_circuit_topologies.py", "--tech", tech,
                 "--case", case.case_id, "--reference-repeats", "3", "--level72-control"],
                group=("dnf", "small", tech), role="reference-stability-diagnostic")
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("duplicate evaluation jobs")
    return sorted(jobs, key=lambda job: (job["stage"], job["kind"] != "pool",
                                        job.get("size", ""),
                                        job.get("tech", ""), job.get("suite", ""),
                                        job.get("tag", ""), job["id"]))


def assert_source(source: str) -> None:
    if git("rev-parse", "HEAD") != source or git("status", "--porcelain"):
        raise RuntimeError("evaluation source drift; preserve this arm and use a fresh worktree")


def verify_saved_bundles() -> None:
    """Rehash saved artifacts and each distinct dataset, without rewriting markers."""
    done = json.loads((SAVED / "training_done.json").read_text())
    path = SAVED / "saved_checkpoint_inventory.json"
    if sha256(path) != done["inventory_sha256"]:
        raise ValueError("saved training inventory drift")
    saved = json.loads(path.read_text())
    expected = {f"{tech.lower()}_{tag}_{size}_{device}"
                for tag in ("dnf", "tff") for size in CLEAN_VARIANTS
                for tech in TECHS for device in ("nmos", "pmos")}
    if saved["source_commit"] != TRAINING_SOURCE or {r["model"] for r in saved["bundles"]} != expected:
        raise ValueError("saved bundle matrix/source differs")
    checked: dict[str, str] = {}
    for row in saved["bundles"]:
        marker = CHECKPOINTS / Path(row["completion_marker"]).name
        if sha256(marker) != row["marker_sha256"]:
            raise ValueError(f"completion marker drift: {marker}")
        bundle = json.loads(marker.read_text())
        if bundle != row["bundle"] or bundle["dataset_source_commit"] != TRAINING_SOURCE:
            raise ValueError(f"saved bundle identity drift: {marker}")
        for field in ("checkpoint", "normalization", "configuration"):
            if field in bundle and sha256(CHECKPOINTS / bundle[field]) != bundle[field + "_sha256"]:
                raise ValueError(f"bundle artifact drift: {row['model']}/{field}")
        for field in ("dataset", "dataset_completion_marker"):
            filename = bundle[field]
            if filename not in checked:
                print(f"verify data {filename}", flush=True)
                checked[filename] = sha256(DATA / filename)
            if checked[filename] != bundle[field + "_sha256"]:
                raise ValueError(f"dataset artifact drift: {filename}")
    write_json(STATE / "bundle_verification.json", dict(
        verified_at=now(), models=80, inventory_sha256=done["inventory_sha256"],
        training_source=TRAINING_SOURCE, datasets=checked))


def prepare(source: str, jobs: list[dict[str, Any]]) -> None:
    assert_source(source)
    schedule = dict(source_commit=source, training_source=TRAINING_SOURCE,
                    checkpoint_dir=str(CHECKPOINTS), python=sys.executable, jobs=jobs)
    path = STATE / "schedule.json"
    if path.exists() and json.loads(path.read_text()) != schedule:
        raise ValueError("immutable evaluation schedule drift")
    write_json(path, schedule)
    verify_saved_bundles()
    for pool, lines in build_pools().items():
        output = ROOT / "results" / f"{CAMPAIGN}_full_{pool}"
        output.mkdir(parents=True, exist_ok=True)
        job_list = output / "jobs.txt"
        job_list.write_text("\n".join(lines) + "\n")
        subprocess.run([sys.executable, "scripts/v710_regate_manifest.py",
                        "--output", str(output / "campaign_manifest.json"),
                        "--jobs", str(job_list), "--checkpoints", str(CHECKPOINTS),
                        "--ngspice", str(NGSPICE), "--osdi", str(OSDI),
                        "--pdk-root", str(ROOT / "PDKs"),
                        "--dataset-source-commit", TRAINING_SOURCE,
                        "--evaluation-runtime-commit", source],
                       cwd=ROOT, env=environment(), check=True)
    write_json(STATE / "scope.json", dict(
        jobs=len(jobs), stages=dict(Counter(str(j["stage"]) for j in jobs)),
        roles=dict(Counter(j["role"] for j in jobs)),
        pools={key: len(value) for key, value in build_pools().items()},
        verify_scripts=sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests").rglob("verify_*.py"))))


def classify(job: dict[str, Any], returncode: int, log: str) -> str:
    if "Traceback (most recent call last):" in log or returncode < 0 or returncode >= 124:
        return "infrastructure_error"
    if job["kind"] == "pool":
        return "complete" if returncode == 0 else "infrastructure_error"
    if returncode in (0, 1):
        return "complete"
    # This legacy driver explicitly uses 2 for characterized ERROR rows.
    if (returncode == 2 and job["role"] == "parametric-diagnostic"
            and "RESULT exit-code = 2" in log):
        return "complete"
    return "infrastructure_error"


def run_one(job: dict[str, Any], source: str) -> dict[str, Any]:
    record = STATE / "jobs" / (job["id"] + ".json")
    if record.exists():
        previous = json.loads(record.read_text())
        if previous.get("status") == "complete":
            if previous["source_commit"] != source or sha256(Path(previous["log"])) != previous["log_sha256"]:
                raise ValueError(f"completed job evidence drift: {job['id']}")
            if previous.get("worker_log") and sha256(Path(previous["worker_log"])) != previous["worker_log_sha256"]:
                raise ValueError(f"completed worker evidence drift: {job['id']}")
            return previous
        archive = STATE / "attempts" / job["id"] / str(time.time_ns())
        archive.mkdir(parents=True)
        shutil.copy2(record, archive / "job.json")
        # The legacy worker replaces an invalid verdict log on retry. Retain
        # it first, together with the old simulator outputs it describes.
        if previous.get("worker_log") and Path(previous["worker_log"]).exists():
            shutil.copy2(previous["worker_log"], archive / "worker.log")
        if job["kind"] == "pool":
            tag, size, tech, suite, omp = job["line"].split()
            previous_artifacts = STATE / "pool_artifacts" / f"{tag}_{size}_{tech.lower()}_{suite}_omp{omp}"
        else:
            previous_artifacts = STATE / "artifacts" / job["id"]
        if previous_artifacts.exists():
            previous_artifacts.rename(archive / "artifacts")
    if shutil.disk_usage(ROOT).free < 60 * 1024**3:
        raise RuntimeError("less than 60 GiB free; pause before launching further simulations")
    env = environment()
    output = ROOT / "results" / f"{CAMPAIGN}_full_{job.get('pool', 'clean')}"
    manifest = output / "campaign_manifest.json"
    digest = sha256(manifest)
    artifacts = STATE / "artifacts" / job["id"]
    env.update(PYCIRCUITSIM_SIMPLE_RESULTS=str(artifacts / "simple"),
               PYCIRCUITSIM_NN_RESULTS=str(artifacts / "nn"),
               PYCIRCUITSIM_CAMPAIGN_MANIFEST_SHA256=digest)
    if "tag" in job:
        _verify_group(manifest, CHECKPOINTS, (job["tag"], job["size"], job["tech"].lower()))
        env["PYCIRCUITSIM_NN_FORCE_LEVEL"] = {"dnf": "75", "tff": "76"}[job["tag"]]
        for device in ("nmos", "pmos"):
            env[f"PYCIRCUITSIM_NN_CHECKPOINT_{job['tag'].upper()}_{device.upper()}"] = (
                f"{job['tech'].lower()}_{job['tag']}_{job['size']}_{device}")
    if job["kind"] == "pool":
        env.update(V710_OUT=str(output), V710_MANIFEST=str(manifest),
                   V710_CAMPAIGN_DIGEST=digest, V710_SCRATCH=str(STATE / "pool_artifacts"))
        command = ["bash", "scripts/v710_regate.sh", "_one", *job["line"].split()]
    else:
        command = [sys.executable, "-u", *job["command"]]
    logs = STATE / "logs"
    logs.mkdir(exist_ok=True)
    log = logs / f"{job['id']}.{time.time_ns()}.log"
    result = dict(job, command=command, source_commit=source, status="running",
                  started_at=now(), log=str(log), manifest_sha256=digest,
                  environment={k: v for k, v in env.items() if k.startswith(
                      ("PYCIRCUITSIM_", "BSIMAR_", "V710_", "OMP_", "MKL_", "CUDA_"))})
    write_json(record, result)
    started = time.monotonic()
    with log.open("w") as handle:
        handle.write(json.dumps({"source_commit": source, "command": command}) + "\n")
        handle.flush()
        process = subprocess.Popen(command, cwd=ROOT / job.get("cwd", "."),
                                   env=env, stdout=handle, stderr=subprocess.STDOUT)
        result["pid"] = process.pid
        write_json(record, result)
        rc = process.wait()
    result.update(returncode=rc, finished_at=now(), elapsed_seconds=time.monotonic() - started,
                  status=classify(job, rc, log.read_text(errors="replace")), log_sha256=sha256(log))
    if job["kind"] == "pool":
        tag, size, tech, suite, omp = job["line"].split()
        worker_log = output / tag / size / tech.lower() / f"{suite}.omp{omp}.log"
        if worker_log.exists():
            result.update(worker_log=str(worker_log), worker_log_sha256=sha256(worker_log))
    if "tag" in job:
        _verify_group(manifest, CHECKPOINTS, (job["tag"], job["size"], job["tech"].lower()))
    write_json(record, result)
    return result


def run_stage(stage_jobs: list[dict[str, Any]], source: str, parallel: int) -> bool:
    """Drain admitted work, retaining infra failures before advancing stages."""
    failures: list[str] = []
    # Fixed-output reference entry points run serially to avoid file collisions.
    serial = [j for j in stage_jobs if j["role"] in ("reference", "contract", "preflight")]
    concurrent = [j for j in stage_jobs if j not in serial]
    for job in serial:
        result = run_one(job, source)
        if result["status"] != "complete" or (job["stage"] == 0 and result["returncode"]):
            failures.append(job["id"])
    if failures:
        write_json(STATE / "infrastructure_failures.json", failures)
        return False
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        pending: dict[Any, dict[str, Any]] = {}
        iterator = iter(concurrent)
        while True:
            while not failures and len(pending) < parallel:
                job = next(iterator, None)
                if job is None:
                    break
                pending[pool.submit(run_one, job, source)] = job
            if not pending:
                break
            future = next(as_completed(pending))
            job = pending.pop(future)
            try:
                result = future.result()
                if result["status"] != "complete":
                    failures.append(job["id"])
            except Exception as exc:
                failures.append(f"{job['id']}: {exc}")
            print(f"stage {job['stage']} completed {job['id']}", flush=True)
    if failures:
        write_json(STATE / "infrastructure_failures.json", failures)
    return not failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.parallel < 1:
        parser.error("parallel must be positive")
    jobs = inventory()
    if args.plan:
        print(json.dumps(dict(jobs=len(jobs), stages=dict(Counter(j['stage'] for j in jobs)),
                              roles=dict(Counter(j['role'] for j in jobs))), indent=2))
        return 0
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = git("rev-parse", "HEAD")
        write_json(STATE / "state.json", dict(status="preflight", updated_at=now(), pid=os.getpid()))
        try:
            prepare(source, jobs)
            if args.prepare_only:
                return 0
            for stage in sorted({job["stage"] for job in jobs}):
                assert_source(source)
                state = dict(status="running", stage=stage, updated_at=now(),
                             source_commit=source, pid=os.getpid(), total=len(jobs))
                write_json(STATE / "state.json", state)
                write_json(STATE / "stage_event.json", state)
                if not run_stage([j for j in jobs if j["stage"] == stage], source, args.parallel):
                    raise RuntimeError(f"stage {stage} has infrastructure failures; see retained logs")
                if stage == 5:
                    # A final rehash must still reproduce the original manifests
                    # before any complete campaign is collected or published.
                    prepare(source, jobs)
                    for pool in build_pools():
                        subprocess.run([sys.executable, "scripts/v710_regate_collect.py", "--root",
                                        f"results/{CAMPAIGN}_full_{pool}", "--require-manifest"],
                                       cwd=ROOT, env=environment(), check=True)
            # Only publish after the complete planned matrix has returned.
            assert_source(source)
            for check in (False, True):
                subprocess.run([sys.executable, "scripts/v730_docs_build.py", "--campaign",
                                f"{CAMPAIGN}_full_clean", *(["--check"] if check else [])],
                               cwd=ROOT, env=environment(), check=True)
            write_json(STATE / "state.json", dict(status="evaluated", updated_at=now(),
                        total=len(jobs), reports_generated=True, final_review_required=True))
        except Exception as exc:
            write_json(STATE / "state.json", dict(status="needs_review", error=str(exc),
                        updated_at=now(), source_commit=source))
            raise
        finally:
            write_json(STATE / "stage_event.json", json.loads((STATE / "state.json").read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
