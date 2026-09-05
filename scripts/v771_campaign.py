#!/usr/bin/env python3
"""Run and resume the V7.7.1 data, training, and evaluation dependencies."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import time
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "results/v771_campaign"
DATA = ROOT / "results/v771_full_data"
CHECKPOINTS = ROOT / "results/v771_full_checkpoints"
TECHS = ("tsmc5", "tsmc6", "tsmc7", "tsmc12", "tsmc16")
DEVICES = ("nmos", "pmos")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def assert_source(commit: str) -> None:
    if git("rev-parse", "HEAD") != commit or git("status", "--porcelain"):
        raise RuntimeError("source drift: freeze a clean source before resuming")


def training_jobs() -> list[tuple[str, str, str, str]]:
    """Exercise both families early, then drain the longest model tiers first."""
    pilot = [(model, "small", "tsmc5", device)
             for model in ("direct", "transformer") for device in DEVICES]
    rest = [(model, size, tech, device)
            for model in ("transformer", "direct")
            for size in ("xl", "large", "medium", "small")
            for tech in TECHS for device in DEVICES]
    return pilot + [job for job in rest if job not in pilot]


def validate_bundle(stem: str, commit: str) -> None:
    """A completed job must still match its model, sidecars, and source data."""
    marker = json.loads((CHECKPOINTS / f"{stem}_best.pt.complete").read_text())
    pairs = [("checkpoint", "checkpoint_sha256"),
             ("normalization", "normalization_sha256")]
    if "_tff_" in stem:
        pairs.append(("configuration", "configuration_sha256"))
    for name, checksum in pairs:
        if sha256(CHECKPOINTS / marker[name]) != marker[checksum]:
            raise ValueError(f"checkpoint checksum mismatch: {stem}/{name}")
    dataset = DATA / marker["dataset"]
    completion = dataset.with_suffix(".npz.complete")
    data_marker = json.loads(completion.read_text())
    if (marker["dataset_source_commit"] != commit
            or data_marker["source_commit"] != commit
            or data_marker["source_dirty"] is not False
            or marker["dataset_sha256"] != data_marker["dataset_sha256"]
            or marker["dataset_completion_marker_sha256"] != sha256(completion)):
        raise ValueError(f"dataset provenance mismatch: {stem}")


def run_job(
    name: str, command: list[str], env: dict[str, str], commit: str,
    validate: Callable[[], None] | None = None, *, resume: bool = True,
) -> None:
    """Keep failed attempts and reject live orphan workers before resuming."""
    assert_source(commit)
    record_path = STATE / "jobs" / f"{name}.json"
    identity = {"command": command, "source_commit": commit,
                "environment": {key: value for key, value in env.items()
                                if key != "CUDA_VISIBLE_DEVICES"}}
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f"job configuration drift: {name}")
        if previous["status"] == "complete" and resume:
            if validate is not None:
                validate()
            print(f"{timestamp()} SKIP {name}", flush=True)
            return
        if previous["status"] == "running":
            process = Path(f"/proc/{previous['pid']}")
            if process.exists():
                raise RuntimeError(f"worker {previous['pid']} still exists: {name}")
    log = STATE / "logs" / f"{name}.{time.time_ns()}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record = {**identity, "gpu": env.get("CUDA_VISIBLE_DEVICES", ""),
              "started_at": timestamp(), "log": str(log),
              "status": "running"}
    print(f"{timestamp()} START {name}", flush=True)
    try:
        with log.open("w") as output:
            child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **env},
                                     stdout=output, stderr=subprocess.STDOUT)
            record["pid"] = child.pid
            write_json(record_path, record)
            record["returncode"] = child.wait()
        if record["returncode"] != 0:
            raise RuntimeError(f"{name}: exit {record['returncode']}; see {log}")
        assert_source(commit)
        if validate is not None:
            validate()
    except Exception as exc:
        record.update(status="failed", error=str(exc), finished_at=timestamp(),
                      elapsed_seconds=time.monotonic() - started)
        write_json(record_path, record)
        raise
    record.update(status="complete", finished_at=timestamp(),
                  elapsed_seconds=time.monotonic() - started)
    write_json(record_path, record)
    print(f"{timestamp()} DONE {name} ({record['elapsed_seconds']:.0f}s)", flush=True)


def base_environment() -> dict[str, str]:
    return {
        "PYTHONPATH": str(ROOT / "external_compact_models") + os.pathsep + str(ROOT),
        "BSIMAR_DATA_DIR": str(DATA), "BSIMAR_CHECKPOINT_DIR": str(CHECKPOINTS),
        "NGSPICE_BIN": "/usr/local/ngspice-45.2/bin/ngspice",
        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "PYCIRCUITSIM_TORCH_THREADS": "1",
        "OMP_WAIT_POLICY": "passive", "KMP_BLOCKTIME": "0",
        "PYTHONUNBUFFERED": "1",
    }


def generate(python: str, commit: str, env: dict[str, str], workers: int) -> None:
    def one(tech: str, device: str) -> None:
        dataset = DATA / f"{tech}_dnf_{device}.npz"
        run_job(f"data-{tech}-{device}", [python, "-u",
                str(ROOT / "external_compact_models/bsim_cmg/scripts/generate_nn_data.py"),
                "--device", device, "--tech", tech, "--enable-inv-trip",
                "--enable-subvt-off", "--allow-safety-rejections",
                "--max-l-ratio", "1.35", "--n-workers", str(workers),
                "--data-dir", str(DATA)], env, commit)
        # Validate even a resumed dataset before any trainer consumes it.
        code = (
            "import sys; from neural_network.data.dataset import validate_canonical_dataset; "
            "from neural_network.eval.loo_labels import get_or_build_tech_variant_labels; "
            "validate_canonical_dataset(sys.argv[1]); "
            "get_or_build_tech_variant_labels(sys.argv[1], sys.argv[2])"
        )
        run_job(f"labels-{tech}-{device}", [python, "-c", code, str(dataset), device],
                env, commit, resume=False)

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(one, tech, device) for tech in TECHS for device in DEVICES]
        errors = []
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("\n".join(errors))


def train(python: str, commit: str, env: dict[str, str], gpus: list[str]) -> None:
    pending: Queue[tuple[str, str, str, str]] = Queue()
    for job in training_jobs():
        pending.put(job)

    def worker(gpu: str) -> list[str]:
        errors = []
        while True:
            if pending.empty():
                return errors
            # Shared hardware can become occupied after kickoff. Do not claim
            # a job until this GPU is free; other workers can drain the queue.
            occupants = subprocess.check_output(
                ["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid",
                 "--format=csv,noheader"], text=True,
            ).strip()
            if occupants:
                time.sleep(30)
                continue
            try:
                model, size, tech, device = pending.get_nowait()
            except Empty:
                return errors
            tag = "dnf" if model == "direct" else "tff"
            stem = f"{tech}_{tag}_{size}_{device}"
            command = [python, "-u", "-m", "neural_network.cli.train",
                       "--model", model, "--size", size, "--device-type", device,
                       "--tech-scope", tech, "--data", str(DATA / f"{tech}_dnf_{device}.npz"),
                       "--swa-mode", "ema", "--seed", "42", "--cuda", "--overwrite"]
            try:
                run_job(f"train-{stem}", command,
                        {**env, "CUDA_VISIBLE_DEVICES": gpu}, commit,
                        lambda: validate_bundle(stem, commit))
            except Exception as exc:
                errors.append(str(exc))
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        results = list(pool.map(worker, gpus))
    errors = [error for result in results for error in result]
    if errors:
        raise RuntimeError("\n".join(errors))


def evaluate(python: str, commit: str, env: dict[str, str], parallel: int) -> None:
    for model, size, tech, device in training_jobs():
        tag = "dnf" if model == "direct" else "tff"
        validate_bundle(f"{tech}_{tag}_{size}_{device}", commit)
    job_lists = STATE / "job_lists"
    run_job("gate-jobs", [python, "scripts/v710_regate_jobs.py", str(job_lists)], env, commit)
    for pool in ("clean", "simple_v2"):
        output = ROOT / "results" / f"v771_full_{pool}"
        gate_env = {**env, "NN_PY": python, "PAR": str(parallel),
                    "JOBS": str(job_lists / f"jobs_{pool}.txt"),
                    "V710_OUT": str(output), "V710_SCRATCH": str(output / "artifacts"),
                    "V710_MANIFEST": str(output / "campaign_manifest.json")}
        run_job(f"evaluate-{pool}", ["bash", "scripts/v710_regate.sh"], gate_env, commit)
        run_job(f"collect-{pool}", [python, "scripts/v710_regate_collect.py",
                "--root", str(output), "--require-manifest"], env, commit)
    for tag in ("dnf", "tff"):
        run_job(f"coverage-{tag}", [python, "scripts/v730_coverage.py",
                "--tag", tag, "--set", "clean", "--passes", "v771-full-clean",
                "--require-complete", "--fail-on-gaps"], env, commit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "data", "train", "evaluate"), default="all")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="0,3,4")
    parser.add_argument("--generation-workers", type=int, default=4)
    parser.add_argument("--gate-parallel", type=int, default=16)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        records = [json.loads(path.read_text()) for path in sorted((STATE / "jobs").glob("*.json"))]
        print(json.dumps(dict(Counter(row["status"] for row in records)), sort_keys=True))
        for row in records:
            if row["status"] != "complete":
                print(row["status"], row.get("pid"), row["log"], row.get("error", ""))
        return 0
    gpus = args.gpus.split(",")
    if (not gpus or any(not gpu.isdigit() for gpu in gpus)
            or len(set(gpus)) != len(gpus)
            or args.generation_workers < 1 or args.gate_parallel < 1):
        parser.error("select unique GPU IDs and positive worker counts")
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "runner.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another V7.7.1 campaign runner is active")
        commit = git("rev-parse", "HEAD")
        assert_source(commit)
        env = base_environment()
        # An inherited experiment pin or optimization must never select another model.
        for key in list(os.environ):
            if key.startswith(("PYCIRCUITSIM_", "BSIMAR_", "V710_")):
                del os.environ[key]
        stages = {"data": lambda: generate(args.python, commit, env, args.generation_workers),
                  "train": lambda: train(args.python, commit, env, gpus),
                  "evaluate": lambda: evaluate(args.python, commit, env, args.gate_parallel)}
        for stage, action in stages.items():
            if args.stage not in ("all", stage):
                continue
            state = {"stage": stage, "status": "running", "pid": os.getpid(),
                     "source_commit": commit, "updated_at": timestamp()}
            write_json(STATE / "state.json", state)
            try:
                action()
            except Exception as exc:
                state.update(status="failed", error=str(exc), updated_at=timestamp())
                write_json(STATE / "state.json", state)
                print(str(exc), file=sys.stderr, flush=True)
                return 1
            state.update(status="complete", updated_at=timestamp())
            write_json(STATE / "state.json", state)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
