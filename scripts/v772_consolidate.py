#!/usr/bin/env python3
"""Keep the frozen training queue and start V7.7.2 evaluation when it completes."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__:
    from . import v771_campaign as campaign
    from .v710_regate_manifest import validate_dataset_source
else:
    import v771_campaign as campaign
    from v710_regate_manifest import validate_dataset_source


def training_progress(root: Path, commit: str) -> dict:
    """Only successful jobs with the frozen training identity satisfy the dependency."""
    complete, running, failed = [], [], []
    for model, size, tech, device in campaign.training_jobs():
        tag = "dnf" if model == "direct" else "tff"
        stem = f"{tech}_{tag}_{size}_{device}"
        path = root / "results/v771_campaign/jobs" / f"train-{stem}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if record["source_commit"] != commit:
            raise ValueError(f"training source changed: {stem}")
        if record["status"] == "complete":
            if record.get("returncode") != 0:
                raise ValueError(f"training did not exit successfully: {stem}")
            complete.append(stem)
        elif record["status"] == "failed":
            failed.append(stem)
        else:
            running.append(stem)
    return {"complete": complete, "running": running, "failed": failed,
            "total": len(campaign.training_jobs()), "updated_at": campaign.timestamp()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", required=True, type=Path)
    parser.add_argument("--training-source", required=True)
    parser.add_argument("--training-service", default="pycircuitsim-v771-campaign.service")
    parser.add_argument("--gate-parallel", type=int, default=16)
    args = parser.parse_args()
    if args.gate_parallel < 1:
        parser.error("--gate-parallel must be positive")
    campaign.configure_campaign("v772")
    campaign.DATA = args.training_root / "results/v771_r2_data"
    campaign.CHECKPOINTS = args.training_root / "results/v771_r2_checkpoints"
    gate_source = campaign.git("rev-parse", "HEAD")
    campaign.assert_source(gate_source)
    identity = validate_dataset_source(campaign.ROOT, {args.training_source},
                                       gate_source, args.training_source)
    campaign.STATE.mkdir(parents=True, exist_ok=True)
    with (campaign.STATE / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"stage": "train", "status": "running", "pid": os.getpid(),
                 "source_commit": gate_source, "training_source_commit": args.training_source,
                 "training_root": str(args.training_root), "updated_at": campaign.timestamp()}
        campaign.write_json(campaign.STATE / "state.json", state)
        campaign.write_json(campaign.STATE / "source_equivalence.json", identity)
        try:
            while True:
                campaign.assert_source(gate_source)
                source = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                 cwd=args.training_root, text=True).strip()
                dirty = subprocess.check_output(["git", "status", "--porcelain"],
                                                cwd=args.training_root, text=True).strip()
                if source != args.training_source or dirty:
                    raise RuntimeError("frozen training source drifted")
                progress = training_progress(args.training_root, args.training_source)
                campaign.write_json(campaign.STATE / "training_progress.json", progress)
                if progress["failed"]:
                    raise RuntimeError(f"training jobs failed: {progress['failed']}")
                if len(progress["complete"]) == progress["total"]:
                    break
                active = subprocess.run(["systemctl", "--user", "is-active", "--quiet",
                                         args.training_service], check=False)
                if active.returncode != 0:
                    # Restarts use the same frozen queue with --stage train.
                    subprocess.run(["systemctl", "--user", "start", args.training_service], check=True)
                time.sleep(30)
            # The original live supervisor predates --stage train. Stop its
            # optional old evaluation tail only after every training job ended.
            subprocess.run(["systemctl", "--user", "stop", args.training_service], check=True)
            env = campaign.base_environment()
            env["V710_DATASET_SOURCE_COMMIT"] = args.training_source
            for key in list(os.environ):
                if key.startswith(("PYCIRCUITSIM_", "BSIMAR_", "V710_")):
                    del os.environ[key]
            state.update(stage="evaluate", status="running", updated_at=campaign.timestamp())
            campaign.write_json(campaign.STATE / "state.json", state)
            campaign.evaluate(sys.executable, gate_source, env, args.gate_parallel)
        except Exception as exc:
            state.update(status="failed", error=str(exc), updated_at=campaign.timestamp())
            campaign.write_json(campaign.STATE / "state.json", state)
            raise
        state.update(status="complete", updated_at=campaign.timestamp())
        campaign.write_json(campaign.STATE / "state.json", state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
