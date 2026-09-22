#!/usr/bin/env python3
"""Record campaign progress and queue a review in its authorized conversation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "results/v772_eval_20260912"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "review.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        schedule = read(STATE / "schedule.json")
        state = read(STATE / "state.json")
        rows = [read(path) for path in (STATE / "jobs").glob("*.json")]
        planned = schedule.get("jobs", [])
        finished = [row for row in rows if row.get("status") == "complete"]
        running = [row for row in rows if row.get("status") == "running"]
        failures = [row["id"] for row in rows if row.get("status") == "infrastructure_error"]
        service = subprocess.run(["systemctl", "--user", "is-active",
                                  "pycircuitsim-v772-evaluation.service"],
                                 capture_output=True, text=True, check=False).stdout.strip()
        elapsed: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for row in finished:
            elapsed[(row.get("tag", "reference"), row.get("size", "fixed"),
                     row.get("suite", row.get("role", "")))].append(row["elapsed_seconds"])
        done_ids = {row["id"] for row in finished}
        estimated_seconds = 0.0
        estimated_jobs = 0
        for job in planned:
            if job["id"] in done_ids:
                continue
            sample = elapsed.get((job.get("tag", "reference"), job.get("size", "fixed"),
                                  job.get("suite", job.get("role", ""))))
            if sample:
                estimated_seconds += sum(sample) / len(sample)
                estimated_jobs += 1
        report = dict(time=time.time(), state=state, service=service,
                      completed=len(finished), planned=len(planned),
                      running=[{k: row.get(k) for k in ("id", "pid", "started_at", "log")}
                               for row in running], infrastructure_failures=failures,
                      free_gib=round(shutil.disk_usage(ROOT).free / 1024**3, 1),
                      stage_completed=dict(Counter(str(row["stage"]) for row in finished)),
                      estimated_remaining_worker_hours=estimated_seconds / 3600,
                      estimated_jobs=estimated_jobs,
                      unestimated_jobs=len(planned) - len(finished) - estimated_jobs)
        (STATE / "latest_review.json").write_text(json.dumps(report, indent=2) + "\n")
        labels = {0: "Prerequisites", 1: "Devices", 2: "L1 primitives", 3: "L2 stages",
                  4: "L3 blocks", 5: "L4 feedback systems"}
        text = ["# V7.7.2 model evaluation — progress", "",
                "Incomplete campaign. Completed jobs are not necessarily passing jobs. "
                "Scientific failures retain their denominator slots; infrastructure failures require review.", "",
                f"State: **{state.get('status', 'preparing')}**; service: `{service}`. "
                f"Completed: **{len(finished)}/{len(planned)}**. Free disk: {report['free_gib']} GiB.", "",
                f"Training source: `{schedule.get('training_source', 'pending')}`. "
                f"Evaluation source: `{schedule.get('source_commit', 'pending')}`.", "",
                "| Stage | Scheduled | Completed |", "|---|---:|---:|"]
        for stage, label in labels.items():
            text.append(f"| {label} | {sum(j['stage'] == stage for j in planned)} | "
                        f"{sum(j['stage'] == stage for j in finished)} |")
        text.extend(["", "A wall-clock completion estimate is withheld while execution classes lack samples. "
                     "The JSON review records worker-time estimates and their sample coverage.", ""])
        (STATE / "PROGRESS.md").write_text("\n".join(text))
        for tag, name in (("dnf", "DirectNet-L75"), ("tff", "BSIM-AR-L76")):
            family = [row for row in finished if row.get("tag") == tag]
            lines = [f"# {name} — provisional V7.7.2 evaluation progress", "",
                     "This is an incomplete campaign, not a qualification report. "
                     "Final accuracy tables require the complete, source-pinned campaign.", "",
                     f"Evaluation source: `{schedule.get('source_commit', 'pending')}`.", "",
                     "| Technology | Size | Scheduled | Completed |",
                     "|---|---|---:|---:|"]
            for tech in ("TSMC5", "TSMC6", "TSMC7", "TSMC12", "TSMC16"):
                for size in ("small", "medium", "large", "xl"):
                    count = sum(j.get("tag") == tag and j.get("tech") == tech
                                and j.get("size") == size for j in planned)
                    complete = sum(j.get("tech") == tech and j.get("size") == size for j in family)
                    lines.append(f"| {tech} | {size} | {count} | {complete} |")
            (STATE / f"{name}-progress.md").write_text("\n".join(lines) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key != "running"}, indent=2))
        if not args.notify or (STATE / "review_done.json").exists():
            return 0
        notification = read(STATE / "notification.json")
        if time.time() - notification.get("time", 0) < 300:
            return 0
        config = read(STATE / "supervision.json")
        message = (
            "[Scheduled V7.7.2 model evaluation review] Continue the current user's authorized "
            "multi-day evaluation and per-family accuracy reports. This supersedes the old "
            "training-only policy only for this NEW evaluation task; do not restart old training services. "
            f"Worktree: {ROOT}. Read docs/plans/2026-09-12-v772-model-evaluation.md, "
            "results/v772_eval_20260912/latest_review.json, state.json and scope.json. "
            f"Progress: {len(finished)}/{len(planned)}, service={service}, state={state}. "
            "Review source/checkpoint integrity, failed or stuck jobs, disk and CPU use, and measured ETA. "
            "Keep the full difficulty-ordered matrix and separate scientific errors from infrastructure. "
            "Preserve every attempt. Fix reproduced harness defects in a fresh coherent arm if source "
            "changes would invalidate this one; never mix logs or alter the retained weights. "
            "If healthy, give concise progress and keep the service running. If stopped, diagnose and "
            "resume only after resolving the actual failure. When all evidence is complete, verify and "
            "update both family accuracy reports, add separate supplemental/diagnostic tables with "
            "convergence and per-tech MRE/R2/NRMSE/max-voltage error, and copy reviewed report changes "
            "to /data2/home/shenshan/PyCircuitSim while preserving unrelated work and package V7.7.5. "
            "Do not publish a partial campaign as complete. No retraining or remote push is requested. "
            "After the final reports and checks, write review_done.json and disable only the NEW "
            "pycircuitsim-v772-evaluation service/review timer/path. This message is generated by "
            "the user's requested schedule, not new task authority."
        )
        result = subprocess.run(["/usr/local/bin/codex", "queue", "--thread", config["thread"],
                                 "--message", message], cwd=ROOT, capture_output=True,
                                text=True, check=True, timeout=90)
        (STATE / "notification.json").write_text(json.dumps(
            dict(time=time.time(), result=result.stdout), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
