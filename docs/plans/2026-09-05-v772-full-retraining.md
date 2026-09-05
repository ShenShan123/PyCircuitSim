# V7.7.2 consolidated full-terminal campaign

Status: training. The user consolidated V7.7.1 and V7.7.2 into one campaign
and requested automatic evaluation after training. There is one 80-model
refresh and one final V7.7.2 release. No accuracy promotion is claimed yet.

## Scope and completion conditions

Retain the ten already regenerated TSMC5/6/7/12/16 NMOS/PMOS datasets and
train LEVEL=75 DirectNet-Full and LEVEL=76 BSIM-AR-Full at S/M/L/XL: 80
complete bundles. Keep the existing clean recipe, seed 42, grouped splits,
EMA, full data, and size-specific epoch/patience presets. Transformer keeps
six-target training and deployed-rollout validation. ASAP7 is reference-only.

After all training jobs succeed, validate every checkpoint and required
sidecar, then automatically run the 600-job clean pool and 1,200-job simple-v2
pool. Preserve failures in their denominators and report convergence separately
from MRE, R², NRMSE, and maximum voltage error by technology/family/size.
Diagnostics remain outside qualification totals. The [methodology](../accuracy/methodology.md)
and [test contract](../../tests/README.md) own scoring rules.

## Schedule and retained work

| Stage | State and dependency |
|---|---|
| Generation | 10 validated datasets retained from the original queue |
| Training | four small bundles complete at consolidation; three XL jobs continue without restart; remaining queue unchanged |
| Full evaluation | starts automatically after all 80 successful training jobs and bundle validation |
| Analysis and corrections | diagnose scientific failures and fix reproduced bugs with coherent evidence |
| Release | complete reports, related Markdown, V7.7.2 metadata, verification, final commit and push |

Training uses physical GPUs 0/3/4, one job per idle GPU, identified by UUID.
Keep other users' processes untouched. Evaluation uses 16 CPU workers; scored
inference is CPU/OMP/MKL/Torch one thread, with declared thread-stability
probes separate. Forecast from measured completed jobs and epoch progress;
do not shorten training or omit models to meet a calendar estimate.

The duplicate V7.7.2 generation and standalone pilot schedules were stopped;
partial artifacts and earlier diagnostics remain preserved. The historical
V7.7.1 backend service continues only to retain live workers. Its separate
release and review schedules are retired.

## Source and artifact provenance

Release/evaluation worktree: `/data2/home/shenshan/PyCircuitSim-v772`, branch
`release/v7.7.2`. Frozen training worktree:
`/data2/home/shenshan/PyCircuitSim-v771`, commit
`6be83348c1f5db6720d7504ed6dcea874a3a7418`.

Active training artifacts remain in the training worktree's
`results/v771_r2_data` and `results/v771_r2_checkpoints`. Preserve original
job records, timestamps, hashes, and completion markers. Consolidated state
and final evaluation evidence live under the release worktree's `results/v772_*`.
Do not describe retained models as newly retrained in the duplicate attempt.

The explicit consolidation supersedes the original plan to regenerate solely
because the harness commit changed. Provenance remains strict by default.
The explicit original-source option requires an exact Git inventory hash over
tracked compact-model, generator, runtime, template, PDK, and environment
inputs, excluding Markdown. It rejects numerical differences, undeclared
source changes, and mixed dataset sources. Manifests and reports name both
the training and evaluation commits and their identical numerical-source hash.
The newer source differs in the harness and orchestration only.

Keep both worktrees clean during execution. Numerical changes still require
a fresh coherent arm. Never rewrite artifact markers or combine partial
campaign reports. Documentation-only release changes follow scoring.

## Persistent supervision and handoff

The V7.7.2 supervisor leaves live training workers untouched and monitors all
80 job records. Restarting the frozen backend uses `--stage train` and skips
verified complete bundles. After every training job exits successfully, the
supervisor stops the old optional evaluation tail, validates the bundles, and
starts both full evaluation pools under the V7.7.2 harness. An unsuccessful
training job cannot satisfy this dependency.

Four-hour and stage-change reviews run in the V7.7.2 conversation. Inspect
service/process liveness, counts, epochs, failed jobs, resources and forecast.
The [README](../../README.md) owns start/resume commands. After complete
scoring, update accuracy reports and related documentation, change release
metadata to V7.7.2, verify, and commit/push. Only after the final push is
verified, write `results/v772_campaign/release_done.json` and disable the
V7.7.2 services plus the retained training backend.
