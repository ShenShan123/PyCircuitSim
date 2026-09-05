# V7.7.2 full-terminal regeneration and retraining

Status: preparation. V7.7.2 is the target release; no new accuracy result or
model promotion is claimed. Package release metadata changes after scoring.

## Scope and acceptance

Regenerate all ten canonical TSMC5/6/7/12/16 NMOS/PMOS datasets from the
BSIM-CMG OSDI evaluator. Train both current four-terminal families,
DirectNet-Full LEVEL=75 and BSIM-AR-Full LEVEL=76, at S/M/L/XL: 80 complete
bundles. Keep the existing clean recipe, seed 42, grouped split, EMA, full
generated dataset, and each size's epoch/patience preset. Transformer uses
the current default six-target training with deployed-rollout validation.
ASAP7 remains reference-only and LEVEL=73/74 remain retired.

Success means every requested model and harness cell has valid provenance and
an explicit outcome. It does not mean changing thresholds until every model
passes. Run both the 600-job clean pool and 1,200-job simple-v2 topology pool
from the current job catalog. Report convergence separately from MRE, R²,
NRMSE, and maximum voltage error, by technology, family, and size. Keep
diagnostics out of qualification totals. The [methodology](../accuracy/methodology.md)
and [test contract](../../tests/README.md) own the scoring rules.

## Dependency schedule

| Stage | Allocation and sequence | Completion condition |
|---|---|---|
| Preflight | first session, CPU | project and PyCMG tests, clean source, private cards and OSDI verified |
| Fresh generation | 10 jobs, 4 workers each | 10 canonical datasets and validated label sidecars |
| Training dependency | preserve active V7.7.1 | predecessor records completed training or enters evaluation |
| Training | physical GPUs 0/3/4, one job per idle GPU | 80 checksum-valid bundles with required sidecars and completion markers |
| Early canaries | paired TSMC5 small bundles, 2 CPU workers | DC, inverter transient, AC, device and terminal integrity reviewed |
| Final evaluation | after all training, 16 CPU workers | all clean and simple-v2 cells collected with matching provenance |
| Release | after analysis and any necessary corrections | reports, related Markdown, V7.7.2 metadata, verification, final commit and push |

V7.7.1 was already training at kickoff. Preserve its processes and evidence;
V7.7.2 is a fresh successor unless the user explicitly supersedes that work.
Data generation can overlap; GPU training waits for its training dependency.
Foreign GPU processes are never stopped. The worker resolves physical GPU
UUIDs and waits for an idle device before claiming a job.

Allow multiple days after GPU availability. At kickoff, predecessor XL jobs
had reached about 25–30 of 300 epochs in 3–4 hours. Those partial runs are not
completed-duration evidence. Recalculate forecasts at reviews from completed
jobs and epoch progress; do not reduce epochs or omit matrix cells to meet a
calendar estimate. Run small pairs first to expose pipeline failures, then
long Transformer tiers to reduce the final queue tail.

## Source and artifact isolation

Worktree: `/data2/home/shenshan/PyCircuitSim-v772`, branch `release/v7.7.2`.
Base: `82c4a8c`, including the tested V7.7.1 subthreshold-generator,
physical-GPU-mapping, and unavailable-device-metric fixes. The original main
worktree's edits remain untouched; their patch is preserved in
`results/v772_setup/preexisting.patch`.

Fresh data, checkpoints, scheduler state, and evidence live under
`results/v772_*`. Never reuse predecessor models as newly trained V7.7.2
models. Freeze a clean source commit before generating data and preserve it
through scoring. If a behavior change becomes necessary, preserve the prior
arm, reproduce the failure, fix and verify it, and create a fresh coherent
source epoch. Never rewrite completion markers or combine partial reports.
Documentation-only release changes follow completed scoring.

## Persistent supervision and release handoff

The existing campaign runner supports an isolated `v772` selection and records
commands, environments, source identity, PIDs, elapsed times, and errors.
Resume verifies finished bundles; interrupted training restarts from its
recorded seed. Per-job completion markers distinguish best-so-far weights
from finished training.

The user service persists across terminal disconnection. Queue a review in
the originating conversation every four hours, on stage changes, and after
early canaries. Reviews check service/process liveness, data/model counts,
epoch progress, failures, GPU occupancy, and forecast changes. Missing
predecessor state fails visibly; a failed predecessor never silently satisfies
the training dependency. Concrete commands belong in the [README](../../README.md).

Before final publication, verify both pool denominators and source manifests,
analyze all scientific failures and infrastructure errors, generate the clean
family reports, and summarize simple-v2 results by tier. Update the accuracy
index, relevant package/test documentation, changelog, README and version
identity. Run required checks and verify the final commit is pushed. Only
then write `results/v772_campaign/release_done.json` and disable the V7.7.2
services/timers. Keep the V7.7.1 schedule independent.
