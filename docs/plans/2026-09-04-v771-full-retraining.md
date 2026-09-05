# V7.7.1 full-terminal regeneration and retraining

Superseded by the user-authorized [V7.7.2 consolidation](2026-09-05-v772-full-retraining.md).
The existing queue is retained as its frozen training backend. There is no
separate V7.7.1 release. The original planning record follows.

Status: preparation; no new accuracy result or model promotion is claimed.
The release remains V7.7.0 until the final evidence is collected.

## Scope and acceptance

Regenerate ten canonical per-technology/polarity datasets from BSIM-CMG OSDI.
Train DirectNet-Full LEVEL=75 and BSIM-AR-Full LEVEL=76 for small, medium,
large, and XL across TSMC5/6/7/12/16, NMOS and PMOS: 80 complete bundles.
Use the current clean recipe, seed 42, grouped splits, EMA, all generated
rows, and the CLI's existing epoch/patience presets. Transformer uses its
default six-target teacher-forced training and deployed-rollout validation.
ASAP7 is reference-only; reduced LEVEL=73/74 remain retired.

Success means complete, provenance-valid execution, not forcing every
scientific gate to pass. Retain failing and unconverged configurations in
their declared denominators. Ground truth, metrics, thresholds, and diagnostic
boundaries are owned by [accuracy methodology](../accuracy/methodology.md)
and the [harness contract](../../tests/README.md). Do not tune thresholds.

## Dependency schedule

| Stage | Planning allowance | Exit condition |
|---|---|---|
| Preparation | first session | clean source commit; root and PyCMG tests pass; OSDI/private cards available |
| Data generation and labeling | first day | 10 checksum-valid datasets and validated label sidecars |
| Training | days 1–4, revise from measured progress | 80 checksum-valid bundles, including all TFF config sidecars |
| Device and circuit evaluation | days 3–5 after complete training | all 600 clean jobs and the full simple-v2 diagnostic pool have explicit verdicts |
| Analysis and corrections | after each failed stage | reproducible bug fixes and fresh evidence for changed behavior |
| Release | after complete evidence | related Markdown updated, V7.7.1 metadata consistent, checks pass, commit pushed |

These are planning allowances, not a deadline. Earlier completed training jobs
took approximately 13 minutes–7 hours for DirectNet and 1.5–13 hours for
Transformer; incomplete large/XL Transformer runs are not duration evidence.
Adjust the forecast from recorded start/end times without reducing the matrix
or shortening training merely to meet a date.

Use physical GPUs 0, 3, and 4, initially one training process per GPU.
GPUs 1 and 2 were occupied at kickoff. Use at most 40 generation workers and
wait when another user's compute process occupies a selected GPU; GPU 0 became
occupied during preparation, leaving GPUs 3 and 4 available initially. Use
16 concurrent CPU gate workers. Main scored execution stays CPU/OMP/MKL/Torch
one thread; the pre-existing OMP 2/4 latch probes remain separate stability
checks. Run short small-model jobs first to expose pipeline failures, then
schedule longer Transformer jobs before shorter jobs to reduce the final tail.

## Source and artifact isolation

Worktree: `/data2/home/shenshan/PyCircuitSim-v771`, branch `release/v7.7.1`.
The original worktree's four pre-existing edits are preserved there and copied
into this isolated worktree; their original patch is in
`results/v771_setup/preexisting.patch`. Existing control artifacts stay intact.
All new raw evidence is below `results/v771_*`; private PDKs remain ignored.

The first clean epoch (`a82369b`) completed all datasets and the TSMC5
DirectNet-small pair. Its early pilot and temperature-attribution evidence
remain under `results/v771_pilot_dnf_small/`. CUDA ordinal ordering on this
mixed A100/RTX host exposed a physical GPU mapping bug when GPU 0 became free.
The corrected scheduler uses UUIDs. The replacement epoch regenerates data
into `results/v771_r2_data/` and trains into `results/v771_r2_checkpoints/`;
the earlier directories remain preserved. Its partial Transformer attempts
are interrupted runs, not completed models or scientific failures.

Freeze the implementation commit before generating data. The current campaign
manifest requires dataset and campaign source commits to match. If a later
behavior fix changes that commit, preserve the previous arm and regenerate a
new coherent arm; never edit markers or merge partial campaigns to pass this
check. Documentation-only release changes follow the completed campaign.

## Monitoring and handoff

The persistent campaign runner records commands, PIDs, stage state, elapsed
times, and failures under `results/v771_campaign/`. Resume verifies completed
artifacts and reruns incomplete jobs; a failed process cannot publish a
completion marker. Label datasets before parallel training to avoid concurrent
writers creating the same sidecar.

Schedule reviews in the originating conversation every four hours and at
stage failure/completion. Each review checks process liveness, log progress,
GPU availability, failed jobs, and revised completion estimates. Stop scheduled
reviews after the final verified push. The concrete run/resume commands belong
in the [repository README](../../README.md).
