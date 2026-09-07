# Accuracy reports

Ground truth is NGSPICE using the identical BSIM-CMG LEVEL=72 OSDI model. Read
[`methodology.md`](methodology.md) before comparing results.

The [V7.7.2 refresh](../plans/2026-09-05-v772-full-retraining.md) is in progress.
Its fresh S/M/L/XL evidence will replace the reports only after complete
collection and validation; the measurements below remain historical controls.
The corrected solver on `main` awaits its separately provenanced
[V7.7.3 evaluation arm](../plans/2026-09-06-v773-corrected-solver-arm.md).
The package release is recorded in the [root README](../../README.md); V7.7.5
cleanup does not change these campaign verdicts.

## Current NN policy

V7.7.0 retired the reduced LEVEL=73/74 families. DirectNet-Full (LEVEL=75) is
the default NN path and BSIM-AR-Full (LEVEL=76) is the autoregressive
alternative. This is an architecture-maintenance decision, not a new accuracy
campaign or a retroactive scientific promotion.

The latest LEVEL=75 evidence remains mixed: its declared simple-circuit matrix
passes 20/20, its inverter matrix passes 100/100, and device AC passes 10/10;
parametric device DC is 115/129 and Miller open-loop AC is 2/5. LEVEL=76 still
lacks a complete five-technology clean matrix; the tracked recovery report is
TSMC5-only and leaves the Miller opamp as `ERROR`.

| LEVEL | family | runtime role | latest applicable evidence |
|---:|---|---|---|
| 75 | DirectNet-Full | default | [`DirectNet-L75-clean.md`](DirectNet-L75-clean.md) |
| 76 | BSIM-AR-Full | autoregressive alternative | [`BSIM-AR-L76-simple-circuits.md`](BSIM-AR-L76-simple-circuits.md) |

## Reports, diagnostics, and historical audits

| file | scope |
|---|---|
| [`DirectNet-L75-clean.md`](DirectNet-L75-clean.md) | latest clean LEVEL=75 qualification and open gaps |
| [`BSIM-AR-L76-clean.md`](BSIM-AR-L76-clean.md) | explicit absence of a complete five-technology clean matrix |
| [`DirectNet-L75-v763-targeted.md`](DirectNet-L75-v763-targeted.md) | targeted four-scale recovery; not a clean replacement |
| [`DirectNet-L75-v764-terminal-followup.md`](DirectNet-L75-v764-terminal-followup.md) | terminal-length, globalization, and matched-data experiments |
| [`DirectNet-L75-V760-recovery.md`](DirectNet-L75-V760-recovery.md) | initial full-terminal attribution and recovery |
| [`BSIM-AR-L76-simple-circuits.md`](BSIM-AR-L76-simple-circuits.md) | TSMC5 autoregressive recovery and capacity study |
| [`simple-circuits-v2-topologies.md`](simple-circuits-v2-topologies.md) | held-out topology/corner and promotion contract |
| [`device-and-feedback-coverage-v767.md`](device-and-feedback-coverage-v767.md) | device-integrity and feedback diagnostics |
| [`v772-harness-audit.md`](v772-harness-audit.md) | gate-reachability, coverage, and contract-drift review |
| [`v7610-harness-audit.md`](v7610-harness-audit.md) | metric-oracle, hierarchy, CLI, and stale-test audit |
| [`v769-harness-audit.md`](v769-harness-audit.md) | harness coverage and engine-agreement audit |
| [`v768-template-harness-audit.md`](v768-template-harness-audit.md) | template inventory and harness repairs |

## Retired-family history

LEVEL=73/74 clean and recipe reports, the superseded V7.5.15 recheck, and
pre-fix claims register are retained in Git history. The
[V7.7.5 cleanup ledger](../CHANGELOG.md#v775--repository-cleanup) identifies the
last revision and the retractions that still apply. These reports describe
removed runtimes and cannot qualify the current LEVEL=75/76 models.

## Evidence reproduction

The [root clean-campaign workflow](../../README.md#run-the-complete-clean-checkpoint-matrix)
owns launch, collection, coverage, and report-build commands. The
[root stage interface](../../README.md#nn-workflow-from-the-project-root)
provides selected device/circuit evaluations in isolated run directories.

The job generator writes clean, simple-v2, and source-frame canary pools.
The current campaign runner dispatches all three after checking geometry
coverage against the selected dataset root. The frozen V7.7.2 worktree has its
own inventory; the [campaign plan](../plans/2026-09-05-v772-full-retraining.md)
records that distinction.

The shared report builder publishes clean LEVEL=75/76 reports only. Without a
complete matrix it preserves an existing report only if its pinned checksum
matches. Named training recipes remain supported, but their run summaries do
not become clean-report evidence. Raw artifacts stay under `results/`, ignored
by Git; [methodology §8](methodology.md#8-measurement-caveats-and-harness-corrections)
owns explicit training/harness source equivalence.
