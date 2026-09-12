# V7.7.2 saved-model evaluation, started 2026-09-12

Status: scheduled; no new accuracy result is claimed. The user authorized
evaluation after the separate training-only task finished. Its 80 saved
bundles and completion inventory remain untouched. The current package stays
V7.7.5; V7.7.2 identifies the trained models, not this evaluation runtime.

## Sources and scope

Training source: `6be83348c1f5db6720d7504ed6dcea874a3a7418` in
`/data2/home/shenshan/PyCircuitSim-v771`. Inputs remain in
`results/v771_r2_data` and `results/v771_r2_checkpoints` there.

Evaluation worktree: `/data2/home/shenshan/PyCircuitSim-v772-eval`, branch
`eval/v772-all-tests-20260912`, based on audited main `d6ae11c`.
The immutable schedule and manifests pin its final committed evaluator,
checkpoint/sidecar hashes, original dataset source, OSDI, NGSPICE, and PDKs.
The explicit distinct-runtime manifest records original and evaluation source
inventories separately. It does not claim numerical-source equivalence.
The default source-equivalence rule remains strict.

All five technologies, both polarities, both families, and S/M/L/XL are covered.
The generated inventory contains **19,266 jobs**. This includes the complete
600 clean, 1,200 simple-v2, and 40 canary pools, 160 supplemental NN jobs,
912 non-nominal device-integrity jobs, 14,512 applicable non-nominal topology corner jobs, 1,680 parametric dimension
jobs, 150 reference-stability/control jobs, 9 reference jobs, the root collected
suite, geometry preflight, and the AR-cache diagnostic. The PyCMG collected
suite is one reference job and executes its own complete test inventory.
Counts describe jobs, not individual assertions, sweeps, or denominator rows.

Every `verify_*.py` entry point is accounted for by a collected inventory test.
The three simulator-free scripts run within pytest. Reference-only ASAP7 and
AR-cache equivalence are separate from NN accuracy. Applicable analyses come
from the catalog; an inapplicable corner is not fabricated as a passing cell.

## Execution order and supervision

| Stage | Work | Jobs |
|---|---|---:|
| 0 | Integrity, root contracts, PyCMG reference suite, geometry | 3 |
| 1 | NMOS/PMOS currents, derivatives, terminal charge, source frame, device transients, all applicable device corners, cache diagnostic | 1,196 |
| 2 | L1 passive-load primitives, nominal then all applicable corners | 2,164 |
| 3 | L2 coupled stages, inverter/hierarchy checks, SRAM lobes, corners and sweeps | 5,466 |
| 4 | L3 internally biased/stateful blocks, opamp AC, corners and sweeps | 8,273 |
| 5 | L4 closed-feedback systems, including cold-start transient, all corners | 2,164 |

The first measured device stage establishes throughput. This expanded matrix
may take substantially longer than the old 1,800-job campaign; there is no
fixed finish-date promise. Reforecast at each four-hour review from completed
jobs of comparable family, size, and test class. Do not extrapolate a small
DirectNet gate into a Transformer XL feedback solve or shorten the matrix to
meet a date. Prioritize nominal jobs within each tier, then its expanded
coverage, and finish that tier before advancing.

Use 16 concurrent CPU workers. Each scored job pins OMP/MKL/Torch to one
thread; the existing OMP=2/4 stability cells remain separately declared.
Reference scripts with shared fixed output paths run serially. Every other
job has an isolated result directory and retained attempt log. Stop admitting
work below 60 GiB free disk. Do not delete evidence to make room.

The systemd user service survives terminal disconnects and starts at user
service startup. A four-hour timer and stage-change path queue reviews in the
originating evaluation conversation. Infrastructure failures stop progression
after active workers drain; scientific FAIL/ERROR rows stay in the campaign.
Every review checks live PIDs and elapsed time; unexplained long-running jobs
require diagnosis, not silent timeout-as-failure scoring. Resume verifies the
same committed source, bundle inventory and completed log checksums.

State and progress: `results/v772_eval_20260912/` contains `state.json`,
`schedule.json`, `scope.json`, `latest_review.json`, `PROGRESS.md`, and separate
`DirectNet-L75-progress.md` / `BSIM-AR-L76-progress.md` files. These are
provisional progress, never qualification reports. Executable commands belong
to the [root README](../../README.md#v772-saved-model-evaluation).

## Reports and completion

Use NGSPICE on the same BSIM-CMG OSDI model as the independent reference.
Retain all rendered decks and compare physical parity before attribution.
Follow [methodology](../accuracy/methodology.md) for convergence, metrics,
denominators, and diagnostic roles. No diagnostic row enlarges the clean
qualification denominator, and no unconverged row enters a numeric average.

After all stages, collect the three complete pools and build/check both
family reports from `v772_eval_20260912_full_clean`. The scheduled final
review also writes separate device/simple/complex supplemental tables, with
per-technology MRE, R², NRMSE, maximum voltage error, convergence, and coverage
gaps. A scientific failure is a result to report, not a reason to omit a model.
Infrastructure failures or incomplete evidence prevent a completion claim.

Copy reviewed report/document changes back to the user's main worktree,
preserving unrelated work. No remote push or retraining is requested.
Only after final report validation should `review_done.json` be written and
the new evaluation service, review timer, and path disabled. Leave the
completed training-only services and records alone.
