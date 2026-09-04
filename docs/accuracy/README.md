# Accuracy reports

Ground truth is NGSPICE using the identical BSIM-CMG LEVEL=72 OSDI model. Read
[`methodology.md`](methodology.md) before comparing results.

## Current NN policy

V7.7.0 retires the reduced LEVEL=73/74 families. DirectNet-Full (LEVEL=75) is
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

## Current reports and diagnostics

| file | scope |
|---|---|
| [`DirectNet-L75-clean.md`](DirectNet-L75-clean.md) | latest clean LEVEL=75 qualification and open gaps |
| [`DirectNet-L75-v763-targeted.md`](DirectNet-L75-v763-targeted.md) | targeted four-scale recovery; not a clean replacement |
| [`DirectNet-L75-v764-terminal-followup.md`](DirectNet-L75-v764-terminal-followup.md) | terminal-length, globalization, and matched-data experiments |
| [`DirectNet-L75-V760-recovery.md`](DirectNet-L75-V760-recovery.md) | initial full-terminal attribution and recovery |
| [`BSIM-AR-L76-simple-circuits.md`](BSIM-AR-L76-simple-circuits.md) | TSMC5 autoregressive recovery and capacity study |
| [`simple-circuits-v2-topologies.md`](simple-circuits-v2-topologies.md) | held-out topology/corner and promotion contract |
| [`device-and-feedback-coverage-v767.md`](device-and-feedback-coverage-v767.md) | device-integrity and feedback diagnostics |
| [`v7610-harness-audit.md`](v7610-harness-audit.md) | metric-oracle, hierarchy, CLI, and stale-test audit |
| [`v769-harness-audit.md`](v769-harness-audit.md) | harness coverage and engine-agreement audit |
| [`v768-template-harness-audit.md`](v768-template-harness-audit.md) | template inventory and harness repairs |

## Retired-family historical evidence

These reports preserve measurements and dead ends for the removed 3-terminal
families. They are not supported runtime or training documentation:

- [`DirectNet-L73-clean.md`](DirectNet-L73-clean.md)
- [`DirectNet-L73-recipes.md`](DirectNet-L73-recipes.md)
- [`BSIM-AR-L74-clean.md`](BSIM-AR-L74-clean.md)
- [`BSIM-AR-L74-recipes.md`](BSIM-AR-L74-recipes.md)
- [`simple-circuits-recheck-2026-08-19.md`](simple-circuits-recheck-2026-08-19.md)
- [`archive-pre-gds-fix.md`](archive-pre-gds-fix.md)

## Evidence reproduction

Generate the current full-terminal job pool and collect one complete campaign:

```bash
conda run -n pycircuitsim python \
  scripts/v710_regate_jobs.py results/v770_full_clean/job_lists

BSIMAR_CHECKPOINT_DIR="$PWD/results/v770_full_checkpoints" \
V710_OUT="$PWD/results/v770_full_clean" \
V710_SCRATCH=/tmp/pycircuitsim-v770-full \
NGSPICE_BIN=/usr/local/ngspice-45.2/bin/ngspice \
JOBS="$PWD/results/v770_full_clean/job_lists/jobs_clean.txt" PAR=32 \
NN_PY="$(conda run -n pycircuitsim which python)" \
bash scripts/v710_regate.sh

conda run -n pycircuitsim python scripts/v710_regate_collect.py \
  --root results/v770_full_clean --require-manifest
```

Coverage and report generation fail closed on incomplete bundles or campaign
cells:

```bash
for family in dnf tff; do
  BSIMAR_CHECKPOINT_DIR="$PWD/results/v770_full_checkpoints" \
  conda run -n pycircuitsim python scripts/v730_coverage.py \
    --tag "$family" --set clean --passes v770-full-clean \
    --require-complete --fail-on-gaps
done

conda run -n pycircuitsim python scripts/v730_docs_build.py
conda run -n pycircuitsim python scripts/v730_docs_build.py --check
```

Raw evidence and generated artifacts remain under `results/` and are ignored by
Git. Never combine partial passes or reuse checkpoint provenance across source
commits.
