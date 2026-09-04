# Accuracy reports

Ground truth is always NGSPICE using the identical BSIM-CMG LEVEL=72 OSDI
model. See [`methodology.md`](methodology.md) for the shared gate and evidence
contract before comparing results.

## Reports

| file | scope |
|---|---|
| [`v769-harness-audit.md`](v769-harness-audit.md) | V7.6.9 test-harness coverage audit: uncollected suites, untested shipped features, engine-agreement guard, corner enrichment; not a score |
| [`v768-template-harness-audit.md`](v768-template-harness-audit.md) | V7.6.8 template inventory, redundancy review, harness repairs, and smoke evidence; not a score |
| [`simple-circuits-v2-topologies.md`](simple-circuits-v2-topologies.md) | held-out simple-topology catalog, corner matrix, harness and promotion contract |
| [`device-and-feedback-coverage-v767.md`](device-and-feedback-coverage-v767.md) | V7.6.7 device-integrity and self-bias/feedback diagnostics; characterization, not a score |
| [`DirectNet-L73-clean.md`](DirectNet-L73-clean.md) | production DirectNet, all clean tiers |
| [`BSIM-AR-L74-clean.md`](BSIM-AR-L74-clean.md) | autoregressive Transformer, all clean tiers |
| [`DirectNet-L75-clean.md`](DirectNet-L75-clean.md) | full-terminal DirectNet production qualification |
| [`DirectNet-L75-v763-targeted.md`](DirectNet-L75-v763-targeted.md) | V7.6.3 targeted four-scale recovery; not a clean-matrix replacement |
| [`DirectNet-L75-v764-terminal-followup.md`](DirectNet-L75-v764-terminal-followup.md) | terminal-length, globalization, matched-data, and Jacobian experiments; no promotion |
| [`DirectNet-L75-V760-recovery.md`](DirectNet-L75-V760-recovery.md) | V7.6.0 attribution and experimental full-terminal status |
| [`BSIM-AR-L76-simple-circuits.md`](BSIM-AR-L76-simple-circuits.md) | TSMC5 full-terminal BSIM-AR recovery and S/M/L/XL checkpoint evaluation |
| [`DirectNet-L73-recipes.md`](DirectNet-L73-recipes.md) | historical DirectNet recipe study |
| [`BSIM-AR-L74-recipes.md`](BSIM-AR-L74-recipes.md) | historical BSIM-AR recipe study |
| [`simple-circuits-recheck-2026-08-19.md`](simple-circuits-recheck-2026-08-19.md) | superseded V7.5.15 campaign audit trail |
| [`archive-pre-gds-fix.md`](archive-pre-gds-fix.md) | retracted pre-V6.13 claims |

## Clean scoreboard

| LEVEL | family | role | current / best clean | historical best recipe | CPU cost |
|---|---|---|---|---|---|
| 73 | **DirectNet** | **production** | V7.5.17 `large` **9/20** served; `xl` **10/20** best | V7.3 `crit15m`@xl **19/20** | 1.5 ms @ `large` |
| 74 | **BSIM-AR** | higher fidelity | V7.5.17 `large` **12/20** | V7.3 `corroft`@medium **20/20** | 61.5 ms @ `medium` |
| 75 | **DirectNet-Full** | experimental, rejected | V7.6.2 `large` **5/20** | `small` **8/20** | not gated |

Strict = passes at OMP ∈ {1, 2, 4}. Totals are **/20** — 4 circuits × 5 techs, TSMC6 included (`methodology.md` §2). Earlier reports scored /16 over four techs, so a /20 total here and a /16 total there can be the same measurement.

The reduced DirectNet and BSIM-AR rows come from the V7.5.17 CPU-pinned
campaign. DirectNet-Full comes from the isolated 240-job V7.6.2 pass;
BSIM-AR-Full remains on V7.6.1 evidence. Recipe columns are historical and are
not direct deltas against the current clean contracts.

TSMC6 remains in the denominator as a controlled repeat of TSMC7. Their
LEVEL=72 data are identical, so disagreement measures training and Newton-basin
variability rather than an independent technology result.

## Evidence and reproduction

Current tables are published only when all 480 DirectNet/BSIM-AR clean jobs and
all family-required checkpoint artifacts are present. Partial passes are never
mixed.

```bash
conda run -n pycircuitsim python scripts/v710_regate_collect.py \
  --root results/v7517_clean --require-manifest

for family in dn tf; do
  BSIMAR_CHECKPOINT_DIR="$PWD/results/v7516_clean/checkpoints" \
  conda run -n pycircuitsim python scripts/v730_coverage.py \
    --tag "$family" --set clean --passes v7517-clean \
    --require-complete --fail-on-gaps
done

conda run -n pycircuitsim python scripts/v730_docs_build.py
conda run -n pycircuitsim python scripts/v730_docs_build.py --check
```

The V7.6.2 DirectNet-Full row is a separate 240-job family pass:

```bash
conda run -n pycircuitsim python scripts/v710_regate_collect.py \
  --root results/v762_directnet_full_clean --require-manifest

BSIMAR_CHECKPOINT_DIR="$PWD/results/v762_directnet_full_checkpoints" \
conda run -n pycircuitsim python scripts/v730_coverage.py \
  --tag dnf --set clean --passes v762-directnet-full \
  --require-complete --fail-on-gaps

conda run -n pycircuitsim python scripts/v730_docs_build.py \
  --check --only dnf --recipes clean
```

Current raw evidence is local and gitignored under `results/v7517_clean/` and
`results/v762_directnet_full_clean/`.
