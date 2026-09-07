# Neural compact-model package

This package contains the shared data, normalization, model, loss, training,
and evaluation code for PyCircuitSim's two full-terminal compact-model families.

| tag | LEVEL | architecture | output contract |
|---|---:|---|---|
| `dnf` | 75 | DirectNet MLP | full-terminal six-surface |
| `tff` | 76 | BSIM-AR Transformer | full-terminal six-surface |

The repository [README](../../README.md) owns environment setup, dataset
generation, training commands, checkpoint selection, and the five-stage
verification workflow. Current measurements and qualification decisions live
in [`docs/accuracy/`](../../docs/accuracy/).

The root [`main.py`](../../main.py) provides a shared interface for
independent `data`, `train`, and `evaluate` stages and their combined `flow`.
Start with the [root workflow guide](../../README.md#nn-workflow-from-the-project-root)
for isolated run directories, GPU selection, previews, and evaluation reports.
Generator and trainer argument definitions are shared in
[`../cli_options.py`](../cli_options.py). The root CLI exposes those settings,
named training recipes, and evaluation selection by device suite/circuit name.
The versioned campaign tools continue to own their release inventories.
The [V7.7.5 cleanup](../../docs/CHANGELOG.md#v775--repository-cleanup) removed
unused helpers and retired report scaffolding. It did not retrain or promote
these models; the repository README owns the current package version.

The [V7.7.2 campaign](../../docs/plans/2026-09-05-v772-full-retraining.md)
retains the already regenerated dataset and ongoing training queue from the
consolidated V7.7.1 task. Both families cover all four sizes. Original artifact
provenance is preserved; in-progress models are not promoted bundles.

## Package boundaries

```text
external_compact_models/neural_network/
├── cli/train.py       unified training entry point
├── config.py          technology vocabularies, paths, and size configuration
├── data/              dataset contracts, loading, sampling, and normalization
├── eval/              metrics and technology-label sidecars
├── losses/            physical-space training losses
├── models/            DirectNet and BSIM-AR implementations
├── training/          shared training and checkpoint lifecycle
└── checkpoints/       standalone trainer's default runtime bundles (ignored)
```

PyCMG generates source-relative training data from the same BSIM-CMG OSDI
binary used by NGSPICE ground truth. `data/contracts.py` owns the ordered
six-surface schema; `data/normalize.py` owns transforms and persisted
normalization statistics. Runtime model classes own conversion of their full
terminal Jacobians to physical units; the old scalar
`denormalize_derivative()` API was removed. Use `NormStats.load()` to load
statistics and `normalizer_from_stats()` to select the matching transform.

Root workflow runs store datasets, checkpoints, logs, and evidence under their
isolated `results/<run>/` directory. See the root
[artifact policy](../../README.md#performance-and-artifact-policy) before
removing local outputs or reusing a preserved comparison.

## Full-terminal contract

LEVEL=75/76 learn `i_d`, `i_g`, `i_b`, `qd`, `qg`, and `qb`. Source
current and charge are reconstructed analytically to preserve closure. The
canonical dataset and normalization order is
`i_d,i_g,i_b,qd,qg,qb`; BSIM-AR emits the same surfaces in its declared
autoregressive order and records that order in the configuration sidecar.

Every campaign-ready full-terminal bundle is checksum bound to its clean-source
dataset:

- DirectNet-Full: `_best.pt`, `_norm.npz`, `_best.pt.complete`.
- BSIM-AR-Full: `_best.pt`, `_norm.npz`, `_config.npz`, `_best.pt.complete`.

A bare checkpoint is best-so-far output, not a completed model. Training and
inference reject missing, stale, dirty-source, or checksum-mismatched bundles.

## Development checks

The [root test guide](../../tests/README.md) maps runtime, training, and
provenance contracts. The [PyCMG guide](../bsim_cmg/README.md) describes the
separate evaluator suite. LEVEL=73/74 checkpoints and their 13-output
compatibility paths are intentionally unsupported; their reports remain in
Git history at the revision recorded by the cleanup ledger. Named training
recipes are still supported, but the shared report builder publishes only
clean LEVEL=75/76 reports.
