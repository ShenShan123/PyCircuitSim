# Neural compact-model package

This package contains the shared data, normalization, model, loss, training,
and evaluation code for PyCircuitSim's four neural compact-model families.

| tag | LEVEL | architecture | output contract |
|---|---:|---|---|
| `dn` | 73 | DirectNet MLP | reduced 13-output |
| `tf` | 74 | BSIM-AR Transformer | reduced 13-output |
| `dnf` | 75 | DirectNet MLP | full-terminal six-surface |
| `tff` | 76 | BSIM-AR Transformer | full-terminal six-surface |

The repository [README](../../README.md) owns environment setup, dataset
generation, training commands, checkpoint selection, and the five-stage
verification workflow. Current measurements and qualification decisions live
in [`docs/accuracy/`](../../docs/accuracy/).

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
├── checkpoints/       generated runtime bundles (ignored)
└── results/           generated training output (ignored)
```

PyCMG generates source-relative training data from the same BSIM-CMG OSDI
binary used by NGSPICE ground truth. `data/contracts.py` owns output-contract
names and ordered schemas; `data/normalize.py` owns transforms and persisted
normalization statistics.

## Full-terminal contract

LEVEL=75 and 76 learn `i_d`, `i_g`, `i_b`, `qd`, `qg`, and `qb`. Source
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

Run package changes through the root unit suite and the PyCMG reference suite
documented in the repository README. Preserve the optional monotone, EKV,
Sobolev, subthreshold, and charge-Sobolev structures: existing checkpoints
depend on their state-dict shapes even when those options are disabled.
