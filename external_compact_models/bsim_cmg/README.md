# PyCMG — BSIM-CMG OSDI evaluator

PyCMG is the vendored Python ctypes host for the BSIM-CMG OSDI binary used by
PyCircuitSim LEVEL=72. NGSPICE loads the identical binary for independent
reference comparisons. The package also generates the canonical training data
for DirectNet-Full LEVEL=75 and BSIM-AR-Full LEVEL=76.

The repository [README](../../README.md) owns the current release, conda setup,
OSDI build, and the data → training → evaluation workflow. See its
[PyCMG reference tools](../../README.md#pycmg-reference-tools) for a working
single-device example, CSV export, and the separate PyCMG test commands.
[AGENTS.md](AGENTS.md) records evaluator implementation and debugging rules.

## Package boundaries

| Path | Responsibility |
|---|---|
| [`pycmg/model.py`](pycmg/model.py) | Public `Model` and `Instance` API, DC/transient evaluation, terminal matrices |
| [`pycmg/core.py`](pycmg/core.py), [`pycmg/osdi_types.py`](pycmg/osdi_types.py) | ctypes bindings, OSDI state, and internal-node solves |
| [`pycmg/parser.py`](pycmg/parser.py), [`pycmg/tech.py`](pycmg/tech.py) | Modelcard parsing, technology registry, and geometry-specific card resolution |
| [`pycmg/sweep.py`](pycmg/sweep.py) | General device sweeps and CSV/NPZ export |
| [`pycmg/nn_config.py`](pycmg/nn_config.py), [`pycmg/nn_generate.py`](pycmg/nn_generate.py) | PDK-driven sampling and canonical full-terminal NN datasets |
| [`pycmg/sensitivity.py`](pycmg/sensitivity.py) | Process-parameter sensitivity diagnostics |
| [`scripts/`](scripts/) | Generator, modelcard, and sensitivity CLIs |
| [`tests/`](tests/) | API contracts and comparisons against NGSPICE |
| `build/osdi/` | Compiled reference binary; generated locally |

Source modelcards are in the repository-root [`PDKs/`](../../PDKs/). ASAP7 is
tracked and supports reference checks; private TSMC cards remain untracked.
ASAP7 is outside the LEVEL=75/76 checkpoint scope. Technology names and legal
geometries come from the registries and cards, not a copied inventory here.

## Evaluator API

`Model(osdi_path, modelcard_path, model_name)` loads a card and binary.
`Instance(model, params=..., temperature=...)` supplies geometry and temperature
in **Kelvin**. `eval_dc(nodes)` returns a dictionary; terminal keys in `nodes`
are `d`, `g`, `s`, and `e` (bulk).

| Result group | Dictionary keys |
|---|---|
| Terminal currents | `id`, `ig`, `is`, `ie` |
| Derived drain-minus-source current | `ids` |
| Charges | `qg`, `qd`, `qs`, `qb` |
| Small-signal derivatives | `gm`, `gds`, `gmb` |
| Scalar capacitance summaries | `cgg`, `cgd`, `cgs`, `cdg`, `cdd` |

Use terminal `id` for drain-current comparisons. `ids = id - is` is a
separate derived quantity and is approximately twice the drain current in a
common-source device. The NN generator and circuit adapter negate PyCMG's
terminal-current fields to implement their positive-current-leaving convention.
Their full 4×4 current and charge stamps are not replaced by these scalar
summaries. Read the [runtime contract](../../AGENTS.md#mosfet-current-and-jacobian-contract)
before changing that boundary.

`eval_tran(nodes, time, delta_t, prev_state=None)` evaluates transient terminal
currents and charges. After `eval_dc`, `condense_last_jacobian()` and
`condense_last_react()` reuse its loaded buffers for the full terminal matrices;
`get_jacobian_matrix(nodes)` performs a fresh DC evaluation. These APIs have
their own sign and ordering contracts; preserve the existing adapter conversions.

## Canonical NN datasets

Use [`scripts/generate_nn_data.py`](scripts/generate_nn_data.py), exposed by
root `main.py data`, for training-ready bundles. The generic CSV generator is
a device-inspection tool and does not create the canonical completion markers.

The canonical arrays contain source-relative voltages (`Vs=0`), geometry and
process metadata, sample-class labels, and the six independent output surfaces
`i_d,i_g,i_b,qd,qg,qb`. Source current and charge follow analytically from
closure. The authoritative column order is
[`neural_network/data/contracts.py`](../neural_network/data/contracts.py);
sampler choices and defaults are defined in [`cli_options.py`](../cli_options.py).
The runtime uses seven continuous voltage/geometry inputs plus its technology
embedding; retained process metadata does not add twelve runtime inputs.

Generation rejects non-finite values, terminal currents above 1 A, and failed
internal-node solves. Unstable NFIN=1 bins are excluded. A dataset becomes
training-ready only with its validated provenance, label sidecars, and
checksum-bound completion marker. The root workflow places new datasets,
checkpoints, and evaluation evidence in an isolated run directory under
`results/`.

## Verification and artifacts

Run the PyCMG suite separately from the root project suite because each has a
`tests` package. API/sweep smoke tests do not certify circuit accuracy;
NGSPICE comparisons and complete circuit campaigns supply that evidence.
See [accuracy methodology](../../docs/accuracy/methodology.md).

Disposable exports belong under root `results/`. The standalone PyCMG helpers
still cache resolved modelcards and NGSPICE scratch files under local `build/`;
those paths are implementation defaults, not published campaign evidence.
Keep the compiled OSDI binary and every artifact referenced by an active run or
preserved comparison. The [V7.7.5 ledger](../../docs/CHANGELOG.md#v775--repository-cleanup)
records the cleanup and recovery location for retired documentation.
