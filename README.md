# PyCircuitSim

PyCircuitSim is a pure-Python, SPICE-like circuit simulator with a shared
solver for BSIM-CMG and neural-network compact models. Its primary workflow is
to generate BSIM-CMG data, train a compact model, and increase validation scope
from devices to circuits while keeping NGSPICE on the identical OSDI model as
ground truth.

Current release: **V7.5.17**.

## Documentation map

Each project document has one job:

- This `README.md` is the executable user guide: setup, data generation,
  training, verification, and simulator use.
- [`AGENTS.md`](AGENTS.md) contains architecture contracts and debugging rules
  for contributors and coding agents.
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) records releases, measurements,
  retractions, and failed approaches.
- [`docs/accuracy/README.md`](docs/accuracy/README.md) indexes current compact-
  model scores; [`docs/accuracy/methodology.md`](docs/accuracy/methodology.md)
  defines gates, thresholds, and evidence provenance.
- [`examples/complex_circuits/RESULTS_TSMC.md`](examples/complex_circuits/RESULTS_TSMC.md)
  is the detailed AnalogGym campaign report.

## Model levels

| LEVEL | Model | Use |
| --- | --- | --- |
| 72 | BSIM-CMG through PyCMG/OSDI | Reference compact model |
| 73 | DirectNet | Production NN fast path |
| 74 | BSIM-AR Transformer | Higher-fidelity, slower NN |

Supported analyses are `.op`, `.dc`, `.ac`, and `.tran`. Devices include
resistors, capacitors, independent voltage/current sources, PULSE sources,
LEVEL=72–74 MOSFETs, and flattened `X` subcircuit instances.

## 0. Set up the environment

### Requirements

- Python 3.10 in a conda environment named `pycircuitsim`
- NGSPICE 45.2 or newer with OSDI support
- OpenVAF at `/usr/local/bin/openvaf`
- PyTorch for LEVEL=73–74
- A built BSIM-CMG OSDI binary at
  `external_compact_models/bsim_cmg/build/osdi/bsimcmg.osdi`

The configured NGSPICE binary is `/usr/local/ngspice-45.2/bin/ngspice`. No
NGSPICE executable is bundled in this checkout; set `NGSPICE_BIN` to another
existing OSDI-capable binary when necessary.

TSMC runs also require the private raw modelcards below. They are intentionally
untracked; only the ASAP7 cards are bundled:

```text
PDKs/TSMC5/cln5_1d2_sp_v1d2_2p2.l
PDKs/TSMC6/cln6_1d8_sp_v1d0_2p2.l
PDKs/TSMC7/cln7_1d8_sp_v1d2_2p2.l
PDKs/TSMC12/cln12ffcll_1d8_sp_v1d0_2p4.l
PDKs/TSMC16/crn16ffcll_1d8_sp_v1d0_2p1.l
```

### Clone and install

```bash
http_proxy=http://127.0.0.1:2080 \
https_proxy=https://127.0.0.1:2080 \
git clone https://github.com/ShenShan123/PyCircuitSim.git
cd PyCircuitSim

conda create -n pycircuitsim --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
  python=3.10
conda activate pycircuitsim

pip install -r requirements.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Build the OSDI model if it is absent:

```bash
cd external_compact_models/bsim_cmg
mkdir -p build
cd build
cmake ..
cmake --build . --target osdi
cd ../../..
```

Verify the reference path before generating data or running gates:

```bash
test -f external_compact_models/bsim_cmg/build/osdi/bsimcmg.osdi
conda run -n pycircuitsim python tests/single_devices/verify_bsimcmg_op.py
```

A fresh checkout does not include those private cards, generated datasets, or
NN checkpoints. Supply the cards, then build the artifacts in stages 1 and 2.

### Command interface conventions

Python entry points use `argparse`. Put their options after the script/module
name and inspect the exact interface with `--help`:

```bash
conda run -n pycircuitsim python main.py --help
conda run -n pycircuitsim python -m neural_network.cli.train --help
conda run -n pycircuitsim python scripts/v710_regate_jobs.py --help
conda run -n pycircuitsim python scripts/v730_coverage.py --help
conda run -n pycircuitsim python scripts/v730_docs_build.py --help
```

The training and re-gate shell wrappers use environment variables for their
campaign matrix and concurrency. They accept only the positional arguments
shown by their own help; Python CLI flags belong in `EXTRA_ARGS` when the
training wrapper explicitly forwards them:

```bash
bash scripts/recipe_train.sh --help
bash scripts/v710_regate.sh --help
```

`v710_regate.sh` requires `NN_PY` to name an executable Python environment
with NumPy and PyTorch. It fails before dispatch if that interpreter is absent
or incomplete; it never falls back to a different Python. A shell-independent
way to resolve the project environment is:

```bash
NN_PY="$(conda run -n pycircuitsim which python)"
```

## 1. Generate datasets

The generator evaluates PyCMG/BSIM-CMG over bias, geometry, temperature, and
technology bins. Production datasets require both overlay flags:
`--enable-inv-trip` and `--enable-subvt-off`. Omitting the latter silently
produces a smaller, non-production dataset.

Generate both polarities for one technology:

```bash
conda run -n pycircuitsim python \
  external_compact_models/bsim_cmg/scripts/generate_nn_data.py \
  --device both \
  --tech tsmc5 \
  --enable-inv-trip \
  --enable-subvt-off \
  --n-workers 8
```

Use `--max-l-ratio 1.35` when reproducing the geometry-densified V7.4.2
dataset. Inspect all generator choices with:

```bash
conda run -n pycircuitsim python \
  external_compact_models/bsim_cmg/scripts/generate_nn_data.py --help
```

Datasets are written under
`external_compact_models/neural_network/data/datasets/`. Each NPZ contains
source-relative terminal inputs, geometry/technology features, 13 BSIM-CMG
targets, and a sample-class label. Canonical generation aborts on any rejected
point or dropped bin and writes a checksum-bound `.npz.complete` marker.
`--allow-rejected-points` is diagnostic only; the training CLI rejects those
artifacts, missing/stale markers, dirty-source provenance, and incomplete row
counts. NFIN=1 is not part of the training domain.

For the five-technology production sweep, use the parallel driver:

```bash
bash scripts/benchmark_gen_data.sh
```

This creates NMOS and PMOS data for TSMC5/6/7/12/16. TSMC6 deliberately uses
the TSMC7 BSIM-CMG source data as a controlled repeat; its NN training run is
independent.

## 2. Train an NN compact model

The unified trainer supports:

- `--model direct` for LEVEL=73 DirectNet;
- `--model transformer` for LEVEL=74 BSIM-AR;
- `--size small|medium|large|xl`;
- `--tech-scope tsmc5|tsmc6|tsmc7|tsmc12|tsmc16|universal`.

Train a clean DirectNet checkpoint for one polarity:

```bash
PYTHONPATH=external_compact_models \
conda run -n pycircuitsim python -m neural_network.cli.train \
  --model direct \
  --size large \
  --device-type nmos \
  --tech-scope tsmc5 \
  --apply-filter off \
  --swa-mode ema \
  --seed 42 \
  --cuda
```

Repeat with `--device-type pmos`. Checkpoints are written to
`external_compact_models/neural_network/checkpoints/` using stems such as
`tsmc5_dn_large_nmos`. The CLI writes the model and normalization sidecar;
Transformer also requires a configuration sidecar.

For parallel DirectNet production training, let the campaign wrapper assign
jobs across GPUs:

```bash
GPUS="0 1 2" NSTREAMS=9 \
TECHS="tsmc5 tsmc6 tsmc7 tsmc12 tsmc16" \
SIZES="large" \
bash scripts/benchmark_train_sml.sh
```

For a clean matrix, use the family-aware wrapper and keep its outputs isolated
from preserved controls:

```bash
BSIMAR_CHECKPOINT_DIR="$PWD/results/v7516_clean/checkpoints" \
RECIPE_TRAIN_LOG_DIR="$PWD/results/v7516_clean/training/tf" \
MODEL=transformer RECIPES=clean SIZES="small medium large xl" \
TECHS="tsmc5 tsmc6 tsmc7 tsmc12 tsmc16" \
GPUS="0 1 2" NSTREAMS=6 \
bash scripts/recipe_train.sh
```

Set `MODEL=direct` or `transformer` for the two supported families.

The wrapper creates a
`*_best.pt.complete` marker only after a successful job. Treat that marker as
the campaign completion record: a bare `*_best.pt` can be best-so-far output
from an interrupted run.

### Select checkpoints at inference

The parser first honors explicit environment pins, then resolves a per-tech
checkpoint. Examples:

```bash
export PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS=tsmc5_dn_large_nmos
export PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS=tsmc5_dn_large_pmos
```

Pins are checkpoint stems inside the package checkpoint directory, without
`_best.pt`; they are not arbitrary file paths. Use `TF` instead of `DN` for
LEVEL=74. A missing pinned stem is an error.

The device and inverter gates below have dedicated DirectNet and BSIM-AR
passes, so family-specific pins are sufficient. Other gates render LEVEL=73
decks. Retarget one of those gates to a trained Transformer checkpoint
like this:

```bash
export PYCIRCUITSIM_NN_FORCE_LEVEL=74
export PYCIRCUITSIM_NN_CHECKPOINT_TF_NMOS=tsmc5_tf_small_nmos
export PYCIRCUITSIM_NN_CHECKPOINT_TF_PMOS=tsmc5_tf_small_pmos

# Run one TSMC5 LEVEL=73-based gate as LEVEL=74.
conda run -n pycircuitsim python \
  tests/single_devices/verify_nn_multi_tech_dc.py --tech TSMC5
```

Unset `PYCIRCUITSIM_NN_FORCE_LEVEL` before returning to the default DirectNet
gates.

## 3. Verify `single_devices` against ground truth

All accuracy gates compare PyCircuitSim with NGSPICE running the same
BSIM-CMG OSDI model. Pin CPU thread counts for scored runs so OpenMP variance
does not change the verdict:

```bash
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYCIRCUITSIM_TORCH_THREADS=1
export NGSPICE_BIN="${NGSPICE_BIN:-/usr/local/ngspice-45.2/bin/ngspice}"
test -x "$NGSPICE_BIN"
```

First validate LEVEL=72 and the generated geometry domain:

```bash
conda run -n pycircuitsim python tests/single_devices/verify_bsimcmg_op.py
conda run -n pycircuitsim python \
  tests/single_devices/verify_bsimcmg_dc_comprehensive.py
conda run -n pycircuitsim python \
  tests/single_devices/verify_data_geometry_coverage.py
```

Then run the mixed-family device gate, the DirectNet source-shift canary, and
the LEVEL=73 multi-technology sweep:

```bash
conda run -n pycircuitsim python \
  tests/single_devices/verify_nn_dc.py --tech TSMC5
conda run -n pycircuitsim python \
  tests/single_devices/verify_nn_lifted_source_dc.py
conda run -n pycircuitsim python \
  tests/single_devices/verify_nn_multi_tech_dc.py
```

Run NMOS and PMOS checks before moving to circuits. A device checkpoint that
passes pointwise regression but fails derivative-sensitive DC gates is not
ready for circuit validation.

## 4. Verify `simple_circuits` against ground truth

Start with the mixed DirectNet/BSIM-AR inverter gate. Follow it with the
LEVEL=73 multi-technology transient and AC gates:

```bash
conda run -n pycircuitsim python \
  tests/simple_circuits/verify_nn_inverter.py
conda run -n pycircuitsim python \
  tests/simple_circuits/verify_nn_multi_tech_tran.py
conda run -n pycircuitsim python \
  tests/simple_circuits/verify_nn_ac.py
```

Verify parser and LEVEL=72 circuit behavior independently:

```bash
conda run -n pycircuitsim python tests/simple_circuits/verify_subckt.py
conda run -n pycircuitsim python tests/simple_circuits/verify_ac.py
```

Finish the default LEVEL=73 score matrix with the four multi-device cells:

```bash
conda run -n pycircuitsim python \
  tests/simple_circuits/verify_complex_ring_osc.py
conda run -n pycircuitsim python \
  tests/simple_circuits/verify_complex_opamp.py
conda run -n pycircuitsim python \
  tests/simple_circuits/verify_complex_sram_snm.py
conda run -n pycircuitsim python \
  tests/simple_circuits/verify_complex_switchcap.py
```

### Run the complete clean checkpoint matrix

Generate the full family pool, then select the DirectNet/BSIM-AR S/M/L/XL
matrix used by V7.5.17. It is exactly **480 jobs**: two families × four tiers ×
five technologies, with the required OMP repeats. Require complete coverage
for both current families before rebuilding the generated reports:

```bash
conda run -n pycircuitsim python \
  scripts/v710_regate_jobs.py /tmp/pycircuitsim-v7517-jobs
awk '$1 == "dn" || $1 == "tf"' \
  /tmp/pycircuitsim-v7517-jobs/jobs_clean.txt \
  > /tmp/pycircuitsim-v7517-jobs/jobs_clean_dn_tf.txt

BSIMAR_CHECKPOINT_DIR="$PWD/results/v7516_clean/checkpoints" \
V710_OUT="$PWD/results/v7517_clean" \
V710_SCRATCH=/tmp/pycircuitsim-v7517-clean \
NGSPICE_BIN="${NGSPICE_BIN:-$PWD/tools/ngspice-45.2/bin/ngspice}" \
JOBS=/tmp/pycircuitsim-v7517-jobs/jobs_clean_dn_tf.txt PAR=32 \
NN_PY="$(conda run -n pycircuitsim which python)" \
bash scripts/v710_regate.sh

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

These gates own NN circuit accuracy. Their definitions and thresholds are in
[`docs/accuracy/methodology.md`](docs/accuracy/methodology.md); generated
family reports are indexed by [`docs/accuracy/README.md`](docs/accuracy/README.md).

## 5. Verify `complex_circuits` with AnalogGym

The AnalogGym migration expands validation to a large, translated circuit
corpus under `examples/complex_circuits/`. NGSPICE always runs the identical
BSIM-CMG OSDI model as ground truth. PyCircuitSim defaults to LEVEL=72 and can
instead run a completed, explicitly pinned DirectNet LEVEL=73 checkpoint pair.
Each result records the checkpoint and normalization hashes; resume refuses to
mix rows from different weights.

The per-design BSIM-CMG libraries are ignored private artifacts derived from
the local TSMC cards. Materialize them after a fresh checkout without rewriting
the tracked AnalogGym decks:

```bash
for tech in tsmc5 tsmc6 tsmc7 tsmc12 tsmc16; do
  conda run -n pycircuitsim python -m \
    examples.complex_circuits.tools.materialize_modelcards \
    --tree "examples/complex_circuits/designs_${tech}"
done
```

Run one deck while developing a translation or solver fix:

```bash
PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS=tsmc5_dn_large_nmos \
PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS=tsmc5_dn_large_pmos \
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYCIRCUITSIM_TORCH_THREADS=1 \
conda run -n pycircuitsim python -m \
  examples.complex_circuits.pycircuitsim_bench.run_compare \
  --tech tsmc5 \
  --category amplifier \
  --design Fan_SMC_Pin_3 \
  --deck tb_gain.cir \
  --model-level 73 \
  --out results/analoggym-one
```

Run and refine a complete DirectNet technology campaign. The driver requires
both `*_best.pt.complete` markers, pins inference to CPU with one OpenMP, MKL,
and Torch thread, and keeps NGSPICE on LEVEL=72:

```bash
conda run -n pycircuitsim python -m \
  examples.complex_circuits.pycircuitsim_bench.campaign \
  --tech tsmc5 \
  --model-level 73 \
  --checkpoint-size large \
  --families ac,dc_source,dc_temp,tran \
  --jobs 12 \
  --refine \
  --out results/analoggym-directnet-large-tsmc5
```

Repeat for `tsmc6`, `tsmc7`, `tsmc12`, and `tsmc16`. TSMC6 and TSMC7 share
LEVEL=72 ground truth but use independently trained DirectNet checkpoints, so
the NN campaign keeps both as its controlled training-repeat axis. For a pure
LEVEL=72 campaign, omit `--model-level` and `--checkpoint-size`; TSMC6 is then
normally reported as the exact TSMC7 repeat instead of rerun. Read the current
results and quarantined-deck list in
[`examples/complex_circuits/RESULTS_TSMC.md`](examples/complex_circuits/RESULTS_TSMC.md).

## Run a netlist directly

Use the analysis directive in a deck to select the solver:

```bash
conda run -n pycircuitsim python main.py \
  examples/simple_circuits/bsimcmg_inverter_op.sp
conda run -n pycircuitsim python main.py path/to/deck.sp \
  --output results/my-run
```

A MOS model declaration selects its implementation:

```spice
.model nmos_ref NMOS (LEVEL=72 TECH=tsmc5 VT=lvt)
.model nmos_nn  NMOS (LEVEL=73 TECH=tsmc5 VT=lvt)
M1 out in 0 0 nmos_nn L=16n NFIN=10
.dc Vin 0 0.8 0.01
.end
```

LEVEL=74 uses the same netlist shape. NN technologies require
`TECH` and `VT`; ASAP7 has no NN checkpoints. Parser-supported suffixes include
`f`, `p`, `n`, `u`, `m`, `k`, `meg`, and `g`.

Simulation output is organized as:

```text
results/<circuit>/<analysis>/
├── <circuit>_simulation.lis
├── <analysis-data>.csv
└── <analysis-plot>.png
```

Use `.sp` files as PyCircuitSim inputs. The paired `.cir` files in `examples/`
are NGSPICE reference templates rendered by the verification infrastructure;
tests do not carry private copies of those netlists.

## Performance controls

CPU, flags-off execution is the scored compatibility contract. GPU and
floating-point-order-changing paths are opt-in:

| Variable | Purpose |
| --- | --- |
| `PYCIRCUITSIM_NN_DEVICE=cuda` | Run NN inference on CUDA |
| `PYCIRCUITSIM_NN_FUSED_JAC=1` | Fuse NN Jacobian work |
| `PYCIRCUITSIM_NN_AR_CACHE=1` | Cache autoregressive inference state |
| `PYCIRCUITSIM_TORCH_THREADS=N` | Set PyTorch CPU threads |

Keep perturbing flags off when reproducing accuracy reports. Contributor rules
for promoting performance paths are in [`AGENTS.md`](AGENTS.md).

## Repository layout

```text
pycircuitsim/                         parser, solver, simulation, models
PDKs/                                technology modelcards (TSMC files private)
external_compact_models/bsim_cmg/    BSIM-CMG evaluator and data generator
external_compact_models/neural_network/  shared NN data/model/training package
examples/single_devices/             device decks and NGSPICE templates
examples/simple_circuits/             compact-model circuit decks
examples/complex_circuits/            AnalogGym corpus and campaign harness
tests/single_devices/                 device-level gates
tests/simple_circuits/                circuit-level gates
tests/common/                         shared render/compare infrastructure
docs/accuracy/                        generated evidence and methodology
```

Raw TSMC PDK modelcards live under `PDKs/TSMC*/` and are intentionally
untracked. Generated single-device cards stay under
`external_compact_models/bsim_cmg/build/modelcards/`. Do not publish `cln*.l`
files or `modelcards.tar.gz`.
