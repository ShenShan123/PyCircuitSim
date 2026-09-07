# PyCircuitSim

PyCircuitSim is a pure-Python, SPICE-like circuit simulator for BSIM-CMG and
neural compact models. NGSPICE running the identical BSIM-CMG OSDI model is
ground truth for every accuracy claim.

Current release: **V7.7.5**.

The NN runtime is full-terminal-only. DirectNet-Full (LEVEL=75) is the default;
BSIM-AR-Full (LEVEL=76) is the autoregressive alternative. The old reduced
LEVEL=73/74 families are retired and rejected. Current measurements and known
limitations are indexed in [`docs/accuracy/`](docs/accuracy/).

## Documentation map

- This file owns setup, commands, netlist use, and the five-stage workflow.
- [`AGENTS.md`](AGENTS.md) owns implementation and debugging contracts.
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) owns outcomes and dead ends.
- [`docs/accuracy/README.md`](docs/accuracy/README.md) indexes measurements;
  [`docs/accuracy/methodology.md`](docs/accuracy/methodology.md) defines gates.
- [`circuit_templates/README.md`](circuit_templates/README.md) owns templates;
  [`tests/README.md`](tests/README.md) owns test organization.

## Model levels

| LEVEL | Model | Role |
|---:|---|---|
| 72 | BSIM-CMG through PyCMG/OSDI | Reference adapter |
| 75 | DirectNet-Full | Default six-surface NN |
| 76 | BSIM-AR-Full | Six-surface autoregressive NN |

Supported analyses are `.op`, `.dc`, `.ac`, and `.tran`. Components include
resistors, capacitors, inductors in DC/AC, independent sources, PULSE sources,
LEVEL=72/75/76 MOSFETs, and flattened subcircuits.

## 0. Set up the environment

Requirements: a conda environment named `pycircuitsim`, Python 3.10+, PyTorch,
NGSPICE 45.2+ with OSDI, OpenVAF, and a built
`external_compact_models/bsim_cmg/build/osdi/bsimcmg.osdi`.

```bash
http_proxy=http://127.0.0.1:2080 \
https_proxy=https://127.0.0.1:2080 \
git clone https://github.com/ShenShan123/PyCircuitSim.git
cd PyCircuitSim

conda create -n pycircuitsim --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
  python=3.10
conda activate pycircuitsim

http_proxy=http://127.0.0.1:2080 \
https_proxy=https://127.0.0.1:2080 \
pip install -r requirements.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

http_proxy=http://127.0.0.1:2080 \
https_proxy=https://127.0.0.1:2080 \
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Build and verify the reference model:

```bash
cd external_compact_models/bsim_cmg
mkdir -p build
cd build
cmake ..
cmake --build . --target osdi
cd ../../..

conda run -n pycircuitsim python tests/single_devices/verify_bsimcmg_op.py
conda run -n pycircuitsim python -m pytest -q tests
```

TSMC work also needs the private cards under `PDKs/TSMC*/`; they are untracked.
ASAP7 cards are bundled, but ASAP7 has no NN checkpoints.

Inspect current interfaces with `--help`:

```bash
conda run -n pycircuitsim python main.py --help
conda run -n pycircuitsim python main.py train --help
PYTHONPATH=external_compact_models conda run -n pycircuitsim python -m neural_network.cli.train --help
conda run -n pycircuitsim python scripts/v710_regate_jobs.py --help
bash scripts/recipe_train.sh --help
bash scripts/v710_regate.sh --help
```

## NN workflow from the project root

Use [`main.py`](main.py) for netlist simulation and the entire NN workflow.
The `data`, `train`, and `evaluate` commands each run one independent stage;
`flow` runs those same stages in order, stopping if a stage fails. The
`simulate` command runs a netlist. The numbered sections below describe the
underlying stages and advanced interfaces.

Activate the environment above. Preview a run, then execute it:

```bash
conda activate pycircuitsim
python main.py flow --run-dir results/my_first_nn --dry-run
python main.py flow --run-dir results/my_first_nn --gpus 0
```

Defaults are TSMC5, both NMOS and PMOS, DirectNet-Full (LEVEL=75), size
`small`, and the `clean` and `canary` evaluation pools. Without `--gpus`,
training uses CPU. Even the small model uses a full canonical dataset;
generation, training, and circuit gates can take substantial time.

Canonical execution requires a **clean, committed Git worktree**, the private
cards for the selected technologies, and the built OSDI model. Commit the
workflow changes before generating canonical data: the existing trainer
rejects datasets marked as coming from dirty source. `--dry-run` needs only
Python, works before dependencies or artifacts are installed, and writes nothing.

Run stages separately with the same directory:

```bash
python main.py data --run-dir results/my_first_nn --workers 8
python main.py train --run-dir results/my_first_nn --gpus 0
python main.py evaluate --run-dir results/my_first_nn --parallel 4
```

Choose either the `flow` command or the separate sequence for a fresh run.
Keep the same `--tech`, `--model`, and `--size` selections across applicable
stages. `all` is an alias for `flow`. Selections are space-separated; unknown
or duplicate entries fail. Add `--dry-run` to any NN command to preview just
that stage or the complete flow.

| Option | Applies to | Purpose |
|---|---|---|
| `--run-dir results/NAME` | all stages | Run output root; default `results/v773_nn` |
| `--tech tsmc5 tsmc7` | all stages | TSMC5/6/7/12/16; standalone data also accepts `asap7`/`all`, standalone train accepts `universal` |
| `--data-dir PATH` | all stages | Explicit dataset root; default `<run-dir>/data` |
| `--model direct transformer` | train, evaluate, flow | LEVEL=75 and LEVEL=76 families |
| `--size small medium large xl` | train, evaluate, flow | Existing training size presets |
| `--recipe clean ar3roll` | train, evaluate, flow | Named strategies; use the same selection for training and evaluation |
| `--checkpoint-dir PATH` | train, evaluate, flow | Explicit bundle root; default `<run-dir>/checkpoints` |
| `--device nmos` | data, train | Work on one polarity; `flow` and evaluation require both |
| `--workers 8` | data, flow | Generator processes per dataset |
| `--gpus 0 1 2` | train, flow | Physical GPU IDs, resolved to UUIDs; one job per selected GPU |
| `--epochs 80 --batch-size 2048 --seed 42` | train, flow | Optional training overrides |
| `--pools clean canary simple_v2` | evaluate, flow | Add the separate held-out topology diagnostic pool |
| `--device-suite nn_multi_tech_dc` | evaluate, flow | Select named single-device suites |
| `--circuit ring_osc current_mirror` | evaluate, flow | Select named circuit cases; required pools are chosen automatically |
| `--evaluation-group single_devices` | evaluate, flow | Select all cases in a test group; also accepts `simple_circuits` |
| `--list-cases` | evaluate, flow | List case names, groups, pools, and evidence roles without running |
| `--parallel 4` | evaluate, flow | Concurrent gate jobs; each runs on CPU with one thread |
| `--ngspice /path/to/ngspice` | evaluate, flow | OSDI-capable reference executable; also reads `NGSPICE_BIN` |
| `--dry-run` | all stages | Print commands, paths, and per-job environment changes |

For both families across the full technology and size matrix:

```bash
python main.py flow --run-dir results/v773_matrix \
  --tech tsmc5 tsmc6 tsmc7 tsmc12 tsmc16 \
  --model direct transformer --size small medium large xl \
  --gpus 0 1 2 --workers 8 --parallel 8 \
  --pools clean canary simple_v2 --dry-run
```

Remove `--dry-run` to execute after choosing available GPUs. Both architectures
share the generated `*_dnf_{nmos,pmos}.npz` datasets. Training retains the
existing EMA/seed defaults and the six-target, teacher-forced Transformer
recipe when `--recipe clean` is selected.

### Configure each stage with arguments

The root CLI and backend CLIs share their argument definitions in
[`external_compact_models/cli_options.py`](external_compact_models/cli_options.py).
All generator and trainer settings are exposed. On `data` and `train`, use
the backend's ordinary option names. The `--data-*` and `--train-*` forms
also work and keep settings separate in a combined `flow`:

```bash
python main.py data --help
python main.py train --help
python main.py evaluate --help

python main.py flow --run-dir results/custom_nn --tech tsmc5 \
  --data-sampler grid --data-grid-per-axis 30 --data-vbs-levels 5 \
  --data-temperatures 248.15,300.15,398.15 --data-seed 17 \
  --model transformer --size small --recipe ar3roll --gpus 0 \
  --train-subthresh --train-lam-subthresh 0.1 --train-swa-mode ema \
  --train-class-weights subvt_off=3.0 --train-seed 42 \
  --device-suite nn_multi_tech_dc terminal_integrity \
  --circuit ring_osc current_mirror --dry-run
```

Shared identity/path options keep their short names: `--tech`, `--device`,
`--data-dir`, `--checkpoint-dir`, `--model`, and `--size`. `--tech-scope`,
`--device-type`, and `--n-workers` are accepted aliases; generation workers
use `--workers`. `--seed` on `flow` controls training; `--data-seed` controls
generation. Explicit arguments override the selected recipe.

Generator settings below use their standalone `data` names; prefix them
with `data-` on `flow` (for example, `--data-voltage-box-factor`):

| Configuration | Arguments |
|---|---|
| Variants and technology aggregation | `--variants`, `--universal`, `--exclude-techs` |
| Temperature and repeatability | `--temperatures` (comma-separated Kelvin), `--seed` |
| Sampler and voltage coverage | `--sampler grid\|lhs`, `--n-lhs-samples`, `--voltage-box-factor` |
| Grid and densification | `--grid-per-axis`, `--vbs-levels`, `--hot-per-axis`, `--jitter-sigma-frac` |
| Targeted overlays | `--enable-inv-trip`, `--enable-subvt-off` |
| Experimental overlays (disabled by default) | `--overshoot-per-axis`, `--n-vbs-lhs` (nonnegative counts; 0 disables) |
| Progress output | `--quiet` suppresses per-bin progress, retaining summaries and errors |
| Evaluation tolerance and geometry sampling | `--dc-solve-tol`, `--max-l-ratio` |
| Output variants | `--version`, `--finetune-size` |
| Rejected samples | `--allow-safety-rejections`, `--allow-rejected-points` |

The root defaults enable both targeted overlays and declared safety
rejections. Use `--no-data-enable-inv-trip`, `--no-data-enable-subvt-off`, or
`--no-data-allow-safety-rejections` to disable them. Boolean backend settings
have corresponding `--no-data-*`/`--no-train-*` overrides.

```bash
python main.py data --run-dir results/lhs_data --device nmos --tech tsmc5 \
  --variants lvt --sampler lhs --n-lhs-samples 5000 \
  --temperatures 300.15 --voltage-box-factor 2 --max-l-ratio 1.35 \
  --version trial --finetune-size 8000 --workers 8 --dry-run

python main.py data --run-dir results/universal_data --universal --dry-run
```

Standalone universal generation defaults to all five TSMC technologies;
an explicit `--tech` selection controls its included technologies. ASAP7 can
be generated separately and remains excluded from universal NN training.
Diagnostic generation with `--allow-rejected-points` writes artifacts but
does not label them as training-ready.

Training settings use the standalone `train` names below, or the same names
prefixed with `train-` on `flow`:

| Configuration | Arguments |
|---|---|
| Explicit input and split | `--data`, `--max-rows`, `--split-mode combo\|random`, `--exclude-techs` |
| Optimizer and stopping | `--epochs`, `--batch-size`, `--lr`, `--patience` |
| Averaging and initialization | `--swa-mode none\|ema\|swa`, `--ema-decay`, `--init-from` |
| Precision and repeatability | `--amp`, `--cuda` (physical GPU 0), `--seed`; use `--gpus` for an explicit GPU list |
| Technology embeddings | `--num-tech-codes`, `--p-unknown` |
| Class weighting and training-only overlays | `--class-weights`, `--training-overlay-classes` |
| Autoregressive strategy | `--full-terminal-ar-targets 3\|6`, `--autoregressive-training` |
| Subthreshold auxiliary loss | `--subthresh`, `--lam-subthresh`, `--subthresh-s2`, `--subthresh-upper`, `--subthresh-floor`, `--subthresh-off-floor`, `--subthresh-ceiling-k`, `--subthresh-ceiling-w` |
| Output naming and replacement | `--exp-name`, `--overwrite` |

The existing base loss is class-weighted BNI-MAE; BSIM-AR additionally supports
the subthreshold auxiliary term. These flags select the implemented losses
and strategies. They do not introduce a new numerical loss implementation.

| Recipe | Strategy |
|---|---|
| `clean` | Existing baseline; seed 42 and EMA |
| `s7`, `s17`, `s123` | Alternative seed |
| `sub` | BSIM-AR with the subthreshold auxiliary loss |
| `ar3` | BSIM-AR with three autoregressive charge targets |
| `ar3roll` | Three-target BSIM-AR with autoregressive training rollout |
| `corridor` | Prepared trajectory-overlay data and class weighting; standalone training |

```bash
python main.py train --run-dir results/strategy_compare \
  --data-dir results/my_first_nn/data --model transformer \
  --recipe clean sub ar3 ar3roll --size small medium --gpus 0 1 \
  --lr 0.0008 --patience 30 --swa-mode swa --dry-run
```

Recipe bundles use `{tech}_{tag}_{recipe}_{size}_{device}`; `clean` keeps the
original `{tech}_{tag}_{size}_{device}` stem. Match `--recipe` when evaluating.
Keep differently configured runs in distinct directories.

Standalone training accepts explicit versioned datasets, universal scope,
custom experiment names, and prepared corridor datasets. `--data` requires
one technology and one polarity; `--exp-name` requires one technology,
model, size, and recipe. The combined `flow` uses the campaign's canonical
per-technology names and rejects universal/versioned/diagnostic datasets,
custom input/output names, and corridor preparation before starting. Such
artifacts retain their standalone use; they do not bypass campaign contracts.

### Select device suites and named circuits

```bash
python main.py evaluate --list-cases
python main.py evaluate --list-cases --evaluation-group single_devices

python main.py evaluate --run-dir results/named_cases \
  --data-dir results/my_first_nn/data \
  --checkpoint-dir results/strategy_compare/checkpoints \
  --tech tsmc5 --model transformer --size small --recipe ar3roll \
  --device-suite nn_multi_tech_dc terminal_integrity \
  --circuit ring_osc current_mirror inverter_chain --parallel 4 --dry-run
```

The list is derived from the campaign suites and circuit catalog, including
single-device integrity/DC/source-frame suites and named simple circuits.
Without explicit case/group selection, evaluation retains `clean` and
`canary`. Named selections choose their owning pools automatically; conflicting
explicit pools/groups, unknown names, and duplicates are errors. Both NMOS
and PMOS checkpoint bundles remain required. Selected catalog circuits receive
nominal geometry checks; device/benchmark suites retain the parametric geometry
guard. Legacy release runners keep their original complete geometry inventory.

NN workflow paths are resolved relative to the repository, including when invoking
`/absolute/path/to/main.py` from another directory. Every child uses the invoking
Python interpreter; no implicit conda switch occurs. The workflow clears
inherited checkpoint pins and numerical experiment flags for its children.

```text
results/my_first_nn/
├── data/                  canonical .npz files, completion markers, labels
├── checkpoints/           weights, normalization/configuration, completion markers
├── logs/                  exact commands and separate logs for each attempt
└── evaluation/
    ├── job_lists/         selected jobs from the authoritative campaign catalog
    ├── clean/             REPORT.md, data.json, provenance, gate logs, artifacts
    └── canary/            separate source-frame contract evidence
```

The geometry guard runs before evaluation against the selected technologies
and dataset root. Evaluation uses only the selected catalog cells at OMP=1;
it is a scoped result, not the full release campaign with OMP=2/4 stability
probes. `simple_v2` remains diagnostic and does not enlarge qualification
totals. Gate definitions and metric interpretation belong to
[`docs/accuracy/methodology.md`](docs/accuracy/methodology.md).

Exit codes: **0** means the requested stages completed with no failed selected
gates; **1** means complete evaluation contains scientific FAIL/ERROR verdicts;
**2** means invalid options, missing prerequisites, a failed child, or incomplete
evidence; **130** means interruption. Read each pool's `REPORT.md` and `data.json`
for convergence and numerical errors separately.

Data generation and training refuse to replace existing artifacts, including
partial bundles, by default. Training supports explicit `--overwrite`; use it
only in the intended experiment directory. For a retry, use a new run/checkpoint directory or train only
the missing polarity. Reuse data with `--data-dir results/old_run/data`.
Evaluation can resume its existing logs when source, selection, and bundles
still match the immutable manifest. Use a new `--run-dir` when changing the
selection, and pass explicit `--data-dir` and `--checkpoint-dir` to evaluate
existing artifacts. An older dataset source requires
`--dataset-source-commit FULL_SHA` and must pass the existing exact source
inventory comparison. This command does not promote models or overwrite the
published accuracy reports. The release-specific V7.7.2 supervisor below
continues to own its ongoing campaign.

## 1. Generate full-terminal datasets

The generator always stores six independent surfaces:
`i_d,i_g,i_b,qd,qg,qb`. Source current and charge are reconstructed by
closure. Canonical dataset names retain the architecture-neutral `dnf` tag.

```bash
conda run -n pycircuitsim python \
  external_compact_models/bsim_cmg/scripts/generate_nn_data.py \
  --device both --tech tsmc5 \
  --enable-inv-trip --enable-subvt-off \
  --allow-safety-rejections --n-workers 8

BSIMAR_DATA_DIR="$PWD/results/v770_full_data" \
BENCHMARK_GEN_LOG_DIR="$PWD/results/v770_full_generation" \
bash scripts/benchmark_gen_data.sh 20
```

A canonical dataset has a checksum-bound `.npz.complete` marker and label
sidecars. Training rejects missing, stale, dirty-source, diagnostic, or
incomplete artifacts. NFIN=1 is outside the training domain.

The V7.7.2 regeneration/retraining campaign is scheduled in
[`docs/plans/2026-09-05-v772-full-retraining.md`](docs/plans/2026-09-05-v772-full-retraining.md).
V7.7.1 and V7.7.2 now share one training queue, released as V7.7.2. From the
clean V7.7.2 worktree, start or resume its training-to-evaluation supervisor:

```bash
conda run --no-capture-output -n pycircuitsim python -u \
  scripts/v772_consolidate.py \
  --training-root /data2/home/shenshan/PyCircuitSim-v771 \
  --training-source 6be83348c1f5db6720d7504ed6dcea874a3a7418 --gate-parallel 16

cat results/v772_campaign/state.json results/v772_campaign/training_progress.json
```

The supervisor writes state under `results/v772_campaign/`. The frozen
training worktree retains its regenerated data in `results/v771_r2_data/`
and bundles in `results/v771_r2_checkpoints/`; these are the consolidated
campaign's active artifacts. Duplicate generation remains an abandoned
attempt. Running training jobs continue without restart.

After all 80 training jobs exit successfully, the supervisor validates their
bundles and starts the 600 clean and 1,200 simple-v2 evaluation jobs. Scoring
uses the V7.7.2 harness and records both source commits. An explicit original
dataset source is accepted only when tracked model, generator, runtime,
template, PDK and environment inputs are identical (Markdown excluded).
Missing pins, mixed sources, or numerical differences fail closed. Keep both
worktrees clean while jobs run. After evaluation completes:

```bash
conda run -n pycircuitsim python scripts/v730_docs_build.py --campaign v772_full_clean
conda run -n pycircuitsim python scripts/v730_docs_build.py --campaign v772_full_clean --check
```

## 2. Train a full-terminal compact model

Train DirectNet-Full for one polarity:

```bash
PYTHONPATH=external_compact_models \
conda run -n pycircuitsim python -m neural_network.cli.train \
  --model direct --size large --device-type nmos \
  --tech-scope tsmc5 --swa-mode ema --seed 42 --cuda
```

Train BSIM-AR-Full:

```bash
PYTHONPATH=external_compact_models \
conda run -n pycircuitsim python -m neural_network.cli.train \
  --model transformer --size large --device-type nmos \
  --tech-scope tsmc5 --full-terminal-ar-targets 3 \
  --autoregressive-training --swa-mode ema --seed 42 --cuda
```

The default Transformer autoregresses all six surfaces. The three-target mode
autoregresses charges and emits currents through a parallel tail. Training is
teacher-forced unless `--autoregressive-training` is supplied; validation
always follows deployed rollout.

Run an isolated clean matrix:

```bash
BSIMAR_DATA_DIR="$PWD/results/v770_full_data" \
BSIMAR_CHECKPOINT_DIR="$PWD/results/v770_full_checkpoints" \
RECIPE_TRAIN_LOG_DIR="$PWD/results/v770_full_training/direct" \
MODEL=direct RECIPES=clean \
SIZES="small medium large xl" \
TECHS="tsmc5 tsmc6 tsmc7 tsmc12 tsmc16" \
GPUS="0 1 2" NSTREAMS=6 \
bash scripts/recipe_train.sh
```

Use `MODEL=transformer` for `tff`. DirectNet bundles require `_best.pt`,
`_norm.npz`, and `_best.pt.complete`; BSIM-AR also requires `_config.npz`.

Pin runtime checkpoints by stem:

```bash
export PYCIRCUITSIM_NN_CHECKPOINT_DNF_NMOS=tsmc5_dnf_large_nmos
export PYCIRCUITSIM_NN_CHECKPOINT_DNF_PMOS=tsmc5_dnf_large_pmos
```

Use `TFF` for LEVEL=76. Missing pins fail loudly. Without pins, the parser
uses the largest available per-technology bundle.

## 3. Verify devices against ground truth

Pin scored inference to CPU and one thread:

```bash
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYCIRCUITSIM_TORCH_THREADS=1
export NGSPICE_BIN="${NGSPICE_BIN:-/usr/local/ngspice-45.2/bin/ngspice}"
test -x "$NGSPICE_BIN"
```

```bash
conda run -n pycircuitsim python tests/single_devices/verify_bsimcmg_op.py
conda run -n pycircuitsim python tests/single_devices/verify_cmg_multiplier.py
conda run -n pycircuitsim python tests/single_devices/verify_nn_dc.py --tech TSMC5
conda run -n pycircuitsim python tests/single_devices/verify_nn_lifted_source_dc.py
conda run -n pycircuitsim python tests/single_devices/verify_nn_multi_tech_dc.py
conda run -n pycircuitsim python tests/single_devices/verify_device_integrity.py \
  --tech TSMC5 --suite output,subthreshold,linear,derivative
conda run -n pycircuitsim python tests/single_devices/verify_terminal_integrity.py \
  --tech TSMC5 --device nmos,pmos --corner nominal,temp_hot,nfin_high
```

## 4. Verify circuits against ground truth

```bash
conda run -n pycircuitsim python tests/simple_circuits/verify_nn_inverter.py
conda run -n pycircuitsim python tests/simple_circuits/verify_nn_multi_tech_tran.py
conda run -n pycircuitsim python tests/simple_circuits/verify_nn_ac.py
conda run -n pycircuitsim python tests/simple_circuits/verify_circuit_ring_osc.py
conda run -n pycircuitsim python tests/simple_circuits/verify_circuit_opamp.py
conda run -n pycircuitsim python tests/simple_circuits/verify_circuit_sram_snm.py
conda run -n pycircuitsim python tests/simple_circuits/verify_circuit_switchcap.py
conda run -n pycircuitsim python tests/simple_circuits/verify_circuit_opamp_ac.py
conda run -n pycircuitsim python tests/simple_circuits/verify_nn_subckt.py \
  --tech TSMC5 --analysis dc,tran,ac
```

Run the complete clean campaign:

```bash
conda run -n pycircuitsim python \
  scripts/v710_regate_jobs.py results/v770_full_clean/job_lists

BSIMAR_CHECKPOINT_DIR="$PWD/results/v770_full_checkpoints" \
V710_OUT="$PWD/results/v770_full_clean" \
V710_SCRATCH=/tmp/pycircuitsim-v770-full \
NGSPICE_BIN="${NGSPICE_BIN:-/usr/local/ngspice-45.2/bin/ngspice}" \
JOBS="$PWD/results/v770_full_clean/job_lists/jobs_clean.txt" PAR=32 \
NN_PY="$(conda run -n pycircuitsim which python)" \
bash scripts/v710_regate.sh

conda run -n pycircuitsim python scripts/v710_regate_collect.py \
  --root results/v770_full_clean --require-manifest

for family in dnf tff; do
  BSIMAR_CHECKPOINT_DIR="$PWD/results/v770_full_checkpoints" \
  conda run -n pycircuitsim python scripts/v730_coverage.py \
    --tag "$family" --set clean --passes v770-full-clean \
    --require-complete --fail-on-gaps
done

conda run -n pycircuitsim python scripts/v730_docs_build.py \
  --campaign v770_full_clean
conda run -n pycircuitsim python scripts/v730_docs_build.py \
  --campaign v770_full_clean --check
```

`v710_regate.sh` requires `NN_PY` to name an executable interpreter with NumPy
and PyTorch. It never falls back to another environment.

`--campaign` requires the selected campaign's complete metrics and matching
collection provenance before any report is written. Omitting it checks or
rebuilds the preserved report selection, currently V7.6.6.

## 5. Sweep unified circuit templates

```bash
conda run -n pycircuitsim python \
  tests/simple_circuits/verify_circuit_topologies.py --list

conda run -n pycircuitsim python \
  tests/simple_circuits/verify_circuit_topologies.py \
  --case current_mirror,inverter_chain \
  --tech TSMC5 --corner all --level72-control

conda run -n pycircuitsim python \
  tests/simple_circuits/verify_circuit_sweep.py opamp \
  --tech TSMC5 --dimension all
```

Every candidate and reference deck is rendered from one parameterized template
under `circuit_templates/`. Generated artifacts go under `results/`.

## Run a netlist directly

```spice
.model nmos_ref NMOS (LEVEL=72 TECH=tsmc5 VT=lvt)
.model nmos_nn  NMOS (LEVEL=75 TECH=tsmc5 VT=lvt)
.model nmos_ar  NMOS (LEVEL=76 TECH=tsmc5 VT=lvt)
M1 out in 0 0 nmos_nn L=16n NFIN=10
.dc Vin 0 0.8 0.01
.end
```

`FAMILY=directnet-full` or `FAMILY=bsimar-full` may be supplied as an
assertion, but LEVEL uniquely selects the family. NN declarations require
`TECH` and `VT`; LEVEL=73/74 fail explicitly.

```bash
conda run -n pycircuitsim python main.py simulate path/to/deck.sp \
  --output results/my-run
```

The original `python main.py path/to/deck.sp` invocation is also supported.
Netlist and simulation output paths remain relative to your current directory.

## Performance and artifact policy

CPU, flags-off inference is the scored contract. BSIM-AR's
`PYCIRCUITSIM_NN_AR_CACHE=1` remains opt-in because it changes float32
summation order. Training may use CUDA; scored inference stays CPU-only.

Datasets and checkpoints are ignored by Git. Keep comparison jobs in isolated
`BSIMAR_DATA_DIR` and `BSIMAR_CHECKPOINT_DIR` roots. Preserve only artifacts
behind a current score or active comparison, and put materialized simulations
under `results/`.

## Repository layout

| Path | Role |
|---|---|
| `main.py` | Unified CLI: simulate, data, train, evaluate, and flow |
| `pycircuitsim/` | Parser, circuit representation, solvers, runtime device models |
| `external_compact_models/bsim_cmg/` | OSDI evaluation and dataset generation |
| `external_compact_models/neural_network/` | NN data contracts, models, losses, training |
| `PDKs/` | Technology cards; only ASAP7 is tracked, private TSMC cards stay untracked |
| `circuit_templates/` | Single-source parameterized circuit topologies |
| `tests/` | Shared comparison infrastructure, verification gates, unit contracts |
| `scripts/` | Advanced training recipes and versioned campaign/report tooling |
| `docs/`, `results/` | Maintained documentation and generated run evidence |
