# PyCircuitSim

PyCircuitSim is a pure-Python, SPICE-like circuit simulator for BSIM-CMG and
neural compact models. NGSPICE running the identical BSIM-CMG OSDI model is
ground truth for every accuracy claim.

Current release: **V7.7.0**.

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
conda run -n pycircuitsim python -m neural_network.cli.train --help
conda run -n pycircuitsim python scripts/v710_regate_jobs.py --help
bash scripts/recipe_train.sh --help
bash scripts/v710_regate.sh --help
```

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
conda run -n pycircuitsim python main.py path/to/deck.sp \
  --output results/my-run
```

## Performance and artifact policy

CPU, flags-off inference is the scored contract. BSIM-AR's
`PYCIRCUITSIM_NN_AR_CACHE=1` remains opt-in because it changes float32
summation order. Training may use CUDA; scored inference stays CPU-only.

Datasets and checkpoints are ignored by Git. Keep comparison jobs in isolated
`BSIMAR_DATA_DIR` and `BSIMAR_CHECKPOINT_DIR` roots. Preserve only artifacts
behind a current score or active comparison, and put materialized simulations
under `results/`.
