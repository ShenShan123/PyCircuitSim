# PyCMG -- BSIM-CMG Training Data Generator for Neural Network Compact Models

PyCMG is a Python ctypes wrapper around the BSIM-CMG OSDI binary, purpose-built for generating training data for neural network compact models. It evaluates the full BSIM-CMG FinFET model (currents, charges, derivatives, capacitances) across systematic voltage, geometry, and temperature sweeps, producing ready-to-train CSV datasets verified against NGSPICE.

## What is PyCMG?

PyCMG loads a compiled BSIM-CMG `.osdi` binary via ctypes (no C++ compilation needed) and calls the model's evaluation functions directly. Given terminal voltages, geometry parameters, and temperature, it returns 17 model outputs (5 currents, 4 charges, 3 derivatives, 5 capacitances). The sweep engine drives this evaluator across PDK-defined geometry combinations (L, NFIN bin boundaries), temperatures, and voltage grids to produce million-row datasets for ML training.

## Quick Start

### 1. Build the OSDI Binary

```bash
mkdir -p build && cd build && cmake .. && cmake --build . --target osdi && cd ..
```

### 2. Generate Your First Dataset

```bash
python scripts/generate_training_data.py \
    --osdi build/osdi/bsimcmg.osdi \
    --tech ASAP7
# Output: training_data/ASAP7_dc.csv (~300MB, 1.5M rows)
```

### 3. Load into Your ML Framework

```python
import pandas as pd
df = pd.read_csv("training_data/ASAP7_dc.csv")
inputs = df[["L", "NFIN", "TFIN", "temp_K", "Vg", "Vd", "Vs", "Ve"]].values
outputs = df[["ids", "gm", "gds", "cgg", "cgd", "cgs"]].values
```

## Installation

### Prerequisites

- **Python 3.8+** with NumPy
- **OpenVAF compiler** (v23.5.0+) -- compiles Verilog-A to OSDI
- **CMake** (v3.20+) -- build system
- **NGSPICE** (v45+) -- optional, for verification tests only

### Building the OSDI Binary

**Option A: CMake (recommended)**

```bash
mkdir -p build && cd build
cmake ..
cmake --build . --target osdi
```

**Option B: Direct OpenVAF**

```bash
mkdir -p build/osdi
openvaf -I bsim-cmg-va/code -o build/osdi/bsimcmg.osdi bsim-cmg-va/code/bsimcmg_main.va
```

Verify the output exists: `build/osdi/bsimcmg.osdi` (should be ~2-3 MB shared object).

### Install Python Dependencies

```bash
pip install numpy pytest
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NGSPICE_BIN` | Path to custom NGSPICE binary | `/usr/local/ngspice-45.2/bin/ngspice` |
| `ASAP7_MODELCARD` | Override ASAP7 modelcard path (file or directory) | `modelcards/ASAP7/7nm_TT_160803.pm` |

## Generating Training Data

### CLI Usage

The main entry point is `scripts/generate_training_data.py`.

```bash
# All 5 technologies, all 42 devices, PDK-defined geometry sweep
python scripts/generate_training_data.py \
    --osdi build/osdi/bsimcmg.osdi

# Single technology
python scripts/generate_training_data.py \
    --osdi build/osdi/bsimcmg.osdi \
    --tech TSMC7

# Specific devices with glob patterns
python scripts/generate_training_data.py \
    --osdi build/osdi/bsimcmg.osdi \
    --tech ASAP7 --devices nmos_*

# Custom voltage grid and temperature
python scripts/generate_training_data.py \
    --osdi build/osdi/bsimcmg.osdi \
    --tech ASAP7 \
    --temps -40 27 85 125 \
    --vg-points 80 --vd-points 80

# Quick test (single default geometry, minimal grid)
python scripts/generate_training_data.py \
    --osdi build/osdi/bsimcmg.osdi \
    --tech ASAP7 --devices nmos_rvt \
    --no-sweep-geometry --temps 27 \
    --vg-points 5 --vd-points 5

# List available devices
python scripts/generate_training_data.py \
    --osdi build/osdi/bsimcmg.osdi --list-devices
```

By default, `--sweep-geometry` is enabled: the sweep enumerates all PDK-defined (L, NFIN) bin boundary combinations from the technology modelcard. Use `--no-sweep-geometry` for a single (min_l, 1) point per device.

### Python API: One-Liner

```python
from pycmg import generate_dataset

paths = generate_dataset(
    osdi_path="build/osdi/bsimcmg.osdi",
    techs=["ASAP7"],
    output_dir="./training_data",
)
# Returns: ["/.../training_data/ASAP7_dc.csv"]
```

### Python API: Composable Pipeline

```python
from pycmg.sweep import SweepConfig, sweep_dc, to_csv

config = SweepConfig(
    techs=["ASAP7", "TSMC7"],
    sweep_geometry=True,              # use PDK-defined (L, NFIN) combos
    temperatures=[300.15, 358.15],    # 27C, 85C in Kelvin
    vg_points=80,
    vd_points=80,
)

result = sweep_dc("build/osdi/bsimcmg.osdi", config, verbose=2)
paths = to_csv(result, "./training_data", split_by="tech")
```

### PDK-Defined Geometry Sweep

TSMC PDKs organize model variants as a 2D grid of (L_bin x NFIN_group), where each combination has specifically fitted binning coefficients. The sweep engine reads these bin boundaries directly from the PDK file:

```python
from pycmg import scan_pdk_geometry_combos

# Enumerate all (L, NFIN) sweep points for TSMC7 PMOS LVT
combos = scan_pdk_geometry_combos(
    "modelcards/TSMC7/cln7_1d8_sp_v1d2_2p2.l",
    "pch_lvt_mac",
)
# Returns 42 (L, NFIN) pairs: 6 L bins x 7 unique NFIN boundaries
# [(8e-9, 1.0), (8e-9, 2.0), ..., (1.2e-7, 24.888)]
```

For each variant, the sweep generates two points using `{nfinmin, nfinmax}` boundaries. This ensures correct binning coefficients are used for every (L, NFIN) combination.

### Process Variation

Sweep model parameters like EOT (oxide thickness) or TOXP alongside the voltage grid:

```bash
python scripts/generate_training_data.py \
    --osdi build/osdi/bsimcmg.osdi \
    --tech ASAP7 \
    --process-var eot=0.9e-9,1.0e-9,1.1e-9 \
    --process-var toxp=1.8e-9,2.1e-9
```

Process variation parameters are passed as `model_overrides` to `Instance`, overriding the modelcard value for each combination in the Cartesian product. The resulting CSV includes extra columns for each varied parameter.

### Extended Voltage Range (2*VDD)

For NN models used in circuit simulators, the Newton-Raphson solver may temporarily evaluate voltages beyond the nominal VDD. Training data covering `[0, 2*VDD]` prevents the NN from extrapolating in these regions:

```bash
# Extend voltage sweep to 2x VDD
python scripts/generate_training_data.py \
    --osdi build/osdi/bsimcmg.osdi \
    --tech ASAP7 --devices nmos_rvt \
    --voltage-scale 2.0

# Or via Python API
from pycmg import generate_dataset
paths = generate_dataset(
    osdi_path="build/osdi/bsimcmg.osdi",
    techs=["ASAP7"],
    voltage_scale=2.0,
)
```

The dense region around Vth keeps the same width regardless of `voltage_scale` -- only the sparse grid and Vd grid extend to `VDD * voltage_scale`.

### Sensitivity Analysis: Finding Dominant Process Parameters

Before modeling process variation, identify which parameters matter most. The sensitivity analysis tool perturbs each BSIM-CMG model parameter independently and ranks them by influence on I-V, Q-V, and C-V characteristics:

```bash
# Identify top 9 process parameters for ASAP7 NMOS
python scripts/sensitivity_analysis.py \
    --osdi build/osdi/bsimcmg.osdi \
    --tech ASAP7 --device nmos_rvt

# Custom: TSMC5, 10% perturbation, top 15
python scripts/sensitivity_analysis.py \
    --osdi build/osdi/bsimcmg.osdi \
    --tech TSMC5 --device nmos_svt \
    --delta 0.10 --top-n 15

# Save full results to CSV
python scripts/sensitivity_analysis.py \
    --osdi build/osdi/bsimcmg.osdi \
    --tech ASAP7 --device nmos_rvt \
    --output sensitivity_results.csv
```

**Example output** (TSMC5 nmos_svt):

```
=== I-V Sensitivity (top 9) ===
Rank  Parameter      ids       gm        gds       gmb       Score
1     phig           2.04e+01  3.16e+03  3.21e+03  5.13e+03  1.15e+04
2     easub          1.91e+01  2.42e+03  2.44e+03  3.23e+03  8.10e+03
3     nu0            2.13e+00  2.82e+00  2.83e+00  2.85e+00  1.06e+01
...
```

**Python API:**

```python
from pycmg import compute_sensitivity

result = compute_sensitivity(
    osdi_path="build/osdi/bsimcmg.osdi",
    modelcard_path="modelcards/ASAP7/7nm_TT_160803.pm",
    model_name="nmos_rvt",
    inst_params={"L": 21e-9, "TFIN": 6.5e-9, "NFIN": 1.0},
    vdd=0.9,
    device_type="nmos",
    delta_fraction=0.05,  # 5% perturbation
    top_n=9,
)

# Top 9 parameters for each category
print(result.rankings["iv"])  # I-V: ['phig', 'easub', ...]
print(result.rankings["qv"])  # Q-V: ['phig', 'easub', ...]
print(result.rankings["cv"])  # C-V: ['phig', 'easub', ...]

# Full sensitivity data per parameter per output
print(result.sensitivities["phig"])  # {'ids': 94.02, 'gm': 29.70, ...}
```

The analysis evaluates at 4 representative bias points (subthreshold, linear, saturation, strong inversion) using central-difference perturbation and normalized sensitivity.

### NN Training Data Generation (.npz)

For neural network compact model training, PyCMG provides a dedicated `.npz` data generator that:
- Enumerates **PDK-legal (L, NFIN) bin boundaries** for TSMC techs (or uses a fallback NFIN list for ASAP7)
- Extracts **process parameters on-the-fly** from the resolved modelcard for each (L, NFIN) bin
- Sweeps in a **source-relative frame** (Vs=0) over a `[0, voltage_box_factor·VDD]` box (default 2·VDD; PMOS mirrors through the origin) to cover Newton-Raphson overshoot
- Writes geometry as 15 columns: `[NFIN, L, T, PHIG, U0, VSAT, EOT, ETA0, CIT, RDSW, CFS, TOXP, CGSL, UA, EU]`
- Tags every kept row with a `sample_class` int8 code (`anchor / vds_zero / subthresh / small_vds / grid / hot / lhs`) so downstream loss/data-augmentation code can subset rows by origin

Each `(variant, L, NFIN, T)` bin produces:

- **~489 targeted points** — anchors + `Vds=0` boundary line (60) + subthreshold transition (300) + small-`Vds` linear region (120). These enforce `Id(Vds=0)=0`, switching-region accuracy, and linear-region fidelity.
- **Bulk samples** — selected by `--sampler`:
  - `grid` (default): hybrid uniform `(grid_per_axis × grid_per_axis × vbs_levels)` grid in `(Vgs, Vds, Vbs)` with `N(0, jitter_sigma_frac·VDD)` Gaussian jitter, plus a `(hot_per_axis × hot_per_axis × vbs_levels)` densification on the saturation plateau (`Vgs ∈ [0.5,1]·VDD`, `Vds ∈ [0.4,1]·VDD`). Defaults give 4500 grid + 720 hot ≈ 5220 bulk samples per bin (~2.3× hot-box density boost vs uniform).
  - `lhs`: legacy Latin Hypercube (default 5000 samples per bin), retained for ablation.

```bash
# Generate universal dataset across all 5 techs and 21 variants
# (defaults: grid sampler, 30×30×5 base + 12×12×5 hot, T = {-25,27,125}°C, box = 2·VDD)
python scripts/generate_nn_data.py --device both --universal

# Single technology / variant subset
python scripts/generate_nn_data.py --device nmos --tech tsmc7
python scripts/generate_nn_data.py --device nmos --tech tsmc7 --variants svt,lvt

# Tune the hybrid grid sampler
python scripts/generate_nn_data.py --device both --universal \
    --grid-per-axis 40 --hot-per-axis 16 --vbs-levels 5 \
    --jitter-sigma-frac 0.05

# Switch back to legacy LHS for ablation
python scripts/generate_nn_data.py --device both --universal \
    --sampler lhs --n-lhs-samples 8000

# Multi-process bin generation + custom temperatures + output dir
python scripts/generate_nn_data.py --device both --universal \
    --temperatures 248.15,300.15,398.15 \
    --n-workers 8 --seed 42 --data-dir ./my_data

# Also write a stratified fine-tune split alongside the main file
python scripts/generate_nn_data.py --device both --universal --finetune-size 200000
```

**CLI flags** (see `scripts/generate_nn_data.py --help` for the full list):

| Flag | Default | Notes |
|------|---------|-------|
| `--sampler {grid,lhs}` | `grid` | Bulk-sample sampler. `grid` = hybrid uniform grid + jitter + hot densification. |
| `--grid-per-axis N` | 30 | Base 2D grid size per axis (Vgs, Vds). |
| `--vbs-levels N` | 5 | Number of Vbs levels in `{0, ±0.25, ±0.5}·VDD`. |
| `--hot-per-axis N` | 12 | Hot-region densification grid size (0 to disable). |
| `--jitter-sigma-frac F` | 0.05 | Gaussian jitter sigma in fractions of VDD. |
| `--n-lhs-samples N` | 5000 | LHS samples per bin (only used when `--sampler=lhs`). |
| `--voltage-box-factor F` | 2.0 | Voltage box width in units of VDD. |
| `--temperatures K1,K2,...` | -25,27,125 °C | Comma-separated temperatures in Kelvin. |
| `--n-workers N` | 1 | Parallel worker count (1 = serial). |
| `--seed N` | 42 | Base RNG seed; per-bin seeds derive monotonically. |
| `--finetune-size N` | 0 | If >0, also write `finetune_<base>.npz` with a stratified subset. |
| `--data-dir DIR` | (auto) | Output directory for `.npz` files. |

**Python API:**

```python
from pycmg.nn_generate import generate_dataset, generate_universal_dataset
from pycmg.nn_config import TECH_CONFIGS
from pycmg.sweep import save_npz

# Single tech (defaults: sampler="grid", grid_per_axis=30, hot_per_axis=12, vbs_levels=5)
data = generate_dataset(TECH_CONFIGS["tsmc7"], "nmos")
save_npz(data["inputs"], data["geometry"], data["outputs"],
         "tsmc7_nmos.npz",
         metadata=data["metadata"], sample_class=data.get("sample_class"))

# Universal (all techs)
data = generate_universal_dataset("nmos")
save_npz(data["inputs"], data["geometry"], data["outputs"],
         "universal_nmos.npz",
         metadata=data["metadata"], sample_class=data.get("sample_class"))
```

**Output format:**
- `inputs` (N, 4): source-relative terminal voltages `[Vd, Vg, Vs, Vb]` (NMOS in `[0, 2·VDD]`; PMOS mirrored to `[-2·VDD, 0]`).
- `geometry` (N, 15): `[NFIN, L, T, PHIG, U0, VSAT, EOT, ETA0, CIT, RDSW, CFS, TOXP, CGSL, UA, EU]`.
- `outputs` (N, 13): `[id, gm, gds, gmb, qg, qd, qs, qb, cgg, cgd, cgs, cdg, cdd]`.
- `sample_class` (N,) int8 — origin of each row. Decode via `metadata["sample_class_names"]`.

Process parameters are **per-bin accurate**: for TSMC techs, different (L, NFIN) bins may produce different process param values because the variant overlay differs per bin. This is more accurate than using a single set of process parameters per Vt flavor.

**Coverage:** 1692 total (L, NFIN) geometry combos across 6 techs, 24 variants. TSMC techs have ~36-40 combos per device (from PDK bin boundaries); ASAP7 has 48 combos per device (fallback lists).

## Supported Technologies

| Technology | Node | Vdd | TFIN | Vt Flavors | Devices |
|------------|------|-----|------|------------|---------|
| ASAP7 | 7nm | 0.90V | 6.5nm | rvt, lvt, slvt, sram | 8 |
| TSMC5 | 5nm | 0.65V | 6.0nm | svt, lvt, ulvt, elvt | 8 |
| TSMC6 | 6nm | 0.75V | 6.0nm | svt, lvt, ulvt | 6 |
| TSMC7 | 7nm | 0.75V | 6.0nm | svt, lvt, ulvt | 6 |
| TSMC12 | 12nm | 0.80V | 6.0nm | svt, lvt, hvt, ulvt, lnvt | 10 |
| TSMC16 | 16nm | 0.80V | 6.0nm | svt, lvt, hvt, ulvt, lnvt | 10 |
| **Total** | | | | | **48** |

Each "device" is an NMOS/PMOS pair for a given Vt flavor. For example, ASAP7 has 4 flavors x 2 polarities = 8 devices: `nmos_rvt`, `pmos_rvt`, `nmos_lvt`, `pmos_lvt`, etc.

Gate lengths and NFIN ranges are defined by PDK bin boundaries. The sweep engine reads these directly from the TSMC PDK files (discrete lmin values and nfinmin/nfinmax groups). For ASAP7, TSMC7's NFIN boundaries are used as reference.

## Output Format

### CSV Schema

Each row is one DC operating point. The base schema has 28 columns (more if process variation is enabled):

| Group | Columns | Unit |
|-------|---------|------|
| Identity | tech, device | -- |
| Geometry | L, NFIN, TFIN, temp_K | m, --, m, K |
| Voltage | Vg, Vd, Vs, Ve, Vth | V |
| Currents | id, ig, is, ie, ids | A |
| Charges | qg, qd, qs, qb | C |
| Derivatives | gm, gds, gmb | S |
| Capacitances | cgg, cgd, cgs, cdg, cdd | F |

When `--process-var` is used, the varied parameter columns (e.g., `eot`, `toxp`) are inserted between the geometry and voltage groups, sorted alphabetically.

### Dataset Size Estimates

With default settings (`sweep_geometry=True`, 5 temperatures, 50x50 voltage grid), data size depends on each technology's PDK variant structure. TSMC nodes typically have 25-42 geometry combos per device, ASAP7 has 6-7 (one L, TSMC7 NFIN boundaries).

### Non-Uniform Voltage Sampling

The Vg grid uses threshold-aware non-uniform sampling to concentrate points near the subthreshold-to-saturation transition region, which is critical for NN training accuracy:

- **Dense region** (default 60% of points): +/- 0.15*Vdd centered on Vth
- **Sparse region** (remaining 40%): uniformly distributed across [0, Vdd]
- **Vd grid**: uniform across [0, Vdd]

Threshold voltage (Vth) is auto-detected per device configuration via the peak-gm method before building the grid.

### Loading Data for Training

```python
import pandas as pd
df = pd.read_csv("training_data/ASAP7_dc.csv")
nmos = df[df["device"].str.startswith("nmos")]

# PyTorch
import torch
from torch.utils.data import Dataset, DataLoader

class MosfetDataset(Dataset):
    def __init__(self, csv_path, input_cols, output_cols):
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(df[input_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[output_cols].values, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

ds = MosfetDataset(
    "training_data/ASAP7_dc.csv",
    input_cols=["L", "NFIN", "TFIN", "temp_K", "Vg", "Vd", "Vs", "Ve"],
    output_cols=["ids", "gm", "gds", "cgg", "cgd", "cgs"],
)
loader = DataLoader(ds, batch_size=1024, shuffle=True)
```

## Advanced Usage

### Single-Point Evaluation

For interactive exploration or custom sweep logic, use the `Model` and `Instance` API directly:

```python
from pycmg import Model, Instance

model = Model(
    osdi_path="build/osdi/bsimcmg.osdi",
    modelcard_path="modelcards/ASAP7/7nm_TT_160803.pm",
    model_name="nmos_rvt",
)

inst = Instance(model, params={"L": 7e-9, "TFIN": 6.5e-9, "NFIN": 2.0})

result = inst.eval_dc({"d": 0.7, "g": 0.5, "s": 0.0, "e": 0.0})
print(f"Ids = {result['ids']:.3e} A")
print(f"gm  = {result['gm']:.3e} S")
print(f"gds = {result['gds']:.3e} S")
print(f"cgg = {result['cgg']:.3e} F")
```

All 17 outputs are available: `id`, `ig`, `is`, `ie`, `ids`, `qg`, `qd`, `qs`, `qb`, `gm`, `gds`, `gmb`, `cgg`, `cgd`, `cgs`, `cdg`, `cdd`.

### Temperature Sweeps

Temperature is specified in Kelvin. Convert from Celsius: `temp_K = temp_C + 273.15`.

```python
for temp_c in [-40, 27, 85, 125]:
    temp_k = temp_c + 273.15
    inst = Instance(model, params={"L": 7e-9, "TFIN": 6.5e-9, "NFIN": 2.0},
                    temperature=temp_k)
    result = inst.eval_dc({"d": 0.7, "g": 0.5, "s": 0.0, "e": 0.0})
    print(f"T={temp_c:4d}C: Ids={result['ids']:.3e} A, gm={result['gm']:.3e} S")
```

### Body Bias Effects

Apply body bias via the `e` (extended/bulk) terminal:

```python
for ve in [0.0, 0.1, 0.2, 0.3]:
    result = inst.eval_dc({"d": 0.7, "g": 0.5, "s": 0.0, "e": ve})
    print(f"Ve={ve:.1f}V: Ids={result['ids']:.3e} A, gmb={result['gmb']:.3e} S")
```

In sweep mode, use `--ve-values 0.0 0.1 0.2 0.3` on the CLI.

### NFIN Scaling

```python
for nfin in [2, 3, 5, 10]:
    inst = Instance(model, params={"L": 7e-9, "TFIN": 6.5e-9, "NFIN": float(nfin)})
    result = inst.eval_dc({"d": 0.7, "g": 0.5, "s": 0.0, "e": 0.0})
    print(f"NFIN={nfin}: Ids={result['ids']:.3e} A")
```

> **Note:** NFIN=1 causes convergence failures for certain TSMC process variants (e.g., tsmc5:ulvt, tsmc16:lnvt) where BSIM-CMG parameters go negative. Use NFIN >= 2 for reliable results. See [Known Limitations](#known-limitations).

### Custom Voltage Grids

Build your own voltage grid for finer control:

```python
from pycmg.sweep import build_voltage_grid, find_threshold, build_nodes

vth = find_threshold(inst, vdd=0.9, device_type="nmos")

# Standard range [0, VDD]
vg_arr, vd_arr = build_voltage_grid(vdd=0.9, vth_mag=vth, vg_points=100, vd_points=100)

# Extended range [0, 2*VDD] for simulator convergence training
vg_arr, vd_arr = build_voltage_grid(vdd=0.9, vth_mag=vth, vg_points=100, vd_points=100,
                                     voltage_scale=2.0)

# Source-relative range for NN training (NMOS: [-VDD, 2*VDD])
vg_arr, vd_arr = build_voltage_grid(vdd=0.9, vth_mag=vth, vg_points=71, vd_points=71,
                                     voltage_scale=2.0, v_min=-0.9,
                                     vth_center=vth, n_dense_mid=30)

# PMOS source-relative frame ([-2*VDD, +VDD])
vg_arr, vd_arr = build_voltage_grid(vdd=0.9, vth_mag=vth, vg_points=71, vd_points=71,
                                     voltage_scale=1.0, v_min=-1.8,
                                     vth_center=-vth, n_dense_mid=30)

for vg in vg_arr:
    for vd in vd_arr:
        nodes = build_nodes(vg, vd, 0.0, 0.9, "nmos")
        result = inst.eval_dc(nodes)
```

### PMOS Conventions

PMOS devices use source-at-Vdd convention. In magnitude space (used by the sweep engine), all voltages are positive. The `build_nodes()` function handles the polarity mapping:

```python
# NMOS: source at ground
nodes_nmos = build_nodes(vg_mag=0.5, vd_mag=0.7, ve_mag=0.0, vdd=0.9, device_type="nmos")
# -> {"g": 0.5, "d": 0.7, "s": 0.0, "e": 0.0}

# PMOS: source at Vdd, voltages reflected
nodes_pmos = build_nodes(vg_mag=0.5, vd_mag=0.7, ve_mag=0.0, vdd=0.9, device_type="pmos")
# -> {"g": 0.4, "d": 0.2, "s": 0.9, "e": 0.9}
```

### Transient Analysis

```python
result = inst.eval_tran(
    nodes={"d": 0.7, "g": 0.5, "s": 0.0, "e": 0.0},
    time=1e-9,
    delta_t=1e-12,
)
# Returns: id, ig, is, ie, ids, qg, qd, qs, qb
```

### Jacobian Matrix

Extract the condensed 4x4 Jacobian (dI/dV):

```python
J = inst.get_jacobian_matrix({"d": 0.7, "g": 0.5, "s": 0.0, "e": 0.0})
# J is a 4x4 numpy array, terminals ordered as [d, g, s, e]
# J[i,j] = dI_terminal_i / dV_terminal_j
```

## Verification Against NGSPICE

### Strategy

PyCMG wraps the OSDI binary directly via ctypes, while NGSPICE loads the **same** OSDI binary via the `.osdi` command. Tests compare PyCMG output vs NGSPICE output to verify:

1. **Binary-level consistency**: Both use the identical `bsimcmg.osdi` file
2. **Ctypes wrapper correctness**: Proper OSDI function call sequences
3. **Numerical accuracy**: Direct comparison of currents, charges, derivatives
4. **Full model coverage**: DC, AC (capacitance), and transient analysis

The OSDI binary is the single source of truth for all model physics calculations.

### Tolerances

| Parameter | Absolute Tolerance | Relative Tolerance |
|-----------|--------------------|--------------------|
| Current (A) | 1e-9 | 0.5% |
| Charge (C) | 1e-18 | 0.5% |
| Conductance (S) | 1e-6 | 1% |
| Capacitance (F) | 1e-18 | 1% |

### Running the Test Suite

All NGSPICE-backed DC tests use shared infrastructure: `run_dc_comparison()` (helpers.py), `standard_bias_points()` and `@requires_osdi` (conftest.py). Each DC test file is a single parametrized function.

```bash
# Quick smoke tests (no NGSPICE required)
pytest tests/test_api.py tests/test_tech.py tests/test_sweep.py -v

# Base technology verification (5 techs, NGSPICE required)
pytest tests/test_dc_regions.py tests/test_dc_jacobian.py tests/test_transient.py -v

# Full suite (280 tests, ~18 min with NGSPICE)
pytest tests/ -v
```

| Test File | Tests | Description | NGSPICE |
|-----------|-------|-------------|---------|
| `test_api.py` | 20 | API smoke tests | No |
| `test_tech.py` | 13 | Technology registry | No |
| `test_sweep.py` | 27 | Sweep engine | No |
| `test_sensitivity.py` | 7 | Sensitivity analysis | No |
| `test_nfin_scaling.py` | 2 | NFIN scaling sanity | No |
| `test_dc_regions.py` | 30 | DC off/linear/saturation, 5 techs × 2 devices × 3 regions | Yes |
| `test_dc_jacobian.py` | 30 | DC Jacobian (central finite-diff), 5 techs × 2 devices × 3 regions | Yes |
| `test_transient.py` | 10 | Transient waveforms, 5 techs × 2 devices | Yes |
| `test_ac_caps.py` | 15 | AC capacitances (cgg/cgd/cgs/cdg/cdd) | Yes |
| `test_body_bias.py` | 20 | Body bias, 5 techs × 2 devices × 2 bias types | Yes |
| `test_temperature.py` | 10 | Temperature (-40/85/125C), ASAP7+TSMC7 | Yes |
| `test_vt_variants.py` | 96 | Vt variant DC, 16 variants × 2 devices × 3 regions | Yes |

## Project Structure

```
pycmg-wrapper/
├── pycmg/                        # Python package
│   ├── __init__.py              # Public API exports (Model, Instance, generate_dataset, ...)
│   ├── osdi_types.py            # OSDI constants, ctypes structures, function types
│   ├── core.py                  # Low-level OSDI interface (OsdiLibrary, OsdiModel, OsdiInstance)
│   ├── parser.py                # Modelcard parsing, PDK introspection (scan_pdk_geometry_combos)
│   ├── model.py                 # Public API (Model, Instance, eval_dc, eval_tran)
│   ├── tech.py                  # Technology registry (TECH_REGISTRY, DeviceConfig, TechConfig)
│   ├── sweep.py                 # Sweep engine (generate_dataset, SweepConfig, sweep_dc, to_csv)
│   ├── sensitivity.py           # Sensitivity analysis (compute_sensitivity, SensitivityResult)
│   ├── nn_config.py             # NN training config (ProcessParams, NNTechConfig, extract_process_params)
│   └── nn_generate.py           # NN .npz data generation using PDK-driven (L, NFIN) combos
├── tests/                        # Test suite (280 tests)
│   ├── conftest.py              # Tiered technology registry (24 entries)
│   ├── helpers.py               # NGSPICE runner helpers, comparison functions
│   ├── test_api.py              # API smoke tests
│   ├── test_tech.py             # Technology registry tests
│   ├── test_sweep.py            # Sweep engine tests (incl. voltage_scale)
│   ├── test_sensitivity.py      # Sensitivity analysis tests
│   ├── test_dc_jacobian.py      # DC Jacobian verification
│   ├── test_dc_regions.py       # DC operating region tests
│   ├── test_transient.py        # Transient waveform verification
│   ├── test_ac_caps.py          # AC capacitance verification
│   ├── test_body_bias.py        # Body bias verification
│   ├── test_temperature.py      # Temperature sweep verification
│   ├── test_nfin_scaling.py     # NFIN scaling sanity
│   └── test_vt_variants.py      # Vt variant DC verification
├── scripts/                      # CLI utilities
│   ├── generate_training_data.py # Training data generation CLI (CSV format)
│   ├── generate_nn_data.py      # NN training data CLI (.npz format, PDK-driven geometry)
│   ├── sensitivity_analysis.py  # Process parameter sensitivity CLI
│   └── generate_naive_tsmc.py   # Naive TSMC modelcard generator
├── modelcards/                   # Technology model cards
│   ├── ASAP7/                   # ASAP7 PDK model files
│   ├── TSMC5/                   # TSMC 5nm model files
│   ├── TSMC6/                   # TSMC 6nm model files
│   ├── TSMC7/                   # TSMC 7nm model files
│   ├── TSMC12/                  # TSMC 12nm model files
│   └── TSMC16/                  # TSMC 16nm model files
├── bsim-cmg-va/                  # Verilog-A source and documentation
│   └── code/                    # BSIM-CMG Verilog-A source files
├── build/                        # Build artifacts (generated)
│   ├── osdi/bsimcmg.osdi       # Compiled OSDI binary
│   └── modelcards/              # Cached generated TSMC modelcards
└── CMakeLists.txt                # Build system
```

## API Reference

### pycmg.sweep

**`generate_dataset(osdi_path, techs, devices, output_dir, ...)`** -- Convenience wrapper. Builds a `SweepConfig`, runs `sweep_dc()`, writes CSVs via `to_csv()`. Returns list of output file paths.

**`SweepConfig`** -- Dataclass configuring the full sweep: `techs`, `devices`, `sweep_geometry` (bool, default True), `temperatures`, `vg_points`, `vd_points`, `ve_values`, `process_vars`, `dense_ratio`, `voltage_scale`.

**`SweepResult`** -- Container with `columns` (ordered column names), `data` (list of rows), `metadata` (timing, counts).

**`sweep_dc(osdi_path, config, verbose)`** -- Core sweep loop. When `sweep_geometry=True`, iterates technologies x devices x PDK-defined (L, NFIN) combos x temperatures x process combos x voltage grid. Returns `SweepResult`.

**`to_csv(results, output_dir, split_by)`** -- Writes `SweepResult` to CSV files. `split_by` controls grouping: `"tech"` (default), `"device"`, or `"none"`.

**`build_voltage_grid(vdd, vth_mag, vg_points, vd_points, dense_ratio, voltage_scale, v_min, n_dense_mid, vth_center)`** -- Non-uniform Vg + uniform Vd grid builder. `voltage_scale` extends the upper bound to `vdd * voltage_scale` (default 1.0). `v_min` sets the lower bound (default 0.0; use `-vdd` for source-relative NN training). `n_dense_mid` adds extra dense points near mid-supply (default 0). `vth_center` overrides the dense region center (default None = use `vth_mag`; pass negative for PMOS source-relative frame).

**`NN_OUTPUT_COLUMNS`** -- List of 13 NN training target column names (subset of `OUTPUT_KEYS`, excludes `ig`, `is`, `ie`, `ids`).

**`save_npz(inputs, geometry, outputs, output_path, metadata)`** -- Save NN training arrays to `.npz` file with optional metadata.

**`find_threshold(inst, vdd, device_type, n_coarse)`** -- Peak-gm threshold detection.

**`build_nodes(vg_mag, vd_mag, ve_mag, vdd, device_type)`** -- Magnitude-space to terminal-voltage mapping.

### pycmg.model

**`Model(osdi_path, modelcard_path, model_name)`** -- Loads BSIM-CMG model from OSDI binary + modelcard file.

**`Instance(model, params, temperature, model_overrides)`** -- Device instance with geometry. `params` sets instance parameters (L, TFIN, NFIN). `temperature` in Kelvin (default: 300.15). `model_overrides` overrides modelcard parameters (for process variation).

- `eval_dc(nodes) -> dict` -- DC operating point. Returns 17 outputs. Raises `RuntimeError` if internal node NR fails to converge (e.g., NFIN=1 with certain TSMC variants).
- `eval_tran(nodes, time, delta_t) -> dict` -- Transient evaluation. Returns 9 outputs. Warns (instead of raising) on internal node convergence failure; the circuit-level NR provides outer convergence.
- `get_jacobian_matrix(nodes) -> np.ndarray` -- 4x4 condensed Jacobian (dI/dV).
- `set_params(params, allow_rebind)` -- Update instance parameters.

### pycmg.tech

**`TECH_REGISTRY`** -- Dict mapping technology names to `TechConfig` objects. 5 technologies, 42 devices total.

**`TechConfig`** -- Technology node config: `name`, `vdd`, `tfin`, `devices` (dict of `DeviceConfig`), `pdk_path`.

**`DeviceConfig`** -- Single device config: `model_name`, `inst_params`, `modelcard`, `pdk_device`, `get_min_l()`, `get_geometry_combos()`.

**`resolve_modelcard(device, tech, L, NFIN=None)`** -- Returns modelcard path. For ASAP7, returns the static file. For TSMC, generates a naive modelcard from the PDK on-the-fly and caches it under `build/modelcards/`. When `NFIN` is provided, selects the correct NFIN-group variant.

**`get_tech_config(name)`** / **`list_techs()`** -- Registry lookup helpers.

### pycmg.sensitivity

**`compute_sensitivity(osdi_path, modelcard_path, model_name, inst_params, vdd, device_type, temperature, delta_fraction, top_n, verbose)`** -- OAT sensitivity analysis. Perturbs each real-valued model parameter by `+/- delta_fraction` and measures normalized output change at 4 representative bias points. Returns `SensitivityResult`.

**`SensitivityResult`** -- Container with `param_names`, `sensitivities` (per-param per-output normalized sensitivity), `rankings` (per-category top-N lists: `"iv"`, `"qv"`, `"cv"`), `bias_points`, `delta_fraction`.

**`enumerate_model_params(desc, model)`** -- Discovers all real-valued model-level parameters from the OSDI descriptor. Returns list of `ParamInfo(index, name, value)`.

**`rank_parameters(sensitivities, categories, top_n)`** -- Ranks parameters by aggregate sensitivity within each output category.

**`format_sensitivity_table(result, category)`** -- Formats a ranked sensitivity table for terminal output.

### pycmg.parser

**`parse_modelcard(path, target_model_name)`** -- Parses a SPICE `.model` block. Returns `ParsedModel` with `name` and `params` dict.

**`parse_number_with_suffix(s)`** -- Parses SPICE numbers with engineering suffixes (e.g., `"16n"` -> `16e-9`, `"1.5meg"` -> `1.5e6`).

**`scan_pdk_geometry_combos(path, base_name)`** -- Enumerates PDK-defined (L, NFIN) sweep points for a TSMC device. For each variant, returns `(lmin, nfinmin)` and `(lmin, nfinmax)`. Sorted and deduplicated.

### pycmg.nn_config

**`ProcessParams`** -- Dataclass with 12 BSIM-CMG process parameters used as NN input features: `phig`, `u0`, `vsat`, `eot`, `eta0`, `cit`, `rdsw`, `cfs`, `toxp`, `cgsl`, `ua`, `eu`. Methods: `as_array()` (ordered list), `as_dict()`.

**`extract_process_params(modelcard_params)`** -- Extracts the 12 NN process params from a parsed modelcard dict (lowercase keys, as from `Model.modelcard_params`). Missing params default to 0.0.

**`NNTechConfig`** -- NN training config wrapping PyCMG's `TechConfig`. Stores training VDD, variant name list, temperature, and optional fallback NFIN values (for ASAP7). Properties: `name`, `vdd`, `tfin`, `pycmg_tech`. Methods: `get_geometry_combos(device_type, variant)` (returns PDK-legal `(L, NFIN)` pairs), `get_model_name(device_type, variant)`, `resolve_modelcard(device_type, variant, L, NFIN)`.

**`TECH_CONFIGS`** -- Dict of 6 NNTechConfig objects: `asap7`, `tsmc5`, `tsmc6`, `tsmc7`, `tsmc12`, `tsmc16`.

**`INPUT_COLUMNS`** -- 19 NN input feature names: 4 voltages + `NFIN` + `L` + `T` + 12 process params.

**`OUTPUT_COLUMNS`** -- 13 NN output target names (same as `NN_OUTPUT_COLUMNS`).

### pycmg.nn_generate

**`generate_dataset(tech, device_type, *, variant_names, temperatures, n_lhs_samples, voltage_box_factor, n_workers, seed, verbose, sampler, grid_per_axis, vbs_levels, hot_per_axis, jitter_sigma_frac)`** -- Generate NN training data for one tech/polarity. Iterates PDK-legal (L, NFIN) combos x temperatures x bins, extracts process params on-the-fly, runs the targeted+bulk sampler chosen by `sampler`. Returns dict with `inputs` (N,4), `geometry` (N,15), `outputs` (N,13), `sample_class` (N,) int8, `metadata`.

**`generate_universal_dataset(device_type, *, ...)`** -- Same signature as `generate_dataset` (minus `tech`). Concatenates results across all 5 technologies.

**`SAMPLE_CLASS_NAMES`** -- Tuple of class labels indexed by the `sample_class` int8 codes: `("anchor", "vds_zero", "subthresh", "small_vds", "grid", "hot", "lhs")`.

**`eval_single_point(inst, vd, vg, vs, vb)`** -- Evaluate one DC bias point. Returns dict of 13 outputs or None on failure.

## Known Limitations

### NFIN=1 Convergence Failures

BSIM-CMG computes NFIN-dependent instance parameters (ETA0_i, U0_i, UA_i) that can become negative at NFIN=1 for certain process variants. The OSDI binary warns but does not abort. The internal node Newton-Raphson then diverges monotonically (0.2 V/step × 200 iterations → ~40 V internal drain), producing `id ≈ 40 kA` and `NaN` for all derivatives.

**Affected variants** (known): `tsmc5:ulvt`, `tsmc16:lnvt` at NFIN=1. Other techs (ASAP7, TSMC7, TSMC12) and NFIN ≥ 2 are unaffected.

**Behavior**: `eval_dc` raises `RuntimeError` when internal NR fails to converge. Callers that sweep bias points should catch this exception. `eval_tran` warns instead of raising.

**Recommendation**: Use NFIN ≥ 2 for data generation and sweeps. NFIN=1 single-fin devices are an edge case rarely used in real designs.

## License

This project is provided for educational and research purposes. The BSIM-CMG model is licensed separately by the BSIM Group at UC Berkeley.
