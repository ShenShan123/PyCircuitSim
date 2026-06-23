# Project: PyCircuitSim

## Overview

Python-based SPICE-like circuit simulator emphasizing educational clarity and modular architecture.
**Primary Goal:** specific support for three compact model families:

- **BSIM-CMG** (LEVEL=72) — PyCMG-wrapped OSDI FinFET model (ground truth).
- **DirectNet** (LEVEL=73) — baseline feed-forward MLP compact model (PyTorch).
- **BSIM-AR Transformer** (LEVEL=74) — autoregressive Transformer compact model (PyTorch).

DirectNet and BSIM-AR share the same data, normalization, and evaluation pipelines via the unified `bsimar` package at `external_compact_models/bsimar/`. DirectNet is the baseline for comparison against BSIM-AR.

Must support **Operating Point**, **DC Sweep**, and **Transient Analysis** for all model types.

**Core Principles:** pure Python; Solver ↔ Device Models decoupled; production-grade compact models via PyCMG/OSDI; basic HSPICE netlist compatibility.

## Architecture

### Module Structure

```
pycircuitsim/
├── config.py           # Path configuration (OSDI binary, modelcards)
├── simulation.py       # Orchestration (run_simulation, run_dc_sweep, run_transient)
├── parser.py           # Two-pass netlist parsing, .model directive support
├── circuit.py          # Circuit topology
├── solver.py           # MNA matrix + Newton-Raphson; DC/Transient/AC solvers
├── logger.py           # HSPICE-like .lis output
├── visualizer.py       # Matplotlib plotting
└── models/
    ├── base.py               # Component abstract base
    ├── passive.py            # R, C, V, I sources (PULSE)
    ├── mosfet_cmg.py         # BSIM-CMG (LEVEL=72) via PyCMG
    ├── mosfet_nn.py          # Shared _MOSFETNNBase (LEVEL=73/74) — voltage prep, autograd, Vds correction
    ├── mosfet_directnet.py   # DirectNet (LEVEL=73, primary)
    └── mosfet_bsimar.py      # BSIMAR Transformer (LEVEL=74, parked — see Rule 15)

external_compact_models/
├── bsimar/             # Unified NN compact model package (importable as `bsimar`)
│   ├── config.py                   # NNTechConfig + TECH_CODE_MAP + local-vocab helpers
│   ├── data/{normalize,dataset}.py
│   ├── models/{direct_net,transformer}.py    # nn.Embedding tech-code
│   ├── losses/bni_mae.py           # MAELoss + per-target LDS weights
│   ├── training/trainer.py
│   ├── eval/{metrics,loo_labels}.py
│   ├── cli/train.py                # `python -m bsimar.cli.train --model direct ...`
│   └── checkpoints/                # *.pt + _norm.npz (gitignored)
└── PyCMG/              # BSIM-CMG OSDI wrapper (git submodule)
    ├── pycmg/{core,model,parser,osdi_types,tech}.py
    ├── build/osdi/bsimcmg.osdi
    └── modelcards/     # ASAP7/*.pm committed; TSMC{5,7,12,16}/cln*.l gitignored (IP)

main.py                 # CLI entry point
examples/*.sp           # Example netlists
results/                # Simulation output
tests/
├── common/             # Shared test infra
│   ├── base.py         # PROJECT_ROOT, OSDI_PATH, TechProfile, ALL_TECHS, NGSPICE runner
│   ├── bsimcmg_{dc,tran}.py
│   └── nn.py           # nrmse, mre, checkpoint resolution, sys.path bootstrap
├── references/         # NGSPICE reference netlists
└── verify_*.py         # 3-level DC/transient tests + NN verification
```

### Key Algorithms

* **MNA** — Sparse construction (scipy.sparse lil_matrix → CSR + spsolve).
* **Newton-Raphson** — SPICE-standard convergence (RELTOL + VNTOL).
* **BE → Trap → BDF-2 integration** — Backward Euler step 1, Trapezoidal default, BDF-2 auto on stiffness.
* **Source + GMIN stepping** — homotopy; GMIN stepping opt-in for bistable.
* **LTE sub-stepping** — adaptive internal sub-steps (opt-in via `max_substeps`).
* **Bistable convergence** — DC oscillation detection, adaptive damping, hard `.ic` mode.
* **AC small-signal** — `ACSolver` linearizes about the DC OP and solves complex `Y = G + jωC` per frequency (see Supported Features).

### Key Compact Models

Three MOSFET compact-model families plug into the same solver and share one data,
normalization, and evaluation pipeline (the `bsimar` package). BSIM-CMG is the
authoritative ground truth; DirectNet is the production NN; BSIMAR is parked.

| LEVEL | Model                        | Implementation                            | Role                                 |
| ----- | ---------------------------- | ----------------------------------------- | ------------------------------------ |
| 72    | **BSIM-CMG**           | `models/mosfet_cmg.py` via PyCMG / OSDI | FinFET**ground truth**         |
| 73    | **DirectNet**          | `models/mosfet_directnet.py` (PyTorch)  | **Primary** NN compact model   |
| 74    | **BSIMAR Transformer** | `models/mosfet_bsimar.py` (PyTorch)     | Autoregressive NN (**parked**) |

- **BSIM-CMG (LEVEL=72)** — PyCMG-wrapped OSDI FinFET model. The reference every
  NN trains against and is gated on; all 5 techs (ASAP7, TSMC5/7/12/16) at
  DC <0.1 % / transient ~0.2 % NRMSE vs NGSPICE. Never substitute simplified
  equations for it.
- **DirectNet (LEVEL=73)** — single-shot feed-forward MLP. 7-dim input
  (Vgs, Vds, Vbs, NFIN, L, T, tech_code) with an `nn.Embedding` tech-code;
  gm/gds/gmb are the **autograd Jacobian** of the predicted `id`.
  Production size is `medium`; per-tech NMOS/PMOS checkpoints for TSMC5/7/12/16
  use a local embedding vocab (Rule 16). Charges are predicted and the AC caps
  are their `dQ/dV` autograd.
- **BSIMAR Transformer (LEVEL=74)** — autoregressive Transformer sharing
  DirectNet's data pipeline and inference rules. Parked (Rule 15); no checkpoints
  on disk. Resurrect the cap-head / AR-loop structure from CHANGELOG / git
  (Rules 9–10).

## Supported Features

* **Devices:** R, C; NMOS/PMOS LEVEL=72 (BSIM-CMG, ground truth), LEVEL=73 (DirectNet, primary NN), LEVEL=74 (BSIMAR, parked); DC + AC voltage/current sources (`AC=mag phase`), PULSE.
* **Analyses:** `.op`, `.dc`, `.tran`, `.ac`.
* **Directives:** `.model` (LEVEL=72/73/74), `.include`, `.ic`.
* Legacy LEVEL=1 (Shichman-Hodges) removed.

**AC (small-signal frequency-domain) analysis** — `ACSolver` (`solver.py`) linearizes about the DC operating point and solves the complex MNA `Y = G + jωC` per frequency: R/C admittances, the full MOSFET small-signal model (gm/gds/gmb **+ the source-referenced transcapacitance matrix Cgg/Cgd/Cdg/Cdd** from `get_capacitances`, i.e. Miller-coupled roll-off), and AC V/I source stimulus. Sweep types `dec`/`oct`/`lin`. Outputs `*_ac_sweep.csv` (mag/phase per node) + a Bode plot. Validated NGSPICE-exact (LEVEL=72) — see Validation.

## Validation

Inverter circuit must PASS Transient Analysis against NGSPICE ground truth within reasonable numerical tolerance. Never use simplified/self-defined equations as reference.

**AC:** `tests/verify_ac.py` gates `.ac` against ground truth — L1 passive RC vs NGSPICE `.ac` AND the closed-form `1/(1+jωRC)`; L2 BSIM-CMG (LEVEL=72) NMOS common-source amp vs NGSPICE `.ac` on the identical OSDI model. Both agree to ~machine precision (L2: gain err 5e-6 dB, phase err 2e-5°, mag NRMSE 5e-7). **DirectNet (LEVEL=73) AC is NGSPICE-gated** across all 24 capacity checkpoints (`tests/verify_nn_ac.py` device CS-amp + `tests/verify_complex_opamp_ac.py` opamp open-loop, shared infra `tests/common/complex_ac.py`). The NN AC caps are autograd `dQ/dV` of its predicted charges. Result: AC **gain** fidelity excellent everywhere (24/24 gain0 err <1.5 dB); the cap-driven **pole/bandwidth** is good but tech-variable (device gate 13/24); the Cgd-feedforward **RHP-zero phase is not reproduced** (diagnostic, not gated); the **opamp** AC inherits the DC value-surface fragility (0/12, but tsmc12-large reproduces GBW 0.97× / PM 1.3°). The gaps are value-surface/feedforward-owned, not a charge-derivative deficiency. See `docs/CHANGELOG.md` + REPORT "AC small-signal accuracy". RO + SRAM excluded (no stable amplifying OP).

> **Sprint history, version-by-version status, dead-ends, and the open
> known-issue roadmap live in `docs/CHANGELOG.md` + `MEMORY.md`** — not duplicated
> here. CLAUDE.md tracks the durable architecture, rules, and how-to-run; the
> CHANGELOG tracks what changed when.

## Setup

```bash
conda create -n pycircuitsim python=3.10 -y
conda activate pycircuitsim
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch
git submodule update --init --recursive
```

**Prerequisites:**

- NGSPICE 45.2+: `/usr/local/ngspice-45.2/bin/ngspice`
- OpenVAF 23.5.0+: `/usr/local/bin/openvaf`
- BSIM-CMG OSDI binary: `external_compact_models/PyCMG/build/osdi/bsimcmg.osdi`

## Quick Start

### Run a simulation

```bash
conda activate pycircuitsim
python main.py examples/bsimcmg_inverter_tran.sp           # -> results/
python main.py examples/rc_lowpass_ac.sp -o my_out -v      # custom output dir + verbose
```

The analysis is chosen by the directive **inside** the netlist (`.op` / `.dc` / `.tran` / `.ac`) — `main.py` takes only the netlist path, an optional `-o/--output` dir (default `results/`), and `-v/--verbose`. Ready-to-run decks live in `examples/`.

### Write a netlist (HSPICE-style)

A netlist is component lines + `.model` cards + one analysis directive, terminated by `.end`. Node `0` is ground; `*` starts a comment. Example — single-pole RC low-pass `.ac` sweep (`examples/rc_lowpass_ac.sp`):

```spice
V1 in 0 DC=0 AC=1 0          * AC source: DC bias 0, |AC|=1V, phase 0 deg
R1 in out 1k
C1 out 0 159.155n
.ac dec 20 10 1e6            * 20 pts/decade, 10 Hz .. 1 MHz
.end
```

**MOSFETs — pick a compact model by LEVEL.** Model card `.model <name> {N|P}MOS (LEVEL=<72|73|74> [TECH=tsmc5 VT=lvt])`; instance line `M<id> <drain> <gate> <source> <bulk> <name> L=30n NFIN=10` (geometry `L`, `NFIN`, optional `TFIN`/`HFIN`/`FPITCH`).

- **LEVEL=72** — BSIM-CMG ground truth (`examples/bsimcmg_inverter_dc.sp`).
- **LEVEL=73** — DirectNet NN; `TECH`/`VT` select the per-tech checkpoint (`examples/nn_inverter_dc.sp`).
- **LEVEL=74** — BSIMAR (parked; no checkpoints on disk).

CMOS inverter DC sweep (LEVEL=72, `examples/bsimcmg_inverter_dc.sp`):

```spice
Vdd 1 0 1.0
Vin 2 0 0.0
Mp1 3 2 1 1 pmos1 L=30n NFIN=10     * drain gate source bulk
Mn1 3 2 0 0 nmos1 L=30n NFIN=10
.model nmos1 NMOS (LEVEL=72)
.model pmos1 PMOS (LEVEL=72)
.dc Vin 0 1.0 0.01
.end
```

### Analysis directives

- `.op` — DC operating point.
- `.dc <src> <start> <stop> <step>` — DC sweep (e.g. `.dc Vin 0 1.0 0.01`).
- `.tran <tstep> <tstop> [uic]` — transient; drive with a `PULSE v1 v2 td tr tf pw period` source (e.g. `.tran 10p 5n`). `uic` (use-initial-conditions, NGSPICE-style) starts the transient from the `.ic` state — pins `.ic` nodes during the OP so a high-impedance node (e.g. a switched-cap hold node) starts at its `.ic` value instead of its off-device leakage equilibrium. Default-off; non-`uic` decks are byte-identical.
- `.ac {dec|oct|lin} <N> <fstart> <fstop>` — small-signal; requires `AC=mag phase` on a source.
- `.ic V(node)=...` (hard initial condition) and `.include` are also supported.

### NN training (per-tech DirectNet, LEVEL=73)

Dedicated per-tech NMOS/PMOS DirectNet checkpoints for **TSMC5 / TSMC7 / TSMC12 / TSMC16**. `--tech-scope` ∈ `{tsmc5,tsmc7,tsmc12,tsmc16,universal}`; `--size` ∈ `{small,medium,large,xl}` (**production = `medium`**; `large`/`xl` are the capacity-study tiers — `large` is the over-fit sweet spot, `xl` the boundary — see CHANGELOG).

```bash
# 1. Generate per-tech data (one .npz per tech+device). --enable-inv-trip adds the
#    inverter-trip overlay; the grid sampler also carries the reverse-Vds corridor.
#    --tech ∈ {tsmc5,tsmc7,tsmc12,tsmc16,asap7,all}. Repeat per tech.
conda run -n pycircuitsim python external_compact_models/PyCMG/scripts/generate_nn_data.py \
    --device both --tech tsmc5 --enable-inv-trip --n-workers 8

# 2. Train a dedicated per-tech DirectNet. --tech-scope auto-sets --exclude-techs
#    (all other techs), --num-tech-codes (per-tech local vocab + UNKNOWN), the
#    default --data path (datasets/<scope>_<dev>.npz), and the save_prefix
#    `tsmc{X}_dn_<size>_<dev>` that the parser preempt cascade recognizes (Rule 16).
conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --size medium \
    --device-type {nmos,pmos} --tech-scope {tsmc5,tsmc7,tsmc12,tsmc16} --cuda --overwrite
```

**Full capacity sweep** — the S/M/L benchmark drives the whole matrix (4 techs × N/P × sizes) on one clean recipe: `scripts/benchmark_gen_data.sh` (datasets) → `scripts/benchmark_train_sml.sh` (the 24 checkpoints) → `scripts/benchmark_run_tests.sh` → `scripts/benchmark_collect.py` (`results/benchmark_sml/REPORT.md`). The older `scripts/train_per_tech_8cells.sh` is a TSMC5/7-only S+M convenience sweep.

**Checkpoints** (`external_compact_models/bsimar/checkpoints/`, each `*_best.pt` + `_norm.npz`):

- Per-tech DirectNet: `tsmc{5,7,12,16}_dn_{small,medium,large,xl}_{nmos,pmos}`. Each uses a SHRUNK local-vocab embedding (per-tech variant count + 1 UNKNOWN slot, e.g. TSMC5: 5, TSMC7: 4; Rule 16). Production is `medium`; the V6.4.7 campaign also parks specialized shipping variants (`v6_4_7_pivcor_*`, `v6_4_7_s12cor_*`, `*_c17`, …) — install/repoint per the shipping mix in CHANGELOG.
- Resolver cascade (`pycircuitsim/parser.py`): for a TSMC5/7/12/16 netlist the per-tech slot `tsmc{X}_dn_{medium,small,large,xl}` (medium-first) preempts the universal fallback chain (`refac_dn_* > v4_re_dn_universal > v4_dn_universal > bare`). The universal fallbacks are unreachable until someone retrains a universal stack (`refac_dn_*` / `v4_*` artifacts were deleted 2026-05-12). Resolutions log at parse time as `[NN-resolver] L73 <name> TECH=<x> VT=<y> -> <chk> (scope=<s>, tech_code=<c>)`. Override via `--exp-name` at train time, or `PYCIRCUITSIM_NN_CHECKPOINT_*` / `PYCIRCUITSIM_NN_CHECKPOINT_DN_{NMOS,PMOS}` env vars at runtime (the latter is read first, before the medium-first preempt — used by the benchmark to pin a capacity tier).

**Netlist usage:** `.model nmos_nn NMOS (LEVEL=73 TECH=tsmc5 VT=lvt)` with `L=16n NFIN=10`. Parser auto-resolves the per-tech checkpoint and the local-vocab tech_code via `bsimar.config.local_variant_code(scope, tech, variant)`.

### Output files

Results in `results/<circuit_name>/<analysis_type>/`: an HSPICE-like `*_simulation.lis` log, the analysis data (`*_dc_sweep.csv` / `*_transient.csv` / `*_ac_sweep.csv`), and a Matplotlib plot (VTC / waveform / Bode).

## Testing & Verification

All tests require `conda activate pycircuitsim`.

**Shared infra:** `tests/common/{base,bsimcmg_dc,bsimcmg_tran,nn,nn_sweep}.py` and `tests/references/`.

**BSIM-CMG DC:** L1 `verify_bsimcmg_dc.py` (2) · L2 `verify_bsimcmg_dc_comprehensive.py` (67) · L3 `verify_multi_tech_dc.py` (44).
**BSIM-CMG Transient:** L1 `verify_bsimcmg_tran.py` (1) · L2 `verify_bsimcmg_tran_comprehensive.py` (37) · L3 `verify_multi_tech_tran.py` (72).
**NN device + inverter gate:** `verify_nn_dc_tran.py --tech TSMC5,TSMC7,TSMC12,TSMC16 [--inverter-only]` — per-tech NMOS/PMOS single-OP / DC-sweep / inverter VTC + transient (LEVEL=73) vs NGSPICE BSIM-CMG; `--inverter-only` restricts to the inverter VTC/transient gate. Production per-tech baseline: inverter gate 8/8, DC 55/55, tran 64/64. The bare `--tech` default (`TECH_ORDER`) also lists ASAP7 variants — out of scope, no checkpoints (Rule 14).
**NN parametric harness (V6.3.2):** the PyCMG L3 parametric sweeps ported to DirectNet (LEVEL=73) via `tests/common/nn_sweep.py`. `verify_nn_multi_tech_dc.py` — single-device NMOS/PMOS Id-Vgs over L/NFIN/VT (55 configs, 4 TSMC techs). `verify_nn_multi_tech_tran.py` — inverter VTC + transient over P/N ratio, VDD, Cload, input slew, pulse width. Baseline-gated: the parametric sweep runs only for techs that pass baseline. Geometry/VT/VDD ride on `dataclasses.replace(TestTechConfig)`; only the inverter-transient circuit knobs needed a (behaviour-preserving) refactor of `verify_nn_dc_tran.py` (`InvCircuitParams`). Run with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` — the NN inverter VTC has ~±1% NRMSE run-to-run scatter (high-gain trip point; harness pins `torch` to 1 thread).
**Complex circuits:** `verify_complex_{ring_osc,opamp,sram_snm,switchcap}.py` + `tests/common/complex.py` vs NGSPICE BSIM-CMG (4 circuits × 4 techs = 16 gates + the authoritative `force_ic` single-point ship gate). Parametric sweep mirror: `verify_complex_{opamp,ringosc,switchcap,sram}_sweep.py` + `tests/common/complex_sweep.py` (tech / VT / geometry / VDD / stimulus; baseline-gated, sha256-pinned). CPU-pinned, repo NGSPICE.
**AC (LEVEL=72):** `verify_ac.py` — L1 passive RC (vs NGSPICE `.ac` + analytic), L2 BSIM-CMG common-source amp (vs NGSPICE `.ac`, identical OSDI model). 2/2 PASS, ~machine precision. CPU-pinned; needs `NGSPICE_BIN` (repo `tools/ngspice-45.2/bin/ngspice` on this machine).
**NN AC (LEVEL=73):** `verify_nn_ac.py --tech TSMC<XX> [--device nmos,pmos]` — per-checkpoint NMOS/PMOS common-source amp vs NGSPICE BSIM-CMG (no load cap → device caps set the pole; each side at its own fresh-solved mid-rail OP; gate = gain0 ≤1.5 dB / f3db ratio ∈[0.7,1.43] / mag NRMSE ≤10%; phase = diagnostic). `verify_complex_opamp_ac.py --tech TSMC<XX>` — two-stage Miller opamp open-loop (gate = DC-gain ≤3 dB / GBW ratio ∈[0.6,1.67] / PM ≤15°). Shared infra `tests/common/complex_ac.py`. Both run in the S/M/L benchmark (`scripts/benchmark_run_tests.sh` → `benchmark_collect.py` "AC small-signal accuracy" section). CPU-pinned, `NGSPICE_BIN` + `PYCIRCUITSIM_NN_CHECKPOINT_DN_{NMOS,PMOS}`. RO + SRAM excluded (no stable amplifying OP for `.ac`).
**Other:** `verify_bsimcmg_op.py` (OP <0.02% vs NGSPICE); lifted-source canary `verify_nn_lifted_source_dc.py` (NRMSE ≤10%, guards Rule 2).

Quick sanity:

```bash
python tests/verify_bsimcmg_op.py && python tests/verify_bsimcmg_dc.py && python tests/verify_bsimcmg_tran.py
# AC (set NGSPICE_BIN to the repo ngspice if /usr/local is absent):
NGSPICE_BIN="$PWD/tools/ngspice-45.2/bin/ngspice" python tests/verify_ac.py
```

---

## Development Guidelines

**Coding standards:** type hints on all signatures; clear names (`v_gate`, `i_drain`); docstrings for complex algorithms; voltage clamping Vgs±5V, Vds±10V.

**Separation principle:**

- `solver.py` builds MNA + executes NR (no device equations).
- `models/` calculates current/conductances (no matrix ops).
- `simulation.py` orchestrates (parse → solve → visualize).
- All devices inherit from `Component`.

**Key numerical techniques:**

- Sparse MNA solver: `lil_matrix` assembly, CSR + `spsolve` solve. O(n) memory, O(n·log n) solve.
- SPICE-standard convergence: `|ΔV| < VNTOL + RELTOL × max(|V_old|, |V_new|)` (RELTOL=1e-4, VNTOL=1e-7).
- GMIN (1e-12 S) prevents singular matrices. DC GMIN stepping opt-in via `use_gmin_stepping=True`: 2-level schedule [1e-8, 1e-12]. NN circuits use `_solve_dc_with_retry` (fast path first, GMIN retry on `_last_solve_converged=False`). BSIM-CMG never enters the retry branch.
- BE → Trap → BDF-2: BE step 1, Trap step 2+, BDF-2 auto on stiffness (NR>20 iters); one-way switch.
- Source stepping (20 steps); supply-relative adaptive damping with stuck-counter.
- DC oscillation detection: 5-snapshot ring, accepts averaged solution if variance < 10× tolerance.
- Hard `.ic` mode (`force_ic=True`): stamps `.ic` nodes as temporary V-source constraints, re-solves unconstrained. Required for SRAM latches.
- LTE sub-stepping (opt-in via `max_substeps`, default 1=disabled).

**Entry points:** CLI `main.py`; API `pycircuitsim.simulation.run_simulation()`; module exports (Circuit, Parser, Visualizer, run_simulation).

**Environment & tools:** conda env `pycircuitsim` at `/home/shenshan/.conda/envs/pycircuitsim`; PyTorch 2.10.0 (CPU); OpenVAF `/usr/local/bin/openvaf`; NGSPICE `/usr/local/ngspice-45.2/bin/ngspice`.

---

## Critical Design Rules

These rules were learned from bugs. Violating them causes NR divergence or wrong results.

### Sign Convention for Device Models

1. **Use terminal current `id`, NOT channel `ids`** — `ids = id - is ≈ 2*id` (2× error).
2. **NMOS** `calculate_current()` returns `-result["id"]`; **PMOS** returns `result["id"]` (positive = leaving drain).
3. **Solver stamping** uses unified "current leaving drain" convention. All VCCS conductances (g_ds, g_m, g_mb) need full 4-entry stamps (drain,ctrl+; drain,ctrl-; source,ctrl-; source,ctrl+). An incomplete stamp breaks Jacobian symmetry.
   ```python
   i_leaving = -i_ds if is_pmos else i_ds
   i_eq = i_leaving - g_ds * v_ds - g_m * v_gs - g_mb * v_bs
   rhs[d_idx] -= i_eq    # same for NMOS and PMOS
   rhs[s_idx] += i_eq
   ```
4. **gds floor** for stamping: `max(gds, 1e-12)`. Never `abs(gds)` — it flips large-negative to large-positive and diverges NR. Preserve gm/gmb signs.
5. **Update `_is_mosfet()`** in `solver.py` when adding new device types.
6. **Test both NMOS and PMOS** vs NGSPICE: single OP, DC sweep, inverter VTC, inverter transient.

### NN Model Rules (LEVEL=73 DirectNet primary; LEVEL=74 BSIMAR parked)

Both LEVEL=73 (single-shot MLP, primary) and LEVEL=74 (autoregressive Transformer, parked — Rule 15) share the data pipeline and inference rules, and use `nn.Embedding` for tech-code identity (7-dim input: Vgs, Vds, Vbs, NFIN, L, T, tech_code). Rules 9–10 are parked BSIMAR-specific structure — resurrect from CHANGELOG / git if needed. Rule numbers are internal to this document and are no longer cited in code.

1. Deleted by user.
2. **Source-relative frame for BOTH device types** — shift all terminal voltages by -Vs before NN eval (`v_d_nn = v_d - v_s`, Vs ≡ 0). Training uses Vs=0; shift invariance makes this exact. Until V6.4.7 only PMOS was shifted — lifted-source NMOS (opamp tail pair, SC pass device, SRAM access) saw phantom Vgs/Vds with Vbs=0; the lifted-source canary `tests/verify_nn_lifted_source_dc.py` (NRMSE ≤10 %) guards this permanently.
3. Deleted by user.
4. Deleted by user.
5. Deleted by user.
6. **ASAP7 modelcard name mapping** — parser auto-maps netlist names to `nmos_rvt` / `pmos_rvt`.
7. **PyCMG integration** — `bsimar/config.py` re-exports `NNTechConfig`, `TECH_CONFIGS`, `TECH_CODE_MAP`, `OUTPUT_COLUMNS` from `pycmg.nn_config`. Backward-compat alias `TechConfig = NNTechConfig`. Training VDD may differ from PyCMG's runtime VDD; check `NNTechConfig.VDD` per tech.
8. **Data validation** — `eval_single_point` rejects NaN/Inf and `|id| > 1A`. PyCMG `eval_dc` raises `RuntimeError` on internal-node convergence failure. NFIN=1 is excluded from training data: although `DEFAULT_NFIN_VALUES` lists it, unstable `(variant, NFIN=1)` bins fail OSDI convergence and are dropped per-bin during generation, so NFIN≥2 is what actually trains.
9. **(parked, LEVEL=74)** BSIMAR output uses `BSIMAR_COLUMN_ORDER`, not `OUTPUT_COLUMN_ORDER` — see CHANGELOG / `mosfet_bsimar.py` if resurrected.
10. **(parked, LEVEL=74)** BSIMAR parallel cap head + 8-step AR loop (`parallel_caps`, `grouped_inputs`, structural) — see CHANGELOG if resurrected.
11. **Unified CLI** — `python -m bsimar.cli.train --model direct --size {small,medium,large,xl} --device-type {nmos,pmos} --tech-scope {tsmc5,tsmc7,universal} ...` (xl = 512×8 ~2.13M p, over-fit-boundary tier; production stays medium). With `--tech-scope tsmc{5,7}` the default save_prefix is `tsmc{X}_dn_<size>_<device>` (recognized by the parser preempt cascade). Same `.npz` from PyCMG; checkpoints under `external_compact_models/bsimar/checkpoints/`. Flags (all default-off / behavior-preserving): `--swa-mode {none,ema,swa}` + `--ema-decay`; `--apply-filter {on,off}` + `--class-weights`; `--enable-subvt-off`; the optional loss terms `--sobolev` / `--subthresh` / `--charge-sobolev`; and the EKV backbone `--ekv-core` / `--ekv-alpha` / `--ekv-hidden`.
12. **Charge conservation** — simulator always computes `qs = -(qg + qd + qb)` analytically, even for 13-output models that directly predict `qs`. Guarantees Kirchhoff conservation at every transient timestep.
13. Always report MRE (%), R^2, NRMSE, Max error (mV) metrics per tech.
14. **Exclude ASAP7** — out of scope at this stage (no checkpoints; see Overview).
15. **DirectNet only** — do NOT train/eval the LEVEL=74 BSIMAR Transformer (parked).
16. **Per-tech models use a LOCAL embedding vocab.** When `--tech-scope` is `tsmc5` or `tsmc7`, the dataset loader remaps universal tech codes to a 0-indexed per-tech vocab and the trainer instantiates `DirectNet(num_tech_codes=N, unknown_code_id=N-1)`, where N = variants+1 (TSMC5: 5, TSMC7: 4). The training-time `p_unknown` dropout writes `unknown_code_id` into the embedding, so a misaligned UNKNOWN id → CUDA assert. **Derive `unknown_code_id` from `num_tech_codes`; do NOT hardcode the universal value (17).** Parser uses `bsimar.config.local_variant_code(scope, tech, variant)` to remap at inference; the scope is read from the resolved checkpoint stem (`tsmc{5,7}_dn_*` → local; everything else → universal).

> **Load-bearing code (do NOT delete):** the V6.4.2 `_MonotoneVgResidual` + `--monotonic` path (`bsimar/{cli/train,models/direct_net,training/trainer}.py`, `pycircuitsim/models/mosfet_directnet.py`) must stay committed — on-disk checkpoints carry `mono.*` state_dict keys and fail to load without it. Stock checkpoints route `mono=None` (no inference change). The default-off EKV backbone (`_EKVCore`, `core.*` keys) and the Sobolev/subthreshold/charge-Sobolev loss terms are likewise kept recoverable; all leave stock checkpoints byte-identical.

---

## Important Paths

- **PyCMG submodule:** `external_compact_models/PyCMG/` (21 device variants).
- **OSDI binary:** `build/osdi/bsimcmg.osdi` (PyCMG-relative).
- **Modelcards:** `modelcards/` (PyCMG-relative); ASAP7 `*.pm` committed; TSMC raw PDK `cln*.l` is gitignored/IP-protected — naive modelcards regenerated on-the-fly via `pycmg.tech.resolve_modelcard` into `build/modelcards/`. Never commit `modelcards.tar.gz` (bundles the IP `cln*.l`).
- **Results output:** `results/<circuit_name>/<analysis_type>/`.
- **Test results:** `tests/verify_*_results/` (generated, not tracked).
- **Sprint history:** `docs/CHANGELOG.md`. **Note (2026-06-15 cleanup):** the pre-V6.4.7 plan files and the old iteration result dirs (`results/{v6_4_4_iter2,v6_4_5,v6_4_6}/`, `results/v4_*`/`v5_*`) were pruned; the durable dead-end records remain in this CHANGELOG and CLAUDE.md, so path references to those removed gate files in older notes are intentionally dangling.

## Other Tips

* **Start every complex task in plan mode** — pour energy into the plan for 1-shot implementation. Re-plan the moment something goes sideways; enter plan mode for verification steps too.
* If the plan has several solutions or stages, implement them in sequence. Use git commit first before you modify anything, keep the useful one that make progress and incorperate it. Otherwise, revert the solutions that were proven to be no help with git reset.
* **Update CLAUDE.md before every git commit**.
* Whenever there is a version update, update the `docs/CHANGELOG.md`.
* Always record the dead end proposal (the one being reverted), they are as important as the successful ones.
* **Never be lazy** — never simplify code or skip tests. **NEVER** use simplified equations or self-defined CMG models as reference; ALWAYS use simulation results as ground truth.
* **Use subagents** — second agent for staff-engineer plan review; multiple subagents on separate branches to try multiple solutions; roll back to main when a subagent hits a dead end.
* Use **GPUs in parallel** when you train NN models.
