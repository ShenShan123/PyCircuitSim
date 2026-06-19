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
├── solver.py           # MNA matrix + Newton-Raphson
├── logger.py           # HSPICE-like .lis output
├── visualizer.py       # Matplotlib plotting
└── models/
    ├── base.py               # Component abstract base
    ├── passive.py            # R, C, V, I sources (PULSE)
    ├── mosfet_cmg.py         # BSIM-CMG (LEVEL=72) via PyCMG
    ├── mosfet_nn.py          # Shared _MOSFETNNBase (LEVEL=73/74) — voltage prep, autograd, Vds correction
    ├── mosfet_directnet.py   # DirectNet (LEVEL=73, primary)
    └── mosfet_bsimar.py      # BSIMAR Transformer (LEVEL=74, parked — see Rule 18)

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

## Supported Features

* **Devices:** R, C; NMOS/PMOS LEVEL=72 (BSIM-CMG, ground truth), LEVEL=73 (DirectNet, primary NN), LEVEL=74 (BSIMAR, parked); DC voltage/current sources, PULSE.
* **Analyses:** `.op`, `.dc`, `.tran`.
* **Directives:** `.model` (LEVEL=72/73/74), `.include`, `.ic`.
* Legacy LEVEL=1 (Shichman-Hodges) removed.

## Validation

Inverter circuit must PASS Transient Analysis against NGSPICE ground truth within reasonable numerical tolerance. Never use simplified/self-defined equations as reference.

## Status

**Current ship: V6.4.7** (branch `feat/v6.4.7`) — per-tech complex-circuit mix at **14/16 gates + `force_ic` 8/8**; the full success criterion (headline > 11/16 AND force_ic 8/8) is **MET** (+3 vs the S8 baseline, +6 vs V6.4.4 canonical 8/16).

**In progress: V6.4.8** (branch `feat/v6.4.8`, from `d9c3d6b`) — value-surface accuracy campaign (plan `docs/plans/2026-06-17-directnet-v6.4.8-accuracy.md`). **S0** floor-k diagnostic = KILL (basin-hopping, not an accuracy lever; env-gate `PYCIRCUITSIM_GDS_FLOOR_K` kept default-off). **S1** `--size large` (384×6) = KILL — **capacity is not the bind**: the larger net fits the value surface better (val ~3e-4) but COLLAPSES the tsmc7 opamp (3/4 seeds → gain 0) and regresses RO; tsmc5 SC stays flat at ~11.3 %. **S1 re-run (2026-06-19/20) reproduces the KILL**: original large ckpts under the new continuation-first solver still fail (s7→0, s17→361), and a fresh re-train (`tsmc7_dn_lgB_s*`) gives opamp 4/4 FAIL (s17→361.4 byte-identical to the original) — capacity KILL robust; not a solver/training-noise artifact. **S2** continuation-first DC sweep = **KEEP**: `run_dc_sweep` now solves warm-started points (`point>0`, NN circuits) directly from the neighbour with source-stepping disabled (GMIN retry restores it as fallback; BSIM-CMG path byte-identical). **tsmc7 opamp FLIPS 10.78% FAIL → 8.63% PASS, deterministic across OMP∈{1,2,4}**; tsmc16 opamp now NG-locus-faithful (NRMSE 69.5→17.0, trip −146→−10mV); no regression on tsmc7/tsmc16 cells (ring/sram/SC/inverter/DC-55). The plan's basin-de-fragilization hypothesis is REFUTED (continuation doesn't change the 0/197/383 seed split — value-surface-owned), but path-preservation is a real win. ⚠ tsmc7's pass is a gain-gate pass on a still-unfaithful locus (trip −144mV) → S3 still motivated. Remaining: **S3** (EKV-like analytic backbone — primary lever) + **S4** (install tsmc5/tsmc12 baselines, re-verify S2 there + lifted-source 12/12, compose+promote). Gate-running methodology: **CPU only** (`CUDA_VISIBLE_DEVICES=""`, `OMP=MKL=1`, repo `tools/ngspice-45.2`) reproduces S19's 10.78 %; CUDA mis-lands the fragile opamp basin (47 %). ⚠ tsmc5/tsmc12 V6.4.4 baselines absent on this machine (install from S19 sha256 manifest before the S4 board). Headline **14/16 → 15/16 conditional on tsmc5/tsmc12 holding S2 in S4**.

**Shipping checkpoint mix** (resolver/install names in `results/v6_4_7/S19_promotion.md`):
- tsmc7 = `v6_4_7_pivcor_w2_s7_tsmc7`
- tsmc16 = `v6_4_7_s12cor_w3_s17_tsmc16`
- tsmc5 + tsmc12 = V6.4.4 baseline `tsmc{5,12}_dn_medium` (unchanged; **absent on the campaign machine** — install from the sha256 manifest, see gate file)

> Sprint narrative, dead-ends, and durable findings live in **`docs/CHANGELOG.md` "V6.4.7"** + `MEMORY.md`; per-step gate files in `results/v6_4_7/`. Not duplicated here.

### What we've built

- **BSIM-CMG (LEVEL=72) ground truth** — all 5 techs (ASAP7, TSMC5/7/12/16), DC <0.1 % / transient ~0.2 % NRMSE vs NGSPICE.
- **DirectNet (LEVEL=73, primary)** — per-tech NMOS/PMOS checkpoints (size `medium`) for TSMC5/7/12/16. Inverter gate 8/8, DC 55/55, tran 64/64, lifted-source canary 12/12.
- **Complex-circuit harness** — `tests/verify_complex_{ring_osc,opamp,sram_snm,switchcap}.py` + `tests/common/complex.py` vs NGSPICE BSIM-CMG: 4 circuits × 4 techs = 16 gates + force_ic.
- **Training infra (default-off, recoverable):** EMA/SWA (`--swa-mode`), the trajectory-corridor pipeline (P5), `SobolevIdLoss`, `SubthresholdIdLoss`, the regen-v2 data pipeline (generator floor fix, decade-occupancy gate, deriv-fidelity scorer).
- **Solver:** sparse MNA, source + 2-level GMIN stepping, BE→Trap→BDF-2, LTE sub-stepping, oscillation detection, hard `.ic` mode.

### What's next (open known-issues — structural-change scope, not cheap-lever)

- **tsmc5 switchcap 12.14 %** over-conduction — a moderate/strong-region forward-conduction limit.
- **tsmc7 opamp** gain over-prediction — S2 continuation-first crossed the gain gate (10.78%→8.63% PASS, deterministic) but the Vout LOCUS stays unfaithful (trip −144mV, NRMSE 68%); the systematic over-gain bias is value-surface-rooted and only a faithful structural fix (S3 EKV backbone) closes it.
- **inverter VTC MaxErr ≤25 mV** never met (V6.4 at 29.7–62 mV) — needs a larger seed sweep or network constraints (monotonicity / spectral-norm).
- **ASAP7 + LEVEL=74 BSIMAR out of scope** — no checkpoints (universal artifacts deleted 2026-05-12); each needs a dedicated retrain.

> **Load-bearing code:** the V6.4.2 Phase-7a `_MonotoneVgResidual` + `--monotonic` path (`bsimar/{cli/train,models/direct_net,training/trainer}.py`, `pycircuitsim/models/mosfet_directnet.py`) must stay committed — on-disk checkpoints carry `mono.*` state_dict keys and fail to load without it. Stock checkpoints route `mono=None` (no inference change).

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

### Basic simulation

Create a `.sp` netlist (examples in `examples/`). BSIM-CMG geometric params: `L`, `NFIN`, optional `TFIN`/`HFIN`/`FPITCH`.

### NN training (V6.2 — per-tech dedicated)

```bash
# Generate per-tech data. --enable-inv-trip overlay covers all 4 TSMC techs
# inside pycmg/nn_generate.py. V6.3 re-centered it on VDD/2; V6.3.1 dropped the
# ±0.25·VDD Vbs sweep, so the overlay is ~3.5% of rows (was ~9.8% pre-V6.3.1).
# V6.3 also added the reverse_vds corridor class (~7.5% of rows, always on).
conda run -n pycircuitsim python external_compact_models/PyCMG/scripts/generate_nn_data.py \
    --device both --tech tsmc5 --enable-inv-trip --n-workers 8
conda run -n pycircuitsim python external_compact_models/PyCMG/scripts/generate_nn_data.py \
    --device both --tech tsmc7 --enable-inv-trip --n-workers 8

# Train dedicated per-tech DirectNet. --tech-scope auto-sets:
#   --exclude-techs (all other techs), --num-tech-codes (per-tech vocab + UNKNOWN),
#   default --data path (datasets/<scope>_<dev>.npz), and the save_prefix
#   (`tsmc{5,7}_dn_<size>_<dev>`) recognized by the parser preempt cascade.
conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --size {small,medium} \
    --device-type {nmos,pmos} --tech-scope {tsmc5,tsmc7} --cuda --overwrite

# Convenience: full 8-cell sweep (S+M × NMOS/PMOS × TSMC5/TSMC7) at GPU 2.
bash scripts/train_per_tech_8cells.sh
```

**Checkpoints** (in `external_compact_models/bsimar/checkpoints/`):

- V6.2 DirectNet per-tech: `tsmc{5,7}_dn_{small,medium}_{nmos,pmos}_best.pt` + `_norm.npz`. Embedding vocab shrunk to per-tech variant count + 1 UNKNOWN slot (TSMC5: 5, TSMC7: 4). Production size is `medium`. No other checkpoints are present — universal `refac_dn_*` / `v4_*` artifacts were deleted on 2026-05-12.
- Resolver cascade (`pycircuitsim/parser.py`): for TSMC5/TSMC7 netlists, the per-tech slot `tsmc{X}_dn_{medium,small,large}` preempts the universal fallback chain (`refac_dn_* > v4_re_dn_universal > v4_dn_universal`). At V6.2 only `tsmc{5,7}_dn_medium_{nmos,pmos}` exist on disk; the universal fallbacks are unreachable until someone retrains a universal stack. Resolutions are logged at parse time as `[NN-resolver] L73 <name> TECH=<x> VT=<y> -> <chk> (scope=<s>, tech_code=<c>)`. Override via `--exp-name` at train time or `PYCIRCUITSIM_NN_CHECKPOINT_*` env vars at runtime.

**Netlist usage:** `.model nmos_nn NMOS (LEVEL=73 TECH=tsmc5 VT=lvt)` with `L=16n NFIN=10`. Parser auto-resolves the per-tech checkpoint and the local-vocab tech_code via `bsimar.config.local_variant_code(scope, tech, variant)`.

### Output files

Results in `results/<circuit_name>/<analysis_type>/`: `*_simulation.lis`, `*_dc_sweep.csv` / `*_transient.csv`.

## Testing & Verification

All tests require `conda activate pycircuitsim`.

**Shared infra:** `tests/common/{base,bsimcmg_dc,bsimcmg_tran,nn,nn_sweep}.py` and `tests/references/`.

**BSIM-CMG DC:** L1 `verify_bsimcmg_dc.py` (2) · L2 `verify_bsimcmg_dc_comprehensive.py` (67) · L3 `verify_multi_tech_dc.py` (44).
**BSIM-CMG Transient:** L1 `verify_bsimcmg_tran.py` (1) · L2 `verify_bsimcmg_tran_comprehensive.py` (37) · L3 `verify_multi_tech_tran.py` (72).
**NN V6.2 gate:** `verify_nn_dc_tran.py --tech TSMC5,TSMC7 --inverter-only` (12/12 PASS on the full TSMC5/7 sweep without `--inverter-only`).
**NN parametric harness (V6.3.2):** the PyCMG L3 parametric sweeps ported to DirectNet (LEVEL=73) via `tests/common/nn_sweep.py`. `verify_nn_multi_tech_dc.py` — single-device NMOS/PMOS Id-Vgs over L/NFIN/VT (55 configs, 4 TSMC techs). `verify_nn_multi_tech_tran.py` — inverter VTC + transient over P/N ratio, VDD, Cload, input slew, pulse width. Baseline-gated: the parametric sweep runs only for techs that pass baseline. Geometry/VT/VDD ride on `dataclasses.replace(TestTechConfig)`; only the inverter-transient circuit knobs needed a (behaviour-preserving) refactor of `verify_nn_dc_tran.py` (`InvCircuitParams`). Run with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` — the NN inverter VTC has ~±1% NRMSE run-to-run scatter (high-gain trip point; harness pins `torch` to 1 thread).
**Other:** `verify_bsimcmg_op.py` (OP <0.02% vs NGSPICE).

Quick sanity:

```bash
python tests/verify_bsimcmg_op.py && python tests/verify_bsimcmg_dc.py && python tests/verify_bsimcmg_tran.py
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

Both LEVEL=73 (single-shot MLP, primary) and LEVEL=74 (autoregressive Transformer, parked — Rule 18) share the data pipeline and inference rules, and use `nn.Embedding` for tech-code identity (7-dim input: Vgs, Vds, Vbs, NFIN, L, T, tech_code). Rules 11–12 are parked BSIMAR-specific structure — resurrect from CHANGELOG / git if needed.

1. **Jacobian consistency is mandatory** — gm/gds/gmb MUST be `torch.autograd.grad(id, V)`, never independent predictions. Holds for LEVEL=73 and LEVEL=74.
2. **Source-relative frame for BOTH device types** — shift all terminal voltages by -Vs before NN eval (`v_d_nn = v_d - v_s`, Vs ≡ 0). Training uses Vs=0; shift invariance makes this exact. Until V6.4.7 only PMOS was shifted — lifted-source NMOS (opamp tail pair, SC pass device, SRAM access) saw phantom Vgs/Vds with Vbs=0; the lifted-source canary `tests/verify_nn_lifted_source_dc.py` (NRMSE ≤10 %) guards this permanently.
3. **Training range covers NR overshoot** — margin ±VDD beyond operating range, not ±0.1V.
4. **Smooth voltage clamping** — softplus-based, NOT `torch.clamp`. Hard clamp creates zero-gradient cliffs that stall NR. Margin = 5% of per-dim training range.
5. **Physics-based gds floor** — `gds = max(gds, |id|*0.5, 1e-12)`. NN autograd gds ≈ 0 in saturation; without the floor inverter gain → ∞ and NR diverges. At FinFET 16nm BSIM-CMG λ=0.3-1.2 V⁻¹. Floor only affects the NR Jacobian, not the converged solution.
6. *(Removed — see git history.)*
7. **ASAP7 modelcard name mapping** — parser auto-maps netlist names to `nmos_rvt` / `pmos_rvt`.
8. **PyCMG integration** — `bsimar/config.py` re-exports `NNTechConfig`, `TECH_CONFIGS`, `TECH_CODE_MAP`, `OUTPUT_COLUMNS` from `pycmg.nn_config`. Backward-compat alias `TechConfig = NNTechConfig`. Training VDD may differ from PyCMG's runtime VDD; check `NNTechConfig.VDD` per tech.
9. **Data validation** — `eval_single_point` rejects NaN/Inf and `|id| > 1A`. PyCMG `eval_dc` raises `RuntimeError` on internal-node convergence failure. NFIN=1 is excluded from training data: although `DEFAULT_NFIN_VALUES` lists it, unstable `(variant, NFIN=1)` bins fail OSDI convergence and are dropped per-bin during generation, so NFIN≥2 is what actually trains.
10. *(Removed — see git history.)*
11. **(parked, LEVEL=74)** BSIMAR output uses `BSIMAR_COLUMN_ORDER`, not `OUTPUT_COLUMN_ORDER` — see CHANGELOG / `mosfet_bsimar.py` if resurrected.
12. **(parked, LEVEL=74)** BSIMAR parallel cap head + 8-step AR loop (`parallel_caps`, `grouped_inputs`, structural) — see CHANGELOG if resurrected.
13. **Unified CLI** — `python -m bsimar.cli.train --model direct --size {small,medium,large} --device-type {nmos,pmos} --tech-scope {tsmc5,tsmc7,universal} ...`. With `--tech-scope tsmc{5,7}` the default save_prefix is `tsmc{X}_dn_<size>_<device>` (recognized by the parser preempt cascade). Same `.npz` from PyCMG; checkpoints under `external_compact_models/bsimar/checkpoints/`. V6.4.7 flags (all default-off / behavior-preserving): `--swa-mode {none,ema,swa}` + `--ema-decay`; `--apply-filter {on,off}` + `--class-weights`; `--enable-subvt-off`; and the optional loss terms `--sobolev` / `--subthresh`.
14. **Charge conservation** — simulator always computes `qs = -(qg + qd + qb)` analytically, even for 13-output models that directly predict `qs`. Guarantees Kirchhoff conservation at every transient timestep.
15. *(Removed — Vds-correction behavior is self-documented in `_apply_vds_correction`, `pycircuitsim/models/mosfet_nn.py`; see git history.)*

16. Always report MRE (%), R^2, NRMSE, Max error (mV) metrics per tech.
17. **Exclude ASAP7** — out of scope at this stage (see Status → What's next).
18. **DirectNet only** — do NOT train/eval the LEVEL=74 BSIMAR Transformer (parked; see Status → What's next).
19. **Per-tech models use a LOCAL embedding vocab.** When `--tech-scope` is `tsmc5` or `tsmc7`, the dataset loader remaps universal tech codes to a 0-indexed per-tech vocab and the trainer instantiates `DirectNet(num_tech_codes=N, unknown_code_id=N-1)`, where N = variants+1 (TSMC5: 5, TSMC7: 4). The training-time `p_unknown` dropout writes `unknown_code_id` into the embedding, so a misaligned UNKNOWN id → CUDA assert. **Derive `unknown_code_id` from `num_tech_codes`; do NOT hardcode the universal value (17).** Parser uses `bsimar.config.local_variant_code(scope, tech, variant)` to remap at inference; the scope is read from the resolved checkpoint stem (`tsmc{5,7}_dn_*` → local; everything else → universal).
20. *(Removed — see git history.)*

---

## References

- **ngspice** — physics equation verification.
- **Xyce** — architecture patterns for device/solver separation.
- **BSIM-CMG** — FinFET compact model (LEVEL=72), via PyCMG.
- **ASAP7** — https://github.com/The-OpenROAD-Project/asap7_pdk_r1p7.git
- **PyCMG** — https://github.com/ShenShan123/PyCMG.git

## Important Paths

- **PyCMG submodule:** `external_compact_models/PyCMG/` (21 device variants).
- **OSDI binary:** `build/osdi/bsimcmg.osdi` (PyCMG-relative).
- **Modelcards:** `modelcards/` (PyCMG-relative); ASAP7 `*.pm` committed; TSMC raw PDK `cln*.l` is gitignored/IP-protected — naive modelcards regenerated on-the-fly via `pycmg.tech.resolve_modelcard` into `build/modelcards/`.
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