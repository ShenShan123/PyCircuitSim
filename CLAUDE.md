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
    ├── mosfet_directnet.py   # DirectNet V6 (LEVEL=73) — tech-code embedding, _MOSFETNNBase + NMOS_NN/PMOS_NN
    └── mosfet_bsimar.py      # BSIMAR V6 (LEVEL=74) — tech-code embedding, _MOSFETBSIMARBase + NMOS_BSIMAR/PMOS_BSIMAR

external_compact_models/
├── bsimar/             # Unified NN compact model package (importable as `bsimar`)
│   ├── config.py                   # NNTechConfig + Direct/TransformerConfig + TECH_CODE_MAP
│   ├── data/{normalize,dataset,analyze}.py
│   ├── models/{direct_net,transformer,tsmc5_residual}.py    # nn.Embedding tech-code; tsmc5_residual = V6 Tier M2 residual head on frozen DN backbone
│   ├── losses/bni_mae.py           # MAELoss + per-target LDS weights
│   ├── training/{trainer,tsmc5_residual_train}.py
│   ├── eval/{metrics,visualization}.py
│   ├── cli/train.py                # `python -m bsimar.cli.train --model {direct,transformer} ...`
│   └── checkpoints/                # *.pt + _norm.npz + _config.npz (gitignored)
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

* **Devices:** R, C; NMOS/PMOS LEVEL=72 (BSIM-CMG, ground truth), LEVEL=73 (DirectNet, baseline), LEVEL=74 (BSIMAR, primary); DC voltage/current sources, PULSE.
* **Analyses:** `.op`, `.dc`, `.tran`.
* **Directives:** `.model` (LEVEL=72/73/74), `.include`, `.ic`.
* Legacy LEVEL=1 (Shichman-Hodges) removed.

## Validation

Inverter circuit must PASS Transient Analysis against NGSPICE ground truth within reasonable numerical tolerance. Never use simplified/self-defined equations as reference.

## Status

Current shipping revision is **V6** (7-dim + tech-code embedding architecture, asinh+zscore output norm for both DirectNet and Transformer, B1 hybrid-grid training data, `refac_*` checkpoint prefix). V4/V5 history in `docs/CHANGELOG.md`.

- **BSIM-CMG (LEVEL=72):** all 5 techs (ASAP7, TSMC5/7/12/16), DC <0.1% NRMSE, transient ~0.20% NRMSE vs NGSPICE.
- **DirectNet V6 (LEVEL=73, primary):** universal NMOS/PMOS checkpoints under `refac_dn_{small,medium,large}_*` (production size: `medium`); `e1`/`e2`/`e3` loss-preset variants on disk. Legacy `v4_*` checkpoints loadable via resolver fallback.
- **BSIMAR V6 (LEVEL=74, deprioritized per Rule 18):** `refac_tf_small_*` checkpoints exist; medium/large not yet trained. Inverter transient 6-12% NRMSE (legacy V4-re probe).
- **Test infrastructure:** 3-level DC + transient suites (BSIM-CMG: 2+67+44 DC, 1+37+72 tran; NN V6: ~6 DC + ~4 tran in `verify_nn_dc.py`/`verify_nn_tran_v4.py` plus all-tech sweeps via `verify_nn_dc_tran.py`).
- **Solver upgrades shipped:** sparse MNA (lil→CSR+spsolve), 2-level GMIN stepping [1e-8, 1e-12] with retry, BE→Trap→BDF-2, LTE sub-stepping, oscillation detection, hard `.ic` mode.
- **ASAP7 exclusion:** ASAP7 tech codes (18-21) exceed the trained embedding vocabulary (18 codes). Running ASAP7 with universal checkpoints crashes the embedding. Requires separate fine-tuning.

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

### NN training (V6)

```bash
# Generate universal data (954 geometry combos across 5 techs/21 variants)
conda run -n pycircuitsim python external_compact_models/PyCMG/scripts/generate_nn_data.py \
    --device both --universal

# Train DirectNet — asinh+zscore outputs, MAE + per-target LDS, ASAP7 excluded.
# --size {small,medium,large} selects the SIZE_PRESETS architecture+epochs+batch.
# --loss-preset {default,e1,e2,e3} controls column-weight / output-subset variants.
# Default save_prefix: refac_dn_<size>[_<loss-preset>]_<device>.
conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --size medium --device-type {nmos,pmos} \
    --exclude-techs asap7 --num-tech-codes 18 --cuda

# Optional: TSMC5-only residual head on a frozen DirectNet backbone (V6 Tier M2).
# Separate trainer (not under bsimar.cli.train); reuses backbone norm.npz.
conda run -n pycircuitsim python -u -m bsimar.training.tsmc5_residual_train \
    --device-type {nmos,pmos} --epochs 30 --cuda
```

**Checkpoints** (in `external_compact_models/bsimar/checkpoints/`):

- V6 DirectNet: `refac_dn_{small,medium,large}_{nmos,pmos}_best.pt` + `_norm.npz`. Production size is `medium`.
- V6 Transformer: `refac_tf_{small,medium,large}_{nmos,pmos}_best.pt` + `_norm.npz` + `_config.npz`.
- Resolver cascade (`pycircuitsim/parser.py`): `refac_dn_medium > refac_dn_small > refac_dn_large > v4_re_dn_universal > v4_dn_universal > per-tech > bare` for LEVEL=73; analogous `refac_tf_*` cascade for LEVEL=74. Override via `--exp-name` at train time or `PYCIRCUITSIM_NN_CHECKPOINT_*` env vars at runtime.
- Legacy `v4_*` checkpoints still load via the tail of the cascade.

**Netlist usage:** `.model nmos_nn NMOS (LEVEL=73 TECH=tsmc5 VT=lvt)` with `L=16n NFIN=10` (LEVEL=74 for BSIMAR). Parser auto-resolves tech-code from TECH+VT.

### Output files

Results in `results/<circuit_name>/<analysis_type>/`: `*_simulation.lis`, `*_dc_sweep.csv` / `*_transient.csv`.

## Testing & Verification

All tests require `conda activate pycircuitsim`.

**Shared infra:** `tests/common/{base,bsimcmg_dc,bsimcmg_tran,nn}.py` and `tests/references/`.

**BSIM-CMG DC:** L1 `verify_bsimcmg_dc.py` (2) · L2 `verify_bsimcmg_dc_comprehensive.py` (67) · L3 `verify_multi_tech_dc.py` (44).
**BSIM-CMG Transient:** L1 `verify_bsimcmg_tran.py` (1) · L2 `verify_bsimcmg_tran_comprehensive.py` (37) · L3 `verify_multi_tech_tran.py` (72).
**NN V6 DC:** L1 `verify_nn_dc.py` (~6, TSMC12 SVT) · all-tech sweeps via `verify_nn_dc_tran.py --{dc,pmos,inverter}-only`.
**NN V6 Transient:** L1 `verify_nn_tran_v4.py` (~4, filename historical) · all-tech via `verify_nn_dc_tran.py --tran-only`.
**Other:** `verify_bsimcmg_op.py` (OP <0.02% vs NGSPICE), `verify_nn_leave_one_out.py` (zero-shot transfer).

Quick sanity:

```bash
python tests/verify_bsimcmg_op.py && python tests/verify_bsimcmg_dc.py && python tests/verify_bsimcmg_tran.py
```

Note: `verify_nn_universal*.py` / `verify_nn_multi_tech.py` need porting to the V6 tech-code API (use `TECH_CODE_MAP` lookup instead of `extract_process_params`).

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

### NN Model Rules (LEVEL=73 DirectNet V6 + LEVEL=74 BSIMAR V6)

Both share the same data pipeline and inference-time rules. DirectNet is the V6 primary (single-shot MLP); BSIMAR (autoregressive Transformer with parallel cap head) is deprioritized at the current stage per Rule 18. Both use `nn.Embedding` for tech-code identity (7-dim input: Vgs, Vds, Vbs, NFIN, L, T, tech_code).

1. **Jacobian consistency is mandatory** — gm/gds/gmb MUST be `torch.autograd.grad(id, V)`, never independent predictions. Holds for LEVEL=73 and LEVEL=74.
2. **PMOS source-relative frame** — shift voltages by -Vs before NN eval (`v_d_nn = v_d - v_s`). Training uses Vs=0; in CMOS PMOS Vs=VDD.
3. **Training range covers NR overshoot** — margin ±VDD beyond operating range, not ±0.1V.
4. **Smooth voltage clamping** — softplus-based, NOT `torch.clamp`. Hard clamp creates zero-gradient cliffs that stall NR. Margin = 5% of per-dim training range.
5. **Physics-based gds floor** — `gds = max(gds, |id|*0.5, 1e-12)`. NN autograd gds ≈ 0 in saturation; without the floor inverter gain → ∞ and NR diverges. At FinFET 16nm BSIM-CMG λ=0.3-1.2 V⁻¹. Floor only affects the NR Jacobian, not the converged solution.
6. **TSMC asymmetric L** — NMOS L=16nm, PMOS L=20nm; NNTechConfig uses `L_nmos`/`L_pmos`.
7. **ASAP7 modelcard name mapping** — parser auto-maps netlist names to `nmos_rvt` / `pmos_rvt`.
8. **PyCMG integration** — `bsimar/config.py` re-exports `NNTechConfig`, `TECH_CONFIGS`, `TECH_CODE_MAP`, `OUTPUT_COLUMNS` from `pycmg.nn_config`. v3 process-param exports (`ProcessParams`, `extract_process_params`, `INPUT_COLUMNS`) removed. Backward-compat alias `TechConfig = NNTechConfig`. Training VDD may differ from PyCMG (ASAP7 train=0.7V, PyCMG=0.9V).
9. **Data validation** — `eval_single_point` rejects NaN/Inf and `|id| > 1A`. PyCMG `eval_dc` raises `RuntimeError` on internal-node convergence failure. Default NFIN range `[2, 3, 5, 10, 15, 20, 24]` (NFIN=1 excluded).
10. **Loss layer** — both models use `bsimar.losses.MAELoss` with **per-target LDS weights only** (3-axis stack collapsed to 1 in v5 Phase A). Hard-wired in `train_directnet` / `train_transformer`. DO NOT re-add: `DirectLoss`, `ChargeConsistencyLoss`, `SignConsistencyLoss`, `BoundaryLoss`, `SlopeMatchLoss`, Vov-LDS / subthreshold-LDS axes. Structural Vds gate (`apply_id_gate`) and slope-match loss deleted 2026-05-03 — rule 15's inference-time correction already enforces Id(Vds=0)=0.
11. **BSIMAR output ordering** — Transformer output in `BSIMAR_COLUMN_ORDER` (`qg, qb, qd, qs, id, gm, gds, gmb, cgg, cgd, cgs, cdg, cdd`), not `OUTPUT_COLUMN_ORDER`. Consumer code (`mosfet_bsimar.py`) takes autograd derivatives at the right column indices.
12. **Parallel cap head** — Transformer emits 5 capacitances in parallel from gmb hidden state, not sequential AR steps. AR loop runs 8 steps (charges + currents/conds). `parallel_caps` and `grouped_inputs` structural, not configurable.
13. **Unified CLI** — `python -m bsimar.cli.train --model {direct,transformer} --size {small,medium,large} [--loss-preset {default,e1,e2,e3}] ...` (default save_prefix `refac_{dn,tf}_<size>[_<preset>]_<device>`). TSMC5 residual head trained via the separate `bsimar.training.tsmc5_residual_train` entry. Same `.npz` from PyCMG; checkpoints under `external_compact_models/bsimar/checkpoints/`.
14. **Charge conservation** — simulator always computes `qs = -(qg + qd + qb)` analytically, even for 13-output models that directly predict `qs`. Guarantees Kirchhoff conservation at every transient timestep.
15. **Analytical Vds correction** — `_MOSFETNNBase._apply_vds_correction()` enforces Id(Vds=0)=0 and Id=0 for reverse-Vds at inference. Four-part (order matters):

- (a) **Rail-restoring extrapolation** when `|Vds| > VDD_train` (= `self._vdd_estimate`, from training norm stats): quadratic Id ramp `½·g_max·overshoot² / x_ref` and linear gds ramp `g_max·overshoot / x_ref` (g_max=1mS, x_ref=½·VDD_train). Both zero-valued zero-sloped at the boundary so NR sees a smooth join (linear ramp tried first, caused NR oscillation for TSMC12/16 with ops at the boundary). Must run BEFORE the fast-path early-return.
- (b) one-sided `1-exp(-|Vds|/VT)` with VDD-proportional `VT = max(0.06·VDD, 0.026)V` for Id/gm/gmb.
- (c) symmetric gds with linear-region conductance `|Id_raw|·exp(-|Vds|/VT)/VT`.
- (d) sign enforcement (NMOS id≤0, PMOS id≥0).

  Step (a) fixed the BSIMAR transient bug: NN extrapolates flat-near-zero outside `[-VDD_train, VDD_train]`, creating a false KCL plateau the DCSolver mistakes for equilibrium (inverter OP locking at V(out)=4.4V instead of VDD). Step (a) replicates PyCMG's restoring leakage/impact-ionization physics so NR converges to the true rail. Verified across all 4 TSMC techs on probe (670K) and production (5.15M) checkpoints — inverter transient drops from 18-300% (FAIL) to 6-12% NRMSE (PASS) without retraining.

16. Always report MRE (%) metric.
17. Exclude ASAP7 tech at the current stage.
18. Do NOT train/eval BSIMAR Transformer model at this stage. Only care about DirectNet model.

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
- **Sprint history:** `docs/CHANGELOG.md`.

## Other Tips

* **Start every complex task in plan mode** — pour energy into the plan for 1-shot implementation. Re-plan the moment something goes sideways; enter plan mode for verification steps too.
* If the plan has several solutions or stages, implemente them in sequence. Use git commit first before you modify anything, keep the useful one that make progress and incorperate it. Otherwise, revert the solutions that were proven no help with git reset.
* **Update CLAUDE.md before every git commit**.
* Whenever there is a version update, update the `docs/CHANGELOG.md`.
* Always record the dead end proposal (the one being reverted), they are as important as the successful ones.
* **Never be lazy** — never simplify code or skip tests. **NEVER** use simplified equations or self-defined CMG models as reference; ALWAYS use simulation results as ground truth.
* **Use subagents** — second agent for staff-engineer plan review; multiple subagents on separate branches to try multiple solutions; roll back to main when a subagent hits a dead end.
* Enable "Explanatory" / "Learning" output style in `/config` to see *why* behind changes.
