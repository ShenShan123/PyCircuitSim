# Project: PyCircuitSim

> **Doc layout:** user-facing how-to (install, run, netlist syntax, examples,
> output files, training commands, performance flags) lives in **`README.md`**;
> sprint history, version-by-version verdicts and dead-ends live in
> **`docs/CHANGELOG.md`**; accuracy evidence lives in **`docs/accuracy/`**.
> CLAUDE.md holds only what those don't: durable architecture, current
> production state, testing discipline, and the design rules learned from bugs.

## Overview

Pure-Python SPICE-like circuit simulator emphasizing educational clarity and a
decoupled Solver ↔ Device-Model architecture. **Primary goal:** four coexisting
compact-model families on one solver, all gated against NGSPICE ground truth:

- **BSIM-CMG** (LEVEL=72) — PyCMG-wrapped OSDI FinFET model; the **ground truth**
  every NN trains against and is gated on.
- **DirectNet** (LEVEL=73) — feed-forward MLP; the **production** NN fast path.
- **BSIM-AR Transformer** (LEVEL=74) — autoregressive Transformer; the validated
  **higher-fidelity** option (V7.4 clean best 18/20, ~30–100× slower AR inference).
- **PFN / TabPFN** (LEVEL=75) — TabPFN-v3-style in-context transformer;
  **research** family (latest evidence: V7.3 clean small 14/20 at 0.69M params;
  ~10× DN eval cost, 4× faster than BSIM-AR).

The three NN families share one data / normalization / loss / training / eval
pipeline via the unified `bsimar` package (`external_compact_models/bsimar/`).
Supports **`.op` / `.dc` / `.ac` / `.tran`** for all model types. Techs:
ASAP7 + TSMC5/7/12/16, plus **TSMC6** — which is *TSMC7 relabelled* under
BSIM-CMG (`docs/2026-07-21-systematic-audit.md` §D1), retired for that in
V6.13.0 and **restored in V7.1.0 as a deliberate controlled repeat**: same data,
same recipe, different training run. Score it in its own /4 column, never inside
the old /16 when reading pre-V7.3 reports; current reports deliberately fold it
into /20 as an inline controlled repeat, not as an independent technology.

**Core Principles:** pure Python; Solver ↔ Device Models decoupled; production-grade
compact models via PyCMG/OSDI; basic HSPICE netlist compatibility.

## Architecture

Full module tree + responsibilities: README §Architecture. The load-bearing map:

- `pycircuitsim/solver.py` — MNA + Newton-Raphson; DC/Transient/AC solvers.
- `pycircuitsim/parser.py` — two-pass netlist parsing, `.model`/`.subckt`/`.ic`,
  subckt flattening, NN checkpoint resolver cascade.
- `pycircuitsim/models/` — `mosfet_cmg.py` (L72), `mosfet_nn.py` (shared
  `_MOSFETNNBase`: voltage prep, autograd, Vds correction), `mosfet_directnet.py`
  (L73), `mosfet_bsimar.py` (L74), `mosfet_pfn.py` (L75), `passive.py`.
- `external_compact_models/bsimar/` — the unified NN package (config, data,
  models, losses, training, eval, `cli/train.py`, gitignored `checkpoints/`).
- `external_compact_models/PyCMG/` — BSIM-CMG OSDI wrapper (git submodule);
  OSDI binary at `build/osdi/bsimcmg.osdi`.
- `tests/common/` — shared gate infra; `tests/references/` — NGSPICE decks.

Solver internals worth knowing (beyond README §Algorithms): BE step 1 → Trap
step 2+ → BDF-2 auto on stiffness (NR>20, one-way); DC oscillation detection
(5-snapshot ring, accepts averaged solution if variance < 10× tol);
supply-relative adaptive damping with stuck-counter; hard `.ic` mode
(`force_ic=True`) stamps `.ic` nodes as temporary V-source constraints then
re-solves unconstrained — required for SRAM latches; LTE sub-stepping opt-in
via `max_substeps`; NN circuits use `_solve_dc_with_retry` (fast path first,
GMIN retry on `_last_solve_converged=False`) — BSIM-CMG never enters the retry
branch; AC solves complex `Y = G + jωC` about the DC OP including the full
MOSFET transcapacitance stamp.

### Key Compact Models

| LEVEL | Model                  | Implementation                          | Role                                   |
| ----- | ---------------------- | --------------------------------------- | -------------------------------------- |
| 72    | **BSIM-CMG**           | `models/mosfet_cmg.py` via PyCMG/OSDI   | FinFET **ground truth**                |
| 73    | **DirectNet**          | `models/mosfet_directnet.py` (PyTorch)  | **Production** NN (fast path)          |
| 74    | **BSIM-AR Transformer**| `models/mosfet_bsimar.py` (PyTorch)     | Higher-fidelity AR NN                  |
| 75    | **PFN (TabPFN port)**  | `models/mosfet_pfn.py` (PyTorch)        | In-context NN (**research**)           |

- **BSIM-CMG (72)** — the reference every NN trains against and is gated on;
  all techs at DC <0.1 % / transient ~0.2 % NRMSE vs NGSPICE. Never substitute
  simplified equations for it.
- **DirectNet (73)** — single-shot MLP; 7-dim input (Vgs, Vds, Vbs, NFIN, L, T,
  tech_code with `nn.Embedding`). gm/gds/gmb are the **autograd Jacobian** of the
  predicted `id`; AC caps are the `dQ/dV` autograd of predicted charges; per-tech
  checkpoints use a local embedding vocab (Rule 16). **V7.4 production is the
  freshly rebuilt clean `large` tier: 14/20 strict, zero flips; clean `xl` is
  best at 15/20 but costs 2.3× more for one cell.** The V7.3 recipe results
  (`crit15m@xl` 19/20 best) remain historical evidence and were neither
  retrained nor promoted on the new hardware. Report:
  `docs/accuracy/DirectNet-L73-clean.md` (per tech / scale / testcase) +
  `-recipes.md` (production state, universal scope, dead ends); anything
  cross-cutting is in `methodology.md`.
- **BSIM-AR (74)** — autoregressive Transformer sharing DirectNet's pipeline.
  V7.4 rebuilt all clean tiers; `small` is best at **18/20 strict, zero flips**,
  with capacity declining 18→17→15→13. V7.3 corridor recipes reached 20/20,
  but those checkpoints were not rebuilt in V7.4 and are historical evidence,
  not the current artifact set. AR inference is ~30–100× slower on CPU, so
  DirectNet stays production. Per-tech
  `tsmc{X}_tf_{small,medium,large,xl}_{nmos,pmos}`; parser LEVEL=74 preempt
  cascade + `PYCIRCUITSIM_NN_FORCE_LEVEL=74` hook. Report:
  `docs/accuracy/BSIM-AR-L74-{clean,recipes}.md`.
- **PFN / TabPFN (75)** — faithful scaled-down port of TabPFN-v3's in-context
  transformer (tech code = 8th column token, local vocab), with two deviations: a
  **frozen learned context** (stratified K-row buffer baked into the checkpoint,
  context-KV cached at inference) and a direct 13-output value head (NR needs
  smooth autograd). Latest clean `small` = 14/20 strict in V7.3; PFN was not
  rebuilt and no PFN checkpoints are present on the V7.4 hardware.
  Env pins `PYCIRCUITSIM_NN_CHECKPOINT_PFN_{NMOS,PMOS}`, hook
  `PYCIRCUITSIM_NN_FORCE_LEVEL=75`, drivers take `MODEL=tabpfn`. The `_config.npz`
  sidecar is **required** to rebuild the arch. Report: `docs/accuracy/PFN-L75-{clean,recipes}.md`.

## Supported Features

Devices (R, C, V/I sources, PULSE, NMOS/PMOS L72–75, `X` subckt instances),
analyses (`.op`/`.dc`/`.tran`+`uic`/`.ac`), directives (`.model`, `.include`,
`.ic`, `.subckt`/`.ends`), and the full netlist syntax are documented in
README §Features / §Netlist Syntax. Legacy LEVEL=1 removed. Subckt expansion is
**flattening at parse time** (internal nodes → `X1.n1`, devices → `M.X1.Mp1`,
ground stays global, `.model`/`.include` hoisted, loud errors on unknown
subckt / port mismatch / recursion >64); gate: `tests/verify_subckt.py` (11
checks, subckt ≡ flat bit-identical).

## Validation

Inverter circuit must PASS Transient Analysis against NGSPICE ground truth within
reasonable numerical tolerance. Never use simplified/self-defined equations as reference.

> **Accuracy evidence lives in `docs/accuracy/`** (index + scoreboard:
> `README.md`). Restructured V7.3.0 into **two files per family** —
> `{DirectNet-L73,BSIM-AR-L74,PFN-L75}-{clean,recipes}.md`. The *clean* report
> is the control (one training run, no addendum) and answers **per tech, per
> scale, per testcase**; the *recipes* report carries the training addenda
> measured against it, filtered to the arms that earn a row. Everything
> cross-cutting — gate definitions, strict-OMP discipline, the `gds` code-state
> ladder, the TSMC6 repeat, the measured noise floor — is stated once in
> **`methodology.md`**. Register of retracted claims: `archive-pre-gds-fix.md`.
>
> Tables are generated by `scripts/v730_docs_build.py` from gate logs. On the
> V7.4 hardware, only DirectNet/BSIM-AR clean reports have local raw evidence;
> each report is pinned to one complete campaign pass. The builder preserves
> incomplete-source V7.3 recipe/PFN reports only when their committed SHA-256
> matches, so partial data cannot mix passes and full `--check` stays valid.
> `scripts/v730_coverage.py` reports coverage by pass and emits gaps as a
> runnable job file.
>
> **Denominators changed in V7.3.0:** TSMC6 folds into the headline, so complex
> totals are **/20**, device AC **/10**, opamp AC **/5**. Older documents scored
> /16, /8, /4 over four techs — *no total is comparable across that boundary
> without rescaling*. TSMC6 is still TSMC7 relabelled (`methodology.md` §7).
>
> **Sprint history, version-by-version status, dead-ends, and the open known-issue
> roadmap live in `docs/CHANGELOG.md` + `MEMORY.md`** — not duplicated here.

## How to Run

Setup, quick start, netlist syntax, examples, output layout, NN training
commands, and the performance/GPU flags are all in **README.md**. Essentials:

```bash
conda activate pycircuitsim          # env at /home/shenshan/.conda/envs/pycircuitsim
python main.py examples/<deck>.sp    # analysis chosen by the directive in the deck
```

**Prerequisites:** NGSPICE 45.2+ (`/usr/local/ngspice-45.2/bin/ngspice`; repo
fallback `tools/ngspice-45.2/bin/ngspice` via `NGSPICE_BIN`), OpenVAF
(`/usr/local/bin/openvaf`), OSDI binary
`external_compact_models/PyCMG/build/osdi/bsimcmg.osdi`, PyTorch for L73/74/75.

## Performance Discipline (V7.0.x / V7.2.0)

User-facing knob documentation: README §Performance & GPU Acceleration.
The rules that bind development:

- Every perf change is **bit-identical** (ships default-on) or **perturbing**
  (ships default-off behind an env flag, promoted only after a full re-gate).
  Fidelity is a **CPU, flags-off property** — no scoreboard number changes
  under a perturbing flag. The V7.2.0 CPU bundle {commit, stamp, order} passed
  its §8.4 gates (T3 15/16 strict 0-flip = production cell-for-cell; T4
  latch-basin 8/8). V7.4 closes the GPU axis on the rebuilt clean artifacts:
  T3 is 48/48 executed, binding SRAM+switchcap 24/24, Rule 2 15/15, zero
  flips/errors and **12/16 strict exactly matching the current CPU basket**;
  full-bundle T4 is **8/8, zero basin flips**, worst max|ΔV| 0.1206 mV.
  `PYCIRCUITSIM_NN_DEVICE=cuda` stays opt-in because CPU/flags-off remains the
  scored compatibility contract, not because a GPU fidelity gate is open.
- **`_require_nn_caps` contract** (V7.0.1): DC/OP skips the charge Jacobians;
  `TransientSolver`/`ACSolver` declare `_require_nn_caps`. Adding a third caps
  consumer means calling it there too (`get_capacitances` self-heals, so the
  failure mode is slow, never wrong).
- **`Circuit.invalidate_topology()` contract** (V7.2.0): call it after any
  direct `components` mutation — the node list/map and solver-side caches key
  on the topology version.
- `PYCIRCUITSIM_NN_FUSED_JAC=1` and `PYCIRCUITSIM_NN_AR_CACHE=1` **must stay
  default-off until a 16-gate re-gate clears them** (same math, different
  summation order; no incremental AR form can be bit-identical in float32 —
  `F.linear` is not row-stable on CPU). `verify_ar_cache.py` /
  `verify_batched_tail.py` guard the opt-in paths no accuracy gate reaches.
- Measurements + dead ends (do not retry TF32/compile/bf16 for DirectNet):
  `docs/plans/2026-07-25-v700-nn-perf.md`, `docs/plans/2026-07-26-v720-gpu-scaling.md`.

## NN Training Notes (beyond README §NN Compact Models)

Per-tech NMOS/PMOS checkpoints for TSMC5/7/12/16 (+ TSMC6), all three families
at all four scales. `--tech-scope` ∈ `{tsmc5,tsmc6,tsmc7,tsmc12,tsmc16,universal}`;
`--size` ∈ `{small,medium,large,xl}`. Traps and facts the README doesn't carry:

- **Data generation needs BOTH `--enable-inv-trip` AND `--enable-subvt-off`**
  to reproduce the production datasets — omitting `--enable-subvt-off` silently
  yields a set 4.7 % smaller that is otherwise class-for-class identical, which
  is exactly how it goes unnoticed.
- Curriculum recipes train via `scripts/recipe_train.sh` — warm-start from the
  clean same-size base. The `v660clean`/`crit30f` production detour is historical;
  V7.4 serves newly rebuilt clean checkpoints. Full capacity
  sweep: `scripts/benchmark_gen_data.sh` → `benchmark_train_sml.sh` (`GPUS`,
  `NSTREAMS`) → `benchmark_run_tests.sh` → `benchmark_collect.py`.
- **Current V7.4 checkpoint set** (`bsimar/checkpoints/`): all 40 clean
  DirectNet and all 40 clean BSIM-AR (five techs × four tiers × two polarities).
  Recipe, universal and PFN rows in the reports are V7.3 historical evidence;
  those checkpoint families are not present on the new hardware.
- **Universal DirectNet (V6.7.0):** `u716_dn_{clean,csob,corroft,crit30u}_large` +
  `_{clean,corroft}_xl` + TSMC5 fine-tunes `u716f5_plain_n{1000000,full}_large` —
  18-code vocab, env-pin-only. Best = `u716_dn_corroft_large` (10/12 strict, 0 FLIPs).
  See `docs/accuracy/DirectNet-L73-recipes.md` §7.
- **Resolver cascade** (`pycircuitsim/parser.py`): env pin
  `PYCIRCUITSIM_NN_CHECKPOINT_{DN,PFN}_{NMOS,PMOS}` read FIRST (since V6.6.6 an absent
  pinned stem RAISES — no silent fallback); then per-tech `tsmc{X}_{dn,tf,pfn}_{large,
  medium,small,xl}` (**large-first**) preempts the dormant universal fallback.
  Completed runs carry a `*_best.pt.complete` marker (a bare `_best.pt` may be a
  killed run). Resolutions log `[NN-resolver] L73 <name> TECH=.. VT=.. -> <chk> ...`.

**Netlist usage:** `.model nmos_nn NMOS (LEVEL=73 TECH=tsmc5 VT=lvt)` with
`L=16n NFIN=10`. Parser auto-resolves the checkpoint + local-vocab tech_code via
`bsimar.config.local_variant_code(scope, tech, variant)`.

### Output files

`results/<circuit_name>/<analysis_type>/`: an HSPICE-like `*_simulation.lis` log, the
data (`*_dc_sweep.csv` / `*_transient.csv` / `*_ac_sweep.csv`), and a Matplotlib plot.

## Testing & Verification

All tests require `conda activate pycircuitsim`. Ground truth is **always** NGSPICE
on the identical BSIM-CMG (LEVEL=72) OSDI model. Gates are CPU-pinned
(`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`) and honor
`NGSPICE_BIN`; the complex/AC gate infra pins torch to 1 thread
(`PYCIRCUITSIM_TORCH_THREADS` overrides). Shared infra in `tests/common/`.

- **Subcircuit hierarchy:** `verify_subckt.py` — 11/11 (L1 subckt≡flat
  bit-identical, L2 L72 inverter vs NGSPICE, L3 nested buffer).
- **BSIM-CMG (72):** OP `verify_bsimcmg_op.py`; DC L1 `verify_bsimcmg_dc.py` ·
  L2 `..._comprehensive.py` (67) · L3 `verify_multi_tech_dc.py` (43 PASS + 1
  known ERROR `TSMC5_lvt_inv_l_24nm`, pre-existing pure-L72 NR divergence);
  tran L1/L2/L3 (1/37/72); AC `verify_ac.py` (2/2). Counts grew/shrink with
  TSMC6 registration — same coverage, one duplicate tech column; quote TSMC6
  separately.
- **DirectNet (73):** `verify_nn_dc_tran.py` (inverter 8/8, DC 55/55, tran
  64/64); `verify_nn_multi_tech_{dc,tran}.py` (baseline-gated — pin OMP/MKL=1,
  VTC trip has ~±1 % scatter); `verify_nn_ac.py`;
  `verify_nn_lifted_source_dc.py` (NRMSE ≤10 %, guards Rule 2).
- **Complex circuits (4 × 5 = 20 scored gates):** `verify_complex_{ring_osc,opamp,
  sram_snm,switchcap}.py` scored vs NGSPICE (ring period, opamp gain, switchcap
  charge/droop, SRAM butterfly positivity + NRMSE). SRAM `force_ic` probe is a
  printed **diagnostic**, not a gate. Parametric mirrors
  `verify_complex_*_sweep.py` (baseline-gated, sha256-pinned) +
  `verify_complex_sweep_canaries.py`; opamp AC `verify_complex_opamp_ac.py`.
- **Perf-path gates (no NGSPICE):** `verify_ar_cache.py` (10 checks — flag-off
  bit-identical, cached ≡ stock within 1e-4, lever >1.15×);
  `verify_batched_tail.py` (22 checks — exact bit equality per element across
  all 3 families × polarity × caps, §8.1 source tripwires);
  `verify_latch_basin_gpu.py --config {commit,gpu,stamp,order,…}` (full 6T
  latch, both states, 4 techs, 100 %-same-basin binding — the failure mode the
  perturbing flags could cause that no other gate tests).
- **Diagnostics** (`tests/diag_*.py`, **not** gates): `diag_l72_*` controls
  prove L72-in-PyCircuitSim ≈ NGSPICE, isolating NN-surface gaps;
  `diag_nn_jacobian_consistency.py`.

**Quick sanity:**

```bash
python tests/verify_bsimcmg_op.py && python tests/verify_bsimcmg_dc.py && python tests/verify_bsimcmg_tran.py
python tests/verify_subckt.py
NGSPICE_BIN="$PWD/tools/ngspice-45.2/bin/ngspice" python tests/verify_ac.py   # if /usr/local absent
```

---

## Development Guidelines

**Coding standards:** type hints on all signatures; clear names (`v_gate`,
`i_drain`); docstrings for complex algorithms; voltage clamping Vgs±5V, Vds±10V.

**Separation principle:** `solver.py` builds MNA + executes NR (no device
equations); `models/` computes current/conductances (no matrix ops);
`simulation.py` orchestrates; all devices inherit from `Component`.
Convergence: `|ΔV| < VNTOL + RELTOL × max(|V_old|,|V_new|)` (RELTOL=1e-4,
VNTOL=1e-7); GMIN 1e-12 S; DC GMIN stepping opt-in (`use_gmin_stepping=True`).

**Entry points:** CLI `main.py`; API `pycircuitsim.simulation.run_simulation()`;
module exports (Circuit, Parser, Visualizer, run_simulation).

**Environment & tools:** conda env `pycircuitsim`; PyTorch 2.10.0 (CPU);
OpenVAF `/usr/local/bin/openvaf`; NGSPICE `/usr/local/ngspice-45.2/bin/ngspice`.

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

### NN Model Rules (LEVEL=73 DirectNet production; 74 BSIM-AR; 75 PFN)

All three NN levels share the data pipeline and inference rules and use `nn.Embedding`
for tech-code identity (7-dim input: Vgs, Vds, Vbs, NFIN, L, T, tech_code). Rules
3–5, 9–10, 15 were deleted by the user (parked BSIMAR-specific structure — recover
from CHANGELOG/git if needed). Rule numbers are internal to this doc and no longer
cited in code.

1. Feel free to **re-generate datasets and re-train all models** as you want.
2. **Source-relative frame for BOTH device types** — shift all terminal voltages by
   -Vs before NN eval (`v_d_nn = v_d - v_s`, Vs ≡ 0). Training uses Vs=0; shift
   invariance makes this exact. Until V6.4.7 only PMOS was shifted — lifted-source
   NMOS saw phantom Vgs/Vds; the canary `verify_nn_lifted_source_dc.py` (NRMSE ≤10%)
   guards this permanently.
6. **ASAP7 modelcard name mapping** — parser auto-maps netlist names to `nmos_rvt`/`pmos_rvt`.
7. **PyCMG integration** — `bsimar/config.py` re-exports `NNTechConfig`, `TECH_CONFIGS`,
   `TECH_CODE_MAP`, `OUTPUT_COLUMNS` from `pycmg.nn_config` (alias `TechConfig = NNTechConfig`).
   Training VDD may differ from PyCMG's runtime VDD; check `NNTechConfig.VDD` per tech.
8. **Data validation** — `eval_single_point` rejects NaN/Inf and `|id| > 1A`. PyCMG
   `eval_dc` raises on internal-node convergence failure. NFIN=1 is excluded: unstable
   `(variant, NFIN=1)` bins fail OSDI convergence and are dropped per-bin, so NFIN≥2 trains.
11. **Unified CLI** — `python -m bsimar.cli.train --model {direct,transformer,tabpfn}
    --size {small,medium,large,xl} --device-type {nmos,pmos} --tech-scope {...} ...`
    (production = `large`; V7.4 clean `xl` improves the circuit score by one
    cell but costs 2.3× more). Per-tech `--tech-scope` → default
    save_prefix `tsmc{X}_{dn,tf,pfn}_<size>_<device>`. Flags (all default-off /
    behavior-preserving): `--swa-mode {none,ema,swa}`+`--ema-decay`; `--apply-filter
    {on,off}`+`--class-weights`; `--enable-subvt-off`; loss terms `--sobolev` /
    `--subthresh` / `--charge-sobolev`; EKV backbone `--ekv-core`/`--ekv-alpha`/`--ekv-hidden`.
12. **Charge conservation** — simulator always computes `qs = -(qg + qd + qb)`
    analytically, even for 13-output models. Guarantees Kirchhoff conservation every timestep.
13. Always report MRE (%), R², NRMSE, Max error (mV) per tech.
14. **Exclude ASAP7** — out of scope (no checkpoints; see Overview).
16. **Per-tech models use a LOCAL embedding vocab.** For a per-tech `--tech-scope` the
    loader remaps universal tech codes to a 0-indexed vocab and the trainer instantiates
    `num_tech_codes=N, unknown_code_id=N-1`, N = variants+1 (TSMC5:5, TSMC7:4,
    TSMC12:6, TSMC16:6). Training-time `p_unknown` dropout writes `unknown_code_id`, so a
    misaligned UNKNOWN id → CUDA assert. **Derive `unknown_code_id` from `num_tech_codes`;
    do NOT hardcode the universal 17.** Parser remaps at inference via
    `bsimar.config.local_variant_code(scope, tech, variant)`; scope read from the ckpt stem.

> **Load-bearing code (do NOT delete):** the V6.4.2 `_MonotoneVgResidual` + `--monotonic`
> path (`bsimar/{cli/train,models/direct_net,training/trainer}.py`,
> `pycircuitsim/models/mosfet_directnet.py`) must stay committed — on-disk checkpoints
> carry `mono.*` state_dict keys and fail to load without it (stock checkpoints route
> `mono=None`, no inference change). The default-off EKV backbone (`_EKVCore`, `core.*`
> keys) and the Sobolev/subthreshold/charge-Sobolev loss terms are likewise kept
> recoverable; all leave stock checkpoints byte-identical.

---

## Important Paths

- **PyCMG submodule:** `external_compact_models/PyCMG/` (21 device variants).
- **OSDI binary:** `build/osdi/bsimcmg.osdi` (PyCMG-relative).
- **Modelcards:** `modelcards/` (PyCMG-relative); ASAP7 `*.pm` committed; TSMC raw PDK
  `cln*.l` gitignored/IP-protected — naive cards regenerated on-the-fly via
  `pycmg.tech.resolve_modelcard` into `build/modelcards/`. Never commit `modelcards.tar.gz`.
- **Results:** `results/<circuit_name>/<analysis_type>/`. **Test results:**
  `tests/verify_*_results/` (generated, not tracked).
- **Sprint history:** `docs/CHANGELOG.md`. House-cleans have pruned old plans, one-off
  scripts, gate-specific diagnostics, stale datasets, and result dirs — so some path
  references in older CHANGELOG entries are intentionally dangling (the narrative, not
  the file, is the record).

## Other Tips

* **Start every complex task in plan mode** — pour energy into the plan for 1-shot
  implementation. Re-plan the moment something goes sideways.
* If the plan has several stages, implement in sequence. Git commit before modifying;
  keep what makes progress, `git reset` what doesn't.
* On every version update, add a `docs/CHANGELOG.md` entry. **Record dead-end proposals
  (the ones reverted) — they are as important as the successful ones.**
* **Never be lazy** — never simplify code or skip tests. **NEVER** use simplified
  equations or self-defined CMG models as reference; ALWAYS use simulation results as
  ground truth.
* Use **GPUs in parallel** when you train NN models.
