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
compact-model families on one solver, all gated against NGSPICE ground truth.
Supports **`.op` / `.dc` / `.ac` / `.tran`** for all model types. Techs:
ASAP7 + TSMC5/7/12/16, plus **TSMC6** — *TSMC7 relabelled* under BSIM-CMG
(`methodology.md` §7), kept as a deliberate controlled repeat: same data, same
recipe, different training run. Pre-V7.3 reports score it in its own /4
column; current reports fold it into /20 as an inline repeat, never as an
independent technology.

**Core principles:** pure Python; Solver ↔ Device Models decoupled;
production-grade compact models via PyCMG/OSDI; basic HSPICE netlist
compatibility.

### Model families (current state)

| LEVEL | Model                  | Implementation                          | Role                       |
| ----- | ---------------------- | --------------------------------------- | -------------------------- |
| 72    | **BSIM-CMG**           | `models/mosfet_cmg.py` via PyCMG/OSDI   | FinFET **ground truth**    |
| 73    | **DirectNet**          | `models/mosfet_directnet.py` (PyTorch)  | **Production** NN fast path|
| 74    | **BSIM-AR Transformer**| `models/mosfet_bsimar.py` (PyTorch)     | Higher-fidelity AR NN      |
| 75    | **PFN (TabPFN port)**  | `models/mosfet_pfn.py` (PyTorch)        | In-context NN (**research**)|

- **BSIM-CMG (72)** — the reference every NN trains against and is gated on;
  all techs at DC <0.1 % / transient ~0.2 % NRMSE vs NGSPICE. Never substitute
  simplified equations for it.
- **DirectNet (73)** — single-shot MLP; 7-dim input (Vgs, Vds, Vbs, NFIN, L, T,
  tech_code via `nn.Embedding`); gm/gds/gmb are the autograd Jacobian of the
  predicted `id`, AC caps the dQ/dV autograd of predicted charges. **V7.4
  production = the freshly rebuilt clean `large` tier: 14/20 strict, zero
  flips** (`xl` best at 15/20 but 2.3× cost for one cell). V7.3 recipe results
  (`crit15m@xl` 19/20) are historical evidence, not rebuilt. Report:
  `docs/accuracy/DirectNet-L73-{clean,recipes}.md`.
- **BSIM-AR (74)** — autoregressive Transformer on the shared pipeline. V7.4
  clean `small` is best at **18/20 strict, zero flips**; capacity declines
  18→17→15→13. V7.3 corridor recipes reached 20/20 but were not rebuilt. AR
  inference ~30–100× slower on CPU, so DirectNet stays production. Stems
  `tsmc{X}_tf_{size}_{nmos,pmos}`; parser LEVEL=74 preempt cascade +
  `PYCIRCUITSIM_NN_FORCE_LEVEL=74`. Report: `BSIM-AR-L74-{clean,recipes}.md`.
- **PFN / TabPFN (75)** — faithful scaled-down TabPFN-v3 port (tech code =
  8th column token) with two deviations: a **frozen learned context** baked
  into the checkpoint (context-KV cached) and a direct 13-output value head
  (NR needs smooth autograd). Latest evidence: V7.3 clean `small` 14/20 at
  0.69 M params; ~10× DN eval cost, 4× faster than BSIM-AR. Not rebuilt in
  V7.4; no PFN checkpoints on this hardware. Env pins
  `PYCIRCUITSIM_NN_CHECKPOINT_PFN_{NMOS,PMOS}`, force hook LEVEL=75, drivers
  take `MODEL=tabpfn`; the `_config.npz` sidecar is **required** to rebuild
  the arch. Report: `PFN-L75-{clean,recipes}.md`.

The three NN families share one data / normalization / loss / training / eval
pipeline via the unified `bsimar` package.

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
- `external_compact_models/PyCMG/` — BSIM-CMG OSDI wrapper;
  OSDI binary at `build/osdi/bsimcmg.osdi`.
- `tests/common/` — shared gate infra; `tests/references/` — NGSPICE decks.

Solver internals worth knowing (beyond README §Algorithms): BE step 1 → Trap
step 2+ → BDF-2 auto on stiffness (NR>20, one-way); DC oscillation detection
(5-snapshot ring, accepts averaged solution if variance < 10× tol);
supply-relative adaptive damping with stuck-counter; hard `.ic` mode
(`force_ic=True`) stamps `.ic` nodes as temporary V-source constraints then
re-solves unconstrained — required for SRAM latches; LTE sub-stepping opt-in
via `max_substeps`; NN circuits use `_solve_dc_with_retry` (fast path first,
GMIN retry on `_last_solve_converged=False`); AC solves complex `Y = G + jωC`
about the DC OP including the full MOSFET transcapacitance stamp.

V7.5.0/V7.5.1 (AnalogGym parity) added, for **LEVEL=72 only**: the full
4-terminal Newton stamp (all four terminal currents + the condensed OSDI
Jacobian via `get_terminal_stamp` — junction/gate-leakage conductances
participate in NR) **and the full 4-terminal charge companion**
(`get_charge_stamp` from `condense_last_react` — the old 3×3 SPICE-cap
expansion stamps SIGN-FLIPPED transcap off-diagonals for floating-bulk
devices, which at small dt turns Newton into an ~15× error amplifier; the
3-conductance/3×3 stamps remain the NN contract); SPICE-style damped limiting
(`nr_limit_voltages`: fetlim/limvds/pnjlim in the NMOS-normalized frame,
±2.5 V window, retreat-to-anchor on eval failure; anchors reset per NR sweep;
an iteration that limited is never accepted as converged); source-referenced
OSDI evaluation (the internal-node solve is not shift-robust); a wide DC gmin
ladder (1e-2→GMIN) with an **automatic fallback** when a plain L72 solve
fails (so "BSIM-CMG never retries" is no longer true — it has its own
in-`DCSolver` ladder now, honest `final_converged` = last level/last source
step only); the transient retry ladder re-walks the interval with a locally
halved dt (down to `dt·2⁻²⁴`) instead of stiffening companions against a
fixed target time, with stagnation fail-fast; and the oscillation-average
acceptance KCL gate covers L72, not just NN. `PYCIRCUITSIM_NR_TRACE=1`
traces NR.

V7.5.2 closed both V7.5.1 follow-ups: **AC now stamps the full 4-terminal
`Y = G4 + jωC4`** for L72 from the same condensed OSDI Jacobians (the 3×3
expansion + channel-only conductances are gone from AC too; NO external
gmin in the AC load — it measurably pollutes high-impedance bulk nodes —
gmin remains a DC Newton aid only; NN AC path unchanged), and transients
gained **opt-in LTE-driven output refinement** (`refine_output=True` /
`PYCIRCUITSIM_TRAN_REFINE=1`, default off = byte-identical): every
committed march piece is emitted (non-uniform time axis, grid points
preserved exactly), PULSE corners become breakpoints (land-on + small
BE restart scaled to the local corner gap — kills trapezoid corner ring),
and each piece is LTE-checked (TRTOL=7, voltage state) with depth-1
un-commit rollback. `integration_method='trap'` pins the trapezoid (no
stiffness swap). Dead ends recorded in CHANGELOG §V7.5.2: post-corner
hold window (over-rings), branch-current LTE (NR-noise d³ is
dt-independent → thrash).

V7.5.3 added **per-device charge-state LTE** to the refine mode (still
opt-in, flags-off byte-identical): NGSPICE-CKTterr-shaped truncation
control on MOSFET terminal-charge states — solver-side 3-deep accepted
(t, q[4], i_cap[4]) histories, trap LTE as a current error, the CKTterr
`/h` charge loosener disarming the test where DD3 is NR-noise (exactly
the V7.5.2 branch-current dead end), CHGTOL=1e-18 because the stock
1e-14 floor sits 100× above a FinFET terminal charge and never fires
(stock NGSPICE marches fast spikes via ITL4 iteration cuts, not charge
terr — measured from cktterr.c). Charge pump: **6/6 stride-independent**
(`up_imin` 1.49 %/1.84 %). Known cost: LDO load-step decks grind under
refine (96× on Basic_LDO for a correct 5/5) — open pathology, do not run
campaign-wide refine-on until fixed. The AnalogGym bench harness also
became NGSPICE-exact in V7.5.3 (sample-exact `.meas` windows, dctrcurv.c
grid rule with Kelvin accumulation, grid-matched strided extrema,
corroborated fork recovery, explicit NGSPICE-failure recovery, altns
fallback, honest `ng_ran` verdicts) — see CHANGELOG §V7.5.3 and
RESULTS_TSMC.md before quoting any bench number.

## Supported Features

Devices (R, C, V/I sources, PULSE, NMOS/PMOS L72–75, `X` subckt instances),
analyses (`.op`/`.dc`/`.tran`+`uic`/`.ac`), directives (`.model`, `.include`,
`.ic`, `.subckt`/`.ends`) — full syntax in README §Features / §Netlist Syntax.
Legacy LEVEL=1 removed. Subckt expansion is **flattening at parse time**
(internal nodes → `X1.n1`, devices → `M.X1.Mp1`, ground global,
`.model`/`.include` hoisted, loud errors on unknown subckt / port mismatch /
recursion >64); gate: `tests/verify_subckt.py` (11 checks, subckt ≡ flat
bit-identical).

## Validation

Inverter circuit must PASS Transient Analysis against NGSPICE ground truth
within reasonable numerical tolerance. Never use simplified/self-defined
equations as reference.

> **Accuracy evidence lives in `docs/accuracy/`** (index + scoreboard:
> `README.md`): two files per family — the *clean* report (the control: one
> training run, per tech / scale / testcase) and the *recipes* report (the
> training addenda measured against it). Cross-cutting material — gate
> definitions, strict-OMP discipline, the `gds` code-state ladder, the TSMC6
> repeat, the measured noise floor — is stated once in **`methodology.md`**;
> retracted claims in `archive-pre-gds-fix.md`. Tables are generated by
> `scripts/v730_docs_build.py` from gate logs (each report pinned to one
> complete campaign pass; committed-SHA guard so partial data cannot mix
> passes); `scripts/v730_coverage.py` reports coverage and emits gaps as a
> runnable job file. On the V7.4 hardware only DirectNet/BSIM-AR clean reports
> have local raw evidence.
>
> **Denominators changed in V7.3.0:** TSMC6 folds into the headline — complex
> totals /20, device AC /10, opamp AC /5 (older docs: /16, /8, /4). *No total
> is comparable across that boundary without rescaling.*
>
> Sprint history, dead-ends, and the open known-issue roadmap live in
> **`docs/CHANGELOG.md` + `MEMORY.md`** — not duplicated here.

## How to Run

Setup, quick start, netlist syntax, examples, output layout, NN training
commands, and performance/GPU flags: **README.md**. Essentials:

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
  under a perturbing flag. The V7.2.0 CPU bundle passed its §8.4 gates; V7.4
  closed the GPU axis (T3 48/48 executed, binding 24/24, 12/16 strict matching
  the CPU basket exactly; T4 8/8, zero basin flips).
  `PYCIRCUITSIM_NN_DEVICE=cuda` stays opt-in because CPU/flags-off remains the
  scored compatibility contract, not because a GPU fidelity gate is open.
- **`_require_nn_caps` contract** (V7.0.1): DC/OP skips the charge Jacobians;
  `TransientSolver`/`ACSolver` declare `_require_nn_caps`. A third caps
  consumer must call it too (`get_capacitances` self-heals, so the failure
  mode is slow, never wrong).
- **`Circuit.invalidate_topology()` contract** (V7.2.0): call it after any
  direct `components` mutation — node list/map and solver caches key on the
  topology version.
- `PYCIRCUITSIM_NN_FUSED_JAC=1` and `PYCIRCUITSIM_NN_AR_CACHE=1` **must stay
  default-off until a 16-gate re-gate clears them** (same math, different
  summation order; no incremental AR form can be bit-identical in float32).
  `verify_ar_cache.py` / `verify_batched_tail.py` guard those opt-in paths.
- Measurements + dead ends (do not retry TF32/compile/bf16 for DirectNet):
  `docs/plans/2026-07-26-v720-gpu-scaling.md`; the V7.0.x measurements live in
  `docs/CHANGELOG.md` §V7.0.0–V7.0.4.

## NN Training Notes (beyond README §NN Compact Models)

Per-tech NMOS/PMOS checkpoints for TSMC5/7/12/16 (+ TSMC6), all three families
at all four scales. `--tech-scope` ∈ `{tsmc5,tsmc6,tsmc7,tsmc12,tsmc16,universal}`;
`--size` ∈ `{small,medium,large,xl}`. Traps and facts the README doesn't carry:

- **Data generation needs BOTH `--enable-inv-trip` AND `--enable-subvt-off`**
  to reproduce the production datasets — omitting `--enable-subvt-off`
  silently yields a set 4.7 % smaller that is otherwise class-for-class
  identical.
- Curriculum recipes train via `scripts/recipe_train.sh` (warm-start from the
  clean same-size base). Full capacity sweep: `scripts/benchmark_gen_data.sh`
  → `benchmark_train_sml.sh` (`GPUS`, `NSTREAMS`) → `benchmark_run_tests.sh`
  → `benchmark_collect.py`.
- **Current V7.4 checkpoint set** (`bsimar/checkpoints/`): all 40 clean
  DirectNet + all 40 clean BSIM-AR (five techs × four tiers × two polarities).
  Recipe, universal and PFN rows in the reports are V7.3 historical evidence;
  those checkpoint families are not on this hardware.
- **Universal DirectNet (V6.7.0):** `u716_dn_*` stems, 18-code vocab,
  env-pin-only; best = `u716_dn_corroft_large` (10/12 strict, 0 FLIPs). See
  `DirectNet-L73-recipes.md` §7.
- **Resolver cascade** (`pycircuitsim/parser.py`): env pin
  `PYCIRCUITSIM_NN_CHECKPOINT_{DN,PFN}_{NMOS,PMOS}` read FIRST (an absent
  pinned stem RAISES — no silent fallback); then per-tech
  `tsmc{X}_{dn,tf,pfn}_{large,medium,small,xl}` (**large-first**) preempts the
  dormant universal fallback. Completed runs carry a `*_best.pt.complete`
  marker (a bare `_best.pt` may be a killed run). Resolutions log
  `[NN-resolver] ...`.

**Netlist usage:** `.model nmos_nn NMOS (LEVEL=73 TECH=tsmc5 VT=lvt)` with
`L=16n NFIN=10`. Parser auto-resolves checkpoint + local-vocab tech_code via
`bsimar.config.local_variant_code(scope, tech, variant)`.

### Output files

`results/<circuit_name>/<analysis_type>/`: an HSPICE-like `*_simulation.lis`
log, the data CSV, and a Matplotlib plot.

## Testing & Verification

All tests require `conda activate pycircuitsim`. Ground truth is **always**
NGSPICE on the identical BSIM-CMG (LEVEL=72) OSDI model. Gates are CPU-pinned
(`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`) and honor
`NGSPICE_BIN`; the complex/AC gate infra pins torch to 1 thread
(`PYCIRCUITSIM_TORCH_THREADS` overrides). Shared infra in `tests/common/`.

- **Subcircuit hierarchy:** `verify_subckt.py` — 11/11.
- **BSIM-CMG (72):** OP `verify_bsimcmg_op.py`; DC `verify_bsimcmg_dc.py` /
  `..._comprehensive.py` (81) / `verify_multi_tech_dc.py` (**53 PASS, 0
  ERROR since V7.5.0** — the former known-ERROR `TSMC5_lvt_inv_l_24nm` NR
  divergence is fixed by the full-terminal stamp); tran
  `verify_bsimcmg_tran{,_comprehensive}.py` (1/45) +
  `verify_multi_tech_tran.py` (86); AC `verify_ac.py` (**3/3 since V7.5.2**
  — L3 = floating-bulk NMOS+PMOS CS amps gating v(out) AND the bulk node;
  every earlier AC gate rail-tied the bulk, masking the 3×3-expansion
  hazard). Counts move with
  TSMC6 registration — same coverage, one duplicate column; quote TSMC6
  separately.
- **DirectNet (73):** `verify_nn_dc_tran.py` (inverter 8/8, DC 55/55, tran
  64/64); `verify_nn_multi_tech_{dc,tran}.py` (baseline-gated — pin OMP/MKL=1,
  VTC trip has ~±1 % scatter); `verify_nn_ac.py`;
  `verify_nn_lifted_source_dc.py` (NRMSE ≤10 %, guards Rule 2).
- **Complex circuits (4 × 5 = 20 scored gates):** `verify_complex_{ring_osc,
  opamp,sram_snm,switchcap}.py` scored vs NGSPICE (ring period, opamp gain,
  switchcap charge/droop, SRAM butterfly positivity + NRMSE). SRAM `force_ic`
  probe is a printed **diagnostic**, not a gate. Parametric mirrors
  `verify_complex_*_sweep.py` (baseline-gated, sha256-pinned) +
  `verify_complex_sweep_canaries.py`; opamp AC `verify_complex_opamp_ac.py`.
- **Perf-path gates (no NGSPICE):** `verify_ar_cache.py` (10 checks),
  `verify_batched_tail.py` (22 checks — exact bit equality per element),
  `verify_latch_basin_gpu.py --config {commit,gpu,stamp,order,…}` (full 6T
  latch, both states, 4 techs, 100 %-same-basin binding).
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

These rules were learned from bugs. Violating them causes NR divergence or
wrong results.

### Sign Convention for Device Models

1. **Use terminal current `id`, NOT channel `ids`** — `ids = id - is ≈ 2*id` (2× error).
2. **NMOS** `calculate_current()` returns `-result["id"]`; **PMOS** returns `result["id"]` (positive = leaving drain).
3. **Solver stamping** uses unified "current leaving drain" convention. All VCCS conductances (g_ds, g_m, g_mb) need full 4-entry stamps. An incomplete stamp breaks Jacobian symmetry.
   ```python
   i_leaving = -i_ds if is_pmos else i_ds
   i_eq = i_leaving - g_ds * v_ds - g_m * v_gs - g_mb * v_bs
   rhs[d_idx] -= i_eq    # same for NMOS and PMOS
   rhs[s_idx] += i_eq
   ```
   **LEVEL=72 does NOT use this path since V7.5.0** — it stamps the full
   4-terminal companion (`get_terminal_stamp`: i_out = −I_pycmg·m,
   G = −jac4·m; the [d,:] row reproduces gds/gm/gmb when junctions are off).
   The channel-only opvars are blind to junction/gate-leakage conductances,
   which at 125 °C carry the drain current (measured id=+1.8 mA vs
   gds=4.3e-13 S) — a 3-conductance Jacobian locks NR into a limit cycle.
   The 3-conductance stamp above remains the NN (LEVEL≥73) contract.
4. **gds floor** for stamping: `max(gds, 1e-12)`. Never `abs(gds)` — it flips large-negative to large-positive and diverges NR. Preserve gm/gmb signs.
   (V7.5.0 removed a violating `abs(gds)` from `MOSFET_CMG.get_conductance`;
   the floor lives at the stamp, and the full-J stamp adds GMIN across
   d-s/d-b/s-b instead.)
5. **Update `_is_mosfet()`** in `solver.py` when adding new device types.
6. **Test both NMOS and PMOS** vs NGSPICE: single OP, DC sweep, inverter VTC, inverter transient.

### NN Model Rules (LEVEL=73 DirectNet production; 74 BSIM-AR; 75 PFN)

All three NN levels share the data pipeline and inference rules and use
`nn.Embedding` for tech-code identity (7-dim input: Vgs, Vds, Vbs, NFIN, L, T,
tech_code). Rules 3–5, 9–10, 15 were deleted by the user (parked
BSIMAR-specific structure — recover from CHANGELOG/git if needed). Rule
numbers are internal to this doc and no longer cited in code.

1. Feel free to **re-generate datasets and re-train all models** as you want.
2. **Source-relative frame for BOTH device types** — shift all terminal
   voltages by -Vs before NN eval (`v_d_nn = v_d - v_s`, Vs ≡ 0). Training
   uses Vs=0; shift invariance makes this exact. Until V6.4.7 only PMOS was
   shifted; the canary `verify_nn_lifted_source_dc.py` guards this permanently.
6. **ASAP7 modelcard name mapping** — parser auto-maps netlist names to
   `nmos_rvt`/`pmos_rvt`.
7. **PyCMG integration** — `bsimar/config.py` re-exports `NNTechConfig`,
   `TECH_CONFIGS`, `TECH_CODE_MAP`, `OUTPUT_COLUMNS` from `pycmg.nn_config`.
   Training VDD may differ from PyCMG's runtime VDD; check `NNTechConfig.VDD`.
8. **Data validation** — `eval_single_point` rejects NaN/Inf and `|id| > 1A`.
   PyCMG `eval_dc` raises on internal-node convergence failure. NFIN=1 is
   excluded (unstable bins dropped per-bin, so NFIN≥2 trains).
11. **Unified CLI** — `python -m bsimar.cli.train --model {direct,transformer,
    tabpfn} --size {small,medium,large,xl} --device-type {nmos,pmos}
    --tech-scope {...}` (production = `large`). Flags all default-off /
    behavior-preserving: `--swa-mode`+`--ema-decay`; `--apply-filter`+
    `--class-weights`; `--enable-subvt-off`; loss terms `--sobolev` /
    `--subthresh` / `--charge-sobolev`; EKV backbone `--ekv-*`.
12. **Charge conservation** — simulator always computes `qs = -(qg + qd + qb)`
    analytically, even for 13-output models.
13. Always report MRE (%), R², NRMSE, Max error (mV) per tech.
14. **Exclude ASAP7** — out of scope (no checkpoints).
16. **Per-tech models use a LOCAL embedding vocab.** The loader remaps
    universal tech codes to a 0-indexed vocab; the trainer instantiates
    `num_tech_codes=N, unknown_code_id=N-1` (TSMC5:5, TSMC7:4, TSMC12:6,
    TSMC16:6). Training-time `p_unknown` dropout writes `unknown_code_id`, so
    a misaligned UNKNOWN id → CUDA assert. **Derive `unknown_code_id` from
    `num_tech_codes`; do NOT hardcode the universal 17.** Parser remaps at
    inference via `bsimar.config.local_variant_code(scope, tech, variant)`;
    scope read from the ckpt stem.

> **Load-bearing code (do NOT delete):** the V6.4.2 `_MonotoneVgResidual` +
> `--monotonic` path (`bsimar/{cli/train,models/direct_net,training/trainer}.py`,
> `pycircuitsim/models/mosfet_directnet.py`) must stay committed — on-disk
> checkpoints carry `mono.*` state_dict keys and fail to load without it
> (stock checkpoints route `mono=None`). The default-off EKV backbone
> (`_EKVCore`, `core.*` keys) and the Sobolev/subthreshold/charge-Sobolev loss
> terms are likewise kept recoverable; all leave stock checkpoints
> byte-identical.

---

## Important Paths

- **PyCMG:** `external_compact_models/PyCMG/` (21 device variants).
- **OSDI binary:** `build/osdi/bsimcmg.osdi` (PyCMG-relative).
- **Modelcards:** `modelcards/` (PyCMG-relative); ASAP7 `*.pm` committed; TSMC
  raw PDK `cln*.l` gitignored/IP-protected — naive cards regenerated
  on-the-fly via `pycmg.tech.resolve_modelcard` into `build/modelcards/`.
  Never commit `modelcards.tar.gz`.
- **Results:** `results/<circuit_name>/<analysis_type>/`. **Test results:**
  `tests/verify_*_results/` (generated, not tracked).
- **Sprint history:** `docs/CHANGELOG.md`. House-cleans have pruned old plans,
  one-off scripts, gate-specific diagnostics, stale datasets, and result dirs —
  some path references in older CHANGELOG entries are intentionally dangling
  (the narrative, not the file, is the record).

## Other Tips

* **Start every complex task in plan mode** — pour energy into the plan for
  1-shot implementation. Re-plan the moment something goes sideways.
* If the plan has several stages, implement in sequence. Git commit before
  modifying; keep what makes progress, `git reset` what doesn't.
* On every version update, add a `docs/CHANGELOG.md` entry. **Record dead-end
  proposals (the ones reverted) — they are as important as the successful ones.**
* **Never be lazy** — never simplify code or skip tests. **NEVER** use
  simplified equations or self-defined CMG models as reference; ALWAYS use
  simulation results as ground truth.
* Use **GPUs in parallel** when you train NN models.
