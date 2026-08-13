# Project: PyCircuitSim

> **Doc layout — four documents, four jobs, no overlap.** Say a thing once.
>
> | Where | What it holds |
> | --- | --- |
> | **`README.md`** | The tutorial: install, run, netlist syntax, the deck library, the Python API, output files, training commands, performance flags, the gate inventory and how to run it. |
> | **`docs/CHANGELOG.md`** | The evolution: one entry per version, what it measured, and **every dead end that was reverted** — those are as load-bearing as the successes. |
> | **`docs/accuracy/`** | The scores. Tables are *generated* from gate logs; never hand-edited. |
> | **CLAUDE.md** (here) | The goal, the rules, and the lessons. Current production state, and the design rules learned from specific bugs — each one with the failure it prevents, because a rule without its reason gets optimized away. |
>
> If something here reads like history, it belongs in the CHANGELOG. If it
> reads like instructions, it belongs in the README.

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
- `tests/common/` — shared gate infra. Gates carry no netlists: every circuit
  they simulate is a deck in `examples/`, rendered by
  `tests.common.base.render_reference_deck` (V7.5.8).

Solver internals worth knowing (beyond README §Algorithms): BE step 1 → Trap
step 2+ → BDF-2 auto on stiffness (NR>20, one-way); DC oscillation detection
(5-snapshot ring, accepts averaged solution if variance < 10× tol);
supply-relative adaptive damping with stuck-counter; hard `.ic` mode
(`force_ic=True`) stamps `.ic` nodes as temporary V-source constraints then
re-solves unconstrained — required for SRAM latches; LTE sub-stepping opt-in
via `max_substeps`; NN circuits use `_solve_dc_with_retry` (fast path first,
GMIN retry on `_last_solve_converged=False`); AC solves complex `Y = G + jωC`
about the DC OP including the full MOSFET transcapacitance stamp.

**V7.5.x reshaped the LEVEL=72 solver for AnalogGym parity.** The per-version
narrative — what each change was, what it measured, and every dead end that
was reverted — is `docs/CHANGELOG.md` §V7.5.0–V7.5.7. What survives as a rule:

- **LEVEL=72 stamps the full 4-terminal companion**, both conductance
  (`get_terminal_stamp`) and charge (`get_charge_stamp` from
  `condense_last_react`). The channel-only 3-conductance/3×3 form is the
  **NN contract only** — using it for L72 is a bug, not an approximation:
  the 3×3 expansion sign-flips transcap off-diagonals on floating-bulk
  devices, which at small dt makes Newton a ~15× error amplifier. AC stamps
  the same condensed Jacobians (`Y = G4 + jωC4`), and takes **no external
  gmin** — it measurably pollutes high-impedance bulk nodes; gmin is a DC
  Newton aid only.
- **"BSIM-CMG never retries" is no longer true.** L72 has its own wide DC gmin
  ladder (1e-2→GMIN) with automatic fallback, so `final_converged` is honest
  about the last level / last source step only.
- **OSDI evaluation is source-referenced** — the internal-node solve is not
  shift-robust.
- **An NR iteration that limited is never accepted as converged**
  (`nr_limit_voltages`: fetlim/limvds/pnjlim, ±2.5 V window, anchors reset
  per sweep). `PYCIRCUITSIM_NR_TRACE=1` traces NR.
- **Transient refinement is opt-in and flags-off byte-identical**
  (`refine_output=True` / `PYCIRCUITSIM_TRAN_REFINE=1`): LTE on voltage AND
  per-device charge state, PULSE corners as breakpoints, ngspice dctran.c
  step-control semantics. `PYCIRCUITSIM_REFINE_TRACE` dumps a march trace.
  **ITL4 iteration-count control is measured dead on this corpus** (damped NR
  never exceeds 6 iterations) — do not re-derive it.
- **The AnalogGym corpus IS the tree** — `campaign.corpus()` enumerates
  whatever `examples/complex_circuits/designs_tsmc*/` holds; there is no
  selection flag. Curating the basket means deleting design directories and
  filtering the per-tree artifacts, never adding a filter to the driver.
  Three traps that outlive their sprint:
  - **Totals are not comparable across a curation.** `/159`, `/679`, `/795`,
    38/38, 17/17, 13/13 are the original corpus; `/75`, `/375`, 18/18 are
    V7.5.6; `/51`, `/255`, 12/12 are V7.5.9. Rescale or don't compare.
  - `cross_tech_report.py` **overwrites RESULTS_TSMC.md wholesale**,
    destroying its hand-written sections — splice its tables in, never run it
    blind.
  - **`designs_tsmc6` is an exact L72 duplicate of tsmc7** — re-measured at
    V7.5.9, 75/75 decks identical in verdict *and* in every miss's relative
    error to four decimals. It stays on disk because the NN families train
    separate checkpoints on it (the training-variance control), but running
    it in the **L72 bench** buys nothing and costs a fifth of the campaign.
    Score `tsmc5,tsmc7,tsmc12,tsmc16` and quote tsmc6 as the repeat.
- **`run_compare` must pin `AG_TREE` before the lazy `from meas import
  run_deck`** (`_pin_design_tree`). The shared `tools/` resolve their tech
  from `AG_TREE`/`AG_TECH`/cwd and RAISE otherwise, and the bench runs from
  the repository root, which is none of those — V7.5.8 shipped with every
  campaign deck dying at that import, undetected because no campaign was run
  between the refactor and V7.5.9. A module that resolves global state at
  import time needs its caller to set that state explicitly, not to happen to
  be in the right directory.

- **Prune on measured discrimination, not on structure.** A design that
  agrees with NGSPICE on every deck of every tech is paying for itself with
  nothing; a design that misses is the reason the corpus exists. V7.5.9's
  removals are the four fully-saturated designs plus the two whose
  distinguishing property another survivor now carries more cheaply — and
  the pass that justified them is also the pass that *rescued* two designs
  the structural argument had marked for removal. Measure first; the
  saturated set is not the set you would guess.

## Validation

**Ground truth is ALWAYS NGSPICE on the identical BSIM-CMG (LEVEL=72) OSDI
model. Never a simplified or self-defined equation, ever, for any purpose.**
The inverter transient in particular must pass against it.

Accuracy evidence lives in `docs/accuracy/` (index + scoreboard: its
`README.md`; cross-cutting gate definitions and the measured noise floor once
in `methodology.md`; retracted claims in `archive-pre-gds-fix.md`). Two traps
there:

- **Denominators changed in V7.3.0** — TSMC6 folded into the headline, so
  complex totals are /20, device AC /10, opamp AC /5 (older docs: /16, /8,
  /4). *No total crosses that boundary without rescaling.*
- Tables are **generated** by `scripts/v730_docs_build.py` from gate logs,
  each report pinned to one complete campaign pass with a committed-SHA guard
  so partial data cannot mix passes. Do not hand-edit them.
  `scripts/v730_coverage.py` emits gaps as a runnable job file. On this
  hardware only the DirectNet/BSIM-AR *clean* reports have local raw evidence.

**Subckt expansion is flattening at parse time** (internal nodes → `X1.n1`,
devices → `M.X1.Mp1`, ground global, `.model`/`.include` hoisted, loud errors
on unknown subckt / port mismatch / recursion >64). Legacy LEVEL=1 is removed.

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

> **`TECH=` is not optional on LEVEL≥73 — omitting it fails at parse time.**
> The parser defaults to `TECH=asap7` (`model_params.get('TECH') or "asap7"`),
> which maps to the **UNKNOWN embedding row** (a warning, not an error) *and*
> resolves the **universal-scope** stem `nmos_best.pt` / `ar_nmos_best.pt` —
> a checkpoint family not built on this hardware, so the run dies with
> `NN model not found`. This silently broke all five LEVEL=73/74 example
> decks until V7.5.7. ASAP7 is out of scope for the NN families (Rule 14),
> so an untagged NN deck is always wrong, never merely imprecise.

**`examples/` is the circuit library, and it is load-bearing** (V7.5.8): three
tiers — `single_devices/`, `simple_circuits/`, `complex_circuits/` (the
AnalogGym corpus) — named `<family>_<circuit>_<analysis>.{sp,cir}` so the
LEVEL 72/73/74 triplets sit adjacent, with `.cir` the NGSPICE ground-truth
half of a pair. **Editing a deck changes what a gate runs.** The `.cir` files
are templates carrying `<TOKEN>` placeholders; the `.sp` files are runnable
decks authored at one tech that the harness rewrites per run.

## Testing & Verification

Per-gate inventory, counts and how to run them: **README §Verification**.
Scores: `docs/accuracy/`. The rules that bind development:

- **Gates carry no netlists.** Every circuit a gate simulates is a deck under
  `examples/`, rendered by `tests.common.base.render_reference_deck` (V7.5.8).
  A topology that lives in both a deck and an f-string will drift, and the
  deck is the copy nobody re-runs. If you add a gate, add its deck.
- **One gate per question** (V7.5.9). A gate whose configs are a *subset* of
  another gate's matrix, or that differs from its neighbour only by a string
  argument, is not extra coverage — it is the same measurement paid for
  twice, and it dilutes the suite's signal. Merge behind a flag and record in
  the survivor's docstring what it absorbed, so the next reader does not
  re-add it. Same rule for `examples/`: a deck no gate reads is documentation,
  and it has to earn that on its own.
- **`examples/` and `tests/` share one taxonomy** — `single_devices/`,
  `simple_circuits/`, `complex_circuits/` — so a gate sits in the tier of the
  circuit it gates. `tests/perf/` and `tests/diag/` sit outside it
  deliberately: perf gates run no NGSPICE, and `diag_*` reference
  L72-in-PyCircuitSim rather than NGSPICE, which makes them **controls that
  must never be quoted as gate results**.
- **Gates are CPU-pinned** (`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1`) and honor `NGSPICE_BIN`; complex/AC infra pins torch to
  one thread (`PYCIRCUITSIM_TORCH_THREADS` overrides). This is not optional —
  the opamp gain and SRAM/VTC trips are high-gain fixed points and a
  multi-threaded GEMM reduction order moves the NR basin (~±1 % VTC scatter).
- **Quote TSMC6 separately.** Counts move with its registration: same
  coverage, one duplicate column.
- The SRAM `force_ic` probe is a printed **diagnostic**, not a gate.

---

## Development Guidelines

Architecture, entry points, design principles and the algorithm walkthroughs:
**README §Architecture / §Algorithms / §Python API**. Setup and tool paths:
**README §Installation**. What binds development beyond those:

- **Type hints on every signature**; clear names (`v_gate`, `i_drain`);
  docstrings for anything non-obvious. Voltage clamping Vgs±5 V, Vds±10 V.
- **The separation is a hard boundary, not a preference:** `solver.py` builds
  MNA and runs NR and contains **no device equations**; `models/` computes
  currents/conductances and does **no matrix ops**; `simulation.py`
  orchestrates. A device equation in the solver is the bug class this rule
  exists to prevent.
- Convergence: `|ΔV| < VNTOL + RELTOL × max(|V_old|,|V_new|)` (RELTOL=1e-4,
  VNTOL=1e-7); GMIN 1e-12 S; DC GMIN stepping opt-in
  (`use_gmin_stepping=True`).

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
