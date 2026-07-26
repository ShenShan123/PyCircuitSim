# PyCircuitSim — Detailed Changelog

This is the long-form history of PyCircuitSim. CLAUDE.md keeps a one-paragraph
"current state" summary; everything below is here so the conversation context
isn't burdened with chronology. (Compressed 2026-07-03, re-condensed 2026-07-23 —
every entry and verdict retained, verbose prose pruned; the full original text
lives in git history.)

---

## V7.0.0–V7.0.4 — NN compact-model performance (2026-07-25)

**Scan of the NN inference and training paths, then four infra changes.
Inference DC solve 1.68× with byte-identical output; training 4.9× per epoch;
BSIM-AR 1.6× behind an opt-in flag. Full measurements, routing and dead ends:
`docs/plans/2026-07-25-v700-nn-perf.md`.**

The governing constraint throughout: shipped checkpoints and gate results are
the product, and this repo has repeatedly watched a last-bit NN perturbation
land a different NR basin in a high-gain circuit. So every change is
classified **bit-identical** (ships default-on) or **perturbing** (ships
default-off behind an env flag, promoted only after a full 16-gate re-gate).

### V7.0.0 — the scan

Measured, not estimated. DirectNet `large` (907,565 params), 1 thread, batch
1 — the gate configuration.

**Inference is bandwidth-bound, not FLOP-bound.** 907k fp32 params = 3.6 MB;
one forward streams that in 290 µs ≈ 12.5 GB/s, at the single-core ceiling.
The shipped path streams the weights **four times** — 1 forward + 3 backwards
(id, qg, qd) = 1610 µs/eval. Profiling a 70-point NN inverter DC sweep:
`run_backward` 0.674 s tottime of a 2.64 s run, `batch_eval` 1.285 s cumtime.
**Half the simulation is the NN eval, and the backward engine is the largest
single line item.**

**Training is dominated by the data loader, not the model.** Per step at batch
2048: shipped `DataLoader(TensorDataset, num_workers=8)` 11.00 ms → GPU-resident
index slicing 3.29 ms → + fused AdamW **1.77 ms**.

### V7.0.1 — inference, bit-identical (DC solve 1.68×)

- **DC/OP skips the charge Jacobians.** The qg / qd autograd sweeps exist only
  to produce the 5 capacitances, which are read in exactly two places —
  `TransientSolver` and `ACSolver`. A `.dc` / `.op` run computed both and threw
  them away. `_caps_required` now defaults False and the two cap consumers
  declare it via `_require_nn_caps` at construction; `get_capacitances`
  self-heals (latch + recompute) for any other caller, so the flag is a
  performance hint and never a correctness precondition. 1610 → 784 µs/eval.
- **Per-eval Python overhead.** `_stats_col` did a `list.index` per scalar,
  13× per eval, to rediscover constants fixed at construction; `_denorm` /
  `_denorm_deriv` re-read numpy arrays per scalar. Both now use values resolved
  in `__init__`. The arithmetic is reproduced in its **original association
  order** — folding `out_std / in_std` would round differently — and `np.sinh`
  is deliberately kept over `math.sinh` (libm and numpy may disagree in the
  last ulp, and this feeds every stamped current). `math.sqrt` does replace
  `np.sqrt`: IEEE-correctly-rounded, so bit-identical.
- **`_mosfet_types()` / `_pmos_types()` / `_nn_mosfet_types()` memoized.** They
  re-ran four `try: import` blocks on every call — 4462 + 2596 calls in a
  70-point sweep.

Verified: DC, transient and AC CSVs **sha256-identical** to the pre-change
baseline. Gates: NN inverter 2/2, `verify_nn_ac` 8/8, `verify_complex_ring_osc`
4/4, L72 op 3/3 + dc 2/2 + tran 1/1, `verify_subckt` 11/11.

### V7.0.2 — training, 4.9× per epoch (no shipped checkpoint touched)

- **`_DeviceBatches`** — the splits are already in-memory tensors, but were
  wrapped in a `TensorDataset` behind a `DataLoader` with 8 workers: 2048
  `__getitem__` calls, a collate and an IPC copy per batch, to deliver one
  contiguous slice of a tensor that already existed. Now a permutation + index
  slice on-device. `_pick_loader` checks free GPU memory (needs < 50 % of free)
  and falls back to the `DataLoader` otherwise; `BSIMAR_LOADER=torch|device`
  overrides.
- **Fused AdamW** on CUDA — the step is launch-overhead-bound at these sizes.
- **No per-step host sync** — losses accumulate on-device, `.item()` once per
  epoch instead of ~800 times.

Measured on the real CLI (DirectNet large, TSMC5, 640k train rows, incl.
validation): **3.4 s/epoch → 0.7 s/epoch**. DirectNet, Transformer and TabPFN
all train through the shared loop.

These change the shuffle order, so a **re-train** produces different weights.
No checkpoint on disk is touched, so no gate moves until someone retrains, and
`BSIMAR_LOADER=torch` reproduces the legacy path.

### V7.0.3 — fused analytic Jacobian, opt-in, DEFAULT OFF

`DirectNet.forward_with_jacobian` propagates the Jacobian *forward* in closed
form alongside the value — carry `[h ; ∂h/∂v]` as 5 rows, one GEMM per layer,
one weight stream instead of four — returning the full 13×4 Jacobian.
Behind `PYCIRCUITSIM_NN_FUSED_JAC=1`. Falls back to autograd for the EKV-core
and monotone-residual variants, which re-compose the `id` column.

**Scope correction found during implementation:** the isolated 2.16× is against
the *old* 3-backward path. After V7.0.1 a DC eval is already 784 µs vs the
fused pass's 748 µs, and the fused pass computes the whole Jacobian whether or
not caps are wanted — so it cannot exploit DC mode. End-to-end: **DC 1.48 →
1.64 s (slightly slower), transient 3.04 → 2.20 s (1.38× faster)**. It is a
transient/AC lever only; I1 is strictly better for DC.

Not bit-identical (same math, different summation order): max |ΔV| = 7e-7 (DC)
and 4e-7 (transient) on the inverter — but the inverter is the benign case, and
the opamps are what a re-gate must clear. **Stays off until that re-gate runs.**
With the flag off, the three CSVs remain sha256-identical.

### V7.0.4 — LEVEL=74 AR prefix cache, opt-in, DEFAULT OFF

The BSIM-AR inference loop re-ran the **entire encoder over the whole growing
prefix, once per AR step** — 8 passes over 4, 5, … 11 tokens = 60 token-passes
to produce 11 hidden states. Under the causal mask a prefix hidden state cannot
change once computed, so 49 of the 60 were pure waste. `_encoder_append` now
streams each token through the stack once, keeping per-layer K/V so the new
token still attends to the whole prefix. Behind `PYCIRCUITSIM_NN_AR_CACHE=1`.

Delivered **1.60× DC / 1.56× transient / 1.21× AC** end-to-end, and **4.3× at
batch 2048** in the trainer's `no_grad` AR metrics (that regime is genuinely
FLOP-bound). Gains grow with device count — 2.06× at batch 4.

**Routing correction: I4 was classified "exact in exact arithmetic"; the
bit-wise verification refuted it, and the cause is a hard floor.** `F.linear`
is not row-stable on this CPU — a 1-row GEMV accumulates in a different order
than the same row of an L-row GEMM (measured **0/96 shapes stable**, up to
8e-6 abs). *Any* formulation computing fewer rows than the stock recompute
moves the last bits, however attention is arranged. That also kills the
rescue of caching Q/K/V and replaying the identical full-prefix SDPA call:
attention is only ~1 % of the cost, and the projections holding the other
99 % are exactly what cannot be shrunk exactly. So I4 ships default-off like
V7.0.3, promoted only on a full 16-gate `MODEL=transformer` re-gate.

Deviation is small but real: outputs ≤ 5.3e-6, autograd Jacobians ≤ 1.6e-6,
solved node voltages ≤ 1.6 µV.

**Why 1.6× and not the 5.45× the token count implies:** the encoder is
weight-bandwidth-bound like DirectNet, and the weight stream is paid *per
encoder call, not per token* — 8 AR steps means 8 streams either way. The
cache removes the redundant arithmetic, not the traffic, and the 8 sequential
streams are the autoregressive data dependence itself.

Two PyTorch behaviours found and recorded:

- `nn.TransformerEncoder` runs `_detect_is_causal_mask` on the mask it is
  given and forwards `is_causal=True` with `attn_mask=None`, so the stock call
  reaches SDPA's **fused causal** kernel. Rebuilding an equivalent triangular
  mask selects the math kernel and shifts the primer by 7e-7 for nothing;
  `_encoder_append` matches the hint.
- `TransformerEncoderLayer.forward` has a second fused fast path
  (`torch._transformer_encoder_layer_fwd`) that fires **only under `no_grad`** —
  it is disqualified whenever grad is enabled and a parameter requires grad,
  because the fused op is not differentiable. Every simulator eval wraps the
  forward in `torch.enable_grad()` for the Jacobian, so the shipped LEVEL=74
  surface has never touched it.

Verified — flag off: LEVEL=74 DC/tran/AC CSVs **sha256-identical** to the
pre-change baseline; L72 op 3/3 + dc 2/2 + tran 1/1, `verify_ac` 2/2,
`verify_subckt` 11/11, DirectNet L73 inverter **8/8** across four techs.
Flag on: LEVEL=74 TSMC5 inverter gate vs NGSPICE **2/2 PASS with identical
scores** (VTC 1.09 %, inverter transient 0.97 %), and the LEVEL=74 device AC
gate — the derivative-sensitive one, whose caps are the autograd dQ/dV of the
predicted charges — **2/2 PASS with every metric identical to the last printed
digit** (NMOS 0.435 dB / f3db 1.259 / 4.67 % / 67.40°; PMOS 0.189 dB / 0.891 /
2.08 % / 79.58°). The perturbation is far below gate resolution.

### Dead ends (measured, rejected — do not retry)

- **TF32** 1.77 → 1.92 ms/step, **`torch.compile`** → 2.00, **bf16 autocast**
  → 2.61. All *slower*: DirectNet is launch-overhead-bound, so every one of
  them adds conversion or guard work to a matmul that was never the cost.
  (Worth re-testing for the Transformer/PFN families, which are genuinely
  compute-bound.)
- **Replica-batch trick for the 3 inference gradients** — replicate the input
  3× and take one backward with a column-selecting seed. Only 1.53× (vs 2.16×
  for the analytic Jacobian) **and** not bit-identical: GEMV→GEMM changes the
  summation order, grads differ 4e-6 at N=1 and 1.8e-4 at N=10. Strictly
  dominated.
- **Larger training batch** (8192/16384 → 0.43 s/epoch, the largest raw number
  measured) is a *recipe* change, not infra: it needs LR rescaling and would
  invalidate every recipe comparison. Deliberately excluded.

### Open / not started

**I4 — the LEVEL=74 AR prefix cache.** `TransformerEncoderModel`'s inference
branch re-runs the *entire* encoder over the whole growing prefix 8 times, with
no KV cache. Under a causal mask the prefix hidden states cannot change once
computed, so the recompute is pure waste — and it is why BSIM-AR is 30–100×
slower than DirectNet. Estimated 3–4×. Requires hand-rolling incremental
attention against `nn.TransformerEncoderLayer`'s pre-LN math, so it is real
work and needs bit-level verification before it could ship default-on.

---

## V6.13.1 — systematic-audit fix wave 1: 22 gate-neutral findings (2026-07-24)

**Closed the 22 findings from `docs/2026-07-21-systematic-audit.md` that cannot
change a number any gate reports. The remaining 19 are gate-affecting and are
staged on `audit-fixes-wave2` behind a re-gate — triage, evidence and ordering
in `docs/plans/2026-07-24-audit-fix-waves.md`.**

All 43 open P1/P2 findings were first re-verified against `d2ea720`: re-located
in the current source (many line citations had drifted), re-reproduced where a
pure-Python repro was possible, and split by whether the fix can move a gate
number. **Three were dropped on the evidence** — see the dead-end note below.

Verified on the patched tree: `verify_bsimcmg_op` 3/3, `verify_bsimcmg_dc` 2/2,
`verify_subckt` **11/11** with Levels 1-3 byte-identical to baseline
(`max|dV| = 0.000e+00`, L72 inverter NRMSE 0.187 % of VDD).

### The silent-wrong class (parser)

- **C1** — `+` continuation lines were buffered only after `.model`. After any
  other line the fragment flushed as a space-prefixed orphan and `parse_line`
  dropped it with no error, losing X-instance params, `.ic` assignments, MOSFET
  geometry and **`AC=` stimulus** (a full Bode sweep driven by a zero source).
  Now every logical line buffers; an orphan `+` raises. Replayed HEAD vs patched
  over **4442** deck/library files: 5 differ, all raw TSMC PDK cards this parser
  never opens, and 0 raise.
- **C6i** — ground globality was case-sensitive, so a lowercase `gnd` inside a
  `.subckt` became a node tied only to GMIN. Canonicalized at every
  node-ingestion point. A `.subckt` may no longer declare a ground-named port,
  which the canonicalization would otherwise have turned into a new silent short.
- **C6h** — duplicate `X` instance names silently merged both instances'
  internal nodes onto one net. Keyed on the full flattened path, so nested
  `Xbuf.X1` / `Xbuf2.X1` stay distinct.
- **C6j** — duplicate `.model` names overwrote **retroactively** (the pre-pass
  resolves models before devices), changing polarity and LEVEL for devices
  written above. A conflicting redefinition now raises; an identical one still
  passes, so a doubly-`.include`d library stays legal.
- **C6l / C6m** — an env pin ending in the opposite polarity was ignored and
  fell through to production `large`; the UNKNOWN-tech-code warning covered only
  the universal scope, so a per-tech scope silently used the UNKNOWN embedding
  slot. Both are now loud (`PYCIRCUITSIM_NN_STRICT_TECH_CODE=1` escalates the
  latter to an error). Measured inert against every resolver line on disk.

### The silent-green class (test harness and dispatchers)

- **B3** — 11 dispatchers exited 0 regardless of sub-job outcome: workers ended
  in `exit 0` so `xargs`'s 123 never fired, and a trailing `echo` overwrote `$?`.
  Workers now exit 3 for "reached no verdict" (never 255, which aborts `xargs`),
  and dispatchers capture the status before printing and check that every
  launched cell recorded a row.
- **B4** — `benchmark_collect.py` fabricated tier names and counted only cells
  whose log existed, so an empty tree published a report and exited 0.
- **B5c** — `verify_complex_opamp` had no minimum-gain guard: `ng_gain=0.30` vs
  `dn_gain=0.31` passed at 3.3 %, certifying an opamp with no gain. The sweep
  harness had `OPAMP_MIN_GAIN=5.0`; the ship gate now does too.
- **B5l** — a typo'd `--tech` SKIPped silently, so `TSMC5,TSMC7X` scored 1/1 and
  exited 0. Closed at all six sites.
- **B5g** — the three V6.12.0 loud subckt errors (unknown subckt, port-count
  mismatch, recursion >64) had **zero** tests: deleting all three raises still
  reported 8/8. A Level 0 batch now covers them, and the gate's own total moves
  **8/8 → 11/11** (Levels 1-3 unchanged).
- **B5h / B5j** — `diag_l72_complex_control` took the reassuring branch on NaN,
  so a ring that never oscillated printed "L72 matches NGSPICE";
  `diag_l72_switchcap_control` printed ">> the solver is faithful"
  unconditionally. Per MEMORY.md these verdicts routed whole campaigns.
- **B5a** — `train_per_tech_8cells.sh` wrote into **production** checkpoint slots
  (no `--exp-name`, with `--overwrite`) while omitting the clean-recipe flags,
  and CLAUDE.md advertised it as a convenience sweep. Deleted; superseded by
  `benchmark_train_sml.sh` with `TECHS`/`SIZES`.
- **B5d / B5e / B5i / B5n / B5m** — pinned-but-absent checkpoint became a silent
  SKIP; the gate printed a checkpoint it had not scored; `--idvds-diagnostic`
  was green with no NGSPICE comparison and set `passed=True` on an empty mask;
  `--techs TSMC4` reported "0/0 PASSED"; and the `===BENCH_DONE no-ckpt===`
  marker contained the resume sentinel, so it skipped its own cell forever even
  after the checkpoint was trained.

### Data-pipeline integrity

- **C6o** — the tech-variant label sidecar was validated by **row count alone**,
  and Rule 1 explicitly invites regenerating datasets. Adds a geometry-sha256
  `.meta.npz` fingerprint written by every producer — `loo_labels` plus the
  concat / append / subsample scripts, which had been leaving every universal
  dataset on the weakest validation path. `benchmark_gen_data.sh` retires the
  stale sidecar when it rewrites a dataset, so a regen cannot wedge the next
  training run.
- **C6q / C6r** — `tech_scope_vocab_size("universal")` disagreed with
  `NUM_TOTAL_CODES` (the range is now guarded rather than the vocabulary
  enlarged, which would have changed the `nn.Embedding` row count of every
  universal checkpoint); TabPFN's `_ctx_cache` was not invalidated by
  `load_state_dict` or EMA in-place writes (measured drift 4.0e-2), and was safe
  by call ordering only.

### Dead ends recorded

- **C2 dropped, and the audit's own prescribed fix for it is a hazard.** The
  NaN mechanism (`torch.where` evaluating both branches of the softplus voltage
  clamp) is real, but the clamp sits in no autograd graph — both callers cut the
  graph after it — so it cannot fire. And the prescribed
  `F.softplus(bx, beta=1)/beta` rewrite is measurably **not** forward
  bit-identical: 23 of 401 samples differ in the last fp32 bit, because the
  linear branch becomes `bx/beta` instead of `v_raw - v_min`. On a codebase
  whose opamp and ring basins are documented as sensitive at that magnitude that
  is a gate-affecting change in exchange for nothing. The argument-clamp variant
  measured bit-identical at 401/401 if hardening is ever wanted.
- **B5k and C6p dropped** — both closed in passing by the TSMC6 retire.

### Not included

`scripts/gate_matrix_iso.sh` is the 12th B3 dispatcher and was driving the live
V6.13.0 re-gate while this wave was written, so it was deliberately left
byte-identical. It gets the same treatment now that the campaign has landed.

---

## V6.13.0 — gds sign + guard fix shipped, TSMC6 retired, every checkpoint re-gated (2026-07-24)

**Shipped the audit's last P0 (`A3`, the NN `gds` sign bug), retired TSMC6 as a
duplicate technology, then re-gated all 36 checkpoint sets on disk against
NGSPICE. The fix moves the production DirectNet matrix 14 → 15/16 strict, closes
`tsmc16-opamp`, eliminates every observed OMP flip in the universal set, and
retracts the project's standing claim that `tsmc7-opamp` is unreachable.**

### The fix (`8ed35bd`)

Inference negated `gm` and `gmb` but not `gds`. All three are derivatives of the
same signed `id`, so the sign comes from `id`'s convention, not from which
variable is differentiated — `losses/bni_mae.py` had negated all three since
V6.4 and explicitly documented commit `930c274`'s "gds is the diagonal so no
flip" rule as wrong for this stored convention. The correction never reached
inference. Measured: autograd `d(id)/dVd` vs `-gds_head` = 0.12 rel err, vs
`+gds_head` = 2.08, and 2.0 is the arithmetic signature of a pure sign flip.

The two-sided floor `max(gds, |id|*0.5)` asserted an Early voltage <= 2 V, below
the true median of every device (OSDI amplifying p50 3.3-9.8 V), overriding the
learned output conductance at 90.9 % of amplifying points — and was load-bearing
only because it masked the sign error. Replaced by **guard F**: positives pass
through bit-identical to the raw Jacobian, negatives clamp to `|id|/50 V`. 50 V
sits above the measured maximum true Early voltage (43.4 V), so the guard can
never bind on a physically correct value. OSDI `-d(id)/dVd` is positive at
100.0000 % of conducting points across 111,630 evals on all 10 production
devices, so every negative is model error.

`_floor_gds` -> `_guard_gds`; `PYCIRCUITSIM_GDS_FLOOR_K` is now rejected loudly
rather than silently reinterpreted; the new knob is `PYCIRCUITSIM_GDS_GUARD_K`
(default 0.02). **The sign and the guard must ship together** — measured, the
sign alone is bit-identical and the guard alone regresses device AC 8/10 -> 5/10.

### TSMC6 retired (`38c47d8` + PyCMG `23b0ace`)

Audit §D1: TSMC6 is TSMC7 relabelled under BSIM-CMG — bit-identical datasets and
identical L72 currents, because the differing PDK keys are TMI-proprietary and
absent from the BSIM-CMG Verilog-A. Deleted 22 checkpoints, the datasets,
`results/tsmc6_gate`, and every registry/driver/test entry. Its codes were the
tail (22-24) so nothing renumbered; `NUM_TOTAL_CODES` 25 -> 22. Added
`bsimar.config.assert_tech_is_distinct()`, which compares resolved modelcards on
BSIM-CMG-implemented parameters only — it flags tsmc6 <-> tsmc7 and confirms
5/7/12/16 distinct. The raw vendor PDK is kept (IP) but unreferenced.

Consequence for the L72 suites: the counts drop with the duplicate column, same
coverage. DC L2 81 -> 67, DC L3 53 -> 44, tran L2 45 -> 37, tran L3 86 -> 72.
CLAUDE.md and README carried the old numbers and are corrected.

### Re-gate — what moved

Every number below is single-run OMP=1 unless stated strict, measured at
`d2ea720`, and directly comparable to the pre-fix single-run columns in the
accuracy reports at `a96112a`.

**One signature dominates: every gained cell is an opamp.** Across DirectNet's
four sizes, PFN's three and BSIM-AR's four clean tiers, not one ring, SRAM or
switchcap cell moved. `gds` sets small-signal output resistance — what a
high-gain operating point is made of — and cancels at the Newton fixed point
everywhere else.

- **DirectNet production** (`tsmc{X}_dn_large_*`, crit30 weights): 14 -> **15/16
  strict, zero flips**. `tsmc16-opamp` closes at 7.69 %; `tsmc7-opamp` is the
  sole open cell and only at `large`. Thinnest margin in the matrix is
  `tsmc5-opamp` at 9.54 % against a 10 % gate.
- **`crit15m@xl` = 16/16 STRICT, zero flips** — the first DirectNet checkpoint
  set ever to sweep the full matrix under one uniform recipe, holding all four
  opamp basins including tsmc7 (7.56 %). Not promoted: 2.13 M params, 2.3x
  inference cost, no device-fidelity gain. Documented as an env-pin alternate.
- **DirectNet clean capacity curve** (s/m/l/xl): 7/10/13/10 -> **10/10/13/12**.
- **Universal DirectNet**: net **+3** strict cells and **all three pre-existing
  OMP FLIPs eliminated** — every one of the 8 stems is now flip-free. That is
  the more durable half: a FLIP means the verdict depended on GEMM thread count,
  and removing a wrong-signed Jacobian entry removed that sensitivity.
- **PFN**: 11/11/9 (was 11/10/8); device AC at `large` 5/8 -> **8/8**.
- **BSIM-AR clean**: **14/16 at every tier**, failing the same two cells at every
  tier (was 12/14/13/13). The published "capacity peaks at medium" was largely
  this bug. All four opamps now pass at all four sizes (0.55-8.53 % against a
  10 % gate); **the ceiling moved opamp -> ring**, and the two low-VDD rings are
  not close (5.55-12.55 % against a 5 % gate, worsening with capacity).
- **BSIM-AR corridor recipes**: `corroft@medium` and `corro15@xl` are **16/16
  STRICT, zero flips** — the second and third checkpoint sets in the project to
  sweep the full matrix, after DirectNet's `crit15m@xl`. `corroft@medium` holds
  every cell with room: opamp 2.52-6.73 % against a 10 % gate, ring 2.13-3.33 %
  against 5 %, switchcap 1.99-4.15 %, worst SRAM lobe 6.11 %.
- **Every corridor recipe at `xl` sweeps the matrix.** corroft, crit15m, crit30
  and corro15 are all 16/16 single-run at xl (pre-fix 15/15/14/15); at `large`
  the same four all sit at 15/16 missing only tsmc7-opamp. The corridor's effect
  turns out to be uniform, not recipe-specific — the opposite of the pre-fix
  reading in which recipe choice appeared to decide which opamp basin you got.
  The non-corridor recipes (clean, csob) gain only +1 and stay stuck on the two
  low-VDD rings: **gds moved opamps, the corridor moves rings, and the two
  levers are independent.**

**Device / parametric / L72 suites:** device AC DirectNet **8/8** and PFN **8/8**
(both were railed on cells that now pass); `verify_nn_dc_tran` 24/24;
`verify_nn_lifted_source_dc` 12/12 (Rule 2 canary); parametric tran 64/64;
parametric DC 54/55 — and the single failure (`TSMC12_pmos_nfin_10`, NRMSE
16.15 %) is **bit-identical** pre- and post-fix, which is the confirmation at
scale that DC is exactly invariant rather than a hidden regression. All L72
controls pass; `verify_multi_tech_dc` reports 43 PASS + 1 pre-existing ERROR
(`TSMC5_lvt_inv_l_24nm`, internal-node NR divergence at d=2559 V in the pure
BSIM-CMG path, unrelated to the NN surface — and the suite still exits 0, which
is audit B3).

`verify_complex_opamp_ac` is 0/4: the operating point un-rails, but no tech meets
DC-gain and GBW and phase-margin and magnitude together. TSMC5 is closest
(3.31 dB vs a 3 dB gate, GBW 0.94x, PM 18.9° vs 15°). Pre-fix this suite read
0/12 with nothing close.

### What this retracts

`tsmc7-opamp` was documented across CLAUDE.md, all three accuracy reports and
several memories as **the universal ceiling cell, reachable only by the V6.5.9
T3 differentiable-DC-solver fine-tune**. It is not. DirectNet passes it at
`small` (1.81 %) and `xl` (4.20 %), `crit15m@xl` passes it strict, and BSIM-AR
passes it at every size. The gds floor was masking a railed operating point.
Also retracted: DirectNet §6.2's "the three-basin simultaneous hold is the open
15/16 target", and BSIM-AR's capacity-peak-at-medium reading.

### Also in this version

- `0126c44` — `SobolevIdLoss` now raises rather than silently supervising the
  reverse-corridor-corrupted `gds` column (audit A3-data). Latent: `--sobolev`
  is default-off.
- `88db8f3` — `uni_gate_sweep.sh` was truncating `SUMMARY.tsv` on every dispatch,
  erasing other sections' verdicts (same class as the `gate_matrix_iso` clobber
  fixed in `65b7764`). `UNI_OUT` is now overridable.
- `1559583` — the collector's `dn_large` baseline slot carries crit30 weights,
  not clean; its pre-fix baseline is crit30f's 14/16, and the genuine clean
  `large` lives at `dn_v660clean_large`.
- New campaign tooling: `scripts/a3_regate_collect.py`,
  `scripts/a3_regate_uni_collect.py`, `scripts/a3_regate_omp_collect.py`
  (strict/OMP verdicts), `scripts/a3_omp_one.sh` (resume one OMP value of an
  interrupted multirun sweep instead of re-paying for the runs it already
  banked — worth ~40 BSIM-AR gate-hours on this campaign alone).

### Methodology note — the campaign ran off a frozen snapshot

The resumed half of the re-gate ran from an rsync of the repo at `d2ea720` with
`checkpoints/`, `tools/` and `PyCMG/` symlinked back, verified byte-identical for
`pycircuitsim/`, `tests/` and the `bsimar` package. That decouples an 8-hour gate
campaign from the working tree, so documentation and bug-fix edits could land
concurrently without any chance of changing the numerics half-way through. Worth
repeating for any future long campaign.

### Audit disposition

The remaining 43 P1/P2 findings were re-verified against `d2ea720`, three were
dropped on the evidence, and the rest split into a gate-neutral wave 1 and a
gate-affecting wave 2 — see `docs/plans/2026-07-24-audit-fix-waves.md`.

**Dead end recorded: audit C2 was dropped, and the audit's prescribed fix for it
is itself a hazard.** The NaN mechanism (`torch.where` evaluating both branches
of the softplus voltage clamp) is real, but the clamp sits in no autograd graph —
both callers cut the graph after it — so it cannot fire. And the prescribed
`F.softplus(bx, beta=1)/beta` rewrite is measurably *not* forward-bit-identical:
23 of 401 samples differ in the last fp32 bit, because the linear branch becomes
`bx/beta` instead of `v_raw - v_min`. On a codebase whose opamp and ring basins
are documented as sensitive at that magnitude, that is a gate-affecting change in
exchange for nothing. If defence-in-depth is wanted later, the argument-clamp
variant measured bit-identical at 401/401.

---

## V6.12.1 — silent-green P0 branch merged + accuracy reports consolidated per family (2026-07-24)

**Merged `fix/silent-green-p0` into `main` (fast-forward, 9 commits) and replaced the
seven per-version accuracy reports with three unified per-family reports under
`docs/accuracy/`.**

**Merged from the branch (already recorded in their own commits + the audit register):**

- **P0 silent-green fixes** (`e756481`): `tests/common/base.py` never checked NGSPICE's
  exit status, so a dead binary left the previous run's CSV in a never-cleaned work dir
  and every downstream check passed against **stale ground truth** (reproduced:
  `NGSPICE_BIN=/bin/false tests/verify_subckt.py` → 8/8 PASS with byte-identical NRMSE).
  Fix: unlink CSV+log before invoking, raise on non-zero return. And `solver.py` —
  `spsolve` returns NaN rather than raising on a singular matrix, so every
  `except LinAlgError` guard on the sparse path was dead code; now detected and raised.
- **`docs/2026-07-21-systematic-audit.md`** — the 5-area, ~70-finding audit register
  (solver, parser/models, data generation, NN architecture, test infra), including the
  **gds sign bug** root-cause (§A3: inference negates gm/gmb but not gds; `_floor_gds`
  then discards the learned output conductance at 98–100 % of conducting points),
  its measured 4-arm impact (device AC 8/10 → 10/10, complex 17/20 → 18/20, `force_ic`
  2/5 → 5/5, DC exactly invariant, transient mean NRMSE −19 %), and **§D1 TSMC6 ≡ TSMC7
  relabelled** (bit-identical datasets; the differing PDK keys are TMI-proprietary and
  absent from the BSIM-CMG Verilog-A). **The gds fix is NOT shipped** — sign and floor
  are coupled, and the floor value still needs a k-scan.
- **V6.8.1 BSIM-AR xl-tier fill** documentation + `docs/plans/2026-07-07-bsimar-transformer-xl-fill.md`.

**Accuracy-report consolidation (this entry's new work):**

- New `docs/accuracy/`: `DirectNet-L73-accuracy.md` (V6.6.0 baseline + capacity curve,
  V6.6.1 recipe study, V6.6.2–4 curriculum → production 14/16, V6.6.5/6 cross-tier + xl,
  V6.6.7 negative round, V6.7.0 universal + TSMC5 transfer, V6.11.0 TSMC6, plus **Part II**
  = the frozen collector tables A/B/C verbatim), `BSIM-AR-L74-accuracy.md`
  (V6.8.0 + V6.8.1 xl + TSMC6), `PFN-L75-accuracy.md` (V6.10.0 + TSMC6), and a
  `README.md` index.
- **Deleted** (content fully carried; git history retains them at `1fe1cdb`):
  `V6.6.0-accuracy-report.md`, `V6.6.1-recipe-accuracy-report.md`,
  `V6.6.6-accuracy-report.md`, `V6.7.0-universal-transfer-report.md`,
  `V6.8.0-bsimar-transformer-report.md`, `V6.10.0-tabpfn-pfn-report.md`. Every
  reference in CLAUDE.md / README.md / this file / `docs/plans/` /
  `scripts/recipe_retest_collect.py` was redirected.
  `V6.9.0-tsmc6-onboarding-pdk-parse-audit.md` stays — it is a tech-onboarding + PDK
  audit record, not an NN accuracy report.
- Each unified report now carries the two cross-cutting caveats explicitly: **every
  published number was measured with the gds sign bug present**, and **TSMC6 rows are a
  second training run on the TSMC7 data**, not a sixth technology.

**Also:** `scripts/recipe_train.sh` — balanced round-robin. The GPU index now advances
only for jobs that will actually train; an already-`.complete` job used to consume its
slot, so a resume where one tech was finished starved a GPU (observed 3:3:1 on the
reboot-resume of the xl clean wave). A fresh run skips nothing → byte-identical
assignment. GPU choice never affects results (seeds pinned); this only rebalances
wall-clock.

---

## V6.12.0 — .subckt/.ends hierarchical netlists + hierarchical test-circuit conversion (2026-07-18)

**Added `.subckt`/`.ends` + hierarchical `X` instances to the parser, extended
`.ic` into subckt bodies, converted the project's test circuits to hierarchy, and
re-ran the full matrix: zero regressions (commit 1744b28; merged to `main`
2026-07-18 via `3dadb34`, carrying V6.11.0+V6.12.0 together).**

**Feature (`parser.py`, flattening expansion at parse time):** `.subckt <name>
<ports…> [param=default …]` … `.ends`; instances `X<id> <nodes…> <subckt>
[param=val …]`. Nested defs + X-in-X; defs registered globally (ngspice-flat
scoping); recursion guarded (depth 64, loud error). Flattening: internal nodes →
`X1.n1` (nested `X1.X2.n1`), devices → `M.X1.Mp1` (type char preserved for
first-char dispatch); ground `0`/`GND` global; ports map to connecting nodes so
top-level names are unchanged (node names are opaque strings — no circuit/solver
change needed). Params: card defaults, per-instance overrides, body refs as bare
names or `{expr}`/`'expr'` arithmetic (+ - * /, unit suffixes); `.ic` in bodies
node-remapped AND param-resolved, top-level `.ic V(X1.n1)=v` reaches internals,
`uic`/`force_ic` consume unchanged; `.model`/`.include` hoisted global; other
in-body directives / unknown subckt / port-count mismatch = loud errors.
Robustness fixes riding along: `.include` save/restores the current file (sibling
includes resolve); `.ic`/`.dc`/`.tran`/`.ac` dispatch + `V(...)` matching
case-insensitive.

**New gate `tests/verify_subckt.py` — 8/8 PASS:** L1 subckt≡hand-flattened at
max|ΔV|=0 (RC .tran/.ac, nested {expr}-param DC OP, `.ic`-in-body + uic); L2 L72
inverter-in-subckt ≡ flat (max|ΔV|=0), 0.187% NRMSE vs NGSPICE; L3 nested
2-inverter buffer (X-in-X, NFIN params, `.ic` on internal `Xbuf.m`) 0.64/0.86%.

**Test-circuit conversion** (probed nodes stay top-level as ports → harness
keys/baselines unchanged; NGSPICE refs + single-device Id-Vgs decks stay flat):
inverter VTC/tran generators, all 4 complex builders + `examples/complex`
templates (opamp `ota1`+`cs2`, ring `ringinv`×5, switchcap `ckinv`+`tgate`, SRAM
`sraminv`), NN inverter + CS-amp AC decks, 7 examples. Sweep equivalence canaries
PASS all 5 techs.

**Full re-run (2026-07-18, CPU-pinned):** subckt 8/8; bsimcmg OP/DC-L1/tran-L1/AC
all PASS (tran 0.19%); DC L2 **81/81**, tran L2 **45/45** (grew with TSMC6); DC L3
**52/53 + 1 ERROR** (documented `TSMC5_lvt_inv_l_24nm` OSDI divergence), tran L3
**86/86**; verify_nn_dc_tran **24/24**; sweep canaries ALL PASS; complex ring 5/5,
opamp 2/5 (tsmc7/16 crit30 misses + tsmc6 clean-large miss rail as documented),
switchcap 5/5, SRAM SNM 5/5; lifted-source 15/15; nn_multi_tech_dc **68/69** (1
FAIL `TSMC12_pmos_nfin_10` reproduced BIT-IDENTICALLY pre-feature = pre-existing
DN off-grid-NFIN corner), nn_multi_tech_tran 80/80. **Complex matrix 17/20 =
exactly V6.11.0 production state** (legacy 14/16 + TSMC6 3/4); **484/489 checks**,
all 5 non-passes pre-existing/documented. Hierarchical decks are numerically
transparent — no pass-rate moved either direction. (Complex sweeps / NN AC / opamp
AC not re-run — canaries pin their line-set equivalence.) README brought up to
date in the same pass (was stranded pre-V6.5, still listed `.subckt` unsupported).

**Three pre-existing CLI/solver defects surfaced while smoke-testing the README
commands — NOT regressions (all reproduce on the pre-merge flat decks at
6b3a890). All FIXED, baselines bit-identical after:**

1. **Mislabeled point counts** — `simulation.py` logged `len(results)` (a
   trace-keyed dict) as "time points", so a 3-node inverter reported "3 time
   points" after integrating 502 steps. Now reports points *and* trace count
   via `_sweep_point_count()` (`502 time points (3 traces)`).
2. **Transient wrote no numerical data** — `run_transient` emitted only the
   `.png`; now also writes `_transient.csv` (stdlib csv, `Time (s)` + one column
   per trace) and a run-summary `_simulation.lis` (header/circuit/final state;
   `TransientSolver` has no per-step logger, iteration detail via `debug=True`).
   `Logger.log_final_results` gained `sweep_label`/`point_label`/`final_label`
   kwargs so transient isn't labeled "DC Sweep"; defaults byte-identical.
3. **Duplicated final transient sample** — `num_steps = ceil(t_stop/dt)+1` with
   `5e-9/1e-11 = 500.00000000000006` (IEEE-754) → `ceil` 501 → 502 steps, the
   extra one clamped to `t_stop` and duplicating the end row. Fixed by snapping
   the quotient to the nearest integer within rel-eps 1e-9, `ceil` only for a
   genuinely partial final step. Only float-error-integer decks change (500.0…6
   → 500); exact (550.0) and fractional (1666.667→ceil) ratios bit-identical.
   Inverter now yields 501 unique timepoints ending exactly at 5e-9.

Re-run after all fixes: op/dc/tran L1 PASS, AC 2/2, subckt 8/8, lifted-source
15/15; transient L1 1/1 (0.19%/7.6 mV), L2 **45/45**, L3 **86/86** — every
quoted ASAP7 row reproduces exactly (baseline 0.19/7.6, vdd_0p8 0.21/12.9,
cload_1fF 0.84/42.0, cload_100fF 0.02/0.9). No gate touches `main.py`;
`run_dc_sweep`'s return contract was left untouched. README correction: L2 = 45
(not 37), L3 = 86; removed nonexistent configs (`vdd_0p5`, `nfin_20`).

---

## V6.11.0 — TSMC6 NN family: all three NN compact models trained + gated at every scale (2026-07-14/17)

**Completed the V6.9.0 NN-training deferral: trained + NGSPICE-gated all three NN
families (DN/BSIM-AR/PFN) at every scale on the 6th tech TSMC6 (CLN6). 22
clean-recipe ckpts (DN s/m/l/xl + PFN s/m/l + TF s/m/l/xl, each N/P), one control
recipe (`--apply-filter off --swa-mode ema --seed 42`), per-tech local vocab
(tsmc6 svt/lvt/ulvt = 0/1/2 + UNKNOWN; nn_vt=ulvt). Reports updated (DN
V6.6.6 / TF V6.8.0 / PFN V6.10.0).** ⚠ Note: `MEMORY.md` later found TSMC6 data
≡ TSMC7 (relabelled) — so the opamp-XOR-ring split below is a basin coin-flip, not
distinct-tech fidelity.

**Result — TSMC6 complex 4-cell matrix (ring/opamp/sram/switchcap) vs NGSPICE:**

| family | small | medium | large | xl |
|---|---|---|---|---|
| DirectNet | 1/4 | 2/4 | **3/4** | 2/4 |
| PFN       | 2/4 | 2/4 | 2/4 | — |
| BSIM-AR   | 2/4 | **3/4** | 2/4 | 2/4 |

- **DN peaks large 3/4** (ring 4.82% + sram + switchcap; opamp rails at every size)
  — same curve as other techs (over-fits at xl). Production tier = large.
- **BSIM-AR peaks medium 3/4, the ONLY family to pass the tsmc6 opamp (9.83%)** —
  opamp gain non-collapsed even at small (13.61%) where DN/PFN rail; mirrors V6.8.0
  (BSIM-AR's opamp basin). Over-fits large/xl (ring 7.4→11.2→12.6%).
- **PFN flat 2/4** all sizes (sram+switchcap always; ring 8–12% + opamp fail) —
  consistent with V6.10.0.
- **opamp-XOR-ring split:** DN-large banks the ring, BSIM-AR-medium the opamp, none
  takes all four. Device fidelity COMPLETE for all 11 cells (DN/PFN 6/6, BSIM-AR
  11/11): PMOS DC <1% (best 0.03%), NMOS DC 2–7%, VTC ~1–3%, tran ~0.8–1.2%;
  device-AC 0–2/3; opamp-AC fails (railed OP).

**Harness wiring** (TSMC6 now first-class): `verify_nn_dc_tran.py` (TestTechConfig
mirroring TSMC7 + stem sentinels), `nn_sweep.py`, `complex.py` (BENCH_TECHS + VT→ulvt),
`recipe_eval.sh`/`gate_matrix_iso.sh` (TECHS-overridable, dynamic denominator), the
5 collectors (dynamic `/16`→`/20`), new `scripts/tsmc6_collect.py`.

**Bug fixed — `tech_code_in_vocab` rejected TSMC6 (`tests/common/nn.py`):** the
ASAP7-guard checked the *universal* code against an 18-ceiling, so TSMC6
(tail-appended at universal 22–24) silently SKIPPED even though per-tech uses a
*local* vocab. Fixed: any tech in `LOCAL_VARIANT_CODES` passes; the ceiling now
only gates ASAP7. Preserves tsmc5/7/12/16 + ASAP7 exactly.

**Environment caveat (dead-end / partial):** campaign ran under sustained cluster
overload (loadavg ~1400/192 cores for days). BSIM-AR **large+xl opamp** gates
(AR inference ~30–100× DN) exceeded an 8 h cap but a 12 h retry completed both —
**complex matrix 100% resolved** (large opamp 12.78%, xl 10.13%, both ✗). Only
never-gated cells = the *secondary* large/xl opamp-AC diagnostics (not in the
complex count; opamp-AC rails everywhere). Gate timeout raised 1800→7200s mid-run;
parallel re-gate `scripts/tsmc6_regate.sh` recovered the rest. No conclusion
depends on the two open opamp-AC cells (BSIM-AR already peaks at medium).

## V6.10.0 — TabPFN port: the "PFN" compact-model family, LEVEL=75 (branch `worktree-tabpfn-pfn`, 2026-07-11/14)

**Ported TabPFN v3 (Prior Labs' tabular ICL transformer) into the bsimar stack as
a third NN family and ran the full scale campaign (24 ckpts: 4 techs × N/P ×
s/m/l, clean recipe) on the V6.8.0 harness. Report:
`docs/accuracy/PFN-L75-accuracy.md`.** (V6.9.0 = TSMC6, separate branch.)

**Result — PFN clean small (0.69M) = 11/16 STRICT (OMP∈{1,2,4}) with ZERO flips**
— the first family with no OMP multistability; strongest clean small on record
(DN clean small 7/16); still below curriculum families (DN 14/16, TF 15/16).

- **Architecture** (`tabpfn.py`, `TabPFNCompact`): faithful scaled-down port of
  the three v3 stages (per-column induced-attention embedder / column aggregator
  with CLS+RoPE / ICL transformer with context-only K/V) + two deviations: FROZEN
  LEARNED CONTEXT (stratified K-row buffer baked into the ckpt, context-KV cached
  at inference) and a direct 13-output value head (NR needs smooth autograd). Tech
  code = 8th column token (local vocab).
- **New identity:** LEVEL=75, `--model tabpfn`, tag `pfn`, stems `tsmc{X}_pfn_*`,
  env pins `PYCIRCUITSIM_NN_CHECKPOINT_PFN_*`, `PYCIRCUITSIM_NN_FORCE_LEVEL=75`;
  solver enumerators + parser cascade + all 5 drivers wired; DN/TF byte-identical.
- **Gate curve DECLINES monotonically 11/10/8 (s/m/l)**; misses = tsmc5/7
  ring+opamp (corridor cells, curriculum untested on PFN), tsmc12 switchcap
  5.1–5.3% near-miss, tsmc16-opamp rails from medium up (all 4 at large). Device
  fidelity peaks MEDIUM (id MRE 0.10–0.30%, best clean tier here); AC non-monotone
  (5/0/5 of 8); opamp-AC 0/4 (OP-MISBIAS).
- **Runtime:** 15.6 ms/eval CPU (DN 1.5, BSIM-AR medium 61.5) — 4× faster than BSIM-AR.
- **Off-grid geometry (root-caused):** `nmos_nfin_10` DC fails at s/m on tsmc12/16
  (NRMSE 27–31%, +20% flat over-pred) — NFIN=10 is OFF-GRID (data {2,3,4,6,20.9})
  and the context-relative embedding interpolates the 6→21 gap poorly vs DN's MLP;
  **capacity repairs it** (large ~6% PASS). Fix: denser NFIN or context re-freeze.
- **Large-tier optimization instability:** 8 divergence-collapse events (train
  0.02→0.77 at ep 25–80; both lr 4e-4 AND 3e-4; suspect unconstrained
  SoftmaxScalingMLP logit scaling in the W=256 ICL stack). Diverged runs bank
  pre-divergence EMA bests (val 0.0020–0.0029 = healthy band).
- **Pretrained-TabPFN ICL baseline** (58M, zero training): charges 0.3–1.1% MRE
  but id 3.2–5.7% — the from-scratch port beats it ~10× on id at 1.2% of params.
  Not solver-viable.
- **Dead ends:** fp32 large @300ep (killed ~ep50, 2.5 GPU-days/ckpt);
  lr-3e-4-fixes-divergence (refuted); large epoch-budget flip-flop (150→300→150).
- Production unchanged (DN crit30f@large). Highest-EV next: corridor `corroft@small`
  on PFN (aimed at the tsmc5/7 cells), context enrichment for the NFIN gap.

---

## V6.9.0 — TSMC6 (CLN6) onboarding + TSMC PDK parse audit (branch `feat/tsmc6-onboarding`, 2026-07-12)

**Onboarded the TSMC N6 iPDK card (`cln6_1d8_sp_v1d0_2p2.l`, BSIM-CMG 106.1,
0.75 V SVT/LVT/ULVT) as the sixth tech `TSMC6` — LEVEL=72 ground-truth + NN config
plumbing + datasets (NN training deferred to V6.11.0) — and audited PyCMG's card
parsing across all 5 TSMC cards. Report:
`docs/V6.9.0-tsmc6-onboarding-pdk-parse-audit.md`.** ⚠ Later corrected: `MEMORY.md`
found TSMC6 is TSMC7 relabelled (bit-identical datasets, identical L72 currents) —
NOT a distinct 6th tech under BSIM-CMG.

- **N6 = structural clone of N7:** identical section/model-name sets + binning (zero
  grammar changes); 153/204 core blocks byte-identical to N7, but `.global`/`.30`/pch
  bins differ → datasets md5-distinct.
- **Wiring (PyCMG):** `TECH_REGISTRY["TSMC6"]` + `TSMC6_CONFIG` (mirror TSMC7: vdd
  0.75, tfin 6n); generate/inv-trip/conftest (280→314 tests). Raw card gitignored (IP).
- **Wiring (main repo):** universal codes **22/23/24 tail-appended** (0–21 + the
  18-code pre-train vocab untouched, so all existing ckpts/sidecars stay valid);
  `NUM_TOTAL_CODES` 22→25; `LOCAL_VARIANT_CODES` += tsmc6 (local vocab 4). Parser
  needed NO change (config-map-driven resolver).
- **Labeller collision FIXED (`loo_labels.py`)** — the one real defect: the 12-param
  fingerprint labeller used silent last-writer-wins, and tsmc6↔tsmc7 fingerprints
  collide **108/108** (N7-copied bins). Per-tech datasets now label against their own
  tech (from the `<tech>_<device>.npz` stem); the universal scan keeps the legacy
  5-tech order and RAISES on any cross-tech collision.
- **Verification (all vs NGSPICE):** PyCMG pytest **314/314** (34 new TSMC6 PASS
  first-run); `verify_multi_tech_dc --tech TSMC6` **9/9** (≤0.0097%), tran **14/14**
  (≤0.64%); no empirical pruning needed (passes all 3 VTs × L=16/20/24nm, incl. cases
  N7 fails). Full 6-tech regression: tran **86/86**, DC **52/53 + 1 ERROR**
  (`TSMC5_lvt_inv_l_24nm` NR divergence, reproduced BIT-IDENTICALLY on main =
  pre-existing).
- **Datasets:** `tsmc6_{nmos,pmos}.npz` (1.82M/2.19M rows, 0 dropped) + sidecars.
- **PDK parse audit (new tool `audit_pdk_parse.py`): PASS — 40 devices, 0 round-trip
  mismatches.** Verdicts: (1) zero OSDI-known params among the expression-valued
  assignments each device skips (all HSPICE corner/stat/aging scaffolding); (2)
  mid-line `*` in blocks is expression multiplication NOT a comment — 1,112 legit
  params ride after such expressions, so `_extract_model_params`'s no-strip is
  load-bearing (do NOT unify with `parse_modelcard`'s comment stripping); (3) bin
  selection inclusive-both-ends vs HSPICE's `lmin ≤ L < lmax` — 28–40 boundary points
  ambiguous but self-consistent PyCMG↔NGSPICE, accepted; (4) ±999·10ⁿ sentinels only
  in TSMC5, correctly dropped; (5) ~1.1–1.9k TMI/stat params per card unknown to OSDI,
  ignored IDENTICALLY by PyCMG + NGSPICE, accepted.
- **Follow-ups:** TSMC6 NN training (→V6.11.0); a universal+tsmc6 model needs
  `num_tech_codes=25` + `uni_concat_npz.py` code-subset extension.

---

## V6.8.1 — BSIM-AR Transformer xl-tier fill (2026-07-11/23)

**Trained the untrained xl preset (384×8L×ff1536, 14.81M params — 3× tf-large,
5.5× DN-xl) across the full Phase-B recipe mirror — 48 checkpoints (6 recipes ×
4 techs × 2 devs), full 300/120-epoch fidelity — and gated vs NGSPICE. Report
§9 in `docs/accuracy/BSIM-AR-L74-accuracy.md`; driver + logs in
`results/recipe_bench/xl_campaign/`.**

**Result — xl TIES medium at 15/16 strict, does NOT exceed it, and does NOT
basin-shuffle.** Single-run complex gate (`gate_iso_xl_tf/`, 96 cells): corroft
/ crit15m / corro15 = **15/16** (all miss ONLY tsmc7-opamp), crit30 14/16, clean
& csob 13/16. corroft@xl OMP-strict-confirmed (opamps tsmc5 4.02 / tsmc12 5.90 /
tsmc16 5.98% gain-err, OMP-invariant across {1,2,4}). **The DirectNet xl
basin-shuffle (DN-xl banked tsmc16-opamp) does NOT replicate on the Transformer**
— it already banks tsmc16-opamp at medium and just holds the same 15/16 basket at
xl. Capacity ceiling = 15/16 across three tiers (medium/large/xl); tsmc7-opamp is
genuinely a solver-only (T3/EKV) cell no capacity/recipe reaches.

**AC COLLAPSES at xl** (the sharp negative result): **opamp-AC 0/4 every recipe**
(tsmc5/16 rail on OP-misbias, tsmc12 GBW/PM good but magNRMSE 102%; tsmc7-opamp-AC
does not converge — a ~6 h non-convergent solver spin, seed-skip it), **device
nn_ac weak** (corroft 4/8 device-passes, csob 4/8, clean 2/8), and **csob's
charge-Sobolev does NOT recover AC** — AC weakness at xl is a tier property, not
recipe-fixable. Confirms the "AC peaks at small" law (xl = worst AC tier, like
DN-xl→0). Device DC stays excellent: corroft@xl 55/55 configs PASS (baseline
NRMSE 0.03–0.19%).

**Dead-end / no-promote:** xl costs 3× params + ~30–100× AR inference for the
*same* 15/16 as medium, with collapsed AC and zero basin gain. **corroft@medium
(1.9M) remains the validated best BSIM-AR config; xl is not promoted.**

**Ops notes (shared-box campaign):** ~11-day run under heavy co-tenancy
(Xyce/Swin fleets + the TSMC6 campaign); reboot-resilient `setsid`+`@reboot`-cron
driver with a `.complete`-gated idempotent resume and a 3→30 retry cap
(exactly one transient SIGKILL all campaign, auto-recovered; cap never needed).
Gate lesson: never reuse a `gate_iso_*` output dir across model families (a stale
DirectNet `gate_iso_xl/` nearly contaminated the tally → fresh `gate_iso_xl_tf/`).

---

## V6.8.0 — BSIM-AR Transformer (LEVEL=74) un-parked: recipe campaign (branch `V6.7`, 2026-07-06/07)

**Shipped the entire DirectNet training/recipe/eval stack to the parked BSIM-AR
decode-only Transformer and ran the full scale × recipe campaign per tech, gated
vs NGSPICE. Report: `docs/accuracy/BSIM-AR-L74-accuracy.md`.**

**Result — BSIM-AR `corroft@medium` (corridor curriculum, 1.9M params) = 15/16
STRICT (OMP∈{1,2,4}), beating DN production `crit30f@large` (14/16).** Better basket:
banks tsmc16-opamp (DN FAILS it) + both low-VDD rings, all deterministic; misses
ONLY tsmc7-opamp — the hard cell DN reached only via the V6.5.9 T3 solver fine-tune
(out of scope here). Device DC 44/44 all tiers; AC peaks at small (7/8).

- **Port (code, all default-off / behavior-preserving):** (1) `transformer.py`
  `unknown_code_id` — was hardcoded universal 17; local vocab would CUDA-assert on
  p_unknown → now `num_tech_codes-1`. (2) `init_from` warm-start. (3) Sobolev /
  charge-Sobolev / subthreshold aux losses made model-agnostic; column-sum autograd
  trick VERIFIED valid for the Transformer (grad diff 1.3e-7). Blocker fixed: fused
  SDPA has no double-backward → aux losses force the MATH backend at forward. (4)
  Transformer `xl` preset (14.8M). (5) `--amp` bf16. (6) Parser LEVEL=74 preempt
  cascade `tsmc{X}_tf_{size}_{dev}` (large-first) + `_tf_` scope detection. (7)
  `PYCIRCUITSIM_NN_FORCE_LEVEL=74` hook — retargets LEVEL=73 decks to BSIM-AR at
  parse time (whole gate infra runs the Transformer, zero deck changes). (8)
  `MODEL={direct,transformer}` in the 5 drivers. (9) Solver batched-eval includes
  LEVEL=74 (per-ckpt grouped forward — biggest CPU-gate lever for the AR loop).
- **Latent L74 bug fixed (`mosfet_nn.py`):** `_out_col` read outputs via
  `norm_stats.output_columns` (canonical order) BEFORE the BSIMAR layout → qg
  denormed as id (~5× current, 389% device NRMSE) though the module scored 0.38% on
  the trainer's test set. Historical BSIM-AR norm files predate `output_columns` so
  it never fired. Fixed: BSIMAR layout branch ranks first. Classic silent-green,
  caught by the first real checkpoint.
- **Scale study (clean, S/M/L):** complex 12→14→13 (peaks MEDIUM, one tier earlier
  than DN); AC 7→4→4 of 8 (peaks small); device 44/44 all tiers (tsmc7-NMOS NRMSE
  GROWS with capacity 3.37→4.77% — the tech whose opamp is unreachable).
- **Recipe study:** 3 large corridor recipes (corroft, crit30, crit15m) all →15/16
  single / 14/16 strict, all rail only tsmc7-opamp. **inv_trip anchor is INERT on
  the Transformer** (corroft ≡ crit30 <0.5%/cell — opposite of DN). **corridor is
  the whole ring lever** (invtrip-alone stays ~13/16). **tsmc7-ring ⊥ tsmc7-opamp
  under the corridor at EVERY tier** (ring-open = opamp-kill perturbation) → ceiling
  15/16. Strict best is MEDIUM (both rings deterministic), not large.
- **Datasets:** existing `{tech}_{dev}.npz` + `_corro_` sets (no regen). Ckpts
  `tsmc{X}_tf_{small,medium,large}_{nmos,pmos}` (24) + recipe variants. Production
  unchanged (DN stays fast path; BSIM-AR un-parked as the higher-fidelity option).

---

## V6.7.1 — house-clean after the V6.7.0 campaign (branch `V6.7`, 2026-07-05)

**~12.7 G reclaimed; nothing load-bearing touched; retired checkpoints archived to
`/data2/shenshan/v66x_v670_retired_ckpts_2026-07-05.tar.gz` (1.9 G) before deletion.**

- **Checkpoints (2.5 G → 459 M):** pruned the V6.6.x recipe-study pool (all recorded
  losers, verdicts in the accuracy reports): cor/corft/corr/corrft/corro15, crit*@large,
  crit*@xl, csob@{s,m,xl}, csobcrit (relocation-not-composition), csobekv (EKV breaks
  SRAM), cs7, ekv, invtrip (refuted as ring lever), s123/s17/s7, sob ×4 techs; + failed
  V6.7.0 fine-tunes `u716f5_{plain,crit}_n{2k,10k,50k,200k}` (0/4; ≤10k DIVERGE) and
  `u716f5_crit_n{1M,full}`. **Kept:** production s/m/l/xl, v660clean, crit30f@large,
  alternates csob@large + corroft/crit10@xl, promote-candidate crit15m@xl (14/16 ties
  production), 6 u716 universal bases, u716f5_plain_n{1M,full}.
- **Datasets (26 G → 19 G):** deleted V6.7.0 `uni716_*` concats + `tsmc5ft_n*` (both
  regenerable via seeded `uni_{concat,subsample}_npz.py`). KEPT `tsmc{X}_{cor,corr}_*`
  (9.1 G, referenced by `recipe_train.sh`'s `cor*` arm — retrainability over reclaim).
- **Generated outputs (~3.0 G):** `results/uni_bench/*/complex_omp*` work dirs (SUMMARY /
  logs / transfer_curve.tsv kept) + `tests/verify_*_results/`.
- **Docs:** removed the closed V6.6.2 campaign plan (git-recoverable; verdict preserved in
  entries + memory). **Report merge (user request):** the 580-line collector output
  `results/recipe_bench/ACCURACY_REPORT.md` merged verbatim into
  `docs/accuracy/DirectNet-L73-accuracy.md` as Part II (Part I = analysis, Part II = all data
  tables). The results path keeps a marker stub so collectors still regenerate in place.
- Trap-list survivors re-verified (corridor pipeline scripts, base+corro datasets, PyCMG,
  tools/ngspice); no orphan `_norm.npz`.

---

## V6.7.0 — universal DirectNet + TSMC5 transfer study (branch `V6.6`, 2026-07-04/05)

**Campaign: resurrect the universal-scope DirectNet (retired V6.1) — ONE
18-code-embedding model on TSMC16+12+7 (~7M rows), rank the Core-4 recipes
(clean/csob/corroft/crit30u) on the 12 shared complex gates, then measure TSMC5
few-shot transfer (fine-tune tiers N ∈ {2k…1M,full}).** Report:
`docs/accuracy/DirectNet-L73-accuracy.md`. **No production change:** all artifacts use
`u716_*`/`u716f5_*` stems (env-pin-only; per-tech resolution untouched). New standalone
scripts (zero edits to existing): `uni_concat_npz.py`, `uni_subsample_npz.py`,
`uni_train.sh`, `uni_gate_sweep.sh` (12 gates + zero-shot + OMP sweep + RESOLVER-MISS check).

Executed (36 ckpts, 264 eval cells, zero RESOLVER-MISS, ~1.5 days). **Headlines:**
(1) **universal is VIABLE** — device fidelity per-tech-grade (id NRMSE ≤0.09%, R²≥0.996)
and **corroft = 10/12 strict, 0 FLIPs = per-tech parity with full OMP determinism** (which
per-tech large never had); ranking corroft 10 > clean 9 > crit30u 9+FLIP > csob 8+FLIP;
corridor fixes tsmc7-ring 14.89→3.61% (ring-lever confirmed at universal scope); anchor +
csob basins do NOT survive the scope change (recipe→basin maps are SCOPE-dependent).
(2) **TSMC5 onboarding = ~1M stratified rows: plain@n1M = 4/4 STRICT** (ties per-tech
production at half the data); threshold sharp (0/4 ≤200k); ≤10k DIVERGE (tier-refit
normalizer); n1M beats nfull (opamp basin non-monotone in N). (3) **No free retention** —
source techs collapse at gate level (1–3/12); fine-tune = de-facto per-tech ckpt.
**Phase 1b (xl) CLOSED:** clean@xl 8/12+1FLIP (banks both opamps, loses all rings),
corroft@xl 8/12 (holds rings, all opamps rail) — the V6.6.1 opamps-XOR-rings wall reappears
at universal xl → **corroft@large (10/12 strict, 0 FLIPs) = final best universal config.**

---

## V6.6.7 — 15/16 hunt round 1: csobcrit + crit30a1 both 13/16 (branch `V6.6`, 2026-07-03)

**Both V6.6.6-routed arms trained (16 ckpts, `large`) and strict-gated: NEGATIVE — 13/16 strict
each; production stays crit30f@large 14/16.** `csobcrit` (csob base + crit30 curriculum,
`--charge-sobolev` retained, corro data): the curriculum **relocates rather than composes**
basins — csob's deterministic tsmc16-opamp hold (1.28 %) degrades to a FLIP (OMP1 100 % /
OMP2,4 7.49 %) while tsmc12-opamp is gained deterministically (5.80 %); plan-§5 P2a REFUTED.
`crit30a1` (ring-only corridor w3.0 + HALF inv_trip anchor 1.0, v660clean base): reproduces
corroft (anchor 0) almost exactly (tsmc16-opamp 7.34 % detPASS, identical rings) — anchor 1.0 is
sub-threshold; **the {16} → {5,12} opamp-basin hop is DISCONTINUOUS in anchor weight ∈ (1.0,
2.0)**. Remaining cheap arm: anchor ~1.5 at w3.0; otherwise the uniform lever is exhausted for
the 5+12+16 triple → 15/16 routes to structural levers (T3 class). Harness: recipe map gained
csobcrit/crit30a1; crit-family `--init-from` at `large` now redirects to the **v660clean
archive** (the production slots carry crit30f since V6.6.4 — warm-starting from them would
silently stack curricula). Reports consolidated (user request): `RETEST_ACCURACY.md` +
`RECIPE_REPORT.md` merged into the single `results/recipe_bench/ACCURACY_REPORT.md` (RETEST +
MATRIX marker sections, each regenerated in place by its collector). Same-day house-clean:
CHANGELOG compressed, CLAUDE.md de-duplicated, `results/` pruned to the live evidence set.

---

## V6.6.6 — xl curriculum ties production at 14/16 strict + full test-infra audit (branch `V6.6`, 2026-07-03)

**The 9 curriculum recipes trained at `xl` (72 ckpts, warm-start clean xl) and
strict-gated: `corroft`/`crit10`/`crit15m`@xl = 14/16 STRICT, tying production — all
three bank the tsmc16-opamp deterministically (6.2–6.7%) which production FAILS**
(crit15m also banks tsmc5-opamp). Production UNCHANGED; analysis:
`docs/accuracy/DirectNet-L73-accuracy.md`. Also re-gated the 13 base recipes' xl ckpts (V6.6.3
methodology, `SIZE`-parameterized) — reproduces V6.6.5 cell-for-cell; **xl basins are
OMP-deterministic** (sole FLIP corft tsmc5-ring), unlike large's endemic opamp flips.
Findings: curriculum warm-start rescues xl wholesale (clean@xl 10→14); the **weight→basin
map is TIER-dependent** (crit30 14@large/12@xl; crit10/crit15m peak at xl); ring-only
corridor is the only safe corridor at xl; tsmc7-opamp 0 over 22 recipes × 2 tiers;
opamp-AC 0/4. **Every opamp basin except tsmc7 is individually reachable; none of the
5+12+16 triple holds simultaneously = the 15/16 target.**

**Test-infra audit — 17 verified fixes (4-agent sweep, every finding re-verified).**
Silent-green class: `bsimcmg_dc` scored >100% divergences as ERROR-skip → exit 0 (now hard
FAIL); all-ERROR suites exited 0 (now →1); **an absent `PYCIRCUITSIM_NN_CHECKPOINT_*` pin
silently fell back to production `large`** — a recipe eval could report production numbers
under the recipe's name (now FileNotFoundError); sweep ring/switchcap lacked the `not
partial` guard; SRAM dropped errored NFIN corners; 4 mains exited 0 on empty results;
flat-reference `nrmse()` returned 0 = auto-PASS (now inf); truncated transients now FAIL.
Pipeline class: `gate_matrix_iso.sh` clobbered SUMMARY.txt on subset re-runs (had wiped the
large tier's rc records — numbers stayed correct only because log-status ≡ rc) and broke
`gate_grid.py` to 0/16 (now per-cell verdict files + non-destructive merge); `benchmark_collect`
NUM regex dropped nan rows; the sweep drift manifest hashed `medium` while the resolver loads
`large` (now resolver-consistent); train-resume trusted `_best.pt` (now `.complete` markers).
**The v664-P0 thread-pin landed:** `torch.set_num_threads(PYCIRCUITSIM_TORCH_THREADS or 1)` in
`complex.py`+`complex_ac.py` — verdict-neutral (tsmc12-opamp reproduces 6.25% exactly). Audited
clean: NN-vs-NGSPICE deck parity in all 4 circuits, every collector regex, 512 ckpt-stem →
local-vocab mappings. Accepted LOWs (documented, unfixed): non-inscribed-square SNM diagnostic,
midpoint-touch crossing count, uic-vs-OP tran-start semantic, non-atomic modelcard bakes.

---

## V6.6.5 — Recipe×size matrix COMPLETED: all 13 recipes at all 4 sizes (branch `V6.6`, 2026-07-03)

**The V6.6.1 study became a FULL 13-recipe × 4-size matrix: the 27 large-only combos were
uniform-trained (208 ckpts) and gated (864 eval cells, zero blanks; all logs audited
rc∈{0,1}).** Findings: `clean@large` 13/16 stays the unbeaten in-matrix cell; **the corridor
inverts the capacity curve** (`cor` 11/12/11/5 s→xl — dominates below `large` with the best
device fits, collapses above — weight-space non-monotonicity now seen in capacity-space);
**xl is basin-shuffled, not uniformly over-fit** (invtrip/s17 12/16@xl vs clean 10; sob, worst
everywhere else, jumps 5→10); **AC pass-rate peaks at SMALL** across recipes; device-NRMSE
bottoms at medium; tsmc7-opamp 0/52. Ops (uniformity discipline): 22 killed-run "best-so-far"
ckpts quarantined + retrained to full spec (a `_best.pt` on disk is NOT evidence of a completed
run — gate on the log tail); an eval-overlap incident truncated 17 xl logs → deleted + re-run
serialized. Campaign driver `results/recipe_bench/fill_campaign.sh` (~24 h wall, 3×4090).

---

## V6.6.4 — crit30f PROMOTED to production (branch `V6.6`, 2026-07-02)

All 8 `tsmc{X}_dn_large_*` production slots replaced with bit-identical copies of the
V6.6.3-validated `crit30f` checkpoints (clean base + one identical curriculum fine-tune
`--class-weights traj_corridor=3.0,inv_trip=2.0 --lr 3e-4 --epochs 120 --patience 40
--init-from` own clean large, ring-only `corro` data); the clean originals archived as
`tsmc{X}_dn_v660clean_large_*` (re-gateable as recipe `v660clean`); small/medium/xl stay clean.
Post-promotion verification on the default resolver path (no env pins) reproduces the record —
**14/16 deterministic** (tsmc5-ring 4.04 %, tsmc5-opamp 0.21 %, tsmc12-opamp 6.25 %; tsmc7 +
tsmc16 opamp remain the FAILs). Production moves 13/16 (12 strict) → **14/16 strict**.
Checkpoints are gitignored — this entry + CLAUDE.md are the record.

---

## V6.6.3 — Full-recipe re-test: crit30 supersedes crit15 at 14/16 STRICT (branch `V6.6`, 2026-07-02)

**All 22 on-disk uniform recipes re-tested under one discipline (16-cell isolated matrix +
OMP∈{1,2,4} determinism sweep per recipe + opamp-AC + finalist device suites): best =
`crit30` (`traj_corridor=3.0,inv_trip=2.0` curriculum) at 14/16 STRICT all-OMP — clean+2 —
superseding V6.6.2's crit15 (13 strict).** Validated by `crit30f`: the original crit30 had been
killed at heterogeneous epochs 30–92, so all 8 ckpts were retrained to full spec — the honest
rerun reproduces the artifact cell-for-cell. crit30 banks deterministically: all 4 rings
(tsmc5 12.66→4.04 %), all SRAM/switchcap, tsmc12-opamp 6.25 %, **and tsmc5-opamp 0.21 %** (a
coin-flip in clean, detFAIL in crit15); device level ≥ clean everywhere (DC mean NRMSE
1.64→1.46 %, device-AC 4/8→6/8). V6.6.2 single-run corrections: "crit20/30 collapse opamps" was
a coin-flip read (crit20 13, crit30 14 strict); the corridor-weight → tsmc5-opamp basin map is
**non-monotone** (w1.0 FLIP, w1.5/2.0 detFAIL, w3.0 detPASS — the inv_trip anchor makes w3.0
safe where corroft alone railed). csob re-scoped: only tsmc16-opamp detPASS + only opamp-AC
PASS + best tsmc12 device fit, but tsmc12-opamp flips at OMP=8 and 2 rings detFAIL → 12 strict
(stays the AC/device alternative). tsmc7-opamp: 100 % across all 23 artifacts × all OMP. New
infra: `scripts/{recipe_retest_collect,device_retest_collect}.py`, `device_matrix_iso.sh`,
env-overridable results dirs; prior §14 runs archived as `*_v662_prior`.

---

## V6.6.2 — The cross-wall combo breaks 13/16: crit15 = clean+1 (branch `V6.6`, 2026-07-02)

**REFUTED the V6.6.1/§13 "13/16 uniform ceiling": `crit15` (`traj_corridor=1.5,inv_trip=2.0`
curriculum warm-start on the ring-only `corro` data — the corridor and inv_trip levers had only
ever been tested SEPARATELY, on opposite sides of the shared-MLP wall) nets 14/16 single-run /
13/16 strict = clean+1, the +1 being the DETERMINISTIC tsmc5-ring opening (12.66 → 4.0 %).**
Confirmed live that the opamp gate is a multistable OMP coin-flip (tsmc5/tsmc16 flip 0–8 % ↔
100 % across OMP∈{1,2,4} in both clean and crit15 — single-run opamp passes are unbankable;
`opamp_sweep_def.sh` became the standing probe). Round-2 (crit15m/h, crit10, crit20,
crit30-undertrained) read NEGATIVE on single runs — corrected by V6.6.3's strict re-test.
Production unchanged (promotion deferred to user). New harnesses: `gate_matrix_iso.sh`,
`opamp_sweep_def.sh`, `gate_grid.py`; `recipe_train.sh` gained the `crit*` family.

---

## V6.6.1 — Uniform-recipe comparison sweep (branch `V6.6`, 2026-07-01)

**Swept uniform recipes (charge-Sobolev `csob`, Sobolev `sob`, EKV `ekv`, seeds `s7/s17/s123`,
combos `csobekv`/`cs7`) across all 4 sizes: NO uniform recipe beats clean's 13/16 at `large`
(csob 12 > cs7/s7/s17 11 > csobekv/ekv/s123 10 > sob 5).** The ceiling is a mutual exclusivity
of value-surface basins — each recipe/seed lands a different subset (tsmc12-opamp passes only on
seed-42); combo stacking is zero-sum (`csobekv` uniquely holds tsmc7-ring + tsmc16-opamp but the
EKV core breaks tsmc12/16 SRAM → 10/16; it also converges 10–40× worse val-loss). **`csob` =
best all-rounder** (best device NRMSE at `large` 1.50 vs 1.70, best AC, lands tsmc16-opamp, −1
complex gate) → promoted to documented alternative; refutes the V6.5.x "charge-Sobolev dead on
arrival" verdict (which was measured only at `medium`). `sob` reconfirms deriv-fidelity ⟂ opamp
(5/16). Report: `docs/accuracy/DirectNet-L73-accuracy.md`. Infra: `scripts/recipe_{train,eval}.sh`
+ `recipe_collect.py`; checkpoints saved `tsmc{X}_dn_{recipe}_{size}_{dev}` (clean never
clobbered).

---

## V6.6.0 — House-clean + uniform-recipe reset (branch `V6.6`, 2026-06-29)

**Deliberate reset from V6.5.9's hand-tuned 16/16 to the honest uniform baseline: all 32
DirectNet checkpoints retrained from scratch on ONE identical recipe (`--apply-filter off
--swa-mode ema --seed 42`); production = the uniform `large` tier at 13/16 complex gates
(capacity curve 7/10/13/10 s→xl — peaks at `large`, over-fits at xl).** The 3 open gates
(tsmc5 ring, tsmc7/tsmc16 opamp) are the true fidelity frontier the V6.5.x per-case specials
had force-closed. Device 24/24, inverter 16/16 at every size, lifted-source canary 12/12, L72
OP/DC/TRAN/AC all PASS post-clean. Report: `docs/accuracy/DirectNet-L73-accuracy.md`. House-clean:
datasets purged of stale variants (16 GB → 4.5 GB); version-pinned results/plans/one-off
scripts and 11 gate-specific diagnostics removed (4 reusable controls + all 23 `verify_*` gates
kept); 2 dead `config.py` helpers removed; checkpoints dir cleared with the V6.5.x specials
archived off-repo (`/data2/shenshan/v6.5.9_production_specials.tar.gz`; rollback = `git
checkout V6.5.4` + untar); resolver now prefers per-tech `large` first;
`benchmark_train_sml.sh` gained a `GPUS` override.

---

## V6.5.9 — ★ 16/16: T3 differentiable-DC-solver lands the tsmc7 opamp (branch `V6.5.4`, 2026-06-29)

First-ever tsmc7 opamp PASS (DirectNet gain 178.0 vs NGSPICE 163.4, 8.92%) → production 16/16. Put the DC solve **inside the loss**: a differentiable unrolled Newton solver supervises the emergent transfer curve `Vout(Vin; θ)` against L72, so r_o is shaped by the gain target instead of the residual-minimisation shortcut that over-flattens it. This broke the V6.5.8 "gain stuck ~370" wall — the gain-163 root DOES exist on the ekvhr substrate; "370" was the gate's continuation landing on a different over-flattened branch. Findings: the gate gain is **bimodal + sampling-noisy** (peak |dVout/dVin| bounces 147–187 even on a 3%-NRMSE-faithful curve); the `--lam-lo-override` r_o cap is fine WITH curve supervision (it rails without it); **preservation (ring/switchcap), not existence, was the binding work** — the faithful good-curve root and a passing ring are mutually exclusive, so the shipped candidate sits on the ring-compatible shifted root. Installed `tsmc7_dn_t3` via the `tsmc7_dn_medium` symlink (retired in V6.6.0). Infra `scripts/v6_5_8_{harvest_opamp_topology,t3_solver_finetune,gate_t3,install_t3}`. Memory `[[v659-t3-solver-lands-opamp-16of16]]`.

---

## Test-infrastructure correctness sprint — 11 bugs fixed (branch `V6.5.4`, 2026-06-28)

Audit-driven fix of the verify harness; production pass-rates unchanged (NN device 24/24, complex 15/16), every fix re-checked vs NGSPICE. Durable fixes still in code: **B1 (CRITICAL)** — the per-tech device gates in `verify_nn_dc_tran.py` were pinning **tsmc5's** net (at UNKNOWN tech-code) for ALL techs; routed the 5 `model_path` sites through `_cascade_handles_stem` so each tech resolves its own checkpoint. **B3/B5** — SRAM scored PASS when every corner errored (`all([])==True`) and never compared to ground truth; now ANDs point-by-point NGSPICE-NRMSE tracking (≤10%), with `force_ic` reconciled as a printed **diagnostic** (not a gate; rails on TSMC7/12 only). **B4** — a diverged inverter transient could false-PASS (`_nr_partial` set-but-unread); now auto-FAIL. B2/B6–B11: sweep↔gate `uic` equivalence canary, ASAP7 skip (Rule 14), real-deck canary (32/32), `partial` gating, honest additive exit codes.

---

## V6.5.8 — EKV high-r_o core + vout-weighted KCL breaks the tsmc7-opamp rail (branch `V6.5.4`, 2026-06-28)

First non-railed tsmc7 opamp in the whole campaign — a high-r_o EKV **structural** core + a vout-weighted KCL existence fine-tune produced a real amplifying transfer curve (gain ~350–381), where every prior V6.4.x–V6.5.7 attempt railed to 0. **REFUTES the V6.5.6/7 "only T3 can create a reachable high-gain OP" verdict.** BUT gain ⟺ existence are **coupled through the output-stage r_o**: the reachable OP is over-gained (~2.2× the L72 target 163) and every calibration lever is a binary rail↔370 switch — vout-weight, lam-kcl, and (decisively) `--freeze-core` / `lam_lo`-cap all RAIL rather than lower the gain (the over-flattened r_o is *required* for reachability). ±10% gate not passed; nothing installed (15/16). Routes to T3 (V6.5.9), which sits on this EKV+KCL substrate. `_EKVCore` redesigned to floor-scaled physical `id_core + sqrt(id_core²+(κ·id_s)²)·α·tanh(trunk)` + exposed `--ekv-lam-lo`. Memory `[[v658-ekv-core-breaks-opamp-rail]]`.

---

## V6.5.7 — panel-review correction of the V6.5.6 opamp verdict (branch `V6.5.4`, 2026-06-27)

A 5-agent adversarial review found V6.5.6's "no high-gain zero exists / probe-closed / only-T3" **over-strong**. The real bind is full-system STABLE EXISTENCE with **`vout` the never-supervised node** (T1 pinned only the stage-1 balance node `vo1i`; the solver-conditioning probe was 20 *cold* multistarts along a 1-D line — reachability, not an existence proof; the "vout residual at V*_L72" non-existence argument is falsified by the PASSING tsmc12 at F_rel≈0.19). The cheap vout-prioritized existence retrain was then RUN & **KILLED**: `vout` F_rel floors 0.062 (wrecks the base surface +492%) → 0.13 (preservation-safe) vs the ~0.006 a high-gain zero needs; 0 high-gain solutions on all candidates; opamp gain 0.0 FAIL ⇒ the soft-wall is near-hard for the KCL-loss family. **fetlim also dead** (the L72 control lands gain 163 on the same fetlim-less path — voltage-limiting is not what discards the NN's step). Routes to EKV/T3. `finetune_kcl.py --vout-weight/--vout-target`. Memory `[[v657-vout-existence-retrain-kill]]`.

---

## V6.5.6 — 3-operator Phase-0 routing + T1 KCL-residual lever (branch `V6.5.4`, 2026-06-26)

**Organizing frame — the 3-operator taxonomy (durable):** DirectNet emits ONE surface but the solver reads it through THREE operators, each owning a different gap + a structurally different fix-class — id-VALUES→KCL→NR fixed point (G1: opamp gain, ring period); autograd dQ/dV→pole (G3: f3db); off-diagonal cgd→RHP zero (G4: HF phase). Charge-head retrains are DC-safe; id-surface retrains are DC-unsafe; the recurring ledger failure is applying the wrong fix-class. Phase-0 (four zero-GPU diagnostics) routed the gaps: **D1 EXISTENCE** — at the L72 high-gain OP the net signed NN current is NOT a residual zero (tsmc7 vo1i F_rel 0.128 vs passing tsmc12 0.002); **G3 dead on arrival** — autograd ∂qd/∂Vd already == cdd head == OSDI ~0.1%, so f3db is OP-drift/value-surface owned; **Jacobian-blend closed analytically** — the fixed-point LOCATION is a pure function of `id` VALUES (gm/gds set only the Newton path). **T1 net-node KCL-residual loss SOLVED existence** (vo1i 0.128→0.007 — the corridor never did) but the OP is an **unstable Newton fixed point (contraction)**, and preservation is binding (any λ strong enough to move existence regresses the ring; N2 Sobolev blocks existence on the shared id head). Track B: ring-anchor WORKS (ring 6.44%→2.29% PASS). Not installed (15/16). ⚠ Its "only-T3 / no-zero-exists" conclusion was **corrected by V6.5.7**. Memory `[[v656-t1-existence-to-contraction]]`, `[[nn-accuracy-3operator-taxonomy]]`.

---

## V6.5.5 — diagnostic-routed corridor retrain → 15/16 (branch `V6.5.4`, 2026-06-24/25)

Three zero-risk diagnostics localized the V6.5.4 open gates and routed a targeted corridor retrain. **tsmc5 ring = NMOS-conduction-owned** (the pull-down under-drives id ~23% at the switching edge; the charge-ON transient reproduces the 12.66% gate, ~66% of the period error) → lifted 3/4→4/4 via `large`+ring-corridor+seed7 (`tsmc5_dn_corringL_s7`); **capacity was the bind** — medium trades ring↔opamp, large+seed threads both. **tsmc7 opamp = value-surface-owned** — seeding the NN sweep from the L72 ground-truth OP at every point STILL rails to gain 0 (the high-gain OP is unstable on the NN surface; PTC/homotopy/OP-seed cannot fix it), confirmed unrecoverable by corridor exhaustively. Net 14→15/16. Memory `[[v655-diagnostic-routing-verdicts]]`, `[[v655-corridor-retrain-15of16]]`.

---

## V6.5.4 — fresh full retrain + best-config-per-tech → 14/16 (branch `V6.5.4`, 2026-06-23/24)

Retrained the entire capacity matrix from scratch on freshly regenerated data (32 models, one clean recipe `--apply-filter off --swa-mode ema --seed 42`), best config per tech → 14/16 (matching the V6.4.7 ship but clean, no stale/specialized checkpoints). **Native-L72 control (decisive new diagnostic, `tests/diag_l72_complex_control.py`):** running the exact gate circuits through PyCircuitSim's own solver with the ground-truth OSDI model (no NN) matches NGSPICE at ring 0.00% / opamp 0.00–0.10% ⇒ both remaining gaps are **genuinely NN-value-surface-owned**, not solver/harness (and the 2ps timestep is adequate). Residual 2 gates (tsmc5 ring, tsmc7 opamp) resist every clean-data size AND seed — the value-surface limit V6.4.7 only cleared with corridor-augmented data. Memory `[[v653-l72-control-ring-opamp-model-owned]]`.

---

## V6.5.3 — ★ the switchcap gap was a HARNESS CLOCK BUG, not solver/NN-owned (branch `V6.5.2`, 2026-06-23)

**Overturns V6.5.2.** The tsmc5 switchcap "11.84% over-charge" chased across the ENTIRE V6.4.x–V6.5.2 campaign (XL capacity, µA-band loss, charge-Sobolev, TG-corridor, EKV backbone, the "switchcap-is-solver-owned" verdict) was **two harness bugs**: (1) `render_directnet_netlist` rescaled `Vdd`/`=0.80` to the tech VDD but MISSED the space-delimited PULSE clock rail, so the DirectNet clock over-drove the tsmc5 pass gates to 0.80 V while the NGSPICE deck clocked to 0.65 V — exactly explaining the tech pattern (tsmc5 +0.15 over-drive → 11.8% FAIL; tsmc12/16 0.80 → no over-drive → PASS) → **11.84% FAIL → 1.56% PASS, switchcap 4/4**; (2) the "14.65% L72 floor" (V6.5.2) was a control DC-op with no `uic` pinning (the hold node seeded at the off-transistor leakage equilibrium). **LESSON (load-bearing): when an NN gate fails vs NGSPICE, FIRST diff the rendered NN netlist against the NGSPICE deck token-by-token — clock amplitude, supply rails, bias, sweep, geometry — BEFORE blaming the model or solver.** `uic` was also made first-class in the product path (`.tran ... uic` parsing + pinning in `run_transient`, default-off → non-uic decks byte-identical). Memory `[[v652-switchcap-is-harness-clock-bug]]`.

---

## V6.5.2 — charge-derivative levers + the (later-refuted) switchcap-is-SOLVER-owned finding (branch `feat/ac-analysis`, 2026-06-22)

> SUPERSEDED by V6.5.3 — the "switchcap is solver-owned / 14.65% L72 floor / not NN-fixable" conclusion was two harness bugs. The cap-fidelity sub-findings remain valid reference.

Both candidate switchcap levers KILLED (correctly — there was no model gap): **charge-Sobolev** (`--charge-sobolev`, couples autograd dQ/dV to the supervised cgg/cgd/cdg/cdd) left the switchcap unmoved (11.84→11.32%) and did NOT move AC f3db (⇒ f3db is OP-drift / value-surface owned, not cap-under-prediction); **TG-corridor data-aug** fixed PMOS cdd 62%→5% yet the switchcap charge didn't move (the tell it was never cap/model-owned). Valid reference: the NN autograd caps match OSDI ~0.3–2.5%; per-channel sign map `+cgg,−cgd,−cdg,+cdd` (OSDI off-diagonals SPICE-negated). Kept default-off recoverable. Memory `[[v67-switchcap-is-solver-owned]]` (⚠ refuted).

---

## V6.5.1 — XL capacity tier + µA-band loss lever (KILLED) (branch `feat/ac-analysis`, 2026-06-22)

**XL tier (512×8, 2.13M p) = the over-fit boundary:** complex-gate pass-rate PEAKS at `large` then DECLINES — **6→9→12→9/16 (S→M→L→XL)**; XL fits the device surface ~10× tighter (val 2e-4) yet has the WORST off-nominal NRMSE and **loses every value-surface-fragile gate `large` won** (tsmc5/tsmc12 opamp, tsmc7 ring flip PASS→FAIL) — the cleanest confirmation of V6.4.8-S1 ("capacity is not the bind"; more capacity over-fits and collapses the high-gain NR basins). **µA-band loss de-compression KILL (refutes the V6.4.8 roadmap):** retuning the default-off `SubthresholdIdLoss` to the µA band moved the tsmc5 switchcap <0.2% — the over-charge is sample-and-hold charge/transient behaviour, not µA-band-DC-current owned. Also fixed the `xargs -L1` trailing-blank silent job-collapse bug. Memory `[[v66-xl-overfit-and-uA-lever-kill]]`.

---

## V6.5 — AC small-signal accuracy of the NN models (branch `feat/ac-analysis`, 2026-06-22)

First time NN AC fidelity was gated against ground truth (NGSPICE `.ac` on the identical BSIM-CMG OSDI), across 24 DirectNet checkpoints + the opamp — the NN's small-signal caps are autograd dQ/dV of its predicted charges, the direct probe of the charge-surface derivatives no prior gate measured. AC **gain** excellent everywhere (24/24 gain0 err <1.5 dB ⇒ autograd gm/gds accurate); the cap-driven **pole** is good but tech-variable (device gate 13/24); the **Cgd-feedforward RHP-zero HF phase is NOT reproduced** (a transcapacitance-sign limitation, diagnostic); the **opamp AC inherits the DC value-surface fragility (0/12)** but where the OP lands well (tsmc12-large) GBW 0.97× / PM 1.3° — dynamics right, DC-gain level is the miss. **No retrain warranted** (a dQ/dV deficiency would show bad gain AND pole everywhere — the opposite). Harness `tests/common/complex_ac.py`, `verify_nn_ac.py`, `verify_complex_opamp_ac.py`. Memory `[[v65-nn-ac-accuracy]]`.

---

## AC analysis — small-signal frequency-domain (branch `feat/ac-analysis`, 2026-06-21)

Brought `.ac` from a ~60%-scaffolded, dead-on-arrival state to a working, NGSPICE-validated feature (`ACSolver` solves complex `Y = G + jωC` per swept frequency about the DC OP). Fatal fixes: `run_ac_sweep` imported an absent `pandas` (→ stdlib `csv`); `_stamp_mosfet_ac` skipped the MOSFET caps → added the **transcapacitance stamp** from the source-referenced 2-port `M=[[cgg,−cgd],[−cdg,cdd]]` embedded in the nodal 3×3 (charge-conserving, vanishing at ω→0). Validation `tests/verify_ac.py` 2/2 (passive RC 0.0000% NRMSE; BSIM-CMG NMOS CS-amp vs NGSPICE on the identical OSDI, gain err 5.4e-6 dB — transcapacitance sign confirmed). Gotcha: ngspice `wrdata vp()` emits radians (dump complex `v()`, compute phase in Python). Memory `[[ac-analysis-feature]]`.

---

## V6.4.9 — DirectNet small/medium/large capacity benchmark (branch `feat/v6.4.8`, 2026-06-21)

Clean single-recipe capacity study (24 ckpts, S 128×3 / M 256×5 / L 384×6). Circuit pass-rate rises with capacity **6→9→12/16 (S→M→L)** — but the composition is the finding: device Id-Vgs + inverter accuracy is excellent at EVERY size (NOT the bind; large slightly over-fits the device surface). The **opamp is the value-surface-fragile gate** (gain≈0 at S/M all techs; recovers only at `large`, and only tsmc5/tsmc12). Switchcap needs capacity (0→3/4; tsmc5 never); ring-osc tsmc12/16 every size + tsmc7 at large; SRAM butterfly 4/4 every size. More capacity does NOT close the recipe-sensitive gaps. Harness `scripts/benchmark_*`. Memory `[[v649-sml-capacity-benchmark]]`.

---

## V6.4.8+ — complex-circuit parametric sweep harness + TSMC7 broad retrain (KILL) (branch `feat/v6.4.8`, 2026-06-20)

Built the complex-circuit parametric sweep harness (`tests/common/complex_sweep.py` + 4 `verify_complex_*_sweep.py`, per-circuit stimulus, baseline-gated, sha256-pinned). **TSMC7 broad retrain = KILL (confirms S1):** retraining `medium` on broad `tsmc7_v2` data (to widen the swept envelope) drove opamp **gain→0** — breadth fits the value surface but COLLAPSES the offset-dominated opamp; reverted to the specialized pivcor (8.63% PASS, 15/16 protected). Sweep envelope: the opamp holds gain under *load* perturbations but collapses under almost ANY OP change (VT/NFIN/VDD/vcm); ring/switchcap robust. Also fixed an L cache-key bug + a single-point switchcap clock-render bug (the same class V6.5.3 later traced as the real switchcap failure). Memory `[[v648-broad-retrain-collapses-opamp]]`.

---

## V6.4.8 — value-surface accuracy campaign; ship the S2 win, 14 → 15/16 conditional (branch `feat/v6.4.8`, 2026-06-17→20)

**Methodology locked: all gates run CPU** (the fragile opamp lands a different NR basin on CUDA — `[[v648-gate-cpu-vs-cuda-basin]]`). **S0 floor-k KILL** — gain is wildly non-monotone in the `_floor_gds` coeff (it hops NR basins, not a gain∝1/k lever; gds cancels at the fixed point; the k=2.0 "PASS" is a false-pass) `[[v648-gds-floor-inert-on-opamp-gain]]`. **S1 `--size large` KILL** — the larger net fits the value surface BETTER yet collapses the opamp; capacity is not the bind `[[v648-s1-capacity-not-the-bind]]`. **S2 continuation-first DC sweep KEEP (the sole win, load-bearing)** — `run_dc_sweep` solves warm-started NN points from the neighbour with source-stepping OFF (GMIN retry as fallback; gated on `has_nn` so BSIM-CMG is byte-identical); tsmc7 opamp 10.78% FAIL → 8.63% PASS deterministically; the win is **path-preservation**, not basin-de-fragilization `[[v648-s2-continuation-first-opamp]]`. **S3 EKV analytic backbone KILL** — the additive-in-asinh residual overwhelms the offset-dominated µA band, neutral on switchcap `[[v648-s3-ekv-backbone-kill]]`.

---

## V6.4.7 — serialized accuracy campaign; SHIP at 14/16 + force_ic 8/8 (2026-06-10→16)

A strict serial S1–S19 chain (every lever committed-or-rewound before the next). Start = V6.4.4 canonical 8/16. **Durable behavioral changes still in the code:** S2 — the NMOS source-frame fix (NN Rule 2; `_raw_voltages` shifted only PMOS, so lifted-source NMOS saw phantom Vgs/Vds; new permanent canary `verify_nn_lifted_source_dc.py`, was 10–64% NRMSE, flipped tsmc12 opamp); S7 — the reverse-Vds clamp relaxation in `_apply_vds_correction` (C¹ taper, Id(Vds=0)=0 exact; shipped window 0.20/0.30·VDD_train — the wider 0.30/0.40 corridor was KILLED: tsmc5 opamp veto + force_ic collapse). **Key dead-ends / findings:** S6 — simulator EXONERATED (native-L72 ring-osc control = NGSPICE ratio 1.000); S9b — regen-v2 data + 2 load-bearing data-gen bug fixes: the `NN_DC_SOLVE_TOL` floor (legacy 1e-9 returned exact-0 for |id|<1e-9 → the zero-row artifact; 1e-12 for generation) + an atomic-write fix for a parallel modelcard-cache write race (partial card → degenerate, physically-wrong rows); S10 — Sobolev id-derivative KILL, MAJOR finding **derivative fidelity is ANTI-correlated with the opamp** (the Jacobian guides NR convergence but cancels at the fixed point; opamp gain / RO period are value-surface / NR-fixed-point owned) `[[v647-s10-deriv-fidelity-vs-opamp]]`; S12 — trajectory-corridor KEEP (11→14/16; harvest the bias tubes the transistors visit along the ground-truth trajectory, OSDI-label, append with `--class-weights` — the V6.5.5 corridor descends from this); S11 — subthreshold-id KILL (moved force_ic the wrong way); **S17c — force_ic 0/8→8/8 was a HARNESS BUG** (the 6T netlist pinned wordline=VDD with both bitlines forced = a non-physical read-disturb that exact OSDI physics ALSO fails; wordline-OFF retention → both NN and ground truth rail 8/8) — LESSON: run the native-L72 control before blaming the NN `[[v647-s11-subthreshold-vs-forceic]]`; S19 — replication discipline caught a bistable false-pass; trust the `verify_complex` gate, not the scorer proxy `[[v647-s19-scorer-vs-gate-opamp-replication]]`. Open at ship: tsmc5 switchcap 12.14% (later a clock-bug, V6.5.3), tsmc7 opamp 10.78%.

---

## Condensed history (pre-V6.4.7)

> Full detail for these iterations lives in `git log` and `MEMORY.md`. Only the
> durable outcomes are retained here; the verbose per-phase narratives and the
> pre-V6.0 (v3/v4/v5) exploration logs were pruned in the 2026-06-20 slim.

### V6.4.6 — diagnosis-first iteration (2026-06-01/02, no behavioral change)
Gated every GPU-spend behind a 0-GPU diagnostic. Closed the measurement framing of
two gates (TSMC7 ring_osc, SRAM `force_ic`) and localised the RO error to the
**id VALUE surface** (not the derivative). Probe/measurement fixes only; the
inference path was unchanged. Set up the agenda V6.4.7 then executed.

### V6.4.5 — Track A no-ship iteration (2026-05-29)
Ran all 5 planned phases; **shipped nothing**. Built + validated the multi-circuit
scorer (durable infra, reused later). Ruled out several value-surface levers and
confirmed the RO/SRAM gaps were architectural, not tuning — feeding V6.4.6/7.

### V6.4.4 — DirectNet per-tech checkpoint mix (2026-05-28, inference-only)
First per-tech medium checkpoint mix for TSMC5/7/12/16; complex-circuit pass rate
+2 vs the V6.4.1 baseline (canonical 8/16). Restored the load-bearing V6.4.2
Phase-7a `_MonotoneVgResidual` + `--monotonic` code (on-disk checkpoints carry
`mono.*` state_dict keys; stock checkpoints route `mono=None`, no inference change).

### V6.1 – V6.3.2 — per-tech DirectNet establishment (2026-05-12 → 05-15)
- **V6.1**: per-tech dedicated DirectNet for TSMC5/7; destructive cleanup of the
  universal `refac_*`/`v4_*` artifacts (deleted 2026-05-12).
- **V6.2**: Rule 15(a) terminal-current sign fix; Rule 20 dead-band closed.
- **V6.2.1**: per-tech TSMC12/TSMC16 extension (3 registry edits + data/train).
- **V6.3 / V6.3.1**: inverter spike-removal sprint — dataset regen (`_inv_trip_points`
  recenter on VDD/2 + `_reverse_vds_points` corridor); shipped V6.3.1 with one open
  VTC MaxErr gate.
- **V6.3.2**: ported the PyCMG L3 parametric DC/transient sweeps to DirectNet
  (`tests/common/nn_sweep.py` + `verify_nn_multi_tech_{dc,tran}.py`).

### Pre-V6.0 (v3/v4/v5, package refactors, early milestones)
The BSIMAR package refactors (2026-03/04), the v3 LOO cross-tech sprint, the v4
tech-code migration, the analytical Vds-correction + rail-restoring fixes, and the
v5 inverter-transient phases are recorded in `git log` and `MEMORY.md`. Legacy
LEVEL=1 (Shichman-Hodges) was removed; LEVEL=72/73/74 are the supported models.
