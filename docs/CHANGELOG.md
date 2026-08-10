# PyCircuitSim — Detailed Changelog

This is the long-form history of PyCircuitSim. CLAUDE.md keeps a one-paragraph
"current state" summary; everything below is here so the conversation context
isn't burdened with chronology. (Compressed 2026-07-03, re-condensed 2026-07-23,
2026-07-30 and 2026-08-10 — every entry and verdict retained, verbose prose
pruned; the full original text lives in git history.)

---

## V7.4.0 — clean rebuild + GPU fidelity re-gate on new hardware (branch `v720-gpu-scaling`, 2026-07-30 → 2026-08-06)

**COMPLETE for the CPU accuracy axis.** The box changed (`memlab-gpu2`, fresh
clone, no datasets/checkpoints), so every NN number was re-earned: all 10
datasets regenerated (`--enable-inv-trip --enable-subvt-off`), DirectNet and
BSIM-AR rebuilt 40/40 checkpoints each on the one clean recipe
(`--apply-filter off --swa-mode ema --seed 42`), 240/240 suite runs per family,
reports regenerated and `--check` clean. Recipe rows and PFN were deliberately
NOT retrained — their reports stand as labelled V7.3.0 historical evidence
(an empty local source cannot reconstruct measured rows). Training ran on GPU;
headline gating stayed CPU-pinned (fidelity is a CPU/flags-off property).

**Infra:** `scripts/v740_campaign.sh` (drain-loop orchestrator),
`v740_tf_rescue.sh` (claims OOM-abandoned stems mid-wave), `v740_tf_fill.sh`
(post-wave refill), `v740_tf_reaper.sh` (two-cycle kill of dead-incomplete
stems — a killed run's leftover `_best.pt` otherwise makes every retry SKIP;
retrain is from scratch with `--force` so the control stays one uninterrupted
run). `v730_docs_build.py` gained `--only/--recipes`, a complete-matrix guard,
per-report pass pin and SHA-256 manifest, so a partial pass cannot overwrite a
coherent report; `v740_regate` registered as the V7.4 evidence pass.

### Verdicts — three V7.3.0 claims retracted

- **"DirectNet peaks at `large`" — RETRACTED.** Monotonic 11→12→14→**15/20**,
  `xl` best; but the climb is switchcap+opamp only (ring frozen 2/5, SRAM
  saturated 5/5). `large` stays production on cost (`xl` = 2.3× for one cell).
- **"`tsmc7-opamp` at `large` is open" — RETRACTED.** TSMC7 passes the opamp
  at all four tiers (3.9–6.1 %), strongest opamp column.
- **"`tsmc5-opamp` is the thinnest margin" — RETRACTED.** 0/4: three tiers
  rail, `large` misses 11.6 % vs the 10 % gate.

### Measured noise floor — the TSMC6 repeat paid off

TSMC6/TSMC7 train on bit-identical rows; they agree 14/16 (all rings, SRAMs,
switchcaps) and split twice, **both opamp** — a 2-cell spread in per-tech
totals. **A ≤2-cell opamp difference is noise, not a result.** Applied
immediately: `tsmc16-opamp` went 0/4 (passed twice in V7.3.0) — exactly 2
cells, so flagged *not established pending a second seed*, not a regression.

### New findings

- **`xl` breaks parametric DC for the first time** — 66/69 vs 69/69 at every
  other tier: capacity now damages the device surface *while the circuit score
  climbs* (device-vs-circuit inversion: `medium` best device fit, `xl` best
  circuit tier).
- **Low-VDD rings are an exception-free partition** — TSMC5/6/7 fail all 12
  ring cells, TSMC12/16 pass all 8; TSMC5 worsens with capacity.
- **`switchcap` at `small` fails 4/5 on hold droop**, not headline charge.
- **TSMC5 device AC reports `f3db = nan` at `large`/`xl`** — degenerate fit.

### BSIM-AR clean verdict — stronger, but not flat

Strict curve **18→17→15→13/20** (`small`→`xl`), zero flips — retracts the
V7.3.0 claim that BSIM-AR is flat across capacity. `small` misses only TSMC5
switchcap (hold droop) + the noisy TSMC7 ring. The decline is circuit-surface,
not device damage (transient 80/80 every tier; device AC 9–10/10; parametric DC
65–68/69). TSMC6/TSMC7 agree 15/16 (sole split inside ring noise) — the old
"all 16 repeat cells reproduce" claim is retracted. V7.3.0 corridor recipes
still record 20/20 but were not retrained; they are historical evidence, not
the current artifact set.

### GPU acceleration fidelity — remaining V7.2 gate closed

- **T3 full bundle** (`commit+gpu+stamp+NATURAL`, DirectNet clean `large`):
  **48/48 executed** (4 techs × 4 circuits × OMP {1,2,4}); binding
  SRAM+switchcap 24/24, Rule 2 15/15 on CUDA, zero flips/OOMs/errors.
  Report-only basket 12/16 strict — exactly the V7.4 CPU clean-`large` basket,
  metrics identical except one 0.02 pp opamp delta with the same verdict.
- **T4 full latch basin: 8/8 PASS, zero basin flips**; worst max|ΔV|
  0.1206 mV, worst q-NRMSE 0.0101 %VDD.
- CUDA stays opt-in because CPU/flags-off remains the scored compatibility
  contract, not because a gate is open. Evidence:
  `results/v720_gpu_regate/t{3,4}_gpu_bundle/`.

### TSMC6 ≡ TSMC7: §7 confirmed exhaustively

Of 1748 shared modelcard parameters 11 differ + 74 TSMC6-only + 12 TSMC7-only
= **97 keys, all with zero occurrences in the BSIM-CMG Verilog-A** (vs 333 of
the identical keys that do appear). All 97 are TSMC TMI layout-effect
parameters; no core device-physics parameter differs. Over-determined: OpenVAF
never compiles them AND `nn_generate.py` stamps only `{L, NFIN, TFIN,
DEVTYPE}`. The regenerated `tsmc6_*.npz` are bit-identical to `tsmc7_*` bar
the tech-name field.

### Dead end recorded

- **Round-robining BSIM-AR training over a shared GPU** (`GPUS='1 1 0'`;
  GPU 0 held ~9.5 GB free) — `large`/`xl` jobs OOM there at epoch 0. The wave
  was not restarted (would discard a 75-epoch in-flight job); recovery layered
  instead (rescue → fill → `--require-complete`). Pin big tiers to a GPU with
  real headroom.

---

## V7.3.0 — accuracy reports restructured, re-gated on one code state (2026-07-27 → 29)

**Nine cross-cutting accuracy documents became two per family**
(`{DirectNet-L73,BSIM-AR-L74,PFN-L75}-{clean,recipes}.md`): the clean report
answers per tech/scale/testcase, the recipes report carries the addenda,
cross-cutting material lives once in `methodology.md`, and the pre-fix archive
keeps only the register of retracted claims (frozen tables recoverable via
`git show 37cef77:docs/accuracy/archive-pre-gds-fix.md`). **All tables are
generated**: `v730_docs_build.py` (`--check` fails on stale files),
`v730_coverage.py` (coverage by pass + runnable gap job file),
`v730_control.py` (cross-pass comparison). 2173 → 1768 lines with more
measurement behind them.

**Denominators /16 → /20:** TSMC6 folds into the headline (complex /20, device
AC /10, opamp AC /5). No total is comparable across the boundary without
rescaling. TSMC6 remains TSMC7 relabelled; only presentation changed.

**Measured** (1536 cells, 12 clean + 14 recipe groups; the control reproduced
176/176 jointly-measured cells at OMP=1, licensing the V7.1.0 merge):

- **BSIM-AR clean strict at all four tiers, identical scores at each**, zero
  flips over a 22× capacity range; the open set is the low-VDD ring column only.
- **`corroft@medium`, `corro15@medium` and all four corridor recipes at `xl`
  sweep 20/20** — `corroft@medium` the only checkpoint passing every cell.
- **TSMC6 retires DirectNet's sweep:** `crit15m@xl` passes `tsmc7-opamp` and
  fails `tsmc6-opamp` — same data, same recipe, different training run/basin.
  A sweep a duplicate column can break was never a sweep. Every such split is
  in the bimodal opamp column.
- **PFN's first curriculum arm** (`corroft@small`): rings 3/5→5/5, opamps
  2/5→0/5 — total unchanged, failure set replaced. Cleanest instance of
  curricula *relocating* basins rather than composing them; third architecture
  where the corridor is the ring lever.
- TSMC6 recipe checkpoints trained (26); its freshly harvested ring corridor is
  `array_equal` to TSMC7's — a fourth reproduction of the duplicate finding.

**Fixed:** `--cuda` no longer silently falls back to CPU (a faulted 4090 turned
a run into a 50× slower CPU run that would have been gated normally; guard
exits 1 — note torch enumerates FASTEST_FIRST so nvidia-smi index ≠ CUDA
index). `v730_coverage.py --require-complete` keeps half-trained checkpoints
(bare `_best.pt`, no `.complete`) out of gates.

**Dead ends / non-fixes:** the opamp open-loop AC bias-resolution defect is
documented, not fixed (gate changes are decisions, not side effects). "Every
family is flip-free" retracted a second time — the rule is now "a nonzero flip
count is unbankable, re-measure".

---

## V7.2.0 — GPU-accelerated large-scale SRAM transient (branch `v720-gpu-scaling`, 2026-07-27/28)

**All phases of `docs/plans/2026-07-26-v720-gpu-scaling.md` landed.**
Bit-identical work ships default-on; perturbing levers ship default-off behind
env flags (same discipline as V7.0.x). Workload context: a 200-step SRAM write
op = 483 NR iterations + 200 commit points; the rev-4 discovery was the
post-step charge-history commit running one cache-cold batch-1 eval per device
per step = 75–85 % of transient wall.

### Bit-identical, default-on

- **Phase 1, parse** `4d76c22` — cache-first `torch.load`/norm/sidecars,
  memoized resolver. `PYCIRCUITSIM_NN_DEVICE` default **cpu** (fixes the
  silent-CUDA provenance bug). 32×32 parse ~44 → 4.67 s.
- **Phase 2a-lite + 2c** `146f05f` — one D2H block transfer, deduped constant
  tensors, value-keyed geometry/tech-code cache.
- **Phase 2a-full, batched denorm tail** `e6f8154` — per-device `_unpack_eval`
  → one float64 numpy pass, all three families. §8.1 bit-exactness enforced in
  code: per-element libm `math.exp` on the Vds-correction (np.exp differs 1 ULP
  on ~4.6 % of args, amplified ~60× by the `1−exp` cancellation), float64
  casts + dtype asserts. New gate `verify_batched_tail.py` 22/22 exact.
- **Phase 2d + 4a, NR-loop vectorisation** `1a576ed` — per-node loops
  vectorised; `Circuit` caches nodes/map behind `invalidate_topology()`; one
  LIL→CSR conversion per NR iteration. A/B byte-identical.

### Perturbing, default OFF (env-flag opt-ins)

- **Phase 2t** `d1b8e40`, `PYCIRCUITSIM_TRAN_BATCH_COMMIT=1` — batch-eval the
  commit loop. 4×4 write op 146 → 34.7 s (4.2×); 16×16 latch end-states
  512/512, max final ΔV 0.52 mV.
- **Phase 3a** `2496b9a`+`38b6920`, `PYCIRCUITSIM_NN_DEVICE=cuda[:N]` — GPU NN
  eval with runtime-enforced T0 determinism pins (TF32 off, deterministic
  algorithms, CUBLAS workspace).
- **Phase 3b** `4e9c396`, `PYCIRCUITSIM_BATCHED_STAMP=1` — NN stamps as one
  COO from cached index arrays; 7.0×/6.1× on stamp+convert; perturbation is
  last-bit (max rel 3e-15).
- **Phase 4a′** `4e9c396`, `PYCIRCUITSIM_MNA_ORDERING=<spec>` — explicit splu
  ordering. **Re-measure on real matrices REFUTED §5.2's synthetic claim:**
  COLAMD fill is benign, `MMD_AT_PLUS_A` is *slower* than shipped, and
  **NATURAL wins 2.4–30×** (128×64 `.op` factor 152.9 → 4.7 ms). Phase 4b
  (KLU-class) demoted.

### §8.4 gating — CPU flag bundle GATED AND PASSING

- **T4 latch-basin** (new gate `verify_latch_basin_gpu.py`): all five CPU flag
  configs 8/8, 0 basin flips, ≤60 µV; `commit+gpu` 8/8 (≤0.37 mV, RTX 4090).
- **T3 16-gate CPU-bundle campaign** (`v720_t3_flag_bundle.sh`, 48 cells):
  **15/16 strict, 0 flips — production cell-for-cell**; binding sram+switchcap
  deviate 0.00 pp. Flags stay default-off; the GPU-axis T3 pass remained open
  (closed in V7.4.0).
- **Phase 0 (partial)** `8a2a18b` — the guard-F discontinuity is the **bulk
  regime** of an SRAM array (37.4 % of evals hit the negative branch), so the
  T1 branch-disagreement counter can never bind; binding tiers are T2/T4.

**Version summary:** same 4×4 write op: 146 s baseline → 100.3 s flag-off
(default-on work) → 20.3 s with {commit, stamp} = 7.2×, latch basins 8/8 in
every T4 config. Regression state (CPU flags-off): reference CSVs
sha256-identical; subckt 11/11; L72 op/dc/tran PASS; AC 2/2; inverter 8/8.
Docs consolidation 2026-07-30: README gained §Performance & GPU Acceleration;
CLAUDE.md de-duplicated.

---

## V7.1.0 — accuracy pivots, pre-fix device/AC re-measure, TSMC6 restored, PFN xl (2026-07-25)

Four threads: `docs/accuracy/` reorganized into cross-cutting pivots
(by-tech/by-scale/by-recipe — retired again in V7.3.0); every number still
standing on pre-`gds`-fix code re-measured (resumable driver
`scripts/v710_regate*`; control confirmed HEAD reproduces the V6.13.0
verdicts); TSMC6 restored; PFN gained an xl preset (14.86 M params, mirroring
BSIM-AR xl; lr 3e-4 after the V6.10 divergence collapses).

- **"AC peaks at SMALL" — retracted.** DirectNet device AC 7/8·8/8·8/8·7/8
  across small→xl (pre-fix 5/12·4/12·4/12·4/12): level *and* shape were
  artifacts of the wrong-signed gds.
- **The production curriculum improved the charge surface too:**
  `v660clean@large` fails TSMC5-NMOS AC where `crit30f` weights are 8/8.
- **"Opamp open-loop AC is 0/4 everywhere" — false** (DirectNet `small` banks
  TSMC16; BSIM-AR banks more, un-railed OPs).
- **Finding (not fixed):** the opamp AC gate has a bias-resolution defect —
  2 mV sweep grid across a 3–14 mV transition, `op_valid` applied to the NN
  only.
- **TSMC6 restored as a controlled repeat, not a technology** — a bit-identical
  duplicate is the only instrument for run-to-run variance with data held
  fixed (the first repeat's 68.2 % vs 2.0 % SRAM disagreement collapsed to
  5.2 % vs 6.2 % once gds was fixed). Tail codes 22–24;
  `assert_tech_is_distinct()` kept with tsmc6↔tsmc7 the sole acknowledged
  duplicate. Scoring rule then: own /4 column, never folded into /16.

---

## V7.0.0–V7.0.4 — NN compact-model performance (2026-07-25)

**Inference DC solve 1.68× byte-identical; training 4.9×/epoch; BSIM-AR 1.6×
behind an opt-in flag. Full measurements + dead ends:
`docs/plans/2026-07-25-v700-nn-perf.md`.** Governing constraint: a last-bit NN
perturbation can land a different NR basin, so every change is bit-identical
(default-on) or perturbing (default-off flag, promoted only after a re-gate).

- **V7.0.0 scan:** inference is **bandwidth-bound** (DirectNet `large` streams
  3.6 MB of weights 4× per eval); training was loader-dominated.
- **V7.0.1:** DC/OP skips charge Jacobians (`_require_nn_caps` contract;
  `get_capacitances` self-heals). 1610 → 784 µs/eval; CSVs sha256-identical.
- **V7.0.2:** `_DeviceBatches` on-device slicing + fused AdamW; 3.4 → 0.7
  s/epoch. Changes shuffle order (retrain ≠ same weights); `BSIMAR_LOADER=torch`
  reproduces legacy.
- **V7.0.3:** fused analytic Jacobian `PYCIRCUITSIM_NN_FUSED_JAC=1`, DEFAULT
  OFF — transient 1.38×, DC slightly slower (transient/AC lever only). Not
  bit-identical; stays off until a 16-gate re-gate.
- **V7.0.4:** LEVEL=74 AR prefix cache `PYCIRCUITSIM_NN_AR_CACHE=1`, DEFAULT
  OFF — 1.60× DC / 1.56× tran. "Exact in exact arithmetic" refuted bit-wise:
  `F.linear` is not row-stable on CPU, so **no incremental AR form can be
  bit-identical in float32**; deviation ≤1.6 µV.

**Dead ends (do not retry):** TF32 / `torch.compile` / bf16 autocast all
*slower* for DirectNet (launch-overhead-bound); replica-batch gradient trick
dominated by the analytic Jacobian; larger training batch excluded (a recipe
change invalidating every comparison).

---

## V6.13.1 — systematic-audit fix wave 1: 22 gate-neutral findings (2026-07-24)

Closed the 22 findings from `docs/2026-07-21-systematic-audit.md` that cannot
change a gated number; the 19 gate-affecting ones staged behind a re-gate.
Classes: **silent-wrong parser** (C1 `+` continuation lines dropped outside
`.model` — now every logical line buffers, orphan `+` raises; case-sensitive
ground; duplicate `X` names merging internal nodes; duplicate `.model`
redefinition; polarity-mismatched env pins falling through); **silent-green
harness** (11 dispatchers exited 0 regardless of sub-jobs; report published
from an empty tree; opamp gate without a minimum-gain guard; typo'd `--tech`
silently SKIPped; production checkpoint slots writable with `--overwrite` —
deleted); **data-pipeline integrity** (geometry-sha256 `.meta.npz` sidecar
fingerprints; universal vocab-size guard; TabPFN `_ctx_cache` invalidation on
state-dict/EMA writes). **Dead end recorded:** the audit's prescribed
`F.softplus` NaN rewrite is measurably NOT forward-bit-identical (23/401
samples) — dropped; the argument-clamp variant measured 401/401 identical if
ever wanted.

---

## V6.13.0 — gds sign + guard fix shipped, TSMC6 retired, every checkpoint re-gated (2026-07-24)

**Shipped the audit's last P0 (`A3`, the NN `gds` sign bug), retired TSMC6,
re-gated all 36 checkpoint sets. Production DirectNet 14 → 15/16 strict.**

**The fix (`8ed35bd`):** inference negated `gm`/`gmb` but not `gds` — all
three derive from the same signed `id`; the loss had negated all three since
V6.4 and the correction never reached inference (autograd vs `-gds_head` =
0.12 rel err, vs `+gds_head` = 2.08 — the signature of a pure sign flip). The
old two-sided floor `max(gds, |id|*0.5)` asserted an Early voltage ≤2 V,
overriding the learned conductance at 90.9 % of amplifying points — load-
bearing only because it masked the sign error. Replaced by **guard F**:
positives pass bit-identical, negatives clamp to `|id|/50 V` (OSDI −d(id)/dVd
is positive at 100.0000 % of 111,630 conducting evals). **Sign and guard must
ship together** — sign alone is bit-identical, guard alone regresses device AC.

**TSMC6 retired** (`38c47d8` + PyCMG `23b0ace`): audit §D1 — bit-identical
datasets, identical L72 currents. 22 checkpoints + registry entries deleted;
codes were tail so nothing renumbered. (Restored as a controlled repeat in
V7.1.0.)

**Re-gate — one signature dominates: every gained cell is an opamp** (`gds`
sets small-signal output resistance; it cancels at the Newton fixed point
everywhere else). DirectNet production 14→15/16 strict zero flips
(`tsmc16-opamp` closes; `tsmc7-opamp` sole open cell at `large`).
**`crit15m@xl` = 16/16 STRICT, zero flips** — first full-matrix sweep under
one uniform recipe; not promoted (2.3× inference, no device gain). Clean
capacity curve 10/10/13/12. Universal: +3 strict and **all three OMP FLIPs
eliminated** (a wrong-signed Jacobian entry was the thread-count sensitivity).
PFN 11/11/9; device AC at `large` 5/8→8/8. **BSIM-AR clean 14/16 at every
tier** — "capacity peaks at medium" was largely this bug; ceiling moved opamp
→ ring. Corridor recipes: `corroft@medium`/`corro15@xl` 16/16 STRICT. **gds
moved opamps, the corridor moves rings — independent levers.** DC confirmed
exactly invariant (the single parametric-DC failure is bit-identical pre/post).

**Retracted:** "`tsmc7-opamp` reachable only by the V6.5.9 T3 fine-tune"; "the
three-basin simultaneous hold is the open 15/16 target"; BSIM-AR's
capacity-peak-at-medium.

**Also:** methodology note — the resumed re-gate half ran from a frozen rsync
snapshot of `d2ea720` (verified byte-identical), decoupling an 8-hour campaign
from the working tree; worth repeating.

---

## V6.12.1 — silent-green P0 branch merged + accuracy reports per family (2026-07-24)

Merged `fix/silent-green-p0`: `tests/common/base.py` never checked NGSPICE's
exit status — a dead binary left a stale CSV and everything passed against
**stale ground truth** (reproduced with `NGSPICE_BIN=/bin/false` → 8/8 PASS);
now unlink-before-invoke + raise. `solver.py`: `spsolve` returns NaN on
singular matrices so the `LinAlgError` guards were dead code; now detected.
Also landed the 5-area ~70-finding audit register
(`docs/2026-07-21-systematic-audit.md`, incl. gds §A3 and TSMC6 §D1) and
consolidated seven per-version accuracy reports into three per-family ones.

---

## V6.12.0 — .subckt/.ends hierarchical netlists (2026-07-18)

Added `.subckt`/`.ends` + `X` instances (flattening at parse time,
ngspice-style — solver untouched), `.ic` into subckt bodies, converted the
test circuits to hierarchy, re-ran the full matrix: **zero regressions**
(484/489 checks, all 5 non-passes pre-existing). New gate `verify_subckt.py`
(8/8 → 11/11 in V6.13.1), subckt ≡ flat at max|ΔV|=0. Three pre-existing CLI
defects surfaced by README smoke-tests fixed (trace-count log, missing
transient CSV/lis, duplicated final sample from an IEEE-754 quotient — now
snapped at rel-eps 1e-9); README brought up to date (was stranded pre-V6.5).

---

## V6.11.0 — TSMC6 NN family trained + gated at every scale (2026-07-14/17)

Completed the V6.9.0 deferral: all three NN families trained/gated on TSMC6
(22 clean ckpts). ⚠ TSMC6 later found ≡ TSMC7 — the splits below are basin
coin-flips, not distinct-tech fidelity. Complex 4-cell matrix: DN peaks
`large` 3/4 (opamp rails at every size), BSIM-AR peaks `medium` 3/4 (the only
family to pass the tsmc6 opamp), PFN flat 2/4. Device fidelity complete for
all 11 cells. **Bug fixed:** `tech_code_in_vocab` rejected TSMC6 — the
ASAP7-guard checked the *universal* code against an 18-ceiling, silently
SKIPping per-tech TSMC6; now any tech in `LOCAL_VARIANT_CODES` passes.
Campaign ran under sustained cluster overload (loadavg ~1400/192); gate
timeout raised 1800→7200 s; complex matrix 100 % resolved.

---

## V6.10.0 — TabPFN port: the "PFN" family, LEVEL=75 (2026-07-11/14)

Ported TabPFN v3 into the bsimar stack as a third family and ran the full
scale campaign (24 ckpts). **PFN clean small (0.69 M) = 11/16 STRICT with ZERO
flips** — the first family with no OMP multistability; strongest clean small
on record. Architecture: faithful scaled-down port of the three v3 stages +
two deviations (frozen learned context baked into the ckpt with context-KV
caching; direct 13-output value head for smooth autograd). Gate curve
**declines** 11/10/8 (s/m/l); device fidelity peaks medium; 15.6 ms/eval CPU
(4× faster than BSIM-AR). Root-caused: `nmos_nfin_10` off-grid geometry fails
at s/m (context-relative embedding interpolates the NFIN 6→21 gap poorly);
capacity repairs it. 8 divergence-collapse events at `large` (both lr tried);
diverged runs bank pre-divergence EMA bests. **Dead ends:** fp32 large @300ep;
"lr 3e-4 fixes divergence" refuted. Pretrained 58 M TabPFN ICL baseline: not
solver-viable (id 3.2–5.7 % — the from-scratch port beats it ~10× at 1.2 % of
params).

---

## V6.9.0 — TSMC6 (CLN6) onboarding + TSMC PDK parse audit (2026-07-12)

Onboarded the N6 iPDK card as sixth tech (LEVEL=72 + NN plumbing + datasets;
NN training deferred to V6.11.0) and audited PyCMG's card parsing across all 5
TSMC cards. ⚠ Later corrected: TSMC6 is TSMC7 relabelled. Universal codes
22–24 tail-appended (existing ckpts stay valid). **Real defect fixed:** the
12-param fingerprint labeller (`loo_labels.py`) used silent last-writer-wins
and tsmc6↔tsmc7 collide 108/108; per-tech datasets now label against their own
tech and the universal scan raises on cross-tech collision. Verification:
PyCMG pytest 314/314; TSMC6 DC 9/9, tran 14/14; full 6-tech regression clean
bar the pre-existing `TSMC5_lvt_inv_l_24nm` ERROR. **Parse audit PASS** (40
devices, 0 round-trip mismatches); durable verdicts: mid-line `*` in blocks is
multiplication NOT a comment (`_extract_model_params`'s no-strip is
load-bearing); bin selection inclusive-both-ends is self-consistent
PyCMG↔NGSPICE; TMI/stat params ignored identically by both.

---

## V6.8.1 — BSIM-AR xl-tier fill (2026-07-11/23)

Trained the xl preset (14.81 M) across the full Phase-B recipe mirror (48
ckpts) and gated. **xl TIES medium at 15/16 strict, does NOT exceed it, does
NOT basin-shuffle** (the DirectNet xl shuffle does not replicate); tsmc7-opamp
is the only miss for corroft/crit15m/corro15. **AC COLLAPSES at xl** (opamp-AC
0/4 every recipe; device AC weak; charge-Sobolev does not recover it — a tier
property, not recipe-fixable). **No-promote:** 3× params + ~30–100× AR
inference for the same 15/16; `corroft@medium` remains the validated best.
Ops lesson: never reuse a `gate_iso_*` output dir across model families.

---

## V6.8.0 — BSIM-AR Transformer (LEVEL=74) un-parked: recipe campaign (2026-07-06/07)

Shipped the DirectNet training/recipe/eval stack to the parked AR Transformer
and ran the full scale × recipe campaign. **`corroft@medium` (1.9 M) = 15/16
STRICT, beating DN production (14/16)** — banks tsmc16-opamp + both low-VDD
rings; misses only tsmc7-opamp. Port notes: `unknown_code_id` was hardcoded
universal 17 (local vocab would CUDA-assert) → `num_tech_codes-1`; aux losses
force the MATH SDPA backend (fused SDPA has no double-backward); LEVEL=74
parser cascade + `PYCIRCUITSIM_NN_FORCE_LEVEL=74`. **Latent L74 bug fixed:**
`_out_col` ranked the canonical column order before the BSIMAR layout → qg
denormed as id (~5× current) though the module scored 0.38 % on the trainer's
test set — classic silent-green, caught by the first real checkpoint. Scale
study: complex peaks MEDIUM; AC peaks small; **inv_trip anchor is INERT on
the Transformer** (opposite of DN); the corridor is the whole ring lever;
tsmc7-ring ⊥ tsmc7-opamp under the corridor at every tier.

---

## V6.7.1 — house-clean after the V6.7.0 campaign (2026-07-05)

~12.7 G reclaimed; nothing load-bearing touched; retired checkpoints archived
to `/data2/shenshan/v66x_v670_retired_ckpts_2026-07-05.tar.gz` before deletion
(recorded-loser recipe pool + failed fine-tunes pruned; production/alternates/
universal bases kept). Datasets 26 G → 19 G (regenerable concats deleted;
corridor sets kept — retrainability over reclaim). The 580-line collector
output merged into the DirectNet accuracy report as Part II.

---

## V6.7.0 — universal DirectNet + TSMC5 transfer study (2026-07-04/05)

Resurrected the universal-scope DirectNet (ONE 18-code model on TSMC16+12+7),
ranked the Core-4 recipes on the 12 shared complex gates, measured TSMC5
few-shot transfer. Headlines: (1) **universal is VIABLE** — device fidelity
per-tech-grade; **`corroft` = 10/12 strict, 0 FLIPs = per-tech parity with
full OMP determinism** (which per-tech large never had); corridor fixes
tsmc7-ring 14.89→3.61 %; anchor/csob basins do NOT survive the scope change
(recipe→basin maps are SCOPE-dependent). (2) **TSMC5 onboarding = ~1 M
stratified rows** (`plain@n1M` 4/4 STRICT at half the data; ≤10 k DIVERGE;
n1M beats nfull). (3) **No free retention** — source techs collapse at gate
level; fine-tune = de-facto per-tech ckpt. Phase 1b: the opamps-XOR-rings wall
reappears at universal xl → `corroft@large` is the final best universal
config. Env-pin-only; per-tech resolution untouched.

---

## V6.6.7 — 15/16 hunt round 1: csobcrit + crit30a1 both 13/16 (2026-07-03)

Both routed arms NEGATIVE; production stays `crit30f@large` 14/16. `csobcrit`:
the curriculum **relocates rather than composes** basins (csob's deterministic
tsmc16-opamp hold degrades to a FLIP while tsmc12-opamp is gained). `crit30a1`
(half anchor): reproduces corroft almost exactly — **the {16} → {5,12}
opamp-basin hop is DISCONTINUOUS in anchor weight ∈ (1.0, 2.0)**; the uniform
lever is exhausted for the 5+12+16 triple → 15/16 routes to structural levers.
Harness: crit-family `--init-from` at `large` redirected to the v660clean
archive (production slots carry crit30f — warm-starting from them would
silently stack curricula).

---

## V6.6.6 — xl curriculum ties production 14/16 strict + full test-infra audit (2026-07-03)

`corroft`/`crit10`/`crit15m`@xl = 14/16 STRICT (all bank tsmc16-opamp which
production fails); production unchanged. **The weight→basin map is
TIER-dependent**; curriculum warm-start rescues xl wholesale (clean@xl 10→14);
xl basins are OMP-deterministic unlike large's endemic opamp flips.
**Test-infra audit — 17 verified fixes**, silent-green class: >100 %
divergences scored as ERROR-skip → exit 0; all-ERROR suites exited 0; **an
absent `PYCIRCUITSIM_NN_CHECKPOINT_*` pin silently fell back to production**
(now raises); flat-reference `nrmse()` returned 0 = auto-PASS (now inf);
SUMMARY clobbering on subset re-runs; train-resume trusted `_best.pt` (now
`.complete` markers). The v664-P0 torch thread-pin landed in the complex
harness (verdict-neutral).

---

## V6.6.5 — recipe×size matrix completed: 13 recipes × 4 sizes (2026-07-03)

208 ckpts, 864 eval cells, zero blanks. `clean@large` 13/16 stays the unbeaten
in-matrix cell; **the corridor inverts the capacity curve** (dominates below
`large`, collapses above); **xl is basin-shuffled, not uniformly over-fit**;
AC peaks at SMALL across recipes; device-NRMSE bottoms at medium; tsmc7-opamp
0/52. Ops: 22 killed-run "best-so-far" ckpts quarantined + retrained (a
`_best.pt` on disk is NOT evidence of a completed run — gate on the log tail).

---

## V6.6.4 — crit30f PROMOTED to production (2026-07-02)

All 8 `tsmc{X}_dn_large_*` production slots replaced with the V6.6.3-validated
`crit30f` checkpoints (clean base + one curriculum fine-tune, ring-only
`corro` data); clean originals archived as `tsmc{X}_dn_v660clean_large_*`.
Production 13/16 (12 strict) → **14/16 strict**, verified on the default
resolver path. Checkpoints are gitignored — this entry is the record.

---

## V6.6.3 — full-recipe re-test: crit30 supersedes crit15 at 14/16 STRICT (2026-07-02)

All 22 on-disk recipes re-tested under one discipline (isolated matrix + OMP
determinism sweep). Best = `crit30` 14/16 STRICT; validated by `crit30f` (all
8 ckpts retrained to full spec — the honest rerun reproduces cell-for-cell).
The corridor-weight → tsmc5-opamp basin map is **non-monotone** (w1.0 FLIP,
w1.5/2.0 detFAIL, w3.0 detPASS — the inv_trip anchor makes w3.0 safe). csob
re-scoped to the AC/device alternative (12 strict). tsmc7-opamp: 100 % FAIL
across all 23 artifacts × all OMP.

---

## V6.6.2 — the cross-wall combo breaks 13/16: crit15 = clean+1 (2026-07-02)

REFUTED the V6.6.1 "13/16 uniform ceiling": `crit15` (corridor + inv_trip
curriculum — the two levers had only ever been tested separately) nets 13/16
strict = clean+1, the +1 the DETERMINISTIC tsmc5-ring opening (12.66→4.0 %).
Confirmed live that the opamp gate is a multistable OMP coin-flip — single-run
opamp passes are unbankable. Round-2 arms read NEGATIVE on single runs —
corrected by V6.6.3's strict re-test.

---

## V6.6.1 — uniform-recipe comparison sweep (2026-07-01)

Swept uniform recipes (csob/sob/ekv/seeds/combos) across all 4 sizes: **NO
uniform recipe beats clean's 13/16 at `large`**. The ceiling is mutual
exclusivity of value-surface basins — each recipe/seed lands a different
subset; combo stacking is zero-sum (EKV core breaks tsmc12/16 SRAM). `csob` =
best all-rounder → documented alternative; refutes the V6.5.x "charge-Sobolev
dead on arrival" verdict (measured only at `medium`). `sob` reconfirms
deriv-fidelity ⟂ opamp.

---

## V6.6.0 — house-clean + uniform-recipe reset (2026-06-29)

Deliberate reset from V6.5.9's hand-tuned 16/16 to the honest uniform
baseline: all 32 DirectNet ckpts retrained on ONE recipe; production = uniform
`large` at 13/16 (capacity curve 7/10/13/10 s→xl). The 3 open gates are the
true fidelity frontier the V6.5.x per-case specials had force-closed.
House-clean: datasets 16 G → 4.5 G; V6.5.x specials archived off-repo
(`/data2/shenshan/v6.5.9_production_specials.tar.gz`; rollback = `git checkout
V6.5.4` + untar); resolver prefers per-tech `large` first.

---

## V6.5.9 — ★ 16/16: T3 differentiable-DC-solver lands the tsmc7 opamp (2026-06-29)

First-ever tsmc7 opamp PASS (gain 178.0 vs 163.4, 8.92 %) → production 16/16.
Put the DC solve **inside the loss**: a differentiable unrolled Newton solver
supervises the emergent transfer curve against L72, so r_o is shaped by the
gain target. Broke the V6.5.8 "gain stuck ~370" wall — the gain-163 root DOES
exist; "370" was the gate's continuation landing on an over-flattened branch.
The gate gain is bimodal + sampling-noisy; **preservation, not existence, was
the binding work**. Installed via symlink (retired in V6.6.0). Memory
`[[v659-t3-solver-lands-opamp-16of16]]`.

---

## Test-infrastructure correctness sprint — 11 bugs fixed (2026-06-28)

Production pass-rates unchanged; every fix re-checked vs NGSPICE. **B1
(CRITICAL):** the per-tech device gates pinned **tsmc5's** net for ALL techs;
routed through `_cascade_handles_stem`. **B3/B5:** SRAM scored PASS when every
corner errored (`all([])==True`) and never compared to ground truth; now ANDs
point-by-point NGSPICE tracking, `force_ic` reconciled as a printed
diagnostic, not a gate. **B4:** a diverged inverter transient could
false-PASS. Plus sweep↔gate `uic` canary, ASAP7 skip, real-deck canary,
honest exit codes.

---

## V6.5.8 — EKV high-r_o core breaks the tsmc7-opamp rail (2026-06-28)

First non-railed tsmc7 opamp of the campaign (structural EKV core +
vout-weighted KCL fine-tune → real amplifying curve, gain ~350–381) —
**REFUTES the V6.5.6/7 "only T3" verdict**. BUT gain ⟺ existence are coupled
through the output-stage r_o: every calibration lever is a binary rail↔370
switch (the over-flattened r_o is *required* for reachability). Nothing
installed (15/16); routes to T3. Memory `[[v658-ekv-core-breaks-opamp-rail]]`.

---

## V6.5.7 — panel-review correction of the V6.5.6 opamp verdict (2026-06-27)

5-agent adversarial review found "no high-gain zero exists / only-T3"
over-strong (the probe measured reachability, not existence). The cheap
vout-prioritized existence retrain was then RUN & KILLED: the vout F_rel floor
compatible with preservation is ~20× above what a high-gain zero needs — the
soft-wall is near-hard for the KCL-loss family. fetlim also dead. Memory
`[[v657-vout-existence-retrain-kill]]`.

---

## V6.5.6 — 3-operator Phase-0 routing + T1 KCL-residual lever (2026-06-26)

**Durable organizing frame — the 3-operator taxonomy:** the solver reads the
one NN surface through three operators, each owning a different gap:
id-values→KCL→NR fixed point (opamp gain, ring period); autograd dQ/dV→pole
(f3db); off-diagonal cgd→RHP zero (HF phase). Charge-head retrains are
DC-safe; id-surface retrains are not; the recurring ledger failure is applying
the wrong fix-class. Phase-0 diagnostics: existence (not conditioning) is the
tsmc7 gap; f3db is OP-drift-owned (caps already match OSDI); the fixed-point
LOCATION is a pure function of `id` VALUES. T1 KCL-residual loss solved
existence but produced an unstable fixed point, and preservation is binding.
⚠ Its "only-T3" conclusion corrected by V6.5.7. Memory
`[[v656-t1-existence-to-contraction]]`, `[[nn-accuracy-3operator-taxonomy]]`.

---

## V6.5.5 — diagnostic-routed corridor retrain → 15/16 (2026-06-24/25)

tsmc5 ring = NMOS-conduction-owned → lifted 3/4→4/4 via `large` + ring
corridor + seed7 (capacity was the bind). tsmc7 opamp = value-surface-owned —
seeding the sweep from the L72 ground-truth OP at every point STILL rails
(the high-gain OP is unstable on the NN surface; PTC/homotopy/OP-seed cannot
fix it). Net 14→15/16. Memory `[[v655-corridor-retrain-15of16]]`.

---

## V6.5.4 — fresh full retrain + best-config-per-tech → 14/16 (2026-06-23/24)

Full capacity matrix retrained from scratch on regenerated data (one clean
recipe); best config per tech → 14/16, clean. **Native-L72 control (decisive
diagnostic, `diag_l72_complex_control.py`):** the exact gate circuits through
PyCircuitSim's own solver with the OSDI model match NGSPICE at ring 0.00 % /
opamp ≤0.10 % ⇒ the remaining gaps are genuinely NN-value-surface-owned, not
solver/harness. Memory `[[v653-l72-control-ring-opamp-model-owned]]`.

---

## V6.5.3 — ★ the switchcap gap was a HARNESS CLOCK BUG (2026-06-23)

**Overturns V6.5.2.** The tsmc5 switchcap "11.84 % over-charge" chased across
the entire V6.4.x–V6.5.2 campaign was two harness bugs: (1)
`render_directnet_netlist` rescaled supply rails but MISSED the
space-delimited PULSE clock rail — the NN clock over-drove tsmc5 pass gates to
0.80 V vs NGSPICE's 0.65 V → 11.84 % FAIL became 1.56 % PASS, switchcap 4/4;
(2) the "14.65 % L72 floor" was a control op with no `uic` pinning. **LESSON
(load-bearing): when an NN gate fails vs NGSPICE, FIRST diff the rendered NN
netlist against the NGSPICE deck token-by-token BEFORE blaming the model or
solver.** `uic` made first-class in the product path. Memory
`[[v652-switchcap-is-harness-clock-bug]]`.

---

## V6.5.2 — charge-derivative levers + the (refuted) switchcap-is-SOLVER-owned finding (2026-06-22)

> SUPERSEDED by V6.5.3 — the conclusion was two harness bugs.

Both switchcap levers KILLED (correctly — there was no model gap):
charge-Sobolev left switchcap unmoved and did NOT move f3db (⇒ OP-drift
owned); TG-corridor fixed PMOS cdd 62 %→5 % yet the charge didn't move. Valid
reference: NN autograd caps match OSDI ~0.3–2.5 %; per-channel sign map
`+cgg,−cgd,−cdg,+cdd`.

---

## V6.5.1 — XL capacity tier + µA-band loss lever (KILLED) (2026-06-22)

**XL (2.13 M) = the over-fit boundary:** pass-rate 6→9→12→9/16 (S→M→L→XL); XL
fits the device surface ~10× tighter yet loses every value-surface-fragile
gate `large` won. µA-band `SubthresholdIdLoss` retune KILL (moved switchcap
<0.2 %). Also fixed the `xargs -L1` trailing-blank silent job-collapse bug.

---

## V6.5 — AC small-signal accuracy of the NN models (2026-06-22)

First NN AC gate vs NGSPICE (24 ckpts + opamp): AC **gain** excellent
everywhere (autograd gm/gds accurate); cap-driven pole good but tech-variable;
the **Cgd-feedforward RHP-zero HF phase is NOT reproduced**; opamp AC inherits
the DC value-surface fragility, but where the OP lands well GBW is 0.97× —
dynamics right, DC-gain level the miss. No retrain warranted. Harness
`complex_ac.py`, `verify_nn_ac.py`, `verify_complex_opamp_ac.py`.

---

## AC analysis — small-signal frequency-domain (2026-06-21)

Brought `.ac` from dead-on-arrival to NGSPICE-validated: `ACSolver` solves
complex `Y = G + jωC` about the DC OP; added the missing MOSFET
**transcapacitance stamp** (source-referenced 2-port embedded in the nodal
3×3, charge-conserving). `verify_ac.py` 2/2 (RC 0.0000 %; CS-amp gain err
5.4e-6 dB). Gotcha: ngspice `wrdata vp()` emits radians.

---

## V6.4.9 — DirectNet S/M/L capacity benchmark (2026-06-21)

Pass-rate rises 6→9→12/16 (S→M→L) but device accuracy is excellent at EVERY
size (not the bind). The opamp is the value-surface-fragile gate (recovers
only at `large`, only tsmc5/12); switchcap needs capacity; SRAM 4/4
everywhere. More capacity does NOT close recipe-sensitive gaps.

---

## V6.4.8+ — parametric sweep harness + TSMC7 broad retrain (KILL) (2026-06-20)

Built the complex-circuit parametric sweep harness (baseline-gated,
sha256-pinned). **TSMC7 broad retrain = KILL:** breadth fits the value surface
but COLLAPSES the offset-dominated opamp (gain→0); reverted. The opamp holds
gain under load perturbations but collapses under almost ANY OP change.

---

## V6.4.8 — value-surface accuracy campaign; 14 → 15/16 conditional (2026-06-17→20)

**Methodology locked: all gates run CPU** (the fragile opamp lands a different
NR basin on CUDA). S0 floor-k KILL (gain non-monotone in the floor coeff — it
hops NR basins; gds cancels at the fixed point). S1 capacity KILL (larger net
fits better yet collapses the opamp). **S2 continuation-first DC sweep KEEP
(the sole win, load-bearing):** warm-started NN points with source-stepping
off, gated on `has_nn` so BSIM-CMG is byte-identical; tsmc7 opamp 10.78 %
FAIL → 8.63 % PASS — the win is path-preservation. S3 EKV backbone KILL.

---

## V6.4.7 — serialized accuracy campaign; SHIP at 14/16 + force_ic 8/8 (2026-06-10→16)

Strict serial S1–S19 chain from the V6.4.4 canonical 8/16. **Durable changes
still in code:** S2 — the NMOS source-frame fix (NN Rule 2; permanent canary
`verify_nn_lifted_source_dc.py`); S7 — reverse-Vds clamp relaxation (C¹ taper;
the wider corridor was KILLED: tsmc5 opamp veto). **Key findings:** S6
simulator EXONERATED (native-L72 ring control ratio 1.000); S9b regen-v2 data
+ two load-bearing data-gen fixes (the `NN_DC_SOLVE_TOL` floor causing the
zero-row artifact; an atomic-write fix for a modelcard-cache race); S10 —
MAJOR: **derivative fidelity is ANTI-correlated with the opamp** (the Jacobian
guides NR convergence but cancels at the fixed point)
`[[v647-s10-deriv-fidelity-vs-opamp]]`; S12 — trajectory-corridor KEEP
(11→14/16; ancestor of every later corridor recipe); S17c — the force_ic
0/8→8/8 "gap" was a HARNESS BUG (non-physical read-disturb both NN and ground
truth fail) — LESSON: run the native-L72 control before blaming the NN; S19 —
trust the `verify_complex` gate, not the scorer proxy.

---

## Condensed history (pre-V6.4.7)

> Full detail for these iterations lives in `git log` and `MEMORY.md`. Only the
> durable outcomes are retained here.

### V6.4.6 — diagnosis-first iteration (2026-06-01/02, no behavioral change)
Gated every GPU-spend behind a 0-GPU diagnostic; localised the RO error to the
**id VALUE surface** (not the derivative). Probe/measurement fixes only.

### V6.4.5 — Track A no-ship iteration (2026-05-29)
Ran all 5 planned phases; shipped nothing. Built the multi-circuit scorer
(durable infra); confirmed the RO/SRAM gaps were architectural, not tuning.

### V6.4.4 — DirectNet per-tech checkpoint mix (2026-05-28, inference-only)
First per-tech medium checkpoint mix (canonical 8/16). Restored the
load-bearing `_MonotoneVgResidual` + `--monotonic` code (on-disk checkpoints
carry `mono.*` keys).

### V6.1 – V6.3.2 — per-tech DirectNet establishment (2026-05-12 → 05-15)
- **V6.1**: per-tech DirectNet for TSMC5/7; universal `refac_*`/`v4_*`
  artifacts deleted.
- **V6.2**: terminal-current sign fix; dead-band closed. **V6.2.1**:
  TSMC12/16 extension.
- **V6.3 / V6.3.1**: inverter spike-removal sprint (dataset regen with
  inv-trip recenter + reverse-Vds corridor).
- **V6.3.2**: PyCMG L3 parametric DC/transient sweeps ported to DirectNet.

### Pre-V6.0 (v3/v4/v5, package refactors, early milestones)
The BSIMAR package refactors (2026-03/04), the v3 LOO cross-tech sprint, the
v4 tech-code migration, the analytical Vds-correction + rail-restoring fixes,
and the v5 inverter-transient phases are recorded in `git log` and
`MEMORY.md`. Legacy LEVEL=1 (Shichman-Hodges) was removed; LEVEL=72/73/74/75
are the supported models.
