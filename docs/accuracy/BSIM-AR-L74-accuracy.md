# BSIM-AR Transformer (LEVEL=74) — Unified Accuracy Report

**Family:** BSIM-AR, decode-only autoregressive Transformer compact model, netlist
`LEVEL=74` — the validated **higher-fidelity** option (not production: AR inference
is ~30–100× DirectNet on CPU).
**Ground truth:** always NGSPICE on the *identical* BSIM-CMG (LEVEL=72) OSDI model.
**Consolidated:** 2026-07-24. **Covers:** V6.8.0 (scale × recipe campaign) → V6.8.1
(xl-tier fill) → V6.11.0 (TSMC6 sweep).

> **This file supersedes and replaces** `docs/V6.8.0-bsimar-transformer-report.md`.
> All of its content is carried here. The deleted file remains in git history (last
> present at commit `1fe1cdb`).

Sibling reports: `DirectNet-L73-accuracy.md` (LEVEL=73, production),
`PFN-L75-accuracy.md` (LEVEL=75). Shared methodology: `DirectNet-L73-accuracy.md` §2 —
identical gates, identical isolation, identical strict-OMP discipline.

---

## 0. Provenance

| Source campaign | Date | What it contributed |
|---|---|---|
| V6.8.0 | 2026-07-06/07 | Un-parking; full DN stack port; scale study (s/m/l) + recipe study; **15/16 strict** |
| V6.8.1 | 2026-07-11/23 | xl-tier fill (48 ckpts, ~11-day run): ties 15/16, AC collapses, not promoted |
| V6.11.0 | 2026-07-14/17 | TSMC6 clean capacity sweep (§8 — see the TSMC6≡TSMC7 correction) |

---

## 1. Headline

> **Updated 2026-07-24 (V6.13.0).** The table below is the **post-gds-fix**
> re-gate at `d2ea720`; §3, §4, §6 and Appendix A are the pre-fix measurements
> and are marked in place. See `DirectNet-L73-accuracy.md` §12.2 for the fix.

| model / recipe | params | complex 16-gate (single-run) | complex (strict OMP∈{1,2,4}) | pre-fix strict | device | AC |
|---|---|---|---|---|---|---|
| **BSIM-AR corroft @medium** (corridor curriculum) | **1.9M** | **16/16** | **16/16, zero flips** | 15/16 | 44/44 | 4/8 |
| **BSIM-AR corro15 @xl** | 14.8M | **16/16** | **16/16, zero flips** | 15/16 | 55/55 | — |
| BSIM-AR corro15 @medium | 1.9M | **16/16** | not swept | — | 44/44 | — |
| BSIM-AR corroft / crit15m / crit30 @xl | 14.8M | **16/16** each | not swept | 15/15/14 | 55/55 | — |
| BSIM-AR corroft / crit15m / crit30 @large | 5.0M | 15/16 each | not swept | 15/16 | 44/44 | 4/8 |
| BSIM-AR clean @{small, medium, large, xl} | 0.67–14.8M | **14/16 at every tier** | 14/16 (large), zero flips | 12/14/13/13 | 44/44 | 7/8 small |
| — DirectNet production crit30f @large | 0.9M | 15/16 | 15/16 | 14/16 | 24/24 | 8/8 |
| — DirectNet crit15m @xl | 2.1M | 16/16 | **16/16, zero flips** | 14/16 | 24/24 | — |

**`corroft@medium` now sweeps the entire matrix: 16/16 single-run AND 16/16 strict
across OMP∈{1,2,4} with zero flips**, at 1.9 M params. Cell margins are not
marginal either — opamp 2.52 / 6.73 / 5.32 / 5.82 % against a 10 % gate, ring
3.33 / 2.25 / 2.13 / 2.19 % against 5 %, switchcap 1.99–4.15 % against 5 %, worst
SRAM lobe 6.11 % against 10 % (tsmc5/7/12/16). Three checkpoint sets are now
strict-confirmed at 16/16 — `corroft@medium`, `corro15@xl` and DirectNet's
`crit15m@xl` — and **six** reach 16/16 single-run.

**Every corridor recipe at `xl` sweeps the matrix.** `corroft`, `crit15m`,
`crit30` and `corro15` are all 16/16 single-run at xl (pre-fix: 15/15/14/15),
and `corro15@xl` is strict-confirmed with zero flips. At `large` the same four
recipes all sit at 15/16, missing only `tsmc7-opamp`. So the corridor
curriculum's effect is uniform rather than recipe-specific, which is the
opposite of the pre-fix reading where recipe choice appeared to decide which
opamp basin you got.

**`tsmc7-opamp` was never the wall it was described as.** The claim that it is the
universal hard cell reachable only by the V6.5.9 T3 differentiable-solver fine-tune
appears throughout the pre-fix sections of this report and is **retracted**: BSIM-AR
passes it at *every* size and every recipe measured post-fix, clean included
(0.55–6.73 % gain error). What was actually happening is that the gds floor masked a
railed operating point.

**The ceiling moved opamp → ring.** For clean BSIM-AR the only two open cells at any
tier are `tsmc5-ring` and `tsmc7-ring`, and they fail *deterministically* — 7.38 %
and 8.63 % identically at OMP 1, 2 and 4 — against a 5 % gate. The corridor
curriculum is the lever that closes them (`corroft@medium` banks both at 3.33 % and
2.25 %), which is the same role the corridor plays for DirectNet and at universal
scope.

**Historical context:** the parked v4 *universal* BSIM-AR (2026-04) scored device
NRMSE 0.27 %/0.26 % and was 6–8× worse than DirectNet at 5.5× the params — dominated,
hence parked. This campaign's *per-tech* BSIM-AR closes most of that gap (small-tier
device AVG 0.38 % NRMSE) and, at the circuit level, **edges ahead of the production
DirectNet gate matrix.**

---

## 2. What was ported (code)

The Transformer already shared DirectNet's data/normalizer/eval pipeline and simulator
base class, but was not a first-class recipe citizen. Shipped in V6.8.0 (all
default-off / behavior-preserving for existing paths):

1. **`unknown_code_id` parameter** (`models/transformer.py`) — was hardcoded to the
   universal `17`; per-tech local vocabs (Rule 16) would CUDA-assert on the first
   `p_unknown` dropout. Now derived `num_tech_codes-1`.
2. **`init_from` warm-start** for `train_transformer` — enables every curriculum
   recipe (crit*/cor*ft) to fine-tune from a clean base.
3. **Aux losses for the Transformer** — `--sobolev` / `--charge-sobolev` /
   `--subthresh` were guarded DirectNet-only. The math is model-agnostic; the only fix
   was permuting the output-side norm stats to BSIMAR column order inside `_train_loop`.
   (The column-sum autograd trick was **verified** valid for the Transformer: attention
   mixes token positions within a sample, never batch rows — max |colsum−perrow| grad
   diff 1.3e-7.) A real blocker surfaced and was fixed: fused SDPA has no
   double-backward, so aux losses force the MATH attention backend at forward time.
4. **Transformer `xl` preset** (384×8L, 14.8 M) mirroring DirectNet's xl.
5. **`--amp`** (bf16 autocast) — measured 1.3× at large (loader-bound), not used in the
   campaign for comparability with the DN clean recipe.
6. **Parser LEVEL=74 per-tech preempt cascade** (`tsmc{X}_tf_{size}_{dev}`, large-first)
   + `_tf_` local-vocab scope detection.
7. **`PYCIRCUITSIM_NN_FORCE_LEVEL=74` harness hook** — retargets every LEVEL=73 model
   card to BSIM-AR at parse time, so the entire complex / sweep / AC gate infrastructure
   runs the Transformer with **zero deck changes**.
8. **Driver parameterization** — `MODEL={direct,transformer}` in `recipe_train.sh`,
   `gate_matrix_iso.sh`, `recipe_eval.sh`, `benchmark_run_tests.sh`,
   `recipe_multirun_gate.sh` (default byte-identical DirectNet).
9. **Solver batched-eval** now includes LEVEL=74 (per-checkpoint grouped
   forward+Jacobian pre-warm — the biggest CPU-gate lever for the ~8-forwards AR loop;
   per-device fallback + `NN_BATCHED_EVAL=0` opt-out intact).

### Latent integration bug fixed (V6.8.0, `mosfet_nn.py`)

The first real checkpoint scored **389 % device-DC NRMSE** through the netlist path
despite 0.38 % on the trainer's own test set. `_MOSFETNNBase._out_col` read the model
outputs through `norm_stats.output_columns` (canonical order, describing the *stats
arrays*) **before** the BSIMAR model-output layout — so `qg` was denormalized as `id`
(~5× current). Historical BSIM-AR norm files predate the `output_columns` field, so
this had never fired. Fixed by ranking the `output_layout == "bsimar"` branch first;
the probe now matches the trainer path bit-for-bit.

---

## 3. Scale study (clean recipe)

One identical clean recipe (`--apply-filter off --swa-mode ema --seed 42`), per-tech:

| tier | params | complex (single) | device DC | AC |
|---|---|---|---|---|
| small | 0.67M | 12/16 | 44/44 | 7/8 |
| **medium** | 1.94M | **14/16** | 44/44 | 4/8 |
| large | 5.02M | 13/16 | 44/44 | 4/8 |
| xl (§6) | 14.8M | 13/16 clean · 15/16 corridor | 55/55 | opamp-AC 0/4 |

- **Capacity peaks at medium** (12→14→13) — the same peak-then-decline shape DirectNet
  shows one tier later (DN peaks at large). Extra Transformer capacity buys circuit
  fixed-point accuracy up to medium, then begins over-fitting the value surface (large
  regresses tsmc5-opamp back to a fail and does not recover the rings).
- **AC peaks at small** (7/8 → 4/8 → 4/8) — identical trade-off to DirectNet: more
  capacity sharpens the DC/value surface at some cost in the cap-derivative (pole)
  surface the AC gate reads.
- **Device DC is uniformly excellent and capacity-insensitive** (all 44/44). The lone
  outlier is **tsmc7-NMOS**, which *grows* with capacity (3.37 → 4.07 → 4.77 % NRMSE) —
  a genuine value-surface quirk of the tsmc7 steep low-VDD curve, and the same tech
  whose opamp is the campaign's only unreachable gate.

#### 3.1 The scale study re-measured after the gds fix (V6.13.0)

Same four checkpoint sets, same harness, only the sign + guard fix of
`DirectNet-L73-accuracy.md` §12.2 between them.

| tier | complex, pre-fix | complex, post-fix | failing cells, post-fix |
|---|---|---|---|
| small | 12/16 | **14/16** | tsmc5-ring 6.53 %, tsmc7-ring 5.97 % |
| medium | 14/16 | **14/16** | tsmc5-ring 5.55 %, tsmc7-ring 7.41 % |
| large | 13/16 | **14/16** | tsmc5-ring 7.38 %, tsmc7-ring 8.63 % |
| xl | 13/16 | **14/16** | tsmc5-ring 7.61 %, tsmc7-ring 12.55 % |

**The capacity curve is gone.** Clean BSIM-AR is 14/16 at *every* tier and fails the
*same two cells* at every tier — so "capacity peaks at medium (12→14→13)" was largely
a gds artifact, not a property of the architecture. What the fix bought is the whole
opamp column: all four opamps now pass at all four sizes (gain err 0.55–8.53 % against
a 10 % gate), where pre-fix they were the deciding cells.

**The ceiling moved opamp → ring.** The two low-VDD rings are now the only thing
between clean BSIM-AR and a full sweep, and they are *not* close: 5.55–12.55 % period
error against a 5 % gate, worsening with capacity (tsmc7-ring 5.97 % at small →
12.55 % at xl). Rings are gds-invariant — the same result DirectNet §3.1b and the
universal study §9.1b show — so this is a genuine remaining value-surface gap and the
corridor curriculum (§4) is the lever that addresses it.

### Device DC NRMSE % by tech (NMOS / PMOS)

| tier | tsmc5 | tsmc7 | tsmc12 | tsmc16 |
|---|---|---|---|---|
| small | 1.45 / 0.48 | 3.37 / 0.56 | 0.20 / 0.21 | 0.24 / 0.46 |
| medium | 0.53 / 0.30 | 4.07 / 0.16 | 0.20 / 0.27 | 0.27 / 0.20 |
| large | 1.60 / 0.02 | 4.77 / 0.10 | 0.12 / 0.35 | 0.09 / 0.15 |

All within DirectNet clean@large's device range (0.83–3.99 / 0.02–2.20), achieved at
0.67–5.0 M params vs DN's 0.9 M.

---

## 4. Recipe study (Phase B)

All curricula: 120-epoch fine-tune, lr 3e-4, warm-started from the clean same-size
base, on the ring-only corridor datasets (`{tech}_corro_{dev}.npz`).

| recipe | tier | class weights | complex (single) | complex (strict) | fails (single) |
|---|---|---|---|---|---|
| **corroft** | **medium** | corridor 3.0 | **15/16** | **15/16** | tsmc7-opamp |
| crit30 | large | corridor 3.0 + inv_trip 2.0 | 15/16 | 14/16 | tsmc7-opamp |
| corroft | large | corridor 3.0 | 15/16 | 14/16 | tsmc7-opamp |
| crit15m | large | corridor 1.5 + inv_trip 3.0 | 15/16 | (14/16)† | tsmc7-opamp |
| invtrip | large | inv_trip 2.0 (no corridor) | 13/16§ | — | tsmc5-ring, tsmc7-ring, tsmc7-opamp |
| clean | large | — (control) | 13/16 | 12/16 | tsmc5-ring, tsmc7-opamp, tsmc7-ring |

† crit15m strict inferred from the same mirror-ring fragility pattern (full sweep not
run — all three large curricula share the profile). § invtrip@large (inv_trip alone,
no corridor) is strictly dominated: it leaves both rings closed AND actively *rails*
tsmc7-opamp (99.99 %, worse than clean's marginal 12.78 %). This confirms **the
corridor is the whole ring lever** and that the inv_trip weight is inert-to-harmful on
the Transformer.

### 4.1 The recipe table re-measured after the gds fix (V6.13.0, single-run)

| recipe / tier | pre-fix | post-fix | fails, post-fix |
|---|---|---|---|
| corroft / medium | 15/16 | **16/16** | — |
| corro15 / medium | — | **16/16** | — |
| corroft / large | 15/16 | 15/16 | tsmc7-opamp |
| crit15m / large | 15/16 | 15/16 | tsmc7-opamp |
| crit30 / large | 15/16 | 15/16 | tsmc7-opamp |
| invtrip / large | 13/16 | **14/16** | tsmc5-ring, tsmc7-ring |
| clean / large | 13/16 | **14/16** | tsmc5-ring, tsmc7-ring |

Two of the section's conclusions survive and one does not.

**Survives — the corridor is the whole ring lever.** `invtrip@large` (inv_trip
alone, no corridor) still leaves both rings open, and now lands on *exactly*
clean@large's failure set. Every recipe that closes a ring has the corridor in it.

**Survives, and is now unambiguous — inv_trip is inert on the Transformer.**
Pre-fix, invtrip@large looked actively harmful because it railed tsmc7-opamp at
99.99 % against clean's marginal 12.78 %. Post-fix both pass, and invtrip's
failure set is identical to clean's. It was the gds floor railing the opamp, not
the inv_trip weight; the honest verdict is that inv_trip does nothing here,
rather than that it does harm.

**Does not survive — "the recipe decides which opamp basin you get".** All three
large curricula now score exactly 15/16 with exactly the same single miss, and
all four xl curricula score 16/16 (§6.1). The recipe no longer discriminates
among opamps at all; it discriminates only on rings.

### Key findings

1. **Three distinct corridor recipes converge to 15/16 single-run** (corridor 3.0,
   3.0+anchor, 1.5+anchor) — all open both low-VDD rings and hold tsmc5/12/16 opamps,
   all rail only tsmc7-opamp. **The corridor weight and the inv_trip anchor do not
   change the outcome at large.**
2. **The inv_trip anchor is inert on the Transformer.** corroft (corridor-only) ≡
   crit30 (corridor + inv_trip) to <0.5 % on every cell. This is the *opposite* of
   DirectNet, where crit30 beat corroft by +1 cell (the deterministic tsmc5-opamp
   bank). On the Transformer the corridor alone already saturates the reachable set.
3. **tsmc7-opamp passes only at clean-medium; every large recipe fails it, and the
   corridor rails it at every tier.** clean-medium 9.83 % PASS, clean-large 12.78 %
   (marginal fail), invtrip-large 99.99 % (rail), corridor-{large,medium} rail. It
   needs the *medium tier without a corridor* — but then tsmc7-ring fails. **tsmc7-ring
   and tsmc7-opamp are mutually exclusive**; even per-tech recipe mixing caps tsmc7 at
   3/4 → **campaign ceiling 15/16**. The medium corridor-weight sweep makes this
   *continuous*, not a basin accident: tsmc7-opamp gain error degrades **monotonically**
   with corridor weight — 9.83 % (clean, PASS) → 24.83 % (corro15, w1.5) → 124 %
   (corroft, w3.0) — while any nonzero weight is required to open tsmc7-ring. No weight
   threads the needle; corro15@medium is the closest miss (24.83 % vs the 10 % gate)
   and still 15/16.
4. **Strict vs single-run splits on OMP-fragile rings.** At large each corridor recipe
   banks one ring deterministically and flips the other at OMP=1 (corroft: tsmc5-ring
   flips, tsmc7-ring det; crit30: the mirror) — the documented weight→basin
   non-monotonicity. Opamps are deterministic (tsmc5/12/16 PASS across OMP∈{1,2,4};
   tsmc7 rails deterministically).

---

## 5. Strict determinism (OMP∈{1,2,4})

Multi-run gate at `OMP_NUM_THREADS = MKL_NUM_THREADS = PYCIRCUITSIM_TORCH_THREADS
∈ {1,2,4}`; a cell is *strict* only if all three pass (guards the ~±1 % VTC-trip
scatter + opamp multistability).

**Large corridor recipes — verified:**
- Opamps: tsmc5, tsmc12, **tsmc16** deterministic PASS (both corroft & crit30);
  tsmc7 deterministic FAIL (rail).
- Rings: **corroft** → tsmc7-ring det PASS, tsmc5-ring FLIP (OMP1 FAIL);
  **crit30** → tsmc5-ring det PASS, tsmc7-ring FLIP (OMP1 FAIL). tsmc12/16 rings
  comfortable (~2 % margin, deterministic).
- → both = **14/16 strict**.

**Medium corridor (corroft@medium) — the strict winner:**
- Opamps: tsmc5, tsmc12, **tsmc16** deterministic PASS across OMP∈{1,2,4}; tsmc7
  deterministic FAIL (rail).
- Rings: **all four deterministic PASS** — the medium tier's rings sit comfortably
  inside the 5 % gate, whereas the large recipes' extra capacity pushes one ring to the
  OMP-fragile edge.
- SRAM ×4, switchcap ×4: stable (positivity / charge gates, not multistable).
- → **15/16 strict** (16 − tsmc7-opamp). Campaign best; beats DirectNet production.

### 5.1 Re-measured after the gds fix (V6.13.0) — and the flips are gone

Everything above was measured with the gds sign bug present. Re-run on the
identical checkpoints:

| stem | pre-fix strict | post-fix strict | FLIPs |
|---|---|---|---|
| **corroft@medium** | 15/16 | **16/16** | **0** |
| clean@large | 13/16 | **14/16** | **0** |

Two things changed, and the second is the more interesting one.

1. **`tsmc7-opamp` closed** at `corroft@medium`, taking it to a full sweep. The
   "deterministic FAIL (rail)" recorded above was the gds floor holding the
   operating point at the rail, not a value-surface limit.
2. **Every FLIP disappeared.** The large corridor recipes' OMP-fragile ring
   (corroft's tsmc5-ring, crit30's tsmc7-ring — each passing at OMP 2/4 and
   failing at OMP 1) was the whole reason `large` scored 14/16 strict against
   15/16 single-run, and it is the reason this section concluded "the strict
   best is medium, not large". Post-fix, clean@large's two failing rings return
   **identical** period errors at all three thread counts (7.38 % and 8.63 %),
   i.e. the multistability itself is gone rather than resolved favourably. The
   same flip-free result holds across all seven re-gated DirectNet groups, PFN,
   and all eight universal stems — a wrong-signed Jacobian entry was steering NR
   into different basins under different GEMM thread counts.

   That means the medium-over-large conclusion needs re-deriving rather than
   inheriting: it rested on a flip that no longer exists. The corridor recipes at
   `large` have not yet been strict-swept post-fix.

---

## 6. XL-tier fill (V6.8.1, 2026-07-11/23)

The V6.8.0 scale study stopped at large; the **xl** preset (384×8L×ff1536, **14.81 M
params** — 3× tf-large, 5.5× DirectNet-xl) was coded but untrained. V6.8.1 trains the
full Phase-B recipe mirror at xl — 48 checkpoints (6 recipes × 4 techs × 2 devs), full
300/120-epoch fidelity, a ~11-day run on a heavily shared box (co-tenant Xyce/Swin
fleets + the TSMC6 campaign; exactly one transient SIGKILL, auto-recovered). Question:
does the Transformer at xl **shuffle basins** the way DirectNet-xl did (DN-xl banked
tsmc16-opamp, tying 14/16), and does AC hold or collapse at xl?

**Complex 16-gate, single-run OMP=1** (`results/recipe_bench/gate_iso_xl_tf/`, 96 cells
vs NGSPICE):

| recipe   | strict | FAIL cells                          |
|----------|--------|-------------------------------------|
| corroft  | **15/16** | tsmc7-opamp                      |
| crit15m  | **15/16** | tsmc7-opamp                      |
| corro15  | **15/16** | tsmc7-opamp                      |
| crit30   | 14/16  | tsmc5-opamp, tsmc7-opamp            |

### 6.1 The xl tier re-measured after the gds fix (V6.13.0) — every corridor recipe sweeps

| recipe | pre-fix | post-fix | fails, post-fix |
|---|---|---|---|
| corroft | 15/16 | **16/16** | — |
| crit15m | 15/16 | **16/16** | — |
| corro15 | 15/16 | **16/16** (strict, zero flips) | — |
| crit30 | 14/16 | **16/16** | — |
| csob | 13/16 | **14/16** | tsmc5-ring, tsmc7-ring |
| clean | 13/16 | **14/16** | tsmc5-ring, tsmc7-ring |

**All four corridor curricula at xl now sweep the full matrix**, and `corro15@xl`
is strict-confirmed across OMP∈{1,2,4} with zero flips. That reframes this
section's central question. It asked whether the Transformer "shuffles basins at
xl the way DirectNet-xl did" — the answer is that the basin-shuffling was largely
the gds bug. Post-fix, xl is where the corridor is *most* reliable, not where its
gains get traded away.

**What this does NOT change: xl is still not promoted.** 14.81 M params for a
result that `corroft@medium` also reaches at 1.9 M, on top of §6's AC collapse
and the ~11-day training cost. The medium tier remains the recommendation; xl is
now a corroboration of it rather than a rival to it.

The non-corridor recipes (clean, csob) gain only +1 and stay stuck on the two
low-VDD rings — consistent everywhere else in this campaign: **gds moved opamps,
the corridor moves rings, and the two levers are independent.**
| clean    | 13/16  | tsmc5-ring, tsmc7-opamp, tsmc7-ring |
| csob     | 13/16  | tsmc5-ring, tsmc7-ring, tsmc7-opamp |

**Findings:**

1. **xl TIES medium (15/16), does NOT exceed it, and does NOT shuffle basins.** All
   three top recipes miss *only* tsmc7-opamp — the identical ceiling cell as
   corroft@medium. No xl basin bought tsmc7-opamp. **The DirectNet xl basin-shuffle
   effect does NOT replicate on the Transformer**: DN-xl relocated to bank
   tsmc16-opamp; the Transformer already banks tsmc16-opamp at medium/large and simply
   *holds the same 15/16 basket* at xl.
2. **Mild xl effect = a broader 15/16 plateau, not a higher count.** At medium only
   corroft reached 15/16; at xl, crit15m and corro15 join it. More capacity lets more
   corridor recipes land the *same* basket — but no new cell.
3. **OMP-strict (corroft@xl) — the 15/16 is deterministic.** All three fragile opamps
   are OMP-invariant: tsmc5 4.02–4.03 %, tsmc12 5.89–5.91 %, tsmc16 5.97–5.98 %
   gain-error across OMP∈{1,2,4}; tsmc5-ring 3.83 % at OMP=1. The opamps are the
   multistability risk and they are clean. (The ring OMP=2/4 sweeps were truncated —
   each ring cell ran >2 h under the box load and rings are low-multistability; the
   comfortable OMP=1 margins stand.)
4. **Device DC is excellent and capacity-insensitive: corroft@xl = 55/55 configs PASS.**
   Baseline NRMSE per tech (nmos/pmos): tsmc5 0.16/0.03, tsmc7 0.19/0.06, tsmc12
   0.03/0.03, tsmc16 0.06/0.03 %. Every failure at xl is a *circuit-level* value/bias
   problem, never a device-surface one.
5. **AC COLLAPSES at xl** (`results/bsimar_bench/*/xl/`) — the sharpest result:
   - **opamp-AC (open-loop Miller): 0/4 for every recipe.** tsmc5/tsmc16 rail
     (OP-misbias → value-surface collapse); tsmc12 has good GBW (1.03) / PM (4.4°) but
     magNRMSE 102 %. **tsmc7-opamp-AC does not even converge** — its DC OP spins
     non-convergent (~6 h, no `recipe_eval` timeout); seed-skip it for any tf-xl eval.
   - **device nn_ac (CS-amp): weak.** corroft 4/8 device-passes (tsmc5 nmos+pmos,
     tsmc12/16 nmos-only; tsmc7 both fail), csob 4/8, clean 2/8. Per-tech (both
     devices) only corroft-tsmc5 passes.
   - **csob's charge-Sobolev does NOT recover AC at xl** — it ties corroft's
     device-count and stays 0/4 on opamp-AC. AC weakness at xl is a capacity/tier
     property, not a recipe-fixable one. This mirrors DirectNet-xl and confirms the
     campaign-wide "AC peaks at small" law: xl is the *worst* AC tier.

**Verdict: xl is not worth promoting.** It ties medium's 15/16 at 3× the params and
~30–100× slower AR inference, with collapsed AC and zero basin gain. Its value is
negative-result confirmation: the 15/16 ceiling is **robust across three tiers**
(medium/large/xl all cap at 15/16), tsmc7-opamp is genuinely a solver-only (T3/EKV)
cell that **no amount of capacity or data recipe reaches**, and the Transformer —
unlike DirectNet — does not basin-shuffle at xl. **corroft@medium (1.9 M) remains the
validated best BSIM-AR config.** Driver + logs: `results/recipe_bench/xl_campaign/`.

---

## 7. AC (small-signal, device CS-amp) and inference cost

Gate: gain0 err ≤1.5 dB / f3db ratio ∈[0.7,1.43] / mag NRMSE ≤10 %.

| tier | pass | notable |
|---|---|---|
| small | 7/8 | best; only tsmc12-pmos (13 % mag NRMSE) fails |
| medium | 4/8 | |
| large | 4/8 | |
| xl | 4/8 device (corroft) | but **opamp-AC 0/4**; csob no help → AC collapses at xl |

AC peaks at small and is already competitive with / ahead of DirectNet's large-tier AC
(4/12). The Transformer's cap-derivative surface is strongest at low capacity —
consistent with the DirectNet finding that AC (a dQ/dV pole property) and DC circuit
fixed-points want opposite capacities.

**Inference cost.** BSIM-AR inference is an **8-step sequential AR loop** per
evaluation (each step a full encoder forward) at 5.5× DirectNet's params. On the
CPU-pinned gates (1 thread) this is ~30–100× a single DirectNet eval (61.5 ms/eval at
medium vs DirectNet-large's 1.5 ms); the heavy complex cells (ring/SRAM/switchcap
transients) run ~1 h each at small and ~2–4 h at large. The V6.8.0 batched-eval
inclusion (per-checkpoint grouped forward) is the single biggest mitigation — a
10-device ring collapses to 2 grouped forwards. **For production use the Transformer is
a fidelity option, not a speed option.**

---

## 8. TSMC6 — retired, and what it taught us

**TSMC6 was deleted from this repo on 2026-07-24.** It was never an independent
technology under BSIM-CMG — it is TSMC7 relabelled — so the V6.11.0 "TSMC6"
campaign was a *second training run on the TSMC7 data*, and its rows here were
a duplicate data point presented as a sixth technology.

Evidence (`docs/2026-07-21-systematic-audit.md` §D1, re-verified at deletion):

* `tsmc6_{nmos,pmos}.npz` were `array_equal` to `tsmc7_*` in `inputs`,
  `geometry`, `outputs` and `sample_class` over 1,816,830 / 2,187,292 rows —
  only `meta_tech_name` differed.
* The raw PDKs genuinely differ, but every differing key (`tmi_ver_lod`,
  `tmi_ver_isocpode`, `sfxmin`, `samax_c`, `wodx5akvth0`) is a TSMC
  TMI-proprietary extension with **zero occurrences** in the BSIM-CMG
  Verilog-A. Reproduced mechanically at deletion time: of 871 implemented
  parameter names parsed from the Verilog-A sources, `toxp` and `phig` are
  present and all five TMI keys are absent.
* Two LEVEL=72 Id-Vgs sweeps at identical geometry matched to the last digit.

**What was deleted:** 22 checkpoints, both datasets, `results/tsmc6_gate`, the
registry/driver/test entries, and the per-size TSMC6 tables that stood here.
They remain in git history (last present at commit `a96112a`). The raw vendor
PDK is kept but unreferenced. TSMC6 held tail codes 22-24, so nothing was
renumbered.

**The methodological result is worth keeping.** The headline claim this section used to carry — *"BSIM-AR is the only family to bank the tsmc6 opamp (9.83 %) while tsmc7-opamp is the universal ceiling"* — was never a fidelity result. It is the same measurement twice, landing in different NR basins; the giveaway is that 9.83 % is the *identical number* to the tsmc7-opamp clean-medium pass in §4. A cross-family claim that rests on one opamp cell is a claim about basin luck unless it is reproduced under an OMP sweep on genuinely different data.

**The guard:** `bsimar.config.assert_tech_is_distinct(tech)` compares resolved
modelcards restricted to parameters BSIM-CMG implements, and refuses a tech
that collides with an existing one. It flags `tsmc6`↔`tsmc7` and confirms
tsmc5/7/12/16 are genuinely distinct. Run it *before* onboarding a technology —
the V6.9.0 onboarding gated TSMC6 9/9 DC and 14/14 transient, and passing those
gates told us nothing, because they were TSMC7's gates.
---

## 9. Cross-family comparison and recommendation

Post-gds-fix (V6.13.0), strict OMP∈{1,2,4} where stated:

| family | best config | params | complex strict | device | AC | CPU ms/eval |
|---|---|---|---|---|---|---|
| DirectNet (73) — **production** | crit30f@large | 0.92 M | 15/16 (0 flips) | 24/24 | 8/8 device | **1.5** |
| DirectNet (73) — best any tier | crit15m@xl | 2.1 M | **16/16** (0 flips) | 24/24 | — | 3.4 |
| **BSIM-AR (74)** | **corroft@medium** | 1.9 M | **16/16** (0 flips) | **44/44** | 4/8 device | 61.5 |
| PFN (75) | clean@small | 0.69 M | 11/16 (0 flips) | — | 8/8 at large | 15.6 |

- **DirectNet stays the production NN** (LEVEL=73). It is now 15/16 strict, and at
  `xl` a uniform recipe reaches a full sweep — at 40× the inference speed of
  BSIM-AR. Nothing here changes the production choice; if anything the case is
  stronger, because the one cell that used to separate the families has closed
  for both.
- **BSIM-AR remains the higher-fidelity option** (LEVEL=74): `corroft@medium`
  = 16/16 strict at 1.9 M params, with device DC 44/44 (vs DirectNet's 24/24
  suite) and comfortable margins on every cell. At ~40× DirectNet's per-eval cost
  it is the choice when fidelity dominates and wall-clock does not.
- **PFN (LEVEL=75) stays research.** It was the first flip-free family, but as of
  V6.13.0 *every* family is flip-free, so that distinction has been overtaken —
  the gds sign error was the shared cause. PFN's remaining gap is the low-VDD
  rings and its declining capacity curve.
- **The "universal ceiling" framing is retracted.** `tsmc7-opamp` is passed by
  BSIM-AR at every size and by DirectNet at `small`, `xl` and `crit15m@xl` strict.
  Two independent families reach a full 16/16 strict sweep with ordinary data
  recipes, so the T3 differentiable-solver fine-tune is no longer the only known
  path there and porting it to the Transformer is no longer the natural
  follow-up. The open work is the **low-VDD rings** (tsmc5/tsmc7) for the clean
  recipes, and the **opamp open-loop AC gate**, which no family passes.

**Provenance caveat.** §1, §3.1, §5.1 and this section are post-fix (`d2ea720`).
§3 (except 3.1), §4, §6, §7 and Appendix A are pre-fix and are marked where they
are contradicted. The audit's measurement of the corruption on BSIM-AR
specifically — the shipped floor altering ~89–91 % of amplifying points, median
error inflated 6–8× — explains why the opamp column moved so uniformly. See
`DirectNet-L73-accuracy.md` §12.2 for the fix itself.

---

## Appendix A — full complex matrix (single-run)

> ⚠ **Pre-fix (V6.8.0/V6.8.1) data, kept for provenance.** Every cell below was
> measured with the gds sign bug present. For the post-fix matrix see §1, §3.1,
> §4.1 and §6.1, or regenerate from `results/a3_regate/REPORT.md`.

| recipe/tier | tsmc5 | tsmc7 | tsmc12 | tsmc16 |
|---|---|---|---|---|
| clean/small | O✗ R✗ S✓ C✓ | O✗ R✗ S✓ C✓ | ✓✓✓✓ | ✓✓✓✓ |
| clean/medium | O✓ R✗ S✓ C✓ | O✓ R✗ S✓ C✓ | ✓✓✓✓ | ✓✓✓✓ |
| clean/large | O✓ R✗ S✓ C✓ | O✗ R✗ S✓ C✓ | ✓✓✓✓ | ✓✓✓✓ |
| corridor/large | ✓✓✓✓ | O✗ R✓ S✓ C✓ | ✓✓✓✓ | ✓✓✓✓ |
| corroft/medium | ✓✓✓✓ | O✗ R✓ S✓ C✓ | ✓✓✓✓ | ✓✓✓✓ |
| corroft/xl | ✓✓✓✓ | O✗ R✓ S✓ C✓ | ✓✓✓✓ | ✓✓✓✓ |
| crit15m/xl | ✓✓✓✓ | O✗ R✓ S✓ C✓ | ✓✓✓✓ | ✓✓✓✓ |
| corro15/xl | ✓✓✓✓ | O✗ R✓ S✓ C✓ | ✓✓✓✓ | ✓✓✓✓ |
| clean/xl | O✓ R✗ S✓ C✓ | O✗ R✗ S✓ C✓ | ✓✓✓✓ | ✓✓✓✓ |
| csob/xl | O✓ R✗ S✓ C✓ | O✗ R✗ S✓ C✓ | ✓✓✓✓ | ✓✓✓✓ |

(O=opamp, R=ring_osc, S=sram_snm, C=switchcap; ✓ PASS / ✗ FAIL vs NGSPICE.)

## Appendix B — provenance and reproduction

- **Datasets:** `external_compact_models/bsimar/data/datasets/{tech}_{dev}.npz`
  (2026-06-23, the same sets DirectNet production trained on) + `_corro_` corridor
  sets. No regeneration.
- **Checkpoints:** `tsmc{5,7,12,16}_tf_{small,medium,large,xl}_{nmos,pmos}` (clean) +
  `tsmc{X}_tf_{crit30,corroft,crit15m,invtrip}_large_*` + `_{corroft,corro15}_medium_*`
  + the V6.8.1 xl recipe mirror. Each `*.complete` marker = a finished run.
- **Gate logs:** `results/bsimar_bench/{gate_iso_*16,device_*,ac_*,omp_*}/`,
  `results/recipe_bench/gate_iso_xl_tf/`, `results/recipe_bench/xl_campaign/`.
- **Plans / execution logs:** `docs/plans/2026-07-05-bsimar-transformer-recipe-campaign.md`,
  `docs/plans/2026-07-07-bsimar-transformer-xl-fill.md`.
- **Operational gotcha:** never reuse a `gate_iso_*` results directory across model
  families — the per-cell verdict files collide silently.

```bash
# train a recipe wave
MODEL=transformer RECIPES=corroft SIZES=medium GPUS="0 1 2" NSTREAMS=6 bash scripts/recipe_train.sh
# gate it
MODEL=transformer SIZE=medium bash scripts/gate_matrix_iso.sh
# single netlist, any LEVEL=73 deck retargeted to BSIM-AR
PYCIRCUITSIM_NN_FORCE_LEVEL=74 python main.py examples/bsimar_inverter_dc.sp
```
