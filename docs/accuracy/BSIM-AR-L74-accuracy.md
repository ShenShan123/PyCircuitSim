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

| model / recipe | params | complex 16-gate (single-run) | complex (strict OMP∈{1,2,4}) | device | AC |
|---|---|---|---|---|---|
| **BSIM-AR corroft @medium** (corridor curriculum) | **1.9M** | **15/16** | **15/16** | 44/44 | 4/8 |
| BSIM-AR corridor @large (crit30/corroft/crit15m) | 5.0M | 15/16 | 14/16 | 44/44 | 4/8 |
| BSIM-AR clean @medium | 1.9M | 14/16 | 13/16 | 44/44 | 4/8 |
| BSIM-AR clean @small | 0.67M | 12/16 | — | 44/44 | 7/8 |
| BSIM-AR corridor @xl (corroft/crit15m/corro15) | 14.8M | 15/16 | 15/16 (corroft) | 55/55 | ~2/8 (opamp-AC 0/4) |
| — DirectNet production crit30f @large | 0.9M | 14/16 | 14/16 | 24/24 | — |
| — DirectNet clean @large | 0.9M | 13/16 | 12/16 | 24/24 | 4/12 |

**BSIM-AR `corroft@medium` reaches 15/16 both single-run AND strict — beating
DirectNet's production 14/16 strict outright**, at 1.9 M params. Its failure set is
strictly *better* than DN production: BSIM-AR banks **tsmc16-opamp** (which DN
production fails) plus both low-VDD rings — all deterministic across OMP∈{1,2,4} —
and misses only **tsmc7-opamp**. DN production misses *both* tsmc7- and tsmc16-opamp.

The large-tier corridor recipes reach the same 15/16 single-run but only 14/16 strict
(one OMP-fragile ring each); the **medium tier is where both rings sit deterministically
inside the gate** — so the strict best is medium, not large.

The one cell BSIM-AR cannot reach with any data recipe — **tsmc7-opamp** — is the
universal hard cell that DirectNet itself only ever passed via the V6.5.9 **T3
differentiable-solver fine-tune** (a solver-level method, not a data recipe).

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

| family | best config | params | complex strict | device | AC | CPU ms/eval |
|---|---|---|---|---|---|---|
| DirectNet (73) | crit30f@large | 0.92 M | 14/16 | 24/24 | 4/12 device | **1.5** |
| **BSIM-AR (74)** | **corroft@medium** | 1.9 M | **15/16** | **44/44** | 4/8 device | 61.5 |
| PFN (75) | clean@small | 0.69 M | 11/16 (zero OMP flips) | — | 5/8 device | 15.6 |

- **DirectNet stays the production NN** (LEVEL=73): 0.9 M params, ~30–100× faster
  inference, 14/16 strict. Nothing here changes that.
- **BSIM-AR is un-parked as a validated fidelity alternative** (LEVEL=74) and it
  **exceeds DN's production strict count**: `corroft@medium = 15/16 strict` at 1.9 M
  params banks tsmc16-opamp + both low-VDD rings deterministically — all cells DN
  production fails except the universal tsmc7-opamp. Device DC 44/44, AC competitive.
  If speed weren't a concern, corroft@medium would be the higher-fidelity NN.
- **tsmc7-opamp** remains the one universal unreachable gate for *both* models via data
  recipes; it is a solver-level (T3 / EKV) problem. Porting the T3 differentiable-solver
  fine-tune to the Transformer is the only known path to 16/16 and is the natural
  follow-up.

**Standing caveat — the gds sign bug.** Every number in this report was measured with
the inference-side `gds` sign error present (`docs/2026-07-21-systematic-audit.md` §A3;
not shipped as of 2026-07-24). The audit measured the corruption on all three NN
families (BSIM-AR: shipped-floor alters ~89–91 % of amplifying points, median error
inflated 6–8×) and the sign+guard fix moved the complex matrix 17/20 → 18/20 and the
`force_ic` latch probe 2/5 → 5/5 on DirectNet, with DC exactly invariant. Expect the
AC numbers in §7 — and the opamp rails — to move when it ships. See
`DirectNet-L73-accuracy.md` §12.2.

---

## Appendix A — full complex matrix (single-run)

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
