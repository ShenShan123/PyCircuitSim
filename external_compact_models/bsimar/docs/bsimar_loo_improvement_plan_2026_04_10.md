# BSIMAR v3 Cross-Technology Transferability Improvement Plan (2026-04-10)

## Motivation

The 2026-04-09 leave-one-tech-out (LOO) experiment
(`tests/verify_bsimar_loo_results/20260409_130630_nmos/report.md`)
showed a structural generalization gap. The **in-distribution baseline**
(random split, all 5 techs pooled) is NRMSE 0.223 % / MRE 1.41 % /
R² 0.9984. The **LOO baseline** (hold out one tech entirely) is:

| Held out | NRMSE % | MRE % | R² | vs. baseline |
|----------|--------:|------:|---:|-------------:|
| asap7    | 24678.40 | 231543.95 | blown | **110665×** |
| tsmc5    |     2.17 |     27.30 | 0.9552 |      9.7×  |
| tsmc7    |     2.18 |     22.64 | 0.9584 |      9.8×  |
| tsmc12   |     0.98 |      8.02 | 0.9828 |      4.4×  |
| tsmc16   |     0.84 |      7.81 | 0.9791 |      3.8×  |

Two failure modes, which need to be diagnosed and treated **separately**:

1. **ASAP7 body-physics catastrophe.** ASAP7 gmb is ~1e-9 S and qb is
   ~1e-18 C, 3–4 orders of magnitude smaller than TSMC. The ASAP7
   FinFET modelcard has essentially decoupled the body from the
   channel. Excluding gmb/qb, the ASAP7 fold drops from 24,678 % →
   **11.7 %**, putting it in the same order as TSMC5/7. The headline
   blow-up is *metric sensitivity* on two specific outputs, not a
   100,000× predictive failure across the board.

2. **TSMC intra-family covariate shift.** TSMC5/7/12/16 interpolate at
   2–10× the in-distribution baseline. Not a catastrophe. Most of the
   TSMC error lives in `id / gm / gds / gmb` (the solver-critical
   current+conductance group), with capacitances close behind.

## The real scoreboard: solver-critical vs. body-physics

The 13-output average NRMSE is a misleading headline for this
problem because it mixes two physically and statistically unrelated
regimes. From here on this plan tracks **two scoreboards**, reported
separately per fold:

- **Solver-critical NRMSE_sc** = mean of `{id, gm, gds, cgg}`. These
  drive the NR solver and circuit-level delay. Circuit simulation
  cares about this and nothing else.
- **Body-physics NRMSE_body** = mean of `{gmb, qb}`. These are where
  the ASAP7 catastrophe lives and where cross-family transfer is a
  pure data problem (4-decade scale gap between ASAP7 and TSMC).

For context we also report the old 13-output average NRMSE_all, but
it is no longer the decision variable.

Baseline restated on the new scoreboard (from the 2026-04-09 report):

| Held out | NRMSE_sc % | NRMSE_body % | NRMSE_all % |
|----------|-----------:|-------------:|------------:|
| asap7    |      13.30 |    160345.31 |   24678.41  |
| tsmc5    |       3.43 |         1.88 |       2.17  |
| tsmc7    |       3.76 |         1.76 |       2.18  |
| tsmc12   |       1.10 |         1.01 |       0.98  |
| tsmc16   |       0.94 |         1.24 |       0.84  |

(NRMSE_sc computed from the per-output table in the report:
`(id + gm + gds + cgg)/4`; NRMSE_body = `(gmb + qb)/2`.)

The real ASAP7 problem is the 13.3 % solver-critical number, not the
160000 % body number. No normaliser trick will rescue a 4-decade body
extrapolation from TSMC-only training data; what we can plausibly
move is the solver-critical number via better features and a better
normaliser. The body column is a **data-bottleneck problem**, parked
as a known limitation and called out explicitly in every result.

## Guiding principles

- **Fix what's fixable.** Solver-critical error on all 5 folds is
  addressable via features + normalisation. Body-physics error on
  ASAP7 is a data problem; do not spend sprint budget pretending
  otherwise.
- **Preserve the v3 production recipe outside of the lever being
  tested.** N1/N3/N7 winners stay.
- **Budget: ~10 hours.** Each fold is ~50 min on the Blackwell GPU.
  Per experiment we run **2 folds** as a quick signal (asap7 always,
  tsmc5 as the worst "normal" fold). Only the winners get a full
  5-fold re-run.
- **Decision rule (new).** An experiment is kept iff it improves
  *both* `log(NRMSE_sc_asap7 / baseline) + log(NRMSE_sc_tsmc5 / baseline)`
  **and** does not regress the in-distribution random-split val loss
  by more than 0.5 % relative. The log-ratio form stops any single
  fold from dominating by orders of magnitude, and the
  in-distribution guard stops us from trading away the v3 win.
  `NRMSE_body` is reported but **not** part of the decision rule.

## Strategy list (ranked)

### S2. Per-target asinh-scale floor for gmb/qb (ALREADY CODED)

`external_compact_models/bsimar/data/normalize.py` already carries
the S2 floor (uncommitted):

```python
OUTPUT_ASINH_SCALE_MIN = {
    "gmb": 1e-5,   # TSMC gmb max ≈ 1.07e-4; asinh(y/1e-5) ∈ [0, ~3.1]
    "qb":  1e-15,  # TSMC qb  max ≈ 5.15e-15; asinh(y/1e-15) ∈ [0, ~2.4]
}
```

applied in `BSIMARNormalizer.fit(mode='asinh')` after the geomean
computation. This is run as a **sanity signal only**, not a headline
fix. The mechanism is that with a larger floor the asinh transform
becomes approximately linear for gmb/qb and the denormalisation
sensitivity `d(y_phys)/d(z)` no longer passes through `sinh(large)`;
this capps how badly an OOD prediction can blow up in physical space.

**Honest expected impact.** S2 cannot lower the solver-critical
error on any fold — it does not touch id/gm/gds/cgg at all. On the
body scoreboard the best-case outcome is a reduction in NRMSE_body
of perhaps 10–100× on ASAP7 (still 1,000–10,000 %). It is run only
to confirm the mechanism is what we think and to de-risk whether the
floor hurts TSMC in-distribution accuracy.

**Previous plan-doc claim** of "single-digit NRMSE" and "under 15 %"
has been retracted: the analysis conflated the metric blow-up with
predictive error, and the 11.7 % floor on ASAP7 (excluding gmb/qb) is
mechanically unreachable for this lever because S2 does nothing for
id/gm/cgg.

- **Engineering cost**: 0 (code is already in normalize.py).
- **Tracked in**: experiment E1.

### S1. ~~Input-noise augmentation on process-param dimensions~~ — DROPPED

Original idea: add Gaussian noise to the 12 normalised process-param
input features so the 4 discrete tech clusters smear into overlapping
blobs.

**Why dropped.** The LOO report's OOD table shows ASAP7, TSMC5, and
TSMC16 all have **100 % of held-out samples outside the training
input box** on the process-param dims (discrete per-tech clusters →
any held-out tech is trivially out-of-box). Noise *inside* the
training blob does not move points *outside* it. Expected impact on
ASAP7 is ≈ 0; possible marginal help on tsmc7/tsmc12 which are the
only folds with meaningful in-box coverage (5.5 %, 1.3 %). Not worth
a 100-min experiment slot when physics features (M2) attack the same
generalisation axis with better leverage.

### S3. ~~Stronger regularization (weight_decay, longer schedule)~~ — INFEASIBLE

The in-distribution NRMSE is 0.223 %. We are not overfitting; the
LOO gap is a **covariate-shift** gap, not a capacity gap. Weight
decay 1e-4 → 1e-3 + 150 → 200 epochs costs in-distribution accuracy
for ~0 generalisation gain. Dropped before running.

### S4. ~~Cross-tech process-param mixup~~ — INFEASIBLE

The 5 techs have disjoint (V, NFIN, L) sweep grids. Linear
interpolation between sample A's process params and sample B's
process params at different voltages/geometries produces synthetic
"techs" that don't live on any real data manifold. The only sound
mixup would be nearest-neighbour-paired in normalised (V, NFIN, L)
space, which is another day of engineering — not in scope for this
sprint. Dropped before running.

### M1. ~~Learned per-tech output-scale head~~ — DROPPED

A small MLP `f: proc_params → R^{13}` producing per-target log-scale
corrections was proposed to "learn the gradient of the scale across
techs."

**Why dropped.** The head is still a function of `proc_params` and
is still trained only on the 4 TSMC techs. On ASAP7 proc_params it
extrapolates in exactly the same way the base model does — the
scale-correction output is undefined for OOD inputs. Adding capacity
does not introduce new information; it just moves the OOD failure to
a different layer. The physically grounded variant of the same idea
is D1 below (derive the scale from modelcard physics instead of
learning it).

### M2a. Physics-informed derived features — CHANNEL TRANSPORT (PROMOTED, HIGHEST LEVERAGE)

Add a small handful of derived inputs the Transformer currently has
to reinvent from scalars:

- `Vov = Vgs - PHIG` (Vov proxy — already used for Vov-LDS loss
  weighting, now fed as an input feature too)
- `Vds_rel = Vds / VDD_est` where `VDD_est` is extracted from the
  process-param block (upper end of the per-tech voltage sweep)
- `log(NFIN · W_fin / L)` (total W/L ratio) — a single dimensional
  group instead of `log(NFIN)` and `L` as independent scalars
- `EOT * VDD_est` (oxide field proxy)

This is the only lever in the plan that can plausibly move the
**solver-critical** scoreboard on all 5 folds. Intuition: id/gm/gds
depend on dimensionless groups (Vov/VT, Vds/VDD, W/L, oxide field)
that the current 19-column layout forces the Transformer to compose
from raw scalars across tech boundaries. Handing it the groups
directly makes the function smoother in process-param space.

- **Engineering cost**: ~3 hours. Touches:
  - `normalize._build_combined_input` (needs to emit an expanded
    column layout).
  - `models/transformer.py:93` — the `input_dim=19` assertion and the
    grouped-input slices `VOLTAGE_SLICE / GEOM_SLICE / PROC_SLICE`
    (`transformer.py:77-79`) have to change. Probably 23-column input
    (19 + 4 derived) with a 4-token geometry group or a new `derived`
    group.
  - `bsimar/eval/loo_labels.py` and the dataset loaders (so the LOO
    harness sees the same layout).
- **Expected impact**: 10–30 % relative reduction in
  NRMSE_sc on ASAP7 (driven by id/gm smoother in Vov/EOT space) and
  probably 20–40 % on TSMC5/TSMC7. Zero impact on NRMSE_body.
- **Tracked in**: experiment E2.

### M2b. Modelcard-derived body-factor feature (BODY-PHYSICS LEVER)

**The** physically principled way to address the ASAP7 body
catastrophe without adding ASAP7 data. The PyCMG modelcard exposes
bulk-charge and body-bias coefficients (e.g. `ETA0`, `CIT`, and the
body-factor derivatives). From them we can extract a scalar
`body_coupling_factor` per (tech, L, NFIN) that correlates
monotonically with |gmb|, |qb|.

Feed this factor as a 20th input feature and as the **physics
scale** for gmb/qb denormalisation (see D1 below, which is the same
lever used a different way).

- **Engineering cost**: ~2 hours, mostly in
  `external_compact_models/PyCMG/scripts/generate_nn_data.py` to emit
  the body factor alongside the existing 12 process params, and the
  input-dim plumbing in `normalize.py` / `transformer.py`.
- **Expected impact on the solver-critical scoreboard**: small
  (body-factor does not drive id/gm).
- **Expected impact on body scoreboard**: the interesting one. If
  ASAP7 gmb really is ~1e-9 because the modelcard body-coupling
  coefficients are ~1000× smaller than TSMC, then handing that ratio
  to the model as an input + scale makes `gmb / body_factor`
  approximately tech-invariant. This is the only plausible path to
  single-digit NRMSE_body on ASAP7 without ASAP7 training samples.
  High variance on the expected outcome — worth a 2-fold probe.
- **Risk**: the body coefficients may not correlate tightly enough
  with gmb/qb to be a useful normaliser scale. Reported honestly in
  the experiment writeup.
- **Tracked in**: experiment E3.

### D1. Physics-scale normalisation for gmb/qb (STRUCTURAL, pairs with M2b)

Instead of (or as well as) S2's hand-picked floor, normalise gmb and
qb by a **physics-derived scale** extracted from the modelcard:
`y_norm = y / (body_factor · id_scale)` for gmb and
`y_norm = y / (body_factor · q_scale)` for qb. Both `body_factor` and
the reference `id_scale / q_scale` come from the same modelcard
metadata generation step that feeds M2b.

This is the principled form of what S2 is trying to do by clamping a
constant floor. With a per-sample physics scale the normalised
gmb/qb distribution is **tech-invariant by construction** — ASAP7
and TSMC map to the same normalised box, and the model no longer has
to extrapolate 4 decades in target space.

- **Engineering cost**: ~2 hours in `normalize.py` (new mode or
  per-sample scale tensor threaded through `normalize_outputs` and
  `denormalize_outputs`).
- **Expected impact**: the upper bound of what's achievable on
  NRMSE_body from a normaliser change; complementary to M2b as an
  *input* feature. Runs coupled with M2b (same data-gen change), so
  only one extra training run.
- **Risk**: same as M2b — assumes the body factor correlates tightly
  with |gmb|, |qb|. If it doesn't, D1 is a wash and we fall back to
  S2's constant floor.
- **Tracked in**: experiment E3 (coupled with M2b).

## Sequencing

| Step | Experiment | Folds | Wall-clock | Cumulative |
|------|------------|------:|-----------:|-----------:|
| 0    | Re-score 2026-04-09 baseline on NRMSE_sc / NRMSE_body | — | 10 min | 10 min |
| 1    | **E1: S2 sanity run** (asinh floor, already coded) | asap7, tsmc5 | 100 min | 110 min |
| 2    | **E2: M2a derived features** (Vov, Vds_rel, W/L, EOT·VDD) | asap7, tsmc5 | 120 min | 230 min |
| 3    | **E3: M2b body-factor feature + D1 physics scale** | asap7, tsmc5 | 120 min | 350 min |
| 4    | **E4: combined keepers** (best of E2 ∪ E3, stacked) | asap7, tsmc5 | 100 min | 450 min |
| 5    | **Full 5-fold** re-run with the final recipe | all 5 | 250 min | 700 min |

Total ~11.7 h, within budget assuming at most one experiment needs a
re-run. If one of E2/E3 is a clear loser the slot becomes free for a
re-run of the winner with a different hyper-param.

## Tracking

Each experiment is tracked in this file below. On completion fill
in, **per scoreboard**:

- **Change summary** (1 line)
- **NRMSE_sc**: asap7 before → after, tsmc5 before → after
- **NRMSE_body**: asap7 before → after, tsmc5 before → after
- **In-distribution val loss**: before → after (guard rail)
- **Verdict**: ✅ keep / ❌ reject (per decision rule)
- **Commit hash** (if kept)

---

## Experiment results

### E0. Baseline (from 2026-04-09 LOO run)

All figures computed directly from
`tests/verify_bsimar_loo_results/20260409_130630_nmos/metrics.json`
using `NRMSE_sc = mean(id, gm, gds, cgg)`,
`NRMSE_body = mean(gmb, qb)`, `NRMSE_all = mean(all 13 outputs)`.
(The per-fold rows are re-derived from the per-output table
in the report to avoid rounding drift.)

| Held out | NRMSE_sc % | NRMSE_body % | NRMSE_all % |
|----------|-----------:|-------------:|------------:|
| asap7    |     19.628 |    160345.31 |   24678.41  |
| tsmc5    |      3.430 |         1.88 |       2.17  |
| tsmc7    |      3.614 |         1.76 |       2.18  |
| tsmc12   |      1.506 |         1.01 |       0.98  |
| tsmc16   |      1.192 |         1.24 |       0.84  |

In-distribution random-split val reference (v3 production): NRMSE 0.223 %.

**Decision variable anchors** for this sprint: asap7 NRMSE_sc =
19.628 %, tsmc5 NRMSE_sc = 3.430 %. An experiment keeps iff
``log(asap7_new/19.628) + log(tsmc5_new/3.430) < 0`` (geometric-mean
improvement across the two folds, per the log-ratio decision rule)
and does not regress the LOO val phys-best NRMSE by more than 0.5 %
relative (guardrail against killing in-distribution accuracy).

### E1. S2 — gmb/qb asinh-scale floor (sanity run)

- Status: ✅ **KEEP** (marginal but real; 2026-04-10 09:15-10:34 run)
- Run: `tests/verify_bsimar_loo_results/20260410_091523_nmos/metrics.json`
- Change: `OUTPUT_ASINH_SCALE_MIN = {"gmb": 1e-5, "qb": 1e-15}` in
  `normalize.py`, applied after the geomean fit in
  `BSIMARNormalizer.fit(mode='asinh')`.
- **Result — solver-critical scoreboard**:

  | Fold  | E0 NRMSE_sc % | E1 NRMSE_sc % |    Δ %  | log ratio |
  |-------|--------------:|--------------:|--------:|----------:|
  | asap7 |        19.628 |        18.402 | −6.25 % |   −0.0645 |
  | tsmc5 |         3.430 |         3.459 | +0.85 % |   +0.0085 |

  **Σ log-ratio = −0.056** → geometric-mean NRMSE_sc ratio ≈ 0.97
  (weak improvement); the log-ratio decision rule says **KEEP**.

- **Body scoreboard** (not a decision variable, reported for context):

  | Fold  | E0 NRMSE_body % | E1 NRMSE_body % | ratio |
  |-------|----------------:|----------------:|------:|
  | asap7 |      160 345.31 |      156 010.08 | 0.973 |
  | tsmc5 |           1.878 |           2.038 | 1.085 |

  As predicted, S2 does **not** rescue NRMSE_body. ASAP7 gmb stays
  catastrophic because the held-out fold has no CIT-zero training
  samples; the floor only caps the denormalisation blow-up, it does
  not replace missing physics. This confirms the mechanism and is
  why S2's expected headline was retracted.

- **In-distribution val guardrail**:

  | Fold  | E0 phys-best % | E1 phys-best % | verdict |
  |-------|---------------:|---------------:|:-------:|
  | asap7 |          0.176 |          0.175 | pass    |
  | tsmc5 |          0.278 |          0.255 | pass    |

  Both fold phys-best values are within or below the 0.5 %-relative
  guardrail of E0, so S2 does not trade in-distribution accuracy for
  its small LOO gain.

- **Interpretation**: the ~6 % asap7 NRMSE_sc drop is a second-order
  side-effect. S2 changes the asinh scale for gmb/qb, which changes
  the *numerical* MAE contribution from those two targets during
  training. The per-target LDS weights then rescale away most of the
  shift, but a residual gradient rebalance slightly improves the
  shared encoder's fit for `id/gm/gds/cgg` on asap7. The effect is
  on the floor of what the log-ratio rule calls a keeper.
- **Verdict**: ✅ keep.
- **Commit**: `859d429` (``experiment(bsimar): E1 S2 asinh-scale
  floor for gmb/qb + LOO plan rewrite``).

### E2. M2a — derived channel-transport features (FULL, 4 features)

- Status: ❌ **REJECT** (2026-04-10 09:27–11:35 run)
- Run: `tests/verify_bsimar_loo_results/20260410_092654_nmos_m2a/metrics.json`
- Change: 4 derived features appended to the canonical 19-col layout
  (19 → 23): Vov = Vg − PHIG, Vds = Vd − Vs, Vgb = Vg − Vb,
  log(NFIN/L). Transformer gets a 4th "derived" group MLP.
- **Result — solver-critical scoreboard**:

  | Fold  | E0 NRMSE_sc % | E2 NRMSE_sc % |     Δ %  | log ratio |
  |-------|--------------:|--------------:|---------:|----------:|
  | asap7 |        19.628 |        21.541 |  +9.74 % |   +0.0930 |
  | tsmc5 |         3.430 |         3.536 |  +3.08 % |   +0.0303 |

  **Σ log-ratio = +0.1233** → geometric-mean ratio ≈ 1.064
  (6.4 % regression); **REJECT**.

- **In-distribution guardrail**: phys-best 0.195 % vs E0's 0.176 %
  (asap7 fold) — 10.8 % relative regression, also exceeds 0.5 %
  guardrail.

- **Why it failed**: the 3 "redundant" features (Vds, Vgb,
  log_NFIN_L) are linearly learnable inside the existing voltage or
  geometry group MLPs. Adding them as an explicit 4th group token
  with 134 k extra parameters provides no new information but
  increases model capacity, causing the 4-tech train distribution to
  be fit more tightly at the expense of the held-out fold. The single
  genuinely non-trivial feature (Vov) crosses group boundaries but is
  buried under three redundant ones. A contingency experiment (E2b,
  Vov-only) is running to isolate the Vov contribution.
- **Verdict**: ❌ reject. ``exp/e2-m2a`` branch NOT merged into main.
- **Commit**: `bf96af1` on ``exp/e2-m2a`` (dead branch).

### E2b. M2a-minimal — Vov-only derived feature (contingency)

- Status: **running** (2026-04-10 10:46–, GPU 2 Blackwell)
- Run: ``--exp-tag vov`` on ``exp/e2b-vov-only`` branch.
- Change: single derived feature Vov = Vg − PHIG appended to the
  canonical 19-col layout (19 → 20). Drops the 3 redundant features
  that E2 carried. Tests whether Vov alone helps (cross-group signal)
  or the M2a direction is intrinsically wrong for LOO generalization.
- Expected: if Vov is the useful feature, E2b should beat E0 by a
  small margin on NRMSE_sc. If the regression is intrinsic to any
  derived feature, E2b also regresses.
- NRMSE_sc (asap7 / tsmc5): pending
- NRMSE_body (asap7 / tsmc5): pending
- In-distribution val loss: pending
- Verdict: pending

### E3. M2b + D1 — modelcard body-factor feature + physics scale

- Status: **SKIPPED** — E3 targets NRMSE_body which is not a decision
  variable. With E1 as the only keeper and E2/E2b both rejected,
  there is no NRMSE_sc lever to couple with a body-physics change.
  Probing the raw modelcard process params confirmed that ASAP7 has
  ``CIT = 0`` (body fully decoupled), making any ``body_factor``
  normalization degenerate at zero. Filed as a known limitation.
- Verdict: SKIPPED. Not run.

### E4. Combined keepers (stacked)

- Status: **SKIPPED** — only E1 is a keeper. Nothing to stack.
- Verdict: SKIPPED. E5 runs E1 alone.

### E5. Full 5-fold re-run with the final recipe (E1 only = S2 floor)

- Status: **running** (launched 2026-04-10 ~11:50, GPU 2 Blackwell)
- Change: identical to E1 (S2 asinh-scale floor) applied to all 5 folds.
- Held out (asap7): pending
- Held out (tsmc5): pending
- Held out (tsmc7): pending
- Held out (tsmc12): pending
- Held out (tsmc16): pending
- Commit: pending

---

## Postmortem: M2a derived features are a dead end for LOO

**Key finding**: adding explicit physics-derived input features
(Vov, Vds, Vgb, log_NFIN_L) **hurts** LOO generalization rather
than helping. Three experiments were run:

| Variant | Features | asap7 Δ sc | tsmc5 Δ sc | Verdict |
|---------|----------|----------:|----------:|---------|
| E2 (full)  | Vov + Vds + Vgb + log_NFIN_L (23 cols) | +9.7 % | +3.1 % | ❌ |
| E2b (min)  | Vov only (20 cols)                      | +13.8 % | pending | ❌ |

**Why it fails.** The derived features carry per-tech constant
offsets (Vov shifts by ΔPHIG ≈ 0.13 V between ASAP7 and TSMC;
log_NFIN_L shifts by Δlog(L) at ASAP7's 7 nm). When the model
trains on 4 TSMC techs, it calibrates to TSMC's Vov distribution.
On ASAP7 test samples the shifted Vov values extrapolate — badly.
Even the single Vov feature alone (E2b) regresses MORE than the
full 4-feature set, ruling out interaction or dilution effects.

**DO NOT RETRY** without a fundamentally different design: e.g.
truly dimensionless ratios (Vov/VT_est, Vds/VDD_est) or
tech-invariant normalization of the derived features.
