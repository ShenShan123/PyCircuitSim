# V6 TSMC5 follow-up — session report

**Date:** 2026-05-11
**Branch:** `feat/v6` (no new branches needed; one commit landed on `feat/v6`)
**Scope:** TSMC5 inverter accuracy. DirectNet (LEVEL=73) only;
BSIMAR Transformer treated as control. ASAP7 excluded.
**Predecessor:** `docs/plans/2026-05-09-v6-accuracy-plan.md` —
"Open follow-ups" §1 (TSMC5 inverter at model-fit floor).

## Starting state — V6-prod on `feat/v6` tip

| Cell | TSMC5 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|
| DN single-dev DC NRMSE % | 0.98 | 3.22 | 0.18 | 0.19 |
| DN inverter VTC | **NR_FAIL (59 286 % bounded)** | 9.97 PASS | 6.50 PASS | 5.40 PASS |
| DN inverter tran | **17.76 FAIL** | 10.88 PASS | 3.79 PASS | 8.90 PASS |
| BSIMAR inverter VTC | **72.66 FAIL** | 11.85 PASS | 10.44 PASS | 48.19 FAIL |
| BSIMAR inverter tran | **20.43 FAIL** | 10.43 PASS | 10.40 PASS | 14.18 PASS |

Goal: identify a lever that unblocks TSMC5 inverter without regressing
the 6 PASSing TSMC12/16/7 inverter cells.

## What we tried

### Lever H1 — Per-tech inverter geometry (harness change) — **SHIPPED**

**Motivation.** Verifier hard-coded `l_nmos=16nm, l_pmos=20nm, nfin=2`
for all TSMC techs. TSMC5's V4 NMOS L bins are `{6, 20, 36, 54, 86}` nm,
so L=16nm sits in a 14 nm extrapolation gap. None of TSMC7/12/16 has
this gap on both axes. Methodological problem: confounds "model is bad
at TSMC5" with "verifier asked an extrapolation question."

**Patch.** Added `inv_l_nmos`, `inv_l_pmos`, `inv_nfin` to
`TestTechConfig` with property fallbacks. TSMC5 + TSMC7 overridden to
`inv_l_nmos=20e-9, inv_l_pmos=20e-9, inv_nfin=2` (in-bin for both).
New `{resolve,create_baked}_inv_{nmos,pmos}_modelcard` helpers handle
the NGSPICE-side modelcard baking. Single-device DC + NMOS/PMOS pulse
tests untouched.

**Verification.** TSMC12 inverter byte-clean (VTC 6.50/10.44, tran
3.79/10.40) — confirms no leak into unchanged techs. TSMC5 with new
in-bin geometry:

| Cell | L=16nm (old) | L=20nm in-bin (new) | Δ |
|---|---:|---:|---|
| BSIMAR VTC | 72.66 | 56.53 | −16.13 pp |
| DN VTC | 59 286 (NR-bound) | 59 286 (NR-bound) | 0 |
| BSIMAR inv-tran | 20.43 | 18.14 | −2.29 pp |
| DN inv-tran | 17.76 | 15.69 | −2.07 pp |

**Decision.** SHIPPED as commit `80fc2d4`. The methodological win is
permanent: future per-tech accuracy reporting now uses each tech's own
training bins. The TSMC5 inverter still FAILs in-bin, which **demotes
the geometry-extrapolation hypothesis** — the failure is not a verifier-
question artefact; it's genuine model weakness.

### Lever H2 — Per-tech VDD → VT in `_apply_vds_correction` — **REVERTED**

**Motivation.** Rule 19b uses `VT = max(0.06·VDD_train, 0.026)` with
`VDD_train` derived from the universal training stats — blended to
≈0.80 V (TSMC12/16's nominal VDD). For TSMC5 at VDD=0.65 V this VT
is ~23 % too loose, over-suppressing Id at small |Vds|. The "low-rail
+ transition" error pattern (~22 %, vs 9 % high-rail) on TSMC5
inv-tran is consistent with this story.

**Patch.** Added `_TECH_CODE_VDD: Dict[int, float]` lookup in
`pycircuitsim/models/mosfet_nn.py`, built from `CODE_TO_TECH_VARIANT`
and `TECH_CONFIGS`. New instance attr `self._vdd_per_tech` set at
construction. `_apply_vds_correction` reads `VT = max(0.06 *
self._vdd_per_tech, 0.026)`. Rail-restoring threshold (step a) kept
on the universal `_vdd_estimate` because the model's actual
extrapolation cliff is the universal Vds-max, not per-tech VDD.

| code | tech | VDD | VT_old | VT_new |
|---|---|---:|---:|---:|
| 0–3 | tsmc5 | 0.65 | 0.048 | **0.039** |
| 4–6 | tsmc7 | 0.75 | 0.048 | **0.045** |
| 7–11 | tsmc12 | 0.80 | 0.048 | 0.048 |
| 12–16 | tsmc16 | 0.80 | 0.048 | 0.048 |

**Verification on TSMC12+TSMC5 inverter.**

| Cell | V6 baseline | H2 | Verdict |
|---|---:|---:|---|
| TSMC12 DN VTC | 6.50 | 6.50 | unchanged ✓ |
| TSMC12 DN tran | 3.79 | 3.79 | unchanged ✓ |
| TSMC12 BSIMAR VTC | 10.44 | 9.48 | drift (GPU non-determinism) |
| TSMC12 BSIMAR tran | 10.40 | 10.40 | unchanged ✓ |
| **TSMC5 BSIMAR VTC** | 56.53 | **227 946 %** | **catastrophic** |
| **TSMC5 DN VTC** | 59 286 | **230 045 %** | **catastrophic** |
| **TSMC5 BSIMAR tran** | 18.14 | **NR ERROR** | **catastrophic** |
| **TSMC5 DN tran** | 15.69 | **NR ERROR** | **catastrophic** |

**Decision.** REVERTED via `git restore pycircuitsim/models/mosfet_nn.py`.

**Why it failed.** The `gds` linear-region term in step (c) scales as
`|id_raw|·exp(-|vds|/VT)/VT`. A smaller VT amplifies Jacobian
magnitude near Vds=0 in proportion to 1/VT. For TSMC5 (VT 0.048 →
0.039, 19 % tighter), this term spikes ~28 % at small |vds|,
destabilizing NR. The exponential-suppression-of-Id story was real,
but the gds-amplification side effect dominates the inverter circuit
solve.

**Pinned learning.** Tightening VT alone is structurally unsafe in
this form. A future per-tech VT attempt must *simultaneously* damp the
`|id|·exp(-|vds|/VT)/VT` term (e.g., clamp at a tech-independent
ceiling, or replace with a softer Vds-dependent floor). See
"What NOT to retry" below.

### Lever H3 — Per-tech asinh `s_k` retrain — **FALSIFIED at pre-flight**

**Motivation.** AsinhNormalizer uses a single universal per-target
scale `s_k` (geomean of |y_k| across all rows). Hypothesis: TSMC5
outputs end up in a low-resolution band of the asinh curve because
the universal scale is dominated by larger-tech magnitudes.

**Pre-flight diagnostic** (no retrain run).
Per-tech geomean of `|y_k|` over ASAP7-excluded rows of
`universal_nmos.npz`:

| col | TSMC5 | TSMC7 | TSMC12 | TSMC16 | max/min | universal `s_k` |
|---|---:|---:|---:|---:|---:|---:|
| id  | 4.87e-5 | 6.70e-5 | 4.37e-5 | 4.44e-5 | **1.5×** | 4.91e-5 |
| gm  | 3.90e-5 | 5.66e-5 | 5.45e-5 | 5.63e-5 | 1.5× | 5.14e-5 |
| gds | 1.82e-6 | 3.24e-6 | 3.18e-6 | 3.38e-6 | 1.9× | 2.86e-6 |
| qg  | 5.48e-17 | 9.51e-17 | 1.26e-16 | 1.33e-16 | 2.4× | 9.99e-17 |
| cgd | 2.97e-17 | 5.28e-17 | 1.09e-16 | 1.17e-16 | 3.9× | 7.14e-17 |
| cdd | 2.76e-17 | 4.97e-17 | 1.05e-16 | 1.13e-16 | 4.1× | 6.79e-17 |

Spread 1.5–4.1×. TSMC5's |y|/`s_k` ratios are 0.4–1.0, all in asinh's
near-linear region where the chain rule preserves resolution. A
per-tech `s_k` would shift the band by at most 0.3–1.0× — too small
to explain a 56 % → PASS gap.

**Decision.** SKIPPED. No code touched. The 25-min retrain + ~200-LOC
normaliser refactor is not justified by the magnitude data.

**Bonus signal from the same diagnostic.** TSMC5's `|gds|` is 1.8×
smaller than the other techs while `|id|` is only 1.5× smaller —
TSMC5 has a more pronounced saturation plateau than its peers.
This led to H4.

### Lever H4 — TSMC5 hot-region input densification — **FALSIFIED at pre-flight**

**Motivation.** V5 D1 diagnostic on TSMC7 NMOS showed the universal
LHS sampler under-samples the strong-inversion + saturation plateau
(high-Vgs, low-Vds) by 16×. Combined with H3's bonus signal
(TSMC5 has the deepest plateau), TSMC5 should be doubly disadvantaged
on the same axis.

**Hot region tested**: TSMC5 NMOS, Vgs ∈ [0.325, 0.65] V (0.5·VDD to
VDD), Vds ∈ [0, 0.195] V (0 to 0.3·VDD), codes 0–3, all 6 NFIN bins,
all 5 L bins, 3 temperatures.

**Pre-flight diagnostic** (executed in worktree by delegated agent,
no destructive actions).

Per-bin median rows-in-box, current `universal_nmos.npz`:

| Tech | TSMC5 | TSMC7 | TSMC12 | TSMC16 |
|---|---:|---:|---:|---:|
| Per-bin median rows | **247** | 255 | 274 | 275 |
| Per-V² density (M rows/V²) | **1.40** | 0.98 | 1.14 | 1.14 |

Spread < 12 % on per-bin counts. TSMC5 is the densest tech per unit
voltage area (smaller VDD packs same fractional coverage into less
space). Tried two alternative interpretations of "saturation plateau
hot region" — TSMC5 highest density in both.

**Decision.** SKIPPED, pre-flight gate `under-sampling ≥ 4×` failed
(actual ratio ≈ 1.1×). Risk profile maps to V5 `inv_trip` overlay
which regressed TSMC7/12/16 in production; expected gain ≈ zero.
Worktree auto-cleaned (no changes).

## What we falsified

After this session, four candidate root causes for the TSMC5 inverter
FAIL have been empirically ruled out **for the v6-shipping recipe**:

1. **Geometry extrapolation** (L=16nm gap). Tested via harness fix;
   modest 2 pp gain on transients only. Net contribution to the FAIL
   is small.
2. **Per-tech VT mismatch** in `_apply_vds_correction`. Tighter VT
   destabilises NR through the `|id|·exp/VT` term in step (c).
   Structural side-effect dominates the intended Id-suppression fix.
3. **Per-tech asinh `s_k` resolution band**. Magnitude spread
   1.5–4.1×, all in asinh's linear region — too small a lever for
   the gap.
4. **Input-side hot-region under-sampling** on the (high-Vgs, low-Vds)
   plateau. TSMC5 is the *densest* tech in this box already.

## What we shipped

| Commit | Change |
|---|---|
| `80fc2d4` | `test(harness): per-tech inverter geometry — TSMC5/TSMC7 to L=20nm/NFIN=2` |

Plus this session report.

## What NOT to retry (and why)

- **Tightening VT (or otherwise rescaling step b/c of rule 19) in
  isolation.** Whatever per-tech behaviour you want from step (b), the
  step (c) `gds` linear-region term needs a matching damping change or
  NR will blow up on TSMC5. See H2 postmortem above.
- **Naïve trip-region or hot-region overlays on the universal dataset.**
  V5 `inv_trip` and V5 D1 (TSMC7 hot-box, E4/E5) both regressed other
  techs by +2.7 pp NMOS DC despite the per-tech embedding. The per-bin
  count data above also shows the proposed hot region is not under-
  sampled to begin with for TSMC5.
- **Per-tech asinh `s_k` at the universal-recipe magnitudes.** Doesn't
  move the needle for the observed 1.5–4.1× spread.

## What might still work (no strong-conviction hypothesis)

These are honest gut calls, not data-backed plans. Pick one only if
the diagnostic argument is sharpened first.

- **TSMC5-only per-tech output residual.** A small MLP head conditioned
  on tech-code, trained only on TSMC5 rows with the universal model
  frozen. Different lever from per-tech `s_k`: nonlinear,
  multi-output, can capture the deeper saturation plateau without
  touching other techs. ~2 h plumbing + 30 min train. Risk: residual
  may not transfer to circuit-level NR because it changes the
  Jacobian shape autograd sees.
- **B0-medium asinh DN retrain at 200 epochs (520 K params).** Already
  named in the V6 plan's Open follow-ups. The 56 K probe saturated
  *test-set* NRMSE; the question is whether capacity helps *circuit-
  level* extrapolation. ~4 GPU-h. Risk: probe-level evidence suggests
  capacity is not the constraint.
- **Alternative-axis under-sampling diagnostic.** Run a D1-style heatmap
  on TSMC5 NMOS but in the *Vbs × T* plane, or in *L × NFIN* corners,
  not the (Vgs, Vds) plane H4 already ruled out. If a different axis
  shows the 16× under-sampling, that's the densification target.
  ~30 min code + diagnostic, no retrain.
- **DCSolver-side fix: detect bounded-numeric VTC as a distinct failure
  mode.** The Tier 1A trust-region clamp caps per-step ΔV; the TSMC5
  DN VTC sits at the clamp ceiling for the whole sweep, indicating
  the solver iterates in place. A "no global progress over N iters"
  watchdog (S3 from the original feasibility review) could escalate
  GMIN or fail fast. Solver-only, doesn't fix the model but stops it
  from spending time on a degenerate solve. ~1 day.

## Open question

Whether any of the above is worth pursuing depends on whether TSMC5
inverter accuracy is on the critical path for the next downstream
goal (SRAM validation? a paper? new tech rollout?). Pure model-
quality work without that anchor is unbounded.
