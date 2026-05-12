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

**Motivation.** Verifier hard-coded `l_nmos=16nm, l_pmos=20nm, nfin=2` for all TSMC techs. TSMC5's V4 NMOS L bins are `{6, 20, 36, 54, 86}` nm, so L=16nm sits in a 14 nm extrapolation gap. None of TSMC7/12/16 has this gap on both axes. Methodological problem: confounds "model bad at TSMC5" with "verifier asked an extrapolation question."

**Patch.** Added `inv_l_nmos`, `inv_l_pmos`, `inv_nfin` to `TestTechConfig` with property fallbacks. TSMC5 + TSMC7 overridden to `inv_l_nmos=20e-9, inv_l_pmos=20e-9, inv_nfin=2` (in-bin). New `{resolve,create_baked}_inv_{nmos,pmos}_modelcard` helpers handle NGSPICE modelcard baking. Single-device DC + NMOS/PMOS pulse untouched.

**Verification.** TSMC12 inverter byte-clean (VTC 6.50/10.44, tran 3.79/10.40) — no leak into unchanged techs. TSMC5 with new in-bin geometry:

| Cell | L=16nm (old) | L=20nm in-bin (new) | Δ |
|---|---:|---:|---|
| BSIMAR VTC | 72.66 | 56.53 | −16.13 pp |
| DN VTC | 59 286 (NR-bound) | 59 286 (NR-bound) | 0 |
| BSIMAR inv-tran | 20.43 | 18.14 | −2.29 pp |
| DN inv-tran | 17.76 | 15.69 | −2.07 pp |

**Decision.** SHIPPED as commit `80fc2d4`. Methodological win is permanent: per-tech accuracy reporting now uses each tech's own training bins. TSMC5 inverter still FAILs in-bin, which **demotes the geometry-extrapolation hypothesis** — failure is not a verifier-question artefact; it's genuine model weakness.

### Lever H2 — Per-tech VDD → VT in `_apply_vds_correction` — **REVERTED**

**Motivation.** Rule 19b uses `VT = max(0.06·VDD_train, 0.026)` with `VDD_train` from universal training stats — blended to ≈0.80 V (TSMC12/16's VDD). For TSMC5 at VDD=0.65 V this VT is ~23 % too loose, over-suppressing Id at small |Vds|. The "low-rail + transition" error pattern (~22 % vs 9 % high-rail) on TSMC5 inv-tran fits.

**Patch.** Added `_TECH_CODE_VDD: Dict[int, float]` in `pycircuitsim/models/mosfet_nn.py`, built from `CODE_TO_TECH_VARIANT` + `TECH_CONFIGS`. New `self._vdd_per_tech`. `_apply_vds_correction` reads `VT = max(0.06 * self._vdd_per_tech, 0.026)`. Rail-restoring (step a) kept on universal `_vdd_estimate` — model's extrapolation cliff is universal Vds-max, not per-tech VDD.

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

**Why it failed.** `gds` linear-region term in step (c) scales as `|id_raw|·exp(-|vds|/VT)/VT`. Smaller VT amplifies Jacobian magnitude near Vds=0 in proportion to 1/VT. TSMC5 (VT 0.048 → 0.039, 19 % tighter) spikes ~28 % at small |vds|, destabilising NR. Id-suppression story was real, but gds-amplification side effect dominates the inverter solve.

**Pinned learning.** Tightening VT alone is structurally unsafe. Any future per-tech VT must *simultaneously* damp `|id|·exp(-|vds|/VT)/VT` (clamp at tech-independent ceiling, or replace with softer Vds-dependent floor). See "What NOT to retry".

### Lever H3 — Per-tech asinh `s_k` retrain — **FALSIFIED at pre-flight**

**Motivation.** AsinhNormalizer uses single universal per-target scale `s_k` (geomean of |y_k|). Hypothesis: TSMC5 outputs land in a low-resolution band of asinh because universal scale is dominated by larger-tech magnitudes.

**Pre-flight diagnostic** (no retrain). Per-tech geomean of `|y_k|` over ASAP7-excluded rows of `universal_nmos.npz`:

| col | TSMC5 | TSMC7 | TSMC12 | TSMC16 | max/min | universal `s_k` |
|---|---:|---:|---:|---:|---:|---:|
| id  | 4.87e-5 | 6.70e-5 | 4.37e-5 | 4.44e-5 | **1.5×** | 4.91e-5 |
| gm  | 3.90e-5 | 5.66e-5 | 5.45e-5 | 5.63e-5 | 1.5× | 5.14e-5 |
| gds | 1.82e-6 | 3.24e-6 | 3.18e-6 | 3.38e-6 | 1.9× | 2.86e-6 |
| qg  | 5.48e-17 | 9.51e-17 | 1.26e-16 | 1.33e-16 | 2.4× | 9.99e-17 |
| cgd | 2.97e-17 | 5.28e-17 | 1.09e-16 | 1.17e-16 | 3.9× | 7.14e-17 |
| cdd | 2.76e-17 | 4.97e-17 | 1.05e-16 | 1.13e-16 | 4.1× | 6.79e-17 |

Spread 1.5–4.1×. TSMC5's |y|/`s_k` ratios 0.4–1.0, all in asinh's near-linear region where chain rule preserves resolution. Per-tech `s_k` would shift band by at most 0.3–1.0× — too small for 56 % → PASS gap.

**Decision.** SKIPPED. No code touched. 25-min retrain + ~200-LOC normaliser refactor not justified.

**Bonus.** TSMC5's `|gds|` is 1.8× smaller than other techs while `|id|` is only 1.5× smaller — TSMC5 has a more pronounced saturation plateau. This led to H4.

### Lever H4 — TSMC5 hot-region input densification — **FALSIFIED at pre-flight**

**Motivation.** V5 D1 on TSMC7 NMOS showed universal LHS sampler under-samples strong-inversion + saturation plateau (high-Vgs, low-Vds) by 16×. With H3's bonus signal (TSMC5 deepest plateau), TSMC5 should be doubly disadvantaged.

**Hot region tested**: TSMC5 NMOS, Vgs ∈ [0.325, 0.65] V (0.5·VDD to VDD), Vds ∈ [0, 0.195] V (0 to 0.3·VDD), codes 0–3, all 6 NFIN bins, all 5 L bins, 3 temperatures.

**Pre-flight diagnostic** (worktree, no destructive actions). Per-bin median rows-in-box, current `universal_nmos.npz`:

| Tech | TSMC5 | TSMC7 | TSMC12 | TSMC16 |
|---|---:|---:|---:|---:|
| Per-bin median rows | **247** | 255 | 274 | 275 |
| Per-V² density (M rows/V²) | **1.40** | 0.98 | 1.14 | 1.14 |

Spread < 12 % on per-bin counts. TSMC5 is the *densest* tech per unit voltage area (smaller VDD packs same fractional coverage). Two alternative interpretations of "saturation plateau hot region" tried — TSMC5 highest density in both.

**Decision.** SKIPPED, pre-flight gate `under-sampling ≥ 4×` failed (actual ≈ 1.1×). Risk profile maps to V5 `inv_trip` overlay which regressed TSMC7/12/16; expected gain ≈ zero. Worktree auto-cleaned.

## What we falsified

Four candidate root causes for TSMC5 inverter FAIL empirically ruled out **for v6-shipping recipe**:

1. **Geometry extrapolation** (L=16nm gap). Modest 2 pp gain on transients only. Small net contribution.
2. **Per-tech VT mismatch** in `_apply_vds_correction`. Tighter VT destabilises NR via `|id|·exp/VT` in step (c). Structural side-effect dominates.
3. **Per-tech asinh `s_k` resolution band**. Spread 1.5–4.1×, all in asinh's linear region — too small.
4. **Input-side hot-region under-sampling** on (high-Vgs, low-Vds) plateau. TSMC5 is *densest* tech in this box.

## What we shipped

| Commit | Change |
|---|---|
| `80fc2d4` | `test(harness): per-tech inverter geometry — TSMC5/TSMC7 to L=20nm/NFIN=2` |

Plus this session report.

## What NOT to retry (and why)

- **Tightening VT (or rescaling step b/c of rule 19) in isolation.** Step (c) `gds` linear-region term needs matching damping or NR blows up on TSMC5. See H2 postmortem.
- **Naïve trip-region or hot-region overlays on universal dataset.** V5 `inv_trip` and V5 D1 (TSMC7 hot-box, E4/E5) both regressed other techs by +2.7 pp NMOS DC despite per-tech embedding. Per-bin counts also show proposed hot region is not under-sampled for TSMC5.
- **Per-tech asinh `s_k` at universal-recipe magnitudes.** Doesn't move needle for 1.5–4.1× spread.

## What might still work (no strong-conviction hypothesis)

Honest gut calls, not data-backed plans. Pick only if diagnostic is sharpened first.

- **TSMC5-only per-tech output residual.** Small MLP head conditioned on tech-code, trained on TSMC5-only rows with frozen universal backbone. Different lever from per-tech `s_k`: nonlinear, multi-output, can capture deeper saturation plateau without touching other techs. ~2 h plumbing + 30 min train. Risk: residual may not transfer to circuit-level NR (Jacobian shape changes).
- **B0-medium asinh DN retrain at 200 epochs (520 K params).** Named in V6 Open follow-ups. 56 K probe saturated *test-set* NRMSE; question is whether capacity helps *circuit-level* extrapolation. ~4 GPU-h. Risk: probe evidence suggests capacity is not the constraint.
- **Alternative-axis under-sampling diagnostic.** D1-style heatmap on TSMC5 NMOS but in *Vbs × T* plane, or *L × NFIN* corners (not Vgs×Vds, H4 ruled it out). If different axis shows 16× under-sampling, that's the densification target. ~30 min code, no retrain.
- **DCSolver-side fix: detect bounded-numeric VTC as distinct failure mode.** Tier 1A trust-region clamp caps per-step ΔV; TSMC5 DN VTC sits at clamp ceiling for full sweep — solver iterates in place. "No global progress over N iters" watchdog (S3 from original feasibility review) could escalate GMIN or fail fast. ~1 day.

## Open question

Whether any of the above is worth pursuing depends on whether TSMC5 inverter is on the critical path (SRAM validation? paper? new tech rollout?). Pure model-quality work without that anchor is unbounded.
