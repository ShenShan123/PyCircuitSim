# v5 Inverter-Transient Improvement Sprint — Session Summary

**Date:** 2026-04-22 / 2026-04-23
**Branch:** `feat/bsimar-v4-tech-code`
**Final state:** v4 production behaviour unchanged. v5 closed with no
new trained checkpoints shipped; substantial diagnostic + postmortem
value accrued.

## Goal

Improve the v4 NN compact-model inverter simulation accuracy, in
particular the worst-case inverter VTC TSMC7 (19.15 % NRMSE, hard
FAIL). Four-model ship (DirectNet {N,P}MOS + BSIMAR {N,P}MOS).

## What ran

Five cycles executed under the user's strict commit / experiment /
revert discipline:

| Step | Artifact | Outcome |
|------|----------|---------|
| v4 baseline | `results/v5_baseline_2026_04_22.md` — 8 cell × 4 tech measurements | Reference |
| E1 | Inference-time VT bump 0.06 → 0.10 × VDD in `_apply_vds_correction` | NEUTRAL on primary target (BSIMAR TSMC5 tran); reverted |
| E3 | BSIMAR NMOS TSMC7 per-tech fine-tune (same distribution) | Training NRMSE 0.454 % / inference NMOS DC 14.74 % — 29× gap; reverted |
| D1 | NN-vs-PyCMG error heatmap on TSMC7 SVT NMOS | Hot region isolated, 16× LHS under-sampling confirmed |
| E4 | Dense hot-box overlay concatenated with universal set | TSMC7 NMOS DC +2.68 pp; reverted |
| E5 | Overlay-only fine-tune (no LDS competition) | TSMC7 NMOS DC +2.75 pp (identical magnitude to E4); reverted |

## Findings

### Confirmed

- **The plan v1.1 baseline numbers were stale.** The rail-restoring
  fix (commit `381bbfc`, 2026-04-20) already moved DirectNet TSMC5
  transient from 17.20 % to 3.75 %. Real worst-case inverter failures
  are TSMC7 VTC (19.15 %), not rail-state transients.
- **Training-space NRMSE and inference-space NRMSE are essentially
  decoupled** at TSMC7. A fine-tune that drops training NRMSE to
  0.454 % leaves inference NMOS DC at 14.74 %.
- **The NN's error at TSMC7 is localised** to the strong-inversion +
  saturation plateau (D1). The LHS training sampler puts only 3.07 %
  of TSMC7-SVT rows in the top-decile error region.

### Ruled out

- Inference-time VT tuning as a primary fix (E1).
- Per-tech fine-tuning at the existing distribution (E3).
- Dense overlay with default LDS (E4).
- Dense overlay without LDS competition (E5).

### New hypotheses raised

1. **Sampling-basis mismatch** (v5 plan §17). The verifier sweeps
   uniform Id-Vgs at fixed Vds=VDD/2. The training set is LHS over
   the full (Vgs, Vds) box. Even when the NN nails the LHS
   distribution, a uniform Id-Vgs sweep's NRMSE is dominated by the
   point where |Id| is largest — and the LHS sampler under-weighs
   that region relative to its NRMSE contribution.
2. The E4/E5 **+2.7 pp NMOS DC regression** appears in both runs at
   identical magnitude despite structurally different training
   setups. This is consistent with the overlay samples shifting the
   Id-Vgs curve's *shape* at Vds=0.375 V away from the verifier's
   sweep basis, not merely its magnitude.

Neither hypothesis was tested.

## Decisions taken

1. **Accept TSMC7 NMOS DC as a known v4 limitation** (per user
   instruction post-E5). Documented in CLAUDE.md with root-cause
   explanation.
2. **Defer §4.1 tanh-gated structural retrain.** Low prior that a
   rail-state fix closes a mid-Vds saturation DC gap, and 24-48 h
   of GPU time is unjustified without evidence the tanh gate helps
   the specific failure mode.
3. **Close the v5 sprint.** No production code or checkpoint
   changes ship from v5.

## What does ship (indirectly)

| Artifact | Why keep |
|----------|----------|
| `tests/diag_d1_tsmc7_nmos_errors.py` | Reusable NN-vs-PyCMG error mapper, any tech/device |
| `external_compact_models/bsimar/training/finetune.py` patch | Legitimate edge-case fix: empty `test_idx` when all rows match `finetune_techs` |
| `external_compact_models/PyCMG/scripts/generate_tsmc7_overlay.py` (submodule) | Hot-box overlay generator; template for future non-LHS sampling work |
| `results/v5_baseline_2026_04_22.md` | Authoritative v4 baseline; supersedes plan v1.1 §1 numbers |
| `results/v5_improvement_plan_2026_04_21.md` | Full experimental narrative + four ruled-out hypotheses, for future sprints |
| `results/v5_d1_tsmc7_nmos_errors/` | Heatmaps + data for reproducing the sampling-basis diagnosis |
| `results/v5_e{1,3,4,5}_*.md` | Per-experiment postmortems |
| `results/v5_session_summary_2026_04_23.md` | This document |

## Recommendations for future work

If the TSMC7 NMOS DC gap becomes blocking (e.g. for SRAM validation
or other NMOS-critical circuits), try in this order:

1. **Uniform-sweep augmentation.** Generate a dense uniform
   (Vgs × Vds) grid per (tech, NFIN, L) bin using PyCMG; add as an
   LDS-bypass overlay. This directly tests the sampling-basis
   hypothesis (v5 plan §17) that no current experiment has.
2. **Shape-enforcing loss.** Add a first-derivative penalty on
   Id-Vgs (e.g. `(∂Id/∂Vgs)_nn ≈ (∂Id/∂Vgs)_pycmg`) at uniformly
   spaced Vgs samples. Addresses "the shape is wrong" hypothesis
   directly.
3. **§4.1 tanh-gated Id head** (full retrain). Only justified if
   rail-state accuracy regresses in some downstream workload, since
   v4's rail-restoring patch already handles that failure mode.
4. **Accept the v4 baseline** as sufficient for inverter-transient
   work (all 8 cells PASS 15 % threshold at TSMC5/7/12/16). VTC
   accuracy at TSMC5/7 can be reported with the 14-19 % NRMSE
   caveat documented in CLAUDE.md.

## Experiment count vs production change

- 5 experimental commits, 4 reverts, 1 diagnostic kept.
- 0 checkpoint files changed.
- 1 source-file patch retained (edge-case guard).
- 2 new diagnostic / generator scripts added.
- CLAUDE.md updated with failure-mode documentation.

Session closed.
