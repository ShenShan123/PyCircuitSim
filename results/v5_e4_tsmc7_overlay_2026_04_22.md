# v5 Experiment E4 — TSMC7 Hot-Box Overlay + BSIMAR NMOS Fine-Tune

**Date:** 2026-04-22
**Plan reference:** `results/v5_improvement_plan_2026_04_21.md` §15 (+ D1 recommendation §14)
**Baseline commit:** `b9d35ce` (state as of `results/v5_baseline_2026_04_22.md`)
**Verdict:** **REVERT**

## 1. Data generation

- **Script:** `external_compact_models/PyCMG/scripts/generate_tsmc7_overlay.py`
  (new file; reuses `pycmg.nn_generate._create_model_and_instance` +
  `eval_single_point`; 2D LHS in `(Vgs, Vds)`; canonical
  `(inputs, geometry, outputs)` layout matching `universal_nmos.npz`).
- **Spec:** TSMC7 {SVT, LVT, ULVT} × NFIN ∈ {3, 5, 10, 15, 20} × L ∈ {14, 16, 18, 20} nm,
  `Vgs ∈ [0.444, 0.750] V`, `Vds ∈ [0.329, 0.750] V`, `Vbs = 0`, `T = 300.15 K`,
  500 LHS samples per bin → 3 × 5 × 4 × 500 = 30 000 rows.
- **Runtime:** 27 s wall-clock (8 workers, OPENBLAS/OMP each 1 thread).
- **File:** `external_compact_models/bsimar/data/datasets/tsmc7_overlay_nmos.npz`
  (7.3 MB, 30 000 samples, 0 failures).
- **Concat:** `universal_nmos_plus_tsmc7_overlay.npz` = 11 856 240 + 30 000 =
  **11 886 240 rows** (2.9 GB). Tech-variant cache built by reusing the existing
  `universal_nmos_tech_variant_labels.npy` for the first 11.86 M rows and
  fingerprinting only the 30 000 overlay rows (60 new `(tech, variant, L, NFIN)`
  combinations). 0 misses. Per-code distribution: TSMC7 codes 4/5/6 go from
  592 812 → 602 812 each (+10 000 / variant).

## 2. Fine-tune

- **Harness:** `bsimar.training.finetune.finetune_v4` on
  `v4_universal_nmos_best.phys.pt` → save prefix `v4_ft_tsmc7_overlay_nmos`.
- **Schedule:** 30 TF epochs (cosine, lr=1e-4, bs=2048) + 3 AR finetune epochs
  (lr=1e-5), `finetune_techs={'tsmc7'}`, `new_num_tech_codes=18`.
- **Split:** train 1 476 678 / val 164 075 / test(regression) 7 362 390.
- **Runtime:** **6 159 s = 102.65 min** on 1× A100-40GB.
- **Final phys metrics on TSMC7 val set** (averaged across 13 targets):
  **NRMSE_phys 0.462 %, MRE 3.29 %, R² 0.9922**. Per-target NRMSE ≤ 0.76 % (qb
  worst). Phys-best checkpoint selected at FT-AR epoch 2.
- **PMOS:** symlinked from `v4_universal_pmos_*` (not retrained).

## 3. Verification results

Baseline = commit `a9837a5`; numbers from `results/v5_baseline_2026_04_22.md`.

### TSMC7 (target of fine-tune)

| Metric    | Baseline | E4 (ft nmos + univ pmos) | Δ (pp) | Gate (>5 pp drop)? |
|-----------|---------:|-------------------------:|-------:|:------------------:|
| NMOS DC   |  14.72 % |              **17.40 %** | +2.68  | no (regressed)     |
| PMOS DC   |   3.06 % |                 3.06 %  | 0.00   | n/a (unchanged)    |
| VTC       |  19.15 % |              **19.34 %** | +0.19  | no (flat)          |
| Transient |   9.14 % |                 9.17 %  | +0.03  | no (flat)          |

### TSMC12 (regression sanity)

| Metric    | Baseline | E4   | Δ (pp) | >3 pp regression? |
|-----------|---------:|-----:|-------:|:-----------------:|
| NMOS DC   |   9.95 % |10.71%| +0.76  | no                |
| VTC       |   4.10 % | 8.23%| **+4.13** | **yes**        |
| Transient |   6.78 % | 7.12%| +0.34  | no                |

## 4. Verdict — REVERT

All three hard acceptance gates (§15) failed:

1. **TSMC7 NMOS DC drop > 5 pp:** FAIL (Δ = **+2.68 pp**, regressed).
2. **TSMC7 VTC drop > 5 pp:** FAIL (Δ = +0.19 pp, flat).
3. **No other-tech NMOS DC regresses > 3 pp:** TSMC12 VTC +4.13 pp crosses the
   cross-tech threshold (VTC is dominated by NMOS+PMOS joint error, so a
   mixed-tech regression is still a systemic fit loss).

## 5. Load-bearing observations

- Training-space phys NRMSE **0.462 %** — indistinguishable from E3's 0.454 %.
  Same 29-30× training↔inference gap noted in E3 §13.
- E3 vs E4 on TSMC7 inverter: TSMC7 NMOS DC went **14.72 → 14.74 → 17.40 %**
  (E3 flat, E4 regressed). Targeted hot-box samples did NOT reduce inference-
  space NMOS DC error; they increased it.
- E3's postmortem (plan §13 "load-bearing lesson") framed the gap as
  *distribution mismatch* between LHS density and verifier bias points.
  E4 tested the specific fix D1 recommended (densify the hot box). The fix
  is insufficient on its own — the 30 K TSMC7 overlay rows are still only
  1.6 % of the TSMC7 training partition (30 000 / 1 808 436), and LDS
  per-target weighting inside `MAELoss` re-normalises them back down.
- TSMC7 VTC (−0.01 pp → +0.19 pp) tracks TSMC7 NMOS DC almost 1:1 as expected
  from E3's diagnosis.
- TSMC12 VTC regression (+4.13 pp) is the surprise — fine-tuning only on
  TSMC7 codes 4/5/6 and freezing the embedding table at 18 still disturbed
  the TSMC12 code 7 representation enough to shift the inverter operating
  point. Consistent with E3's TSMC12 VTC regression (+3.61 pp).

## 6. Next step

Per plan §13 updated sprint order, the sequence "D1 → E4" is now exhausted
as cheap options for P1 (TSMC7 NMOS DC). The remaining levers in priority
order:

- §4.4 inverter-trajectory overlay **with LDS bypass** (not tried yet;
  E4 went through the default LDS path, which likely down-weighted the
  hot-box rows). Requires a `is_overlay` flag plumbed through
  `trainer.py` + `bni_mae.py`.
- §4.1 tanh gate + §4.5 simplified inference (structural, full retrain).

E4 checkpoints (`v4_ft_tsmc7_overlay_nmos_*`) and overlay datasets should be
deleted unless kept for §4.4 LDS-bypass experiment reuse.

## 7. Files produced

- `external_compact_models/PyCMG/scripts/generate_tsmc7_overlay.py` (new)
- `external_compact_models/bsimar/data/datasets/tsmc7_overlay_nmos.npz` (7.3 MB)
- `external_compact_models/bsimar/data/datasets/universal_nmos_plus_tsmc7_overlay.npz` (2.9 GB)
- `external_compact_models/bsimar/data/datasets/universal_nmos_plus_tsmc7_overlay_tech_variant_labels.npy` (95 MB)
- `external_compact_models/bsimar/checkpoints/v4_ft_tsmc7_overlay_nmos_*.pt` (19 MB each)
- `external_compact_models/bsimar/checkpoints/v4_ft_tsmc7_overlay_pmos_*` (symlinks to universal)
- `results/v5_e4_datagen.log`
- `results/v5_e4_finetune.log`
- `results/v5_e4_verify_tsmc7.log`
- `results/v5_e4_verify_tsmc12.log`
- `results/v5_e4_tsmc7_overlay_2026_04_22.md` (this file)

## 8. Summary numbers

- Data gen: **27 s** (30 000 rows, 0 failures).
- Fine-tune: **102.65 min** (30 TF + 3 AR epochs, 1× A100).
- TSMC7 inverter: **14.72 → 17.40 % NMOS DC (+2.68), 19.15 → 19.34 % VTC (+0.19),
  9.14 → 9.17 % Transient (+0.03).**
- TSMC12 sanity: NMOS DC +0.76, VTC +4.13, Transient +0.34.
- **Result: REVERT.** Hot-box overlay alone does not recover TSMC7 NMOS DC;
  cross-tech VTC regression > 3 pp.
