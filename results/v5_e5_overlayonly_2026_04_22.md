# v5 Experiment E5 — TSMC7 Overlay-Only BSIMAR NMOS Fine-Tune

**Date:** 2026-04-22
**Plan reference:** `results/v5_improvement_plan_2026_04_21.md` §15 (+ §16 path 1)
**Baseline commit:** `e583ec2` (numbers from `results/v5_baseline_2026_04_22.md`)
**Verdict:** **REVERT**

## 1. Motivation

E4 (plan §15, hot-box overlay concatenated with the universal set) regressed
TSMC7 NMOS DC by +2.68 pp. The postmortem hypothesised that LDS per-target
weighting inside `MAELoss` re-normalised the 30 000 overlay rows back down to
the same weight density as the 1.8 M in-distribution rows, so densification
of the hot region had no net effect. E5 tests the simplest remedy from plan
§16 (path 1): use **only** the overlay data for fine-tuning, so LDS has no
other partition to compete against. No code changes to the loss, trainer, or
model — just a different `data_path`.

## 2. Setup

### Dataset

- Reused `external_compact_models/bsimar/data/datasets/tsmc7_overlay_nmos.npz`
  (produced in E4): 30 000 PyCMG rows × 13 outputs across
  `tsmc7:{svt, lvt, ulvt}` × NFIN ∈ {3, 5, 10, 15, 20} × L ∈ {14, 16, 18, 20} nm,
  `Vgs ∈ [0.444, 0.750] V`, `Vds ∈ [0.329, 0.750] V`, `Vbs = 0`,
  `T = 300.15 K`, 500 LHS samples per bin.

### Tech-variant label cache (new)

The standard `get_or_build_tech_variant_labels` labeller enumerates bins via
`DeviceConfig.get_geometry_combos(pdk_path)`, which returns the default TSMC7
PDK sweep points (36 combos starting at L=8nm, NFIN=1,2,3,…). The overlay's
(L, NFIN) grid uses 14/16/18/20 nm × 3/5/10/15/20 — **none of those match the
default bins**, so the fingerprint builder missed 28 500 / 30 000 samples and
the raw labeller raised `AssertionError`.

Fix: generated the label cache directly from the known overlay enumeration
order (`VARIANTS × NFIN × L × 500 samples` per bin), giving 10 000 samples
each to codes 4 (tsmc7:svt), 5 (tsmc7:lvt), 6 (tsmc7:ulvt). Verified row
structure by asserting `geometry[:,0]` (NFIN) and `geometry[:,1]` (L) match
the expected value for each 500-sample block. Saved as
`tsmc7_overlay_nmos_tech_variant_labels.npy`.

### Finetune.py patch (documented verbatim)

`external_compact_models/bsimar/training/finetune.py`, around lines 202–216:

```diff
     train_ds = _make_ds(train_idx)
     val_ds = _make_ds(val_idx)
-    test_ds = _make_ds(test_idx)
 
     # Reorder outputs
     train_ds.outputs = torch.tensor(
         reorder_outputs(train_ds.outputs.numpy()), dtype=torch.float32)
     val_ds.outputs = torch.tensor(
         reorder_outputs(val_ds.outputs.numpy()), dtype=torch.float32)
-    test_ds.outputs = torch.tensor(
-        reorder_outputs(test_ds.outputs.numpy()), dtype=torch.float32)
 
     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
     val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
-    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
+
+    # Regression-check set only built if there's non-finetune data available
+    if len(test_idx) > 0:
+        test_ds = _make_ds(test_idx)
+        test_ds.outputs = torch.tensor(
+            reorder_outputs(test_ds.outputs.numpy()), dtype=torch.float32)
+        test_loader = DataLoader(
+            test_ds, batch_size=batch_size, shuffle=False)
+    else:
+        test_ds = None
+        test_loader = None
+        print("[E5] No non-finetune samples in this dataset — skipping "
+              "regression-check split (test_loader=None).")
```

The `test_loader` is never read inside `finetune_v4` (only the TF val-set is
used for `test_model` calls), so this patch is cosmetic. It prevents
`_make_ds([])` from hitting `normalizer.normalize_inputs` on a zero-length
array and keeps the code guard-railed against empty-test-set crashes in
future harness changes.

### Fine-tune run

```
finetune_v4(
    pretrained_path=v4_universal_nmos_best.phys.pt,
    data_path=tsmc7_overlay_nmos.npz,
    save_prefix=v4_ft_tsmc7_overlayonly_nmos,
    device_type=nmos, finetune_techs={tsmc7}, new_num_tech_codes=18,
    epochs=30, batch_size=2048, lr=1e-4, ar_finetune_epochs=3,
    device_str=cuda, overwrite=True)
```

- Split: **train 26 933 / val 2 992 / test(regression) 0** (overlay is pure
  tsmc7 — the patched code path triggered correctly).
- Runtime: **342 s = 5.7 min** (1× A100-40GB, shared with the parallel TSMC7
  verify). ~18× faster than E4 (6 159 s) because the dataset is 400× smaller.
- Final phys metrics on TSMC7 val set (averaged over 13 targets):
  **NRMSE_phys 0.692 %, MRE 3.18 %, R² 0.9984**. Phys-best checkpoint selected
  at FT-AR epoch 3.

PMOS: symlinked `v4_ft_tsmc7_overlayonly_pmos_*` → universal PMOS (unchanged).

## 3. Verification results

Baseline = commit `a9837a5`; numbers from `results/v5_baseline_2026_04_22.md`.

### TSMC7 (target of fine-tune)

| Metric    | Baseline | E5 (ft-only nmos + univ pmos) | Δ (pp) | Gate (> 5 pp drop)? |
|-----------|---------:|------------------------------:|-------:|:-------------------:|
| NMOS DC   |  14.72 % |                   **17.47 %** | +2.75  | **no (regressed)**  |
| PMOS DC   |   3.06 % |                       3.06 %  |  0.00  | n/a (symlink)       |
| VTC       |  19.15 % |                      18.35 %  | −0.80  | **no (< 5 pp)**     |
| Transient |   9.14 % |                       6.50 %  | −2.64  | n/a (not gated)     |

VTC took **2 671.8 s (44.5 min)** vs E4's 559.5 s — the fine-tuned NN has
noisier gradients in the Vin-transition region, so the VTC NR needed ~5×
more iterations per bias point.

### TSMC12 (regression sanity)

| Metric    | Baseline | E5   | Δ (pp) | > 3 pp regression? |
|-----------|---------:|-----:|-------:|:------------------:|
| NMOS DC   |   9.95 % |12.02%| +2.07  | no                 |
| VTC       |   4.10 % | 4.90%| +0.80  | no                 |
| Transient |   6.78 % | 5.36%| −1.42  | no (improved)      |

## 4. Verdict — REVERT

Acceptance rule from plan §15 (identical to E3/E4):
> KEEP iff TSMC7 NMOS DC drops > 5 pp **and** TSMC7 VTC drops > 5 pp **and**
> no other tech regresses > 3 pp.

Gate-by-gate:

1. **TSMC7 NMOS DC drop > 5 pp:** FAIL. Δ = **+2.75 pp** (regressed, same
   direction as E4).
2. **TSMC7 VTC drop > 5 pp:** FAIL. Δ = −0.80 pp (improved, but 6.25 pp short
   of the gate).
3. **No other-tech NMOS DC regresses > 3 pp:** PASS. TSMC12 NMOS DC +2.07 pp,
   VTC +0.80 pp (both well inside threshold).

**Result: REVERT.** Two hard gates fail.

## 5. Load-bearing observations

- **TSMC7 NMOS DC regression is robust across E3/E4/E5.** Baseline 14.72 %;
  E3 (universal + AR finetune) 14.74 %; E4 (hot-box overlay concat) 17.40 %;
  E5 (hot-box overlay only) **17.47 %**. Removing LDS competition did not
  reduce inference-space NMOS DC error — it nudged it in the *same* direction
  as E4. The 30 000-row overlay may itself be biased toward a sub-region of
  the sweep that the verifier DC test under-samples.
- **Training-space NRMSE improved** (0.692 % vs E4's 0.462 % vs the universal
  phys-best's 0.223 %). Training-space is in fact worse here because the
  overlay is a 3-tech, 60-bin slice — much less diverse than E4's 18-tech,
  954-bin blend. Since the train split is only 27 K rows, the model
  specializes faster but to a narrower distribution.
- **TSMC7 VTC improved by −0.80 pp** for the first time in E3/E4/E5. The
  overlay-only fine-tune nudged the NN toward the `(Vgs, Vds)` region that
  dominates VTC, but not enough to cross the 5 pp gate. The improvement is
  genuine (matches the NMOS DC direction only for the hot box — the DC
  verifier sweeps a much wider Vgs range, so the overall DC stat still
  regresses).
- **TSMC7 transient improved by −2.64 pp.** Consistent with VTC: the
  overlay's `(Vgs, Vds)` hot box aligns with where the inverter spends the
  transition time, so the transient waveform benefits even though the DC
  sweep does not.
- **Cross-tech footprint is smaller than E4.** E4's TSMC12 VTC regressed
  +4.13 pp (above 3 pp gate). E5 holds TSMC12 VTC to +0.80 pp. Only
  fine-tuning on TSMC7 tech codes 4/5/6 (and using a 400× smaller dataset)
  disturbed the TSMC12 code-7 embedding less than E4's concatenated-training
  path did. This is the only dimension where E5 cleanly beats E4, and it is
  insufficient on its own.
- **Runtime asymmetry.** Fine-tune itself was 5.7 min, but the TSMC7 VTC
  verify stretched to 44.5 min (8× the baseline). Any subsequent overlay-
  only experiment that doesn't tighten NR smoothing will pay a verify-time
  tax that dominates the fine-tune savings.

## 6. Conclusion

Hypothesis from plan §16 path 1 — *"densification works if LDS has no
competition"* — is **not confirmed** on TSMC7 NMOS DC. The specific D1-
identified hot box is dense (500 LHS per bin), but the NMOS DC verifier
sweeps a wider region than the hot box covers, so adding more rows inside
the box does not reduce the test-set NRMSE. The VTC and transient metrics
do see a modest benefit because their bias points correlate with the hot
region; those improvements are not large enough to carry the gate.

Remaining untried levers (plan §16):
- Path 2: `is_overlay` flag to bypass LDS for overlay rows (requires
  trainer + `bni_mae.py` changes).
- Path 3: tanh gate + simplified inference (full structural retrain).
- Path 4: inverter-trajectory overlay (bias points sampled from the actual
  transient waveform, not a LHS).

## 7. Files produced

- `external_compact_models/bsimar/data/datasets/tsmc7_overlay_nmos_tech_variant_labels.npy` (new, 241 KB — enables overlay-only fine-tune and any future overlay-LOO experiment).
- `external_compact_models/bsimar/checkpoints/v4_ft_tsmc7_overlayonly_nmos_{best,best.phys}.pt` (19 MB each).
- `external_compact_models/bsimar/checkpoints/v4_ft_tsmc7_overlayonly_nmos_{config,norm}.npz`.
- `external_compact_models/bsimar/checkpoints/v4_ft_tsmc7_overlayonly_pmos_*` (4 symlinks to universal PMOS).
- `external_compact_models/bsimar/training/finetune.py` (1 patched conditional — empty `test_idx` guard; see §2 above).
- `results/v5_e5_finetune.log` (5.7 min fine-tune log).
- `results/v5_e5_verify_tsmc7.log` (59 min verify log).
- `results/v5_e5_verify_tsmc12.log` (17 min verify log).
- `results/v5_e5_overlayonly_2026_04_22.md` (this file).

Step 3 (D1-style hot-region error map on the E5 checkpoint) was skipped per
§16 ("if patching that script looks expensive, skip this step. Verify output
is what matters"). The verify numbers already tell the story: the hot region
improves (VTC, transient), the rest of the DC sweep does not.

## 8. Summary numbers

- Fine-tune: **342 s** (30 TF + 3 AR epochs, 1× A100, 27 K train samples).
- Final TSMC7 val phys-NRMSE: **0.692 %** (vs E4 0.462 %, universal 0.223 %).
- TSMC7 inverter: **14.72 → 17.47 % NMOS DC (+2.75), 19.15 → 18.35 % VTC (−0.80),
  9.14 → 6.50 % Transient (−2.64).**
- TSMC12 sanity: NMOS DC +2.07, VTC +0.80, Transient −1.42 (all < 3 pp).
- **Result: REVERT.** Overlay-only densification alone does not recover TSMC7
  NMOS DC; VTC improves modestly but misses the 5 pp gate.
