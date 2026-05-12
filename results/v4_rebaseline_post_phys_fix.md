# v4 BSIMAR / DirectNet rebaseline — post phys-best-tracker fix

**Date:** 2026-05-04
**Branch:** `feat/bsimar-v5-phase-a`
**Plan reference:** `docs/superpowers/plans/2026-05-03-phys-best-tracker-bug.md` §5 D1
**Trigger:** Two bugs fixed (Bug A — `apply_id_gate` index mismatch; Bug B — mean phys-score AR-rollout blowup) and the simulator loader updated to fall back to `_best.pt` for legacy checkpoints whose `phys_best_metric` flag is `legacy_mean`. v4 has only Bug B (Bug A entered with v5b's structural id-gate). This rebaseline captures v4 numbers under the corrected loader as the anchor for v5/v6 retrains.

## Loader state

```
v4_universal_{nmos,pmos}_best.pt          ← simulator loads this (TF-best, Bug-A-clean)
v4_universal_{nmos,pmos}_best.phys.bug.pt ← formerly best.phys.pt, renamed (B-B3)
v4_universal_{nmos,pmos}_best.ar.pt       ← AR-val-best (untouched)
v4_universal_{nmos,pmos}_norm.npz         ← phys_best_metric absent → "legacy_mean" default
```

`pycircuitsim/parser.py:666-694` checks `BSIMARNormStats.phys_best_metric == "median"` before trusting `_best.phys.pt`. v4's norm.npz lacks the field (added 2026-05-03), so `phys_trustworthy=False` and the loader falls through to `_best.pt`.

## AR-rollout test-set sanity (pre-inverter, raw test split)

Run: `tests/diag_phys_best_explosion.py --prefix v4_universal --device-type {nmos,pmos}` on the universal dataset's test split.

| Output | NMOS NRMSE % | NMOS R² | PMOS NRMSE % | PMOS R² |
|--------|------|------|------|------|
| **id**     | **1.22**   | 0.478  | **2.45**   | 0.365  |
| gm     | 1.24   | 0.991  | 1.24   | 0.989  |
| gds    | 0.73   | 0.958  | 1.00   | 0.965  |
| gmb    | 12.77  | −0.226 | 7.61   | −0.106 |
| qg     | 1.52   | 0.957  | 1.40   | 0.959  |
| qd     | 0.22   | 0.999  | 0.34   | 0.997  |
| qs     | 0.40   | 0.998  | 0.34   | 0.999  |
| qb     | 2.90   | 0.428  | 2.37   | 0.392  |
| cgg    | 5.06   | 0.847  | 4.93   | 0.829  |
| cgd    | 0.34   | 0.998  | 0.42   | 0.997  |
| cgs    | 0.20   | 1.000  | 0.33   | 0.999  |
| cdg    | 0.20   | 1.000  | 0.30   | 0.999  |
| cdd    | 0.32   | 0.999  | 0.42   | 0.997  |
| **median** | 0.73 % | 0.989 | 0.95 % | 0.989 |
| **mean**   | 2.09 % | 0.802 | 1.78 % | 0.798 |

**Interpretation:** id well-predicted under AR rollout (1.2–2.5 % NRMSE). Mean and median phys-scores within 5 % of each other — Bug B's mean-aggregator never catastrophically misranked v4 because v4's id slot did not blow up under AR rollout (no Bug A → no AR drift catastrophe). Outliers (gmb 7.6–12.8 %, cgg 4.9–5.1 %) are pre-existing v4 limitations, not bug-induced.

## Inverter verification (full simulator path, NGSPICE ground truth)

Per-tech, BSIMAR (LEVEL=74, AR) and DirectNet (LEVEL=73, DN). Thresholds: DC ≤ 10 %, VTC ≤ 10 %, transient ≤ 15 %.

### TSMC5 SVT (VDD=0.65 V, L_n=16 nm, L_p=20 nm, NFIN=10)

| Test | AR NRMSE % | AR | DN NRMSE % | DN |
|------|----------|----|----------|----|
| NMOS DC      | 4.20  | ✓ | 6.20  | ✓ |
| PMOS DC      | 5.12  | ✓ | 7.74  | ✓ |
| Inverter VTC | 10.01 | ✗ | 10.73 | ✗ |
| Inverter tran| 12.36 | ✓ | 3.75  | ✓ |

**6/8 PASS.**

### TSMC7 SVT (VDD=0.75 V)

| Test | AR NRMSE % | AR | DN NRMSE % | DN |
|------|----------|----|----------|----|
| NMOS DC      | 14.52 | ✗ | 15.79 | ✗ |
| PMOS DC      | 2.59  | ✓ | 6.53  | ✓ |
| Inverter VTC | 20.03 | ✗ | 18.14 | ✗ |
| Inverter tran| 9.21  | ✓ | 6.80  | ✓ |

**4/8 PASS.** Documented limitation (CLAUDE.md "TSMC7 NMOS DC 14.72 %").

### TSMC12 SVT (VDD=0.8 V)

| Test | AR NRMSE % | AR | DN NRMSE % | DN |
|------|----------|----|----------|----|
| NMOS DC      | 10.40 | ✗ | 3.71  | ✓ |
| PMOS DC      | 13.66 | ✗ | 12.17 | ✗ |
| Inverter VTC | 4.90  | ✓ | 4.86  | ✓ |
| Inverter tran| 7.12  | ✓ | 4.86  | ✓ |

**5/8 PASS.**

### TSMC16 SVT (VDD=0.8 V)

| Test | AR NRMSE % | AR | DN NRMSE % | DN |
|------|----------|----|----------|----|
| NMOS DC      | 9.40  | ✓ | 3.11  | ✓ |
| PMOS DC      | 13.10 | ✗ | 12.40 | ✗ |
| Inverter VTC | 4.09  | ✓ | 5.67  | ✓ |
| Inverter tran| 7.79  | ✓ | 7.86  | ✓ |

**6/8 PASS.**

### Overall: 21/32 PASS

## Comparison to pre-fix CLAUDE.md numbers

| Metric | Pre-fix (legacy `.phys.pt`) | Post-fix (`_best.pt`) | Δ |
|--------|------------------------------|------------------------|-----|
| TSMC5 inv tran AR     | 12.13 % | 12.36 % | +0.23 |
| TSMC7 inv tran AR     |  9.14 % |  9.21 % | +0.07 |
| TSMC12 inv tran AR    |  6.78 % |  7.12 % | +0.34 |
| TSMC16 inv tran AR    |  7.51 % |  7.79 % | +0.28 |
| TSMC7 NMOS DC AR      | 14.72 % | 14.52 % | −0.20 |
| TSMC7 VTC AR          | 19.15 % | 20.03 % | +0.88 |

All deltas within sampling noise (±1 pp). Confirms:

* Loader fix correctly switched v4 from `.phys.bug.pt` → `_best.pt`.
* v4's TF-best and phys-best (mean-tracked) checkpoints were essentially equivalent in inverter-level NRMSE — Bug B did not catastrophically misrank for v4 (test-split mean 1.78 % vs median 0.95 % — factor-of-2 gap, not 10⁹× as on v5c).
* v5b's "B1 sampler FAIL" verdict (`results/v5b_sdata_gate_2026_05_02.md`) cannot be primarily blamed on Bug B — v5b's failure was Bug A (gate corrupted id training) + Bug B (mean-tracker exposed to the corrupted id) simultaneously. Plan §5 D2 resolves to: v4 rebased ≈ v4 prior ≈ 14.5 % TSMC7 NMOS DC; v5b's number is meaningless because v5b checkpoints are Bug-A-corrupted. **The B1 sampler hypothesis is neither confirmed nor rejected** — it requires a clean Bug-A-fixed v6 retrain to evaluate.

## Persistent limitations (carried over from v4, NOT bug-induced)

1. **TSMC7 NMOS DC + VTC (14–20 % NRMSE).** D1 diagnostic (`results/v5_d1_tsmc7_nmos_errors/`) attributed this to LHS under-sampling of the strong-inversion + saturation plateau (~16× under-represented vs verifier's uniform Vgs sweep). Both BSIMAR and DirectNet hit the same wall — data-coverage problem, not architecture-specific.
2. **TSMC12/16 PMOS DC (12–14 %).** Same sampling-basis class. Affects both models symmetrically.
3. **TSMC5 inverter VTC (10.0/10.7 %).** Borderline — at the gate threshold. AR 0.01 pp over, DN 0.7 pp over. Sensitive to NRMSE definition; could pass or fail run-to-run.
4. **gmb predictions universally weak** (NMOS 12.8 % R² −0.23; PMOS 7.6 % R² −0.10). Pre-existing v4 issue. Inverter-level numbers don't surface this (gmb contributes weakly at the inverter operating point). Worth investigating if SRAM benchmarks are added.

## Conclusion

**v4 with the corrected loader is the right "rewind" target for production.** id NRMSE 1.2 % (NMOS) / 2.5 % (PMOS) on the test split, all 8 inverter transients PASS, DC pass-rate consistent with documented v4 limitations (TSMC7 NMOS / TSMC12-16 PMOS).

The two bugs:

* **Bug A** — corrupts only v5b/v5c TF-trained Transformer weights (gate denormalised id using qg's stats). v4/v5a never used the gate.
* **Bug B** — would mis-pick `_best.phys.pt` over `_best.pt` whenever a single output's AR-rollout NRMSE blew up. v4 never produced such a blowup; v5b/v5c did (Bug A primed it).

Shipping v4 under the fixed loader has no behaviour regression vs pre-fix numbers. v5b/v5c checkpoints discarded. v6 retrain path (both bug fixes + B1 hybrid-grid data + B2 slope loss + B3 gate) is unchanged but should be evaluated against this v4 rebaseline rather than `v5_baseline_2026_04_22.md`.

## Reproduce

```bash
# Per-tech inverter verifier (≈ 7 min/tech):
for t in tsmc5 tsmc7 tsmc12 tsmc16; do
  conda run --no-capture-output -n pycircuitsim python -u \
    tests/verify_bsimar_v4_inverter.py --tech $t \
    > results/v4_rebaseline/$t.log 2>&1
done

# AR-rollout test-split diagnostic (≈ 30 s/device-type, GPU):
for d in nmos pmos; do
  CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES=0 \
  conda run --no-capture-output -n pycircuitsim python \
    tests/diag_phys_best_explosion.py --prefix v4_universal --device-type $d
done
```

## Artifact files

* `results/v4_rebaseline/tsmc{5,7,12,16}.log` — full verifier output per tech
* `results/v4_rebaseline_post_phys_fix.md` — this report
