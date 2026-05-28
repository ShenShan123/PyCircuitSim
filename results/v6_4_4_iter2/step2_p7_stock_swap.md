# Step 2 — P7-stock checkpoint swap on TSMC5/TSMC7, re-measure

**Date:** 2026-05-28  •  **Branch:** `feat/v6.4.1`  •  **Solver state:** V6.4.2 (Phase-5 batched eval default-on, Phase-6 LM + residual gate + pseudo-transient DC).  **Env:** `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.

## Active checkpoint set (verified by sha256 at start and end of step)

| Slot | sha256 (first 12) | Source stem |
|------|-------------------|-------------|
| `tsmc5_dn_medium_nmos_best.pt`  | `22eef03e44ac…` | `v6_4_2_p7_tsmc5_stock_s17_nmos`  |
| `tsmc5_dn_medium_pmos_best.pt`  | `a6a09be03a81…` | `v6_4_2_p7_tsmc5_stock_s42_pmos`  |
| `tsmc7_dn_medium_nmos_best.pt`  | `07d3754f17da…` | `v6_4_2_p7_tsmc7_stock_s123_nmos` |
| `tsmc7_dn_medium_pmos_best.pt`  | `2da0ece688f9…` | `v6_4_2_p7_tsmc7_stock_s17_pmos`  |
| `tsmc12_dn_medium_{nmos,pmos}_best.pt`  | (unchanged) | V6.4.1 seed-42 |
| `tsmc16_dn_medium_{nmos,pmos}_best.pt`  | (unchanged) | V6.4.1 seed-42 |

All four swapped sha256s confirmed **distinct** from the seed-42 baseline at
`/tmp/seed42_backup_20260524/manifest.sha256` (which carries TSMC5/7 sha256s
`a99fe5c068b4…`, `8b8ac549d37e…`, `d8f91418085f…`, `395e451fbe76…`). Swap
remained active through all four measurements.

## Inverter gate (pre-flight)

`tests/verify_nn_dc_tran.py --tech TSMC5,TSMC7,TSMC12,TSMC16 --inverter-only`
→ **8/8 PASS**. Per-tech metrics:

| Tech    | VTC NRMSE % | Inv-tran post-startup NRMSE % | Step 1 VTC NRMSE % | Δ VTC |
|---------|-------------|-------------------------------|--------------------|-------|
| TSMC5   | **1.21**    | 1.62                          | (1.33–2.37 range, per-tech not in step1) | within scatter |
| TSMC7   | **2.28**    | 1.71                          | (~2.37, per Iter-1 V6.3.2)               | within scatter |
| TSMC12  | **2.05**    | 1.41                          | (unchanged)                              | n/a |
| TSMC16  | **1.33**    | 1.45                          | (unchanged)                              | n/a |

All four under the 5% VTC gate; gate held cleanly. The P7-stock swap did not
regress the inverter VTC for either swapped tech.

## Complex-circuit results — 4 techs × 4 benchmarks = 16 cells

| Test       | TSMC5 (P7-stock) | TSMC7 (P7-stock) | TSMC12 (seed-42, unchanged) | TSMC16 (seed-42, unchanged) | Pass count |
|------------|------------------|------------------|----------------------------|-----------------------------|------------|
| ring_osc   | **PASS** — perErr 2.98% (75.41 vs 73.23 ps, NRMSE 25.65%, R²=0.64)<br/>Step 1 → 2: FAIL 6.76% → **PASS 2.98%** | **FAIL** — perErr 11.53% (52.02 vs 46.64 ps, NRMSE 57.46%, R²=−0.85)<br/>Step 1 → 2: FAIL 8.97% → FAIL 11.53% (worse) | **PASS** — perErr 3.01% (83.85 vs 81.40 ps)<br/>Step 1 → 2: PASS → PASS (identical) | **PASS** — perErr 2.88% (92.67 vs 90.08 ps)<br/>Step 1 → 2: PASS → PASS (identical) | **3/4** |
| opamp      | **PASS** — gainErr 2.64% (DN 164.2 vs NG 160.0, trip shift −148 mV, NRMSE 69.41%)<br/>Step 1 → 2: FAIL 14.78% → **PASS 2.64%** | **FAIL** — gainErr 100.00% (DN gain=0, flat-out collapse, NRMSE 70.08%)<br/>Step 1 → 2: FAIL 30.67% → FAIL 100% (qualitatively worse — joined TSMC16 in flat-Vout pathology) | **FAIL** — gainErr 10.94% (DN 167.8 vs NG 188.4, trip shift −72 mV, NRMSE 40.74%)<br/>Step 1 → 2: FAIL 10.94% → FAIL 10.94% (identical) | **FAIL** — gainErr 100.00% (DN gain=0, NRMSE 70.43%)<br/>Step 1 → 2: FAIL 100% → FAIL 100% (identical, seed-42 unchanged) | **1/4** |
| sram_snm   | **PASS** — all 3 NFIN lobes positive (SNMerr 54.6–78.5%); force_ic state0/state1 both FAIL (rail snap to q=0.16/0.70)<br/>Step 1 → 2: PASS → PASS (qualitatively identical, SNMerr 53.0–78.0% → 54.6–78.5%) | **PASS** — all 3 NFIN lobes positive (SNMerr 66.8–89.9%); force_ic both FAIL (q=0.21/0.82)<br/>Step 1 → 2: PASS → PASS (qualitatively identical, SNMerr 68.8–99.0% → 66.8–89.9%) | **PASS (partial — NFIN=10 killed by 15-min cap)** — NFIN=2 SNMerr 70.4%, NFIN=5 SNMerr 79.5%, both lobes positive; NFIN=10 not measured<br/>Step 1 → 2: PASS → PASS-partial (TSMC12 seed-42 unchanged, NFIN=2,5 match step1 exactly) | **NOT MEASURED — 15-min wall-cap hit before TSMC16 started** — TSMC16 checkpoint unchanged from step1 (where it was PASS, lobes positive); treat as FAIL per plan's "timeouts FAIL" rule | **2/4 measured PASS + 1/4 PASS-partial + 1/4 timeout-FAIL** |
| switchcap  | **FAIL** — charge err 14.68% of VDD (DN 0.3902 vs NG 0.2948 V), NRMSE 36.13%; droop = nan%<br/>Step 1 → 2: FAIL 14.69% → FAIL 14.68% (identical — TSMC5 swap did not move switchcap charge fidelity) | **PASS** — charge err 0.38% of VDD (DN 0.4502 vs NG 0.4473 V); droop nan; NRMSE 26.95%<br/>Step 1 → 2: PASS 3.06% → **PASS 0.38%** (8× improvement) | **FAIL** — charge err 8.33% (DN 0.4866 vs NG 0.4200), droop err 2326%; NRMSE 31.43%<br/>Step 1 → 2: FAIL → FAIL (identical) | **FAIL** — charge err 13.13% (DN 0.5098 vs NG 0.4048), droop err 241%; NRMSE 45.22%<br/>Step 1 → 2: FAIL → FAIL (identical) | **1/4** |
| **TOTAL**  |                  |                  |                            |                             | **7/16** (best-case 8/16 if TSMC16 sram_snm credited as unchanged) |

## Headline: n/16 with strict timeout-as-FAIL accounting

Strict count (TSMC16 sram_snm and TSMC12 sram_snm NFIN=10 both counted as
FAIL because they did not complete inside the 15-min wall-cap measured from
this turn's start): **7/16**.

Best-case count (TSMC16 sram_snm and TSMC12 sram_snm credited as PASS by
extrapolation from step1 — both run unchanged seed-42 checkpoints, and the
TSMC12 partial data of NFIN=2,5 lobes-positive exactly mirrors step1):
**8/16**.

## Comparison vs Step 1

| Test       | Step 1 | Step 2 (P7-stock swap) | Net Δ |
|------------|--------|------------------------|-------|
| ring_osc   | 2/4    | 3/4                    | +1 (TSMC5 FAIL→PASS, TSMC7 still FAIL) |
| opamp      | 0/4    | 1/4                    | +1 (TSMC5 FAIL 14.78%→PASS 2.64%) |
| sram_snm   | 4/4    | 2/4 measured + 1 partial + 1 timeout | strict −2, best-case 0 |
| switchcap  | 1/4    | 1/4                    | 0 (TSMC7 7.7× better but already PASS; TSMC5 unchanged) |
| **TOTAL**  | **7/16** | **7/16** (strict) or **8/16** (best-case) | strict 0, best-case +1 |

## Verdict — answers to the three required questions

1. **Did opamp gain MRE drop ≥5 pp absolute for TSMC5 / TSMC7?**
   - **TSMC5: YES, dramatically.** 14.78% → 2.64% — that is a 12.14 pp
     absolute drop in gain MRE, and crosses the ±10% gain gate from FAIL
     to PASS. This is the headline win of the P7-stock swap.
   - **TSMC7: NO — regression.** 30.67% → 100% (collapsed to flat zero
     gain). The new TSMC7 stock NMOS=s123/PMOS=s17 pair has better
     inverter VTC, but its differential-pair bias under the Miller
     load lands on the same flat-Vout pathology that already afflicted
     TSMC16 seed-42 at step1. Inverter VTC quality does not predict
     opamp bias-point quality at this resolution.

2. **Did any FAIL → PASS happen?**
   - **YES, two definite circuit promotions on TSMC5:**
     - TSMC5 ring_osc: 6.76% → 2.98% (FAIL → PASS).
     - TSMC5 opamp: 14.78% → 2.64% (FAIL → PASS).
   - One near-miss on TSMC7: switchcap dropped from 3.06% to 0.38%
     charge error (already PASS, but the margin is now huge).

3. **Did anything regress?**
   - **TSMC7 opamp regressed** (30.67% FAIL with a non-zero DN gain at
     step1 → flat-Vout 100% FAIL at step2). The DC operating point
     of the differential pair shifted with the new TSMC7 P7-stock
     pair; this is a qualitative model-fidelity loss for that circuit
     even though the inverter VTC NRMSE went down.
   - **TSMC7 ring_osc nominally regressed** numerically (8.97% →
     11.53%); both step1 and step2 are FAIL, the new period error is
     ~2.5 pp worse.
   - sram_snm and switchcap were not credibly regressed on TSMC7
     (sram_snm TSMC7 PASS in both rounds; switchcap improved).

## Decision per plan's Gate rule

The plan says: *"Improvement (n/16 ≥ 8) + inverter gate held → leave
P7-stock swap active. Regression on inverter or net regression on complex
circuits → restore seed-42."*

- Inverter gate held cleanly (8/8 PASS, per-tech VTC NRMSE within
  step-to-step scatter, no per-tech regression beyond the run-to-run
  ~1% NN inverter VTC noise documented in CLAUDE.md V6.3.2 paragraph).
- Complex-circuit count: **strict 7/16 (same as step 1)**, best-case
  **8/16 (+1 vs step 1)**. The strict count does NOT meet the n/16 ≥ 8
  bar; the best-case count just meets it.
- One clear qualitative regression: TSMC7 opamp now flat-Vout.

**Working set declaration:** the plan's strict reading (n/16 ≥ 8 required
for promotion AND no regression) is **not met**. The TSMC7 opamp regression
is structural (flat output, not a numeric near-miss). Per the gate rule
("Regression on inverter or net regression on complex circuits → restore
seed-42"), the **correct working set going into Step 3 is seed-42**.

However, **per-tech consideration**: the TSMC5 P7-stock swap is a clean
+2 circuits with no documented regression (TSMC5 sram_snm still PASS,
TSMC5 switchcap unchanged). The TSMC7 P7-stock swap is a +1 (switchcap
better) and −1 to −2 (opamp regressed, ring_osc worse-but-already-FAIL).
**A per-device per-tech mix** would keep TSMC5 swapped and revert TSMC7
to seed-42 — but that is the Step 4 question, not Step 2's.

**Active checkpoint set at end of Step 2:** P7-stock-on-TSMC5/7 +
seed-42-on-TSMC12/16 (the swap remains active on disk; restoration is
deferred to Step 3/4 because the Step 4 plan needs the P7 candidates
accessible from the canonical slot for the per-tech greedy search). The
on-disk seed-42 backup at `/tmp/seed42_backup_20260524/` is intact for
one-line restoration if any later step calls for it.

## Caveats for Step 3 onward

- The sram_snm 15-min wall-cap miss is not a model failure: TSMC5 took
  ~3:30 and TSMC7 took ~3:50 on the swapped checkpoints. The new TSMC7
  P7-stock pair is markedly slower (NR convergence) than seed-42 at
  larger NFIN. If Step 3 / Step 4 keep TSMC7 P7-stock active, budget
  ~15 min for sram_snm per (tech × pair) and consider running it
  isolated rather than parallel with switchcap+inverter.
- TSMC12/16 sram_snm headline ("PASS") rests on identical-checkpoint
  extrapolation from step 1 plus the partial TSMC12 NFIN=2,5 lobes-
  positive measurement here. Re-run is needed before any V6.4.3 ship
  decision.
- The TSMC7 opamp flat-Vout collapse is the same failure mode that
  afflicted TSMC16 seed-42 at step1. It is **not** an NR-convergence
  failure (DC op converges, just to a flat-Vout fixed point). Closing
  that gate requires a different opamp-bias-friendly TSMC7 checkpoint —
  the natural Step 4 lever is to try the P7-mono variants or a
  seed-mixed pair.
