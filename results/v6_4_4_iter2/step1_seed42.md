# Step 1 — V6.4.1 seed-42 re-baseline on V6.4.2 solver

**Date:** 2026-05-27  •  **Branch:** `feat/v6.4.1`  •  **Checkpoints:**
`external_compact_models/bsimar/checkpoints/tsmc{5,7,12,16}_dn_medium_{nmos,pmos}_best.pt`
(V6.4.1 seed-42, unchanged on disk).  **Solver state:** V6.4.2 (Phase-5
batched eval default-on, Phase-6 LM damping + residual gate +
pseudo-transient DC).  **Env:** `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`,
`NN_BATCHED_EVAL` left at default, `NN_SYMMETRIC_CAPS` left at default (off).

## Inverter gate (pre-flight)

`tests/verify_nn_dc_tran.py --tech TSMC5,TSMC7,TSMC12,TSMC16 --inverter-only`
→ **8/8 PASS**. VTC NRMSE 1.33–2.37%, inverter-tran post-startup NRMSE
1.09–1.56%. Gate held; baseline is valid.

## Complex-circuit results (4 techs × 4 benchmarks = 16 cells)

| Test       | TSMC5 | TSMC7 | TSMC12 | TSMC16 | Pass count |
|------------|-------|-------|--------|--------|------------|
| ring_osc   | FAIL — period err 6.76% (78.18 vs 73.23 ps, NRMSE 72.12%, R²=−1.84) | FAIL — period err 8.97% (50.82 vs 46.64 ps, NRMSE 54.37%, R²=−0.65) | PASS — period err 3.01% (83.85 vs 81.40 ps, NRMSE 33.83%, R²=0.34) | PASS — period err 2.88% (92.67 vs 90.08 ps, NRMSE 29.49%, R²=0.49) | **2/4** |
| opamp      | FAIL — gain err 14.78% (DN 136.3 vs NG 160.0, trip shift −120 mV, NRMSE 71.08%) | FAIL — gain err 30.67% (DN 213.5 vs NG 163.4, trip shift −134 mV, NRMSE 59.86%) | FAIL — gain err 10.94% (DN 167.8 vs NG 188.4, trip shift −72 mV, NRMSE 40.74%) | FAIL — gain err 100.00% (DN gain=0 i.e. flat, trip shift +150 mV, NRMSE 70.43%) | **0/4** |
| sram_snm   | PASS — all 3 NFIN lobes positive (SNMerr 53.0–78.0%); force_ic state1/state0 both FAIL (rail snap to q=0.16/0.70 etc.) | PASS — all 3 NFIN lobes positive (SNMerr 68.8–99.0%); force_ic both FAIL (q=0.82/0.23) | PASS — all 3 NFIN lobes positive (SNMerr 70.4–79.5%); force_ic both FAIL (q=0.87/0.19) | PASS — all 3 NFIN lobes positive (SNMerr 51.4–82.9%); force_ic both FAIL (q=0.87/0.20) | **4/4** |
| switchcap  | FAIL — charge err 14.69% of VDD (DN 0.3902 vs NG 0.2948 V), NRMSE 36.13%; droop = 0 both sides (gate skipped) | PASS — charge err 3.06% (DN 0.4703 vs NG 0.4473 V); NG droop ≈ 0 so droop gate auto-passed; NRMSE 52.38% | FAIL — charge err 8.33% (DN 0.4866 vs NG 0.4200 V); droop err 2326% (DN 0.026 mV vs NG −0.001 mV); NRMSE 31.43% | FAIL — charge err 13.13% (DN 0.5098 vs NG 0.4048 V); droop err 241% (DN 0.002 mV vs NG −0.001 mV); NRMSE 45.22% | **1/4** |
| **TOTAL**  |       |       |        |        | **7/16** |

## Comparison to iter-1 V6.3.1 baseline

| Test       | V6.3.1 (iter-1) | V6.4.1 seed-42 / V6.4.2 solver (this step) |
|------------|-----------------|--------------------------------------------|
| ring_osc   | 2/4 | 2/4 (no change — same TSMC12/16 PASS, TSMC5/7 still over the ±5% period gate) |
| opamp      | 0/4 | 0/4 (no change — TSMC12 now closer at 10.94% but still just over the ±10% gain gate; TSMC16 collapsed to gain=0, the new worst cell) |
| sram_snm   | 4/4 | 4/4 (butterfly-positive gate held; the force_ic full-6T convergence probe lands on the same non-rail NN fixed point at q≈0.16–0.20 / qb≈0.70–0.87 as V6.3.1) |
| switchcap  | 1/4 | 1/4 (TSMC7 still the only PASS; TSMC12 charge err improved 8.3% vs prior worse, TSMC16 worsened) |
| **TOTAL**  | **7/16** | **7/16** |

**Net change: 0 circuits.** The V6.4.2 solver upgrades (Phase 5 batched
eval, Phase 6 LM + residual gate + pseudo-transient DC) are
accuracy-neutral on this harness — they buy NR robustness, not model
fidelity, exactly as the V6.4.2 CHANGELOG predicted. The V6.4.1 seed-42
checkpoint regression (vs V6.4 best-of-N on inverter VTC MaxErr) also
does not move any complex-circuit gate at this resolution: every gate
that was open at V6.3.1 is open here, on the same cells, with
qualitatively similar failure modes (opamp gain miss, RO period >5%,
switchcap charge level offset, SRAM `force_ic` settling on the NN's flat
inter-rail fixed point).

The opamp TSMC16 collapse to gain=0 (vs V6.3.1's 14.78%-ish failure mode
across all four techs) is the only qualitative shift worth flagging —
the seed-42 TSMC16 pmos checkpoint pushes the Miller input pair so far
off its bias that the DC operating point lands on a flat-Vout solution.
Step 2 (swap to V6.4 best-of-N) is the natural test of whether that is a
pure checkpoint-quality issue.
