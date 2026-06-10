# S4 = R0.2 — `NN_SYMMETRIC_CAPS=1` SC-only re-test on post-P0 code (V6.4.7, 2026-06-10)

Zero-code env-gated probe (`s4_switchcap_symcaps.log`), all 4 techs, repaired
R0.1 droop gate, post-P0 frame.

## Result: KILLED for SC — the V6.4.5 charge win trades for catastrophic hold drift

| tech | ChgErr% off / on | Droop \|dn−ng\| off / on | NRMSE% off / on | verdict |
|------|------------------|--------------------------|-----------------|---------|
| TSMC5 | 14.65 / **3.68** | 0.000 / **47.63 mV** | 36.1 / 31.7 | FAIL (droop) |
| TSMC7 | 3.06 / 1.76 | 2.208 / **30.73 mV** | 52.4 / 53.4 | FAIL (droop) |
| TSMC12 | 10.29 / 8.69 | 0.001 / **132.42 mV** | 40.2 / 54.3 | FAIL (charge+droop) |
| TSMC16 | 13.14 / **1.38** | 0.002 / **136.90 mV** | 45.4 / 59.9 | FAIL (droop) |

- The recorded V6.4.5 improvement ("SC charge improved on every failing tech")
  replicates qualitatively post-P0 — TSMC5 and TSMC16 charge would now PASS —
  **but symcaps destroys hold-phase retention**: 30–137 mV of genuine
  simulated drift (no NR truncation; convergence clean), 40–170× the 0.1 %
  VDD allowance. Invisible under the old auto-passing droop gate; caught
  immediately by the S3 repair (which thereby proves its worth).
- **Per-circuit env-gated shipping is OFF the table** — there is no
  parameterization where 100+ mV hold drift is acceptable for a
  sample-and-hold cell. D1 now reads: symmetric caps dead for RO (V6.4.5)
  AND dead for SC (V6.4.7 S4).

## Ownership evidence extracted (feeds S5 dump + P5/P7 EV)

The charge-transfer level is highly sensitive to the cap stamping (TSMC16
13.14 → 1.38 % from symmetrization alone) ⇒ the SC **sample-phase charge
error is substantially charge-model/cap-owned**, not id-owned — consistent
with P0 (id frame fix) having left SC unchanged. TSMC12's residual 8.69 %
under symcaps suggests mixed ownership there. The asymmetric trans-cap terms
(cgd ≠ cdg) that symmetrization averages away are load-bearing during hold —
the NN's cap asymmetry error expresses as hold-phase drift, the cap
symmetry error as sample-phase charge error. S5's dump should therefore
record the full cap matrix (cgg/cgd/cgs/cdg/cdd) along the sample window AND
the phi falling edge, vs OSDI.
