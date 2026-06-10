# S3 = R0.1 — switchcap droop sub-gate repair (V6.4.7, 2026-06-10)

**Change (`tests/verify_complex_switchcap.py`):** hold-droop sub-gate replaced
`|dn−ng|/|ng| ≤ 10 % (nan-skip when |ng| ≤ 1 µV)` with
`|dn−ng| ≤ max(10 %·|ng|, 1e-3·VDD)`. Summary column renamed
`DroopErr%` → `Droop%alw` (now % of the allowance, ≤100 passes); result-dict
key `droop_err_pct` → `droop_pct_of_allowance` (units changed — renamed so no
consumer silently misreads old-vs-new numbers).

## Why the old gate was broken in BOTH directions (measured)

Post-P0 values (`s2_logs/switchcap.log`): NG droops −0.19/−0.33/−1.18/−1.50 µV
(raw csv) on TSMC5/7/12/16; DN droops 0.000/2.207/0.001/0.002 mV.

- **Unpassable direction:** TSMC12/16 droop "FAIL" at 206.9 %/241.2 % on
  |dn−ng| ≈ 2–3 µV — the relative gate demanded 19–150 nV agreement, below
  BOTH solvers' tolerances (NGSPICE RELTOL·V+VNTOL ≈ 0.43 mV/point here).
- **Auto-pass hole:** TSMC7's 2.208 mV DN droop — the largest absolute
  disagreement on the board — auto-PASSED via the `|ng| > 1e-6` nan-guard.
  The 1e-6 threshold sliced through four same-scale (sub-µV) NG readings,
  auto-passing two and assigning nV-level demands to the others.

## The floor is principled (adversarial review, agent aa7fac731530a695b)

Verdict: **CORRECTION (net tightening) — not a loosening, not engineered.**
- The droop is a two-point difference; each point is honest only to
  RELTOL·V_hold+VNTOL. Formal bound 2·(1e-3·V_hold+1e-6) =
  0.61/0.90/0.87/0.84 mV per tech; the 1e-3·VDD floor (0.65/0.75/0.80/0.80
  mV) matches within ±20 %. An engineered-to-pass floor would have needed
  ≥3e-3·VDD to keep TSMC7 passing; 1e-3 flips it to FAIL.
- Waveforms bit-identical pre/post repair (same charge errs, droops,
  NRMSE) — only the verdict logic changed, and the only flip is PASS→FAIL
  on the worst cell. The inverse of the E3 false-PASS pattern.
- The old relative branch never passed any recorded cell (zero
  discrimination); its entire pass record was the nan-guard hole.
- No programmatic consumers of the old key (scorer has no SC cells;
  `baseline_v6_4_4.json` has no droop fields).

## Recorded caveats (from the review)

1. **Blind spot:** the floor admits up to ~50 nA off-state leakage error
   (C·ΔV/T = 100 fF·0.8 mV/1.7 ns) ≈ 500–4000× the true 11–88 pA leak. The
   V6.4.5-era 26 µV phantom-leak artifact class would sail under it. This is
   a circuit-level gate; subthreshold fidelity is gated by P3/P4 work, not
   here.
2. An optional better-grounded floor is `2·(RELTOL_NG·V_hold+VNTOL_NG)` per
   tech — same order; revisit only if a variant runs V_hold ≪ VDD.

## Census and headline restatement (REQUIRED by review finding 5)

| | RO | opamp | SRAM butterfly | SC | headline |
|---|---|---|---|---|---|
| V6.4.4 canonical, old gate | 3 | 1 | 4 | 1 | 9/16 |
| **V6.4.4 canonical, repaired gate** | 3 | 1 | 4 | 0 | **8/16** |
| Post-P0 (S2), old gate | 3 | 2 | 4 | 1 | 10/16 |
| **Post-P0+R0.1 (current), honest** | 3 | 2 | 4 | 0 | **9/16** |

TSMC7 SC: charge 3.06 % PASS, droop 2.208 mV = 294 % of 0.750 mV allowance →
FAIL. TSMC5/12/16 fail charge (14.65/10.29/13.14 % vs ≤5 %) regardless.
The plan's success bar ">9/16" now counts honest cells; the S8 re-freeze
(`baseline_v6_4_7_pre.json`) must use this gate.

Validation: full 4-tech re-run `s3_switchcap_repaired_gate.log` (census 0/4) +
post-rename TSMC7 single-tech run (FAIL at 294 % of allowance, new column).
