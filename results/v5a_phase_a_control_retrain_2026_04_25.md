# v5a Phase A — Control Retrain Gate Report (2026-04-25 → 2026-04-27)

**Plan:** `docs/superpowers/plans/2026-04-24-v5-inverter-accuracy.md` (Phase A4).
**Branch:** `feat/bsimar-v5-phase-a` (HEAD `4cd1b18` at gate time, plus the
verifier `--directnet-prefix`/`SKIP_VTC` patches in this commit).
**Owner:** a4-retrainer (sub-agent of team-lead).
**Goal:** retrain the four production checkpoints (DirectNet NMOS/PMOS +
BSIMAR Transformer NMOS/PMOS) on the **existing** `universal_{nmos,pmos}.npz`
data with the trimmed pipeline (Phase A1+A2+A3 commits) and confirm
inverter-verifier accuracy stays within **±1 pp NRMSE per cell** of the v4
baseline. This is the merge-gate for the entire Phase A trim.

---

## TL;DR — **GATE: FAIL**

| Metric                                       | Result                          |
|----------------------------------------------|---------------------------------|
| Cells WITHIN ±1 pp of v4 baseline            | **5 / 30** (17 %)               |
| Cells IMPROVED beyond −1 pp                  | 8 / 30                          |
| Cells REGRESSED beyond +1 pp                 | **17 / 30** (57 %)              |
| Cells DNF / hung                             | 2 (TSMC7 VTC AR + DN)           |
| Worst single regression                      | DN NMOS DC TSMC12: **+12.40 pp**|
| Best single improvement                      | AR VTC TSMC5: **−9.79 pp**      |
| DN catastrophic failures                     | TSMC12 inverter transient: NR diverged at t=1e-11 s |

**Bottom line.** The trimmed pipeline produced excellent training-space
metrics (DN val ≈3.5× lower than v4; TF phys-NRMSE 0.063 % overall) but
the inverter-verifier physical-space behaviour collapsed in several
cells, in particular DirectNet NMOS DC on TSMC12/16 (+12 pp regression).
The strict ±1 pp gate is not met. **Recommend NOT merging Phase A as-is**
and proceeding to deeper diagnosis before re-trim.

---

## 1. Trained checkpoints

All four checkpoints land under
`external_compact_models/bsimar/checkpoints/` (gitignored):

| Checkpoint                                  | Wall-clock | Best val (norm)         | Phys NRMSE / R²            |
|---------------------------------------------|------------|-------------------------|----------------------------|
| `v5a_dn_universal_nmos_best.pt`             | ~6 h       | **0.000270**            | 0.008 % / 1.0000           |
| `v5a_dn_universal_pmos_best.pt`             | ~6 h       | **0.000276**            | 0.008 % / 1.0000           |
| `v5a_universal_nmos_best.phys.pt`           | ~26 h      | TF 0.002622 / AR 0.002826 | **0.063 %** / 0.9998 (avg) |
| `v5a_universal_pmos_best.phys.pt`           | ~26 h      | TF 0.002548 / AR 0.002662 | **0.070 %** / 0.9997 (avg) |

DN val loss dropped **3.5× below v4** (v4 NMOS 0.00167 → v5a 0.000270).
TF physical NRMSE is comparable to v3 medium (0.063–0.070 %, R² ≥ 0.9997
across 17 tech×variant test slices). On paper the trained models look
better than v4 — the regression is purely an *inference-distribution*
problem (see §5).

Architecture is unchanged from v4 (7-dim input + tech-code embedding,
18 codes, ASAP7 excluded). Loss / norm hard-wired by the trim:
- **DN:** MAE + per-target LDS (after A1 removed `DirectLoss` and
  `ChargeConsistencyLoss`).
- **TF:** MAE + per-target LDS (after A2 removed `SignConsistency` +
  `Boundary`; after A3 collapsed the 3-axis LDS stack to per-target).

Training logs: `results/v5a_retrain_runs/{dn,tf}_{nmos,pmos}.log`.

---

## 2. Verifier delta table (v5a − v4, percentage points)

Reference: v4 baseline `results/v5_baseline_2026_04_22.md` (commit
`706bcdd`, captured 2026-04-22). Verifier:
`tests/verify_bsimar_v4_inverter.py --tech <tech>
--checkpoint-prefix v5a_universal --directnet-prefix v5a_dn_universal`.

Convention: **positive = regression, negative = improvement**. Cells
within ±1 pp are bolded as **WITHIN**; |Δ| ≥ 1 pp marked
**REGRESS** / **IMPROVE**.

### TSMC5 (8 cells)

| Test          | v4 (%) | v5a (%) | Δ (pp)   | Verdict     |
|---------------|--------|---------|----------|-------------|
| NMOS DC AR    |  4.59  |  7.06   | +2.47    | REGRESS     |
| NMOS DC DN    |  6.20  |  7.57   | +1.37    | REGRESS     |
| PMOS DC AR    |  5.97  |  7.43   | +1.46    | REGRESS     |
| PMOS DC DN    |  7.74  |  5.28   | −2.46    | IMPROVE     |
| VTC AR        | 13.96  |  4.17   | **−9.79**| IMPROVE     |
| VTC DN        | 10.73  | 16.74   | +6.01    | REGRESS     |
| Tran AR       | 12.13  |  3.16   | **−8.97**| IMPROVE     |
| Tran DN       |  3.75  | 12.10   | +8.35    | REGRESS     |

### TSMC7 (6 cells; VTC AR+DN are DNF — verifier hung in NR loop)

| Test          | v4 (%) | v5a (%) | Δ (pp)   | Verdict     |
|---------------|--------|---------|----------|-------------|
| NMOS DC AR    | 14.72  | 17.58   | +2.86    | REGRESS     |
| NMOS DC DN    | 15.79  | 16.95   | +1.16    | REGRESS     |
| PMOS DC AR    |  3.06  |  6.66   | +3.60    | REGRESS     |
| PMOS DC DN    |  6.53  |  4.97   | −1.56    | IMPROVE     |
| VTC AR        | 19.15  |   —     | DNF      | (verifier hung 2 attempts — see §4) |
| VTC DN        | 18.14  |   —     | DNF      | (verifier hung 2 attempts — see §4) |
| Tran AR       |  9.14  |  8.78   | −0.36    | **WITHIN**  |
| Tran DN       |  6.80  |  8.94   | +2.14    | REGRESS     |

### TSMC12 (8 cells; DN transient = NaN, NR diverged at t=1e-11 s)

| Test          | v4 (%) | v5a (%) | Δ (pp)   | Verdict     |
|---------------|--------|---------|----------|-------------|
| NMOS DC AR    |  9.95  | 10.23   | +0.28    | **WITHIN**  |
| NMOS DC DN    |  3.71  | 16.11   | **+12.40**| REGRESS    |
| PMOS DC AR    | 13.72  |  8.54   | −5.18    | IMPROVE     |
| PMOS DC DN    | 12.17  | 19.70   | +7.53    | REGRESS     |
| VTC AR        |  4.10  |  6.17   | +2.07    | REGRESS     |
| VTC DN        |  4.86  |  5.89   | +1.03    | REGRESS     |
| Tran AR       |  6.78  |  6.39   | −0.39    | **WITHIN**  |
| Tran DN       |  4.86  |  NaN    | DNF      | REGRESS (NR diverged) |

### TSMC16 (8 cells)

| Test          | v4 (%) | v5a (%) | Δ (pp)   | Verdict     |
|---------------|--------|---------|----------|-------------|
| NMOS DC AR    |  8.96  |  9.49   | +0.53    | **WITHIN**  |
| NMOS DC DN    |  3.11  | 15.25   | **+12.14**| REGRESS    |
| PMOS DC AR    | 13.48  |  9.14   | −4.34    | IMPROVE     |
| PMOS DC DN    | 12.40  | 19.14   | +6.74    | REGRESS     |
| VTC AR        |  3.40  |  5.12   | +1.72    | REGRESS     |
| VTC DN        |  5.67  |  5.33   | −0.34    | **WITHIN**  |
| Tran AR       |  7.51  |  6.46   | −1.05    | IMPROVE     |
| Tran DN       |  7.86  |  5.97   | −1.89    | IMPROVE     |

### Aggregate

|                              | AR cells | DN cells | All     |
|------------------------------|----------|----------|---------|
| WITHIN ±1 pp                 | 4        | 1        | **5**   |
| IMPROVED (Δ < −1 pp)         | 5        | 3        | **8**   |
| REGRESSED (Δ > +1 pp)        | 6        | 11       | **17**  |
| DNF                          | 1        | 2        | **3**   |
| **Total measurable**         | 15       | 15       | **30**  |

---

## 3. Raw verifier SUMMARY blocks

```
TSMC5  v5a (verify_tsmc5.log)         7/8 PASS
TSMC7  v5a (verify_tsmc7_skipvtc.log) 4/6 PASS  (VTC skipped after 2 hangs)
TSMC12 v5a (verify_tsmc12.log)        4/7 PASS  (DN tran NaN omitted)
TSMC16 v5a (verify_tsmc16.log)        6/8 PASS
```

For comparison, v4 baseline result lines:
`TSMC5 6/8 · TSMC7 4/8 · TSMC12 6/8 · TSMC16 6/8`. Total threshold-PASS
cells: v4 **22/32**; v5a **21/29 measurable** (lower absolute count and
strictly worse on PMOS DC + NMOS DC for TSMC12/16 DN).

Logs preserved under `results/v5a_retrain_runs/`.

---

## 4. Anomalies and operator notes

1. **TSMC7 inverter-VTC verifier hangs** (both attempts).
   - First run (`verify_tsmc7.log`): 60 min wall, ~7 h CPU at 99 %, log
     stuck at `--- Test 3: Inverter VTC ---`. Killed.
   - Second run (`stdbuf -oL` + `python -u`): same pattern at 36 min
     wall, ~4.5 h CPU at 1318 %. Killed.
   - Third run with `SKIP_VTC=1` env var (added in this commit) cleared
     in 9 min and produced the 4/6 SUMMARY shown above.
   - Hypothesis: v5a TSMC7 NMOS DC NRMSE is 17.58 % (AR) / 16.95 % (DN),
     up from 14.72 % / 15.79 % in v4. Bistable VTC sweep + degraded
     model → DCSolver enters a non-converging NR loop. Same failure
     mode that drove the *v5 plan §15-17* TSMC7-NMOS investigation.
   - **Action item:** the verifier should grow a per-test wall-clock
     timeout. Filed as A4 follow-up.

2. **TSMC12 inverter-transient DN diverged** (`Failed to converge at
   t=1.00e-11 s … final max delta 1.33e+02`). Same checkpoint passes
   AR transient at 6.39 %. Root cause: DN PMOS DC error 19.70 % means
   the DN inverter pull-up landscape is wrong enough that the very
   first DC OP / first transient step cannot land. v4 DN PMOS DC was
   12.17 % and converged at 4.86 % NRMSE.

3. **DN val-loss vs phys-NRMSE inversion.**
   The DN training-space val loss dropped from v4 0.00167 → v5a
   0.000270 (3.5× better) and per-tech NRMSE in normalised space sits
   at 0.007–0.010 % — yet inverter-verifier NMOS DC NRMSE on TSMC12/16
   *worsened* by 12 pp. This is the textbook signature of a
   training/inference *distribution* mismatch: the LHS training grid
   under-weights exactly the strong-inversion + saturation plateau that
   the verifier's uniform Id-Vgs sweep hits hardest (same mechanism
   diagnosed in `v5_improvement_plan_2026_04_21.md §17` for TSMC7
   NMOS). The trim from `DirectLoss` (13-output weighted MSE) to
   `MAE + per-target LDS` further down-weights that plateau because
   LDS density is low there. The new floor is what is exposing it.

4. **Cosmetic `conda run … failed (See above)` lines** at the end of
   each verifier log are the harmless `conda run` post-exec quirk
   already noted in the v4 baseline file. The Python process always
   completed and printed a well-formed SUMMARY.

5. **3-h cadence reporting** missed an 18-hour window during the
   monitoring phase (operator error: status posted to chat output
   instead of `SendMessage`). Corrected mid-run; future cadence is
   driven by `ScheduleWakeup` chained at 60-min cap (runtime cap
   prevents true 3 h intervals).

---

## 5. Verdict and recommendation

### Gate: **FAIL**

The strict ±1 pp cell-by-cell gate from the Phase A4 plan is not met:
**13 / 30 measurable cells violate the gate** (17 regress > +1 pp,
8 improve > −1 pp; 5 within). 2 cells DNF on TSMC7 VTC; 1 cell DNF on
TSMC12 DN transient. We cannot declare the trimmed pipeline accuracy-
neutral relative to v4.

### Mixed signal — what *did* work

- **AR transient on TSMC5 dropped from 12.13 % → 3.16 %** (−8.97 pp).
- **AR VTC on TSMC5 dropped from 13.96 % → 4.17 %** (−9.79 pp).
- AR PMOS DC on TSMC12/16 improved by 5.18 / 4.34 pp.
- DN tran on TSMC16 improved by 1.89 pp.

### What broke

- DN NMOS DC TSMC12/16 regressed by **+12 pp** (the dominant failure).
- DN PMOS DC TSMC12/16 regressed by **+7 pp**.
- DN VTC TSMC5 regressed by +6 pp; DN tran TSMC5 regressed by +8 pp.
- TSMC7 verifier no longer terminates in VTC.

### Recommended next step

**Do not merge Phase A as-is.** Two paths forward, in priority order:

1. **(Most likely root cause)** — Reinstate the `DirectLoss` 13-output
   weighted MSE from A1 *only* for DN. The trim was justified by
   "deletes dead code" but the audit conflated dead-on-paper with
   actually-shaping-the-trained-distribution. The `DirectLoss` weights
   `id` and `gds` more aggressively than per-target LDS does, which
   matches what the verifier measures on NMOS / PMOS DC. Specifically:
   keep A2 + A3 (Transformer-only trims, where physical-space metrics
   look intact) but back out A1's DN-side change. Re-run the four
   trainings and the verifier on the same gate.

2. **(Independent fix)** — Add a uniform-Vgs sweep augmentation block
   (see `v5_improvement_plan_2026_04_21.md §17`) to the training data
   so LHS density no longer hides the saturation plateau. This is the
   root-cause fix and would benefit v4 as well; it is also a larger
   change than Phase A intended.

If #1 succeeds, the trim ships with one of the three sub-commits
(A1) reverted; A2 + A3 stand. If #1 fails, fall back to **revert all
of Phase A** and reopen the v5 plan with the §17 distribution-
mismatch hypothesis as the new starting point.

### Out-of-scope safety check

- All four v5a checkpoints persist at the paths above and are
  gitignored.
- v4 (`v4_*`) and v5_trim_smoke_* checkpoints were not touched at any
  point during retrain (verified by mtime comparison).

---

## Appendix A — Wall-clock summary

| Phase                              | Wall-clock                          |
|-----------------------------------|-------------------------------------|
| 4 parallel trainings              | ~26 h (TF dominates; DN ~6 h each) |
| 4 verifier runs                   | ~63 min (tsmc5 14 m + tsmc12 19 m + tsmc16 19 m + tsmc7 9 m skipvtc) |
| Plus tsmc7 hangs (kill twice)     | +96 min wasted                      |

## Appendix B — File inventory

- Training logs: `results/v5a_retrain_runs/{dn,tf}_{nmos,pmos}.log`.
- Verifier logs: `results/v5a_retrain_runs/verify_tsmc{5,12,16}.log`,
  `verify_tsmc7_skipvtc.log` (and the killed `verify_tsmc7.log` stub).
- Checkpoints (gitignored): `external_compact_models/bsimar/checkpoints/v5a_*`.
- Verifier patches in this commit:
  - `tests/verify_bsimar_v4_inverter.py`: added `--directnet-prefix`
    + `--checkpoint-prefix` alias and `SKIP_VTC=1` env var bypass for
    Test 3 (needed to recover TSMC7 transient after VTC hang).
- This report: `results/v5a_phase_a_control_retrain_2026_04_25.md`.
