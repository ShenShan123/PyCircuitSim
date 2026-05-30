# V6.4.5 — Final report (Track A, all phases run, 9/16)

> **Merged outcome:** This is the **Track-A** report (no-ship). V6.4.5 ultimately
> **SHIPS at 10/16 via Track B's B8** (differentiable-sim TTFT closing TSMC7
> ring_osc) — see `results/v6_4_5_track_b/V6_4_5_track_b_final.md` and the unified
> V6.4.5 CHANGELOG entry. Track A's key contribution to the ship: it proved RO is
> not retrain-addressable (Phase 5, 32 trainings), which is exactly why B8's
> direct-period-optimisation was the lever that worked.

**Date:** 2026-05-29  •  **Branch:** `feat/v6.4.4` (Track A) → merged into `feat/v6.4.5`  •  **Status:** Track A no-ship; V6.4.5 ships via Track B B8.

## Outcome

V6.4.5 Track A completed **all five phases**. Phases 1, 2, 4 were the earlier
no-ship iteration; **Phases 3 and 5 were then executed** (2026-05-29, this
session). Phase 3 built and validated the multi-circuit scorer; Phase 5 ran
the full 16-seed stock + 8-seed mono TSMC7 retrain (32 fresh trainings) scored
under that vector. **No candidate closes ring_osc TSMC7 ≤ 5 %** — the RO error
is a systematic ~9 % model bias, not seed/recipe-addressable. **KILL.**

Complex-circuit headline stays **9/16** (= V6.4.4). Inverter gate 8/8.
Extended harness 55/55 + 64/64. Canonical checkpoints untouched (sha-verified
against `/tmp/v6_4_4_backup_20260529`).

| Phase | Verdict | Code change | Complex-circuit delta |
|-------|---------|-------------|:---------------------:|
| 1 — V6.4.4 re-baseline | reproduced exactly | none | — |
| 2 — Zero-code solver probes | all 3 killed/diagnostic | none | 0 |
| 3 — Multi-circuit scorer | **built + validated** (scorer accepts V6.4.4 mix after opamp re-calibration) | `scripts/eval_v6_4_5_candidate.py`, `scripts/v6_4_5_search.py` (infra) | 0 |
| 4 — Rule-15 `Ioff_rail` patch | **KILL** (inverter regress ~10×) | reverted | 0 |
| 5 — TSMC7-only retrain | **KILL** (best feasible RO 9.05 % > baseline 8.98 %; 32 trainings) | none shipped | 0 |
| 6 — V6.4.6 split-head | deferred in plan | — | — |

Per-phase gate files: `phase1_v6_4_4_rebaseline.md`, `phase2_zero_code_probes.md`, `phase3_multi_circuit_scorer.md`, `phase4_ioff_rail_sweep.md`, `phase5_tsmc7_retrain.md`.

## Phase 3 + 5 headline (this session)

- **Phase 3 scorer.** `eval_v6_4_5_candidate.py` scores `(inv_vtc_nrmse,
  inv_tran_post_nrmse, ring_osc_period_err, sram_rail_snap_resid,
  opamp_flat_flag)` per candidate, using `BSIMAR_CHECKPOINT_DIR` isolation
  (never mutates canonical slots → parallel-safe). **Re-calibration:** the
  plan's `opamp_flat_flag = |Vout_center − VDD/2| > 0.3·VDD` flagged *every*
  tech including the PASSING TSMC5 opamp (a railed center common-mode is
  normal for a high-gain open-loop opamp); redefined to `gain < 10` (the real
  collapse signal). Under it the V6.4.4 mix clears the hard gates for
  TSMC5/7/12; TSMC16 correctly reads flat (its opamp is a known fail cell).
- **Phase 5 retrain.** 24 stock + 8 mono TSMC7 seeds × 2 devices. Best overall
  RO = 8.21 % (`stock_s31`, opamp-collapsed → infeasible); best feasible RO =
  9.05 % (`stock_s11`), worse than V6.4.4 (8.98 %). 13/16 new candidates
  collapse the TSMC7 opamp to zero gain. The seed lottery moves inverter VTC
  (1.75–5.50 %) but not the RO period (DN ~50.8–53 ps vs NG 46.64) → systematic
  bias, not init variance. See `phase5_tsmc7_retrain.md`.

## Dead ends recorded (CLAUDE.md "always record the dead end proposal")

1. **`NN_SYMMETRIC_CAPS=1` does not move TSMC7 RO.** TSMC7 RO period err
   8.97 % unchanged. The 9 % period drift is not cap-asymmetry.

2. **`max_substeps=4` on RO does not move TSMC7 RO enough.** 8.97 % → 8.04 %
   at ~2× wall time. The 5 % gate stays missed. RO drift is not
   LTE-dominated.

3. **SRAM `force_ic` q ≈ 0.18 is a true NN attractor, not a poor warm
   start.** Warm-started from the NGSPICE butterfly lobe (`near_zero` ≈
   83–123 mV instead of literal 0), force_ic still settles at q ≈ 0.70–0.80
   on every tech.

4. **Plan's Rule-15 `Ioff_rail` formulation `Ioff_rail = max(|id_raw|, k·NFIN·1nA)` is
   unsound.** It doubles the conducting current at the rail, regressing the
   inverter VTC by 10× (NRMSE 1.21 % → 11.56 % at the smallest non-zero k).
   It also moves the SRAM attractor *further* from the rails (q goes from
   0.866 toward 0.375). The patch was reverted.

5. **TSMC7 RO is a systematic ~9 % bias — the seed/recipe lottery does not
   move it (Phase 5).** 16-seed stock + 8-seed mono TSMC7 retrain (32 fresh
   trainings) scored under the Phase-3 vector. Best overall RO 8.21 %
   (`stock_s31`, opamp-collapsed → infeasible); best feasible RO 9.05 %
   (`stock_s11`), *worse* than V6.4.4 (8.98 %). The seed moves inverter VTC
   (1.75–5.50 %) but not the RO period (DN ~50.8–53 ps vs NG 46.64). 13/16 new
   candidates collapse the TSMC7 opamp to zero gain. KILL — closing RO needs
   an architectural change, not a retrain. No checkpoint shipped.

6. **Phase-3 `opamp_flat_flag = |Vout_center − VDD/2| > 0.3·VDD` is not a usable
   discriminator (re-calibrated, not a dead end per se).** It flags every
   tech, including the *passing* TSMC5 opamp, because a high-gain open-loop
   opamp is railed at the exact center common-mode whenever the input pair has
   any offset. Redefined to `gain < 10` (the true collapse signal); under it
   the V6.4.4 mix clears the hard gates as required.

## What can move both open gates

Outside V6.4.5 scope, captured here for V6.4.6 planning:

- **Ring_osc TSMC7 (~9 % → ≤ 5 %)** — model-fidelity, **and now confirmed NOT
  retrain-addressable** (Phase 5 dead end above: 32 trainings, RO floor
  ~8.2 % infeasible / ~9.0 % feasible). Needs an architectural change: V6.4.6
  split-head or Track B B7/B9. Track B B5/B6 (OSDI-Jacobian distill / harvest
  retrain) remain unproven but are *different* levers than the seed sweep.

- **SRAM `force_ic` (0/8 → 4/8)** — model-fidelity. The q ≈ 0.18 attractor
  is intrinsic to the NN's cross-coupled inverter pair. The plan's
  inference-only Vds correction (Phase 4) is unsound; closing it requires
  one of:
  - **Track B B7** — closed-form skeleton + small NN residual; the
    subthreshold exp suppresses Id by ≥ 6 decades at `Vgs < Vth`,
    inheriting rail stability.
  - **V6.4.6 split-head** (plan §6) — spectral-norm Id head bounds the
    cross-coupled-loop-gain at the SRAM rail; softplus cap head removes the
    spurious negative Cgd that loads RO.
  - **Track B B9** — hard-monotone lattice removes spurious mid-rail
    equilibria by construction.

- An off-state floor patch with the *corrected* formula
  `Ioff_extra = max(Ioff_floor - |id_raw|, 0)` (additive only below floor)
  is the cheapest re-attempt at the inference-only path; it was NOT tried
  in V6.4.5 because the plan's exact wording was followed (drop the
  solution, do not silently reformulate).

## Working tree state at sign-off

```
On branch feat/v6.4.4
Untracked:
  scripts/eval_v6_4_5_candidate.py   # Phase-3 multi-circuit scorer
  scripts/v6_4_5_search.py           # parallel diagonal/greedy search
  scripts/run_v6_4_5_p5_train.sh     # Phase-5 TSMC7 retrain driver
  scripts/score_p5_diagonal.sh       # Phase-5 diagonal scoring driver
```

- `results/v6_4_5/` is gitignored (generated-artifact convention); the phase
  gate files live there untracked.
- The 32 `v6_4_5_p5_tsmc7_*` checkpoints are gitignored dead-end artifacts; the
  canonical `tsmc{5,7,12,16}_dn_medium_*` slots are sha-verified unchanged.
- V6.4.4 remains canonical. No model shipped. The Phase-3 scorer scripts are
  reusable infra (commit-or-discard deferred to the user).
