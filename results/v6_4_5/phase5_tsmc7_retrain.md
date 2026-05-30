# Phase 5 — TSMC7 retrain under the V6.4.5 multi-circuit objective

**Date:** 2026-05-29  •  **Branch:** `feat/v6.4.4`  •  **Status:** **KILL — no candidate closes ring_osc TSMC7; nothing shipped**
**Env:** training `OMP_NUM_THREADS=4`, GPUs 1/2/3; scoring `OMP_NUM_THREADS=1`, 8-way.

## What was run

Per plan §Phase 5 (16-seed stock + 8-seed mono, TSMC7 N+P), scored with the
Phase-3 multi-circuit vector. The 4 seeds `{42,7,17,123}` already existed on
disk as the V6.4.2 Phase-7a artifacts and were re-scored for free in Phase 3;
the remaining seeds were trained fresh under the `v6_4_5_p5_tsmc7_*` naming:

- **stock**: 16 seeds `{42,7,17,123}` (existing) + `{1,2,3,5,11,13,31,47,73,91,137,211}` (12 new) → 24 N+P trainings (12 new fresh).
- **mono** (`--monotonic`): 8 seeds `{42,7,17,123}` (existing) + `{1,2,3,5}` (4 new) → 8 new fresh trainings.

32 fresh trainings (`scripts/run_v6_4_5_p5_train.sh`), then diagonal scoring
(same-seed N&P, `scripts/score_p5_diagonal.sh`). All trainings used the
unchanged `data/datasets/tsmc7_{nmos,pmos}.npz` (2026-05-15). The isolated
scorer (`BSIMAR_CHECKPOINT_DIR`) never touched the canonical slots — verified
by sha against the pre-Phase-5 backup `/tmp/v6_4_4_backup_20260529`.

GPU budget: 32 trainings × ~20–30 min, 6-wide ≈ ~2.0 h wall (well under cap).

## Result — RO is a systematic ~9 % bias, not seed/recipe-addressable

DirectNet RO period clusters tightly while NG = 46.64 ps. Reaching the ≤ 5 %
gate needs a period ≤ 48.97 ps — **below the entire observed range**.

### New 16-seed sweep (diagonal, sorted by RO)

| candidate    | inv VTC % | inv tran % | RO %  | SRAM resid | opamp gain | flat(g<10) |
|--------------|----------:|-----------:|------:|-----------:|-----------:|:----------:|
| stock_s31    |      3.52 |       1.10 |  8.21 |      0.292 |        0.0 |     1      |
| stock_s11    |      3.14 |       1.52 |  9.05 |      0.280 |      118.2 |     0      |
| stock_s13    |      3.59 |       1.33 |  9.29 |      0.293 |        0.0 |     1      |
| stock_s5     |      2.22 |       1.14 |  9.72 |      0.282 |      363.6 |     0      |
| stock_s2     |      3.52 |       1.21 | 10.23 |      0.294 |        0.0 |     1      |
| mono_s5      |      3.17 |       1.49 | 10.60 |      0.267 |      394.3 |     0      |
| mono_s3      |      4.12 |       1.15 | 10.71 |      0.300 |        0.0 |     1      |
| stock_s1     |      1.96 |       1.36 | 10.79 |      0.291 |        0.0 |     1      |
| stock_s137   |      2.43 |       1.38 | 10.81 |      0.276 |        0.0 |     1      |
| stock_s91    |      3.79 |       1.48 | 10.95 |      0.283 |        0.0 |     1      |
| mono_s1      |      5.50 |       1.39 | 11.20 |      0.360 |      363.0 |     0      |
| stock_s211   |      2.25 |       1.12 | 11.74 |      0.306 |        0.0 |     1      |
| mono_s2      |      3.94 |       1.07 | 12.24 |      0.295 |        0.0 |     1      |
| stock_s73    |      4.29 |       1.57 | 12.76 |      0.293 |        0.0 |     1      |
| stock_s47    |      2.24 |       1.20 | 12.78 |      0.300 |        0.0 |     1      |
| stock_s3     |      1.75 |       1.24 | 13.84 |      0.297 |        0.0 |     1      |

### Combined landscape (this sweep + Phase-3 existing pool + baseline)

| Pool                            | RO range %  | best RO % | best **feasible** RO % (gain ≥ 10) |
|---------------------------------|------------:|----------:|-----------------------------------:|
| V6.4.4 baseline (canonical)     |        8.98 |      8.98 |                               8.98 |
| Phase-7a existing (8: 4 stock+4 mono) | 8.96–11.53 | 8.96 (mono_s42) |                  8.96 |
| New 16-seed sweep               | 8.21–13.84 | 8.21 (stock_s31, **infeasible**) |       9.05 (stock_s11) |

- **Best overall RO = 8.21 %** (`stock_s31`) — but its opamp **collapsed**
  (gain 0 → flat, infeasible). The lowest RO is bought by a degenerate model.
- **Best feasible RO = 9.05 %** (`stock_s11`) — **worse** than the V6.4.4
  baseline (8.98 %).
- Only **3/16** new candidates even clear the hard gates (gain ≥ 10): the
  retrained TSMC7 opamp collapses to zero gain in 13/16 cases (the same
  collapse iter-2 saw with P7-stock).
- inv VTC stays 1.75–5.50 % (the seed *does* move VTC by ~3.8 % pp) yet RO
  barely budges — the RO period is insensitive to the seed-level variation
  that moves VTC. The ~9 % RO error is a **model/architecture/data bias**, not
  init variance and not the stock↔mono recipe choice.

## Kill criteria (plan §Phase 5) — both fire

> *"If neither stock-16-seed nor mono-8-seed clears the Phase-3 hard gates AND
> beats V6.4.4 by ≥ 1 complex-circuit pass for TSMC7 → stop."*

No candidate clears RO ≤ 5 %; the best feasible candidate (9.05 %) is *worse*
than V6.4.4 (8.98 %). **KILL.**

> *"If best TSMC7 candidate regresses the inverter VTC MaxErr by > +5 mV …
> reject."*

Moot — no candidate passes the RO gate to begin with.

Greedy n/p-seed mixing was **not** run: the diagonal RO floor (8.96 % feasible,
period ~50.8 ps) is a systematic ~2 ps above the gate; cross-mixing seeds
cannot bridge a systematic-bias gap that the full 24-stock + 8-mono diagonal
never approached.

## Decision

- **KILL Phase 5.** No TSMC7 retrain ships. V6.4.5 complex-circuit headline
  stays **9/16** (= V6.4.4). Canonical checkpoints untouched (sha-verified).
- The `v6_4_5_p5_tsmc7_*` checkpoints (32 stems) are kept on disk as
  reproducible dead-end evidence; they do NOT match the parser resolver
  pattern (`tsmc7_dn_*`), so they are inert and never auto-loaded.

## What this rules out, and what remains

- **Seed lottery does not move TSMC7 RO.** 24 stock + 8 mono seeds, RO floor
  ~8.2 % (infeasible) / ~9.0 % (feasible). The lottery moves inverter VTC, not
  the RO period.
- **The `--monotonic` recipe does not help RO** (mono floor 8.96 %, ties stock
  baseline; new mono seeds 10.6–12.2 %).
- **Retraining frequently collapses the TSMC7 opamp** (gain → 0 in 13/16),
  confirming the iter-2 P7-stock finding as a recurring retrain failure mode.
- **Closing ring_osc TSMC7 ≤ 5 % requires an architectural change**, not a
  retrain: V6.4.6 split-head (plan §6, spectral-norm id-head + softplus
  cap-head to fix the cap shape that loads the RO period) or Track B
  B7 (physics-anchored skeleton) / B9 (hard-monotone lattice).

## Files

- `scripts/run_v6_4_5_p5_train.sh`, `scripts/score_p5_diagonal.sh`
- `logs/v6_4_5_p5/` (per-training logs, batch + score driver logs)
- `results/v6_4_5/phase3_logs/search_TSMC7_p5.jsonl` (16 new candidates)
- `results/v6_4_5/phase3_logs/search_TSMC7.jsonl` (Phase-3 existing pool + baseline)
