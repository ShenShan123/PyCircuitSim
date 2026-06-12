# Phase 3 — Multi-circuit selection objective (scorer + re-score)

**Date:** 2026-05-29  •  **Branch:** `feat/v6.4.4`  •  **Status:** Done (scorer built, validated, re-calibrated)
**Env:** `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `pycircuitsim` conda env.

## What shipped (infrastructure, no model change)

- `scripts/eval_v6_4_5_candidate.py` — scores ONE `(tech, nmos-stem, pmos-stem)`
  candidate against the V6.4.5 multi-circuit vector. Runs the four
  micro-benchmarks in one process and prints a machine-parseable `RESULT`
  JSON line.
  - **Isolation, not swapping.** Unlike `eval_v6_4_1_pair.py` (which swapped
    the canonical slots and restored from `/tmp/v6_4_1_phase4_backup`, now
    GONE), this copies the candidate into a private `mkdtemp` dir *under the
    canonical `tsmc{X}_dn_medium_{dev}` stem names* and points
    `BSIMAR_CHECKPOINT_DIR` at it. Vocab-scope detection (Rule 19, keys off
    the `tsmc{5,7}_dn_*` stem) still fires correctly; the real `checkpoints/`
    dir is **never mutated**, so candidates run concurrently and safely.
  - Per-process harness output dirs are also redirected into the isolated
    dir, so concurrent candidates don't race on shared netlist / NGSPICE
    temp / trace files.
- `scripts/v6_4_5_search.py` — parallel (`--jobs N`) diagonal / greedy search
  over candidate stems, ranking by the selection objective below.

**Cost:** ~13 min wall per candidate single-threaded (the un-batched DirectNet
ring-oscillator + inverter transients dominate). 152 cores → 8-way parallel
keeps the wall time of a 9-candidate scan at ~2 batches.

## Scoring vector (per candidate, per tech)

```
inv_vtc_nrmse         hard gate <= 5 %
inv_tran_post_nrmse   hard gate <= 5 %
opamp_flat_flag       hard gate (see re-calibration below)
ring_osc_period_err   primary Pareto objective (TSMC7 open gate <= 5 %)
sram_rail_snap_resid  Pareto objective = max(|q-VDD|,|qb-0|)/VDD at force_ic state-1
```

## Re-calibration (plan §Phase 3: "if the scorer rejects the V6.4.4 mix, fix the thresholds")

The plan defined `opamp_flat_flag = 1 if |Vout_op - VDD/2| > 0.3*VDD` at the
center common-mode `.op`. **That metric flags every tech, including the one
PASSING opamp (TSMC5).** An open-loop high-gain opamp railed at the exact
center common-mode is normal whenever the input pair has any offset — Vout at
the center sweep point sits at a rail for TSMC5 (the gate-passing cell,
gain_err 2.64 %) just as it does for the collapsed TSMC16. So the
vout-at-center form is not a usable discriminator and would reject the V6.4.4
mix by construction.

**Re-calibrated:** `opamp_flat_flag = 1 if peak |dVout/dVin| (gain) < 10`.
This captures the *real* "flat/collapsed" failure — the iter-2 TSMC7 P7-stock
regression that drove gain_err 30.67 % → 100 % (gain → 0). Re-derived from the
already-recorded `opamp_gain` (no re-run).

## Validation — re-score the V6.4.4 mix (canonical slots)

| Tech   | inv VTC % | inv tran % | RO %  | SRAM resid | opamp gain | flat(g<10) |
|--------|----------:|-----------:|------:|-----------:|-----------:|:----------:|
| TSMC5  |      1.21 |       1.64 |  2.98 |      0.251 |        164 |     0      |
| TSMC7  |      2.84 |       1.10 |  8.98 |      0.302 |        370 |     0      |
| TSMC12 |      2.05 |       1.44 |  3.01 |      0.240 |        168 |     0      |
| TSMC16 |      1.33 |       1.47 |  2.88 |      0.249 |          0 |     1      |

- Hard gates clear for **TSMC5/7/12** (inv VTC/tran ≤ 5 %, gain ≥ 10). The
  scorer accepts the V6.4.4 mix where it is supposed to be good.
- **TSMC16 flat=1** is correct, not a mis-tune: TSMC16's opamp is a known
  V6.4.4 *failing* cell (Phase-1 rebaseline: gain 0.0, gain_err 100 %).
- VTC NRMSE shows the documented ~±0.5 % run-to-run scatter on the high-gain
  trip (TSMC7 2.84 % here vs 2.37 % in Phase 1 — both PASS; pin
  `OMP_NUM_THREADS=1`).
- RO / SRAM reproduce Phase-1 exactly (TSMC7 RO 8.98 %, SRAM resid 0.30 = the
  q ≈ 0.81 non-rail attractor).

**Kill criterion (Phase 3):** *"No retrain begins until the scorer accepts
V6.4.4."* — MET after the opamp re-calibration.

## Free re-score of the on-disk TSMC7 Phase-7a pool (feeds Phase 5)

All 4 stock + 4 mono seeds (`v6_4_2_p7_tsmc7_{stock,mono}_s{42,123,7,17}`)
scored against the new vector (diagonal, same-seed N&P). See
`phase5_tsmc7_retrain.md` for the table and the conclusion — in short, the
TSMC7 RO error is a **systematic ~9 % bias** (DN periods 50.8–52.0 ps vs NG
46.64), not seed/recipe-addressable: best = `mono_s42` 8.96 %, essentially the
V6.4.4 baseline (8.98 %); stock is the *worse* cluster (~11.4 %).

## Files

- `scripts/eval_v6_4_5_candidate.py`, `scripts/v6_4_5_search.py`
- `results/v6_4_5/phase3_logs/search_TSMC7.jsonl` (9-candidate diagonal)
- `results/v6_4_5/phase3_logs/baseline_{TSMC5,TSMC12,TSMC16}.out`
- `results/v6_4_5/phase3_logs/diagonal_TSMC7.log`
