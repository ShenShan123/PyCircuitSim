# Phase 1 — V6.4.4 re-baseline on V6.4.5 measurement harness

**Date:** 2026-05-28  •  **Branch:** `feat/v6.4.4` (HEAD `801ac6e`)  •  **Status:** Done
**Env:** `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `pycircuitsim` conda env.
**Checkpoints loaded** (per `[NN-resolver]` lines): canonical V6.4.4 mix —
TSMC5 `tsmc5_dn_medium_{nmos,pmos}_best.pt` (P7-stock-mix), TSMC7
`tsmc7_dn_medium_*` (seed-42), TSMC12 `tsmc12_dn_medium_*` (seed-42), TSMC16
`tsmc16_dn_medium_*` (seed-42).

Kill criterion (§Phase 1): *"None. This is a clean re-baseline; record numbers and move on."*

## Headline

| Gate                     | V6.4.4 expected | Re-baseline measured |
|--------------------------|----------------:|---------------------:|
| Complex-circuit pass     |          9/16   |             **9/16** |
| Inverter gate            |          8/8    |              **8/8** |
| Extended DC harness      |         55/55   |            **55/55** |
| Extended transient harness |       64/64   |            **64/64** |

V6.4.4 reproduces exactly. The V6.4.5 plan numbers (3/4 RO PASS, TSMC7 RO 8.97 %
FAIL, SRAM force_ic 0/4) are confirmed.

## Complex-circuit gates (16 cells)

### Ring oscillator (§Benchmark 3a) — 3/4

| Tech   | NG per (ps) | DN per (ps) | PerErr % | NRMSE % |     R²  | Status |
|--------|------------:|------------:|---------:|--------:|--------:|:------:|
| TSMC5  |       73.23 |       75.41 |     2.98 |   25.65 |  0.6402 | PASS   |
| TSMC7  |       46.64 |       50.82 | **8.97** |   54.37 | -0.6525 | **FAIL** |
| TSMC12 |       81.40 |       83.85 |     3.01 |   33.83 |  0.3405 | PASS   |
| TSMC16 |       90.08 |       92.67 |     2.88 |   29.49 |  0.4917 | PASS   |

Gate: period err ≤ 5 %. TSMC7 reproduces V6.4.4's 8.97 % miss (CLAUDE.md
diagnosis: phase walk under BDF-2 + asymmetric caps along the trip).

### Miller opamp (§Benchmark 3b) — 1/4

| Tech   | NG gain | DN gain | GainErr % | Trip shift mV | NRMSE % | Status |
|--------|--------:|--------:|----------:|--------------:|--------:|:------:|
| TSMC5  |   160.0 |   164.2 |      2.64 |        -148.0 |   69.41 | PASS   |
| TSMC7  |   163.4 |   213.5 |     30.67 |        -134.0 |   59.86 | FAIL   |
| TSMC12 |   188.4 |   167.8 |     10.94 |         -72.0 |   40.74 | FAIL   |
| TSMC16 |   187.7 |     0.0 |    100.00 |        +150.0 |   70.43 | FAIL   |

Gate: |gain err| ≤ 10 %. TSMC5 stays on; TSMC16 collapses to flat-Vout (gain=0)
— matches V6.4.4 final report. **Out of V6.4.5 scope** but tracked.

### SRAM read-SNM (§Benchmark 3c) — 4/4 butterfly, 0/8 force_ic

Butterfly lobes all positive (gate PASS) across the 12 NFIN corners:

| Tech   | NFIN | NG SNM (mV) | DN SNM (mV) | SNMerr % | min(qb) mV | Positive |
|--------|-----:|------------:|------------:|---------:|-----------:|:--------:|
| TSMC5  |    2 |      189.2 |       337.7 |     78.5 |      477.6 |   yes    |
| TSMC5  |    5 |      194.2 |       305.4 |     57.3 |      431.9 |   yes    |
| TSMC5  |   10 |      196.1 |       303.2 |     54.6 |      428.8 |   yes    |
| TSMC7  |    2 |      187.4 |       373.0 |     99.0 |      527.5 |   yes    |
| TSMC7  |    5 |      184.8 |       311.9 |     68.8 |      441.0 |   yes    |
| TSMC7  |   10 |      184.4 |       332.7 |     80.4 |      470.4 |   yes    |
| TSMC12 |    2 |      239.1 |       407.4 |     70.4 |      576.1 |   yes    |
| TSMC12 |    5 |      230.6 |       413.8 |     79.5 |      585.2 |   yes    |
| TSMC12 |   10 |      226.9 |       406.2 |     79.1 |      574.4 |   yes    |
| TSMC16 |    2 |      235.5 |       356.6 |     51.4 |      504.3 |   yes    |
| TSMC16 |    5 |      229.7 |       405.8 |     76.6 |      573.9 |   yes    |
| TSMC16 |   10 |      228.7 |       418.4 |     82.9 |      591.8 |   yes    |

`force_ic` probe — every tech / state fails to rail-snap:

| Tech   | state1 (q=VDD, qb=0) | state0 (q=0, qb=VDD) |
|--------|----------------------|----------------------|
| TSMC5  | q=0.873  qb=0.196 — FAIL | q=0.196 qb=0.873 — FAIL |
| TSMC7  | q=0.867  qb=0.200 — FAIL | q=0.200 qb=0.867 — FAIL |
| TSMC12 | q=0.866  qb=0.192 — FAIL | q=0.192 qb=0.866 — FAIL |
| TSMC16 | q=0.865  qb=0.199 — FAIL | q=0.199 qb=0.865 — FAIL |

All four techs settle on the non-rail attractor (q ≈ 0.87, qb ≈ 0.20) — exactly
the V6.4.4 fingerprint. The cross-coupled DC equilibrium of the NN inverter
pair is genuinely inboard of the rails: off-leak Id at `(Vgs=0, Vds=VDD)` is
under-modelled.

### Switched-cap (§Benchmark 3d) — 1/4

| Tech   | NG chg V | DN chg V | ChgErr % | DroopErr % | NRMSE % | Status |
|--------|---------:|---------:|---------:|-----------:|--------:|:------:|
| TSMC5  |   0.2948 |   0.3902 |    14.68 |        nan |   36.13 | FAIL   |
| TSMC7  |   0.4473 |   0.4703 |     3.06 |        nan |   52.38 | PASS   |
| TSMC12 |   0.4200 |   0.4866 |     8.33 |     2325.7 |   31.43 | FAIL   |
| TSMC16 |   0.4048 |   0.5098 |    13.13 |      241.3 |   45.22 | FAIL   |

Gate: charge err + droop err. **Out of V6.4.5 scope** (CHANGELOG V6.4.2 dead
end); kept here as the regression budget.

### Complex-circuit headline: **9/16 PASS** = 3 (RO) + 1 (opamp) + 4 (SRAM butterfly) + 1 (SC).

## Inverter gate — 8/8

| Tech   | VTC NRMSE % | Tran NRMSE % | Status         |
|--------|------------:|-------------:|:--------------:|
| TSMC5  |        1.21 |         1.62 | PASS (both)    |
| TSMC7  |        2.37 |         1.09 | PASS (both)    |
| TSMC12 |        2.05 |         1.41 | PASS (both)    |
| TSMC16 |        1.33 |         1.45 | PASS (both)    |

Matches CLAUDE.md V6.4.4 numbers exactly: 1.21/2.37/2.05/1.33 (VTC), 1.62/1.09/1.41/1.45 (tran).

## Extended harness

- `verify_nn_multi_tech_dc.py`: **55/55 PASS** (all techs, all parametric sweeps).
- `verify_nn_multi_tech_tran.py`: **64/64 PASS** (all VDD / P-N-ratio / Cload / slew / PW sweeps).

## Decisions for Phase 2

- TSMC7 RO 8.97 % is the live target (gate ≤ 5 %).
- SRAM `force_ic` 0/8 — every tech lands at the q ≈ 0.87, qb ≈ 0.20 fixed point.
  Phase 2 probe (3): butterfly-lobe warm-start to discriminate "poor warm
  start" vs "true NN attractor".
- Inverter / extended baselines logged for Phase 2 regression compare.

No code changes were made in Phase 1.
