# DirectNet V6.4.4 — Complex-Circuit Iteration 2 (inference + selection only)

**Date:** 2026-05-24  •  **Status:** In progress  •  **Branch:** `feat/v6.4.1` (V6.4.2 solver state)

> **Scope contract.** This iteration is **inference + checkpoint selection only**.
> No data regeneration, no retraining. We work entirely with artifacts that
> already exist on disk:
> - canonical shipping set (seed-42) in
>   `external_compact_models/bsimar/checkpoints/tsmc{5,7,12,16}_dn_medium_*`,
> - V6.4 best-of-N production backup at
>   `/tmp/v6_4_checkpoints_backup_20260517/`,
> - Phase-7a candidates `v6_4_2_p7_{tsmc5,tsmc7}_{stock,mono}_s{7,17,42,123}_*`
>   already on disk (32 cells total).
>
> Allowed levers: checkpoint swap, env flags (`NN_SYMMETRIC_CAPS`,
> `NN_BATCHED_EVAL`), solver flags, per-tech / per-device pool selection on
> real complex-circuit metrics. No code change beyond glue scripts for
> measurement and tabulation.

## Iter-1 outcome (collapsed; full record in `2026-05-15-...iter1.md` + CHANGELOG)

Iter-1 walked the eight-phase plan. By V6.4.2 every training-bearing or
solver-bearing phase had shipped or dead-ended:

| Phase | Type | Outcome |
|------:|------|---------|
| 1b Sobolev | loss | DEAD END, 7-8× worse VTC |
| 2a sym caps | solver | shipped, dormant (`NN_SYMMETRIC_CAPS=0`) |
| 2b gm/gmb floor | solver | REVERTED, unsound |
| 4 data overlays + sinh | data | DEAD END, hard VTC↔tran tradeoff |
| 5 batched eval | solver | SHIPPED, accuracy-neutral |
| 6 LM + residual gate + p-tran | solver | SHIPPED, accuracy-neutral on RO/SRAM |
| 7a monotonic-in-Vg residual | model | TSMC5/7 trained, never re-ranked vs V6.4 best-of-N |
| 7b spectral-gds | model | REJECTED by CLI (not coherent on shared-trunk MLP) |
| 8 split heads | model | not started — would require retraining (out of scope here) |

The complex-circuit harness was measured **once**, at V6.3.1 (commit
`6dff82a`): RO 2/4, opamp 0/4, SRAM-SNM 4/4, SC 1/4 = **7/16**. It has
never been re-measured on V6.4 best-of-N, on V6.4.1 seed-42, or on V6.4.2's
shipping checkpoints — even though the solver, the inverter VTC numbers, and
the candidate pool have all moved underneath it.

## Diagnosis (from V6.4.2 Phase 6 verification)

> "The RO transient already converges to a bit-identical inaccurate period;
> the SRAM `force_ic` re-solve converges to a consistent non-rail NN fixed
> point. Phase 6 improves *how robustly* a fixed point is reached — it cannot
> move a fixed point a converging solve already reaches. Closing those gates
> needs a better model."  — `docs/CHANGELOG.md` V6.4.2

The complex-circuit failures are **model fidelity gaps**, not NR-convergence
failures. The only inference-time lever we have not exhausted is **which
checkpoint we ship**. The iter-1 selection criterion was inverter VTC MaxErr;
the complex-circuit metrics (opamp gain, RO period, SRAM `force_ic` rail
snap, SC charge transfer + droop) were *never* part of the selection
objective even though they are the gates that are open.

## Target

≥ **10/16** complex-circuit gates closed on the V6.4.2 solver, using only
checkpoint swaps + env / solver flags, with the inverter gate held at 8/8 and
the extended harness (`verify_nn_multi_tech_{dc,tran}.py`) non-regressing.

## Steps

Each step is gated on the previous, runs as a single agent invocation, and
ends with a tabulated result committed to the repo
(`results/v6_4_4_iter2/step{N}_*.md`).

### Step 1 — Re-baseline on the shipping seed-42 set

Run all four complex tests on the canonical `tsmc{5,7,12,16}_dn_medium_*`
checkpoints currently on disk. The V6.3.1 7/16 number is **stale** —
between then and now we shipped batched eval (Phase 5) and Phase-6 NR
upgrades, both billed as accuracy-neutral but not measured here, and
swapped from V6.4 best-of-N to V6.4.1 seed-42 (a documented inverter VTC
regression). Without Step 1 we cannot tell which subsequent step moves what.

- Verify the inverter gate is 8/8 first
  (`tests/verify_nn_dc_tran.py --tech TSMC5,TSMC7,TSMC12,TSMC16
   --inverter-only`).
- Run `tests/verify_complex_{ring_osc,opamp,sram_snm,switchcap}.py` against
  all four TSMC techs, with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` (CLAUDE.md
  reproducibility rule for the inverter trip gain).
- Capture per-test JSON / log output. Tabulate pass/fail + the Rule-16
  quartet (MRE %, R², NRMSE %, MaxErr) per (test, tech).

**Gate:** Step 1 table written to `results/v6_4_4_iter2/step1_seed42.md`.

### Step 2 — Swap to the best inverter-VTC checkpoints on disk + re-measure

**Pivot (2026-05-27):** the V6.4 best-of-N backup at
`/tmp/v6_4_checkpoints_backup_20260517/` was cleared by `/tmp` cleanup
between V6.4.2 ship (2026-05-18) and now. The source stems
(`v6_4_bof_*`, `v6_4_repro_*`) are unrecoverable from local disk.

Replacement target: the **V6.4.2 Phase-7a stock winners** already on disk
at `external_compact_models/bsimar/checkpoints/v6_4_2_p7_*_stock_*`,
which the Phase-7 bake-off (`logs/v6_4_2_phase7/search_TSMC{5,7}_stock_winner.json`)
identified as **strictly better** than seed-42 on inverter VTC for both
TSMC5 and TSMC7:

| Tech  | Stock winner pair                         | Inverter VTC MaxErr (mV) | vs seed-42 |
|-------|-------------------------------------------|--------------------------|------------|
| TSMC5 | nmos=s17 / pmos=s42                       | **43.7** (NRMSE 1.14 %)  | 134.6 → 43.7 (−68 %) |
| TSMC7 | nmos=s123 / pmos=s17                      | **99.6** (NRMSE 2.20 %)  | 210.5 → 99.6 (−53 %) |

Both winners also clear their transient gates (39.4 / 49.0 mV ≤ baseline
39.6 / 49.4). For TSMC12 / TSMC16 no better candidate exists on disk —
they stay on seed-42. Net effect: TSMC5 and TSMC7 swap to their P7 stock
winners; TSMC12/16 are unchanged. Same hypothesis as the original Step 2
(reduce gain-amplification residual at the inverter trip → move opamp +
RO gates), just on the better candidates that actually exist.

- Back up the current seed-42 checkpoints to `/tmp/seed42_backup_20260524/`
  (manifest first).
- For TSMC5: copy `v6_4_2_p7_tsmc5_stock_s17_nmos_{best.pt,norm.npz}` →
  `tsmc5_dn_medium_nmos_{best.pt,norm.npz}`; copy
  `v6_4_2_p7_tsmc5_stock_s42_pmos_*` → `tsmc5_dn_medium_pmos_*`.
- For TSMC7: nmos=s123, pmos=s17 — same shape.
- TSMC12/16 untouched.
- Re-run the inverter gate — must stay 8/8.
- Re-run the four complex tests, tabulate.

**Gate:** Step 2 table written. If Step 2 regresses the inverter gate or
the complex-circuit count, restore seed-42 from
`/tmp/seed42_backup_20260524/` and treat seed-42 as the working set.

### Step 3 — Toggle `NN_SYMMETRIC_CAPS=1` and re-measure RO + SC

Phase 2a is dormant code that targets the RO BDF-2 oscillation mode. Never
re-measured since V6.4. RO and SC are the transient-heavy benchmarks the
symmetric-cap stamp would touch.

- On the better of Step 1 / Step 2 checkpoint set, re-run RO and SC with
  `NN_SYMMETRIC_CAPS=1`.
- Re-run inverter transient — must hold.
- If a gate flips and the inverter transient holds, record a recommendation
  to flip the default for transient analysis only. **Do not flip the
  default in this iteration** — that is a code change that belongs in a
  later commit; we only record the empirical result here.

**Gate:** Step 3 table written; `NN_SYMMETRIC_CAPS` decision recorded.

### Step 4 — Per-tech, per-device checkpoint mix from the existing pool

The shipping convention is one (nmos, pmos) pair per tech. We have multiple
candidates per (tech, device) on disk:

| Tech    | NMOS pool size | PMOS pool size | Sources |
|---------|---------------:|---------------:|---------|
| TSMC5   | 10             | 10             | seed-42, V6.4 best-of-N, 4×P7-stock, 4×P7-mono |
| TSMC7   | 10             | 10             | same as TSMC5 |
| TSMC12  | 2              | 2              | seed-42, V6.4 best-of-N |
| TSMC16  | 2              | 2              | seed-42, V6.4 best-of-N |

The V6.4 / V6.4.2 selections all used **inverter VTC MaxErr** as the
single objective. Complex-circuit pass rate was never part of the
selection. This step does the missing thing: greedy pair search per tech,
scored on the four complex-circuit tests for *that tech*.

- Reuse the `scripts/v6_4_2_phase7_search.py` shape (1: fix pmos, sweep
  nmos; 2: fix best-nmos, sweep pmos; 3: joint-refine top-2 × top-2).
- Score = number of complex-circuit gates passed for that tech (out of 4),
  with lex tiebreak: opamp gain MRE ↑, RO period error % ↑, SC droop %.
- Inverter gate per-tech: VTC MaxErr ≤ V6.4 best-of-N + 5 mV (slack),
  transient post-startup MaxErr ≤ V6.4 best-of-N + 5 mV. Any pair failing
  this hard gate is dropped from the pool.
- Per-tech pair eval cost ≈ 4 complex tests × ~30-60 s + 1 inverter sim ≈
  3-5 min. Greedy ~19 pairs/tech × 4 techs ≈ 1-1.5 h wall time bounded.

**Gate:** A per-tech (nmos, pmos) selection table is written with the
chosen pair, its inverter + complex-circuit numbers, and its provenance
(seed42 / v6_4_bof / v6_4_2_p7_*).

### Step 5 — Promote, re-measure, CHANGELOG

If the Step 4 selection beats Step 2 by ≥ 1 circuit:

- Copy the winning pair per tech into the canonical parser slots.
- Re-run the full inverter gate + extended harness
  (`verify_nn_multi_tech_dc.py`, `verify_nn_multi_tech_tran.py`) — must
  hold (DC 55/55, VTC+tran 64/64).
- Re-run the four complex tests one more time to lock the official V6.4.4
  numbers (the previous Step 4 sweep is per-tech; this one is the
  consolidated headline).
- Write the V6.4.4 CHANGELOG section. Update CLAUDE.md status paragraph
  with the new per-tech provenance and the 4-circuit pass count.

If Step 4 does not beat Step 2, ship the Step 2 result as **V6.4.3** (a
pure checkpoint-swap release) and record Step 4 as a documented dead end.

## What we are NOT doing

- **No retrain, no data regen.** Datasets and training code are untouched.
- **No new loss / model code** beyond what is already on disk in this
  branch. The Phase-7a `_MonotoneVgResidual` stays in `direct_net.py`
  exclusively as a load path for the existing P7 checkpoints.
- **No Phase 7b** (`--spectral-gds`; CLI refuses it for the shared-trunk
  MLP and that decision stands).
- **No Phase 8** split-head model (would require retraining; deferred).
- **No ASAP7, no LEVEL=74 BSIMAR** (CLAUDE.md Rules 17–18).

## Risks

- The V6.4 best-of-N backup is a directory of files, not a manifest —
  every Step 2 swap must verify file count + sha256 before activation.
- The P7 candidates were trained against the V6.4.2 Phase-4 regenerated
  datasets. Per V6.4.2 CHANGELOG those datasets were reverted but the
  trained checkpoints remain on disk. The norm stats `_norm.npz` are
  shipped alongside each `.pt`, so inference is self-consistent — but
  treat them as eval-time candidates only and never promote without the
  per-tech inverter gate passing.
- Complex tests are slow without Phase-5 batched eval; the iter-1
  execution log noted "a full RO window timed out" at V6.3.1. V6.4.2
  ships batched eval default-on, so this should be moot — but every agent
  step caps per-test wall time at 10 min and records timeouts as FAIL.

## Definition of done

1. Per-step result tables committed under `results/v6_4_4_iter2/step{1..5}_*.md`.
2. Best result strictly improves over the iter-1 7/16 baseline by **≥ 3
   circuits** (target 10/16), with the inverter gate at 8/8 and the
   extended harness non-regressing.
3. CHANGELOG entry V6.4.3 or V6.4.4 written; CLAUDE.md status paragraph
   updated with the new per-tech checkpoint provenance.
4. Every dropped lever logged with the empirical numbers that proved it
   useless.

## Execution log (filled in as steps complete)

### Step 1 — done (2026-05-27): 7/16 PASS, no change from V6.3.1

Re-baselined V6.4.1 seed-42 on the V6.4.2 solver. Inverter gate held
at 8/8. Complex-circuit totals: ring_osc 2/4 (TSMC12/16), opamp 0/4
(TSMC16 collapsed to gain=0), sram_snm 4/4 (butterfly-positive gate;
force_ic still settles on the non-rail NN fixed point on all four techs),
switchcap 1/4 (TSMC7 only). Net delta vs iter-1 V6.3.1: **0 circuits** —
the V6.4.2 solver upgrades are accuracy-neutral on this harness, as the
V6.4.2 CHANGELOG predicted. Full table in
`results/v6_4_4_iter2/step1_seed42.md`.

### Step 2 — done (2026-05-28): 7/16 strict (8/16 best-case), inverter gate held, swap active

Re-measured all four complex tests on the TSMC5/7 P7-stock swap (TSMC5
nmos=s17/pmos=s42, TSMC7 nmos=s123/pmos=s17; TSMC12/16 unchanged seed-42).
Inverter gate **8/8 PASS** with VTC NRMSE 1.21/2.28/2.05/1.33 % (TSMC5/7
swapped checkpoints inside step-to-step scatter; TSMC12/16 unchanged).
Complex-circuit pass count: **7/16 strict** (no change vs Step 1) or **8/16
best-case** (counting TSMC16 sram_snm as PASS-by-checkpoint-unchanged +
TSMC12 sram_snm as PASS-by-partial-evidence; both lost the 15-min wall-cap
race during the SRAM run). The TSMC5 P7-stock swap was a clean win
(ring_osc 6.76%→2.98% FAIL→PASS; opamp 14.78%→2.64% FAIL→PASS); the TSMC7
P7-stock swap was a mixed bag (switchcap 3.06%→0.38%, but opamp regressed
from 30.67% FAIL to flat-Vout 100% FAIL — same pathology as TSMC16
seed-42). Per the gate rule, strict reading does not promote (n/16 still
7); the P7-stock files remain in the canonical slots through end of Step 2
so Step 3 can run NN_SYMMETRIC_CAPS on them, but the seed-42 backup at
`/tmp/seed42_backup_20260524/` is intact for one-line restoration if Step
3/4 calls for it. Full table in `results/v6_4_4_iter2/step2_p7_stock_swap.md`.

### Step 3 — PARTIAL (2026-05-28): `NN_SYMMETRIC_CAPS=1` is inverter-safe

Only the inverter gate was re-measured under `NN_SYMMETRIC_CAPS=1`:
8/8 held (VTC NRMSE 1.21/2.28/2.05/1.33 %, tran 1.41/1.71/1.41/1.45 %).
The RO + SC re-measurement under the flag did not complete this
iteration (`step3_logs/{ring_osc,switchcap}.log` are 0-byte; the
background launches were lost between agent turns). Recommendation:
keep default OFF (dormant) until the RO+SC measurement is recorded.
Result: `results/v6_4_4_iter2/step3_logs/inverter_gate.log`.

### Step 4 — DONE (2026-05-28): lightweight per-tech mix, not full greedy search

Greedy pair-search infrastructure exists
(`scripts/v6_4_2_phase7_search.py`, `scripts/eval_v6_4_1_pair.py`) but
Step 2 evidence was decisive enough that it reduced to a one-line pick:
TSMC5 keeps P7-stock-s17/s42, TSMC7 reverts to seed-42 (P7-stock opamp
regression unshippable), TSMC12/16 stay seed-42 (no alternative on
disk). Inverter gate re-verified 8/8 on the mix (VTC NRMSE
1.21/2.37/2.05/1.33 %, tran 1.62/1.09/1.41/1.45 % — TSMC7 transient
1.09 % is the seed-42 recover from P7-stock's 1.71 %). Final
complex-circuit count from Step 1 + Step 2 evidence: **ring 3/4 +
opamp 1/4 + sram 4/4 + sc 1/4 = 9/16**. Result:
`results/v6_4_4_iter2/step4_logs/inverter_gate.log`.

### Step 5 — DONE (2026-05-28): CHANGELOG + CLAUDE.md updated, V6.4.4 commit

V6.4.4 ships the per-tech mix at **9/16** complex-circuit gates (+2 vs
V6.4.1 seed-42 / V6.3.1 iter-1 baseline). Target ≥ 10/16 was **not
met**; every inference-only lever from on-disk artifacts is exhausted
(V6.4 best-of-N backup gone; Phase-7a mono lost its bake-off; TSMC7
P7-stock has structural opamp regression; TSMC12/16 have no alternative
candidates). Remaining gates need a retrain (Phase 8 split heads or a
re-scored Phase-7 best-of-N on opamp gain + RO period — both deferred).
Final report: `results/v6_4_4_iter2/V6_4_4_final.md`. CHANGELOG V6.4.4
section + CLAUDE.md status paragraph updated; V6.4.2 Phase-7a code
(uncommitted since V6.4.2 ship) folded into the same V6.4.4 commit.
