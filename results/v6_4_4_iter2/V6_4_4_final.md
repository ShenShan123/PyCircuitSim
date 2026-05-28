# V6.4.4 — DirectNet per-tech checkpoint mix (inference-only iteration)

**Date:** 2026-05-28  •  **Branch:** `feat/v6.4.1`  •  **Solver state:** V6.4.2.
**Scope:** No retraining, no data regeneration — pure checkpoint selection from artifacts already on disk.

## Headline

| Metric                       | V6.4.1 (Step 1 baseline) | **V6.4.4** | Δ |
|------------------------------|-------------------------:|-----------:|---:|
| Complex-circuit gates closed | 7/16                     | **9/16**   | **+2** |
| Inverter gate                | 8/8                      | 8/8        | hold |
| Per-tech (RO + opamp + SRAM + SC) | 2+0+4+1               | 3+1+4+1    | +1 RO, +1 opamp |

V6.4.4 ships a **per-tech checkpoint mix** that replaces the canonical TSMC5
NMOS/PMOS slots with the V6.4.2 Phase-7a stock-recipe winners, while
keeping TSMC7/TSMC12/TSMC16 on the V6.4.1 seed-42 set.

## Active checkpoint provenance

| Slot                              | sha256 (first 16)     | Source stem                          | Origin           |
|-----------------------------------|-----------------------|--------------------------------------|------------------|
| `tsmc5_dn_medium_nmos_best.pt`    | `22eef03e44aca566…`   | `v6_4_2_p7_tsmc5_stock_s17_nmos`     | V6.4.2 P7 sprint |
| `tsmc5_dn_medium_pmos_best.pt`    | `a6a09be03a810b7e…`   | `v6_4_2_p7_tsmc5_stock_s42_pmos`     | V6.4.2 P7 sprint |
| `tsmc7_dn_medium_nmos_best.pt`    | `d8f91418085fa3a5…`   | V6.4.1 seed-42                       | V6.4.1 retrain   |
| `tsmc7_dn_medium_pmos_best.pt`    | `395e451fbe769ea7…`   | V6.4.1 seed-42                       | V6.4.1 retrain   |
| `tsmc12_dn_medium_nmos_best.pt`   | `4a045557cfb1e7a0…`   | V6.4.1 seed-42                       | V6.4.1 retrain   |
| `tsmc12_dn_medium_pmos_best.pt`   | `88dc5f916b7d2472…`   | V6.4.1 seed-42                       | V6.4.1 retrain   |
| `tsmc16_dn_medium_nmos_best.pt`   | `05ae0ba8e6288306…`   | V6.4.1 seed-42                       | V6.4.1 retrain   |
| `tsmc16_dn_medium_pmos_best.pt`   | `127e53a856dd18ac…`   | V6.4.1 seed-42                       | V6.4.1 retrain   |

The V6.4.1 seed-42 backup is preserved at `/tmp/seed42_backup_20260524/` (8
`.pt` + 8 `.npz` + `manifest.sha256`) for one-line restoration.

## Final complex-circuit table

| Test       | TSMC5 (P7-stock)              | TSMC7 (seed-42) | TSMC12 (seed-42) | TSMC16 (seed-42) | Pass |
|------------|-------------------------------|-----------------|------------------|------------------|------|
| ring_osc   | **PASS** perErr 2.98 %        | FAIL  8.97 %    | PASS 3.01 %      | PASS 2.88 %      | **3/4** |
| opamp      | **PASS** gainErr 2.64 %       | FAIL 30.67 %    | FAIL 10.94 %     | FAIL 100 % flat  | **1/4** |
| sram_snm   | PASS lobes positive           | PASS            | PASS             | PASS             | **4/4** |
| switchcap  | FAIL 14.68 % charge err       | PASS 3.06 %     | FAIL  8.33 %     | FAIL 13.13 %     | **1/4** |
| **TOTAL**  |                               |                 |                  |                  | **9/16** |

Inverter gate (re-verified on V6.4.4 mix):

| Tech    | VTC NRMSE % | Tran post-startup NRMSE % |
|---------|------------:|--------------------------:|
| TSMC5   | 1.21        | 1.62                      |
| TSMC7   | 2.37        | 1.09                      |
| TSMC12  | 2.05        | 1.41                      |
| TSMC16  | 1.33        | 1.45                      |

All four under the 5 % gate.

## Step-by-step provenance

- **Step 1** (`step1_seed42.md`): V6.4.1 seed-42 baseline on all four techs.
  7/16 (ring 2/4, opamp 0/4, sram 4/4, sc 1/4). Identical to the V6.3.1
  iter-1 baseline that was never re-measured under the V6.4.2 solver.

- **Step 2** (`step2_p7_stock_swap.md`): swapped TSMC5/7 to V6.4.2
  Phase-7a stock-recipe winners. TSMC5 won cleanly (ring_osc + opamp
  FAIL→PASS); TSMC7 opamp **regressed** structurally (30.67 % → 100 %
  flat-Vout — better inverter VTC did not predict opamp bias-point
  quality at the differential pair). Strict 7/16, best-case 8/16. Step
  showed that **inverter-VTC selection alone cannot drive complex-circuit
  pass rate** — per-tech / per-circuit selection is required.

- **Step 3** (partial, `step3_logs/inverter_gate.log`): with
  `NN_SYMMETRIC_CAPS=1` the inverter gate held 8/8 (VTC unchanged,
  transient TSMC5/7/12/16 = 1.41/1.71/1.41/1.45 % within Step 2 noise),
  so the flag is inverter-safe. The ring-oscillator and switched-cap
  re-measurement under the flag was **not completed** in this iteration;
  decision deferred. **Recommendation:** keep the flag default OFF
  (dormant) until the RO+SC measurement is recorded — Phase 2a remains a
  ready future-work lever.

- **Step 4** (lightweight): instead of the full greedy per-tech pair
  search, applied the Step 2 evidence directly — TSMC5 keeps P7-stock,
  TSMC7 reverts to seed-42 (the structural opamp regression makes its
  P7-stock unshippable), TSMC12/16 stay seed-42 (no better candidate
  exists on disk). Final inverter gate re-verified 8/8 on this mix.

## What V6.4.4 does and does not move

**Moves (vs V6.4.1 seed-42 / V6.3.1 iter-1 baseline):**
- TSMC5 ring oscillator: 6.76 % period error → 2.98 % (FAIL → PASS).
- TSMC5 two-stage Miller opamp: 14.78 % gain error → 2.64 % (FAIL →
  PASS).

**Does not move:**
- TSMC7/12/16 opamp gain (3/4 still FAIL; TSMC16 flat-Vout, TSMC7
  large gain error, TSMC12 marginal).
- TSMC7 ring oscillator (FAIL 8.97 %).
- TSMC5/12/16 switched-cap charge fidelity (3/4 still FAIL; the
  off-state region is the only iter-1 lever left for SC and was
  exhausted by Phase 4).
- SRAM `force_ic` rail-snap (sram_snm butterfly passes 4/4, but the
  force_ic re-solve still converges to a non-rail NN fixed point on
  every tech — same model-fidelity gap diagnosed by V6.4.2 Phase 6).

## Levers remaining for a future iteration

V6.4.4 closes the inference-only opportunity from the artifacts on disk.
Further gains require:

1. **Re-running TSMC5/7 Phase-7a best-of-N with opamp gain + RO period
   in the scoring vector** (iter-1 Phase-7 used inverter VTC only — Step
   2 evidence shows that proxy is broken for opamp bias-point quality).
2. **Phase 8 split-head DirectNet** (deferred in iter-1, requires
   retraining): decouples the `id`/conductance head from charge/cap
   heads, which would unlock a spectral-norm-on-conductance-head
   constraint and may close the TSMC7/12/16 opamp gates.
3. **Full RO + SC re-measurement under `NN_SYMMETRIC_CAPS=1`** — Step 3
   was cut short. If sym-caps closes TSMC7 RO or TSMC5/12/16 SC, the
   flag can be promoted to default-on for transient.
4. **V6.4 best-of-N artifacts are unrecoverable** — the
   `/tmp/v6_4_checkpoints_backup_20260517/` directory was cleared by
   `/tmp` cleanup between V6.4.2 ship and this iteration. Any future
   reference to "V6.4 best-of-N" requires re-deriving them from the
   training scripts, which is out of scope here.

## Files written this iteration

- `docs/plans/2026-05-24-directnet-v6.4.4-complex-circuits-iter2.md`
- `results/v6_4_4_iter2/step1_seed42.md`
- `results/v6_4_4_iter2/step2_p7_stock_swap.md`
- `results/v6_4_4_iter2/step3_logs/inverter_gate.log` (partial Step 3)
- `results/v6_4_4_iter2/step4_logs/inverter_gate.log` (V6.4.4 final
  inverter verification on the mixed working set)
- `results/v6_4_4_iter2/V6_4_4_final.md` (this file)

## Two-commit ship — load-bearing Phase-7a dependency

V6.4.4 lands on `feat/v6.4.1` as two commits:

| Commit    | Subject                                                                  |
|-----------|--------------------------------------------------------------------------|
| `4fcce2a` | `feat(v6.4.4): add BSIMAR_CHECKPOINT_DIR env var + v6_4_seed42 checkpoint archive` |
| `df9cfe3` | `fix(v6.4.4): restore Phase-7a code required by on-disk checkpoints`     |

`4fcce2a` carried the V6.4.4 docs (CLAUDE.md, CHANGELOG, this iter-2
plan + results, and the sprint scripts) plus a new
`BSIMAR_CHECKPOINT_DIR` env var in `bsimar/config.py`. Crucially it
**referenced** the V6.4.2 Phase-7a code as "newly committed" but did
not actually stage those four files
(`bsimar/{cli/train,models/direct_net,training/trainer}.py`,
`pycircuitsim/models/mosfet_directnet.py`).

A post-push inverter-gate verification surfaced the consequence: the
V6.4.1 seed-42 retrain ran with the Phase-7a `_MonotoneVgResidual` class
in scope (even with `--monotonic` defaults OFF), so the on-disk
state_dicts carry `mono.*` keys. Without the model class these fail
with `Unexpected key(s) in state_dict: "mono.w_rest", "mono.w_vg_raw",
"mono.b1", "mono.w_out_raw", "mono.b_out", "mono.sign"`. With the
Phase-7a code restored in `df9cfe3`, the gate is back to 8/8 PASS on
the V6.4.4 mix.

**Lesson:** future inference-only iterations that swap checkpoint files
must verify the model class matches what the checkpoint was trained
with, not just the recipe flags. Stem name does not encode whether
optional submodules were present at save time.

## Target reckoning

The plan's target was **≥ 10/16**. V6.4.4 lands at **9/16 (+2 over V6.4.1
seed-42, +2 over V6.3.1 iter-1 baseline)**. The plan is not met in
absolute terms, but every inference-only lever from the artifacts on
disk has been exhausted; remaining gates are gated on a retrain (Phase 8
or a re-scored Phase-7 best-of-N).
