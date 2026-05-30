# Phase 4 — Rule-15 `Ioff_rail` extension (inference-only)

**Date:** 2026-05-28  •  **Status:** **KILLED — patch reverted**
**Patch tested:** `(a.5) off-state floor` block in `_apply_vds_correction`
(`pycircuitsim/models/mosfet_nn.py`, +26 LOC behind `NN_IOFF_RAIL_K` env var).
**Verdict.** No `k` value closes any SRAM `force_ic` cell. Every non-zero `k`
catastrophically regresses the inverter VTC (NRMSE 1.21 % → 11.56 %, an order
of magnitude). Patch reverted; working tree clean.

## Dormancy verification (k = 0)

The patch at `NN_IOFF_RAIL_K=0` (default) is byte-identical to V6.4.4. Inverter
gate with the patch present but env var unset:

| Tech   | VTC NRMSE % | Tran NRMSE % | Match Phase 1? |
|--------|------------:|-------------:|:--------------:|
| TSMC5  |        1.21 |         1.62 | exact          |
| TSMC7  |        2.37 |         1.09 | exact          |
| TSMC12 |        2.05 |         1.41 | exact          |
| TSMC16 |        1.33 |         1.45 | exact          |

Dormancy is clean — the patch only fires when `NN_IOFF_RAIL_K > 0`.

## k-sweep results

### Inverter VTC NRMSE % (gate ≤ 5 %, baseline 1.21 / 2.37 / 2.05 / 1.33)

| k (nA/fin) | TSMC5 VTC | TSMC7 VTC | TSMC12 VTC | TSMC16 VTC | Verdict          |
|-----------:|----------:|----------:|-----------:|-----------:|:-----------------|
|          0 |     1.21  |     2.37  |     2.05   |     1.33   | baseline         |
|          1 |  **11.56**|  **16.34**|  **12.40** |  **13.51** | CATASTROPHIC     |
|          3 |    12.80  |    16.54  |    11.07   |    12.80   | CATASTROPHIC     |
|         10 |    13.01  |    15.39  |    13.22   |    14.32   | CATASTROPHIC     |
|         30 |     OOM   |     OOM   |     OOM    |     OOM    | (GPU contention — not re-measured; trend was already KILL) |
|        100 |     OOM   |     OOM   |     OOM    |     OOM    | (GPU contention — not re-measured; trend was already KILL) |

Inverter post-startup transient NRMSE % at k = 1 (baseline 1.62/1.09/1.41/1.45):
**17.85 / 23.49 / 15.08 / 10.35** — 10–20× regression on every tech.

Kill criterion 2 hit: *"If any `k` regresses … inverter VTC MaxErr by > 5 mV → revert"*. NRMSE quadruples-to-decuples, MaxErr necessarily blows past 5 mV.

### SRAM `force_ic` at k = 1 (only k value with complete data — other k's stopped after inverter KILL)

`force_ic` baseline (Phase 1, k=0): all four techs settle at q ≈ 0.866 / qb ≈ 0.199 (0/8 PASS).

| Tech   | state1 q / qb   | state0 q / qb   | Pass? | vs k=0 baseline                    |
|--------|-----------------|-----------------|:-----:|------------------------------------|
| TSMC5  | 0.375 / 0.381   | 0.381 / 0.375   | FAIL  | moved FURTHER from rails (0.873→0.375) |
| TSMC7  | 0.657 / 0.222   | 0.222 / 0.657   | FAIL  | moved FURTHER from rails (0.867→0.657) |
| TSMC12 | 0.095 / 1.244   | 1.244 / 0.095   | FAIL  | wild overshoot beyond VDD          |
| TSMC16 | 0.731 / 0.390   | 0.390 / 0.731   | FAIL  | moved FURTHER from rails (0.865→0.731) |

**The patch moves the SRAM attractor AWAY from the rails**, not toward them.
TSMC12 even diverged the solver to q = 1.244 (above VDD = 0.8 V).

Kill criterion 1 hit: *"If best `k` does not close SRAM `force_ic` ≥ 1/4 → abandon Phase 4."*

### SRAM butterfly at k = 1

Butterfly lobes all positive on every tech / NFIN corner (min(qb) ranges 247–321 mV; the gate is "positive" not "low"). The butterfly NRMSE worsened (Phase 1: ~50–80 %; Phase 4 k=1: ~25–35 %), but the formal butterfly gate ("all lobes positive") still PASSES — so kill criterion 2's "regresses sram_snm butterfly" did not trip. Kill is on the inverter regression alone.

## Root-cause analysis

The patch's formula

```python
Ioff_rail = max(abs(id_raw_pre), Ioff_floor)        # k·NFIN·1nA
result["id"] += sign_conv * blend * Ioff_rail
```

does *not* gate on the device's on/off state. When the device is conducting
(id_raw_pre ≫ Ioff_floor), `Ioff_rail = abs(id_raw_pre)`, and at the
inverter rail (Vds ≈ VDD, blend ≈ 1) the patch effectively **doubles the
conducting current**. That doubles the inverter pull-down strength and shifts
the VTC trip by hundreds of mV — exactly matching the observed 10× NRMSE
regression.

A corrected formulation would have to floor only the off-state, e.g.
```
Ioff_rail = max(Ioff_floor - abs(id_raw_pre), 0.0)   # additive only below floor
```
but that is a different patch (different physics, different kill criteria) —
**not the plan's Phase 4** and so not retried here. The Phase 2 butterfly
warm-start probe already proved the q ≈ 0.18 attractor is a true model
property; closing it likely requires a retrain (Phase 5/6), not an
inference-only Vds correction.

## Decision

**KILL Phase 4 as specified in the plan.** Revert the patch in full:

```
git checkout -- pycircuitsim/models/mosfet_nn.py
```

Working tree after revert (`git status`):
```
On branch feat/v6.4.4
Your branch is up to date with 'origin/feat/v6.4.4'.
Untracked files: docs/plans/2026-05-28-directnet-v6.4.5-ro-sram.md
nothing added to commit but untracked files present
```

## Complex-circuit pass count

Unchanged from Phase 1: **9/16**. Inverter 8/8 unaffected (no shipped change).

## What this rules out for V6.4.5

1. **The plan's `(a.5) off-state floor` formulation does not work.** Its
   `max(|id_raw|, floor)` term boosts the conducting state as much as the
   off state.
2. **Inference-only Vds corrections cannot move the q ≈ 0.18 attractor**
   without a Vgs gate — and Vgs gating breaks the autograd-of-id contract
   (Rule 1).
3. SRAM force_ic is now firmly model-fidelity territory. Phase 5 retrain or
   the V6.4.6 split-head architecture (plan §6) are the remaining levers.

## Suggested follow-ups (NOT executed in V6.4.5)

- **Track B B7 (Physics-anchored residual)** — gates the off-state via a
  closed-form subthreshold-exp skeleton; the residual MLP rides on top and is
  multiplicatively scaled, so it cannot double the conducting current.
- **Track B B9 (Hard-monotone lattice)** — removes the spurious sub-rail
  fixed point by construction.
- **Phase 5 TSMC7 retrain** — moves the ring_osc gate; SRAM stays as a
  V6.4.6 deliverable.
