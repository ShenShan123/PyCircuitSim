# V6.4.7 S11b — pivot to the 2 open headline cells (tsmc5 SC, tsmc7 opamp)

**Date:** 2026-06-15 · After S11/P3 KILL (force_ic), user-directed pivot to the
2 failing cells of the 14/16 mix. **Outcome: both are systematic
model-fidelity limits that resist the available levers; headline stays 14/16.**
One useful artifact: a strictly better-positioned tsmc7 promotion candidate.

## tsmc5 switchcap — over-conduction 12.16 % (gate ≤5 %)

The pass-NMOS transfers too much charge during sampling (DN overshoots Vin;
forward id too strong — S5b). Tested levers:

- **Subthreshold loss (S11):** `v6_4_7_s11sub_w0002_s42_tsmc5` → SC **12.16 →
  11.70 %** (barely moves). The over-conduction is in the moderate/strong
  conduction region, NOT the weak-inversion tail the asinh-s2 term re-scales.
  (That checkpoint also collapses the tsmc5 opamp = v2-data lottery — net
  regression, not promotable.)
- **Trajectory corridor (S12):** the W=3 corridor (which INCLUDES the SC
  trajectory) left tsmc5 SC at 12.16 % ("not corridor-addressable at this dose")
  AND collapsed the tsmc5 opamp. A gentler dose fixes it even less.
- Also resisted P0 frame (S2: "SC unchanged"), P2 reverse clamp (S7: tsmc12
  flipped, tsmc5 did not), symcaps (S4: KILLED).

**Conclusion: a genuine forward-conduction-accuracy limit of the DirectNet for
tsmc5 at the SC sample biases.** Not subthreshold-owned, not corridor-
addressable. Documented hard-fail.

## tsmc7 opamp — systematic ~10–11 % gain over-prediction (gate ≤10 %)

The DN over-predicts the opamp DC gain by ~11 % on the healthy seeds (NG gain
≈163; DN ≈181). This is a **systematic bias, not seed luck or a collapse**:

| checkpoint | opamp gain | gain_err | note |
|---|---|---|---|
| production S8 baseline | — | **10.16 %** | 0.16pp over; but fails RO (8.28 %) |
| control-v2 s7 (W=0) | 181 | 10.99 % | healthy, 0.99pp over |
| control-v2 s31 (W=0) | 186 | 13.77 % | healthy |
| control-v2 s17 (W=0) | 375 | 129 % | over-gain |
| control-v2 s42 (W=0) | 0 | collapsed | lottery |

**Gentle corridor W-sweep (the deferred S12 W-sweep; W∈{1,2} × seeds{7,31}):**
the corridor **PRESERVES the W=0 over-gain (181→181) or COLLAPSES it to 0** — it
does NOT pull the gain down toward NG (the S10 preserve-or-collapse value-surface
fragility). No gentle "reduce gain 11 %" path exists:

| config | opamp | RO | inv_vtc | SC |
|---|---|---|---|---|
| w1 s7 | collapsed | 2.91 % ✓ | 3.06 % | 1.02 % |
| w1 s31 | 14.95 % | 2.91 % ✓ | 5.78 % ✗ | 1.03 % |
| **w2 s7** | **10.78 %** (gain 181) | **2.86 % ✓** | **2.93 % ✓** | **1.02 % ✓** |
| w2 s31 | collapsed | 2.91 % ✓ | 2.73 % | 1.03 % |

**Conclusion: tsmc7 opamp is a systematic sub-1pp over-gain miss** (best ~10.2–
10.8 %, within run-to-run noise of the 10 % gate) that the corridor cannot
reliably close — the gain magnitude is value-surface owned (S10: fixing the
Jacobian collapses it). Documented hard-fail (not robustly flippable).

## Useful artifact for S19 — `v6_4_7_pivcor_w2_s7_tsmc7` is a better tsmc7 candidate

The W=2-corridor s7 checkpoint **passes RO (2.86 %), inverter (2.93 %), SC
(1.02 %) and keeps the opamp near-pass (10.78 %, gain 181)** — strictly better
positioned than the S12 corridor tsmc7 (which passes RO but COLLAPSES the
opamp). Same headline gate count (RO pass, opamp fail), but a far healthier
opamp (0.78pp from the gate vs collapsed). **Recommend it as the tsmc7 promotion
candidate at S19** in place of the S12 corridor (replication-check the opamp
margin; the gentle dose is more robust). Checkpoints gitignored, regenerable
from `scripts/v6_4_7_pivot_corridor.sh` (W=2 SEEDS="7" TECHS="tsmc7").

## Net

**Headline unchanged 14/16.** The 2 open cells (tsmc5 SC over-conduction, tsmc7
opamp over-gain) + `force_ic` are all **systematic value-surface / fixed-point /
forward-conduction limits** that resist the cheap DirectNet levers (subthreshold,
corridor dose, frame, clamp). Closing them needs a more fundamental change
(architecture / physics-core / accept as known-issues). Recommend proceeding to
**S19 promotion at 14/16** with `force_ic` + these 2 cells documented as
model-fidelity known-issues, OR a scoped structural investigation if the 3
remaining gaps are must-close.
