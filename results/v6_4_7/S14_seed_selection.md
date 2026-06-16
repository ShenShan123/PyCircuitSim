# V6.4.7 S14 — authoritative-gate seed selection — headline 13 → **14/16**

**Date:** 2026-06-16 · Continuation step (user-directed, post-S19a). The cheapest
"train-free teachers first" form of P6: before building anything, run the
**authoritative gates** (`verify_complex_*.py`, NOT the scorer) over **every
existing seed checkpoint** for the value-owned cells. Because the scorer and the
gate disagree on bistable cells (the S19a lesson), an existing seed can pass the
*gate* even when the scorer didn't flag it. Logs: `results/v6_4_7/s14_logs/`.
Drivers: `scripts/v6_4_7_s14a_opamp_sweep.sh`, `scripts/v6_4_7_s14c_forceic_sweep.{sh,py}`.

## S14a — opamp gate sweep → tsmc16 RECOVERED

Authoritative `verify_complex_opamp.py` over all tsmc16 + tsmc7 seeds (gain_err,
`OMP_NUM_THREADS=1`):

**tsmc16 (gate ≤10 %):**
| stem | gain | gain_err | status |
|---|---|---|---|
| **`v6_4_7_s12cor_w3_s17_tsmc16`** | **197.3** | **5.14 %** | **PASS** |
| `v6_4_7_ctlv2_s17_tsmc16` | 197.9 | 5.44 % | PASS |
| `v6_4_7_ctlv2_s31_tsmc16` | 197.9 | 5.44 % | PASS |
| `v6_4_7_s12cor_w3_s31_tsmc16` (S19a pick) | 382.8 | 103.98 % | FAIL |
| ctlv2_s7/s42, s12cor_s7/s42, s11sub_s42 | 0.0 | ~100 % | FAIL (collapse) |

The S12 scorer picked **s31** (it measured 197 once, a bistable basin landing);
the authoritative gate puts s31 on the **382 over-gain branch** (the S19a
retraction). But **s17 lands on the correct ~197 branch deterministically.**

**tsmc16 `s12cor_w3_s17` full authoritative-gate verification → 4/4:**
- opamp **5.14 %** PASS — replicated `OMP∈{1,2,4}` (gain 197.3 identical 3/3, NOT a fluke)
- ring_osc **3.99 %** PASS · switchcap **2.01 %** PASS (droop 1 % of allow) · butterfly positive PASS
- force_ic 0/2 (q=0.800 railed, qb=0.117 — part of the open 0/8)

⇒ **tsmc16 opamp cell recovered honestly. Promotion candidate s31 → s17. Headline 13 → 14/16.**

**tsmc7 (gate ≤10 %): NO recovery.** No seed passes with RO also passing — the
corridor (needed for RO) forces the opamp to collapse (`s12cor_w3` s17/s31/s42 →
gain 0; s7 → 356/118 %) or over-shoot (`pivcor_w1_s31` 15.04 %), and
`pivcor_w2_s7` at **10.78 %** remains the best (the documented +0.78 pp
over-gain). `ctlv2_s31` reported "NA" = an errored opamp run; control-v2 has no
corridor ⇒ RO fails regardless. **tsmc7 stays `pivcor_w2_s7`, 3/4.**

## S14c — force_ic seed sweep → NO recovery (cheap dead-end)

Fast `force_ic_probe` (2 DC solves, no butterfly) over **44 checkpoints × 4
techs**. Acceptance = released KCL residual OK **AND** both nodes within
`0.1·VDD` of their seeded rails. **Every checkpoint = 0/2.** Best "0"-node
(storage-0) resting voltage above ground vs the `0.1·VDD` band:

| tech | best stem | q (railed) | qb ("0" node) | band 0.1·VDD | margin outside |
|---|---|---|---|---|---|
| tsmc5  | ctlv2_s42 / s12cor_s31 | 0.650 | 0.086 | 0.065 | ~21 mV |
| tsmc12 | ctlv2_s7/s42 / s12cor_s42 | 0.800 | 0.116 | 0.080 | ~36 mV |
| tsmc16 | s12cor_s17/s42 / ctlv2_s17 | 0.800 | 0.117 | 0.080 | ~37 mV |
| tsmc7  | ctlv2_s42 / s12cor_s42 / s11sub_s31 | 0.749 | 0.121 | 0.075 | ~46 mV |

Many seeds (incl. the promoted `pivcor_w2_s7` and most corridor seeds) land on
the **symmetric metastable point** (q=qb=VDD/2) — even further from railing. The
"0" node is held 21–46 mV above ground by OFF/pull-up leakage the model can't
suppress; **no seed rails.** This confirms force_ic is a deep
regenerative-gain / NR-fixed-point limit (S11: a more-accurate id surface
*removes* the bistability; P0-A: the railed point is NR-unstable), **not
seed-addressable.** The only remaining lever is the structural **S17 = P9**
(physics-anchored OFF core) — whose premise (more-accurate OFF → rails) is in
direct tension with S11 (more-accurate OFF → symmetric collapse), so it is
high-effort / uncertain-payoff.

## Net

**Headline 13 → 14/16** (tsmc16 opamp recovered via s17, authoritative-gate
verified — replacing the retracted s31). Final mix: tsmc5=baseline,
tsmc7=`pivcor_w2_s7`, tsmc12=baseline, **tsmc16=`s12cor_w3_s17`**. force_ic
**0/8** (ship-required, OPEN — confirmed not seed-closeable). Remaining
known-issues: force_ic, tsmc5 SC 12.14 %, tsmc7 opamp 10.78 %.

sha256 of the new tsmc16 pick:
```
39b8c92a…  v6_4_7_s12cor_w3_s17_tsmc16_nmos_best.pt
e4f34165…  v6_4_7_s12cor_w3_s17_tsmc16_pmos_best.pt
7a0bd2f5…  v6_4_7_s12cor_w3_s17_tsmc16_nmos_norm.npz
44534059…  v6_4_7_s12cor_w3_s17_tsmc16_pmos_norm.npz
```
(norm.npz are byte-identical to the s31 pick — same dataset/normalization;
only the state_dict differs.)
