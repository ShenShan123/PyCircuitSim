# S7 = P2 — Rule-15 reverse-Vds clamp relaxation (V6.4.7, 2026-06-12)

**Change (`pycircuitsim/models/mosfet_nn.py`):** the (b) reverse branch of
`_apply_vds_correction` no longer hard-zeroes conduction: reverse id =
`id_raw · f_sym · taper(|Vds|)` with the same `VT = max(0.06·VDD, 0.026)`
blend as forward (keeps Id(Vds=0)=0 exact) and a C¹ smoothstep taper;
matching gm/gmb factors; (c) untouched in both directions (its
`|id_raw|·exp/VT` linear term is the conductance that cures the force_ic NR
fold — measured 1.32e-4 S at the live Mpr bias, 13× the documented 1e-5 S
fold threshold); (d) scoped by direction (reverse allows the physically
flipped sign, clamps forward-sign noise). Forward path structurally
untouched. Authorized by the S7 probe: raw reverse surface sign-correct,
~25–35 % conservative on the trained corridor (`S7_P2_rev_probe.md`).

## Taper-window selection (pre-registered rule: LARGEST window that breaks no protected gate)

| Window (taper start/end ·VDD_train) | opamp | SC | RO | veto status |
|---|---|---|---|---|
| 0.30/0.40 (full trained corridor) | 1/4 — **TSMC5 9.78→13.57 % VETO BREAK** | 2/4 (T12 3.69) | 3/4 all improved | KILLED |
| 0.10/0.20 (conservative) | 2/4 (T5 2.15, T12 4.95) | 1/4 (T12 7.46 — flip lost) | 3/4 identical | clean but dominated |
| **0.20/0.30 (SHIPPED)** | **2/4 (T5 2.49, T12 4.97; T7 resurrected flat→10.16 %)** | **2/4 (T7 1.89, T12 4.13)** | 3/4 | **clean** |

Multiple-comparison caveat (recorded per house adversarial discipline):
three windows were scanned and the best shipped. The window rule was stated
before the bisection ran; 0.20/0.30 is also the closest no-veto window to
the principled corridor bound (0.30). Residual selection-overfit risk is
delegated to the S19 blind holdouts + replicate-3× discipline.

## Shipped-window battery (CPU, OMP_NUM_THREADS=1; `s7_logs/*_vc.log` + opamp_vc/switchcap_vc)

| Suite | Pre-P2 (S5b state) | Post-P2 shipped (0.20/0.30 window) | Verdict |
|---|---|---|---|
| lifted-source 12 | 12/12 | **12/12** | HELD |
| inverter 8 | 8/8 (VTC 1.13/3.90/1.47/1.53; tran 1.62/1.09/1.41/1.45) | **8/8** — VTC 0.96/2.36/1.31/1.11; tran **1.34/1.06/0.84/0.94** (uniform improvement; identical across all three windows — overshoot recovery saturates at small reverse Vds) | HELD + improved |
| DC 55 | 55/55 | **55/55** | HELD |
| tran 64 | 64/64 | **64/64** | HELD |
| ring_osc | 3/4 (2.97/8.98/3.01/2.88 %) | **3/4 — 2.61/8.28/2.19/2.13 %** (identical at all three windows) | all 4 techs improved; TSMC7 −0.7 pp |
| opamp | 2/4 (T5 9.78 fragile, T12 5.21; T7/T16 flat) | **2/4 — T5 2.49 (de-fragilized 4×), T12 4.97, T7 flat→10.16 (0.16 pp from gate)** | HELD + big margin gains |
| switchcap | 1/4 (T7 1.86–3.40 fragile; T12 8.14) | **2/4 — T12 FLIPPED (4.13 %), T7 1.89; T16 charge 3.38 (droop still fails, 802 %)** | **+1 GATE** |
| SRAM butterfly | 4/4 | **4/4** | HELD |
| SRAM force_ic | 0/8; high node off-rail (q 0.785–0.837), qb 0.104–0.117 | 0/8 — shipped window: symmetric release on T5/12/16 (q=qb=0.328/0.415/0.410), T7 railed-high (q=0.751, qb=0.128) | unchanged on the gated metric; see phenomenology + caveat below |

**Headline: 10/16 → 11/16** (SC TSMC12 flipped; all prior gates held).

## force_ic attractor phenomenology across windows (recorded for P3)

- 0.30-window: released cell collapses to the SYMMETRIC metastable point on
  TSMC5/7/16 (q=qb=0.328/0.391/0.410) — deep reverse conduction destabilizes
  the asymmetric attractor. (Part of why the wide window was killed.)
- 0.10-window: asymmetric attractor with the high node railed exactly
  (0.650/0.751/0.802/0.800 ≡ VDD); low node propped at 0.092–0.137 V by the
  pinning NMOS weak-inversion over-prediction (P0-D: 7.5× at Vov≈+45 mV).
- Shipped 0.20-window: symmetric on TSMC5/12/16 (0.328/0.415/0.410),
  railed-high asymmetric on TSMC7 only.
- Net: P2 delivered its mechanism (restoring current + fold-curing
  conductance); closure of force_ic now rests on P3, exactly as the plan's
  joint-ownership prediction (P0+P2+P3) stated.
- **CAVEAT for S11/P3 (recorded now):** both no-veto windows are equally
  0/8 on the gated metric, but the 0.10-window leaves the release at the
  railed-high asymmetric attractor on all 4 techs — plausibly a better
  starting basin for P3's weak-inversion suppression than the symmetric
  point. **If P3 closes force_ic at the 0.10 window but not at 0.20, the
  window trade re-opens, and ship-required force_ic outranks the SC TSMC12
  headline gate.** The taper window is one constant
  (`_reverse_taper`: x0/x1) — re-testing is cheap.

## Self-checks

`scripts/v6_4_7_s7_selfcheck.py`: id(Vds=0)=0 exact; continuity across
Vds=0 clean; gds>0 everywhere; no wrong-sign reverse points; taper leak 0;
recovery row corrected id −6.70 µA (raw −10.18, OSDI −13.71) with gds
1.32e-4 S. One benign threshold trip at the compressed-taper edge of the
0.10 window (50 nA step at µA scale; not present at the shipped window's
wider band — re-run recorded in the log).

## Dead-end record (within-step)

The full-corridor window (taper 0.30/0.40) is DEAD as shipped
configuration: it regresses TSMC5 opamp 9.78 → 13.57 % (gate ±10 %) and
collapses the force_ic release to the symmetric point on 3 techs. The
trained corridor is usable physics, but the opamp DC op-point tolerates
only ≤0.20·VDD of it. Numbers: `s7_logs/opamp.log`, `sram_snm.log`,
`switchcap.log` (as-built run).
