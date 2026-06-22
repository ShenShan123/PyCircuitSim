# DirectNet (LEVEL=73) capacity benchmark — small / medium / large / xl

All checkpoints trained on ONE identical clean recipe (`--apply-filter off --swa-mode ema --seed 42`); capacity is the only variable. Datasets = full Vth + geometry grid per tech (`--variants all`, inv-trip + subvt-off overlays). Ground truth = NGSPICE BSIM-CMG (LEVEL=72), repo ngspice-45.2, CPU-pinned.

Sizes: small=128x3 (~0.06M p) / medium=256x5 (~0.4M p) / large=384x6 (~0.9M p) / **xl=512x8 (~2.13M p)**.

> **V6.6 update — XL capacity tier + µA-band loss lever (KILLED).** This revision adds the **XL** tier (512×8, 2.13M p) on the identical clean recipe and a tested-but-reverted accuracy lever. **Headline: the capacity curve PEAKS at `large` and DECLINES at XL — 6 → 9 → 12 → 9 / 16.** XL fits the device surface ~10× tighter (val loss 2e-4 vs medium's ~2e-3) yet has the *worst* off-nominal parametric NRMSE (2.17% mean, the highest of any tier) and *loses every value-surface-fragile gate `large` had won* (tsmc5 opamp, tsmc12 opamp, tsmc7 ring-osc all PASS→FAIL). This is the cleanest possible confirmation of V6.4.8-S1: **capacity is not the bind; beyond `large` it over-fits the value-surface and collapses the high-gain basins.** The **µA-band loss de-compression** lever (the V6.4.8 roadmap's named next step — retune the default-off `SubthresholdIdLoss` to the µA band, `s2=1e-7 upper=3e-5`) was A/B-tested on tsmc5 switchcap and **KILLED**: charge_err 11.84% → 12.05%/11.69% (unchanged) with a slight DC regress. It *refutes* the 'loss-compression-owned' hypothesis — the tsmc5 switchcap over-charge survives direct µA-band loss de-compression (and survived S3's EKV prior), so it is **charge/transient (sample-and-hold) owned, not DC-current-band owned**. The V6.5 AC study below is unchanged (XL does not move AC: opamp 0/4, device CS-amp 4/12, gain0-err marginally worse).

## Key findings

1. **Circuit pass-rate is NON-monotonic in capacity: 6/16 → 9/16 → 12/16 → 9/16** (small → medium → large → **xl**). It rises through `large`, then **regresses at XL**. The S→L gains come from charge-transfer and timing circuits; the L→XL loss comes from value-surface over-fit collapsing the opamp / high-VDD-RO high-gain basins.
2. **Device-level Id-Vgs / inverter accuracy is excellent at every size, but XL over-fits.** Mean parametric NRMSE 1.63 → 1.29 → 1.7 → 2.17% (S→M→L→XL): it is *best at medium* and *worst at XL*, even though XL's validation loss is ~10× lower than medium's. XL fits the training distribution tightest but generalizes worst to off-nominal geometry/VT sweeps (e.g. tsmc7 DC/pmos 0.51% medium → 2.65% XL; tsmc12 DC/pmos 0.76% → 3.57%). Inverter VTC+transient stays 16/16 PASS at all four sizes. **Device fidelity is NOT the bind; more capacity past `large` hurts generalization.**
3. **The opamp is the hardest, value-surface-fragile gate.** Open-loop gain collapses to ~0 (NRMSE ~70%) at small AND medium for all four techs, recovers to PASS only at **large** and only for tsmc5/tsmc12 (tsmc12 large: gain err 6.25%, locus NRMSE 1.0%), then **collapses again at XL** (tsmc5/tsmc12 opamp PASS→FAIL). tsmc7/tsmc16 never pass on this clean recipe (the *shipping* tsmc7/tsmc16 need the pivcor / s12cor recipes). The high-gain basin is capacity- AND tech-sensitive with a sweet spot at `large` — bigger is not better.
4. **Switched-cap needs capacity up to a point:** 0/4 (small) → 3/4 (medium, large, **xl**). tsmc5 never passes (~11-12% charge error) at ANY size — and V6.6 proves it is **not** µA-band-loss-compression-owned (the lever moved it <0.2%): it is sample-and-hold charge/transient over-charge, independent of both capacity and µA-band DC loss weighting.
5. **Ring-osc** passes for the higher-VDD nodes (tsmc12/16) at every size; tsmc7 passes ONLY at `large` (PASS at L, FAIL at S/M/**XL** — another L→XL over-fit regression); tsmc5 never (period err 6-14%). **SRAM butterfly** (all-lobes-positive gate) passes 4/4 at every size; force_ic is an informational probe.

**Bottom line:** capacity helps circuit-level behaviour up to **`large` (12/16), which is the sweet spot**; **XL over-fits and regresses to 9/16**. Neither more capacity (XL) nor µA-band loss de-compression closes the recipe-sensitive gaps (tsmc7/tsmc16 opamp, tsmc5 switchcap) — the V6.4.x attribution to value-surface / charge-transient ownership (not capacity, not µA-band DC loss) is confirmed and sharpened. `large` remains the production capacity; XL is retained as the empirical over-fit boundary.

## Cross-size summary

| Size | Complex gates PASS | Device mean-NRMSE% (all sweeps) |
|---|---|---|
| small | 6/16 | 1.63 |
| medium | 9/16 | 1.29 |
| large | 12/16 | 1.7 |
| xl | 9/16 | 2.17 |

## Capacity comparison (the headline view)

### Complex-circuit gate verdict by capacity

| Tech | Circuit | small | medium | large | xl |
|---|---|---|---|---|---|
| tsmc5 | ring_osc | FAIL | FAIL | FAIL | FAIL |
| tsmc5 | opamp | FAIL | FAIL | PASS | FAIL |
| tsmc5 | sram_snm | PASS | PASS | PASS | PASS |
| tsmc5 | switchcap | FAIL | FAIL | FAIL | FAIL |
| tsmc7 | ring_osc | FAIL | FAIL | PASS | FAIL |
| tsmc7 | opamp | FAIL | FAIL | FAIL | FAIL |
| tsmc7 | sram_snm | PASS | PASS | PASS | PASS |
| tsmc7 | switchcap | FAIL | PASS | PASS | PASS |
| tsmc12 | ring_osc | PASS | PASS | PASS | PASS |
| tsmc12 | opamp | FAIL | FAIL | PASS | FAIL |
| tsmc12 | sram_snm | PASS | PASS | PASS | PASS |
| tsmc12 | switchcap | FAIL | PASS | PASS | PASS |
| tsmc16 | ring_osc | PASS | PASS | PASS | PASS |
| tsmc16 | opamp | FAIL | FAIL | FAIL | FAIL |
| tsmc16 | sram_snm | PASS | PASS | PASS | PASS |
| tsmc16 | switchcap | FAIL | PASS | PASS | PASS |

### Device-level mean NRMSE% by capacity (lower = better fit)

DC = Id-Vgs (NMOS/PMOS); INV = inverter VTC+transient (combined).

| Tech | Suite/Dev | small | medium | large | xl |
|---|---|---|---|---|---|
| tsmc5 | DC/nmos | 3.62 | 2.49 | 3.99 | 4.12 |
| tsmc5 | DC/pmos | 1.16 | 0.47 | 1.21 | 1.69 |
| tsmc5 | INV/all | 2.52 | 2.16 | 1.98 | 1.75 |
| tsmc7 | DC/nmos | 2.33 | 2.21 | 1.84 | 2.12 |
| tsmc7 | DC/pmos | 1.17 | 0.51 | 0.65 | 2.65 |
| tsmc7 | INV/all | 2.2 | 1.85 | 1.92 | 1.73 |
| tsmc12 | DC/nmos | 0.4 | 0.36 | 1.25 | 1.89 |
| tsmc12 | DC/pmos | 0.74 | 0.76 | 2.2 | 3.57 |
| tsmc12 | INV/all | 2.19 | 1.59 | 1.61 | 1.59 |
| tsmc16 | DC/nmos | 0.59 | 0.51 | 0.83 | 1.23 |
| tsmc16 | DC/pmos | 0.65 | 0.65 | 1.03 | 1.8 |
| tsmc16 | INV/all | 2.01 | 1.96 | 1.89 | 1.9 |

## AC small-signal accuracy (V6.5)

First-ever NGSPICE-gated evaluation of DirectNet (LEVEL=73) AC fidelity. The NN's small-signal capacitances are autograd derivatives of its predicted terminal charges (cgd=∂qg/∂Vd, cdd=∂qd/∂Vd, …) — a quantity no prior gate measured. Ground truth = NGSPICE `.ac` on the identical BSIM-CMG (LEVEL=72) OSDI model. Two circuit classes (AC needs a stable amplifying OP, so the free-running ring oscillator and bistable SRAM are out of scope):

- **Device CS-amp** — per-checkpoint NMOS/PMOS common-source amplifier, no external load cap so the device's own Cgd/Cdd set the pole; gates gain0 err ≤1.5 dB, f3db ratio ∈[0.7,1.43], mag NRMSE ≤10%. The passband phase is reported (not gated): deep in-band it matches (<7°), but at/beyond the −3 dB corner NG carries a strong Cgd-feedforward RHP-zero phase lag the NN does not reproduce — a distinct limitation from the (excellent) cap-driven pole.
- **Opamp open-loop** — two-stage Miller opamp; gates DC-gain err ≤3 dB, GBW ratio ∈[0.6,1.67], phase-margin err ≤15° (linear mag NRMSE reported, not gated — dominated by the 40 dB passband plateau).

**Findings.** (1) AC **gain** fidelity is excellent everywhere — device gain0 err <1.5 dB in 24/24 cells (mean 0.55–0.86 dB) — so the autograd gm/gds the NN feeds the AC stamp are accurate. (2) The dominant **cap-driven pole** is mostly faithful (f3db ratio ≈1.0 for the well-fit cells) but capacity/tech-variable: tsmc5 NMOS and tsmc12/16 PMOS under-predict the output cap (ratio 1.1–1.6), so 13/24 clear the magnitude gate. (3) The high-frequency **phase** (Cgd-feedforward RHP zero) is not reproduced — a clean, specific transcapacitance limitation. (4) The **opamp** AC inherits the DC value-surface fragility (0/12): the gain collapses or over-predicts at most cells, BUT where the OP lands in the good basin (tsmc12-large) the NN reproduces GBW to 0.97× and phase margin to 1.3° — the dynamics are right, the DC-gain *level* is the value-surface-owned miss. **No retraining is warranted:** the gaps are value-surface- and feedforward-owned, not a charge-derivative (dQ/dV) deficiency (which would have shown as bad gain *and* bad pole everywhere, the opposite of what is measured).

**V6.6 — XL adds 8 checkpoints (32 total) and does NOT change the AC story.** Device CS-amp gate holds at 4/12 for XL (17/32 overall), opamp stays 0/4 (0/16 overall), and device mean gain0-err drifts marginally worse (0.86→0.91 dB L→XL) — consistent with the XL device-surface over-fit seen in DC. AC fidelity is capacity-saturated by `large`.

### Cross-size AC summary

| Size | Device CS-amp PASS | Opamp PASS | Device mean gain0-err dB | Device mean magNRMSE% |
|---|---|---|---|---|
| small | 5/12 | 0/4 | 0.55 | 6.14 |
| medium | 4/12 | 0/4 | 0.78 | 8.4 |
| large | 4/12 | 0/4 | 0.86 | 9.21 |
| xl | 4/12 | 0/4 | 0.91 | 9.13 |

### Device CS-amp AC gate by capacity

| Tech | Dev | small | medium | large | xl |
|---|---|---|---|---|---|
| tsmc5 | nmos | FAIL | FAIL | FAIL | FAIL |
| tsmc5 | pmos | PASS | PASS | PASS | PASS |
| tsmc7 | nmos | PASS | FAIL | PASS | FAIL |
| tsmc7 | pmos | PASS | PASS | PASS | PASS |
| tsmc12 | nmos | PASS | PASS | FAIL | PASS |
| tsmc12 | pmos | FAIL | FAIL | FAIL | FAIL |
| tsmc16 | nmos | PASS | PASS | PASS | PASS |
| tsmc16 | pmos | FAIL | FAIL | FAIL | FAIL |

### Device CS-amp gain0 error (dB) by capacity (lower = better)

| Tech | Dev | small | medium | large | xl |
|---|---|---|---|---|---|
| tsmc5 | nmos | 0.83 | 0.7 | 1.46 | 1.58 |
| tsmc5 | pmos | 0.09 | 0.37 | 0.37 | 0.39 |
| tsmc7 | nmos | 0.56 | 1.32 | 0.24 | 1.07 |
| tsmc7 | pmos | 0.32 | 0.79 | 0.92 | 0.85 |
| tsmc12 | nmos | 0.04 | 0.31 | 0.89 | 0.33 |
| tsmc12 | pmos | 1.25 | 1.3 | 1.26 | 1.28 |
| tsmc16 | nmos | 0.06 | 0.19 | 0.25 | 0.26 |
| tsmc16 | pmos | 1.26 | 1.29 | 1.48 | 1.49 |

### Opamp open-loop AC gate by capacity (DC-gain err dB / GBW ratio / PM err°)

| Tech | small | medium | large | xl |
|---|---|---|---|---|
| tsmc5 | FAIL (5.69/21.6/5.9) | FAIL (13.87/5.76/44.4) | FAIL (113.96/n/a/n/a) | FAIL (86.09/n/a/n/a) |
| tsmc7 | FAIL (244.86/n/a/n/a) | FAIL (31.89/1.16/51.4) | FAIL (45.85/0.15/118) | FAIL (19.48/4.64/24.4) |
| tsmc12 | FAIL (84.33/n/a/n/a) | FAIL (3.15/19.6/50.7) | FAIL (5.14/0.966/1.43) | FAIL (42.69/0.125/115) |
| tsmc16 | FAIL (23.46/3.33/4.98) | FAIL (8.34/16.3/39.3) | FAIL (8.68/14.9/31.7) | FAIL (99.77/n/a/n/a) |

## Size = small

### Device-level parametric sweeps (per tech)

DC = Id-Vgs over L/NFIN/VT; INV = inverter VTC+transient. Values: baseline NRMSE% / mean NRMSE% over all sweep configs; MRE% (mean); R2 (min); pass-rate.

| Tech | Suite | Dev | base NRMSE% | mean NRMSE% | mean MRE% | min R2 | max MaxErr | Pass |
|---|---|---|---|---|---|---|---|---|
| tsmc5 | DC | nmos | 2.96 | 3.62 | 10.16 | 0.947 | 52.27uA | 7/7 |
| tsmc5 | DC | pmos | 1.09 | 1.16 | 4.75 | 0.9879 | 5.205uA | 7/7 |
| tsmc5 | INV | all | 3.39 | 2.52 | 12.39 | 0.9767 | 212.892mV | 16/16 |
| tsmc7 | DC | nmos | 3.88 | 2.33 | 8.08 | 0.9852 | 47.7uA | 5/5 |
| tsmc7 | DC | pmos | 0.91 | 1.17 | 3.75 | 0.9968 | 10.11uA | 4/4 |
| tsmc7 | INV | all | 2.62 | 2.2 | 10.32 | 0.9837 | 316.32mV | 16/16 |
| tsmc12 | DC | nmos | 0.34 | 0.4 | 2.0 | 0.9994 | 5.621uA | 9/9 |
| tsmc12 | DC | pmos | 0.25 | 0.74 | 2.6 | 0.9921 | 8.059uA | 9/9 |
| tsmc12 | INV | all | 2.11 | 2.19 | 10.53 | 0.9664 | 483.385mV | 16/16 |
| tsmc16 | DC | nmos | 0.05 | 0.59 | 1.98 | 0.9941 | 5.607uA | 7/7 |
| tsmc16 | DC | pmos | 0.38 | 0.65 | 2.45 | 0.9954 | 27.988uA | 7/7 |
| tsmc16 | INV | all | 1.06 | 2.01 | 8.79 | 0.9823 | 230.83mV | 16/16 |

### Complex circuits (per tech)

Gate verdict + headline + waveform NRMSE%/R2.

| Tech | Circuit | Gate | Headline | NRMSE% | R2 | MaxErr |
|---|---|---|---|---|---|---|
| tsmc5 | ring_osc | FAIL | period_err=8.05% | 64.31 | -1.2607 | 694.37mV |
| tsmc5 | opamp | FAIL | gain_err=100.00% trip_shift=116.00mV | 70.58 | -1.006 | 648.6mV |
| tsmc5 | sram_snm | PASS | NG_SNM=189.2mV DN_SNM=174.4mV  force_ic=ok/ok | — | — | — |
| tsmc5 | switchcap | FAIL | charge_err=11.75% | 15.3 | 0.6271 | 94.33mV |
| tsmc7 | ring_osc | FAIL | period_err=5.94% | 57.74 | -0.8632 | 789.51mV |
| tsmc7 | opamp | FAIL | gain_err=10.33% trip_shift=-146.00mV | 68.69 | -0.9258 | 727.34mV |
| tsmc7 | sram_snm | PASS | NG_SNM=187.4mV DN_SNM=323.1mV  force_ic=FAIL/FAIL | — | — | — |
| tsmc7 | switchcap | FAIL | charge_err=1.51% | 1.83 | 0.9948 | 21.38mV |
| tsmc12 | ring_osc | PASS | period_err=1.95% | 47.67 | -0.3098 | 839.2mV |
| tsmc12 | opamp | FAIL | gain_err=100.00% trip_shift=-12.00mV | 70.48 | -1.0053 | 797.4mV |
| tsmc12 | sram_snm | PASS | NG_SNM=239.1mV DN_SNM=226.8mV  force_ic=ok/ok | — | — | — |
| tsmc12 | switchcap | FAIL | charge_err=4.09% | 6.17 | 0.9447 | 34.84mV |
| tsmc16 | ring_osc | PASS | period_err=1.47% | 39.12 | 0.1055 | 829.41mV |
| tsmc16 | opamp | FAIL | gain_err=10.23% trip_shift=0.00mV | 1.76 | 0.9988 | 129.97mV |
| tsmc16 | sram_snm | PASS | NG_SNM=235.5mV DN_SNM=219.3mV  force_ic=FAIL/FAIL | — | — | — |
| tsmc16 | switchcap | FAIL | charge_err=2.76% | 4.2 | 0.974 | 23.26mV |

### AC small-signal (per tech)

| Tech | Dev | gain0_err dB | f3db ratio | magNRMSE% | phase(inband)° | Gate |
|---|---|---|---|---|---|---|
| tsmc5 | nmos | 0.83 | 1.58 | 8.69 | 69.6 | FAIL |
| tsmc5 | pmos | 0.09 | 1.0 | 1.0 | 79.91 | PASS |
| tsmc7 | nmos | 0.56 | 1.0 | 6.51 | 49.95 | PASS |
| tsmc7 | pmos | 0.32 | 1.0 | 3.66 | 62.17 | PASS |
| tsmc12 | nmos | 0.04 | 1.0 | 0.47 | 29.49 | PASS |
| tsmc12 | pmos | 1.25 | 1.41 | 15.89 | 36.45 | FAIL |
| tsmc16 | nmos | 0.06 | 1.0 | 0.59 | 32.48 | PASS |
| tsmc16 | pmos | 1.26 | 1.26 | 12.35 | 34.87 | FAIL |

Opamp open-loop:

| Tech | DC-gain err dB | GBW ratio | PM err° | magNRMSE% | Gate |
|---|---|---|---|---|---|
| tsmc5 | 5.69 | 21.6 | 5.9 | 83.71 | FAIL |
| tsmc7 | 244.86 | n/a | n/a | 73.18 | FAIL |
| tsmc12 | 84.33 | n/a | n/a | 68.17 | FAIL |
| tsmc16 | 23.46 | 3.33 | 4.98 | 61.91 | FAIL |

<details><summary>checkpoints resolved</summary>

- tsmc5: `tsmc5_dn_small_nmos_best.pt`
- tsmc7: `tsmc7_dn_small_nmos_best.pt`
- tsmc12: `tsmc12_dn_small_nmos_best.pt`
- tsmc16: `tsmc16_dn_small_nmos_best.pt`

</details>

## Size = medium

### Device-level parametric sweeps (per tech)

DC = Id-Vgs over L/NFIN/VT; INV = inverter VTC+transient. Values: baseline NRMSE% / mean NRMSE% over all sweep configs; MRE% (mean); R2 (min); pass-rate.

| Tech | Suite | Dev | base NRMSE% | mean NRMSE% | mean MRE% | min R2 | max MaxErr | Pass |
|---|---|---|---|---|---|---|---|---|
| tsmc5 | DC | nmos | 0.68 | 2.49 | 7.43 | 0.9466 | 16.638uA | 7/7 |
| tsmc5 | DC | pmos | 0.09 | 0.47 | 2.47 | 0.9949 | 8.847uA | 7/7 |
| tsmc5 | INV | all | 1.11 | 2.16 | 10.05 | 0.9902 | 241.731mV | 16/16 |
| tsmc7 | DC | nmos | 6.84 | 2.21 | 7.02 | 0.9541 | 20.766uA | 5/5 |
| tsmc7 | DC | pmos | 0.06 | 0.51 | 1.49 | 0.9972 | 3.889uA | 4/4 |
| tsmc7 | INV | all | 2.48 | 1.85 | 9.43 | 0.9847 | 303.898mV | 16/16 |
| tsmc12 | DC | nmos | 0.08 | 0.36 | 1.73 | 0.9946 | 38.91uA | 9/9 |
| tsmc12 | DC | pmos | 0.04 | 0.76 | 2.11 | 0.978 | 60.534uA | 9/9 |
| tsmc12 | INV | all | 1.14 | 1.59 | 9.51 | 0.9916 | 221.222mV | 16/16 |
| tsmc16 | DC | nmos | 0.04 | 0.51 | 1.91 | 0.9907 | 31.209uA | 7/7 |
| tsmc16 | DC | pmos | 0.03 | 0.65 | 1.84 | 0.9817 | 47.373uA | 7/7 |
| tsmc16 | INV | all | 1.03 | 1.96 | 9.98 | 0.9775 | 310.603mV | 16/16 |

### Complex circuits (per tech)

Gate verdict + headline + waveform NRMSE%/R2.

| Tech | Circuit | Gate | Headline | NRMSE% | R2 | MaxErr |
|---|---|---|---|---|---|---|
| tsmc5 | ring_osc | FAIL | period_err=5.89% | 74.28 | -2.0161 | 698.46mV |
| tsmc5 | opamp | FAIL | gain_err=100.00% trip_shift=74.00mV | 70.58 | -1.006 | 648.6mV |
| tsmc5 | sram_snm | PASS | NG_SNM=189.2mV DN_SNM=180.1mV  force_ic=ok/ok | — | — | — |
| tsmc5 | switchcap | FAIL | charge_err=11.84% | 15.6 | 0.6121 | 101.46mV |
| tsmc7 | ring_osc | FAIL | period_err=10.86% | 54.39 | -0.6537 | 783.43mV |
| tsmc7 | opamp | FAIL | gain_err=99.99% trip_shift=54.00mV | 70.08 | -1.0043 | 727.47mV |
| tsmc7 | sram_snm | PASS | NG_SNM=187.4mV DN_SNM=268.3mV  force_ic=ok/ok | — | — | — |
| tsmc7 | switchcap | PASS | charge_err=1.76% | 2.44 | 0.9908 | 31.46mV |
| tsmc12 | ring_osc | PASS | period_err=2.26% | 51.72 | -0.5414 | 844.36mV |
| tsmc12 | opamp | FAIL | gain_err=100.00% trip_shift=-10.00mV | 70.48 | -1.0053 | 797.4mV |
| tsmc12 | sram_snm | PASS | NG_SNM=239.1mV DN_SNM=226.2mV  force_ic=FAIL/FAIL | — | — | — |
| tsmc12 | switchcap | PASS | charge_err=4.19% | 5.78 | 0.9514 | 34.11mV |
| tsmc16 | ring_osc | PASS | period_err=2.22% | 48.64 | -0.3827 | 841.27mV |
| tsmc16 | opamp | FAIL | gain_err=100.00% trip_shift=148.00mV | 70.43 | -1.0047 | 797.4mV |
| tsmc16 | sram_snm | PASS | NG_SNM=235.5mV DN_SNM=222.1mV  force_ic=FAIL/FAIL | — | — | — |
| tsmc16 | switchcap | PASS | charge_err=3.22% | 5.08 | 0.9619 | 27.76mV |

### AC small-signal (per tech)

| Tech | Dev | gain0_err dB | f3db ratio | magNRMSE% | phase(inband)° | Gate |
|---|---|---|---|---|---|---|
| tsmc5 | nmos | 0.7 | 1.58 | 7.4 | 69.66 | FAIL |
| tsmc5 | pmos | 0.37 | 1.12 | 3.91 | 81.28 | PASS |
| tsmc7 | nmos | 1.32 | 1.26 | 13.4 | 55.62 | FAIL |
| tsmc7 | pmos | 0.79 | 1.26 | 8.18 | 67.15 | PASS |
| tsmc12 | nmos | 0.31 | 1.0 | 3.27 | 30.23 | PASS |
| tsmc12 | pmos | 1.3 | 1.12 | 12.74 | 34.61 | FAIL |
| tsmc16 | nmos | 0.19 | 1.0 | 2.05 | 32.56 | PASS |
| tsmc16 | pmos | 1.29 | 1.41 | 16.27 | 35.84 | FAIL |

Opamp open-loop:

| Tech | DC-gain err dB | GBW ratio | PM err° | magNRMSE% | Gate |
|---|---|---|---|---|---|
| tsmc5 | 13.87 | 5.76 | 44.4 | 54.63 | FAIL |
| tsmc7 | 31.89 | 1.16 | 51.4 | 71.23 | FAIL |
| tsmc12 | 3.15 | 19.6 | 50.7 | 27.69 | FAIL |
| tsmc16 | 8.34 | 16.3 | 39.3 | 41.41 | FAIL |

<details><summary>checkpoints resolved</summary>

- tsmc5: `tsmc5_dn_medium_nmos_best.pt`
- tsmc7: `tsmc7_dn_medium_nmos_best.pt`
- tsmc12: `tsmc12_dn_medium_nmos_best.pt`
- tsmc16: `tsmc16_dn_medium_nmos_best.pt`

</details>

## Size = large

### Device-level parametric sweeps (per tech)

DC = Id-Vgs over L/NFIN/VT; INV = inverter VTC+transient. Values: baseline NRMSE% / mean NRMSE% over all sweep configs; MRE% (mean); R2 (min); pass-rate.

| Tech | Suite | Dev | base NRMSE% | mean NRMSE% | mean MRE% | min R2 | max MaxErr | Pass |
|---|---|---|---|---|---|---|---|---|
| tsmc5 | DC | nmos | 4.06 | 3.99 | 13.02 | 0.9405 | 26.087uA | 7/7 |
| tsmc5 | DC | pmos | 0.02 | 1.21 | 3.88 | 0.9613 | 15.211uA | 7/7 |
| tsmc5 | INV | all | 1.03 | 1.98 | 10.22 | 0.9867 | 269.705mV | 16/16 |
| tsmc7 | DC | nmos | 2.02 | 1.84 | 4.66 | 0.9851 | 53.95uA | 5/5 |
| tsmc7 | DC | pmos | 0.04 | 0.65 | 1.55 | 0.9984 | 15.494uA | 4/4 |
| tsmc7 | INV | all | 1.79 | 1.92 | 10.7 | 0.9846 | 305.212mV | 16/16 |
| tsmc12 | DC | nmos | 0.02 | 1.25 | 3.09 | 0.9257 | 114.924uA | 9/9 |
| tsmc12 | DC | pmos | 0.02 | 2.2 | 5.99 | 0.723 | 203.19uA | 8/9 |
| tsmc12 | INV | all | 1.04 | 1.61 | 9.89 | 0.9915 | 221.848mV | 16/16 |
| tsmc16 | DC | nmos | 0.02 | 0.83 | 2.16 | 0.9903 | 40.379uA | 7/7 |
| tsmc16 | DC | pmos | 0.01 | 1.03 | 2.45 | 0.9745 | 63.252uA | 7/7 |
| tsmc16 | INV | all | 1.03 | 1.89 | 9.54 | 0.9754 | 212.951mV | 16/16 |

### Complex circuits (per tech)

Gate verdict + headline + waveform NRMSE%/R2.

| Tech | Circuit | Gate | Headline | NRMSE% | R2 | MaxErr |
|---|---|---|---|---|---|---|
| tsmc5 | ring_osc | FAIL | period_err=12.66% | 62.4 | -1.1285 | 696.15mV |
| tsmc5 | opamp | PASS | gain_err=2.10% trip_shift=-50.00mV | 36.19 | 0.4727 | 648.25mV |
| tsmc5 | sram_snm | PASS | NG_SNM=189.2mV DN_SNM=181.0mV  force_ic=FAIL/FAIL | — | — | — |
| tsmc5 | switchcap | FAIL | charge_err=11.20% | 14.55 | 0.6626 | 90.88mV |
| tsmc7 | ring_osc | PASS | period_err=4.82% | 63.39 | -1.2463 | 791.0mV |
| tsmc7 | opamp | FAIL | gain_err=99.99% trip_shift=114.00mV | 70.08 | -1.0043 | 727.47mV |
| tsmc7 | sram_snm | PASS | NG_SNM=187.4mV DN_SNM=315.2mV  force_ic=ok/ok | — | — | — |
| tsmc7 | switchcap | PASS | charge_err=1.52% | 1.74 | 0.9953 | 12.91mV |
| tsmc12 | ring_osc | PASS | period_err=4.04% | 62.97 | -1.2854 | 856.45mV |
| tsmc12 | opamp | PASS | gain_err=6.25% trip_shift=0.00mV | 1.01 | 0.9996 | 65.1mV |
| tsmc12 | sram_snm | PASS | NG_SNM=239.1mV DN_SNM=226.9mV  force_ic=ok/ok | — | — | — |
| tsmc12 | switchcap | PASS | charge_err=4.14% | 5.76 | 0.9518 | 34.25mV |
| tsmc16 | ring_osc | PASS | period_err=2.59% | 53.54 | -0.6754 | 846.43mV |
| tsmc16 | opamp | FAIL | gain_err=100.00% trip_shift=-60.00mV | 70.43 | -1.0047 | 797.4mV |
| tsmc16 | sram_snm | PASS | NG_SNM=235.5mV DN_SNM=222.9mV  force_ic=ok/ok | — | — | — |
| tsmc16 | switchcap | PASS | charge_err=3.32% | 5.34 | 0.9579 | 29.23mV |

### AC small-signal (per tech)

| Tech | Dev | gain0_err dB | f3db ratio | magNRMSE% | phase(inband)° | Gate |
|---|---|---|---|---|---|---|
| tsmc5 | nmos | 1.46 | nan | 14.7 | 72.42 | FAIL |
| tsmc5 | pmos | 0.37 | 1.12 | 3.89 | 81.19 | PASS |
| tsmc7 | nmos | 0.24 | 1.0 | 2.59 | 51.32 | PASS |
| tsmc7 | pmos | 0.92 | 1.26 | 9.49 | 67.13 | PASS |
| tsmc12 | nmos | 0.89 | 2.0 | 13.64 | 41.44 | FAIL |
| tsmc12 | pmos | 1.26 | 1.12 | 12.37 | 34.52 | FAIL |
| tsmc16 | nmos | 0.25 | 1.0 | 2.65 | 32.71 | PASS |
| tsmc16 | pmos | 1.48 | 1.26 | 14.38 | 34.61 | FAIL |

Opamp open-loop:

| Tech | DC-gain err dB | GBW ratio | PM err° | magNRMSE% | Gate |
|---|---|---|---|---|---|
| tsmc5 | 113.96 | n/a | n/a | 69.15 | FAIL |
| tsmc7 | 45.85 | 0.15 | 118 | 72.79 | FAIL |
| tsmc12 | 5.14 | 0.966 | 1.43 | 52.11 | FAIL |
| tsmc16 | 8.68 | 14.9 | 31.7 | 42.22 | FAIL |

<details><summary>checkpoints resolved</summary>

- tsmc5: `tsmc5_dn_large_nmos_best.pt`
- tsmc7: `tsmc7_dn_large_nmos_best.pt`
- tsmc12: `tsmc12_dn_large_nmos_best.pt`
- tsmc16: `tsmc16_dn_large_nmos_best.pt`

</details>

## Size = xl

### Device-level parametric sweeps (per tech)

DC = Id-Vgs over L/NFIN/VT; INV = inverter VTC+transient. Values: baseline NRMSE% / mean NRMSE% over all sweep configs; MRE% (mean); R2 (min); pass-rate.

| Tech | Suite | Dev | base NRMSE% | mean NRMSE% | mean MRE% | min R2 | max MaxErr | Pass |
|---|---|---|---|---|---|---|---|---|
| tsmc5 | DC | nmos | 3.95 | 4.12 | 13.77 | 0.938 | 20.701uA | 7/7 |
| tsmc5 | DC | pmos | 0.01 | 1.69 | 6.0 | 0.9501 | 21.258uA | 7/7 |
| tsmc5 | INV | all | 1.08 | 1.75 | 9.22 | 0.9903 | 185.865mV | 16/16 |
| tsmc7 | DC | nmos | 6.69 | 2.12 | 6.65 | 0.9561 | 20.813uA | 5/5 |
| tsmc7 | DC | pmos | 0.04 | 2.65 | 5.16 | 0.9731 | 50.859uA | 4/4 |
| tsmc7 | INV | all | 1.19 | 1.73 | 9.79 | 0.9844 | 306.138mV | 16/16 |
| tsmc12 | DC | nmos | 0.02 | 1.89 | 4.71 | 0.8639 | 149.493uA | 8/9 |
| tsmc12 | DC | pmos | 0.02 | 3.57 | 9.57 | 0.3683 | 291.143uA | 8/9 |
| tsmc12 | INV | all | 1.02 | 1.59 | 9.51 | 0.9915 | 222.411mV | 16/16 |
| tsmc16 | DC | nmos | 0.02 | 1.23 | 3.24 | 0.9516 | 40.039uA | 7/7 |
| tsmc16 | DC | pmos | 0.02 | 1.8 | 4.8 | 0.9025 | 98.302uA | 7/7 |
| tsmc16 | INV | all | 1.01 | 1.9 | 9.52 | 0.9755 | 212.948mV | 16/16 |

### Complex circuits (per tech)

Gate verdict + headline + waveform NRMSE%/R2.

| Tech | Circuit | Gate | Headline | NRMSE% | R2 | MaxErr |
|---|---|---|---|---|---|---|
| tsmc5 | ring_osc | FAIL | period_err=13.50% | 62.53 | -1.1374 | 693.85mV |
| tsmc5 | opamp | FAIL | gain_err=100.00% trip_shift=24.00mV | 70.58 | -1.006 | 648.6mV |
| tsmc5 | sram_snm | PASS | NG_SNM=189.2mV DN_SNM=181.8mV  force_ic=FAIL/FAIL | — | — | — |
| tsmc5 | switchcap | FAIL | charge_err=11.21% | 14.57 | 0.6617 | 91.03mV |
| tsmc7 | ring_osc | FAIL | period_err=14.31% | 58.91 | -0.94 | 783.49mV |
| tsmc7 | opamp | FAIL | gain_err=99.99% trip_shift=126.00mV | 70.08 | -1.0043 | 727.47mV |
| tsmc7 | sram_snm | PASS | NG_SNM=187.4mV DN_SNM=195.4mV  force_ic=FAIL/FAIL | — | — | — |
| tsmc7 | switchcap | PASS | charge_err=1.72% | 2.35 | 0.9914 | 30.68mV |
| tsmc12 | ring_osc | PASS | period_err=3.40% | 56.91 | -0.8664 | 842.63mV |
| tsmc12 | opamp | FAIL | gain_err=100.00% trip_shift=-8.00mV | 70.48 | -1.0053 | 797.4mV |
| tsmc12 | sram_snm | PASS | NG_SNM=239.1mV DN_SNM=226.7mV  force_ic=ok/ok | — | — | — |
| tsmc12 | switchcap | PASS | charge_err=4.19% | 5.82 | 0.9507 | 34.24mV |
| tsmc16 | ring_osc | PASS | period_err=3.05% | 59.29 | -1.0546 | 847.89mV |
| tsmc16 | opamp | FAIL | gain_err=100.00% trip_shift=-16.00mV | 70.43 | -1.0047 | 797.4mV |
| tsmc16 | sram_snm | PASS | NG_SNM=235.5mV DN_SNM=223.0mV  force_ic=FAIL/FAIL | — | — | — |
| tsmc16 | switchcap | PASS | charge_err=3.42% | 5.4 | 0.9569 | 29.72mV |

### AC small-signal (per tech)

| Tech | Dev | gain0_err dB | f3db ratio | magNRMSE% | phase(inband)° | Gate |
|---|---|---|---|---|---|---|
| tsmc5 | nmos | 1.58 | nan | 15.88 | 74.83 | FAIL |
| tsmc5 | pmos | 0.39 | 1.12 | 4.17 | 81.24 | PASS |
| tsmc7 | nmos | 1.07 | 1.41 | 10.93 | 57.52 | FAIL |
| tsmc7 | pmos | 0.85 | 1.12 | 8.86 | 66.9 | PASS |
| tsmc12 | nmos | 0.33 | 1.0 | 3.49 | 30.09 | PASS |
| tsmc12 | pmos | 1.28 | 1.12 | 12.56 | 34.63 | FAIL |
| tsmc16 | nmos | 0.26 | 1.0 | 2.75 | 32.72 | PASS |
| tsmc16 | pmos | 1.49 | 1.26 | 14.42 | 34.68 | FAIL |

Opamp open-loop:

| Tech | DC-gain err dB | GBW ratio | PM err° | magNRMSE% | Gate |
|---|---|---|---|---|---|
| tsmc5 | 86.09 | n/a | n/a | 69.15 | FAIL |
| tsmc7 | 19.48 | 4.64 | 24.4 | 65.08 | FAIL |
| tsmc12 | 42.69 | 0.125 | 115 | 67.64 | FAIL |
| tsmc16 | 99.77 | n/a | n/a | 66.64 | FAIL |

<details><summary>checkpoints resolved</summary>

- tsmc5: `tsmc5_dn_xl_nmos_best.pt`
- tsmc7: `tsmc7_dn_xl_nmos_best.pt`
- tsmc12: `tsmc12_dn_xl_nmos_best.pt`
- tsmc16: `tsmc16_dn_xl_nmos_best.pt`

</details>
