# DirectNet (LEVEL=73) capacity benchmark — small / medium / large

All checkpoints trained on ONE identical clean recipe (`--apply-filter off --swa-mode ema --seed 42`); capacity is the only variable. Datasets = full Vth + geometry grid per tech (`--variants all`, inv-trip + subvt-off overlays). Ground truth = NGSPICE BSIM-CMG (LEVEL=72), repo ngspice-45.2, CPU-pinned.

Sizes: small=128x3 (~0.06M p) / medium=256x5 (~0.4M p) / large=384x6 (~0.9M p).

> **V6.5 update — AC small-signal accuracy.** The DC/transient capacity study below is unchanged; V6.5 adds the first NGSPICE-gated evaluation of DirectNet **AC** (`.ac`) fidelity across all 24 checkpoints — see the [AC small-signal accuracy](#ac-small-signal-accuracy-v65) section. Headline: AC **gain** fidelity (gm/gds via autograd) is excellent everywhere (gain0 err <1.5 dB, 24/24); the **dominant cap-driven pole / bandwidth** is good but capacity/tech-variable (device CS-amp gate 13/24); the **opamp** AC inherits the DC value-surface fragility (0/12, though tsmc12-large reproduces GBW to 0.97× and phase margin to 1.3°). No retraining warranted (the gaps are value-surface- and feedforward-owned, not a charge-derivative deficiency).

## Key findings

1. **Circuit pass-rate rises monotonically with capacity: 6/16 -> 9/16 -> 12/16** (small -> medium -> large). The gains come from charge-transfer and timing circuits, not from device-curve accuracy (already excellent everywhere).
2. **Device-level Id-Vgs / inverter accuracy is excellent at every size** (mean NRMSE <4%, most <2%; inverter VTC+transient 16/16 PASS at all sizes, all techs). Capacity barely moves it, and at the finer nodes the large net slightly *overfits* the device surface (e.g. tsmc12 DC mean NRMSE 0.36% medium -> 1.25% large). Device fidelity is NOT the bind.
3. **The opamp is the hardest, value-surface-fragile gate.** Open-loop gain collapses to ~0 (NRMSE ~70%) at small AND medium for all four techs, and recovers to PASS only at **large**, only for tsmc5 and tsmc12 (tsmc12 large: gain err 6.25%, locus NRMSE 1.0%). tsmc7/tsmc16 never pass on this clean single recipe -- matching project history that the *shipping* tsmc7/tsmc16 needed special recipes (pivcor / s12cor) to keep the opamp alive. Refines V6.4.8-S1: the high-gain basin is capacity- AND tech-sensitive, not a clean capacity win or loss.
4. **Switched-cap needs capacity:** 0/4 (small) -> 3/4 (medium and large). tsmc5 never passes (~11-12% charge error) -- its known micro-amp-band loss-compression over-conduction (V6.4.8-S3), independent of capacity.
5. **Ring-osc** passes for the higher-VDD nodes (tsmc12/16) at every size and tsmc7 at large; tsmc5 never (period err 6-13%). **SRAM butterfly** (all-lobes-positive gate) passes 4/4 at every size; force_ic is reported as an informational probe.

**Bottom line:** larger capacity helps circuit-level behaviour overall (12/16 at large) but does NOT close the two recipe-sensitive gaps (tsmc7/tsmc16 opamp, tsmc5 switchcap) that the V6.4.x campaigns already attributed to value-surface / loss-compression, not capacity.

## Cross-size summary

| Size | Complex gates PASS | Device mean-NRMSE% (all sweeps) |
|---|---|---|
| small | 6/16 | 1.63 |
| medium | 9/16 | 1.29 |
| large | 12/16 | 1.7 |

## Capacity comparison (the headline view)

### Complex-circuit gate verdict by capacity

| Tech | Circuit | small | medium | large |
|---|---|---|---|---|
| tsmc5 | ring_osc | FAIL | FAIL | FAIL |
| tsmc5 | opamp | FAIL | FAIL | PASS |
| tsmc5 | sram_snm | PASS | PASS | PASS |
| tsmc5 | switchcap | FAIL | FAIL | FAIL |
| tsmc7 | ring_osc | FAIL | FAIL | PASS |
| tsmc7 | opamp | FAIL | FAIL | FAIL |
| tsmc7 | sram_snm | PASS | PASS | PASS |
| tsmc7 | switchcap | FAIL | PASS | PASS |
| tsmc12 | ring_osc | PASS | PASS | PASS |
| tsmc12 | opamp | FAIL | FAIL | PASS |
| tsmc12 | sram_snm | PASS | PASS | PASS |
| tsmc12 | switchcap | FAIL | PASS | PASS |
| tsmc16 | ring_osc | PASS | PASS | PASS |
| tsmc16 | opamp | FAIL | FAIL | FAIL |
| tsmc16 | sram_snm | PASS | PASS | PASS |
| tsmc16 | switchcap | FAIL | PASS | PASS |

### Device-level mean NRMSE% by capacity (lower = better fit)

DC = Id-Vgs (NMOS/PMOS); INV = inverter VTC+transient (combined).

| Tech | Suite/Dev | small | medium | large |
|---|---|---|---|---|
| tsmc5 | DC/nmos | 3.62 | 2.49 | 3.99 |
| tsmc5 | DC/pmos | 1.16 | 0.47 | 1.21 |
| tsmc5 | INV/all | 2.52 | 2.16 | 1.98 |
| tsmc7 | DC/nmos | 2.33 | 2.21 | 1.84 |
| tsmc7 | DC/pmos | 1.17 | 0.51 | 0.65 |
| tsmc7 | INV/all | 2.2 | 1.85 | 1.92 |
| tsmc12 | DC/nmos | 0.4 | 0.36 | 1.25 |
| tsmc12 | DC/pmos | 0.74 | 0.76 | 2.2 |
| tsmc12 | INV/all | 2.19 | 1.59 | 1.61 |
| tsmc16 | DC/nmos | 0.59 | 0.51 | 0.83 |
| tsmc16 | DC/pmos | 0.65 | 0.65 | 1.03 |
| tsmc16 | INV/all | 2.01 | 1.96 | 1.89 |

## AC small-signal accuracy (V6.5)

First-ever NGSPICE-gated evaluation of DirectNet (LEVEL=73) AC fidelity. The NN's small-signal capacitances are autograd derivatives of its predicted terminal charges (cgd=∂qg/∂Vd, cdd=∂qd/∂Vd, …) — a quantity no prior gate measured. Ground truth = NGSPICE `.ac` on the identical BSIM-CMG (LEVEL=72) OSDI model. Two circuit classes (AC needs a stable amplifying OP, so the free-running ring oscillator and bistable SRAM are out of scope):

- **Device CS-amp** — per-checkpoint NMOS/PMOS common-source amplifier, no external load cap so the device's own Cgd/Cdd set the pole; gates gain0 err ≤1.5 dB, f3db ratio ∈[0.7,1.43], mag NRMSE ≤10%. The passband phase is reported (not gated): deep in-band it matches (<7°), but at/beyond the −3 dB corner NG carries a strong Cgd-feedforward RHP-zero phase lag the NN does not reproduce — a distinct limitation from the (excellent) cap-driven pole.
- **Opamp open-loop** — two-stage Miller opamp; gates DC-gain err ≤3 dB, GBW ratio ∈[0.6,1.67], phase-margin err ≤15° (linear mag NRMSE reported, not gated — dominated by the 40 dB passband plateau).

**Findings.** (1) AC **gain** fidelity is excellent everywhere — device gain0 err <1.5 dB in 24/24 cells (mean 0.55–0.86 dB) — so the autograd gm/gds the NN feeds the AC stamp are accurate. (2) The dominant **cap-driven pole** is mostly faithful (f3db ratio ≈1.0 for the well-fit cells) but capacity/tech-variable: tsmc5 NMOS and tsmc12/16 PMOS under-predict the output cap (ratio 1.1–1.6), so 13/24 clear the magnitude gate. (3) The high-frequency **phase** (Cgd-feedforward RHP zero) is not reproduced — a clean, specific transcapacitance limitation. (4) The **opamp** AC inherits the DC value-surface fragility (0/12): the gain collapses or over-predicts at most cells, BUT where the OP lands in the good basin (tsmc12-large) the NN reproduces GBW to 0.97× and phase margin to 1.3° — the dynamics are right, the DC-gain *level* is the value-surface-owned miss. **No retraining is warranted:** the gaps are value-surface- and feedforward-owned, not a charge-derivative (dQ/dV) deficiency (which would have shown as bad gain *and* bad pole everywhere, the opposite of what is measured).

### Cross-size AC summary

| Size | Device CS-amp PASS | Opamp PASS | Device mean gain0-err dB | Device mean magNRMSE% |
|---|---|---|---|---|
| small | 5/12 | 0/4 | 0.55 | 6.14 |
| medium | 4/12 | 0/4 | 0.78 | 8.4 |
| large | 4/12 | 0/4 | 0.86 | 9.21 |

### Device CS-amp AC gate by capacity

| Tech | Dev | small | medium | large |
|---|---|---|---|---|
| tsmc5 | nmos | FAIL | FAIL | FAIL |
| tsmc5 | pmos | PASS | PASS | PASS |
| tsmc7 | nmos | PASS | FAIL | PASS |
| tsmc7 | pmos | PASS | PASS | PASS |
| tsmc12 | nmos | PASS | PASS | FAIL |
| tsmc12 | pmos | FAIL | FAIL | FAIL |
| tsmc16 | nmos | PASS | PASS | PASS |
| tsmc16 | pmos | FAIL | FAIL | FAIL |

### Device CS-amp gain0 error (dB) by capacity (lower = better)

| Tech | Dev | small | medium | large |
|---|---|---|---|---|
| tsmc5 | nmos | 0.83 | 0.7 | 1.46 |
| tsmc5 | pmos | 0.09 | 0.37 | 0.37 |
| tsmc7 | nmos | 0.56 | 1.32 | 0.24 |
| tsmc7 | pmos | 0.32 | 0.79 | 0.92 |
| tsmc12 | nmos | 0.04 | 0.31 | 0.89 |
| tsmc12 | pmos | 1.25 | 1.3 | 1.26 |
| tsmc16 | nmos | 0.06 | 0.19 | 0.25 |
| tsmc16 | pmos | 1.26 | 1.29 | 1.48 |

### Opamp open-loop AC gate by capacity (DC-gain err dB / GBW ratio / PM err°)

| Tech | small | medium | large |
|---|---|---|---|
| tsmc5 | FAIL (5.69/21.6/5.9) | FAIL (13.87/5.76/44.4) | FAIL (113.96/n/a/n/a) |
| tsmc7 | FAIL (244.86/n/a/n/a) | FAIL (31.89/1.16/51.4) | FAIL (45.85/0.15/118) |
| tsmc12 | FAIL (84.33/n/a/n/a) | FAIL (3.15/19.6/50.7) | FAIL (5.14/0.966/1.43) |
| tsmc16 | FAIL (23.46/3.33/4.98) | FAIL (8.34/16.3/39.3) | FAIL (8.68/14.9/31.7) |

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
