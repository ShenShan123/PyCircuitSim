# Phase 0 P0-D — SRAM off-transistor attractor instrumentation (TSMC7)

**Date:** 2026-06-01  •  **Branch:** `feat/v6.4.6`  •  **Status:** Done
**Env:** `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `pycircuitsim` conda env.
**Decides:** Open Q3 / plan §4 P0-D — retire or keep the Phase-3
leak-magnitude / off-floor / skeleton family.
**Ground truth:** OSDI BSIM-CMG via PyCMG `eval_single_point` (CLAUDE.md).

---

## Commands

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run -n pycircuitsim python scripts/v6_4_6_p0d_sram_attractor.py \
  > results/v6_4_6/phase0_logs/p0d_attractor.log 2>&1
```

The script (1) builds the TSMC7 6T `force_ic` cell via `_directnet_6t_netlist`,
solves both states with `DCSolver(..., force_ic=True)` to the **stuck**
unconstrained solution, (2) enumerates all 6 MOSFETs, computes `(Vgs,Vds,Vbs)`,
classifies on/off, dumps post-Rule-15 NN `id`/`gds` (`mosfet._eval`) and analytic
OSDI `id`/`gm`/`gds` at the *same absolute bias*, and (3) derives a per-device
constant-current OSDI VTH. A follow-on Id–Vgs NN-vs-OSDI overlay at the
attractor `Vds` was run for the critical pull-down device.

## Per-device OSDI turn-on VTH (constant-current, `|Id| ≥ 1e-7·NFIN = 2.0e-7 A` at Vds=VDD)

| Device | VTH (constant-current) |
|--------|------------------------:|
| NMOS   | **181.4 mV** (Vgs)      |
| PMOS   | **−222.6 mV** (Vgs)     |

VTH definition: linear-interpolated Vgs at which `|Id|` crosses
`Icrit = 1e-7·NFIN` on an OSDI Id–Vgs sweep at `Vds = VDD`, Vs=Vb=0 (NMOS) /
source-at-VDD frame (PMOS). This is the standard constant-current threshold,
scaled by fin count.

## Stuck node voltages (state1, seed q=VDD, qb=0)

`V(vdd)=750  V(q)=815.1  V(qb)=226.4  V(bl)=V(blb)=V(wl)=750` (mV). state0 is the
exact mirror (`q=226.4, qb=815.1`). The latch fails to rail: the "0" node
**qb sits at 226 mV** instead of 0, the "1" node q overshoots to 815 mV.

## Per-transistor table at the stuck solution (state1; state0 is the mirror)

| Dev | type | D/G/S/B    | Vgs(mV) | Vds(mV) | class            | NN id (A)   | OSDI id (A) | \|NN/OSDI\| | NN gds     | OSDI gds    | OSDI gm    |
|-----|------|------------|--------:|--------:|------------------|------------:|------------:|------------:|-----------:|------------:|-----------:|
| Mpl | pmos | qb/q/vdd/vdd |  +65.1 | −523.6 | OFF (Vov=−287.7) | +4.97e-07  | −0.0e+00    | ∞           | 9.83e-07  | 2.81e-12   | 1.83e-10  |
| Mnl | nmos | qb/q/0/0     | +815.1 | +226.4 | ON  (Vov=+633.7) | −1.230e-04 | −1.399e-04  | 0.880       | 7.95e-05  | 1.68e-04   | 2.72e-04  |
| Mpr | pmos | q/qb/vdd/vdd | −523.6 |  +65.1 | ON  (Vov=+301.0) | −0.0e+00   | −1.985e-05  | 0.000       | 8.78e-05  | −3.27e-04  | 6.48e-05  |
| **Mnr** | **nmos** | **q/qb/0/0** | **+226.4** | **+815.1** | **NEAR-VTH (Vov=+45.0)** | **−6.36e-06** | **−8.43e-07** | **7.54** | 1.74e-04 | 2.72e-07 | 2.47e-05 |
| Mal | nmos | bl/wl/q/0    |  −65.1 |  −65.1 | OFF (Vov=−246.5) | −0.0e+00   | −0.0e+00    | ∞ (0/0)     | 7.05e-04  | −7.61e-09  | 6.86e-09  |
| Mar | nmos | blb/wl/qb/0  | +523.6 | +523.6 | ON  (Vov=+342.2) | −1.255e-04 | −5.716e-05  | 2.20        | 6.28e-05  | 6.87e-06   | 3.23e-04  |

**Reading the latch.** Node qb (the intended "0") is held *up* at 226 mV by a
current balance on its pull-down/pull-up. The dominant culprit is **Mnr**, the
pull-down of the high node q (gate tied to qb): at the attractor it sits at
**Vgs = 226 mV ≈ VTH(181 mV), Vov = +45 mV (NEAR-VTH / weak inversion)** and the
NN sources **−6.36 µA vs OSDI −0.84 µA — 7.5× over-modelled current**. The OFF
PMOS pull-up **Mpl** *also* injects a spurious **+0.50 µA where OSDI gives 0**
(Vgs=+65 mV, a hard-off PMOS). These two over-modelled currents pin a
cross-coupled equilibrium ~226 mV off the rail.

## Id–Vgs NN-vs-OSDI overlay — the over-modelled leakage floor (Mnr device, Vds=815.1 mV, Vs=Vb=0)

| Vgs (mV) | NN id (A)   | OSDI id (A) | \|NN/OSDI\| |
|---------:|------------:|------------:|------------:|
|    0.0   | −5.665e-06  | −0.0e+00    | ∞           |
|   50.0   | −5.653e-06  | −2.133e-09  | 2650        |
|  100.0   | −5.632e-06  | −1.247e-08  | 452         |
|  150.0   | −5.668e-06  | −7.127e-08  | 79.5        |
|  175.0   | −5.766e-06  | −1.660e-07  | 34.7        |
|  225.0   | −6.330e-06  | −8.089e-07  | 7.83        |
|  300.0   | −9.618e-06  | −5.197e-06  | 1.85        |
|  400.0   | −2.454e-05  | −2.307e-05  | 1.06        |

**Rule-16 quartet over the sweep:** MRE = 30136 % (dominated by the deep-OFF
band where OSDI≈0), R² = 0.466, NRMSE = 21.49 %, MaxErr = 5.66e-6 A.

The NN carries a **flat ~5.66 µA subthreshold floor** across the *entire* OFF
region (Vgs 0→150 mV) where OSDI rolls off ≥3–4 decades to near-zero. The two
surfaces only converge once Vgs > ~350 mV (strong inversion). The `asinh_scale_id`
normalisation crushes the 1 nA–1 µA leakage band ~6 decades below the on-state,
so the subthreshold roll-off carries near-zero training loss gradient — exactly
the mechanism the plan §2 predicted. **This µA-scale over-modelled floor is the
~µA the latch needs moved (per plan §10, the safe-nA `Ioff_rail` regime was
SRAM-impotent precisely because it could not source µA).**

---

## DECISIONS

### (a) Off-leakage OVER-MODELLED, or already ≈0?

> **OVER-MODELLED — decisively.** In the deep-OFF band the NN sources a flat
> ~5.66 µA where OSDI gives ~0 (ratio 80–2650×). At the actual attractor bias the
> critical near-VTH pull-down **Mnr** is 7.5× over (−6.36 µA NN vs −0.84 µA OSDI),
> and the hard-off PMOS pull-up **Mpl** injects +0.50 µA where OSDI is 0. The
> attractor is **leakage-magnitude driven**, not charge/homotopy-driven. ⇒ A
> Phase-3 multi-region skeleton / closed-form subthreshold core that suppresses
> the OFF current ≥3–4 decades CAN, in principle, help.

### (b) OFF device DEEP-OFF or MODERATE inversion?

> **MIXED — and that is the killer caveat.** The genuinely deep-OFF devices
> (Mpl pull-up at Vov=−288 mV; Mal access at Vov=−246 mV) are *safe* to gate,
> but the device that actually pins the attractor — **Mnr — sits at Vov=+45 mV,
> NEAR-VTH / weak inversion (Vgs 226 mV vs VTH 181 mV)**. Any off-state gate
> `S(Vov)` aggressive enough to crush Mnr's 7.5×-over current would fire in the
> SAME weak-inversion region the inverter trip lives in, where the ~20× VTC gain
> amplifies the perturbation — **this is D4 territory** (the V6.4.5 `Ioff_rail`
> dead end collapsed inverter VTC 1.21→11.56 % for exactly this reason).

### Net verdict for the Phase-3 leak/skeleton family

> **KEEP-WITH-STRONG-CAVEAT, but it is NOT the lever of first resort.** Plan §4
> says: `OSDI id≈0 / Vgs≈VTH ▶ Phase 3 skeleton DEAD`. P0-D lands in the
> *boundary* case: the leak IS over-modelled (the "alive" condition (a)), BUT
> the pinning device is at Vgs≈VTH (the "dead" condition (b)). The two conditions
> point opposite ways. **Because P0-A proved a railed fixed point EXISTS, the
> SRAM gate is closable at 0 GPU via the Phase-1 solver homotopy without touching
> the model at all** — so the correct sequencing is: **close SRAM via Phase 1
> first; fund Phase 3 ONLY if Phase 1 cannot rail the latch.** If Phase 3 is ever
> funded, the off-gate MUST be validated against the inverter VTC regression
> budget on the same checkpoint (the Vov=+45 mV proximity to the trip is the top
> risk), and the fit gate (≥4-decade suppression with `n≤1.3` AND ≤5 % inv_trip
> *simultaneously*) is the go/no-go.

(Rule 16: single-bias comparisons report raw id/gds/gm + ratios in the table;
the Id–Vgs overlay reports the MRE/R²/NRMSE/MaxErr quartet.)
