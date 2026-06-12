# S7_P2_rev_probe — raw reverse-Vds surface vs OSDI (P2 pre-build go/no-go)

Raw tap: identity-monkeypatched `_MOSFETNNBase._apply_vds_correction` (post-denorm, pre-correction). Ground truth: OSDI via PyCMG `eval_single_point` at identical absolute bias (sign-consistency verified at forward bias, all 8 tech/device cells). Corridor grid: Vds -[0.30..0.01]*VDD x Vgs {0.2..1.0}*VDD, Vbs=0, mirrored for PMOS; 60 pts per cell.

## Corridor Rule-16 + agreement (raw NN id vs OSDI id)

| tech | dev | MRE % | R2 | NRMSE % | MaxErr uA | sign-agree (meaningful) | median ratio | <2x | <10x | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| TSMC5 | nmos | 27.65 | 0.9274 | 7.41 | 25.925 | 0.955 (n=44) | 0.789 | 0.932 | 0.977 | **USABLE** |
| TSMC5 | pmos | 34.88 | 0.9065 | 8.07 | 21.595 | 0.951 (n=41) | 0.722 | 0.805 | 0.951 | **USABLE** |
| TSMC7 | nmos | 41.99 | 0.7822 | 13.09 | 55.890 | 1.000 (n=54) | 0.641 | 0.852 | 0.963 | **USABLE** |
| TSMC7 | pmos | 30.97 | 0.8907 | 8.93 | 27.224 | 1.000 (n=52) | 0.753 | 0.846 | 1.000 | **USABLE** |
| TSMC12 | nmos | 30.61 | 0.9269 | 7.57 | 28.265 | 0.979 (n=47) | 0.761 | 0.851 | 0.957 | **USABLE** |
| TSMC12 | pmos | 31.83 | 0.9015 | 8.57 | 28.115 | 0.979 (n=47) | 0.750 | 0.872 | 0.936 | **USABLE** |
| TSMC16 | nmos | 30.08 | 0.9272 | 7.46 | 28.156 | 1.000 (n=47) | 0.760 | 0.872 | 0.979 | **USABLE** |
| TSMC16 | pmos | 32.15 | 0.9111 | 8.06 | 27.520 | 0.956 (n=45) | 0.751 | 0.911 | 0.956 | **USABLE** |

## SRAM force_ic state1 attractor — reverse-biased devices (the P2 beneficiaries)

| tech | device | Vgs mV | Vds mV | raw NN id | post-clamp id | OSDI id | raw/OSDI | sign |
|---|---|---|---|---|---|---|---|---|
| TSMC5 | Mpl (pmos, REV) | -565.6 | 25.9 | -5.592e-06 | -0.000e+00 | -8.029e-06 | 0.696 | Y |
| TSMC5 | Mar (nmos, REV) | -25.9 | -25.9 | +2.095e-07 | +0.000e+00 | -0.000e+00 | nan | n |
| TSMC7 | Mpr (pmos, REV) | -632.7 | 34.6 | -1.018e-05 | -0.000e+00 | -1.371e-05 | 0.743 | Y |
| TSMC7 | Mal (nmos, REV) | -34.6 | -34.6 | -3.387e-08 | -0.000e+00 | -0.000e+00 | nan | Y |
| TSMC12 | Mpr (pmos, REV) | -695.8 | 37.0 | -1.134e-05 | -0.000e+00 | -1.536e-05 | 0.739 | Y |
| TSMC12 | Mal (nmos, REV) | -37.0 | -37.0 | +1.448e-07 | +0.000e+00 | -0.000e+00 | nan | n |
| TSMC16 | Mpr (pmos, REV) | -693.6 | 36.0 | -1.036e-05 | -0.000e+00 | -1.437e-05 | 0.721 | Y |
| TSMC16 | Mal (nmos, REV) | -36.0 | -36.0 | -1.765e-07 | -0.000e+00 | -0.000e+00 | nan | Y |

## Beyond-corridor (Vgs=0.6*VDD)

| tech | dev | Vds | raw NN id | OSDI id | raw/OSDI | sign |
|---|---|---|---|---|---|---|
| TSMC5 | nmos | -0.4*VDD | +4.631e-05 | +7.532e-05 | 0.615 | Y |
| TSMC5 | nmos | -0.5*VDD | +5.177e-05 | +9.990e-05 | 0.518 | Y |
| TSMC5 | pmos | -0.4*VDD | -2.396e-05 | -4.859e-05 | 0.493 | Y |
| TSMC5 | pmos | -0.5*VDD | -2.733e-05 | -6.823e-05 | 0.401 | Y |
| TSMC7 | nmos | -0.4*VDD | +7.483e-05 | +1.298e-04 | 0.576 | Y |
| TSMC7 | nmos | -0.5*VDD | +8.090e-05 | +1.584e-04 | 0.511 | Y |
| TSMC7 | pmos | -0.4*VDD | -5.331e-05 | -8.747e-05 | 0.609 | Y |
| TSMC7 | pmos | -0.5*VDD | -5.869e-05 | -1.120e-04 | 0.524 | Y |
| TSMC12 | nmos | -0.4*VDD | +5.460e-05 | +9.310e-05 | 0.587 | Y |
| TSMC12 | nmos | -0.5*VDD | +6.052e-05 | +1.205e-04 | 0.502 | Y |
| TSMC12 | pmos | -0.4*VDD | -4.691e-05 | -8.060e-05 | 0.582 | Y |
| TSMC12 | pmos | -0.5*VDD | -5.208e-05 | -1.045e-04 | 0.498 | Y |
| TSMC16 | nmos | -0.4*VDD | +5.260e-05 | +8.910e-05 | 0.590 | Y |
| TSMC16 | nmos | -0.5*VDD | +5.825e-05 | +1.159e-04 | 0.502 | Y |
| TSMC16 | pmos | -0.4*VDD | -4.176e-05 | -7.442e-05 | 0.561 | Y |
| TSMC16 | pmos | -0.5*VDD | -4.653e-05 | -9.850e-05 | 0.472 | Y |

## Near-zero-Vds structure (post-hoc CSV analysis — drives the P2 blend design)

* **Every meaningful-subset sign failure sits at the innermost grid point**
  (|Vds| = 0.01·VDD ≈ 6.5–8 mV): 8 failures total, all at vds index 0. At
  |Vds| ≥ 0.03·VDD the meaningful-subset sign agreement is 100 % on all 8
  cells. The raw surface does NOT pass through Id(Vds=0)=0 — the deep-linear
  edge is sub-µA noise with random sign.
* **Noise a relaxation would inject where OSDI is ~0** (|OSDI id| < 1 µA,
  6–19 pts/cell): max |raw id| = 0.32–0.76 µA, median 0.06–0.61 µA
  (worst cell TSMC7 NMOS). This is the per-device noise bound for every
  circuit crossing Vds = 0 if (b)+(d) are relaxed without a near-zero taper.
* **Conductance caveat at the SRAM attractor:** OSDI gds at the live Mpr
  reverse bias is ~3.2–4.3e-4 S (NN frame, the ~3e2 µS scale the railed
  point's NR-stability needs), but the raw NN autograd gds there sits BELOW
  the Rule-5 floor (printed values = 0.5·|id| ≈ 2.8–5.7 µS, i.e. the floor)
  — ~80× short. P2 gets the restoring *current* (~72–74 % of OSDI) but not
  automatically the restoring *conductance*; the Jacobian path needs its own
  look (the floored autograd slope was not separately recorded by this probe).
* Production at the attractor also shows a Rule-15(a) artifact: the OFF
  pull-down NMOS (Mnr / Mnl at Vds ≈ 0.68–0.84 V > VDD_train) carries a
  phantom −1.0…−1.7 µA rail-restoring ramp current (OSDI ≤ 2.3e-8 A) while
  the clamp denies the ON PMOS its −8…−15 µA restoring current — both errors
  act on the same storage node.

## Verdict

**USABLE — go for P2, with a near-zero taper.** All 8 tech/device cells:
sign-correct (0.95–1.00) and order-of-magnitude-correct (median raw/OSDI
0.64–0.79, within-10x 0.94–1.00) wherever OSDI conducts > 1 µA; R² 0.78–0.93,
NRMSE 7.4–13.1 %. The raw current is a systematic ~25–35 % UNDER-prediction
(conservative, not noisy). At the live SRAM attractor the clamp withholds
72–74 % of the OSDI restoring current (raw −5.6…−11.3 µA vs OSDI
−8.0…−15.4 µA, sign-correct on all 4 techs). Beyond the corridor
(−0.4/−0.5·VDD) the surface stays sign-correct and degrades smoothly
(ratio 0.62 → 0.40) — clamp saturation, not garbage; the plan's taper past
−0.30·VDD remains advisable but is not load-bearing. The relaxation MUST
keep an Id(Vds=0)=0 blend (e.g. retain a one-sided 1−exp-style factor in
the reverse direction instead of f_id=0) because the raw surface at
|Vds| ≤ 0.01·VDD is sub-µA random-sign noise.

Full per-point data: `s7_rev_probe_corridor.csv`; log: `s7_logs/s7_rev_probe.log`.
