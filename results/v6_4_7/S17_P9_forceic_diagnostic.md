# V6.4.7 S17 (P9) — force_ic diagnostic — verdict: **P9 KILLED before any build**

**Date:** 2026-06-16 · User-directed "attempt P9, diagnostic-first." Phase-1
diagnosis (`scripts/v6_4_7_s17_forceic_decomp.py`, generalises the V6.4.6 P0-D
dump to any tech + current code + the promoted checkpoint): at the stuck
`force_ic` fixed point, decompose the "0"-storage-node KCL into per-device
NN-vs-OSDI `id`. **P9's premise — the inboard attractor is held by OFF-PMOS
leakage — is FALSIFIED.** force_ic has two failure modes, neither OFF-leakage.

## tsmc16 (s12cor_w3_s17, promoted) — INBOARD mode (qb stuck = 116.7 mV)

| device | role | Vgs/Vds (mV) | class | NN id | OSDI id | NN/OSDI | err into qb |
|---|---|---|---|---|---|---|---|
| **Mnl** | **ON driver NMOS** | 800 / 117 | ON | **−56.30 µA** | **−61.12 µA** | **0.921** | **−4.82 µA (8% UNDER-pull)** |
| Mar | ON access NMOS | 683 / 683 | ON | −59.56 µA | −59.55 µA | 1.000 | −6 nA (exact) |
| Mpl | **OFF load PMOS** | 0 / −683 | OFF | **0.0** | **−0.0** | — | **0 (no leakage error)** |

The node is stuck at 117 mV because the **strongly-ON driver NMOS under-predicts
its LINEAR-region pull-down by ~8 % (4.82 µA)** — it can't sink the
(accurately-modelled) access pull-up, so the cell balances inboard. Substituting
OSDI currents at the same node voltages gives a net ~1.55 µA pulling qb toward
ground (OSDI would rail it). **The OFF load PMOS leakage is exactly 0 (NN=OSDI)**
— P9 has nothing to fix here. The error is **strong-inversion / linear-region
Rdson**, the opposite end of the id surface from P9's target.

## tsmc7 (pivcor_w2_s7, promoted) — SYMMETRIC-SADDLE mode (q=qb=388 mV=VDD/2)

Every device is OSDI-accurate (NN/OSDI ratio **0.999–1.000**, per-device errors
≤23 nA); the OSDI KCL residual at the symmetric point is ~13 nA ≈ 0. The
unconstrained re-solve simply **converges to the metastable saddle** instead of a
railed state. **No model current error to fix** — this is a fixed-point-selection
/ regenerative-gain problem (the P0-A constraint-continuation homotopy that
targeted exactly this was already built and KILLED: the railed point is
NR-unstable). P9 (or any id-accuracy lever) is irrelevant to this mode.

## Verdict — P9 KILLED (pre-build, by diagnosis)

force_ic is **not OFF-leakage-owned**. The two modes are:
1. **inboard** (tsmc16/12, most seeds) — ON-driver **linear-region id under-pull
   ~8 %** (an id-VALUE error at high Vgs / low Vds);
2. **symmetric saddle** (tsmc7 pivcor) — all currents exact, a solver
   fixed-point-selection / gain problem.

P9's compose-at-inference OFF core addresses the OFF/subthreshold region, where
the diagnostic shows **zero error**. Building it would not move force_ic. **No
P9 code written; dead-end recorded** (the diagnostic-first protocol working as
intended — ~5 min of probing avoided a multi-hundred-LOC structural build). This
also closes the S11 tension cleanly: S11 said "force_ic is gain/NR-fixed-point
owned, a more-accurate id removes the bistability"; the decomposition refines
that — the *inboard* mode is a specific linear-region id error, the *symmetric*
mode is pure fixed-point selection.

## The real (uncertain, out-of-scope) force_ic levers

- **inboard mode →** improve the NN driver's **linear-region** (high-Vgs,
  low-Vds) id accuracy ~8 %: a P5-style corridor retrain harvesting the *force_ic*
  driver biases, or a linear-region-weighted id loss. A retrain (S10/S11 collapse
  risk), and it would NOT touch the tsmc7 saddle mode.
- **symmetric-saddle mode →** a solver-side asymmetric-release homotopy (distinct
  from the killed P0-A symmetric continuation) or a higher-gain model. Research-grade.

Both are new levers beyond the V6.4.7 cheap-lever scope. **Recommendation: ship
14/16 with force_ic as a documented known-issue; pursue the linear-region
corridor only if force_ic must close.** Driver/probe: `scripts/v6_4_7_s17_forceic_decomp.py`.
(Caveat: the script's aggregate "into node" sign label is cosmetically inverted;
the per-device error attribution — the decisive output — is correct.)
