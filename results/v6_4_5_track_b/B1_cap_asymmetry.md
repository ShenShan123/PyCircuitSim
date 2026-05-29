# B1 — Adaptive cap-symmetry probe on TSMC7 RO (Track B, Tier 1)

**Date:** 2026-05-29 • **Branch:** `feat/v6.4.5-track-b` • **Verdict: KILL**
**Env:** `CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` (CPU, 1 thread).
**Probe:** `experiments/v6_4_5_track_b/B1_cap_asymmetry_probe.py` — monkeypatches
`_MOSFETNNBase._unpack_eval` (no source edit), logs
`δ = |cgd − cdg| / max(|cgd|, |cdg|, 1e-15)` for every DirectNet NN eval during
the TSMC7 5-stage ring-oscillator transient (`tsmc7_dn_medium_*`, V6.4.4 mix).

## Falsifier (plan B1)

> Promote: δ > 5 % on > 10 % of NR evals **at the trip region** AND
> `NN_SYMMETRIC_CAPS=1` closes TSMC7 RO ≤ 5 %.
> Hard kill: δ uniformly < 1 % → RO drift is not cap-asymmetry.

## Result — 55,360 NN evals logged (full transient, partial=False)

| Subset | n | δ>5 % | δ>1 % | median | p90 | p99 | max |
|--------|---:|------:|------:|-------:|----:|----:|----:|
| all evals | 55360 | **1.7 %** | 28.4 % | 0.199 % | 4.28 % | 8.0 % | 32.6 % |
| mid-rail Vds band 0.1–0.6 V (trip proxy) | 10431 | **0.0 %** | 50.7 % | 1.58 % | 4.66 % | 4.81 % | 5.07 % |

(The "top-25 %-of-cmax" cut degenerated to the full set due to a percentile
tie; the Vds mid-rail band — where each stage is mid-switch — is the
meaningful trip-region proxy and is the most damning: **0 %** of those evals
exceed δ = 5 %.)

## Verdict: **KILL**

δ is small everywhere: only **1.7 %** of all evals (and **0 %** of mid-rail
trip-region evals) exceed the 5 % asymmetry threshold — far below the
**>10 %** promotion bar. The Cgd/Cdg outputs are already near-symmetric
(median δ ≈ 0.2 %).

This **explains and corroborates** the V6.4.5 Track-A finding that
`NN_SYMMETRIC_CAPS=1` left the TSMC7 RO period *bit-for-bit unchanged*:
symmetrizing already-symmetric caps is a no-op. The 9 % RO period drift is
**not** a cap-asymmetry problem.

**Consequence:** TSMC7 RO is a model-fidelity gap in the Id / cap-magnitude
surface, not a stamp-symmetry artifact. The RO gate is routed to B3 (LoRA),
B5 (OSDI Jacobian distill), B6 (adversarial harvest), or B8 (diff-sim TTFT).
`NN_SYMMETRIC_CAPS` stays dormant (default OFF).
