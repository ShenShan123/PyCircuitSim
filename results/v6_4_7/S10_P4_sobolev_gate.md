# V6.4.7 S10 (P4) — Sobolev id-derivative supervision: gate + verdict

**Date:** 2026-06-14 · **Verdict: KILL the Sobolev term** (drop from the final
loss stack; record dead-end next to V5 Phase-C). **No model promoted; headline
stays 11/16; control-v2 stays the attribution baseline.** The committed loss
code (`SobolevIdLoss`) stays as default-off, recoverable infrastructure (it
pairs with the permanent deriv-fidelity scorer), exactly as the V5 Phase-C JAC
loss was kept recoverable.

## What was built (commits e834060, f9144b7, ee955a0)

`SobolevIdLoss` (`bsimar/losses/bni_mae.py`): id-channels only
(∂id/∂{Vg,Vd,Vb} vs OSDI gm/gds/gmb), in the **same asinh normalized-derivative
space the deriv-fidelity gate measures**, so the term supervises exactly the
quantity ruling-4 scores. Trainer/CLI wiring (`--sobolev`, `--lam-sobolev`,
`--sobolev-floor`, `--sobolev-strong-boost`, `--sobolev-corridor-only`,
`--init-from`), second-order autograd, EMA-compatible, default path
bit-unchanged.

**Sign convention (the P0-I §2 trap) — verified, not assumed.**
`scripts/v6_4_7_s10_sign_check.py` on control-v2 confirmed **uniform negation
of all three id channels** (stored gm/gds/gmb = −∂id/∂V): for `gds` (the only
channel where conventions differ) `res_uniform` 7.8e-3 ≪ `res_930` 8.6e-2
(~11×). The 930c274 "gds is the diagonal so no flip" rule is WRONG for this
stored convention. FD selfcheck <0.5 %.

## Screen (from-scratch retrains, seed 17 = clean A/B vs control-v2 s17)

A warm-start fine-tune screen (v1) was abandoned: plain val-MAE selection
always REVERTS the slope correction (λ=0.1 degraded val-MAE 4× and early-stopped
at epoch 2). Replaced by from-scratch retrains at seed 17 — identical init +
data split + normalizer fit to control-v2 s17, so the A/B isolates the Sobolev
term exactly.

| config | λ | opamp | RO% | invVTC | **gds_fwd** | gm_fwd | off_exc |
|---|---|---|---|---|---|---|---|
| ctlv2 s17 (baseline) | — | 10.46 % (gain 180.5) | 8.66 | 3.45 | **55.8** | 137.3 | 2.9e-4 |
| sob_a (global, boost4) | 0.02 | **collapse** (0.0) | 11.93 | 1.72 | 42.6 | 115.2 | 1.5e-7 |
| sob_b (corridor-only) | 0.1 | **collapse** | 15.25 | 3.82 | 11.1 | 16.0 | 1.5e-7 |
| sob_c (corridor-only) | 0.3 | **collapse** | 11.27 | 5.96 | **1.7** | 0.1 | 3.0e-6 |
| sob_d (global, boost4) | 0.005 | **collapse** | 11.05 | 1.49 | 44.4 | 134.0 | 1.4e-4 |
| sob_e (global, boost4) | 0.01 | **collapse** | 12.28 | 3.17 | 44.6 | 131.1 | 1.7e-7 |

- **Derivative fidelity improves monotonically in λ** (gds_fwd 55.8→1.7 %,
  gm_fwd 137→0.1 %; off-state 3 orders) — the ruling-4 core objective, met.
- **Inverter held/improved at low λ** (1.49–1.72 < 3.45); degrades only at
  λ=0.3 (5.96, fails).
- **Opamp collapses on EVERY config, including λ=0.005** whose val-MAE is
  *identical* to control (0.00119 vs 0.00117). So the collapse is
  **λ-independent** — any non-trivial Sobolev gradient tips seed-17 into the
  collapse basin.

## 4-seed arm (config A, λ=0.02 boost4) — attribution

Per ruling 2 (1-seed opamp = collapse-noise), config A was run at 4 seeds and
compared to control-v2's per-seed opamp (s42 collapse, s17 180, s7 362,
s31 187 → control 1/4 collapse).

| seed | opamp | RO% | invVTC | gds_fwd | off_exc | SC | sram q/qb (state-1) |
|---|---|---|---|---|---|---|---|
| ctlv2 s17 | 180 ✓ | 8.66 | 3.45 | 55.8 | 2.9e-4 | ✓ 1.60 | 0.390/0.390 (metastable) |
| sob s42 | **0.0 ✗** | 7.99 | 1.15 | 43.3 | 1.4e-7 | ✓ 1.48 | 0.833/0.069 (railed-ish) |
| sob s17 | **0.0 ✗** | 11.93 | 1.72 | 42.6 | 1.5e-7 | ✗ nan | 0.750/0.125 |
| sob s7 | **0.0 ✗** | 7.77 | 0.96 | 42.8 | 1.7e-7 | ✓ 1.62 | 0.749/0.123 |
| sob s31 | **0.0 ✗** | 11.19 | 2.36 | 42.3 | 1.4e-7 | ✓ 1.86 | 0.390/0.390 (metastable) |

- **Opamp collapses 4/4** (vs control 1/4), INCLUDING s7 and s31 which control
  kept healthy at gain 362 / 187. **Systematic, not seed-luck.**
- Deriv fidelity (gds 42–43 vs 48–69, gm 115–117 vs ~137, off 3–4 orders),
  inverter (0.96–2.36 vs 3.45), SC (3/4 hold) — all robust across seeds.
- RO mixed: 2/4 improved (7.77/7.99 — best-ever tsmc7, < control 8.66), 2/4
  regressed (11–12). Net within seed-variance.
- **Side finding:** 3/4 seeds move SRAM force_ic OUT of the symmetric
  metastable point (0.39/0.39) toward a railed state (q≈0.75–0.83 / qb≈0.07–0.13);
  qb on s42 (0.069) is essentially at the 0.1·VDD band — the off-state deriv
  improvement helps the subthreshold-owned SRAM attractor (P3-adjacent), but
  does not close it (q over-rails).

## Verdict — KILL (pre-registered kill gate triggered)

Plan S10 kill gate: *"best-λ screen does not cut TSMC7 gain err below ~15 %
with inverter held → drop the Sobolev term, record the dead-end next to the V5
Phase-C entry."* The opamp gain err is **~100 % (fully collapsed) on every λ and
every seed** → kill gate triggered decisively. The Sobolev term is **dropped
from the final loss stack.** No Sobolev checkpoint is promoted; the
`v6_4_7_s10{ft,sob,p4}_*` stems are inert dead-end artifacts (do not match the
resolver pattern). Per ruling 5 the full retrain still happens — on the
*surviving* (stock) stack, which is control-v2.

## The major finding — derivative fidelity is ANTI-correlated with the opamp

control-v2's autograd derivatives are badly off on **every** seed (gm_fwd
~137 %, gds_fwd 48–69 %) — yet its opamp gain is within ~10 % of NG (180 vs
163). The Sobolev arm IMPROVES those derivatives (gm 116 %, gds 43 %) and
**collapses the opamp to 0 on all 4 seeds.** Better autograd Jacobian → worse
(collapsed) opamp.

Mechanism (a value-surface / NR-fixed-point property, the **P0-C / P0-I class**
extended from RO to the opamp): the harness opamp gain is the max slope of the
**large-signal DC transfer curve** — the locus of *converged* NR fixed points,
a property of the id **VALUE** surface, not of the autograd Jacobian (which
guides NR convergence but cancels at the fixed point, P0-C). The Sobolev term
necessarily reshapes the value surface (same weights produce value AND slope)
to fix the slope, and that reshape destabilizes the value-owned opamp bias.

**Consequence for the campaign — ruling-4 premise partially falsified.** Precise
∂id/∂V does NOT help — and actively harms — the value-owned opamp gain (and, by
the same P0-C argument, the RO period). The derivative-fidelity metric (ruling 4)
is at best an **NR-robustness** indicator, NOT a proxy for opamp/RO circuit
accuracy; it should be demoted from a circuit-accuracy promotion gate. The
opamp/RO levers must target the id **VALUE** surface (P5 trajectory corridors,
P3 subthreshold), not derivative supervision. The lone deriv-positive
side-channel worth carrying forward is the **off-state/subthreshold derivative
improvement's effect on SRAM force_ic** (3/4 seeds escaped the metastable
point) — feed this observation into P3/S11, not as a Sobolev arm but as
evidence the subthreshold VALUE surface is the SRAM lever.

## Artifacts

- Loss + sign check: `bsimar/losses/bni_mae.py` (`SobolevIdLoss`),
  `scripts/v6_4_7_s10_sign_check.py`.
- Screens: `scripts/v6_4_7_s10_screen{,2,3}.sh`, `_full.sh`, `_score.py`.
- Scores: `results/v6_4_7/S10_screen_score.json`, `S10_screen3_score.json`,
  `S10_4seed_score.json`. Baseline: `S10_screen_baseline.md`,
  `S10_deriv_fidelity_controlv2_ref.md`.
- Logs: `results/v6_4_7/s10_screen{2,3}_logs/`, `s10_full_logs/`.
