# tsmc7 opamp → 16/16: T3, the differentiable-DC-solver campaign (and what comes after)

Date: 2026-06-28 · Branch `V6.5.4` · Forward plan. Successor to
`2026-06-27-tsmc7-opamp-vout-existence-retrain.md` (§8 = V6.5.8 EKV breakthrough)
and `2026-06-26-accuracy-frontier-3operator-phase0.md`. Recorded against CHANGELOG
V6.5.8.

> **Where we are.** Production = **15/16** complex DC gates. The lone open gate is
> the **tsmc7 opamp** (gain→0 historically). **V6.5.8 broke the rail**: a high-r_o
> EKV structural core + vout-weighted KCL existence fine-tune makes the tsmc7 opamp
> *amplify* (gain ~350–381, reachable via continuation, ring preserved) — the first
> non-railed tsmc7 opamp ever, refuting the "unreachable / only-T3-creates-it"
> verdict. The remaining gap is **gain calibration**: the reachable OP is over-gained
> (~2.2×), and gain is COUPLED to OP-reachability through the output-stage r_o, so no
> loss/structure lever tried (vout-weight, lam-kcl, lam_lo, freeze-core) lands gain
> 163 without railing. The remaining lever — **T3** — is the subject of this plan.

## 0. The one fact that defines the T3 design

Everything the fine-tune (V6.5.8) supervised is a **static residual** at the L72 node
voltages: `F(V_L72; θ) ≈ 0`. It never supervises the actual **transfer curve**
`Vout(Vin)` — the quantity the gate measures (gain = peak `|dVout/dVin|`, trip =
crossing). So the existence fine-tune minimizes the residual the *easiest* way:
**over-flatten the output-stage r_o** (which makes the OP a sticky continuation
attractor AND drives `F→0` over a wide vout range), overshooting gain to ~370. The
gate's solve lives *outside* the loss, so the loss has no signal that the gain is wrong.

**T3 closes this loop: put the DC solve IN the loss.** Supervise `Vout(Vin; θ)`
(produced by a differentiable opamp DC solve) against L72's curve. Then existence,
gain, trip, and curve-shape are all enforced jointly, and r_o is shaped by the gain
target instead of by the residual-minimization shortcut.

**Corollary (binding constraint).** T3 MUST be free to reshape r_o, i.e. train the
**full** model (EKV core `param_head` *and* the residual) — NOT `--freeze-core`. The
V6.5.8 coupling map says gain lives in lam (core) and existence needs the residual;
T3's transfer-curve gradient must reach both.

## 1. T3 build — differentiable unrolled opamp DC solver

Reuse the V6.5.8 substrate: `tsmc7_dn_ekvhr_{nmos,pmos}` (high-r_o EKV core) as init;
the `KCLGroups`/harvest infra in `scripts/v6_5_5_finetune_kcl.py` already assembles
`F(node)=Σ signed NN terminal id` into the free nodes {vtail, n1, vo1i, vout} with the
device→node topology+sign and the L72 OP per Vin. That assembler IS the residual map
`F(V; θ)`; T3 wraps a differentiable Newton solve around it.

New module (proposed `scripts/v6_5_8_t3_solver_finetune.py`):
1. **Differentiable Newton solve.** For each `Vin` on a grid spanning the trip
   (±0.06 V, the harvested band): start `V₀` = L72 OP (teacher-forced); iterate
   `V_{k+1} = V_k − (∂F/∂V)⁻¹ F(V_k; θ)` for K≈5–10 steps, `J=∂F/∂V` via autograd
   through the NN, keeping the graph. Output `Vout(Vin; θ) = V_K[vout]`.
   - Damp/clip the Newton step (LM `J+λI`) so the unstable high-gain OP doesn't blow
     the unroll; anneal teacher-forcing → self-start over epochs.
   - Cheaper alt once stable: implicit-function `∂Vout/∂θ = −(∂F/∂V)⁻¹ ∂F/∂θ` at the
     converged V* (skip unrolling). Start with unrolling — robust to non-convergence.
2. **Loss** (all terms already have infra):
   - `L_curve = Σ_Vin (Vout_NN(Vin) − Vout_L72(Vin))²` — the transfer curve.
   - `L_gain = (peak|dVout/dVin|_NN − gain_L72)²` — explicit slope target (the gated
     quantity); compute the slope by autograd of the differentiable `Vout(Vin)`.
   - `+ base-data LDS-MAE anchor` (preserve the 15 gates) `+ ring-anchor` (preserve
     ring). Same anchors as V6.5.8; they held bulk to +1–2 %.
3. **Gating (Rule-16 naming: `tsmc7_dn_*`):** `diag_opamp_solver_conditioning.py`
   (reachable?) → `verify_complex_opamp.py --tech TSMC7` (gain ±10 %, AUTHORITATIVE)
   → `diag_opamp_basin_seed.py` 1c → full 16-gate matrix + device DC/AC +
   lifted-source canary (run the canary WITHOUT a global env override — it is
   multi-tech; pin only tsmc7 via the resolver or a per-tech harness).

## 2. Cheapest-first ladder (gate each rung before the next)

| Rung | Lever | Cost | P(16/16) | Gate-or-kill |
|---|---|---|---|---|
| **T3.0** | **MVP: overfit the single trip point, no anchor.** Unroll the solve at `vin*` only; supervise `Vout=½VDD` + `slope=163`. Full model (core+residual), init from `ekvhr`. | ~2 GPU-hr | — | Does a reachable OP with gain∈[147,180] form AT ALL? If NO → T3 cannot place a gain-163 reachable OP on this surface → **permanent 15/16** (Rung 5). If YES → T3.1. |
| **T3.1** | Add the ±0.06 V curve band + `L_gain` + base/ring anchors; full fine-tune. | ~half day | ~15–25 % | `verify_complex_opamp` gain ±10 % AND ring/switchcap/sram/device-DC/AC/canary unregressed → **install → 16/16**. |
| **T3.2** | If T3.1 lands gain but regresses a preservation gate: widen anchors / trade λ; or implicit-function variant for a cleaner gradient. | ~1 day | — | same gate. |
| **5** | **Document permanent 15/16** with the V6.5.8+T3 verdict: the per-device-NN + external-solver architecture cannot match L72's output-stage curvature to the ~1 %/trip-window precision a gain-163 reachable OP needs. | — | — | honest endpoint. |

**Start at T3.0.** It is the cheap fund-or-kill that V6.5.8's coupling map made the
right question: *can the solve-in-the-loss place a reachable gain-163 OP at all?* One
overfit run answers it before the heavy full campaign.

## 3. Risks / known traps (carry forward)
- **Unroll instability.** The high-gain OP has a near-singular `J` (gain = tiny output
  conductance) — the Newton unroll can diverge. Mitigate: LM damping, gradient
  clipping, teacher-forced warm starts, modest K. If the unroll is hopeless, switch to
  the implicit-function form at a continuation-converged V*.
- **Preservation is the binding risk, not capacity.** Every opamp retrain that moved
  the value surface risked the ring (shared NMOS bias region). V6.5.8's ring-anchor
  held it (ring PASS 2.43 %); keep it in T3. Do NOT chase capacity past `large`
  (v66-xl) or broad data (v648-broad-retrain-collapses-opamp).
- **Don't re-tread the dead levers.** vout-weight/lam-kcl (binary switches), `lam_lo`
  cap (rails), `--freeze-core` (rails), N2 Sobolev (value/slope conflict), Jacobian
  distillation (P0-3: location is id-VALUE-only), fetlim/voltage-limiting (L72 control
  lands 163 on the same path). All recorded dead.
- **Gate honestly.** CPU-pinned (CUDA_VISIBLE_DEVICES='', OMP=MKL=1) — the opamp is
  multistable, CPU vs CUDA land different basins. Trust `verify_complex_opamp`
  (continuation), not the cold probe alone, for the gain; trust the probe for
  *reachability*. The gate keys only on gain — a pass with a residual trip offset
  (~100 mV in V6.5.8) is a legitimate gain-gate pass but note the offset.

## 4. Beyond the opamp (lower-priority future accuracy work)
- **EKV core as a general substrate (DC-safe to explore).** The high-r_o EKV core hit
  ~0.24 % bulk id-MRE AND the most *balanced* opamp node residuals of any checkpoint.
  It is worth A/B-ing the EKV core on the OTHER techs' marginal gates (e.g. tsmc5 ring,
  the device f3db cells) — purely additive, default-off, gated against the full matrix.
  Low odds of moving a gate but cheap and it may tighten margins.
- **G3 device f3db (13/24).** OP-drift / value-surface owned; the charge lever is dead
  (V6.5.6 P0-2 D2=MATCH: autograd cdd ≈ supervised ≈ OSDI). No independent lever — but
  if T3 (or an EKV productionization) improves the value-surface OP placement, f3db
  may improve as a side effect. Re-measure after any value-surface change.
- **G4 RHP-zero phase.** Diagnostic-only (passband-masked); moves no gate. Leave as
  fidelity-only unless a phase gate is added.
- **The honest ceiling is unchanged: 16/16 ≈ 1-in-4 to 1-in-5; 15/16 is the stable
  ship.** V6.5.8 improved the odds (reachability is solved; only gain calibration
  remains) but did not change the count.

## 5. Status / substrate on disk
- Substrate (gitignored): `tsmc7_dn_ekvhr_{nmos,pmos}` (high-r_o EKV core),
  `tsmc7_dn_ekvkcl_*` / `tsmc7_dn_ekvk_l*` / `tsmc7_dn_ekvfz_*` (existence candidates,
  all gain ~370 or railed — kept as the gain⟺r_o coupling evidence).
- Code (committed in V6.5.8): `_EKVCore` floor-scaled residual + `--ekv-lam-lo`
  (`bsimar/models/direct_net.py`, `cli/train.py`, `training/trainer.py`); EKV-aware
  `_build_and_load` + `--freeze-core` (`scripts/v6_5_5_finetune_kcl.py`).
- Prereqs: `results/v6_5_5/kcl_groups/tsmc7_opamp_kcl.npz`,
  `results/v6_5_5/corridors/tsmc7_ring_{nmos,pmos}_corridor.npz`,
  `external_compact_models/bsimar/data/datasets/tsmc7_{nmos,pmos}.npz`. 3× RTX 4090.
