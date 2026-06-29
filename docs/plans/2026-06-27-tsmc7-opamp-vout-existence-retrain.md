# tsmc7 opamp → 16/16: the vout-prioritized existence retrain (and the cheap-first ladder)

Date: 2026-06-27 · Branch `V6.5.4` · Supersedes the "only T3 remains" close of
`2026-06-26-accuracy-frontier-3operator-phase0.md` (kept for rationale; its STATUS
banner now points here). Recorded as CHANGELOG **V6.5.7**.

> **★ RESOLVED 2026-06-29 (CHANGELOG V6.5.9): Rung 4 (T3) WORKS → 16/16.** The
> differentiable-DC-solver fine-tune (the successor plan
> `2026-06-28-tsmc7-opamp-T3-differentiable-solver.md`) landed the tsmc7 opamp gate
> (gain 178.0/8.92 % PASS) and the full 16-gate matrix. The Rung-1 KCL-loss KILL
> below still holds; T3 succeeded where the loss-only levers could not because it
> supervises the SOLVED transfer curve, not a static residual. Production 16/16.

> **TL;DR.** A 5-agent adversarial review of the V6.5.6 verdict found it
> over-stated. The lone open gate (tsmc7 opamp DC gain→0) is NOT "existence-solved,
> contraction-remaining, only T3 left." It is **full-system *stable existence* with
> `vout` as the never-supervised node**, and the cheapest lever that targets it — a
> **vout-prioritized native-µA existence retrain** — was never run. Climb a
> cheapest-first ladder; escalate to the EKV-core and T3 levers only if the cheap
> retrain fails. Production stays **15/16** until a candidate clears the full gate
> matrix. Honest ceiling unchanged: **16/16 ≈ 1-in-5 to 1-in-4; plan for 15/16.**
>
> **⇒ RUNG 1 EXECUTED (2026-06-27) — KILL. See §5.** The vout-prioritized retrain
> CONFIRMS the reframe but does NOT reach 16/16: the output-stage residual `vout`
> floors at ~0.062 (only by wrecking the base surface, +492% val) to ~0.13 (at
> safe preservation) — vs the ~0.006 a high-gain zero needs — and solver-
> conditioning finds **0 high-gain solutions on every candidate** (gate gain still
> 0.0). The soft wall is real and ~10–20× wide. Routes to Rung 3 (EKV core) /
> Rung 4 (T3). Production unchanged 15/16.

## 1. What the review corrected (why the old verdict was over-strong)

The V6.5.6 close rested on three claims the panel falsified or qualified
(unanimously, all code-cited):

1. **"T1 solved existence."** Only *partially*. T1/k3_a pinned the stage-1 balance
   node `vo1i` (`i_Mn2 − i_Mp4`) and **never supervised `vout`** — the KCL loss is a
   uniform `mean` over the 4 free nodes (`finetune_kcl.py:216-222`) and the epoch
   selection checks `vo1i` only (`:438`). So the *full* 4-node high-gain root
   (`vo1i` balanced AND `vout` mid-rail simultaneously) was never created. The bind
   is still full-system **existence** with `vout` unpinned — it only looked like a
   pure "contraction" problem because the partial root existed at `vo1i`.
2. **"No high-gain zero exists (probe-closed)."** Overstated. The
   solver-conditioning probe is 20 *cold* multistarts seeded effectively along a
   1-D `vout` line with `vtail`/`n1` pinned and `vo1i` clipped to L72
   (`diag_opamp_solver_conditioning.py:115-136`). That proves *not reachable by
   multistart/seed/GMIN*, not *non-existent*. Pseudo-arclength branch-tracking
   (which traverses the near-singular fold the killed Norton soft-pin homotopy
   folds at, `solver.py:964-979`) was never run.
3. **"Representational limit, only T3 remains."** Contingent, not established. What
   was shown: `vo1i`-only existence + ring preservation are co-achievable but don't
   pass, and KCL + N2 cannot add Newton contraction without destroying the partial
   existence. A *full* `vout`-inclusive root was never attempted.

Supporting refinements:
- **The wall is a tech-specific *soft ratio* wall, not a universal MLP ceiling.** A
  high-gain zero lands in-band only if `1/gain ≳` the achievable output-stage
  cancellation precision (~1%). tsmc12 passes (lower gain, wider band); tsmc7 fails
  (ulvt + 0.7 V VDD → gain≈163 ⇒ 1/gain≈0.6% AND a narrower/steeper high-gain Vin
  window). 0.6% needed vs ~1–1.5% raw is a **~2–3× gap** — and T1 already closed
  `vo1i` ~18× (0.128→0.007) with the native-µA signed-difference trick. So pulling
  the *output-stage* difference under ~0.6% is hard-but-live.
- **`vout`-residual-at-V\*_L72 is non-predictive** (tsmc12 has `vout` F_rel ≈ 0.19
  there and passes — its NN zero sits at V′ ≠ V\*; `diag_opamp_kcl_residual.py:135-141`).
  Do not use it as an existence proxy; use a *post-retrain* solve/branch-track.

## 2. Confirmed-dead levers (do NOT re-tread)

- **fetlim / SPICE voltage-limiting as a fix.** The L72-in-PyCircuitSim opamp
  control lands gain 163–188 on the SAME continuation-first, fetlim-less path
  (`diag_opamp_op_decomp.py:109-115`) → voltage-limiting absence is not the
  blocker; the gap is purely the NN value surface. fetlim conditions the path,
  cannot manufacture a zero or stabilize a repeller.
- **Jacobian / gm-gds distillation, decoupled- or separate-head stamping.** P0-3:
  the DC fixed-point *location* is a pure function of `id` VALUES; gm/gds set only
  the Newton path (re-verified `mosfet_nn.py:629-643`). Matching L72's high-r_o
  Jacobian moves the path, not the zero — and the only way it touches the value
  surface is the S10 weight-sharing collapse.
- **N2 / id-Sobolev contraction on the shared id head.** k3_b/k3_c left `vo1i`
  stuck at 0.22–0.24 (value/slope conflict on one output) — the S10 collapse.
- **`force_ic` / `uic` pinning of `vout`.** Releases and re-solves unconstrained →
  an unstable OP diverges (strictly weaker than the 1c L72-seed that already rails);
  and for a single-valued opamp transfer it would measure the injected ground
  truth, not the NN's emergent gain (goalpost-moving).
- **Seed/ensemble sweep, broad-data retrain, capacity past `large`, µA-band
  de-compression *value* retrain.** All recorded dead (V6.5.5 sweep; v648-s1;
  v66-xl; v66 µA-band KILL).

## 3. The cheapest-first ladder (each rung gates the next)

| Rung | Lever | Cost | P(16/16) | Status |
|---|---|---|---|---|
| **1** | **vout-prioritized native-µA existence retrain from k3_a** (§4) | ~1 GPU-hr | ~12–30% | **DONE — KILL (§5).** vout floors 0.062–0.13 vs 0.006 needed; 0 high-gain solns; gate gain 0.0. Soft-wall near-hard for KCL family. |
| **2** | Validate any candidate (solver-conditioning + optional arclength; full 16-gate + ring + canary) | ~1 CPU-hr | — | **DONE for Rung-1 candidates** (solver-cond + NGSPICE opamp gate run; all rail). |
| **3** | Region-gated **high-r_o EKV core** as substrate (structural fix for the r_o-shape defect that makes the OP an unstable NN-surface fold) | ~1 day | ~15% | **NEXT if pursuing 16/16.** Changes the representation, not the loss — the only family V6.5.6/Rung-1 did NOT test (they killed the *loss* path to slope, never the *structural* one). |
| **4** | **T3** unrolled-DC-solver MVP (overfit-hard at the trip-point, no anchor) → full campaign | ~1 day → 1–2 wk | ~15% | last resort; puts the SOLVE in the loss (existence+stability+level jointly). |
| **5** | Document **permanent 15/16** with the sharpened verdict | — | — | the honest endpoint; Rung-1 already sharpened it (existence/precision wall, ~10–20× on vout). |

**One live (action-neutral) disagreement:** the soft-wall odds — ~12% (near-hard)
vs ~20–30% (clearly soft). It does not change the next action: run Rung 1 first.

## 4. Rung 1 — build spec (this is what executes next)

**Premise.** k3_a already gives existence at `vo1i` (0.009) + ring PASS (2.29%) +
switchcap/SRAM/device-AC PASS — a clean, non-regressing start. The only missing
ingredient is making `vout` (the output-stage balance `i_Mp6 − i_Mn7`) *also* a
near-zero at the same OPs, so the full high-gain root exists. `F[vout]` is already
assembled by the harvest (it is free-node index 3); it is simply never prioritized.

**Code change (minimal, behavior-preserving at defaults), `scripts/v6_5_5_finetune_kcl.py`:**
1. `KCLGroups.__init__`: store `self.free_nodes` (names from the npz) and the
   `vo1i`/`vout` indices.
2. `KCLGroups.loss_and_frac(node_w=None)`: optional per-free-node weight —
   `loss = (node_w[None,:] * rel**2).mean()` (None ⇒ the current uniform mean,
   byte-identical).
3. `main()`: add `--vout-weight` (default 1.0) and `--vout-target` (default off).
   Build `node_w = ones(n_free)`, set `node_w[vout_idx] = vout_weight`; pass it to
   the training-loop `loss_and_frac`. Extend epoch selection so a candidate counts
   as "fixed" only if `frac[vo1i] < vo1i_target` AND (`vout-target` off OR
   `frac[vout] < vout_target`). All new args at default ⇒ prior k2/k3 runs reproduce.

**Run (start from k3_a, ring-anchored, N2 OFF):**
```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 conda run -n pycircuitsim python -u \
  scripts/v6_5_5_finetune_kcl.py --tech tsmc7 --cuda --overwrite \
  --nmos-init tsmc7_dn_kcl3_a_nmos --pmos-init tsmc7_dn_kcl3_a_pmos \
  --ring-weight <k3_a value> --lam-kcl 1.0 --lam-sob 0 \
  --vout-weight 8 --vo1i-target 0.02 --vout-target 0.05 \
  --epochs 60 --exp-name tsmc7_dn_kclV_w8
```
Sweep `--vout-weight ∈ {4,8,16}` (one GPU each, parallel) if the first is
promising-but-marginal.

**Validation gate (Rung 2 — run on each candidate via
`PYCIRCUITSIM_NN_CHECKPOINT_DN_{NMOS,PMOS}`):**
1. `tests/diag_opamp_kcl_residual.py` — BOTH `vo1i` AND `vout` F_rel must drop.
2. `tests/diag_opamp_solver_conditioning.py` — the **fund-or-kill** signal: does
   any mid-rail seed now converge to a high-gain solution (gain>50)? (Optional
   arclength capstone for rigor.)
3. `tests/diag_opamp_basin_seed.py` (1c) — gain must HOLD >0 when L72-seeded.
4. `tests/verify_complex_opamp.py` — the authoritative tsmc7 opamp gate.
5. Full 16-gate matrix + device DC/AC + lifted-source canary — UNREGRESSED
   (swapping only tsmc7's checkpoints touches only tsmc7's gates; ring is the
   binding preservation risk).

**Decision:**
- vout F_rel drops AND a high-gain solution appears AND the gate passes AND
  preservation holds → **install → 16/16.**
- vout F_rel drops but the OP is still unstable (gate rails / 1c rails) →
  existence-without-contraction confirmed at the *full* system → route to Rung 3
  (EKV core) / Rung 4 (T3).
- vout won't drop below the gain-demanded ~0.6% without nicking the ring →
  the soft wall is binding at tsmc7 → quantified verdict, route to Rung 4 or 5.

## 5. Rung 1 — EXECUTED (2026-06-27): KILL — the soft wall is real and ~10–20× wide

**Code:** `scripts/v6_5_5_finetune_kcl.py` gained `--vout-weight` / `--vout-target`
+ vout-aware epoch selection (behavior-preserving at defaults; the `loss_and_frac`
node weighting and selection now read the `vout` free-node index). Recipe matched
k3_a (`--apply-filter off`, `--ring-weight 1.0 --lam-kcl 20 --lam-sob 0`); the
default `--apply-filter on` was a real mismatch the norm-assert correctly caught
(0.83 % filtered rows drift the normalizer 1.3e-3 off the `large`/k3_a checkpoints).

**Sweep (3× RTX 4090, 80 epochs each):** large-start ×{w16, w64}, k3_a warm-start
×w16. **Baseline confirms the reframe outright:** at the production `large` start
`frac = [vtail 0.011, n1 0.132, vo1i 0.132, vout 0.121]`, and **k3_a's start is
`vo1i 0.009 / vout 0.279`** — i.e. T1/k3_a fixed `vo1i` to 0.009 but left `vout` at
a **28 % residual**: the full high-gain root never existed there, exactly as the
panel argued.

**The vo1i↔vout↔preservation frontier (quantified):**

| run | best `vout` F_rel | `vo1i` there | anchor val-drift | reading |
| --- | --- | --- | --- | --- |
| w16 (large) | 0.108 | ~0.01 | +200–600 % | `vout` floors ~0.11–0.13, surface wrecked |
| **w64 (large)** | **0.062** | 0.018 | **+492 %** | best `vout`, base surface destroyed |
| **w16k3a** (k3_a) | 0.132 | 0.004 | **+31 %** | preservation-safe, `vout` stuck ~0.13 |

A high-gain zero needs `vout` F_rel ≈ 1/gain ≈ **0.006**. Achievable is **0.062
(10× too high, surface-wrecking) → 0.13 (20× too high, preserved)**, with a hard
**anti-correlation** (driving `vout` down ⇒ the shared NMOS/PMOS heads blow up the
base-data anchor — the S10 value-coupling, now on the value side).

**Decisive probes (CPU, L72-in-PyCircuitSim, no inference):**
- `diag_opamp_solver_conditioning.py` on BOTH the preserved (w16k3a) AND the
  surface-wrecking (w64) candidates: **0 high-gain solutions / 20 starts; all rail**
  (best |gain| = 0). No reachable high-gain OP was created.
- `verify_complex_opamp.py --tech TSMC7` (authoritative, NGSPICE) on w16k3a:
  **DirectNet gain = 0.0, gain err 99.99 %, FAIL** (NGSPICE truth 163.4).

**Verdict.** The full `vout`-inclusive high-gain root is **not creatable by KCL
loss-weighting within the preservation budget** on the single-id-head DirectNet
surface. This is the *existence/precision* wall (not contraction): the output-stage
current cancellation the gain demands (~0.6 %) is ~10–20× below what the value
surface can hold while staying accurate. The disagreement between the panel's
near-hard (~12 %) and soft (~20–30 %) reads resolves toward **near-hard** for the
KCL-loss family. Nothing installed; production stays **15/16**. Candidate
`tsmc7_dn_kclV_w16k3a` (preservation-safe, `vout` 0.279→0.13) is kept on disk as
the existence-improved substrate for Rung 3/4; w16/w64 discarded (surface-wrecked).

**Routing forward:** the value surface cannot represent the cancellation by *loss*;
the remaining levers change the *representation* (Rung 3 — region-gated high-r_o EKV
core, which supplies the saturation shape by physics rather than by the conflicted
loss) or put the *solve* in the loss (Rung 4 — T3). Both are heavier; neither is
funded yet.

## 6. Lessons learned (reuse these — do not relearn)
- **Retraining from `tsmc7_dn_large`/k3_a needs `--apply-filter off`.** The
  `finetune_kcl.py` default `--apply-filter on` drops 0.83 % small-Id rows, drifting
  the re-fit normalizer ~1.3e-3 off the checkpoint's stored norm → the
  `_assert_norm_matches` guard (1e-3) aborts. The assert is CORRECT (it catches a
  genuine preprocessing mismatch); match the checkpoint's recipe, don't relax it.
- **The decisive metric is `diag_opamp_solver_conditioning.py`, NOT the `vout`
  residual.** A low residual is necessary, not sufficient; only the multistart
  (or arclength) shows whether a high-gain OP is actually reachable. Always run it
  before claiming a candidate works, and the NGSPICE `verify_complex_opamp.py` for
  the authoritative gain.
- **`vout`↓ and base-anchor preservation are anti-correlated** on the single-id
  head (driving the output-stage difference to zero blows up the device surface —
  the S10 value/derivative conflict, now value-side). Loss-weighting cannot
  separate them; a *structural* prior (Rung 3 EKV core) is the only way to decouple.
- **Run the cheap probe BEFORE the heavy lever.** Rung 1 (~1 GPU-hr) converted the
  open "is 16/16 reachable by the cheap existence lever?" into a hard KILL with a
  quantified ~10–20× gap — saving a blind T3 campaign. Keep this discipline for
  Rung 3/4: gate each on `diag_opamp_solver_conditioning.py` first.
- **Update THIS plan file on every change/lesson** (standing user directive) so the
  next session starts from the current frontier, not the original hypothesis.

## 7. Status / housekeeping
- Prerequisites on disk: `results/v6_5_5/kcl_groups/tsmc7_opamp_kcl.npz`,
  `results/v6_5_5/corridors/tsmc7_ring_{nmos,pmos}_corridor.npz`,
  `tsmc7_dn_kcl3_a_{nmos,pmos}` checkpoints. 3× RTX 4090 free.
- Nothing installed; production resolver symlinks untouched. New candidates are
  `tsmc7_dn_kclV_*` (gitignored), gated before any repoint.
- This plan is the substrate for Rungs 3–4 as well (the ring-anchor + harvest_kcl
  infra is shared).

## 8. Rung 3 EXECUTED (V6.5.8, 2026-06-28) — BREAKTHROUGH: the high-gain OP is realizable

The EKV-core lever (Rung 3) was rebuilt and run. Headline: **the V6.5.7
"representational limit / only T3 remains" verdict is REFUTED.** The single-id-head
DirectNet surface CAN host a stable high-gain tsmc7-opamp DC fixed point — via a
**high-r_o EKV structural core + a vout-weighted KCL existence fine-tune.** This is
the first non-railed tsmc7 opamp in the entire campaign (all of V6.4.x–V6.5.7 gave
gain 0). Calibration (gain magnitude) is the remaining work; production stays 15/16
until a candidate clears the full gate matrix within ±10%.

**What was built.** `_EKVCore` (V6.4.8 S3, killed) was redesigned to fix its two
defects: (1) the residual is now **floor-scaled physical**
`id = id_core + sqrt(id_core²+(κ·id_s)²)·α·tanh(trunk)` (authority scales with the
local current → no µA-band runaway, the V6.4.8 KILL mechanism; smooth, autograd
intact), replacing the additive-in-asinh-z form; (2) the CLM/lam band low end was
widened (0.30→0.05) for high-r_o headroom. `scripts/v6_5_5_finetune_kcl.py`
`_build_and_load` is now EKV-aware (detects `core.*`, rebuilds the core).

**The decisive chain of findings:**
1. **Pure EKV core (bulk-trained, no existence loss): KILL on its own.** Trained
   tsmc7 N+P `large` + `--ekv-core` (held-out id MRE ~0.24%). Solver-conditioning
   probe = **0 high-gain solutions / all railed** — the structural prior alone does
   NOT create the missing zero. BUT its opamp node residuals at the L72 OP are far
   more **balanced** than any plain-MLP checkpoint (`vo1i 0.083 / vout 0.042` vs
   production `0.132 / 0.121` and kcl3_a `0.009 / 0.279`): the EKV physics holds
   both nodes low simultaneously.
2. **EKV substrate DISSOLVES the value-coupling wall.** Fine-tuning the EKV pair
   with the KCL existence loss + ring-anchor drove **BOTH `vo1i` AND `vout` to
   ~0.004–0.018** with the bulk surface preserved (+1–2% anchor drift) — exactly
   what Rung 1 on the plain MLP could NOT do (there `vo1i`→0.007 blew `vout` to
   0.279). The Rung-1 anti-correlation is gone because the physics, not free MLP
   weights, carries the bulk shape.
3. **vout-WEIGHTING creates the high-gain OP.** Uniform vout-weight (l20/l50) still
   railed (gain 0) in the authoritative gate. **vout-weight 3 (`vw3`) → gain 379.7,
   a REAL amplifying transfer curve** (over-gained vs target ~163, but non-railed).
   The cold solver-conditioning probe still missed it; the continuation DC-sweep
   gate (V6.4.8-S2 path) found it. ⇒ vout-weight is the decisive knob; the high-gain
   branch exists and is reachable by continuation once the output-stage balance is
   pushed hard enough.

**Why this matters / corrections to the prior verdict.** V6.5.7 concluded the wall
was a representational/precision limit reachable only by T3 (putting the solve in
the loss). That was over-strong: the EKV *structural prior* (not a loss term)
supplies the high-r_o output-stage shape the bulk MLP couldn't hold, and on that
substrate the vout-weighted existence loss creates a stable high-gain zero. Both the
existence wall AND the contraction wall fall. The remaining task is **calibration**
(land gain within ±10% of 163), not a new lever.

**The calibration wall (EXECUTED, definitive).** The reachable OP is over-gained
(~350–381 vs L72 163) and its gain is COUPLED to its existence via the output-stage
r_o. No available lever lowers gain to 163 without destroying the OP:
- vout-weight {1.5,2,2.5,3}: binary switch (rail ↔ ~370), not a gain knob.
- lam-kcl {6,10,14,20,50}: binary (≤6 rails, ≥10 → ~370).
- **`lam_lo` {0.10,0.13,0.16} (cap r_o): RAILS** — capping r_o kills reachability.
- **`--freeze-core` (data-true r_o ×3): RAILS** — the bounded residual alone can't
  make existence; the over-flat r_o is *required* for the OP to be a continuation
  attractor.
⇒ a reachable high-gain OP REQUIRES over-flattened r_o (gain ~370); the data-true
r_o (gain 163) supports no reachable OP. **Preservation OK** on a gain-370 candidate
(tsmc7 ring PASS 2.43 %; bulk anchor +1–2 %). Trip offset ~100 mV / Vout-NRMSE ~70 %
also remain (the gate keys only on gain). Production stays **15/16**; not installed.

**Routing → T3 (the next campaign).** The structural prior + existence loss CAN host
a reachable high-gain OP (the "unreachable / only-T3-creates-it" close is refuted),
but landing gain at the L72 value needs JOINT existence+gain+curve control: **T3 —
a differentiable unrolled-DC-solver supervising Vout(Vin) against L72.** The EKV+KCL
infra built here is its substrate: `tsmc7_dn_ekvhr_{nmos,pmos}` (high-r_o core),
`--ekv-lam-lo`, `--freeze-core`, EKV-aware `finetune_kcl`, and the gain⟺r_o coupling
map above (T3 must control r_o-shape, not just residual). Checkpoints on disk
(gitignored). Recorded as CHANGELOG V6.5.8.

**→ Forward plan: `docs/plans/2026-06-28-tsmc7-opamp-T3-differentiable-solver.md`**
(the T3 campaign — differentiable unrolled-DC-solver supervising Vout(Vin); starts
at the T3.0 MVP fund-or-kill; also covers EKV-core-as-general-substrate + G3/G4).
