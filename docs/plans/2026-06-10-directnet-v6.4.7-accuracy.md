# DirectNet V6.4.7 — Ranked levers to improve NN compact-model accuracy in complex-circuit simulation

**Date:** 2026-06-10 (rev. 2, same day)  •  **Status:** PROPOSED — REVISED after four-agent adversarial review (dead-end audit, ML methodology, simulator numerics, EV/risk)  •  **Branch:** `feat/v6.4.7` (cut from `feat/v6.4.6` @ `c5155e7`)
**Authoring:** rev 1 — plan-mode synthesis + staff-engineer adversarial review (checked against recorded V5–V6.4.6 dead ends; three conflicts designed around: V5 Phase-C JAC-loss negative, E2-medium head-trim falsification, P0-A NR-instability of the railed SRAM point). rev 2 — four-agent panel found the **P0 NMOS source-frame bug**, retracted the switchcap droop premise, re-powered the campaign, and hardened selection discipline. Proposal IDs are stable from rev 1; the rev-2 ranking is the section order below (P0/R0 new; P4 now precedes P3 among GPU arms).

**User rulings (2026-06-10):**
1. **SRAM `force_ic` is ship-required.** It is not one of the 16 headline cells (`verify_complex_sram_snm.py` counts the butterfly gate only), but it joins the V6.4.7 success criteria. P2's SRAM half, P3, and the P9 fallback are funded on this basis.
2. **Full-arm campaign authorized.** Budget raised to **~250–300 GPU-h** so every arm runs at **≥4 seeds per config** — the documented 139 mV seed lottery and the 13/16 spontaneous opamp-collapse rate make 1-seed arms indistinguishable from noise (a 1-seed arm "collapses" with ~80 % probability regardless of recipe; V6.4.5's central failure was attributing seed luck to recipes).

## Context

V6.4.4 is canonical at **9/16** complex-circuit gates. The 7 failing headline cells: **1 ring_osc** (TSMC7, 8.97 % period err), **3 opamp** (TSMC7 30.7 %; TSMC12 10.94 % against a **±10 % gain gate** (`GAIN_TOL=0.10`, `verify_complex_opamp.py:44`) — i.e. a **+1.1 % gain move** flips it, the cheapest gate on the board; TSMC16 flat/gain-0), **3 switchcap** (charge-transfer — see corrected row below); plus the ship-required **SRAM `force_ic` 0/8** (inboard attractor q≈0.87/qb≈0.20 on all techs). The user has authorized data regeneration and full retraining; project-rule constraints (e.g., Rule 10 no-new-losses) are waived for this proposal.

### Rev-2 headline finding — the NMOS source-frame bug (→ P0)

`_raw_voltages` (`pycircuitsim/models/mosfet_nn.py:232-243`) source-shifts **only PMOS**; NMOS passes **absolute** terminal voltages into a network trained exclusively at Vs≡0 (`pycmg/nn_generate.py:23`; shipped norm stats have `input_min[2]=input_max[2]=0`). The softplus input clamp (`mosfet_nn.py:245-261`) then pins vs≈0 while vd/vg/vb stay absolute — any **lifted-source NMOS** is evaluated at phantom Vgs/Vds inflated by +vs with Vbs=0, and `∂id/∂vs≈0` through the saturated clamp makes the solver's source-coupling stamps fictitious. CLAUDE.md NN Rule 2 documents the PMOS-only shift, encoding the same blind spot; **no test in any suite sweeps an NMOS at Vs≠0**.

The affected-device census maps directly onto the failing families:

- **opamp** — the diff pair Mn1/Mn2 sits on `vtail` (`examples/complex/miller_opamp_directnet.sp:23-24`): the gain-determining devices are mis-framed and blind to their own source node (no source degeneration, no body effect — a natural mechanism for the D5 gain→0 collapse fragility);
- **switchcap** — the pass NMOS has source=`vsamp`: phantom Vgs=VDD/Vds=vin drive while sampling explains DN charging to/past Vin (DN **overshoots** vin=0.48 at 0.4866/0.5098 on TSMC12/16);
- **SRAM** — access transistors Mal/Mar are lifted; Mar reads NN −125.5 vs OSDI −57.2 µA (**+68 µA — the largest single device error at the P0-D attractor**, 3.4× the Mpr reverse-clamp error);
- **ring_osc / inverter** — every source at a rail ⇒ **bit-identical under the fix**, consistent with RO being the smallest, genuinely model-owned gap.

### Root-cause table (corrected)

| Gate | Established owner | Evidence |
|------|------------------|----------|
| ring_osc TSMC7 | **joint (id, charge) model property** — NOT Jacobian (P0-C: inert ≤0.01 ps), NOT integration (P0-G: ~0.4 ps of 4.18 ps), NOT charge values alone (P0-H: ≤2 aC); NMOS dynamic id ~20 % peak under-prediction, but id is **not separable** (P0-I: exact-id injection → 92 ps, wrong direction) | `results/v6_4_6/phase0{C,G,H,I}_*.md` |
| SRAM force_ic | **co-owned by three current errors at the attractor:** (1) Mar lifted-source frame corruption, +68 µA (→ P0); (2) pinning NMOS Mnr weak-inversion over-prediction 7.5× (6.36 vs 0.84 µA at Vov≈+45 mV; → P3); (3) ON-PMOS Mpr restoring current **zeroed by the Rule-15 reverse-Vds clamp** (NN −0.0 vs OSDI −19.85 µA; → P2); asinh maps the 1nA–100nA band to 0.011 % of normalized id range | `phase0_D`, `phase0_E` + rev-2 audit |
| opamp | gain = true autograd derivatives of the learned id surface at the op point; gds NRMSE 20–23 % (P0-B) — inert for RO but **load-bearing for gain**; 13/16 fresh retrains collapse gain→0 (D5); diff-pair frame corruption (P0) compounds on top | `phase0_B`, V6.4.5 P5 + rev-2 audit |
| switchcap | **CORRECTED (rev 2):** the binding failure is the **sample-phase charge-transfer level** (TSMC5 14.68 %, TSMC12 8.33 %, TSMC16 13.13 % of VDD; gate ≤5 %), with DN overshooting Vin on TSMC12/16 — plausibly frame-owned (P0), hold-frozen by the reverse-Vds clamp (P2). **The rev-1 droop≡subthreshold-floor story is retracted:** recorded DN droops are 26 µV / 2 µV (`results/v6_4_5/phase1_logs/switchcap.log`), ~3 orders below any µA-floor prediction; the 2325 %/241 % figures are ratios against a ~1 µV NGSPICE reference at the harness noise floor, and the droop sub-gate as coded demands ~0.1 µV ≈ VNTOL agreement — **numerically unpassable noise** (measurement artifact, repaired in R0). The same gate auto-passes TSMC7's 2.178 mV droop (the largest absolute disagreement) via the `abs(ng_droop)>1e-6` nan-guard (`verify_complex_switchcap.py:133-137`) | `switchcap.log` + rev-2 audit |

Training-pipeline facts behind these: data is DC-only pointwise op-points; loss is MAE × per-target LDS (LDS is inverse-frequency over *output-value* bins → densifying an input region buys **no** loss mass); the analytic OSDI derivative columns (gm/gds/gmb, caps) **are supervision targets of the 13-output loss** (rev-2 correction — rev 1 wrongly called them "unused"; the E2 falsification depends on exactly this supervision), but the **autograd slope NR consumes is never tied to them**; `sample_class` is saved in every npz but **discarded by the loader** (`dataset.py:85-90`).

## Execution order

**P0 → R0 → P1 → P2 (week 1, all 0 GPU) → re-freeze baseline → full-arm campaign (P4 lead, P3, P5, P6, P7 + P8a rider; ≥4 seeds per config) → promotion.** A decision table (Sequencing §2) re-shapes campaign funding from week-1 outcomes. Gate files go under `results/v6_4_7/`.

---

## Ranked proposals

### P0 — Fix the NMOS source-frame bug, A/B, re-baseline (0 GPU, **highest priority**)

~3 LOC: apply the PMOS shift to NMOS too in `_raw_voltages` — `return v_d - v_s, v_g - v_s, 0.0, v_b - v_s`. Voltage-shift invariance makes this exact physics; the shifted SRAM-attractor Vbs (−226 mV) is inside the trained vb box [−541, +750] mV. Rule 15 is unaffected (`_apply_vds_correction` already receives the difference `vds = v_d_nn − v_s_nn`).

- **A/B:** full 16-cell complex harness + force_ic 8 + inverter 8/8 + DC 55/55 + tran 64/64. All grounded-source suites (inverter, DC, tran, RO) are **bit-identical by construction** — which also means they are structurally blind to this change; therefore add a **lifted-source single-device sweep** (NMOS Id–Vgs at Vs∈{0.1, 0.2}·VDD vs OSDI, ~40 LOC in the parametric harness) — the vs input currently has zero verification coverage and is the real canary for P0 and P2.
- **Governance (pre-arbitrated):** TSMC5 opamp (PASS 2.64 %) was *selected* under the buggy frame and may move. This is a correctness fix, not a candidate — a moved TSMC5 opamp does **not** veto it. After P0 (and any R0/P2 code), **re-freeze the baseline as `results/v6_4_7/baseline_v6_4_7_pre.json` over all 16 cells + force_ic** before any campaign comparison; `baseline_v6_4_4.json` is stale the moment P0 lands.
- **Decides:** how much of opamp / switchcap / SRAM is inference-frame error vs model error — re-prices P2/P3/P4/P5 EV wholesale. P5 corridor harvesting must wait for P0 (pre-P0 opamp/SRAM/SC trajectories are frame-corrupted).
- **Kill:** none — the bug is real regardless of how many gates it flips. If the lifted-source sweep shows the shifted frame regressing single-device accuracy vs OSDI (it should not — the shift moves evaluation *into* distribution), stop and investigate before shipping.

### R0 — Switchcap measurement fix + root-cause package (0 GPU)

Three free items, run before any campaign arm is shaped — until rev 2, 3 of the 7 failing gates were diagnosed by inference only.

1. **Droop sub-gate repair (~5 LOC, `verify_complex_switchcap.py:133-137`):** replace the pure-relative criterion with an absolute-floored one (e.g. `|dn−ng| ≤ max(DROOP_TOL·|ng|, f·VDD)`). This simultaneously removes the unpassable ~0.1 µV≈VNTOL direction *and* closes the nan-guard hole that auto-passes TSMC7's 2.178 mV droop. Same measurement-fix class as V6.4.6's SRAM probe correction — subject to the same adversarial false-PASS review (the E3 lesson: a measurement change that flips a gate must be shown to be a correction, not a loosening).
2. **`NN_SYMMETRIC_CAPS=1` SC-only re-test** on post-P0 code: recorded V6.4.5 zero-code probes show it improved SC charge on *every* failing tech (TSMC12 8.33→5.88, TSMC16 13.13→7.65, TSMC5 14.68→13.75 %) — TSMC12 lands 0.9 pp from the ≤5 % gate. D1 stays a dead end **for RO**; this is a per-circuit re-scoping, not a dead-end violation. If it flips a cell, the promotion protocol must explicitly decide per-circuit env-gated shipping.
3. **Per-device id/charge dump aimed at the sample window + phi falling edge** (conduction + charge feedthrough), ~150 LOC reusing P0-D plumbing. The rev-1 dump targeted the hold window under the retracted droop hypothesis — wrong window, wrong hypothesis.

- **Decides:** the SC ownership split between frame (P0), reverse-clamp hold-freeze (P2), charge model (P5/P7), and cap stamping (symcaps) — and updates the failing-gate census the whole campaign budgets against.

### P1 — Complete the causal swap matrix: inject exact OSDI **id AND charges together** into the live TSMC7 RO (0 GPU)

The missing cell nobody ran. P0-I injected exact id with NN charges (92 ps); exact-everything was never measured. (Runs on post-P0 code; the RO is bit-identical under P0, so results remain comparable.)
- **Build:** extend `scripts/v6_4_6_p0i_id_injection_v2.py` (v2-analytic scheme already does live OSDI `eval_dc` in the NR loop; charges+caps come from the same call). ~200–400 LOC, ~4–8 wall-hours.
- **Decides:** period → ~46.6–47 ps ⇒ simulator exonerated, joint model error confirmed, P0-H's "charges exact" shown trajectory-conditional ⇒ fund P5. Period still ~92 ps with everything exact ⇒ **solver/harness bug** — every model-side RO lever pauses (would be the biggest find of the iteration). Also explains the P0-I anomaly: NN id/charge errors currently *cancel*, so id-only improvements can regress RO — every retrain below must score the live RO period early.
- **Kill-criteria:** none — both outcomes decisive.

### P2 — Relax the Rule-15 reverse-Vds clamp (0 GPU; SRAM + switchcap hold-freeze)

The biggest inference-side model lever after P0. `mosfet_nn.py:569` (`f_id = 0` when reverse) + wrong-sign clamp (d) at `:579-585` hard-zero all reverse conduction — yet the V6.3 `reverse_vds` corridor class (7.48 % of all training rows, Vd∈[−0.30, −0.01]·VDD) was added *specifically* so the NN could learn reverse conduction, and the clamp was deliberately deferred, never loosened (`docs/plans/2026-05-14-v6.3-spike-removal.md:173-178` — a deferred lever, **not** a dead end). At the SRAM attractor the ON-PMOS Mpr (Vds = +65 mV, reverse) reads id = −0.0 vs OSDI **−19.85 µA**, with node q stranded 65 mV above the rail and its restoring path zeroed.

- **Jacobian consistency is on you, not autograd (rev 2):** the four-part correction is applied **post-autograd on Python floats** (`mosfet_nn.py:509-587`) — autograd differentiates the raw network only; consistency is maintained by hand-derived factors (`gm *= f_id` at `:574`, product-rule gds at `:575`). Any relaxation of `:569`/`:579-585` **must hand-derive the matching gm/gds factors**.
- **For SRAM, the conductance is the point, not a side effect (rev 2):** the railed point's NR-instability (P0-A/E2: continuation folds at g*≈1e-5 S via `Δqb = residual/g_qb` explosion) is cured precisely by the missing reverse conductance — OSDI-implied at the Mpr bias: |19.85 µA| / 65 mV ≈ **305 µS ≫ the 1e-5 S fold threshold**. Carry the sign into the relaxed gds, then apply the Rule-4 `max` floor (never `abs`).
- **Probe before building (~20 LOC):** dump the NN's **raw pre-clamp** reverse id at the P0-D Mpr bias and on a reverse-corridor grid. The corridor covers the SRAM bias (+65 mV ≈ 0.09·VDD), so the surface *should* be trained there — but if the raw reverse surface is garbage, the relaxation injects noise into every circuit crossing Vds=0 (including the three **passing** ROs, which are blind vetoes). Garbage ⇒ record dead end; SRAM rides on P0+P3 (+P9 fallback).
- **Build (~80–150 LOC in `_apply_vds_correction`):** relax (b) and (d) together, with **(d) scoped by direction** — keep the wrong-sign clamp in the normal direction (it guards forward subthreshold sign noise), bypass it only for reverse conduction; a smooth odd blend across Vds=0 designed in up front (the circuits that *live* at Vds≈0 — switchcap hold, SRAM attractor — are exactly the targets; don't discover the C¹ join by NR chatter); **taper the relaxed reverse current to zero beyond ~0.30·VDD reverse** — past the trained corridor the raw surface is extrapolation.
- **Closes (re-scoped, rev 2):** SRAM force_ic is co-owned three ways (P0 frame + P2 reverse + P3 weak-inversion) — expect **P0+P2+P3 jointly** for 8/8, with P2 removing the NR-instability obstacle. SC: P2 may unfreeze the TSMC12/16 overshoot's bleed-back during hold (the reverse clamp currently freezes it).
- **Kill:** regression on inverter 8/8 / DC 55/55 / tran 64/64 / the new lifted-source sweep / the three passing ROs → soft-blend variant; if that regresses too, record dead end. (The legacy hard gates are grounded-source and nearly blind to the reverse path — the passing ROs and the lifted-source sweep are the real canaries.)

### P4 — Sobolev id-derivative supervision against analytic OSDI gm/gds/gmb (opamp lead arm)

Opamp DC gain is the peak |dVout/dVin| of the converged DC locus = the **true derivatives of the learned id surface** — and 3 of the 7 failing headline cells are opamp, one of them (TSMC12, 10.94 % vs the ±10 % gate) a **+1.1 % gain move** away. Promoted above P3 among GPU arms on gate-count EV. Rev-2 reframing: gm/gds/gmb heads are *already supervised* — P4 **consistency-couples the autograd slope NR consumes to those heads**, converting the E2-documented smoothness prior into a constraint on the surface the solver actually uses. Also attacks the D5 collapse mode (gain→0 = derivative-surface fragility) for every other arm.

- **Design around the V5 Phase-C precedent** (`results/v5_phase_c_jac_loss_ab_2026_05_07.md`: 8-channel JAC loss was *detrimental* at S-scale/159 k params; implementation deleted in the v6 refactor — **recover via `git show 930c274`**, the asinh chain-rule transform is already written there): this time **id-channels only** (one extra `autograd.grad` with `create_graph=True`), M-scale (520 k params), λ-swept, **fine-tune from V6.4.4 first** (needs a ~20-LOC init-from-checkpoint kwarg in `train_directnet`), full retrain second, judged on the actual opamp gain — a metric that didn't exist as a gate in V5.
- **Supervise gds in relative/asinh space and importance-weight the opamp op-point corridor (rev 2):** gain ∝ gm/gds where gds is *small*; a physical-space MAE on ∂id/∂Vd is dominated by linear-region gds and buys nothing at the op point. Use the 930c274 chain-rule transform for scale balance, and reuse P5's `sample_class` plumbing to upweight a harvested opamp op-point neighborhood. A globally-uniform λ small enough to protect the VTC is likely too dilute at the op point — the most probable failure mode of P4 as originally written.
- **Veto set must include the currently passing RO cells** — TSMC5 (2.98 % PASS) and TSMC12/16: P0-I showed id-surface improvements can regress RO through id↔charge cancellation; the rev-1 blind-veto list omitted TSMC5 RO.
- **Prerequisite infra (~30 LOC):** the candidate scorer cannot see opamp gain error (`eval_v6_4_5_candidate.py:148-167` returns only flat-flag + raw gain) — add NG reference gains + `gain_err`.
- **Gotchas:** per-channel sign conventions (OSDI gds = −∂id/∂Vd vs NN frame — probe each channel vs finite differences before training; this exact trap is documented in P0-I §2); use y_true in the asinh sqrt factor; no inference-only clamps at train time.
- **Closes:** opamp TSMC7 + TSMC12; makes every other retrain arm collapse-resistant. **Effort:** ~250 LOC, ~30–60 GPU-h at 4 seeds × λ grid. **Kill:** fine-tune arm at best λ doesn't cut TSMC7 gain err below ~15 % with inverter held → don't fund the full-retrain arm; record next to the V5 entry.

### P3 — Decade-balanced subthreshold id objective + regen (SRAM — ship-required)

The failure is the **loss transform, not coverage** (subthresh class already has ~97 k rows/cell; asinh crushes the leakage band to 0.011 % of range). Primary target: SRAM force_ic, ship-required per user ruling. The rev-1 "closes switchcap droop (2 of 3 SC fails)" claim is **retracted** (R0: droop was a measurement artifact; SC charge-transfer is not subthreshold-owned). Possible opamp TSMC16 assist retained pending P0/R0 evidence.

- **Stage 1 (no regen, ~80 LOC trainer):** auxiliary id loss in small-scale asinh space — `MAE(asinh(id_phys/s2))` with s2≈1e-9, Huber for robustness — with rev-2 design fixes: (i) **upper mask `|id_true| < 1e-6`** (or a decade taper) so the auxiliary is genuinely subthreshold and does not re-weight the trip band where the ~20× VTC gain lives; (ii) **one-sided ceiling hinge on masked-out rows** — `relu(|id_pred| − k·NFIN·1nA)` where `|id_true| ≤ 1e-10` — the hard-OFF device (Mpl: NN +0.50 µA vs OSDI −0.0) otherwise receives zero auxiliary signal; a *ceiling* penalty is sign-agnostic, so the sign-random OSDI floor noise is irrelevant to it (and it is not D4's `Ioff_rail` floor — it suppresses current, never injects it); (iii) **exempt `reverse_vds` rows** from the a-priori sign sanitization; (iv) write the torch-differentiable denorm chain (~20 LOC) — `normalize.py:218-236` is a float/numpy inference helper, not reusable as-is.
- **Stage 2 (regen) is mandatory, not contingent (rev 2):** the 1e-12–1e-9 decade is empty (28 rows total — likely OSDI internal-solve tolerance in `eval_single_point`); no loss can teach sub-nA roll-off from absent data, and SRAM needs multi-decade suppression. Fix the generator floor and densify Vgs∈[VTH−0.1, VTH+0.15] up front; budget it in the campaign.
- **Coordinate with P2:** re-run the P2 A/B after any P3 retrain — the clamp currently hides whatever the reverse surface learned.
- **Effort:** ~3.5 GPU-h per 8-cell sweep × λ grid × 4 seeds. **Risk:** D4 territory — the pinning device sits at Vov≈+45 mV, the same weak-inversion band as the inverter trip; select through the multi-circuit scorer with inverter VTC as a hard gate. **Kill:** weak-inversion id ratio at the P0-D probe biases not improved ≥10× while inverter VTC holds ≤5 % → escalate to P9.

### P5 — Trajectory-corridor overlay data regen — **harvest from NGSPICE, not NN** (conditional on P1)

If P1 lands ~46.6 ps, the confirmed defect is the joint (id, q) surface along circuit-visited biases. **Rev-2 change: harvest the bias corridors from the NGSPICE reference waveforms** — already produced by `tests/common/complex.py` for all four circuits — not from NN sims. Ground-truth-trajectory harvest is the teacher-forced standard; it eliminates rev-1's top stated risk (biases from the wrong trajectory) *and* the budgeted harvest→retrain second iteration. Harvest only **after P0** (pre-P0 NN trajectories are frame-corrupted — doubly wrong). Evaluate OSDI on the corridors, add as a weighted sample class — the same move as `inv_trip`, the project's single most successful data lever (TSMC5 16.90 % → 0.92 %), at the V6.3.1-corrected dosage (~3.5 % of rows).

- **Critical plumbing:** LDS normalizes per output-value bin → densification buys no loss mass. Plumb `sample_class` through `load_and_split_bsimar` (currently dropped at `dataset.py:85-90`) and multiply per-class weights into the LDS tensor — **after** the per-target mean-normalization (`trainer.py:163-176`), then **renormalize the product to unit mean**, otherwise the effective learning rate changes and confounds every A/B against control (~40 LOC).
- **Weight corridor rows by residence time × |id|** (a poor man's adjoint sensitivity, ~10 LOC) rather than uniformly.
- **Closes:** RO (if P1 says model), switchcap charge-transfer (if R0 says charge-model), opamp TSMC16 op-point. **Effort:** ~300–400 LOC + regen 8 datasets + ~10–20 GPU-h. **Risk:** overlay overdose (V6.3 Phase-B cautionary tale) — hold the ~3.5 % dosage. **Kill:** first scored arm's RO err not <7 % → stop at one iteration.

### P6 — Ensemble-mean distillation + SWA/EMA (variance/collapse insurance)

Train-free teachers first (rev 2): the **32 `v6_4_5_p5_tsmc7_*` checkpoints (16 stock + 8 mono recipes) survive on disk** — a 0-GPU TSMC7 seed bank; the distillation script is ~50–100 LOC and the student ~2–4 GPU-h. Other techs: distill from each campaign arm's 4 seeds.
- **Expectation re-set (rev 2):** all 32 recorded seeds land 50.8–53 ps on the RO — the RO error is **seed-invariant bias**. P6 kills *variance* (the 13/16 opamp gain→0 collapse, selection overfitting); it does not close RO. Score it as an opamp-robustness and selection-stability lever; mean-surface distillation smooths derivatives, synergizing with P4.
- **SWA/EMA (~20 LOC, `torch.optim.swa_utils`) becomes a default flag on every campaign arm** — within-run weight averaging smooths derivative surfaces at near-zero cost; cheaper than distillation, always on.
- **Kill:** distilled/averaged net not Pareto-≥ the best single seed on the scorer vector.

### P7 — Split-head architecture with **retained** 13-target supervision (campaign arm)

Decouple the id head from the charge head on a shared trunk (optionally widen). Do **not** drop the 9 unused output heads — that is falsified at M-scale (E2-medium-4out regressed 21.5 % vs 10.2 % VTC NRMSE; `docs/plans/2026-05-08-directnet-target-trim.md`, commit `da52ca5`): supervision is a smoothness prior at 520 k params. ~150 LOC + ~10 GPU-h, run only inside the scorer-selected campaign. **Kill (re-banded, rev 2):** inverter regression **beyond the documented ±1 % run-to-run scatter** at equal data/loss — rev-1's "any regression" would trip on noise.

### P8 — Teacher-forced trajectory supervision (RO endgame; replaces the rev-1 adjoint build)

Rev-1's circuit-in-the-loop adjoint mis-stated the codebase ("PyTorch end-to-end already" — false: the solver is scipy/numpy MNA + `spsolve`; torch is detached at the device leaf) and underestimated the build 2–5×; the adjoint is **out of V6.4.7 scope**. Replacement, same conceptual target — the joint (id, q) property along the *real* trajectory:

- **P8a — teacher-forced one-step companion-residual supervision (~150–250 LOC):** at each NGSPICE reference timestep, supervise the NN's id and charge *values* at the true-trajectory biases — exactly the quantities the transient stamps (`_stamp_mosfet_dc:304`, `_stamp_mosfet_transient:1772`) and exactly where P0-H's ~20 % NMOS peak pull-down deficit lives. No unrolling, no adjoint; composes with P5 (same harvested corridors, adds the per-point supervision). Can run as a cheap campaign rider.
- **P8b (optional) — zeroth-order LoRA on the live period:** SPSA/CMA-ES over a 10–100-parameter low-rank correction, scored directly on the live RO period (~385 s/eval, embarrassingly parallel). Directly optimizes the non-separable joint property P0-I identified. **Must be gated by held-out circuits** — it optimizes the gate itself.
- Fund the full P8b search only if P5 fails while P1 exonerated the simulator.

### P9 — Physics-anchored multi-region subthreshold core (SRAM fallback)

Compose-at-inference: frozen MLP owns strong inversion, closed-form weak-inversion exponential (ideality pinned n≤1.3) owns OFF. A legitimate fallback now that force_ic is ship-required. Keep behind the recorded go/no-go fit gate (≥4-decade suppression AND ≤5 % inv_trip simultaneously). Only after P0+P2+P3 all fail to close force_ic — the P0-D caveat (pinning device ≈VTH) makes it D4-adjacent.

### Dropped / not funded

- **SRAM gate re-spec to transient write-then-hold** — predicted ineffective, not just unprincipled: the railed point is NR-unstable (P0-A/E2), so a hold-test drifts to the same inboard attractor; once the model improves, force_ic passes anyway.
- **Solver/integration levers for RO** — dead per P0-G (truncation ~0.4 ps). gds floor tweaks — Jacobian-only, inert at fixed points.
- **Rev-1 P8 adjoint circuit-in-the-loop** — replaced by P8a/P8b (codebase mis-statement + 2–5× cost underestimate).
- All recorded dead ends as-is: D2 substeps, D3 warm starts, D4 Ioff_rail floors, D5 seed lotteries without recipe change, gds/cap Jacobian distillation for RO, charge-value distillation, force_ic homotopy. **D1 symmetric caps stays dead for RO only** — its recorded SC-charge improvement is re-tested in R0 (per-circuit re-scoping, not a dead-end violation).

## Sequencing

1. **Week 1 (0 GPU):** P0 frame fix + lifted-source sweep → R0 switchcap package → P1 swap matrix → P2 raw-probe, then clamp relax → scorer extension (`gain_err` + **switchcap cells** — the scorer currently has none, and `baseline_v6_4_4.json` contains no SC cells and no opamp TSMC7/16) → **re-freeze `baseline_v6_4_7_pre.json` over all 16 cells + force_ic** → **commit the campaign infra** (`scripts/eval_v6_4_5_candidate.py`, `scripts/v6_4_5_search.py` are still untracked in git).
2. **Decision table — week-1 outcomes re-shape campaign funding:**

   | Week-1 outcome | Re-shape |
   |---|---|
   | P0 flips opamp/SC cells outright | shrink P4/P5 scope to the residual; recount and re-rank |
   | P0 moves TSMC5 opamp (selected under buggy frame) | pre-arbitrated: correctness fix ships regardless; re-baseline |
   | R0 droop-gate repair changes the SC census | recount failing gates; re-budget the campaign |
   | R0: symcaps flips TSMC12 SC | decide per-circuit env-gated shipping explicitly at promotion |
   | R0 dump: SC charge error is frame-owned | drop SC from P3/P5 EV claims |
   | P1 → ~46.6–47 ps | model confirmed → fund P5 |
   | P1 → still ~92 ps | solver/harness bug — pause **all** model-side RO levers |
   | P2 raw reverse surface is garbage | P2 dead end; SRAM rides on P0+P3 (+P9 fallback) |
   | P0+P2 already lift force_ic to ≥6/8 | P3 demoted to a cleanup λ arm |

3. **Weeks 2–4 (full-arm campaign, ~250–300 GPU-h per user ruling):** arms = **P4 (lead)**, P3, P5 (if funded by P1), P6/P7/P8a riders; **≥4 seeds per config**; SWA/EMA flag on every arm. **Stage A:** arms run independently against the post-week-1 control at equal data and seeds. **Stage B:** compose the top-2 compatible winners (P3 loss + P4 loss stack; P5 data composes with either) and re-score — composition is budgeted, not assumed. Hold a **~20 % budget reserve** for scoring wall-clock (V6.4.5's 32-candidate scoring was a full phase; this campaign is larger with a longer vector).
4. **Week 5:** promotion per the V6.4.6 protocol (baseline JSON, blind holdouts, per-tech mix allowed) + the rev-2 selection discipline below; CHANGELOG/CLAUDE.md; dead-end records for killed arms.

## Verification

- P0/R0/P2: full complex harness (`tests/verify_complex_{ring_osc,opamp,sram_snm,switchcap}.py`) + force_ic probe + inverter gate (`verify_nn_dc_tran.py --inverter-only`) + extended harness (55/55, 64/64) + **the new lifted-source DC sweep**, `OMP_NUM_THREADS=1`.
- Retrain arms: multi-circuit scorer vector per candidate (Rule-16 quartet per cell); **blind vetoes extended to ALL currently-passing cells** — the rev-1 list omitted TSMC5 ring_osc (2.98 % PASS) and SC TSMC7; the re-frozen baseline covers all 16 cells + force_ic; snapshot checkpoints + `manifest.sha256` before any campaign.
- **Selection discipline (rev 2):**
  1. **Replicate the top-3 candidates 3×** (different thread counts) before promotion; require winner − runner-up > replicate σ — min-over-N selection across 40+ candidates inflates the winner's apparent margin.
  2. **Held-out perturbed circuit variants** (different RO stage count, opamp Cload, SRAM cell ratio — cheap netlist edits) are never scored during selection and run **once, blind, at promotion** — the only test that a winner generalizes rather than memorizing the 16 gate configs.
  3. **The promotion rule is pre-registered here:** promote the per-tech best on the scorer vector subject to all vetoes; per-tech mix allowed with ≥4-seed evidence per promoted cell; any mid-campaign scorer change requires a recorded amendment (the V6.4.5 flat-flag re-calibration precedent).
- **Success =** headline **>9/16** AND **SRAM force_ic improved (target 8/8)** with no protected gate regressed; every killed arm recorded with the numbers that killed it.
