# DirectNet V6.4.7 — Ranked levers to improve NN compact-model accuracy in complex-circuit simulation

**Date:** 2026-06-10 (rev. 2, same day; rev. 2.1 same day — sequencing serialized per user instruction: every lever is committed-or-rewound before the next starts; **rev. 3, 2026-06-12 — three user rulings reshape the GPU campaign: keep small-current rows + alter the loss, make ∂id/∂V precise, regen+retrain unconditional; new step S9b**)  •  **Status:** **WEEK 1 (S1–S8) COMPLETE 2026-06-12 — headline honest 8/16 → 11/16, zero GPU; see "Week-1 outcomes" §. S9 (SWA/EMA infra) SHIPPED 2026-06-12. S9b COMPLETE 2026-06-14 — regen-v2 + control-v2; go/no-go = PROCEED. **S10 (P4 Sobolev id-derivative arm) COMPLETE 2026-06-14 — verdict KILL: the term improves deriv fidelity + inverter but collapses the opamp 4/4 (systematic); major finding = derivative fidelity is anti-correlated with the value-owned opamp/RO gates (P0-C/P0-I class), partially falsifying ruling-4's premise. See "S10 outcomes" § below.** **S12 (P5 trajectory-corridor arm) COMPLETE 2026-06-15 — verdict KEEP: tsmc7 RO 8.28→2.9 % (all 4 seeds, the P5 dynamic-id thesis confirmed), per-tech mix 11/16 → 14/16 (tsmc7 RO + tsmc16 opamp s31 + tsmc16 SC); cost = collapses *passing* opamps (tsmc5/tsmc12), avoided by per-tech mix; force_ic still 0/8. RESUME AT S11 (P3 SRAM, ship-required force_ic). See "S12 outcomes" § below.** Originally PROPOSED — REVISED after four-agent adversarial review (dead-end audit, ML methodology, simulator numerics, EV/risk)  •  **Branch:** `feat/v6.4.7` (cut from `feat/v6.4.6` @ `c5155e7`; week-1 commits `c2ac02b`…`ad62c68`)
**Authoring:** rev 1 — plan-mode synthesis + staff-engineer adversarial review (checked against recorded V5–V6.4.6 dead ends; three conflicts designed around: V5 Phase-C JAC-loss negative, E2-medium head-trim falsification, P0-A NR-instability of the railed SRAM point). rev 2 — four-agent panel found the **P0 NMOS source-frame bug**, retracted the switchcap droop premise, re-powered the campaign, and hardened selection discipline. Proposal IDs are stable from rev 1; the rev-2 ranking is the section order below (P0/R0 new; P4 now precedes P3 among GPU arms). rev 3 — user-directed amendment after week 1 (recorded under the S5b mid-campaign-change precedent): a 2026-06-12 pipeline audit found the loader silently drops every |id| ≤ 1e-15 row (6.0–7.9 % of each dataset) on top of the known empty generator band, and the three rulings below re-anchor the campaign on unfiltered small-current data, precise ∂id/∂V, and unconditional regen+retrain.

**User rulings (2026-06-10):**
1. **SRAM `force_ic` is ship-required.** It is not one of the 16 headline cells (`verify_complex_sram_snm.py` counts the butterfly gate only), but it joins the V6.4.7 success criteria. P2's SRAM half, P3, and the P9 fallback are funded on this basis.
2. **Full-arm campaign authorized.** Budget raised to **~250–300 GPU-h** so every arm runs at **≥4 seeds per config** — the documented 139 mV seed lottery and the 13/16 spontaneous opamp-collapse rate make 1-seed arms indistinguishable from noise (a 1-seed arm "collapses" with ~80 % probability regardless of recipe; V6.4.5's central failure was attributing seed luck to recipes).

**User rulings (2026-06-12, rev 3 — recorded mid-campaign amendments under the S5b precedent):**

3. **Do NOT filter small currents/values out of the training sample — and alter the loss so they carry signal.** Small currents are load-bearing in complex circuits (SRAM retention/force_ic, the TSMC16 SC hold leak ~3.9 mV exposed by S5b, the NN leakage equilibrium S5 found at vsamp(0)=0.39–0.70 V). The 2026-06-12 audit measured the live pipeline: `filter_small_targets` (`dataset.py:39-52`, `DEFAULT_FILTER_THRESHOLDS={"id": 1e-15}`, hard-wired `apply_filter=True` at `trainer.py:316` for DirectNet and `:400` for the Transformer) drops **6.0–7.9 % of every per-tech dataset** (TSMC5 N 171.5k/7.43 %, TSMC7 N 124.8k/6.00 %, TSMC12 200.1k/6.93 %, TSMC16 203.9k/7.06 %, TSMC7 P 190.2k/7.85 %), and the generator leaves the 1e-15–1e-9 A band essentially empty (14–27 rows per cell in (1e-10, 1e-9], 0–3 below). Net: **the network trains with zero off-state supervision** — the sub-nA leakage band is either absent (generator) or discarded (loader). The filter is removed; because asinh+LDS as-is gives the retained rows ~zero loss mass and sub-floor OSDI rows are solve-tolerance noise with random sign (the filter's original v5 §4-B4 rationale), **the loss must be altered in the same step** — magnitude-aware/sign-agnostic terms, see P3 rev 3.
4. **Derivative-consistency of id in the loss must be re-considered; ∂id/∂V must be made precise.** The autograd gm/gds/gmb the solver consumes (NN Rule 1) is currently tied to nothing — the OSDI derivative columns supervise the 13 output heads but never the id head's slope. P4 is elevated from an opamp-scoped lead arm to a **core campaign requirement**: the Sobolev id-derivative consistency term is carried by every funded retrain arm, and derivative fidelity (autograd gm/gds/gmb NRMSE vs held-out OSDI columns; P0-B baseline gds 20–23 %) becomes a first-class scorer metric and promotion criterion. See P4 rev 3.
5. **Data regeneration and model retraining are necessary — unconditional.** P3 Stage-2 regen and full retrains are no longer contingent on kill-test outcomes: the campaign commits to regenerated datasets (generator floor fix + subthreshold/OFF densification, decade-occupancy acceptance gate) and to full retrain arms trained on them. Fine-tune sub-arms remain as cheap λ-screens only (they pick λ and catch sign-convention bugs early); kill criteria on individual loss terms now gate *which terms enter the final stack*, not *whether a retrain happens*. Executed as new step **S9b** before any arm.

## Context

V6.4.4 is canonical at **9/16** complex-circuit gates. The 7 failing headline cells: **1 ring_osc** (TSMC7, 8.97 % period err), **3 opamp** (TSMC7 30.7 %; TSMC12 10.94 % against a **±10 % gain gate** (`GAIN_TOL=0.10`, `verify_complex_opamp.py:44`) — i.e. a **+1.1 % gain move** flips it, the cheapest gate on the board; TSMC16 flat/gain-0), **3 switchcap** (charge-transfer — see corrected row below); plus the ship-required **SRAM `force_ic` 0/8** (inboard attractor q≈0.87/qb≈0.20 on all techs). The user has authorized data regeneration and full retraining; project-rule constraints (e.g., Rule 10 no-new-losses) are waived for this proposal.

### Rev-2 headline finding — the NMOS source-frame bug (→ P0)

`_raw_voltages` (`pycircuitsim/models/mosfet_nn.py:232-243`) source-shifts **only PMOS**; NMOS passes **absolute** terminal voltages into a network trained exclusively at Vs≡0 (`pycmg/nn_generate.py:23`; shipped norm stats have `input_min[2]=input_max[2]=0`). The softplus input clamp (`mosfet_nn.py:245-261`) then pins vs≈0 while vd/vg/vb stay absolute — any **lifted-source NMOS** is evaluated at phantom Vgs/Vds inflated by +vs with Vbs=0, and `∂id/∂vs≈0` through the saturated clamp makes the solver's source-coupling stamps fictitious. CLAUDE.md NN Rule 2 documents the PMOS-only shift, encoding the same blind spot; **no test in any suite sweeps an NMOS at Vs≠0**.

The affected-device census maps directly onto the failing families:

- **opamp** — the diff pair Mn1/Mn2 sits on `vtail` (`examples/complex/miller_opamp_directnet.sp:23-24`): the gain-determining devices are mis-framed and blind to their own source node (no source degeneration, no body effect — a natural mechanism for the D5 gain→0 collapse fragility);
- **switchcap** — the pass NMOS has source=`vsamp`: phantom Vgs=VDD/Vds=vin drive while sampling explains DN charging to/past Vin (DN **overshoots** vin=0.48 at 0.4866/0.5098 on TSMC12/16);
- **SRAM** — access transistors Mal/Mar are lifted; Mar reads NN −125.5 vs OSDI −57.2 µA (**+68 µA — the largest single device error at the P0-D attractor**, 3.4× the Mpr reverse-clamp error);
- **ring_osc / inverter** — every source at a rail ⇒ **bit-identical under the fix**, consistent with RO being the smallest, genuinely model-owned gap.

### Root-cause table (corrected; **rev-2 snapshot — superseded in part by "Week-1 outcomes" below: the ring_osc row's P0-I non-separability claim is RETRACTED (S6), and the SRAM/switchcap rows are re-owned by S5/S5b/S7 findings**)

| Gate | Established owner | Evidence |
|------|------------------|----------|
| ring_osc TSMC7 | **joint (id, charge) model property** — NOT Jacobian (P0-C: inert ≤0.01 ps), NOT integration (P0-G: ~0.4 ps of 4.18 ps), NOT charge values alone (P0-H: ≤2 aC); NMOS dynamic id ~20 % peak under-prediction, but id is **not separable** (P0-I: exact-id injection → 92 ps, wrong direction) | `results/v6_4_6/phase0{C,G,H,I}_*.md` |
| SRAM force_ic | **co-owned by three current errors at the attractor:** (1) Mar lifted-source frame corruption, +68 µA (→ P0); (2) pinning NMOS Mnr weak-inversion over-prediction 7.5× (6.36 vs 0.84 µA at Vov≈+45 mV; → P3); (3) ON-PMOS Mpr restoring current **zeroed by the Rule-15 reverse-Vds clamp** (NN −0.0 vs OSDI −19.85 µA; → P2); asinh maps the 1nA–100nA band to 0.011 % of normalized id range | `phase0_D`, `phase0_E` + rev-2 audit |
| opamp | gain = true autograd derivatives of the learned id surface at the op point; gds NRMSE 20–23 % (P0-B) — inert for RO but **load-bearing for gain**; 13/16 fresh retrains collapse gain→0 (D5); diff-pair frame corruption (P0) compounds on top | `phase0_B`, V6.4.5 P5 + rev-2 audit |
| switchcap | **CORRECTED (rev 2):** the binding failure is the **sample-phase charge-transfer level** (TSMC5 14.68 %, TSMC12 8.33 %, TSMC16 13.13 % of VDD; gate ≤5 %), with DN overshooting Vin on TSMC12/16 — plausibly frame-owned (P0), hold-frozen by the reverse-Vds clamp (P2). **The rev-1 droop≡subthreshold-floor story is retracted:** recorded DN droops are 26 µV / 2 µV (`results/v6_4_5/phase1_logs/switchcap.log`), ~3 orders below any µA-floor prediction; the 2325 %/241 % figures are ratios against a ~1 µV NGSPICE reference at the harness noise floor, and the droop sub-gate as coded demands ~0.1 µV ≈ VNTOL agreement — **numerically unpassable noise** (measurement artifact, repaired in R0). The same gate auto-passes TSMC7's 2.178 mV droop (the largest absolute disagreement) via the `abs(ng_droop)>1e-6` nan-guard (`verify_complex_switchcap.py:133-137`) | `switchcap.log` + rev-2 audit |

Training-pipeline facts behind these: data is DC-only pointwise op-points; loss is MAE × per-target LDS (LDS is inverse-frequency over *output-value* bins → densifying an input region buys **no** loss mass); the analytic OSDI derivative columns (gm/gds/gmb, caps) **are supervision targets of the 13-output loss** (rev-2 correction — rev 1 wrongly called them "unused"; the E2 falsification depends on exactly this supervision), but the **autograd slope NR consumes is never tied to them**; `sample_class` is saved in every npz but **discarded by the loader** (`dataset.py:85-90`); **and the loader silently drops every row with |id| ≤ 1e-15 A** (`filter_small_targets`, `dataset.py:39-52`, hard-wired `apply_filter=True` at `trainer.py:316/:400` — measured 6.0–7.9 % of rows per cell, 2026-06-12 audit; rev 3), so on top of the empty 1e-15–1e-9 generator band the hard-OFF device class never reaches training at all.

## Execution order

**Strictly serial — one lever at a time, dead ends rewound before the next starts:**
S1 pre-flight ✅ → S2 P0 ✅ → S3 R0.1 ✅ → S4 R0.2 ✅ → S5 R0.3 ✅ → S6 P1 ✅ → S7 P2 ✅ → S8 re-freeze baseline + scorer ✅ → S9 SWA/EMA infra ✅ → S9b regen-v2 + control-v2 ✅ → S10 P4 ✅ **KILLED** (Sobolev deriv arm: deriv fidelity ⟂ value-owned opamp) → **[rev-4 REORDER]** S12 P5 ✅ **KEEP** (trajectory-corridor: tsmc7 RO 8.28→2.9 %, per-tech mix 11→14/16) → S11 P3 → S13 P8a → S14 P6 distillation → S15 P7 → S16 P8b (conditional) → S17 P9 (fallback) → S18 Stage-B composition → S19 promotion. **Resume point: S11 P3** (SRAM subthreshold, ship-required force_ic; carry the S12 force_ic rail-ward side-signal).

Each step starts from a committed clean state, runs its verification gates, and resolves — **progress → commit; kill criteria met → `git reset`/revert + dead-end record** — before the next step begins. The decision table (Sequencing §2) re-shapes the campaign after S8. Gate files go under `results/v6_4_7/`.

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

### P4 — Sobolev id-derivative supervision against analytic OSDI gm/gds/gmb (opamp lead arm) — ❌ KILLED at S10 (see "S10 outcomes")

> **STATUS (2026-06-14): KILLED.** Built + sign-verified; improves deriv fidelity + inverter but collapses the opamp 4/4 seeds (the gain/RO are value-surface-owned, not Jacobian-owned). Term dropped from the final stack; the rev-3 elevation of ∂id/∂V to a *promotion* criterion is amended by rev-4 ruling 6 (deriv gate → NR-robustness metric only). The rest of this section is the original (rev-3) proposal, retained for the record.

Opamp DC gain is the peak |dVout/dVin| of the converged DC locus = the **true derivatives of the learned id surface** — and 3 of the 7 failing headline cells are opamp, one of them (TSMC12, 10.94 % vs the ±10 % gate) a **+1.1 % gain move** away. Promoted above P3 among GPU arms on gate-count EV. Rev-2 reframing: gm/gds/gmb heads are *already supervised* — P4 **consistency-couples the autograd slope NR consumes to those heads**, converting the E2-documented smoothness prior into a constraint on the surface the solver actually uses. Also attacks the D5 collapse mode (gain→0 = derivative-surface fragility) for every other arm.

- **Design around the V5 Phase-C precedent** (`results/v5_phase_c_jac_loss_ab_2026_05_07.md`: 8-channel JAC loss was *detrimental* at S-scale/159 k params; implementation deleted in the v6 refactor — **recover via `git show 930c274`**, the asinh chain-rule transform is already written there): this time **id-channels only** (one extra `autograd.grad` with `create_graph=True`), M-scale (520 k params), λ-swept, **fine-tune from V6.4.4 first** (needs a ~20-LOC init-from-checkpoint kwarg in `train_directnet`), full retrain second, judged on the actual opamp gain — a metric that didn't exist as a gate in V5.
- **Supervise gds in relative/asinh space and importance-weight the opamp op-point corridor (rev 2):** gain ∝ gm/gds where gds is *small*; a physical-space MAE on ∂id/∂Vd is dominated by linear-region gds and buys nothing at the op point. Use the 930c274 chain-rule transform for scale balance, and reuse P5's `sample_class` plumbing to upweight a harvested opamp op-point neighborhood. A globally-uniform λ small enough to protect the VTC is likely too dilute at the op point — the most probable failure mode of P4 as originally written.
- **Veto set must include the currently passing RO cells** — TSMC5 (2.98 % PASS) and TSMC12/16: P0-I showed id-surface improvements can regress RO through id↔charge cancellation; the rev-1 blind-veto list omitted TSMC5 RO.
- **Prerequisite infra (~30 LOC):** the candidate scorer cannot see opamp gain error (`eval_v6_4_5_candidate.py:148-167` returns only flat-flag + raw gain) — add NG reference gains + `gain_err`.
- **Gotchas:** per-channel sign conventions (OSDI gds = −∂id/∂Vd vs NN frame — probe each channel vs finite differences before training; this exact trap is documented in P0-I §2); use y_true in the asinh sqrt factor; no inference-only clamps at train time.
- **Rev 3 (user ruling 4) — elevated from opamp lead arm to CORE CAMPAIGN REQUIREMENT.** ∂id/∂V precision is now an explicit V6.4.7 objective, not a single arm's bet: (i) the Sobolev id-derivative term is **carried (λ per arm, possibly 0 only with recorded justification) by every funded retrain arm** — P3, P5, P7 included — not just S10; (ii) success is judged on TWO gates — (a) the opamp gain gate as before, and (b) a **direct derivative-fidelity gate**: autograd gm/gds/gmb NRMSE vs the held-out OSDI columns (P0-B baseline: gds 20–23 %) reported per Rule 16 on the full test split AND on the harvested op-point corridors; the scorer gains this metric at S9b; (iii) derivative supervision **reuses the P3 trust-floor mask** — the gm/gds/gmb columns of sub-floor rows are the same solve-tolerance noise as the id values, and supervising slopes on noise is the V5 Phase-C failure mode in miniature; (iv) the RO id-deficit connection: S6 re-armed id-only levers, and the ~20 % dynamic id peak pull-down deficit (P0-G/H) is an id-VALUE-plus-slope property along the switching trajectory — corridor-weighted derivative supervision (with P5's harvested classes) is the designed lever.
- **Closes:** opamp TSMC7 + TSMC12; makes every other retrain arm collapse-resistant. **Effort:** ~250 LOC, ~30–60 GPU-h at 4 seeds × λ grid. **Kill (re-scoped, rev 3):** fine-tune λ-screen at best λ doesn't cut TSMC7 gain err below ~15 % with inverter held → the Sobolev term is **dropped from the final loss stack** and the dead end recorded next to the V5 Phase-C entry — but the full retrain still happens (ruling 5) on the surviving stack; the screen gates the *term*, not the *retrain*.

### P3 — Decade-balanced subthreshold id objective + regen (SRAM — ship-required)

The failure is the **loss transform AND the loader filter, not sampling coverage above 1 nA** (rev 3 correction of the rev-2 "loss transform, not coverage" line: the subthresh class has ~97 k rows/cell and asinh crushes the leakage band to 0.011 % of range, *but additionally* `filter_small_targets` discards every |id| ≤ 1e-15 row — 6.0–7.9 % of each dataset — so the hard-OFF class is not merely down-weighted, it is absent). Primary target: SRAM force_ic, ship-required per user ruling. The rev-1 "closes switchcap droop (2 of 3 SC fails)" claim is **retracted** (R0: droop was a measurement artifact; SC charge-transfer is not subthreshold-owned). Possible opamp TSMC16 assist retained pending P0/R0 evidence.

- **Stage 1 (no regen, ~80 LOC trainer):** auxiliary id loss in small-scale asinh space — `MAE(asinh(id_phys/s2))` with s2≈1e-9, Huber for robustness — with rev-2 design fixes: (i) **upper mask `|id_true| < 1e-6`** (or a decade taper) so the auxiliary is genuinely subthreshold and does not re-weight the trip band where the ~20× VTC gain lives; (ii) **one-sided ceiling hinge on masked-out rows** — `relu(|id_pred| − k·NFIN·1nA)` where `|id_true| ≤ 1e-10` — the hard-OFF device (Mpl: NN +0.50 µA vs OSDI −0.0) otherwise receives zero auxiliary signal; a *ceiling* penalty is sign-agnostic, so the sign-random OSDI floor noise is irrelevant to it (and it is not D4's `Ioff_rail` floor — it suppresses current, never injects it); (iii) **exempt `reverse_vds` rows** from the a-priori sign sanitization; (iv) write the torch-differentiable denorm chain (~20 LOC) — `normalize.py:218-236` is a float/numpy inference helper, not reusable as-is.
- **Rev 3 (user ruling 3) — the loader filter is part of the failure and is REMOVED.** Stage 1 additionally: (v) expose `apply_filter` through `train_directnet`/CLI and turn it **off** for every V6.4.7 arm — all |id| ≤ 1e-15 rows stay in training; (vi) honour the filter's original rationale (v5 §4-B4: sub-floor rows are OSDI solve-tolerance noise with **random sign**) by supervising those rows ONLY through the sign-agnostic ceiling hinge of (ii) — magnitude suppression, never sign — with the hinge mask re-anchored to the data trust floor (`|id_true| ≤ 1e-12` once Stage-2 regen lands; `≤ 1e-10` before it) and the plain MAE term masked out below the floor; (vii) **re-fit and audit the asinh scale on unfiltered data**: ~7 % of rows now contribute the 1e-18 `_OUTPUT_LOG_FLOORS` value to the geometric-mean s_id fit, dragging s_id down roughly an order of magnitude, which silently re-allocates asinh resolution away from strong inversion where the ~20× VTC trip gain lives — if the control-v2 inverter VTC degrades, pin s_id at the filtered-fit value and let the auxiliary terms own the small-current signal. Also re-check `max_rows` caps (row counts grow ~7 %).
- **Stage 2 (regen) is mandatory — rev 2, now also user-ruled (rev 3, ruling 5):** the 1e-12–1e-9 decade is empty (28 rows total — likely OSDI internal-solve tolerance in `eval_single_point`); no loss can teach sub-nA roll-off from absent data, and SRAM needs multi-decade suppression. Fix the generator floor (tighten/verify the internal-solve tolerance so sub-nA points are trustworthy, not noise) and densify Vgs∈[VTH−0.1, VTH+0.15] up front. **Acceptance gate: every id decade in 1e-12–1e-6 A non-empty with ≥1 k rows per cell**, recorded in the regen gate file. Executed at S9b, before any arm trains.
- **Coordinate with P2:** re-run the P2 A/B after any P3 retrain — the clamp currently hides whatever the reverse surface learned.
- **Effort:** ~3.5 GPU-h per 8-cell sweep × λ grid × 4 seeds. **Risk:** D4 territory — the pinning device sits at Vov≈+45 mV, the same weak-inversion band as the inverter trip; select through the multi-circuit scorer with inverter VTC as a hard gate. **Kill:** weak-inversion id ratio at the P0-D probe biases not improved ≥10× while inverter VTC holds ≤5 % → escalate to P9.

### P5 — Trajectory-corridor overlay data regen — **harvest from NGSPICE, not NN** (conditional on P1)

If P1 lands ~46.6 ps, the confirmed defect is the joint (id, q) surface along circuit-visited biases. **Rev-2 change: harvest the bias corridors from the NGSPICE reference waveforms** — already produced by `tests/common/complex.py` for all four circuits — not from NN sims. Ground-truth-trajectory harvest is the teacher-forced standard; it eliminates rev-1's top stated risk (biases from the wrong trajectory) *and* the budgeted harvest→retrain second iteration. Harvest only **after P0** (pre-P0 NN trajectories are frame-corrupted — doubly wrong). Evaluate OSDI on the corridors, add as a weighted sample class — the same move as `inv_trip`, the project's single most successful data lever (TSMC5 16.90 % → 0.92 %), at the V6.3.1-corrected dosage (~3.5 % of rows).

- **Critical plumbing (pulled forward to S9b, rev 3 — P4's corridor weighting and P3's class-aware masks need it too):** LDS normalizes per output-value bin → densification buys no loss mass. Plumb `sample_class` through `load_and_split_bsimar` (currently dropped at `dataset.py:85-90`) and multiply per-class weights into the LDS tensor — **after** the per-target mean-normalization (`trainer.py:163-176`), then **renormalize the product to unit mean**, otherwise the effective learning rate changes and confounds every A/B against control (~40 LOC).
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

## Week-1 outcomes (2026-06-12, S1–S8 complete — this section supersedes stale claims above)

**Headline: 9/16 (honest restatement of V6.4.4's 8/16 + P0) → 11/16.** Every
step committed-or-rewound per protocol; commits c2ac02b (S1), e2a121a (S2),
d24c1d7 (S3), 5c6342b (S4), 6162cad (S5), 7454034 (S5b), 4e0b55e (S6),
bdf4102 (S7). Baseline re-frozen: `results/v6_4_7/baseline_v6_4_7_pre.json`.

1. **S2 = P0 frame fix shipped.** Canary 12/12 (pre-fix 10–64 % NRMSE);
   TSMC12 opamp flipped PASS (10.94 → 5.21 %); force_ic attractor halved its
   rail distance; switchcap unchanged ⇒ frame was NOT the SC owner.
2. **S3 = R0.1 droop gate repaired** (absolute-floored; adversarial verdict:
   correction/net-tightening). **V6.4.4's canonical 9/16 was honestly
   8/16** (TSMC7 SC pass was a nan-guard artifact).
3. **S4 = R0.2 symcaps KILLED for SC too** — charge win trades for
   30–137 mV hold drift (invisible under the old gate). D1 dead everywhere.
4. **S5 = R0.3 dump settled SC ownership**: solver numerics clean, charge/cap
   VALUES exonerated (ΔQ_q ≤ 0.5 % of gap), id error REV-clamp-concentrated;
   **dominant owner was a harness `.ic`/uic semantics gap** (DN started from
   the NN leakage equilibrium, vsamp(0)=0.39–0.70 V, vs NGSPICE's uic 0 V).
5. **S5b amendment shipped**: uic-equivalent constrained-`.ic` start in the
   DN runner. SC failures became honest forward-conduction errors; TSMC7 SC
   = fragile PASS (off-default-Vin variant mandatory in S19 holdouts).
6. **S6 = P1 resolved by a control nobody had planned: the native LEVEL=72
   path runs the RO at ratio 1.000 vs NGSPICE (46.64/46.65 ps) ⇒ simulator
   EXONERATED.** The planned exact-id+q injection (93.01 ps) reproduced
   P0-I's ~92 ps and is an **artifact of the injection id-path mapping** —
   **V6.4.6 P0-I IS RETRACTED**; id-only levers (P4, P8a, LoRA) re-armed;
   cap-sign conventions bounded at ±1.3 % (second-order); the RO gap reverts
   to the NN's ~20 % dynamic id peak pull-down deficit (P0-G/H), unclouded.
7. **S7 = P2 shipped** (reverse-Vds clamp relaxed; tapered blend, window
   rule = largest no-veto window → 0.20/0.30·VDD_train; full corridor
   0.30/0.40 KILLED on a TSMC5-opamp veto break — dead-end recorded). Gains:
   SC TSMC12 FLIPPED (4.13 %), TSMC5 opamp de-fragilized 9.78 → 2.49 %,
   TSMC7 opamp resurrected flat → **10.16 % (0.16 pp from its gate)**, RO
   improved on all 4 techs (TSMC7 8.98 → 8.28 %), inverter tran improved.
   force_ic still 0/8 → P3 carries closure (window-trade caveat recorded).

**Decision-table (§2) resolutions:** P0 flipped opamp/SC cells → P4 census
shrinks to TSMC7 (0.16 pp!) + TSMC16 (flat); R0 changed the SC census →
recounted (honest 8/16 start); symcaps row → killed; SC frame-owned row →
no (harness + clamp + forward-id); P1 → exonerated via L72 control ⇒ **P5
funded** (re-scoped: id surface along trajectories — charges are exact);
P2 garbage row → no (usable, shipped); P0+P2 ≥6/8 row → no (0/8) ⇒ **P3
stays a full ship-required arm**, additionally targeting the TSMC16 SC hold
leak (~6.4 mV — same subthreshold class) and the S7 window-trade re-test.

**Re-shaped campaign (weeks 2–4, serial as before; rev 3 inserts S9b):**
S9 SWA/EMA → **S9b data-regen v2 + loader-filter removal + `sample_class`
plumbing + derivative-fidelity scorer metric + control-v2 retrain (rev 3,
user rulings 3+5)** → S10 P4 (lead; elevated to core ∂id/∂V requirement;
TSMC7 opamp needs −0.2 pp, TSMC16 needs un-collapsing; veto set = all 11
passing cells) → S11 P3 (unfiltered small-current loss: decade-balanced
auxiliary + ceiling hinge; force_ic qb-prop + TSMC16 SC leak + re-test
the 0.10 taper window under P3) → S12 P5 (id-corridors: TSMC7 RO 8.28 %,
TSMC5 SC over-conduction 12.14 %) → S13 P8a (re-armed id-VALUE supervision)
→ S14 P6 → S15 P7 → S16 P8b (premise weakened by the P0-I retraction; keep
as fallback only) → S17 P9 → S18 composition → S19 promotion (blind
holdouts now include the off-default-Vin SC variant).

## S12 outcomes (2026-06-15 — P5 trajectory-corridor arm; verdict KEEP, headline 11→14/16)

Full detail: `results/v6_4_7/S12_P5_corridor_gate.md`. Built the corridor
harvest/append/train pipeline (`scripts/v6_4_7_s12_{harvest,append}_corridors.py`,
`scripts/v6_4_7_s12_train_corridor.sh`, `traj_corridor` = SAMPLE_CLASS_CODES
code 12). Harvested the per-device bias **tubes** the benchmark devices visit
along the **ground-truth** trajectories (RO+SC via native L72 == NGSPICE; opamp
+SRAM butterfly via NGSPICE directly — raw L72 DC diverges under PyCircuitSim
NR), OSDI-evaluated at the bench geometry, ±12 mV/20-sample jittered to ~1 % of
each dataset, appended to v2 → `{tech}_v2cor_{dev}.npz` (NMOS L=16n is off the
PDK grid ⇒ corridor rows labeled via a **pre-seeded label cache**, validated in
the live trainer path). Trained 4 seeds × 8 cells, control-v2 recipe +
`--class-weights traj_corridor=3`, A/B vs control-v2.

**Kill gate (first scored arm RO err < 7 %): PASSED — tsmc7 RO 8.28 → 2.87–2.92 %
(all 4 seeds), NEW-PASS.** The P5 thesis is confirmed: the RO period gap is the
~20 % NMOS dynamic-id deficit (P0-G/H), owned by the id VALUE surface along the
switching trajectory; teaching ground-truth id there closes it seed-invariantly.
tsmc5 RO recovered 5.80 (control-v2) → 4.6 % (PASS). **tsmc16: switchcap
13.1→2.01 % (all 4, but also flipped by control-v2's v2 data) and opamp
fail→5.06 % (s31 only, fragile 1/4).**

**Major cost — the corridor collapses *passing* opamps (tsmc5, tsmc12, all 4
seeds, 100 %)** — exactly the S10 value-surface/NR-fixed-point fragility
(reshaping the id surface destabilises the high-gain opamp). So the corridor is
**promoted per-tech, only where it nets a gain with no veto break:** tsmc7
(corridor: RO flip), tsmc16 (corridor s31: opamp+SC flip), tsmc5 + tsmc12
(**baseline** — corridor would regress their passing opamps). Net **11/16 →
14/16** (+3: tsmc7 RO, tsmc16 opamp, tsmc16 SC). butterfly 4/4 held (verified on
the corridor tsmc7/tsmc16 checkpoints; tsmc16 SNMerr 0.0 %, tsmc7 positive at
SNMerr 39.8 %), inverter held. **force_ic still 0/8 — NOT closed (S11/P3's
target); some seeds nudge the released cell rail-ward (tsmc7 s42 probe q=0.75).**
SC over-conduction on tsmc5 NOT fixed (12.16 %). **W-sweep (gentler dose to
preserve passing opamps) deferred** — would not change the tsmc7 headline.
**KEEP — surviving arm; `v6_4_7_s12cor_w3_*` are the S19 per-tech promotion
candidates (tsmc16 s31 opamp flip replication-gated). RESUME AT S11 = P3.**

## S10 outcomes (2026-06-14 — P4 Sobolev id-derivative arm; verdict KILL)

Full detail: `results/v6_4_7/S10_P4_sobolev_gate.md`. Built `SobolevIdLoss`
(id-channels only, asinh normalized-derivative space matching the deriv gate,
**uniform-negation sign verified** by FD/empirics — the 930c274 "gds no-flip"
is wrong, gds res 11× smaller under uniform negation). Warm-start fine-tune
screen reverts under plain val-MAE selection → replaced by **from-scratch
retrains at seed 17** (clean A/B vs control-v2 s17). **Result across λ∈{0.005…
0.3} and a 4-seed arm (config A λ=0.02):**

- **Deriv fidelity improves robustly** (gds_fwd 55.8→1.7 % monotonic; gm_fwd
  137→0.1 %; off-state 3–4 orders better) — ruling-4 core objective MET — and
  the **inverter improves** (VTC 0.96–2.36 vs 3.45) on every seed.
- **But the opamp collapses 4/4 seeds** (gain 180→0), including s7/s31 which
  control-v2 kept healthy (362/187) — **systematic, not seed-luck**; collapse
  is **λ-independent down to λ=0.005** (val-MAE identical to control).
- RO mixed (2/4 improved to 7.77/7.99 — best-ever tsmc7; 2/4 regressed).
  Side finding: 3/4 seeds move SRAM force_ic OUT of the metastable point toward
  the rail (off-state-deriv benefit; P3-adjacent, doesn't close).

**Verdict = KILL** (pre-registered S10 kill gate: opamp not < 15 % with inverter
held → drop the term). No Sobolev checkpoint promoted; `v6_4_7_s10{ft,sob,p4}_*`
inert. **Major finding: derivative fidelity is ANTI-correlated with the opamp.**
control-v2 has gm_fwd ~137 % yet gain within 10 % of NG; the Sobolev arm fixes
the Jacobian and collapses the gain — because the harness opamp gain (and the RO
period) are **value-surface / NR-fixed-point owned (the P0-C/P0-I class)**, NOT
autograd-Jacobian owned. **Consequence: ruling-4's premise is partially
falsified — precise ∂id/∂V does not help (actively harms) the value-owned
opamp/RO gates; the deriv-fidelity metric is an NR-robustness indicator, not a
circuit-accuracy promotion gate.** The opamp/RO levers must target the id VALUE
surface (P5 corridors, P3 subthreshold). `SobolevIdLoss` stays as default-off,
recoverable infra (pairs with the permanent deriv-fidelity scorer).

**User rulings post-S10 (2026-06-14, rev 4):** (6) **deriv-fidelity gate
DEMOTED to an NR-robustness metric only** — it is no longer a circuit-accuracy
promotion gate (S10 showed it anti-correlates with the value-owned opamp);
promote on the actual circuit gates (16 cells + force_ic + inverter). Ruling 4
is amended accordingly: ∂id/∂V precision is still *reported*, not *promoted on*.
(7) **REORDER — do S12 = P5 BEFORE S11 = P3.** The S10 finding most directly
implicates P5 (NGSPICE-trajectory id VALUE corridors) for the value-owned
opamp/RO gaps. **RESUME AT S12 = P5** (value-corridor data regen + retrain;
targets TSMC7 RO 8.28 %, tsmc5 RO, TSMC5 SC over-conduction, opamp/TSMC16);
P3 (S11, SRAM subthreshold) follows — carry the S10 SRAM-escape side finding.

## S9b outcomes (2026-06-14 — regen-v2 + control-v2 + gate; verdict PROCEED)

Full detail: `results/v6_4_7/S9b_controlv2_gate_summary.md`. S9b ran on a
**bare-checkout machine** — the entire runtime stack was rebuilt first (PyCMG
via proxy on `feat/v6`; OpenVAF 23.5 + OSDI; conda env + torch 2.6 CUDA;
NGSPICE 45.2 + OSDI from source; user-supplied TSMC PDKs). The lost-commit
S9b generator code was reconstructed on `feat/v6` (`NN_DC_SOLVE_TOL` floor
fix + `subvt_off` class; patch in `results/v6_4_7/s9b_pycmg_patch/`), and two
real bugs were fixed: a **parallel modelcard-cache write race** (degenerate
cards → labeller assert; fixed with atomic write) and **NFIN<2 inclusion**
(excluded per Rule 9).

- **Regen-v2 acceptance PASS:** 8 datasets; decade gate 8/8 (40k–200k
  rows/decade vs 1k); asinh `drift_id=1.0000` (no s_id pinning); labeller 0
  misses. gm/gds asinh drift 0.73–0.96 (P4-relevant).
- **control-v2:** 32 cells (4 seeds × 8), stock recipe, `--apply-filter off`,
  EMA, v2 data (1 CUDA-contention FAIL re-run clean).
- **Full multi-tech gate (per-cell best vs S8 baseline):** **2 regressions**
  — tsmc5 ring_osc (5.80% vs 2.61%) and tsmc12 opamp (all 4 seeds collapse) —
  **1 new-pass** (tsmc16 SC 13.1% → 3.17%), inverters hold on all 4 techs.
- **Go/no-go = PROCEED.** Both regressions are fresh-retrain variance, NOT
  data defects: EMA ruled out by ablation (RO-neutral 7.23≈7.21%); tsmc5 RO =
  lost best-of-8 cherry-pick (tsmc7 confirms — matches its non-cherry-picked
  8.28% baseline); tsmc12 opamp = the 44%-likely 4-seed spontaneous-collapse
  lottery (tsmc5 s7 passes at 0.79%). Data is sound (gates pass, inverter
  holds, tsmc16 SC win), so the data change is **not rewound**. **control-v2
  is now the fresh-retrain attribution baseline** for S10+ arms; the S8
  `baseline_v6_4_7_pre.json` stays the promotion gatekeeper; **tsmc5 ring_osc
  + tsmc12 opamp join the arms' recover-set** (P4 collapse-resistance; P5/P8a
  RO). Optional: +4 control-v2 seeds on tsmc12 to confirm the opamp lottery.
- **Harness/portability fixes shipped:** scorer/verify_* honor `NGSPICE_BIN`;
  gate driver runs GPU-serial (`--workers 1`; 1 scorer co-exists with
  training, >1 asserts) or `--cpu`; deriv-fidelity split out of the
  protected-gate check (computed separately for the P4 reference).

## Sequencing

**Protocol (applies to every step below):** execution is strictly serial — one lever at a time, no overlap. Each step starts from a clean committed state (`git commit` before touching anything), runs its verification gates, then resolves one of two ways before the next step begins: **(a) progress → commit**, becoming part of the baseline every later step builds on; **(b) kill criteria met → rewind** (`git reset`/revert the code; checkpoints stay on disk as inert artifacts that must not match the resolver pattern) **and record the dead end with the numbers that killed it**. No step starts while the previous one is unresolved. The proposal sections above stay in *rank* order (stable IDs); this section is the *execution* order.

1. **Week 1 (0 GPU), serial S1–S8 — ✅ ALL COMPLETE (2026-06-12), see "Week-1 outcomes" for results, gate files under `results/v6_4_7/`:**
   - **S1 — pre-flight:** snapshot checkpoints + `manifest.sha256`; **commit the still-untracked campaign infra** (`scripts/eval_v6_4_5_candidate.py`, `scripts/v6_4_5_search.py`) so every later rewind has a clean base.
   - **S2 = P0** frame fix + lifted-source sweep. Gates: 16-cell harness + force_ic 8 + inverter 8/8 + DC 55/55 + tran 64/64 + the new sweep. Correctness fix — ships regardless; the only stop is the lifted-source sweep regressing vs OSDI (investigate before proceeding, per P0 kill note).
   - **S3 = R0.1** droop sub-gate repair, under the E3-class adversarial false-PASS review. Shown to be a loosening rather than a correction → rewind + dead-end record. Pass → recount the SC failing-gate census.
   - **S4 = R0.2** symcaps SC-only re-test on post-P0 code (env-gated probe, zero code — nothing to rewind; record the result; per-circuit env-gated shipping is decided at S19, not here).
   - **S5 = R0.3** SC per-device id/charge dump at the sample window + phi falling edge (diagnostic artifact only; settles the SC ownership split frame/clamp/charge/symcaps).
   - **S5b — `.ic`/uic semantics fix (AMENDMENT 2026-06-11, recorded per the mid-campaign-change rule):** S5 found the dominant SC owner is a harness gap — NGSPICE runs `tran uic` from `.ic` while `run_directnet_transient` uses `.ic` only as an OP guess and starts from the NN leakage equilibrium (vsamp(0)=0.39–0.70 V vs 0). Fix: solve the OP with `.ic` nodes pinned (constrained solution, NOT the force_ic released re-solve) and start the transient from it. Measurement-fix class ⇒ E3 adversarial review; re-run SC AND RO (shared runner) with blind vetoes on the three passing RO cells; recount the SC census.
   - **S6 = P1** swap matrix on post-P0 code (diagnostic script; both outcomes decisive — ~46.6–47 ps funds S12, ~92 ps pauses **all** model-side RO levers (S12 RO claim, S13, S16); see §2).
   - **S7 = P2** — raw-reverse probe **first** (~20 LOC); garbage surface → record dead end, **skip the build** (SRAM rides S2+S11+S17). Else build the clamp relaxation; canaries fail → soft-blend variant; that fails too → rewind + dead-end record.
   - **S8 — re-freeze:** scorer extension (`gain_err` + **switchcap cells** — the scorer currently has none, and `baseline_v6_4_4.json` contains no SC cells and no opamp TSMC7/16) → **re-freeze `baseline_v6_4_7_pre.json` over all 16 cells + force_ic on exactly the code that survived S2–S7** → commit. Then apply §2 to re-shape the campaign before any GPU is spent.
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

3. **Weeks 2–4 (GPU campaign, ~250–300 GPU-h per user ruling), serial S9–S18 — one arm at a time, each fully scored and committed-or-rewound before the next starts.** Every arm runs **≥4 seeds per config** and A/Bs at equal data against a frozen control. **Control amendment (rev 3, recorded):** because ruling 3+5 change the *data itself* (regen v2 + unfiltered loader), the attribution control becomes **control-v2** — the stock V6.4.4 recipe retrained at S9b on the new data with the filter off, ≥4 seeds — so every arm's delta isolates its *recipe*, not the shared data change; **the S8-frozen `baseline_v6_4_7_pre.json` remains the promotion gatekeeper** (no protected gate may regress vs S8, and control-v2 itself must clear that bar before any arm trains — if control-v2 regresses protected gates, the data change is investigated/rewound first). The control is *not* advanced mid-campaign even when an arm survives (a surviving arm's recipe is committed, but attribution stays against control-v2; composition happens once, at S18). **Multi-seed training fans out across ALL GPUs on this machine (user ruling 2026-06-10)** — serialization is at the *arm* level, not the seed level; one seed per GPU concurrently. Hold a **~20 % budget reserve** for scoring wall-clock (V6.4.5's 32-candidate scoring was a full phase; this campaign is larger with a longer vector).
   - **S9 — SWA/EMA flag** (pulled forward from P6, ~20 LOC, `torch.optim.swa_utils`): must land and be committed **before the first arm**, since it is a default flag on every arm. **✅ SHIPPED 2026-06-12:** `--swa-mode {none,ema,swa}` + `--ema-decay` in `bsimar/{training/trainer,cli/train}.py`; EMA = per-step `AveragedModel` update (`get_ema_multi_avg_fn`), SWA = equal-weight from 75 % of max_epochs with best-val reset at the switchover; val selection and the saved checkpoint use the averaged weights when active, saved from `avg_model.module` ⇒ on-disk key format unchanged (verified: smoke `none` vs `ema` key lists identical, no `module.`/`n_averaged` contamination). Default `none` is behavior-preserving. Gates: short smoke trains (none + ema, GPU 1) PASS.
   - **S9b — data-regen v2 + loader-filter removal + control-v2 (NEW, rev 3, user rulings 3+5):** (1) generator floor fix + Vgs∈[VTH−0.1, VTH+0.15] densification (P3 Stage 2, pulled forward so EVERY arm trains on it); acceptance gate: every id decade in 1e-12–1e-6 A holds ≥1 k rows per cell; (2) regenerate all 8 per-tech datasets; (3) expose `apply_filter` and turn it off — all small-current rows retained; (4) `sample_class` plumbing + LDS-product renormalization (pulled forward from P5 — P4 corridor weighting and P3 masks need it); (5) asinh-scale audit per P3 rev-3 (vii); (6) extend the scorer with the derivative-fidelity metric (autograd gm/gds/gmb NRMSE vs held-out OSDI, full split + corridors); (7) **retrain control-v2** (stock recipe, new data, filter off, SWA/EMA per S9, ≥4 seeds) and gate it against `baseline_v6_4_7_pre.json` — protected gates must hold; the inverter VTC is the canary for the s_id-drift risk. Control-v2 regresses protected gates → investigate the data change (s_id pinning first) before any arm starts. **✅ COMPLETE 2026-06-14 (verdict PROCEED):** regen-v2 acceptance PASS (decade 8/8, asinh drift_id=1.0, labeller 0 misses); control-v2 32 cells trained; full multi-tech gate = 2 regressions (tsmc5 RO 5.80 vs 2.61; tsmc12 opamp all-4-collapse) + 1 new-pass (tsmc16 SC 13.1→3.17) + inverters hold all 4 techs. Investigation: EMA ruled out (ablation RO-neutral), tsmc5 RO = lost best-of-8 cherry-pick (tsmc7 confirms), tsmc12 opamp = 44%-likely 4-seed collapse lottery (tsmc5 s7 passes) ⇒ **data sound, not rewound**; control-v2 = fresh-retrain attribution baseline; tsmc5 RO + tsmc12 opamp join the recover-set. See "S9b outcomes" § + `results/v6_4_7/S9b_controlv2_gate_summary.md`. NOTE: ran on a bare-checkout machine; full env rebuilt (PyCMG/OSDI/torch/ngspice-45.2), lost generator code reconstructed (patch in `results/v6_4_7/s9b_pycmg_patch/`), 2 bugs fixed (cache-write race, NFIN<2).
   - **S10 = P4 — ✅ COMPLETE 2026-06-14, verdict KILL.** Ran the fine-tune λ-screen (reverts under val-MAE selection), then from-scratch seed-17 A/B + a 4-seed arm. Deriv fidelity improves robustly (gds_fwd 55.8→1.7 %) and the inverter improves, but the opamp collapses 4/4 seeds (systematic, λ-independent) ⇒ pre-registered kill gate triggered, Sobolev term dropped, no model promoted. Major finding: deriv fidelity is anti-correlated with the value-owned opamp (P0-C/P0-I class). **Post-S10 rev-4 rulings:** (6) deriv gate demoted to NR-robustness only; (7) **REORDER — do S12 (P5) before S11 (P3).** `SobolevIdLoss` kept as default-off recoverable infra. See "S10 outcomes" § + `results/v6_4_7/S10_P4_sobolev_gate.md`.
   - **S12 = P5 (lead arm per rev-4 reorder) — ✅ COMPLETE 2026-06-15, verdict KEEP.** Harvested the id-VALUE bias tubes from the native-L72 (RO+SC) and NGSPICE (opamp+SRAM, raw L72 DC diverges under PyCircuitSim NR) ground-truth trajectories, OSDI-evaluated at the bench geometry, ±12 mV/20-sample jittered to ~1 %, appended to v2 as `traj_corridor` (code 12, pre-seeded label cache for the off-grid 16n geometry), retrained 4 seeds × 8 cells with `--class-weights traj_corridor=3`. **Kill gate PASSED: tsmc7 RO 8.28 → 2.9 % (all 4 seeds, NEW-PASS)** — confirms the P5 thesis (RO = id-value-surface deficit along the trajectory). Per-tech mix (corridor where it nets a gain, baseline where it would regress a passing opamp) → **11/16 → 14/16** (tsmc7 RO; tsmc16 opamp s31 + SC). **Cost: collapses *passing* opamps (tsmc5/tsmc12, all 4) — the S10 value-surface fragility — avoided by the per-tech mix.** SC-tsmc5 NOT fixed; force_ic still 0/8 (S11). butterfly 4/4 + inverter held. See "S12 outcomes" + `results/v6_4_7/S12_P5_corridor_gate.md`. Build plan `results/v6_4_7/S12_P5_build_plan.md`.
   - **S11 = P3 (ship-required SRAM) — after S12.** force_ic still 0/8 (S2+S7 did not lift it). Stage 1 loss (decade-balanced auxiliary + ceiling hinge + trust-floor masks, on the S9b unfiltered data — Stage 2 regen already landed at S9b); re-run the P2 A/B after every P3 retrain. **Carry the S10 SRAM-escape side finding** (the off-state deriv improvement moved force_ic out of the metastable point toward the rail on 3/4 seeds ⇒ the subthreshold VALUE surface is the SRAM lever). Kill (weak-inversion ratio not ≥10× with VTC ≤5 %) → rewind, escalate to S17.
   - **S13 = P8a rider** — immediately after S12 (same harvested corridors, adds per-point value supervision). Skipped if S6 paused model-side RO levers (it did not — simulator exonerated).
   - **S14 = P6 distillation:** TSMC7 student from the 32-checkpoint seed bank (0-GPU teachers); other techs distilled from each *surviving* arm's 4 seeds — hence after S10–S13. Kill (not Pareto-≥ the best single seed on the scorer vector) → rewind.
   - **S15 = P7 split-head:** run only against the scorer-selected surviving recipe. Kill (inverter regression beyond the documented ±1 % run-to-run scatter) → rewind.
   - **S16 = P8b** — funded only if S12 failed **and** S6 exonerated the simulator; gated by held-out circuits.
   - **S17 = P9 fallback** — only after S2+S7+S11 have all failed to close force_ic; stays behind the recorded go/no-go fit gate (≥4-decade suppression AND ≤5 % inv_trip simultaneously).
   - **S18 — Stage B composition:** compose the top-2 compatible *surviving* winners (P3 loss + P4 loss stack; P5 data composes with either) and re-score — composition is budgeted, not assumed. Not Pareto-≥ the best single arm → rewind to the best single arm.
4. **Week 5, S19 — promotion** per the V6.4.6 protocol (baseline JSON, blind holdouts, per-tech mix allowed) + the rev-2 selection discipline below; the R0.2 symcaps per-circuit env-gated shipping decision is made here; CHANGELOG/CLAUDE.md; a dead-end record for **every rewound step**, with the numbers that killed it.

## Verification

- P0/R0/P2: full complex harness (`tests/verify_complex_{ring_osc,opamp,sram_snm,switchcap}.py`) + force_ic probe + inverter gate (`verify_nn_dc_tran.py --inverter-only`) + extended harness (55/55, 64/64) + **the new lifted-source DC sweep**, `OMP_NUM_THREADS=1`.
- Retrain arms: multi-circuit scorer vector per candidate (Rule-16 quartet per cell); **blind vetoes extended to ALL currently-passing cells** — the rev-1 list omitted TSMC5 ring_osc (2.98 % PASS) and SC TSMC7; the re-frozen baseline covers all 16 cells + force_ic; snapshot checkpoints + `manifest.sha256` before any campaign. **Rev 3: the scorer vector additionally carries (a) derivative fidelity — autograd gm/gds/gmb NRMSE vs held-out OSDI columns, full test split + harvested corridors (P0-B baseline gds 20–23 %) — and (b) an off-state probe — predicted |id| at Vgs=0/Vds=VDD biases vs OSDI (the Mpl-class hard-OFF error, NN +0.50 µA vs ~0) — so rulings 3/4 are measured, not assumed.**
- **Selection discipline (rev 2):**
  1. **Replicate the top-3 candidates 3×** (different thread counts) before promotion; require winner − runner-up > replicate σ — min-over-N selection across 40+ candidates inflates the winner's apparent margin.
  2. **Held-out perturbed circuit variants** (different RO stage count, opamp Cload, SRAM cell ratio — cheap netlist edits) are never scored during selection and run **once, blind, at promotion** — the only test that a winner generalizes rather than memorizing the 16 gate configs.
  3. **The promotion rule is pre-registered here:** promote the per-tech best on the scorer vector subject to all vetoes; per-tech mix allowed with ≥4-seed evidence per promoted cell; any mid-campaign scorer change requires a recorded amendment (the V6.4.5 flat-flag re-calibration precedent).
- **Success =** headline **>11/16** (net-new flips beyond the S8 state) AND **SRAM force_ic improved (target 8/8)**, with promoted checkpoints trained on the unfiltered regen-v2 data (rulings 3/5), no protected gate regressed, and every killed arm/term recorded with the numbers that killed it. **Rev-4 amendment (ruling 6, post-S10):** derivative fidelity is **demoted from a promotion criterion to an NR-robustness metric** (reported, not promoted on) — S10 showed it anti-correlates with the value-owned opamp/RO gates, so promotion is on the circuit gates (16 cells + force_ic + inverter), not on autograd gm/gds/gmb NRMSE. The rev-3 "deriv fidelity strictly better than control-v2" clause is RETIRED.
