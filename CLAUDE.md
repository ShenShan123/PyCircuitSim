# Project: PyCircuitSim

## Overview

Python-based SPICE-like circuit simulator emphasizing educational clarity and modular architecture.
**Primary Goal:** specific support for three compact model families:

- **BSIM-CMG** (LEVEL=72) — PyCMG-wrapped OSDI FinFET model (ground truth).
- **DirectNet** (LEVEL=73) — baseline feed-forward MLP compact model (PyTorch).
- **BSIM-AR Transformer** (LEVEL=74) — autoregressive Transformer compact model (PyTorch).

DirectNet and BSIM-AR share the same data, normalization, and evaluation pipelines via the unified `bsimar` package at `external_compact_models/bsimar/`. DirectNet is the baseline for comparison against BSIM-AR.

Must support **Operating Point**, **DC Sweep**, and **Transient Analysis** for all model types.

**Core Principles:** pure Python; Solver ↔ Device Models decoupled; production-grade compact models via PyCMG/OSDI; basic HSPICE netlist compatibility.

## Architecture

### Module Structure

```
pycircuitsim/
├── config.py           # Path configuration (OSDI binary, modelcards)
├── simulation.py       # Orchestration (run_simulation, run_dc_sweep, run_transient)
├── parser.py           # Two-pass netlist parsing, .model directive support
├── circuit.py          # Circuit topology
├── solver.py           # MNA matrix + Newton-Raphson
├── logger.py           # HSPICE-like .lis output
├── visualizer.py       # Matplotlib plotting
└── models/
    ├── base.py               # Component abstract base
    ├── passive.py            # R, C, V, I sources (PULSE)
    ├── mosfet_cmg.py         # BSIM-CMG (LEVEL=72) via PyCMG
    ├── mosfet_nn.py          # Shared _MOSFETNNBase (LEVEL=73/74) — voltage prep, autograd, Vds correction
    ├── mosfet_directnet.py   # DirectNet (LEVEL=73, primary)
    └── mosfet_bsimar.py      # BSIMAR Transformer (LEVEL=74, parked — see Rule 18)

external_compact_models/
├── bsimar/             # Unified NN compact model package (importable as `bsimar`)
│   ├── config.py                   # NNTechConfig + TECH_CODE_MAP + local-vocab helpers
│   ├── data/{normalize,dataset,analyze}.py
│   ├── models/{direct_net,transformer}.py    # nn.Embedding tech-code
│   ├── losses/bni_mae.py           # MAELoss + per-target LDS weights
│   ├── training/trainer.py
│   ├── eval/{metrics,visualization}.py
│   ├── cli/train.py                # `python -m bsimar.cli.train --model direct ...`
│   └── checkpoints/                # *.pt + _norm.npz (gitignored)
└── PyCMG/              # BSIM-CMG OSDI wrapper (git submodule)
    ├── pycmg/{core,model,parser,osdi_types,tech}.py
    ├── build/osdi/bsimcmg.osdi
    └── modelcards/     # ASAP7/*.pm committed; TSMC{5,7,12,16}/cln*.l gitignored (IP)

main.py                 # CLI entry point
examples/*.sp           # Example netlists
results/                # Simulation output
tests/
├── common/             # Shared test infra
│   ├── base.py         # PROJECT_ROOT, OSDI_PATH, TechProfile, ALL_TECHS, NGSPICE runner
│   ├── bsimcmg_{dc,tran}.py
│   └── nn.py           # nrmse, mre, checkpoint resolution, sys.path bootstrap
├── references/         # NGSPICE reference netlists
└── verify_*.py         # 3-level DC/transient tests + NN verification
```

### Key Algorithms

* **MNA** — Sparse construction (scipy.sparse lil_matrix → CSR + spsolve).
* **Newton-Raphson** — SPICE-standard convergence (RELTOL + VNTOL).
* **BE → Trap → BDF-2 integration** — Backward Euler step 1, Trapezoidal default, BDF-2 auto on stiffness.
* **Source + GMIN stepping** — homotopy; GMIN stepping opt-in for bistable.
* **LTE sub-stepping** — adaptive internal sub-steps (opt-in via `max_substeps`).
* **Bistable convergence** — DC oscillation detection, adaptive damping, hard `.ic` mode.

## Supported Features

* **Devices:** R, C; NMOS/PMOS LEVEL=72 (BSIM-CMG, ground truth), LEVEL=73 (DirectNet, primary NN), LEVEL=74 (BSIMAR, parked); DC voltage/current sources, PULSE.
* **Analyses:** `.op`, `.dc`, `.tran`.
* **Directives:** `.model` (LEVEL=72/73/74), `.include`, `.ic`.
* Legacy LEVEL=1 (Shichman-Hodges) removed.

## Validation

Inverter circuit must PASS Transient Analysis against NGSPICE ground truth within reasonable numerical tolerance. Never use simplified/self-defined equations as reference.

## Status

Current shipping revision is **V6.4.4** (V6.4.2 solver + per-tech checkpoint mix from V6.4.2 Phase-7a stock winners on TSMC5, V6.4.1 seed-42 on TSMC7/12/16; inference-only iteration, 2026-05-28). V4 → V6.4.2 history in `docs/CHANGELOG.md`; iter-1 plan in `docs/plans/2026-05-15-directnet-complex-circuits.md`, iter-2 plan in `docs/plans/2026-05-24-directnet-v6.4.4-complex-circuits-iter2.md`, final report in `results/v6_4_4_iter2/V6_4_4_final.md`. **V6.4.5 (2026-05-29) was a no-ship Track-A iteration that ran all five phases.** First pass recorded four dead ends (`NN_SYMMETRIC_CAPS=1`, `RO_MAX_SUBSTEPS=4`, SRAM butterfly warm-start, the plan's Rule-15 `Ioff_rail` patch); the second pass built the Phase-3 multi-circuit scorer (`scripts/eval_v6_4_5_candidate.py` + `scripts/v6_4_5_search.py`, infra only) and ran the full Phase-5 TSMC7 retrain (16-seed stock + 8-seed mono, 32 trainings) — a **fifth dead end**: no candidate closes ring_osc TSMC7 ≤5 % (best feasible 9.05 % > baseline 8.98 %), confirming the ~9 % RO error is a systematic model bias, not seed/recipe-addressable. No model shipped; V6.4.4 stays canonical. Plan in `docs/plans/2026-05-28-directnet-v6.4.5-ro-sram.md`, final report in `results/v6_4_5/V6_4_5_final.md`. **V6.4.6 (2026-06-01/02) is another no-behavioral-change iteration — diagnosis-first, 9/16 held.** Phase 0 ran six 0-GPU diagnostics that **killed the RO Jacobian-distillation lever before any GPU** (P0-C: the divergent gds/caps are Jacobian-only and cancel at the NR fixed point, so they are causally inert on the RO period — owned instead by the id-value/charge-value/BDF-2 integration) and unlocked an SRAM solver path (P0-A: a railed fixed point exists). Phase 1 then found the SRAM path is a **measurement fix, not a gate-close**: the `force_ic` early-return left `_last_solve_converged` stale (guaranteed `0/8` regardless of NR), and once hardened the released cell still lands in the inboard attractor on all 4 techs (genuine `0/8`); the plan's constraint-continuation homotopy was built and **killed** (the railed point is NR-unstable). The only code shipped is the corrected SRAM probe (`solver.py` KCL-residual telemetry + honest flag; `verify_complex_sram_snm.py` `resid_ok AND rail_ok`, rail band tightened `VDD/4` → `0.1·VDD`). No model, no checkpoint changed. Plan in `docs/plans/2026-06-01-directnet-v6.4.6-ro-sram.md`, gate files under `results/v6_4_6/`.

> **Branch `feat/v6.4.1`, V6.4.4 (2026-05-28):** inference-only iteration; no retraining, no data regen. Selected the best per-tech checkpoint from artifacts already on disk. TSMC5 NMOS=`v6_4_2_p7_tsmc5_stock_s17`, TSMC5 PMOS=`v6_4_2_p7_tsmc5_stock_s42` (V6.4.2 Phase-7a winners, sha256 `22eef03e…` / `a6a09be0…`); TSMC7/12/16 stay on V6.4.1 seed-42. Complex-circuit pass rate **7/16 → 9/16** vs V6.4.1: TSMC5 ring_osc 6.76 % → 2.98 % (PASS) and TSMC5 opamp 14.78 % → 2.64 % (PASS). TSMC7 Phase-7a was tried and reverted — its better inverter VTC (174 → 100 mV) collapsed the Step-3b Miller opamp to flat-Vout (gain error 30.67 % → 100 %), proving that **inverter-VTC selection alone cannot drive complex-circuit pass rate**. Inverter gate 8/8 held (VTC NRMSE 1.21/2.37/2.05/1.33 %, tran 1.62/1.09/1.41/1.45 %); extended harness unchanged.

> **V6.4 best-of-N artifacts are GONE.** The `/tmp/v6_4_checkpoints_backup_20260517/` directory was cleared by `/tmp` cleanup between V6.4.2 ship and the V6.4.4 iteration; the `v6_4_bof_*` / `v6_4_repro_*` source stems survive nowhere on disk. The V6.4 inverter VTC numbers in earlier CHANGELOG entries (TSMC5 62, TSMC7 60, TSMC12 32, TSMC16 30 mV) are no longer reproducible from on-disk artifacts. The V6.4.1 seed-42 backup at `/tmp/seed42_backup_20260524/` (manifest.sha256 included) is the only rollback target. Remaining opamp/SC gates are now model-fidelity gaps that need a retrain (Phase 8 split heads or a Phase-7 best-of-N re-scored on opamp gain + RO period — both deferred).

> **V6.4.5 Track-A dead ends (no-ship, 2026-05-29).** Four levers tried against TSMC7 ring_osc (8.97 %) and SRAM `force_ic` (0/8), all reverted: (1) `NN_SYMMETRIC_CAPS=1` on RO + SC — TSMC7 RO period err bit-for-bit unchanged (not cap-asymmetry). (2) `RO_MAX_SUBSTEPS=4` on RO — TSMC7 8.97 → 8.04 % at 2× wall time (not LTE). (3) SRAM butterfly warm-start (`near_zero` ≈ 83–123 mV instead of literal 0) — all four techs still settle at q ≈ 0.70–0.80 → q ≈ 0.18 is a **true NN attractor**, not a poor warm start. (4) Plan's Rule-15 `Ioff_rail` patch `Ioff_rail = max(|id_raw|, k·NFIN·1nA)` — doubles the conducting current at the rail, collapsing inverter VTC NRMSE 1.21 → 11.56 % at the smallest non-zero k = 1 (SRAM attractor also moved *away* from rails). A corrected formula `Ioff_extra = max(floor − |id_raw|, 0)` was NOT retried (plan-wording discipline). **Phase 5 (TSMC7 16-seed stock + 8-seed mono, 32 fresh trainings) was then run and also died:** scored under the new Phase-3 multi-circuit vector, no candidate reaches RO ≤5 % — best overall 8.21 % (`stock_s31`, opamp-collapsed/infeasible), best feasible 9.05 % (`stock_s11`) > baseline 8.98 %; the seed moves inverter VTC (1.75–5.50 %) but not the RO period (DN ~50.8–53 ps vs NG 46.64), and 13/16 new candidates collapse the TSMC7 opamp to gain 0. The Phase-3 scorer's `opamp_flat_flag` was re-calibrated from vout-at-center (flagged even the passing TSMC5 opamp) to `gain<10`. The 32 `v6_4_5_p5_tsmc7_*` checkpoints are inert dead-end artifacts (don't match the resolver pattern). Closing TSMC7 RO now requires an **architectural** change (V6.4.6 split-head, or Track B B5/B6/B7), not a retrain. See CHANGELOG "V6.4.5".

> **V6.4.6 dead ends (no behavioral change, 2026-06-01/02).** Three levers recorded, all on canonical V6.4.4 (no checkpoint touched): **(E1) Phase-2 RO Jacobian-distillation — KILLED before any GPU by P0-C.** Swapping the *exact* OSDI gds/caps into the live TSMC7 RO transient moves the period **≤0.01 ps** (baseline 50.82 / cap-swap 50.82 bit-for-bit / gds-swap 50.83 ps; NG 46.64). gds & cap-derivatives are **Jacobian-only** — they enter only the NR Jacobian + matching RHS offset and cancel at the converged fixed point (`_stamp_mosfet_dc:304`, `_stamp_mosfet_transient:1718-1782`). P0-B's divergent gds (NRMSE 20–23 %) is real but *causally inert* on the period; the RO walk is owned by the **id-value + charge-value (qg/qd) trajectories + BE/Trap/BDF-2 truncation** → deferred to a scoped V6.4.7 dynamic-id/qs/BDF-2 investigation. The deferred split-head Softplus cap-head is therefore unnecessary for RO. **(E2) Phase-1 `force_ic` constraint-continuation homotopy (Norton soft-pin g:1→0, + series-R dual) — KILLED.** P0-A proved a railed *residual* fixed point exists (KCL residual 8.5e-5/1.26e-4 ≪ thr, ratio 0.013–0.017) and that the cell is **bistable**, but the continuation folds at **g*≈1e-5 S** into the symmetric metastable point on all 4 techs (0/8): the railed point is **NR-unstable** under the re-stamp map `x→A(x)⁻¹b(x)` — the OFF node's vanishing deep-subthreshold conductance makes `Δqb=residual/g_qb` explode as the soft-pin conductance vanishes. A *stronger* negative than the plan's conjunction foresaw (the railed point exists yet cannot be tracked). Reverted. **(E3) The `VDD/4` rail-proximity band (intermediate "4/8 → 11/16") — RETRACTED as a false-PASS.** Fixing the stale `_last_solve_converged` flag let the pre-existing `VDD/4` band mark TSMC12/16 PASS, but the released cell lands in the *documented inboard failure attractor* (q≈0.87/qb≈0.20) on all 4 techs — storage-"0" at 24–30 % VDD ≈ 1× the true SNM above ground. `VDD/4`=25 % VDD straddles the attractor, so the 0.80 V techs' qb≈0.19 sneaks inside by VDD-scaling while TSMC5 rails *closer in absolute volts* (qb=0.163) yet "failed". The released solution is **byte-identical to V6.4.4** (matching checkpoint sha256s) ⇒ a zero-delta measurement correction, not an improvement. Band tightened to **`0.1·VDD`** → honest **`0/8`** (confirms D3: the inboard point is a true NN attractor co-existing with an NR-unstable railed point). **Net: the only code shipped is the corrected SRAM `force_ic` probe** (stale-flag fix + KCL-residual gate, `solver.py` + `verify_complex_sram_snm.py`); inverter 8/8 byte-identical, butterfly 4/4. SRAM `force_ic` remains a model-fidelity gap; RO and SRAM both deferred to V6.4.7. **Post-ship RO diagnostics P0-G + P0-H (2026-06-02) localised the TSMC7 RO gap to the NMOS dynamic `id` VALUE** (~20 % peak pull-down under-prediction): P0-G drove integration truncation→0 (Trap & BE both converge to a ~50.4 ps continuum limit, ~8 % > NG; the BDF-2 switch never fires) leaving only ~0.4 ps to integration, and P0-H found the charge VALUES (qg/qd/qs) are *exact* (≤2 aC) — so RO is owned by neither the gds/cap Jacobian (P0-C), nor charges, nor integration, but the **id VALUE**. **Post-ship P0-I (2026-06-03) then ran the causal id-injection swap and re-scoped that lever: the id-VALUE is NOT separable from the charge model. [RETRACTED at V6.4.7 S6, 2026-06-11 — the ~92 ps was an artifact of the injection id-path mapping; the native LEVEL=72 control reproduces NGSPICE at ratio 1.000 and exact charges+caps injection changes nothing; see `results/v6_4_7/S6_P1_swap_matrix.md`. id-only levers re-armed.]** Injecting the *exact* OSDI `id` into the live RO (NMOS-only AND symmetric N+P) produces a genuine, full-rail, uniform **~92 ps** oscillation (baseline 50.83 / N+P 92.30 / NMOS 92.74 ps) — ~2× baseline and *further* from NG 46.64, the *opposite* direction from swapping id+charge together (NGSPICE 46.64). Unlike the Jacobian (P0-C: inert/separable, ≤0.01 ps), swapping `id` alone yields an inconsistent hybrid whose period is dominated by the id↔charge mismatch ⇒ the RO period is a **joint (id, charge)** property. So the planned **frozen-base LoRA id-VALUE-only distillation is NO LONGER de-risked** — V6.4.7 must gate any id-only fix on the live RO period and consider a **joint id+charge correction (or retrain)**. (Caveat: the injection bypasses Rule-15 + floors gds, so 92 ps is a proxy warning, not proof a real autograd-consistent LoRA fails.) P0-I was numerically hard: the naive swap diverged (inconsistent-Jacobian artifact), rebuilt as a consistent exact-bias OSDI op-point (`v6_4_6_p0i_id_injection_v2.py`) that converges but is ~20–35× slow. Gate files `results/v6_4_6/phase0{G,H,I}_*.md`. See CHANGELOG "V6.4.6".

> **V6.4.7 plan (2026-06-10, rev 2 after four-agent adversarial review; rev 2.1 same day serialized the sequencing):** `docs/plans/2026-06-10-directnet-v6.4.7-accuracy.md`. Top lever is **P0 — the NMOS source-frame bug**: `_raw_voltages` (`pycircuitsim/models/mosfet_nn.py:232-243`) source-shifts only PMOS, so lifted-source NMOS (opamp `vtail` diff pair, switchcap pass device, SRAM access transistors) is evaluated at phantom Vgs/Vds with Vbs=0 against a Vs≡0-trained network (NN Rule 2 encodes the same blind spot). Rev 2 also retracted the switchcap droop premise (charge-transfer is the real failure; the droop sub-gate demands ~VNTOL agreement — measurement artifact, repaired in R0), and recorded two user rulings: SRAM `force_ic` is ship-required; full-arm campaign at ~250–300 GPU-h, ≥4 seeds per config. Rev 2.1 re-ordered execution into a strict serial chain S1–S19 (commit-or-rewind before the next lever starts; SWA/EMA infra pulled ahead of all GPU arms; every arm A/Bs against the S8-frozen control). **S1 pre-flight done:** checkpoints snapshotted to `/data2/home/shenshan/checkpoint_snapshots/v6_4_7_pre_20260610/` (161 files, manifest mirrored at `results/v6_4_7/checkpoints_pre_manifest.sha256`); V6.4.5 campaign infra committed. **S2 = P0 SHIPPED (2026-06-10): headline 9/16 → 10/16.** The frame fix flipped TSMC12 opamp PASS (10.94 → 5.21 %); lifted-source canary 12/12 (pre-fix 10–64 % NRMSE); inverter 8/8 / DC 55/55 / tran 64/64 / RO bit-identical / butterfly 4/4 all held; force_ic still 0/8 but the inboard attractor halved its distance to the rails (qb → 0.104–0.117 V, ~25 mV outside the 0.1·VDD band on TSMC12/16). TSMC5 opamp moved 2.64 → 9.78 % (still PASS; pre-arbitrated). Switchcap unchanged ⇒ frame was not the SC owner. Gate file `results/v6_4_7/S2_P0_frame_fix.md`. **S3 = R0.1 SHIPPED (2026-06-10): switchcap droop sub-gate repaired** — `|dn−ng| ≤ max(10 %·|ng|, 1e-3·VDD)` replaces the pure-relative gate that was simultaneously unpassable (demanded 19–150 nV ≈ sub-VNTOL agreement when NG droop sat at sub-µV) and auto-passing (nan-guard waved TSMC7's 2.208 mV droop — the worst cell — through). Adversarial review (E3-class): **CORRECTION, net tightening** — the floor matches the solvers' two-point tolerance 2·(RELTOL·V_hold+VNTOL) within ±20 %; column renamed `Droop%alw` (% of allowance). **Headline restated honestly: V6.4.4 canonical = 8/16 under the repaired gate (TSMC7 SC pass was a nan-guard artifact); current post-P0 state = 9/16.** Caveat recorded: the 0.65–0.80 mV floor is a circuit-level gate that admits ~50 nA off-state leakage error — subthreshold fidelity belongs to P3/P4. Gate file `results/v6_4_7/S3_R01_droop_gate_repair.md`. **S4 = R0.2 resolved (2026-06-10): symcaps KILLED for SC too** — on post-P0 code `NN_SYMMETRIC_CAPS=1` still improves SC charge (TSMC5 14.65→3.68 %, TSMC16 13.14→1.38 % — would pass) but explodes hold droop to 30–137 mV genuine drift (40–170× allowance; invisible under the old auto-pass gate, caught by the S3 repair). Per-circuit env-gated shipping is off the table; D1 now dead for RO AND SC. Ownership evidence: SC sample-phase charge error is substantially cap/charge-model-owned (not id-owned), and the asymmetric trans-caps are load-bearing during hold. Gate file `results/v6_4_7/S4_R02_symcaps_retest.md`. **S5 = R0.3 resolved (2026-06-11): SC ownership settled — three owners, none of them charge/cap values.** The per-device dump (`scripts/v6_4_7_s5_sc_dump.py`, KCL decomposition C·ΔV=Q_res+Q_cap+Q_num along the live trajectory) found Q_num≈0 (numerics clean), ΔQ_q≤0.05 fC (charge/cap VALUES exonerated — S4's symcaps charge win was coincidental compensation), and the id error REV-clamp-concentrated (TSMC7 Mnt +11.6 fC withheld in SAMPLE). The trajectory-head probe then found the **dominant owner: a harness `.ic`/uic semantics gap** — NGSPICE integrates `uic` from `.ic v(vsamp)=0`, while the DN runner uses `.ic` only as an OP guess and starts from the NN off-state leakage equilibrium (vsamp(0)=0.390 V on TSMC5 =Vin; 0.704 V on TSMC16). TSMC7's 2.2 mV hold droop = Mpt NEAR0 id leak −2.222 fC ≡ 2.2 mV exactly (P3-adjacent). SC dropped from P5/P7 EV claims. Gate file `results/v6_4_7/S5_R03_sc_device_dump.md`. **S5b SHIPPED (2026-06-11): uic-equivalent `.ic` start in `run_directnet_transient`** (constrained-OP via temporary pins, mirroring force_ic's guards, pins removed before the transient) — under the old protocol a bit-perfect model still failed (TSMC5's "14.65 %" = (Vin−NG_chg)/VDD exactly). A/B: SC census 0/4 → 1/4 — TSMC7 PASS (charge 3.40 %, droop 0.541 mV); TSMC5 11.96 % overshoot, TSMC12 8.14 % / TSMC16 6.20 % now UNDershoot (forward id too weak) + TSMC16 real 3.852 mV hold leak newly exposed; RO blind veto held (periods bit-identical, 3/4). E3 review: CORRECTION with conditions — **TSMC7 robustness probe 2/3 (droop healthy at all Vin; charge fails at 0.65·VDD 5.36 %) ⇒ TSMC7 SC = fragile PASS, off-default-Vin SC variant mandatory in S19 blind holdouts**; production `.ic` semantics (simulation.py, unconstrained-OP start) recorded as known issue; TSMC16's impossible 0.704 V NN equilibrium recorded; P2's SC-side EV shrinks (sample window now forward). **Honest headline: 10/16.** Gate file `results/v6_4_7/S5B_uic_semantics_fix.md`. **S6 = P1 resolved (2026-06-11): simulator EXONERATED, V6.4.6 P0-I RETRACTED.** The planned exact-id+charges injection gave 93.01 ps (and NMOS-only 92.91 — same ~92 ps as P0-I's id-only), but the decisive cell nobody had run was the **native LEVEL=72 control**: pycircuitsim's own BSIM-CMG path runs the identical RO at **46.64 ps vs NGSPICE 46.65 (ratio 1.000, 0.02 %)** — same solver, runner, window, estimator, cards. So the ~92–93 ps numbers are artifacts of the injection id-path mapping (gds floored to |id|/2, Rule-15 bypass); exact charges+caps injection adds nothing (92.30→93.01), and a within-NN cap-convention flip experiment bounds all cap-sign effects at ±1.3 % (NN's own convention is the better one). **Consequences:** RO error confirmed model-owned (the ~20 % NMOS dynamic id peak pull-down deficit of P0-G/H, now unclouded); P5 funded and re-scoped to the id surface along trajectories; P4/P8a/LoRA id-only levers re-armed; no model-side RO lever paused. Injection-style causal probes recorded as convention-fragile — use the native L72 device as the exact-physics endpoint (129 s vs ~4,400 s, too). Gate file `results/v6_4_7/S6_P1_swap_matrix.md`. Next: S7 = P2 reverse-clamp probe→relax.

> **V6.4.2 Phase-7a code is load-bearing.** The `_MonotoneVgResidual` residual + `--monotonic` CLI flag in `bsimar/{cli/train,models/direct_net,training/trainer}.py` + `pycircuitsim/models/mosfet_directnet.py` were documented as shipped in V6.4.2 but never committed. V6.4.4 ships in two commits: `4fcce2a` (docs + env var + scripts) referenced the code as "newly committed" but did not stage it; `df9cfe3` (the follow-up fix) restored the four files. Without `df9cfe3` the on-disk V6.4.1 seed-42 checkpoints fail to load with `Unexpected key(s) in state_dict: "mono.w_rest", "mono.w_vg_raw", "mono.b1", "mono.w_out_raw", "mono.b_out", "mono.sign"` — the local retrain ran with `_MonotoneVgResidual` in scope and the saved state_dicts carry those keys regardless of the `--monotonic` flag. With both commits applied, the inverter gate is 8/8 PASS on the V6.4.4 mix; stock checkpoints route through `mono=None` with no behaviour change at inference.

- **BSIM-CMG (LEVEL=72):** all 5 techs (ASAP7, TSMC5/7/12/16), DC <0.1% NRMSE, transient ~0.20% NRMSE vs NGSPICE.
- **DirectNet V6.4 (LEVEL=73, primary):** dedicated per-tech NMOS/PMOS checkpoints `tsmc{5,7,12,16}_dn_medium_*` (production size `medium`), each selected by **best-of-N over 8 seeds**. Inverter vs NGSPICE BSIM-CMG — VTC MaxErr: **TSMC5 62.0, TSMC7 60.1, TSMC12 32.3, TSMC16 29.7 mV** (NRMSE 1.20–2.13%); transient post-startup MaxErr: **TSMC5 37.9, TSMC7 51.5, TSMC12 57.6, TSMC16 54.9 mV** (NRMSE 1.17–1.38%); ΔVtrip ≤0.3 mV; R² ≥ 0.9981. TSMC12/16 embedding vocab = 6 per scope (5 variants + UNKNOWN).
- **V6.4 wins vs V6.3.1:** all 4 techs improve inverter VTC — TSMC12 −59% (78→32), TSMC16 −35% (45→30), TSMC5 −7%, TSMC7 −9%; transient holds. Key finding: DirectNet retraining is a **seed lottery** (139 mV VTC swing seed-to-seed); best-of-N selection on the real inverter sim — not val loss — is mandatory. See CHANGELOG "V6.4".
- **Open gate:** inverter VTC MaxErr ≤25 mV still unmet (V6.4 at 29.7–62.0; TSMC5/7 lag). Gain amplification at the trip multiplies the NN's Id residual ~20× into Vout. Next levers: a larger seed sweep, or plan Phase 6 (monotonicity / spectral-norm network constraints). A solver-side gm-floor (Phase 2b) was tried and **reverted** — unsound, checkpoint-dependent (see CHANGELOG "V6.4").
- **NO checkpoints for ASAP7 / LEVEL=74 BSIMAR.** Universal `refac_dn_*`, `refac_tf_*`, `v4_*`, and `checkpoints_legacy/` artifacts deleted on 2026-05-12. Simulating ASAP7 (or LEVEL=74) requires a separate retrain — out of scope.
- **Test infrastructure:** 3-level DC + transient suites (BSIM-CMG: 2+67+44 DC, 1+37+72 tran; NN: `verify_nn_dc_tran.py --tech TSMC5,TSMC7,TSMC12,TSMC16 --inverter-only` for the gate; per-tech routing via parser preempt cascade). Inverter metrics report: `scripts/eval_v6_3_1_inverter.py` → `results/v6_3_1_metrics_report/`. **V6.4 complex-circuit harness:** `tests/verify_complex_{ring_osc,opamp,sram_snm,switchcap}.py` + `tests/common/complex.py` (vs NGSPICE BSIM-CMG); not yet re-measured on V6.4 checkpoints.
- **V6.3.2 NN parametric harness:** `verify_nn_multi_tech_dc.py` (single-device L/NFIN/VT, 55 configs) + `verify_nn_multi_tech_tran.py` (inverter VTC+transient over P/N-ratio/VDD/Cload/slew/PW, 64 configs) port the PyCMG L3 sweeps to DirectNet via `tests/common/nn_sweep.py`. Stress-test gates: DC NRMSE <10%, inverter <15%. **V6.3.1 checkpoint results: DC 55/55 PASS; inverter 63/64 PASS** — sole FAIL `TSMC5_vtc_vdd_0p55` (16.8%, VDD−0.1V). Off-bin L/NFIN and VDD±0.1V are the stressors (baseline configs stay on-bin/nominal and pass cleanly). Reproducibility: keep `bsimar/checkpoints/` pointed at a *stable* copy — VTC trip gain ~20× amplifies any mid-run weight change; run with `OMP_NUM_THREADS=1`. Re-run on `feat/v6.4.1` seed-42 checkpoints: DC 55/55, VTC+tran 64/64 PASS (see CHANGELOG "V6.4.1").
- **Solver upgrades shipped:** sparse MNA (lil→CSR+spsolve), 2-level GMIN stepping [1e-8, 1e-12] with retry, BE→Trap→BDF-2, LTE sub-stepping, oscillation detection, hard `.ic` mode. V6.4 added env-gated transient C-stamp symmetrization (`NN_SYMMETRIC_CAPS`, default off — dormant, for future ring-oscillator work).
- **ASAP7 exclusion:** unchanged — would also need a dedicated per-tech checkpoint or fresh universal training.

## Setup

```bash
conda create -n pycircuitsim python=3.10 -y
conda activate pycircuitsim
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch
git submodule update --init --recursive
```

**Prerequisites:**

- NGSPICE 45.2+: `/usr/local/ngspice-45.2/bin/ngspice`
- OpenVAF 23.5.0+: `/usr/local/bin/openvaf`
- BSIM-CMG OSDI binary: `external_compact_models/PyCMG/build/osdi/bsimcmg.osdi`

## Quick Start

### Basic simulation

Create a `.sp` netlist (examples in `examples/`). BSIM-CMG geometric params: `L`, `NFIN`, optional `TFIN`/`HFIN`/`FPITCH`.

### NN training (V6.2 — per-tech dedicated)

```bash
# Generate per-tech data. --enable-inv-trip overlay covers all 4 TSMC techs
# inside pycmg/nn_generate.py. V6.3 re-centered it on VDD/2; V6.3.1 dropped the
# ±0.25·VDD Vbs sweep, so the overlay is ~3.5% of rows (was ~9.8% pre-V6.3.1).
# V6.3 also added the reverse_vds corridor class (~7.5% of rows, always on).
conda run -n pycircuitsim python external_compact_models/PyCMG/scripts/generate_nn_data.py \
    --device both --tech tsmc5 --enable-inv-trip --n-workers 8
conda run -n pycircuitsim python external_compact_models/PyCMG/scripts/generate_nn_data.py \
    --device both --tech tsmc7 --enable-inv-trip --n-workers 8

# Train dedicated per-tech DirectNet. --tech-scope auto-sets:
#   --exclude-techs (all other techs), --num-tech-codes (per-tech vocab + UNKNOWN),
#   default --data path (datasets/<scope>_<dev>.npz), and the save_prefix
#   (`tsmc{5,7}_dn_<size>_<dev>`) recognized by the parser preempt cascade.
conda run -n pycircuitsim python -u -m bsimar.cli.train \
    --model direct --size {small,medium} \
    --device-type {nmos,pmos} --tech-scope {tsmc5,tsmc7} --cuda --overwrite

# Convenience: full 8-cell sweep (S+M × NMOS/PMOS × TSMC5/TSMC7) at GPU 2.
bash scripts/train_per_tech_8cells.sh
```

**Checkpoints** (in `external_compact_models/bsimar/checkpoints/`):

- V6.2 DirectNet per-tech: `tsmc{5,7}_dn_{small,medium}_{nmos,pmos}_best.pt` + `_norm.npz`. Embedding vocab shrunk to per-tech variant count + 1 UNKNOWN slot (TSMC5: 5, TSMC7: 4). Production size is `medium`. No other checkpoints are present — universal `refac_dn_*` / `v4_*` artifacts were deleted on 2026-05-12.
- Resolver cascade (`pycircuitsim/parser.py`): for TSMC5/TSMC7 netlists, the per-tech slot `tsmc{X}_dn_{medium,small,large}` preempts the universal fallback chain (`refac_dn_* > v4_re_dn_universal > v4_dn_universal`). At V6.2 only `tsmc{5,7}_dn_medium_{nmos,pmos}` exist on disk; the universal fallbacks are unreachable until someone retrains a universal stack. Resolutions are logged at parse time as `[NN-resolver] L73 <name> TECH=<x> VT=<y> -> <chk> (scope=<s>, tech_code=<c>)`. Override via `--exp-name` at train time or `PYCIRCUITSIM_NN_CHECKPOINT_*` env vars at runtime.

**Netlist usage:** `.model nmos_nn NMOS (LEVEL=73 TECH=tsmc5 VT=lvt)` with `L=16n NFIN=10`. Parser auto-resolves the per-tech checkpoint and the local-vocab tech_code via `bsimar.config.local_variant_code(scope, tech, variant)`.

### Output files

Results in `results/<circuit_name>/<analysis_type>/`: `*_simulation.lis`, `*_dc_sweep.csv` / `*_transient.csv`.

## Testing & Verification

All tests require `conda activate pycircuitsim`.

**Shared infra:** `tests/common/{base,bsimcmg_dc,bsimcmg_tran,nn,nn_sweep}.py` and `tests/references/`.

**BSIM-CMG DC:** L1 `verify_bsimcmg_dc.py` (2) · L2 `verify_bsimcmg_dc_comprehensive.py` (67) · L3 `verify_multi_tech_dc.py` (44).
**BSIM-CMG Transient:** L1 `verify_bsimcmg_tran.py` (1) · L2 `verify_bsimcmg_tran_comprehensive.py` (37) · L3 `verify_multi_tech_tran.py` (72).
**NN V6.2 gate:** `verify_nn_dc_tran.py --tech TSMC5,TSMC7 --inverter-only` (12/12 PASS on the full TSMC5/7 sweep without `--inverter-only`). The `verify_nn_dc.py` / `verify_nn_tran_v4.py` legacy entry points target TSMC12 SVT which has no V6.2 checkpoint.
**NN parametric harness (V6.3.2):** the PyCMG L3 parametric sweeps ported to DirectNet (LEVEL=73) via `tests/common/nn_sweep.py`. `verify_nn_multi_tech_dc.py` — single-device NMOS/PMOS Id-Vgs over L/NFIN/VT (55 configs, 4 TSMC techs). `verify_nn_multi_tech_tran.py` — inverter VTC + transient over P/N ratio, VDD, Cload, input slew, pulse width. Baseline-gated: the parametric sweep runs only for techs that pass baseline. Geometry/VT/VDD ride on `dataclasses.replace(TestTechConfig)`; only the inverter-transient circuit knobs needed a (behaviour-preserving) refactor of `verify_nn_dc_tran.py` (`InvCircuitParams`). Run with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` — the NN inverter VTC has ~±1% NRMSE run-to-run scatter (high-gain trip point; harness pins `torch` to 1 thread).
**Other:** `verify_bsimcmg_op.py` (OP <0.02% vs NGSPICE).

Quick sanity:

```bash
python tests/verify_bsimcmg_op.py && python tests/verify_bsimcmg_dc.py && python tests/verify_bsimcmg_tran.py
```

---

## Development Guidelines

**Coding standards:** type hints on all signatures; clear names (`v_gate`, `i_drain`); docstrings for complex algorithms; voltage clamping Vgs±5V, Vds±10V.

**Separation principle:**

- `solver.py` builds MNA + executes NR (no device equations).
- `models/` calculates current/conductances (no matrix ops).
- `simulation.py` orchestrates (parse → solve → visualize).
- All devices inherit from `Component`.

**Key numerical techniques:**

- Sparse MNA solver: `lil_matrix` assembly, CSR + `spsolve` solve. O(n) memory, O(n·log n) solve.
- SPICE-standard convergence: `|ΔV| < VNTOL + RELTOL × max(|V_old|, |V_new|)` (RELTOL=1e-4, VNTOL=1e-7).
- GMIN (1e-12 S) prevents singular matrices. DC GMIN stepping opt-in via `use_gmin_stepping=True`: 2-level schedule [1e-8, 1e-12]. NN circuits use `_solve_dc_with_retry` (fast path first, GMIN retry on `_last_solve_converged=False`). BSIM-CMG never enters the retry branch.
- BE → Trap → BDF-2: BE step 1, Trap step 2+, BDF-2 auto on stiffness (NR>20 iters); one-way switch.
- Source stepping (20 steps); supply-relative adaptive damping with stuck-counter.
- DC oscillation detection: 5-snapshot ring, accepts averaged solution if variance < 10× tolerance.
- Hard `.ic` mode (`force_ic=True`): stamps `.ic` nodes as temporary V-source constraints, re-solves unconstrained. Required for SRAM latches.
- LTE sub-stepping (opt-in via `max_substeps`, default 1=disabled).

**Entry points:** CLI `main.py`; API `pycircuitsim.simulation.run_simulation()`; module exports (Circuit, Parser, Visualizer, run_simulation).

**Environment & tools:** conda env `pycircuitsim` at `/home/shenshan/.conda/envs/pycircuitsim`; PyTorch 2.10.0 (CPU); OpenVAF `/usr/local/bin/openvaf`; NGSPICE `/usr/local/ngspice-45.2/bin/ngspice`.

---

## Critical Design Rules

These rules were learned from bugs. Violating them causes NR divergence or wrong results.

### Sign Convention for Device Models

1. **Use terminal current `id`, NOT channel `ids`** — `ids = id - is ≈ 2*id` (2× error).
2. **NMOS** `calculate_current()` returns `-result["id"]`; **PMOS** returns `result["id"]` (positive = leaving drain).
3. **Solver stamping** uses unified "current leaving drain" convention. All VCCS conductances (g_ds, g_m, g_mb) need full 4-entry stamps (drain,ctrl+; drain,ctrl-; source,ctrl-; source,ctrl+). An incomplete stamp breaks Jacobian symmetry.
   ```python
   i_leaving = -i_ds if is_pmos else i_ds
   i_eq = i_leaving - g_ds * v_ds - g_m * v_gs - g_mb * v_bs
   rhs[d_idx] -= i_eq    # same for NMOS and PMOS
   rhs[s_idx] += i_eq
   ```
4. **gds floor** for stamping: `max(gds, 1e-12)`. Never `abs(gds)` — it flips large-negative to large-positive and diverges NR. Preserve gm/gmb signs.
5. **Update `_is_mosfet()`** in `solver.py` when adding new device types.
6. **Test both NMOS and PMOS** vs NGSPICE: single OP, DC sweep, inverter VTC, inverter transient.

### NN Model Rules (LEVEL=73 DirectNet V6.2; LEVEL=74 BSIMAR rules retained for resurrection)

Both LEVEL=73 (single-shot MLP, primary) and LEVEL=74 (autoregressive Transformer, parked per Rule 18) share the same data pipeline and inference-time rules. Both use `nn.Embedding` for tech-code identity (7-dim input: Vgs, Vds, Vbs, NFIN, L, T, tech_code). Rules 11–13 below describe BSIMAR-specific structure; they are not live at V6.2 but must be honoured if BSIMAR is resurrected.

1. **Jacobian consistency is mandatory** — gm/gds/gmb MUST be `torch.autograd.grad(id, V)`, never independent predictions. Holds for LEVEL=73 and LEVEL=74.
2. **Source-relative frame for BOTH device types** — shift all terminal voltages by -Vs before NN eval (`v_d_nn = v_d - v_s`, Vs ≡ 0). Training uses Vs=0; shift invariance makes this exact. Until V6.4.7 only PMOS was shifted — lifted-source NMOS (opamp tail pair, SC pass device, SRAM access) saw phantom Vgs/Vds with Vbs=0; the lifted-source canary `tests/verify_nn_lifted_source_dc.py` (NRMSE ≤10 %) guards this permanently.
3. **Training range covers NR overshoot** — margin ±VDD beyond operating range, not ±0.1V.
4. **Smooth voltage clamping** — softplus-based, NOT `torch.clamp`. Hard clamp creates zero-gradient cliffs that stall NR. Margin = 5% of per-dim training range.
5. **Physics-based gds floor** — `gds = max(gds, |id|*0.5, 1e-12)`. NN autograd gds ≈ 0 in saturation; without the floor inverter gain → ∞ and NR diverges. At FinFET 16nm BSIM-CMG λ=0.3-1.2 V⁻¹. Floor only affects the NR Jacobian, not the converged solution.
6. **TSMC asymmetric L** — NMOS L=16nm, PMOS L=20nm; NNTechConfig uses `L_nmos`/`L_pmos`.
7. **ASAP7 modelcard name mapping** — parser auto-maps netlist names to `nmos_rvt` / `pmos_rvt`.
8. **PyCMG integration** — `bsimar/config.py` re-exports `NNTechConfig`, `TECH_CONFIGS`, `TECH_CODE_MAP`, `OUTPUT_COLUMNS` from `pycmg.nn_config`. Backward-compat alias `TechConfig = NNTechConfig`. Training VDD may differ from PyCMG's runtime VDD; check `NNTechConfig.VDD` per tech.
9. **Data validation** — `eval_single_point` rejects NaN/Inf and `|id| > 1A`. PyCMG `eval_dc` raises `RuntimeError` on internal-node convergence failure. Default NFIN range `[2, 3, 5, 10, 15, 20, 24]` (NFIN=1 excluded).
10. **Loss layer** — both models use `bsimar.losses.MAELoss` with **per-target LDS weights only** (3-axis stack collapsed to 1 in v5 Phase A). Hard-wired in `train_directnet` / `train_transformer`. DO NOT re-add: `DirectLoss`, `ChargeConsistencyLoss`, `SignConsistencyLoss`, `BoundaryLoss`, `SlopeMatchLoss`, Vov-LDS / subthreshold-LDS axes. Structural Vds gate (`apply_id_gate`) and slope-match loss deleted 2026-05-03 — rule 15's inference-time correction already enforces Id(Vds=0)=0.
11. **BSIMAR output ordering** — Transformer output in `BSIMAR_COLUMN_ORDER` (`qg, qb, qd, qs, id, gm, gds, gmb, cgg, cgd, cgs, cdg, cdd`), not `OUTPUT_COLUMN_ORDER`. Consumer code (`mosfet_bsimar.py`) takes autograd derivatives at the right column indices.
12. **Parallel cap head** — Transformer emits 5 capacitances in parallel from gmb hidden state, not sequential AR steps. AR loop runs 8 steps (charges + currents/conds). `parallel_caps` and `grouped_inputs` structural, not configurable.
13. **Unified CLI** — `python -m bsimar.cli.train --model direct --size {small,medium,large} --device-type {nmos,pmos} --tech-scope {tsmc5,tsmc7,universal} ...`. With `--tech-scope tsmc{5,7}` the default save_prefix is `tsmc{X}_dn_<size>_<device>` (recognized by the parser preempt cascade). Same `.npz` from PyCMG; checkpoints under `external_compact_models/bsimar/checkpoints/`.
14. **Charge conservation** — simulator always computes `qs = -(qg + qd + qb)` analytically, even for 13-output models that directly predict `qs`. Guarantees Kirchhoff conservation at every transient timestep.
15. **Analytical Vds correction** — `_MOSFETNNBase._apply_vds_correction()` enforces Id(Vds=0)=0 and Id=0 for reverse-Vds at inference. Four-part (order matters):

- (a) **Rail-restoring extrapolation** when `|Vds| > VDD_train` (= `self._vdd_estimate`, from training norm stats): quadratic Id ramp `½·g_max·overshoot² / x_ref` and linear gds ramp `g_max·overshoot / x_ref` (g_max=1mS, x_ref=½·VDD_train). Both zero-valued zero-sloped at the boundary so NR sees a smooth join. Must run BEFORE the fast-path early-return. **The injected `id_extra` adds in the same direction as the conducting-current sign** (NMOS `id -= id_extra`, PMOS `id += id_extra`) — restoring leakage strengthens the device's pull toward the source rail. The opposite sign creates a current-free dead-band inside `VDD_train < |Vds| < 20·VT` (the V6.1 bug; see CHANGELOG "V6.2 — Rule 15(a) sign fix").
- (b) one-sided `1-exp(-|Vds|/VT)` with VDD-proportional `VT = max(0.06·VDD, 0.026)V` for Id/gm/gmb.
- (c) symmetric gds with linear-region conductance `|Id_raw|·exp(-|Vds|/VT)/VT`.
- (d) sign enforcement (NMOS id≤0, PMOS id≥0).

  Step (a) replicates PyCMG's restoring leakage/impact-ionization physics so NR converges to the true rail instead of locking on the NN's flat-zero plateau outside `[-VDD_train, VDD_train]`. Inference-time only — no retraining required.

16. Always report MRE (%), R^2, NRMSE, Max error (mV) metrics.
17. Exclude ASAP7 tech at the current stage.
18. Do NOT train/eval BSIMAR Transformer model at this stage. Only care about DirectNet model.
19. **Per-tech models use a LOCAL embedding vocab.** When `--tech-scope` is `tsmc5` or `tsmc7`, the dataset loader remaps universal tech codes to a 0-indexed per-tech vocab and the trainer instantiates `DirectNet(num_tech_codes=N, unknown_code_id=N-1)`, where N = variants+1 (TSMC5: 5, TSMC7: 4). The training-time `p_unknown` dropout writes `unknown_code_id` into the embedding, so a misaligned UNKNOWN id → CUDA assert. **Derive `unknown_code_id` from `num_tech_codes`; do NOT hardcode the universal value (17).** Parser uses `bsimar.config.local_variant_code(scope, tech, variant)` to remap at inference; the scope is read from the resolved checkpoint stem (`tsmc{5,7}_dn_*` → local; everything else → universal).
20. **Re-validate Rule 15(a) when resurrecting TSMC12/16 or LEVEL=74 BSIMAR.** The V6.2 sign fix changed which sign convention is shipped; the original sign was load-bearing under the wrong-sign clamp for unshipped code paths and they have not been re-tested.

---

## References

- **ngspice** — physics equation verification.
- **Xyce** — architecture patterns for device/solver separation.
- **BSIM-CMG** — FinFET compact model (LEVEL=72), via PyCMG.
- **ASAP7** — https://github.com/The-OpenROAD-Project/asap7_pdk_r1p7.git
- **PyCMG** — https://github.com/ShenShan123/PyCMG.git

## Important Paths

- **PyCMG submodule:** `external_compact_models/PyCMG/` (21 device variants).
- **OSDI binary:** `build/osdi/bsimcmg.osdi` (PyCMG-relative).
- **Modelcards:** `modelcards/` (PyCMG-relative); ASAP7 `*.pm` committed; TSMC raw PDK `cln*.l` is gitignored/IP-protected — naive modelcards regenerated on-the-fly via `pycmg.tech.resolve_modelcard` into `build/modelcards/`.
- **Results output:** `results/<circuit_name>/<analysis_type>/`.
- **Test results:** `tests/verify_*_results/` (generated, not tracked).
- **Sprint history:** `docs/CHANGELOG.md`.

## Other Tips

* **Start every complex task in plan mode** — pour energy into the plan for 1-shot implementation. Re-plan the moment something goes sideways; enter plan mode for verification steps too.
* If the plan has several solutions or stages, implement them in sequence. Use git commit first before you modify anything, keep the useful one that make progress and incorperate it. Otherwise, revert the solutions that were proven to be no help with git reset.
* **Update CLAUDE.md before every git commit**.
* Whenever there is a version update, update the `docs/CHANGELOG.md`.
* Always record the dead end proposal (the one being reverted), they are as important as the successful ones.
* **Never be lazy** — never simplify code or skip tests. **NEVER** use simplified equations or self-defined CMG models as reference; ALWAYS use simulation results as ground truth.
* **Use subagents** — second agent for staff-engineer plan review; multiple subagents on separate branches to try multiple solutions; roll back to main when a subagent hits a dead end.
* Enable "Explanatory" / "Learning" output style in `/config` to see *why* behind changes.
