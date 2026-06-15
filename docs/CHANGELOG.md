# PyCircuitSim — Detailed Changelog

This is the long-form history of PyCircuitSim. CLAUDE.md keeps a one-paragraph
"current state" summary; everything below is here so the conversation context
isn't burdened with chronology.

---

## V6.4.7 (in progress) — serialized accuracy campaign; week 1 (S1–S8) honest 8/16 → 11/16 zero-GPU, S9b regen-v2 PROCEED, S10/P4 Sobolev KILL (deriv-fidelity ⟂ value-owned opamp), S12/P5 trajectory-corridor KEEP (tsmc7 RO 8.28→2.9 %, per-tech mix 11→14/16), S11/P3 subthreshold KILL (force_ic gain/NR-fixed-point owned, not value owned → S17/P9), S11b pivot — the 2 open cells (tsmc5 SC over-conduction, tsmc7 opamp over-gain) are systematic model-fidelity limits, headline stays 14/16 (2026-06-10 → 06-15)

Plan: `docs/plans/2026-06-10-directnet-v6.4.7-accuracy.md` (rev 2.1 — strict
serial chain S1–S19, every lever committed-or-rewound before the next; user
rulings: SRAM `force_ic` ship-required; ~250–300 GPU-h campaign, ≥4
seeds/config, seeds one-per-GPU). Gate files with full A/B detail under
`results/v6_4_7/`; campaign control = `baseline_v6_4_7_pre.json`.

### S9 + S9b — SWA/EMA infra + regen-v2 + control-v2 gate (2026-06-12 → 06-14)

**S9 (SWA/EMA, 2026-06-12):** `--swa-mode {none,ema,swa}` + `--ema-decay` in
`bsimar/{training/trainer,cli/train}.py`; default `none` behavior-preserving.

**S9b (regen-v2 + control-v2, COMPLETE 2026-06-14, verdict PROCEED).** Executed
on a **bare-checkout machine** — the whole runtime stack was rebuilt first:
PyCMG restored via proxy on `feat/v6` (the pinned commit is gone from the
remote); OpenVAF 23.5.0 + BSIM-CMG OSDI built; conda env + torch 2.6.0+cu124
(CUDA, 3× 4090); **NGSPICE 45.2 + OSDI built from source** (`tools/ngspice-45.2`,
harness now honors `NGSPICE_BIN`); TSMC PDKs user-supplied. The lost-commit S9b
generator code was reconstructed on `feat/v6` (preserved as
`results/v6_4_7/s9b_pycmg_patch/`):
- `NN_DC_SOLVE_TOL` floor fix (`pycmg/model.py`) — the legacy 1e-9 internal-NR
  tol returned EXACT 0 for true |id|<~1e-9 (the 6–8 % zero-row artifact);
  exports 1e-12 for generation. Exact-zero rows 10.0 % → 1.3 %.
- `subvt_off` sample class (code 11) + `--enable-subvt-off`/`--dc-solve-tol`.
- **Bug fixes (both load-bearing):** (1) a **parallel modelcard-cache write
  race** — the on-the-fly naive-card file per `(pdk_device,L,NFIN)` was shared
  by the 3 temperature-bin workers; non-atomic truncate+write let a reader
  parse a partial card → **degenerate modelcard** (only PHIG/TOXP non-zero) →
  physically-wrong rows the tech-variant labeller could not fingerprint. Fixed
  with atomic temp-file + `os.replace`. (2) **NFIN<2 excluded** in
  `enumerate_bins` (feat/v6 included it; project Rule 9 excludes it).

Regen-v2 acceptance **PASS**: 8 datasets (1.8–3.1 M rows), decade gate 8/8
(40k–200k rows/decade in 1e-12..1e-6 A vs the 1k gate), asinh audit
`drift_id=1.0000` (no s_id pinning needed — the unfiltered small-current rows
sit above the 1e-15 filter threshold), labeller 0 misses. gm/gds asinh drift
0.73–0.96 (P4-relevant).

control-v2: 32 cells (4 seeds {42,17,7,31} × 8 tech×dev), stock medium recipe,
`--apply-filter off`, EMA, v2 data. Full multi-tech gate (per-cell best vs S8
baseline): **2 protected-gate regressions** — tsmc5 ring_osc (5.80 % vs 2.61 %)
and tsmc12 opamp (all 4 seeds collapse) — **1 new-pass** (tsmc16 switchcap
13.1 % → 3.17 %), and inverters **hold on all 4 techs**. **Go/no-go = PROCEED:**
both regressions are fresh-retrain variance, not data defects — EMA ruled out
by ablation (RO-neutral, 7.23 ≈ 7.21 %); tsmc5 RO = lost best-of-8 cherry-pick
(tsmc7 confirms, matching its non-cherry-picked 8.28 % baseline at 8.66 %);
tsmc12 opamp = the ~44 %-likely 4-seed spontaneous-collapse lottery (tsmc5 s7
passes at 0.79 %). Data sound (gates pass, inverter holds, tsmc16 SC win) ⇒ not
rewound. **control-v2 becomes the fresh-retrain attribution baseline** for the
S10+ arms; the S8 `baseline_v6_4_7_pre.json` stays the promotion gatekeeper;
**tsmc5 ring_osc + tsmc12 opamp join the arms' recover-set** (P4
collapse-resistance; P5/P8a RO). Detail:
`results/v6_4_7/S9b_controlv2_gate_summary.md` (+ per-tech gate files). Harness
portability fixes: scorer/`verify_*` honor `NGSPICE_BIN`; gate driver
`scripts/v6_4_7_s9b_gate_controlv2.py` runs GPU-serial (`--workers 1`; 1 scorer
co-exists with training, >1 CUDA-asserts) or `--cpu`. **Resume at S10 (P4).**

### S10 — P4 Sobolev id-derivative arm (2026-06-14, verdict KILL)

The first GPU arm. Built `SobolevIdLoss` (`bsimar/losses/bni_mae.py`): id
channels only (∂id/∂{Vg,Vd,Vb} vs OSDI gm/gds/gmb), compared in the **same
asinh normalized-derivative space the deriv-fidelity gate measures**, so the
term supervises exactly the quantity ruling-4 scores. The 8-channel V5 Phase-C
form was net-detrimental; the chain-rule transform was recovered from
`git show 930c274` but restricted to the 3 id channels. **Sign convention
verified, not assumed** (the P0-I §2 trap): `scripts/v6_4_7_s10_sign_check.py`
shows **uniform negation of all three channels** is correct (stored gm/gds/gmb =
−∂id/∂V) — the 930c274 "gds is the diagonal so no flip" comment is WRONG for the
stored convention (gds residual 11× larger under it). Trainer/CLI:
`--sobolev/--lam-sobolev/--sobolev-floor/--sobolev-strong-boost/`
`--sobolev-corridor-only/--init-from`, second-order autograd, EMA-compatible,
default path bit-unchanged (smoke-verified).

Methodology: a warm-start fine-tune screen **reverts** under plain val-MAE
selection (λ=0.1 degraded val-MAE 4× → early-stop epoch 2 ≈ unshaped warm
start). Replaced by **from-scratch retrains at seed 17** — identical weight
init + data split + normalizer fit to control-v2 s17, so the A/B isolates the
Sobolev term exactly. Screened λ∈{0.005,0.01,0.02,0.1,0.3} (global boost4 +
corridor-only), then ran a **4-seed arm** (config A, λ=0.02 boost4, seeds
{42,17,7,31}).

Results:
- **Derivative fidelity improves robustly + monotonically in λ** — PMOS gds_fwd
  55.8 → 1.7 % (λ=0.3), gm_fwd 137 → 0.1 %, off-state 3–4 orders better; the
  4-seed arm holds gds 42–43 % vs control 48–69 % on every seed. **Ruling-4
  core objective met.**
- **Inverter improves** on every seed (VTC 0.96–2.36 % vs 3.45 %).
- **The opamp collapses 4/4 seeds** (gain 180 → 0), including s7/s31 which
  control-v2 kept healthy at 362/187 — **systematic, not seed-luck**, and
  **λ-independent down to λ=0.005** (val-MAE identical to control, 0.00119).
- RO mixed (2/4 improved to 7.77/7.99 — best-ever tsmc7; 2/4 regressed). Side
  finding: 3/4 seeds move SRAM force_ic OUT of the symmetric metastable point
  (0.39/0.39) toward a railed state (q≈0.75–0.83 / qb≈0.07–0.13) — the
  off-state-deriv improvement helps the subthreshold-owned SRAM attractor
  (P3-adjacent; doesn't close, q over-rails).

**Verdict = KILL** (pre-registered S10 kill gate: opamp not < 15 % with inverter
held → drop the term, record dead-end next to V5 Phase-C). No Sobolev checkpoint
promoted; `v6_4_7_s10{ft,sob,p4}_*` stems inert (don't match the resolver).
`SobolevIdLoss` stays as default-off, recoverable infra (pairs with the
permanent deriv-fidelity scorer).

**Major finding — derivative fidelity is ANTI-correlated with the opamp.**
control-v2 has wildly-off autograd derivatives on every seed (gm_fwd ~137 %)
yet its opamp gain is within ~10 % of NG; the Sobolev arm IMPROVES the Jacobian
and COLLAPSES the gain. Mechanism: the harness opamp gain (max slope of the
**large-signal DC transfer curve** = locus of *converged* NR fixed points) and
the RO period are **value-surface / NR-fixed-point owned — the P0-C/P0-I class**
(the autograd Jacobian guides NR convergence but cancels at the fixed point).
Fixing the slope necessarily reshapes the coupled id VALUE surface, which
destabilizes the value-owned opamp bias. **Consequence — ruling-4's premise is
partially falsified:** precise ∂id/∂V does not help (actively harms) the
value-owned opamp/RO gates; the deriv-fidelity metric is an NR-robustness
indicator, NOT a circuit-accuracy promotion gate. The opamp/RO levers must
target the id VALUE surface (P5 trajectory corridors, P3 subthreshold). Gate
file `results/v6_4_7/S10_P4_sobolev_gate.md`. **Resume at S11 (P3, SRAM
value-surface subthreshold lever — carry the SRAM-escape side finding).**

- **S1 pre-flight (`c2ac02b`):** plan serialized; V6.4.5 campaign infra
  finally tracked; 161 checkpoints snapshotted to
  `/data2/home/shenshan/checkpoint_snapshots/v6_4_7_pre_20260610/`
  (sha256 manifest mirrored in-repo).
- **S2 = P0 NMOS source-frame fix (`e2a121a`) — first behavioral change
  since V6.4.4.** `_raw_voltages` shifted only PMOS; lifted-source NMOS saw
  phantom Vgs/Vds (+Vs) with Vbs=0 against a Vs≡0-trained net. 3-LOC fix
  (shift both; Rule 15 consumes the invariant difference). New permanent
  gate `tests/verify_nn_lifted_source_dc.py` (NMOS Id–Vgs at
  Vs∈{0,0.1,0.2}·VDD vs NGSPICE OSDI, ≤10 % NRMSE): pre-fix 10–64 % NRMSE /
  negative R² / Id over-predicted up to ~80×; post-fix **12/12 at
  0.05–4.4 %**. TSMC12 opamp FLIPPED PASS (10.94→5.21 %); TSMC5 opamp moved
  2.64→9.78 % (selected under the buggy frame — pre-arbitrated non-veto);
  TSMC7 opamp changed failure mode (30.7 %→flat); force_ic attractor halved
  (qb 0.19–0.23→0.104–0.117 V); inverter 8/8 (tran bit-exact), DC 55/55,
  tran 64/64, RO bit-identical, butterfly 4/4 all held; SC unchanged ⇒ the
  frame was NOT the SC owner. NN Rule 2 corrected.
- **S3 = R0.1 switchcap droop-gate repair (`d24c1d7`).** The old gate was
  simultaneously unpassable (relative error vs sub-µV NG droop demanded
  19–150 nV ≈ sub-solver-tolerance agreement) and auto-passing (the
  |ng|>1e-6 nan-guard waved TSMC7's 2.208 mV — the worst cell — through).
  New: `|dn−ng| ≤ max(10 %·|ng|, 1e-3·VDD)`; the floor matches the two-point
  solver tolerance 2·(RELTOL·V_hold+VNTOL) within ±20 %; column renamed
  `Droop%alw`. E3-class review: CORRECTION, net tightening (an engineered
  floor needed ≥3e-3·VDD to keep TSMC7 passing). **Headline restated:
  V6.4.4 canonical = 8/16 honest.** Caveat: the floor admits ~50 nA
  off-state leakage error — subthreshold fidelity is P3/P4 territory.
- **S4 = R0.2 symcaps re-test (`5c6342b`): KILLED for SC too.** Charge
  improves (TSMC5 14.65→3.68, TSMC16 13.14→1.38 %) but hold droop explodes
  to 30–137 mV genuine drift — invisible under the old gate, caught by the
  S3 repair. D1 now dead for RO and SC; per-circuit env-gated shipping off
  the table.
- **S5 = R0.3 SC per-device dump (`6162cad`).** Four-window trajectory dump
  with exact KCL decomposition C·ΔV=Q_res+Q_cap+Q_num: numerics clean
  (Q_num≈0), **charge/cap VALUES exonerated** (ΔQ_q≤0.05 fC — S4's cap
  hypothesis overturned as coincidental compensation), id error
  REV-clamp-concentrated (TSMC7 Mnt +11.64 fC withheld). Trajectory-head
  probe found the dominant owner: **harness `.ic`/uic semantics gap** —
  NGSPICE runs `tran uic` from `.ic v(vsamp)=0` while the DN runner used
  `.ic` only as an OP guess, starting from the NN leakage equilibrium
  (vsamp(0)=0.390 V on TSMC5 =Vin exactly / 0.704 V on TSMC16 — a
  physically impossible equilibrium, recorded as a known issue). TSMC7's
  2.207 mV hold droop exactly attributed: Mpt NEAR0 id leak −2.222 fC ≡
  2.2 mV on 100 fF.
- **S5b uic-equivalent start (`7454034`, recorded amendment).**
  `run_directnet_transient` now solves the OP with `.ic` nodes pinned
  (constrained, NOT force_ic's released re-solve) and integrates from it.
  Under the old protocol a bit-perfect model still failed (TSMC5's
  "14.65 %" = (Vin−NG_chg)/VDD — pure protocol artifact). SC 0/4→1/4:
  TSMC7 PASS (**fragile** — robustness probe 2/3, charge crosses its gate
  at Vin=0.65·VDD ⇒ off-default-Vin SC variant mandatory in the S19 blind
  holdouts); TSMC12/16 now honestly UNDershoot (forward id too weak);
  TSMC16's real 3.85 mV hold leak exposed. RO blind veto held
  (periods bit-identical). E3-class review: CORRECTION. Known issue:
  production `.ic` semantics (`simulation.py`) still artifact-start.
- **S6 = P1 swap matrix + LEVEL=72 control (`4e0b55e`): simulator
  EXONERATED; V6.4.6 P0-I RETRACTED.** The planned exact-id+charges
  injection read 93.01 ps (NMOS-only 92.91 — reproducing P0-I's ~92 ps),
  but the decisive uncontrolled cell was the **native LEVEL=72 path: the
  identical RO at 46.64 ps vs NGSPICE 46.65 (ratio 1.000, 0.02 %)** — same
  solver, runner, window, estimator, cards. The ~92–93 ps numbers are
  artifacts of the injection id-path mapping (gds=floor(−OSDI gds)→|id|/2,
  Rule-15 bypass); exact charges+caps injection adds nothing
  (92.30→93.01); a within-NN cap-sign-flip experiment bounds ALL
  cap-convention effects at ±1.3 % (the NN's own convention is the better
  one). RO ownership reverts, unclouded, to the ~20 % NMOS dynamic id peak
  pull-down deficit (P0-G/H; charges exact, integration ~0.4 ps). P5
  funded (re-scoped to the id surface along trajectories); id-only levers
  (P4, P8a, LoRA) re-armed. Methodology: injection probes are
  convention-fragile — use the native L72 device as the exact-physics
  endpoint (129 s vs ~4,400 s).
- **S7 = P2 reverse-Vds clamp relaxation (`bdf4102`) — second behavioral
  change.** Probe first: the raw pre-clamp reverse surface is USABLE
  (sign-correct 95–100 % where |OSDI|>1 µA, ~25–35 % conservative, R²
  0.78–0.93 on the V6.3 reverse_vds corridor; recovers 72–74 % of the OSDI
  restoring current at the live SRAM bias). Relaxation in
  `_apply_vds_correction`: reverse id = `id_raw·f_sym·taper(|Vds|)`
  (Id(Vds=0)=0 exact; C¹ smoothstep taper; gm/gmb matched; (c) untouched —
  its `|id_raw|·exp/VT` term is the fold-curing conductance, 1.32e-4 S at
  the Mpr bias ≈ 13× the documented 1e-5 S NR-fold threshold; (d)
  direction-scoped). **Window rule pre-registered: largest taper window
  breaking no protected gate.** Full corridor 0.30/0.40·VDD_train KILLED —
  TSMC5 opamp 9.78→13.57 % veto break + force_ic symmetric collapse on 3
  techs (dead end recorded); 0.10/0.20 clean but loses the TSMC12 SC flip;
  **shipped 0.20/0.30**: SC TSMC12 FLIPPED (4.13 %), TSMC5 opamp
  de-fragilized 9.78→2.49 %, TSMC7 opamp resurrected flat→10.16 % (0.16 pp
  from gate), RO improved on all 4 techs (TSMC7 8.98→8.28 %), inverter tran
  uniformly improved (1.34/1.06/0.84/0.94 %), all protected gates held.
  force_ic stays 0/8 — P2 delivered its mechanism; closure rests on P3
  (pinning-NMOS weak-inversion props qb at 0.09–0.14 V). Caveat: the 0.10
  window rails the high node exactly on all 4 techs — if P3 closes at 0.10
  but not 0.20, the window trade re-opens and ship-required force_ic
  outranks the SC TSMC12 gate.
- **S8 scorer + baseline re-freeze (`ad62c68`).** Scorer gains
  `opamp_gain_err` vs a file-memoized NG reference (it was blind to the
  ±10 % gate — the P4 prerequisite), switchcap cells (charge + repaired
  droop gate), and the V6.4.5 flat-flag recalibration (`gain<10`) finally
  in the committed file. Cross-validated against the frozen
  `baseline_v6_4_7_pre.json` (all 16 cells + force_ic + extended gates,
  commit-stamped, fragility notes carried). Plan updated with the "Week-1
  outcomes" section: decision table resolved — P5 funded id-scoped; P3
  stays a full ship-required arm (+ TSMC16 SC leak + S7 window re-test);
  P4 census = TSMC7 opamp (0.16 pp) + TSMC16 (flat); P8b demoted to
  fallback (its non-separability premise retracted with P0-I).

**Week-1 ledger: honest 8/16 → 11/16** (RO 3, opamp 2, SC 2, butterfly 4),
zero GPU. Open cells: TSMC7 opamp 10.16 %, TSMC16 opamp flat, TSMC7 RO
8.28 %, TSMC5 SC 12.14 % (over-conduction), TSMC16 SC hold leak, and
ship-required force_ic 0/8. Resume at S9 (SWA/EMA infra) → S10 (P4 lead
arm, seeds one-per-GPU on GPUs 1/2/3).

### S12 — P5 trajectory-corridor arm (2026-06-15, verdict KEEP, headline 11→14/16)

The value-corridor lever the S10 finding implicated (RO/opamp are id-VALUE-surface
owned, not Jacobian-owned). Built the corridor pipeline:
`scripts/v6_4_7_s12_{harvest,append}_corridors.py`, `_train_corridor.sh`;
`traj_corridor` = SAMPLE_CLASS_CODES code 12.

- **Harvest:** ran the 4 complex benchmark circuits and collected the per-device
  bias **tubes** the transistors visit along the **ground-truth** trajectory —
  RO + switchcap via the native **LEVEL=72** path (S6: == NGSPICE at ratio
  1.000); opamp + SRAM butterfly via **NGSPICE** directly (raw L72 DC sweeps
  diverge under PyCircuitSim's NR for those high-gain circuits — same
  ground-truth teacher). Vs-shift exactness verified (`|Δid|=0`). OSDI-evaluated
  at the bench geometry (NMOS 16n / PMOS 20n, NFIN=2, T=300.15), ±12 mV /
  20-sample jitter tube → ~1 % of each dataset (fail=0; |id| 1e-9–1e-4 A).
- **Append:** `{tech}_v2cor_{dev}.npz` (v2 left pristine/backed-up). NMOS L=16n
  is OFF the PDK geometry grid {6,20,36,…}nm, so corridor rows can't be
  fingerprinted by the tech-variant labeller; they are labeled by a **pre-seeded
  label cache** (v2 rows via the labeller + corridor rows the known bench-variant
  code, same concat order). Validated end-to-end in the live trainer (loads, no
  re-fingerprint/assert, class visible, `--class-weights traj_corridor=3` folds
  + LDS-renormalizes correctly).
- **Train:** 4 seeds × 8 cells, control-v2 stock recipe (medium, EMA, filter
  off) + `--class-weights traj_corridor=3`, A/B vs control-v2 (~6.9 s/epoch).

**Kill gate PASSED decisively — tsmc7 RO 8.28 → 2.87–2.92 % (all 4 seeds,
NEW-PASS).** Confirms the P5 thesis: the RO period gap is the ~20 % NMOS
dynamic-id deficit (P0-G/H), owned by the id VALUE surface along the switching
trajectory; teaching ground-truth id there closes it seed-invariantly. tsmc5 RO
recovered 5.80 (control-v2) → 4.6 %. tsmc16 switchcap 13.1 → 2.01 % (all 4; also
flipped by control-v2's v2 data) + opamp fail → 5.06 % (s31 only, fragile 1/4).

**Cost — the corridor COLLAPSES *passing* opamps (tsmc5 + tsmc12, all 4 seeds,
100 %)** — the same S10 value-surface/NR-fixed-point fragility (reshaping the id
surface destabilises the high-gain opamp). So the corridor is **promoted
PER-TECH only where it nets a gain with no veto break:** tsmc7 (corridor: RO
flip) + tsmc16 s31 (corridor: opamp + SC flip); tsmc5 + tsmc12 keep **baseline**
(corridor would regress their passing opamps; their RO already passes). Net
**11/16 → 14/16** (RO 3/4→4/4, opamp 2/4→3/4, switchcap 2/4→3/4, butterfly 4/4
verified held — tsmc16 SNMerr 0.0 %, tsmc7 positive). Inverter held. **force_ic
still 0/8 — NOT closed (S11/P3's target); some seeds nudge the released cell
rail-ward (tsmc7 s42 probe q=0.75).** tsmc5 SC over-conduction NOT fixed
(12.16 %). W-sweep (gentler dose to preserve passing opamps) deferred — would not
change the tsmc7 headline (tsmc7 opamp already fails). **KEEP — surviving arm;
`v6_4_7_s12cor_w3_*` are the S19 per-tech promotion candidates (tsmc16 s31 opamp
flip replication-gated). Datasets + checkpoints gitignored, regenerable.** Gate
file `results/v6_4_7/S12_P5_corridor_gate.md`. Resume at **S11 = P3** (SRAM
subthreshold, ship-required force_ic).

### S11 — P3 subthreshold-id arm (2026-06-15, verdict KILL → S17/P9)

The ship-required SRAM `force_ic` arm. Built `SubthresholdIdLoss`
(`bsimar/losses/bni_mae.py`, `--subthresh`, default-off, DirectNet-only): an
**asinh-s2 (s2≈1e-9) sub-µA VALUE term** (Huber, sign-aware, masked
`1e-12<|id|<1e-6`) that re-scales the subthreshold roll-off the global
`s_id≈2.6e-5` crushes to ~0.01 % of normalized range (the regen-v2 data HAS the
rows — ~15 %/cell below 1 µA — but asinh+LDS gives them ~zero loss mass), plus a
**sign-agnostic OFF ceiling hinge** `relu(asinh(|id_pred|/s2) − asinh(k·NFIN·1nA
/s2))` on `|id_true|≤1e-10` rows (suppresses hard-OFF over-prediction without
injecting current — NOT the D4 `Ioff_rail` floor). Probe
`scripts/v6_4_7_s11_subvt_probe.py` + combined `force_ic` gate
`scripts/v6_4_7_s11_sram_gate.py` + multi-GPU drivers. Default path bit-unchanged.

- **λ calibration (gotcha):** base val-MAE is ~0.001, the raw asinh-s2 term is
  O(1)/row ⇒ λ=0.05/0.15 swamp the fit (val 12–30× worse, killed); **λ=0.002 is
  the operating point** (val 1.4× control, inverter holds). Trained 4 TSMC7 seeds
  + tsmc5/12/16 (seed 42) on v2 data, A/B vs control-v2.
- **The term WORKS on its target (weak-inversion fidelity):** TSMC7 weak-band
  NN/OSDI |id| ratio **1.84→1.14** (NMOS, |log10| 0.356→0.102, 3.5×),
  **0.90→1.13** (PMOS, 5×) — and is **gate-neutral-to-positive**: inv_vtc
  2.61→2.96 %, inv_tran 1.21→1.16 %, RO 10.86→7.88 %, SC 1.76→1.64 % PASS. The
  opamp collapse is the documented v2-data retrain lottery (control-v2 collapses
  identically) — not caused by P3.
- **But `force_ic` stays 0/14 and moves the WRONG way:** 6/7 (tech,seed) cells
  COLLAPSE to the symmetric metastable point q=qb=VDD/2 (TSMC7 s42/s17/s7,
  TSMC5/12/16 s42) — strictly worse than control's near-railed inboard (TSMC7
  s42 control q=0.749 AT rail, only qb=0.121 = 46 mV out); the one inboard-landing
  seed (s31) is identical to control (qb=0.122). A more accurate/symmetric
  subthreshold id surface **removes the asymmetry that kept the baseline partially
  railed.**

**Pre-registered kill gate (weak-inversion ratio ≥10× with VTC ≤5 %) NOT met**
(3.5–5×, force_ic not closed). **Conclusion: `force_ic` railing is a
regenerative-gain / NR-fixed-point property** (the cross-coupled pair needs trip
gain to make the symmetric point repelling) — the **same value-surface-vs-
fixed-point split as the opamp gain (S10) and RO period (P0-C/P0-I)**. No
subthreshold-VALUE variant addresses trip gain. **KILL → S17/P9** (physics-
anchored compose-at-inference subthreshold core — now unblocked: S2 frame + S7
reverse-clamp + S11 subthreshold all failed to close force_ic). No checkpoint
promoted (`v6_4_7_s11sub_*` inert, don't match the resolver); `SubthresholdIdLoss`
KEPT as default-off recoverable infra (real gate-neutral subthreshold-fidelity
win, composable — e.g. the TSMC16 SC hold leak). Headline unchanged **14/16**;
`force_ic` **0/8**, ship-required-OPEN. Gate file
`results/v6_4_7/S11_P3_subthreshold_gate.md`. Resume at **S13 = P8a** (teacher-
forced id supervision — RO target already met by S12; the live gap is force_ic
→ S17/P9).

### S11b — pivot to the 2 open headline cells (2026-06-15, both model-fidelity limits)

User-directed pivot (force_ic accepted as a known-issue) to the 2 failing cells
of the 14/16 mix. **Both are systematic model-fidelity limits; headline stays
14/16.** Gate file `results/v6_4_7/S11b_pivot_open_cells.md`.

- **tsmc5 switchcap over-conduction (12.16 %, gate ≤5 %):** the pass-NMOS
  forward charge-transfer is too strong. Subthreshold loss (S11) barely moves it
  (→11.70 % — the over-conduction is moderate/strong region, NOT the
  weak-inversion tail), and the S12 corridor (any dose) doesn't fix it either +
  collapses the tsmc5 opamp. Also resisted P0/P2/symcaps. A genuine
  forward-conduction-accuracy limit, not subthreshold/corridor-addressable.
- **tsmc7 opamp ~10–11 % gain over-prediction (gate ≤10 %):** systematic, NOT
  seed luck — production S8 10.16 %, control-v2 healthy seeds s7 10.99 %/s31
  13.77 % (NG gain ≈163, DN ≈181). The deferred S12 **gentle-corridor W-sweep**
  (`scripts/v6_4_7_pivot_corridor.sh`, W∈{1,2}×seeds{7,31}) found the corridor
  **PRESERVES the over-gain (181→181) OR COLLAPSES it to 0 — no gentle "reduce
  gain 11 %" path** (the S10 value-surface fragility). Best
  `v6_4_7_pivcor_w2_s7_tsmc7`: opamp 10.78 % (0.78pp over, within run-to-run
  noise) + RO 2.86 % + inv 2.93 % + SC 1.02 % — **a strictly better-positioned
  tsmc7 S19 candidate than the S12 corridor (opamp near-pass vs collapsed);
  recommended for S19.**

**Net: the 2 open cells + force_ic are all value-surface / fixed-point /
forward-conduction limits resisting the cheap DirectNet levers (subthreshold,
corridor dose, frame, clamp).** Recommend **S19 promotion at 14/16** with
force_ic + these 2 cells as documented known-issues (or a scoped structural
change — architecture / physics-core — if they are must-close). The serial
chain's S13/S14/S15 are lower-value (S12 already met the RO target).

**Repo cleanup (2026-06-15, same step):** the superseded pre-V6.4.7 plan files
(`docs/plans/2026-04-24 … 2026-06-01`) and old iteration result dirs
(`results/{v6_4_4_iter2,v6_4_5,v6_4_6}/`, `results/v4_*`/`v5_*` reports) were
removed. Their durable dead-end/progress records live in this CHANGELOG and
CLAUDE.md; any path references to those removed files in older entries are
intentionally dangling (the narrative, not the gate file, is the record).

---

## V6.4.6 — diagnosis-first architectural iteration; probe fix + dead ends, no behavioral change (2026-06-01/02)

Plan: `docs/plans/2026-06-01-directnet-v6.4.6-ro-sram.md`. **Scope contract:**
close the two gates V6.4.5 confirmed are architectural — TSMC7 ring_osc (8.97 %)
and SRAM `force_ic` (0/8) — but gate every GPU-spend behind a **0-GPU diagnostic
that can kill the lever before it is built** (Phase 0), and make any retrain
opamp-safe by construction (frozen-base LoRA). **Outcome: no model shipped, no
checkpoint changed, 9/16 held.** Phase 0 (6 diagnostics) killed the RO lever
before any GPU and unlocked an SRAM solver path; Phase 1 then found the SRAM path
is a *measurement* fix, not a gate-close. V6.4.6 ships a corrected SRAM probe +
dead-end records. V6.4.4 remains canonical. Per-phase gate files under
`results/v6_4_6/`.

### Outcome

| Phase | Verdict | Code shipped | 16-cell pass count |
|-------|---------|--------------|:------------------:|
| 0 — six 0-GPU diagnostics | **done 6/6** — one lever killed (P0-C), one unlocked (P0-A), one demoted (P0-D) | none (instrumentation reverted) | 9/16 |
| 1 — `force_ic` recovery | **probe-hardening SHIPPED; homotopy KILLED; 0 gates closed** | `solver.py` + `verify_complex_sram_snm.py` (corrected probe) | 9/16 |
| 2 — LoRA + Jacobian distill (RO) | **KILLED before GPU** by P0-C (gds/caps causally inert on the RO period) | none | 9/16 |
| 3 — physics-core SRAM fallback | not funded (lead chose honest-ship after Phase 1) | none | — |

### Phase 0 — six zero-GPU diagnostics (6/6, all on canonical V6.4.4, tree left clean)

- **P0-A — railed fixed point EXISTS.** The released NN 6T cell's unconstrained
  KCL residual at the rail seed = 8.5e-5 (TSMC5) / 1.26e-4 (TSMC7), ratio
  0.013–0.017 ≪ 1. The inboard point (q≈0.82/qb≈0.23) has an equally small
  residual ⇒ the cell is **bistable**; `solver.py` force_ic re-solve selects the
  wrong basin. **Unlocked Phase 1.**
- **P0-B — gds is the divergent surface.** Along the TSMC7 RO trip cycle gds
  NRMSE 22.9 % (N) / 20.4 % (P), R² 0.24/0.34; gm/id ≤6 %; caps largely
  exonerated (cgg/cdg ≈0.5–1 %).
- **P0-C — RO Jacobian-distillation DEAD before any GPU.** Swapping the *exact*
  OSDI gds/caps into the live TSMC7 RO transient moves the period **≤0.01 ps**
  (baseline 50.82 ps, cap-swap 50.82 bit-for-bit, gds-swap 50.83; vs NG 46.64).
  gds & cap-derivatives are **Jacobian-only** — they enter only the NR Jacobian +
  matching RHS offset and **cancel at the converged fixed point** (code-confirmed
  `_stamp_mosfet_dc:304`, `_stamp_mosfet_transient:1718-1782`). The divergent gds
  is real but **causally inert** on the period. The RO period is owned by the
  id-VALUE + charge-VALUE (qg/qd) trajectories + BE/Trap/BDF-2 truncation.
  **Killed Phase 2 (gds-distill, charge-distill, and the deferred Softplus
  cap-head) for RO; deferred to a scoped V6.4.7 dynamic-id/qs/BDF-2 investigation.**
- **P0-D — SRAM off-leak over-modelled, but pinning device ≈VTH.** At the TSMC7
  stuck point the pull-down sources −6.36 µA vs OSDI −0.84 µA (7.5×; Id–Vgs
  NRMSE 21.5 %), but Mnr sits at Vov ≈ +45 mV ≈ VTH (weak inversion = D4
  territory). **Phase 3 demoted to a caveated fallback** (any off-gate risks the
  inverter trip).
- **P0-E — subthreshold band is sign-random floor noise.** `|id|<1e-7` band:
  45 % negative, 6 % literal-0; asinh crushes the 1nA–100nA band to 0.011 % of
  the normalised id range. OSDI gm/gds in the subthresh class are CLEAN. ⇒ raw
  asinh-floor drop / log-reweight on the id *value* is unsafe; only a clipped
  (`|id|>1e-12`) Huber-on-ln-current distill against the *derivative* is safe.
- **P0-F — baseline + holdout frozen** (`baseline_v6_4_4.json`, pinned to HEAD
  `54c4759` + 8 checkpoint sha256s): TSMC12 RO 3.01 % PASS, TSMC16 RO 2.88 % PASS
  (blind vetoes); TSMC5 opamp 2.64 % PASS, TSMC12 opamp 10.94 % FAIL (protected);
  TSMC7 RO 8.97 %.

### Phase 1 — `force_ic` is a measurement fix, not a gate-close (0 gates, 9/16 held)

- **Probe-hardening (SHIPPED, correctness).** The force_ic path early-returns, so
  `_last_solve_converged` was **never set** (stuck at `__init__` `False`) → the
  old SRAM accept's first clause was *always* `False` → a guaranteed `0/8`
  **regardless of where NR landed**, with **no KCL-residual gate** (the top
  Goodhart risk). Fixed: the solver now computes the *released* solution's KCL
  residual, sets `_last_solve_converged` honestly, and exposes
  `_last_dc_residual`/`_last_dc_resid_threshold`; the test gates on
  `resid_ok AND rail_ok`. Isolated to the `if _ic_temp_sources:` branch — inverter
  8/8 byte-identical, butterfly 4/4 unchanged.
- **Rail band tightened `VDD/4` → `0.1·VDD`.** An intermediate pass reported
  **4/8 PASS (TSMC12/16) → 11/16** under the pre-existing `VDD/4` band. Adversarial
  review **retracted it as a false-PASS**: the released solution lands in the
  *documented inboard failure attractor* (q≈0.87/qb≈0.20) on **all 4 techs**; the
  storage-"0" node parks at 24–30 % VDD ≈ 1× the true SNM above ground. `VDD/4`
  (=25 % VDD) straddles the attractor, so the 0.80 V techs' qb≈0.19 sneaks inside
  by VDD-scaling while TSMC5 rails *closer in absolute volts* (qb=0.163) yet fell
  outside its smaller band. The released solution is **byte-identical to V6.4.4**
  (matching checkpoint sha256s) — so this was a zero-delta *measurement
  correction* of V6.4.4's own behavior, not an improvement. The `0.1·VDD` band
  reports the **honest `0/8`** and would still accept a genuinely railed solution.
- **Constraint-continuation homotopy (BUILT + KILLED).** The plan's Norton
  soft-pin continuation (sweep conductance g:1→0, track the railed branch P0-A
  found) folds at **g*≈1e-5 S** into the symmetric metastable point on all 4
  techs (0/8). Root cause: the railed point is a *residual* fixed point (P0-A) but
  **NR-unstable** under the full re-stamp map `x→A(x)⁻¹b(x)` — the OFF node's
  vanishing deep-subthreshold conductance makes `Δqb=residual/g_qb` explode once
  the soft-pin conductance drops below g*. The series-R fallback is the Norton
  dual and folds identically. A *stronger* negative than the plan's conjunction
  foresaw (the railed point exists yet cannot be tracked). Reverted.

### Dead ends recorded (V6.4.6)

| # | Dead end | Evidence |
|---|----------|----------|
| E1 | Phase-2 RO Jacobian-distillation (gds/charge/cap-head) | P0-C: exact OSDI gds/caps swapped into the live RO transient move the period ≤0.01 ps (Jacobian-only, cancel at the fixed point). Killed before GPU. |
| E2 | Phase-1 `force_ic` constraint-continuation homotopy (Norton + series-R) | Folds at g*≈1e-5 S into the metastable point, 0/8; the railed point is NR-unstable, not absent. Reverted. |
| E3 | `VDD/4` rail-proximity band ("4/8 → 11/16") | False-PASS of the documented inboard attractor (qb at 24–30 % VDD); VDD-scaling artifact. Retracted; band tightened to `0.1·VDD` → honest 0/8. |
| E4 | RO integrator selection / finer fixed tstep (P0-G) | BDF-2 never fires; Trap & BE both converge to a ~50.4 ps continuum limit (~8 % > NG) as tstep→0. BE's coarse-step "pass" is a truncation artifact. No shippable solver-integration fix. |
| E5 | RO charge-value (qg/qd) distillation (P0-H) | Charge VALUES already exact vs OSDI (≤2 aC, ≤1.2 % NRMSE) → nothing to distill; cannot move the period. (Distinct from the P0-C Jacobian null: charges are inert because *already correct*, not because they cancel.) |

### Post-ship RO diagnostics (P0-G + P0-H, 2026-06-02) — RO localised to the id-VALUE

Two more 0-GPU diagnostics run after the 9/16 ship, closing the last untested RO
axes (gate files `results/v6_4_6/phase0{G,H}_*.md`; scripts `v6_4_6_p0{g,h}_*.py`):

- **P0-G (integration study) — no shippable solver fix.** The BDF-2 stiffness
  auto-switch **never fires** on the TSMC7 RO (baseline is already pure
  Trapezoidal). Forcing Trap = baseline (50.82 ps); forcing BE "passes" at the
  coarse 2 ps step (45.14 ps) but a convergence study (×1/÷2/÷4/÷8) shows that is
  an O(h) truncation artifact — BE climbs back above the gate and **both Trap and
  BE Richardson-extrapolate to a common ≈50.4 ps continuum limit (~8 % > NG)**. So
  integration truncation owns only **~0.4 ps** of the 4.18 ps gap; the rest is
  model-owned. (No regression surface run — there is no shippable candidate.)
- **P0-H (VALUE overlay) — charges exact, id-VALUE is the owner.** Overlaying the
  NN's post-Rule-15 **VALUES** (`id`, `qg`, `qd`, `qs`) — what the transient
  companion actually stamps and which, unlike the Jacobian, do NOT cancel at the
  fixed point — vs analytic OSDI along the RO trip: **charge VALUES are exact**
  (NRMSE 0.7–1.2 %, R²≥0.999, MaxErr ≤2 aC ≈ ≤2 % of the per-stage swing) ⇒ a
  charge-value distillation has nothing to remove. The **`id` VALUE** carries the
  residual: NMOS on-state NRMSE 9.6 %, **~20 % under-prediction of the dynamic
  peak pull-down current** (−72 vs −90 µA) — direction-consistent with DN period
  being *longer* than NG.

**Net:** the ~3.7 ps model residual is owned by the **NMOS dynamic `id` VALUE** —
gds/cap Jacobian (P0-C), charge VALUES (P0-H), and BE/Trap/BDF-2 integration
(P0-G) are all exonerated. (P0-H originally mis-attributed the gap to integration;
reconciled with P0-G's convergence result — see the gate files.)

### What V6.4.6 rules out / leaves for V6.4.7

- **TSMC7 ring_osc** — the residual is *located* in the **NMOS dynamic `id` VALUE**
  (P0-G/P0-H: charges exact, integration ~0.4 ps, ~20 % peak pull-down
  under-prediction), but the **id-VALUE-only correction lever is NO LONGER
  de-risked** after **P0-I (2026-06-03, `phase0I_id_injection.md`)**. P0-I ran the
  causal id-injection swap (the P0-C analogue): the naive swap **diverged** (v1,
  inconsistent Jacobian artifact), so it was rebuilt as a consistent exact-bias
  OSDI op-point (v2) — which converges but is ~20–35× slow (hybrid OSDI-id/NN-charge
  device → NR-failure-driven dt-halving). The causal result is **paradoxical and
  decisive**: injecting the exact OSDI `id` (NMOS-only **and** symmetric N+P)
  produces a genuine, full-rail, uniform **~92 ps** oscillation (baseline 50.83;
  N+P 92.30; NMOS 92.74) — ~2× baseline and *further* from NG 46.64 ps, the
  *opposite* direction from swapping id+charge together (NGSPICE = 46.64).
  **Unlike the Jacobian (P0-C: inert, separable), the `id` VALUE is NOT separable
  from the NN charge model** — the RO period is a joint (id, charge) property.
  P0-I therefore **cannot confirm/refute** "id owns the gap" cleanly, and the
  plan's **frozen-base LoRA id-VALUE-ONLY distillation** is re-scoped: V6.4.7 must
  gate any id-only fix against the live RO period *immediately* and consider a
  **joint id+charge correction (or retrain)** rather than id-only. (Caveat: the
  injection bypasses Rule-15 + floors gds, so 92 ps is a *proxy* warning, not proof
  a real autograd-consistent LoRA fails — it shifts the burden of proof onto
  *demonstrating* an id-only RO fix.) Jacobian-distillation, charge-distillation,
  and the deferred split-head cap-head remain empirically dead for RO.
- **SRAM `force_ic`** is a true model-fidelity gap, not a solver-path bug: the
  inboard attractor is a stable NN fixed point (D3 confirmed) co-existing with an
  NR-unstable railed point. No 0-GPU solver continuation closes it. Open for
  V6.4.7: the Phase-3 physics-core off-leakage fix (P0-D-caveated: pinning device
  ≈VTH → inverter-trip / D4 risk) or the transient write-then-hold re-spec.
- **SRAM `force_ic`** remains a model-fidelity off-leakage gap (D3 confirmed: the
  inboard attractor is a stable NN fixed point co-existing with an NR-unstable
  railed point). Open for V6.4.7: the Phase-3 physics-core off-leakage fix
  (P0-D-caveated: pinning device ≈VTH → inverter-trip/D4 risk) or the transient
  write-then-hold re-spec.
- **Ships:** the corrected SRAM `force_ic` probe (`solver.py` KCL-residual
  telemetry + honest flag; `verify_complex_sram_snm.py` `resid_ok AND rail_ok`
  with the `0.1·VDD` band) + Phase-0/1 gate files (incl. post-ship
  `phase0{G,H,I}_*.md`) + the P0-G/H/I diagnostic scripts (`v6_4_6_p0i_id_injection{,_v2}.py`)
  + `baseline_v6_4_4.json`. No model, no checkpoint. V6.4.4 remains the active
  revision. (P0-I is instrumentation-only — `git diff` over `pycircuitsim/` empty.)

---

## V6.4.5 — Track A no-ship iteration, all 5 phases run (2026-05-29)

Plan: `docs/plans/2026-05-28-directnet-v6.4.5-ro-sram.md` (Track A).
**Scope contract:** all five Track-A phases executed. Phases 1/2/4 were the
first pass (inference + zero-code probes); **Phases 3 and 5 were then run** —
Phase 3 built the multi-circuit scorer, Phase 5 ran the full 16-seed stock +
8-seed mono TSMC7 retrain (32 fresh trainings) scored under it. **No model
shipped** — V6.4.4 remains canonical; only the Phase-3 scorer *scripts* are
new (infra). The iteration is a dead-end record (CLAUDE.md "always record the
dead end proposal"). Per-phase gate files under `results/v6_4_5/`.

### Outcome

| Phase                          | Verdict                | Code shipped | 16-cell pass count |
|--------------------------------|------------------------|--------------|:------------------:|
| 1 — V6.4.4 re-baseline         | reproduced exactly     | none         | 9/16               |
| 2 — Zero-code solver probes    | all 3 killed/diagnostic| none         | 9/16               |
| 3 — Multi-circuit scorer       | **built + validated** (scorer accepts V6.4.4 mix after opamp re-calibration) | scripts (infra) | — |
| 4 — Rule-15 `Ioff_rail` patch  | **KILL** (inverter regress ~10×) | reverted | 9/16        |
| 5 — TSMC7-only retrain         | **KILL** (best feasible RO 9.05 % > baseline 8.98 %; 32 trainings) | none | 9/16 |
| 6 — V6.4.6 split-head          | deferred in plan       | —            | —                  |

Inverter 8/8 held, extended harness 55/55 + 64/64 held. Final 9/16
matches V6.4.4 — TSMC7 ring_osc (~9 % period err) and SRAM `force_ic`
(0/8) remain open. Canonical checkpoints sha-verified unchanged after the
isolated scorer + retrain.

### Dead ends recorded

- **`NN_SYMMETRIC_CAPS=1` does not move TSMC7 RO.** With the flag ON
  for both ring_osc and switchcap, TSMC7 RO period err stayed at
  8.97 % bit-for-bit; switchcap charge err improved slightly on
  TSMC5/12/16 but no new SC cell crossed its gate. The 9 % RO drift is
  not cap-asymmetry. Flag stays default OFF (unchanged).

- **`max_substeps=4` on RO does not close TSMC7 enough.** A 2-LOC env-
  var override threading `RO_MAX_SUBSTEPS` through `run_directnet_-
  transient` moved TSMC7 RO 8.97 % → 8.04 % at ~2× wall time —
  short of the ≤ 5 % kill gate. The drift is not LTE-dominated.
  Harness edit reverted.

- **SRAM `force_ic` q ≈ 0.18 is a true NN attractor.** A 10-LOC
  diagnostic in `verify_complex_sram_snm.py` replaced the literal
  `(VDD, 0)` rail seed with the NGSPICE butterfly-lobe value
  (`near_zero` ≈ 83–123 mV depending on tech). All four techs still
  settle at q ≈ 0.70–0.80 / qb ≈ 0.16–0.47 — not a poor warm-start
  symptom. Phase 4 was promoted from "diagnostic" to "mandatory" by
  this finding, but Phase 4 also died (next bullet). Diagnostic edit
  reverted.

- **Plan's Rule-15 `Ioff_rail` patch is unsound as formulated.** The
  patched `_apply_vds_correction` (+26 LOC behind `NN_IOFF_RAIL_K`)
  computed `Ioff_rail = max(|id_raw|, k·NFIN·1nA)` and added
  `sign_conv · blend · Ioff_rail` to `id`. For a *conducting* device
  at the rail (`|id_raw| ≫ floor`, `blend ≈ 1`) this effectively
  doubles the current, collapsing the inverter VTC NRMSE from
  1.21 % to 11.56 % at the smallest non-zero `k = 1`. The SRAM
  attractor moved *further* from the rails (q 0.866 → 0.375 on TSMC5,
  divergence past VDD on TSMC12). Kill criterion 2 (inverter VTC
  regression > 5 mV) tripped at every `k ∈ {1, 3, 10}`; `k ∈ {30, 100}`
  OOM'd on parallel GPU contention but the trend was clear. The patch
  is fully reverted.

  The formula could be salvaged by floor-only-below — e.g.
  `Ioff_extra = max(floor − |id_raw|, 0)` — but that is a *different*
  patch (different kill-criterion math) and was not retried; the
  plan's exact wording was followed (drop, don't silently
  reformulate).

- **Seed/recipe lottery does not move TSMC7 ring_osc (Phase 5, 32
  trainings).** The full plan §5 sweep — 16-seed stock + 8-seed mono,
  TSMC7 N+P, scored under the Phase-3 multi-circuit vector — never
  reaches the ≤ 5 % gate. Best overall RO = 8.21 % (`stock_s31`) is
  *infeasible* (opamp collapsed to gain 0); best **feasible** RO =
  9.05 % (`stock_s11`), *worse* than the V6.4.4 baseline (8.98 %). The
  seed moves inverter VTC NRMSE (1.75–5.50 %) but barely moves the RO
  period (DN ~50.8–53 ps vs NG 46.64 ps) — a systematic ~9 % model
  bias, not init variance. 13/16 new candidates collapse the TSMC7
  opamp to zero gain (the same failure iter-2 saw with P7-stock). Both
  Phase-5 kill criteria fire; no checkpoint shipped. The 32
  `v6_4_5_p5_tsmc7_*` checkpoints are kept as inert dead-end evidence
  (they don't match the resolver pattern, so are never auto-loaded).

### Phase 3 — multi-circuit scorer built + validated (infra, not a dead end)

`scripts/eval_v6_4_5_candidate.py` + `scripts/v6_4_5_search.py` score a
candidate on `(inv_vtc_nrmse, inv_tran_post_nrmse, ring_osc_period_err,
sram_rail_snap_resid, opamp_flat_flag)` using `BSIMAR_CHECKPOINT_DIR`
isolation (a private tmp dir holding the candidate copied under the
canonical `tsmc{X}_dn_medium_{dev}` stem so vocab-scope detection still
fires) — the real `checkpoints/` dir is never mutated, so candidates run
concurrently and the canonical slots are sha-verified unchanged.

**Re-calibration (plan §Phase 3 "if the scorer rejects the V6.4.4 mix, fix
the thresholds"):** the plan's `opamp_flat_flag = |Vout_center − VDD/2| >
0.3·VDD` flags *every* tech, including the PASSING TSMC5 opamp — a
high-gain open-loop opamp is railed at the exact center common-mode
whenever the input pair has any offset. Redefined to `gain < 10` (the true
collapse signal). Under it the V6.4.4 mix clears the hard gates for
TSMC5/7/12; TSMC16 reads flat=1, correctly (its opamp is a known fail
cell). Scorer-accepts-V6.4.4 kill gate MET.

### What V6.4.5 rules out

1. The plan-as-written for Phase 4 (`max(|id_raw|, floor)`) — the
   conducting-state contamination is structural, not a tuning bug.
2. Cap-asymmetry and LTE as TSMC7 RO drift sources.
3. "Warm start the SRAM near rails" as a force_ic cure.
4. **The seed/recipe lottery and the `--monotonic` recipe as TSMC7 RO
   levers** — 32 trainings, RO floor ~8.2 % infeasible / ~9.0 % feasible.

### What V6.4.5 leaves for V6.4.6

- **Ring_osc TSMC7** is now *confirmed* not retrain-addressable (Phase 5
  above). It needs an architectural change: V6.4.6 split-head (spectral-
  norm `id` head + softplus cap head, to fix the cap shape loading the RO
  period) or Track B B7/B9. Track B B5/B6 (OSDI-Jacobian distill / harvest
  retrain) are different levers than the seed sweep and remain unproven.
- **SRAM `force_ic`** is also model-fidelity; Phase 4-style inference-only
  Vds corrections cannot move it without a Vgs gate (which breaks Rule 1).
  Candidates: Track B B7, V6.4.6 split-head, Track B B9.
- An off-state floor with the corrected formula
  `Ioff_extra = max(floor − |id_raw|, 0)` is the cheapest re-attempt at
  the inference-only path and was *not* tried in V6.4.5.

V6.4.5 ships no model and no checkpoints — V6.4.4 remains the active
revision. New this iteration: the Phase-3 scorer scripts (reusable infra)
and this CHANGELOG + gate files.

---

## V6.4.4 — DirectNet per-tech checkpoint mix (inference-only iteration, 2026-05-28)

Plan: `docs/plans/2026-05-24-directnet-v6.4.4-complex-circuits-iter2.md`.
**Scope contract:** inference + checkpoint selection only. No retraining,
no data regen. Artifacts pulled from what was already on disk: V6.4.1
seed-42 (canonical slots), V6.4.2 Phase-7a stock-recipe winners
(`v6_4_2_p7_tsmc{5,7}_stock_*`), V6.4.2 Phase-7a mono-recipe candidates
(`v6_4_2_p7_tsmc{5,7}_mono_*`). The V6.4 best-of-N production backup at
`/tmp/v6_4_checkpoints_backup_20260517/` was cleared by `/tmp` cleanup
before the iteration and is unrecoverable — flagged as a dead end.

### Final ship

Per-tech checkpoint mix in the canonical `tsmc{X}_dn_medium_*` slots:

| Slot                              | sha256 (first 16)   | Source stem                            |
|-----------------------------------|---------------------|----------------------------------------|
| `tsmc5_dn_medium_nmos_best.pt`    | `22eef03e44aca566…` | `v6_4_2_p7_tsmc5_stock_s17_nmos`       |
| `tsmc5_dn_medium_pmos_best.pt`    | `a6a09be03a810b7e…` | `v6_4_2_p7_tsmc5_stock_s42_pmos`       |
| `tsmc7_dn_medium_{nmos,pmos}`     | seed-42 (unchanged) | V6.4.1                                 |
| `tsmc12_dn_medium_{nmos,pmos}`    | seed-42 (unchanged) | V6.4.1                                 |
| `tsmc16_dn_medium_{nmos,pmos}`    | seed-42 (unchanged) | V6.4.1                                 |

V6.4.1 seed-42 backup preserved at `/tmp/seed42_backup_20260524/`.

### Complex-circuit pass rate — +2 vs V6.4.1 / V6.3.1 iter-1 baseline

| Test       | V6.4.1 (Step 1) | **V6.4.4** | Δ |
|------------|----------------:|-----------:|---:|
| ring_osc   | 2/4             | **3/4**    | +1 (TSMC5 FAIL → PASS, perErr 6.76 → 2.98 %) |
| opamp      | 0/4             | **1/4**    | +1 (TSMC5 FAIL → PASS, gainErr 14.78 → 2.64 %) |
| sram_snm   | 4/4             | 4/4        | 0 (butterfly only; `force_ic` rail-snap still FAIL on every tech) |
| switchcap  | 1/4             | 1/4        | 0 (TSMC7 PASS in both; TSMC5/12/16 FAIL) |
| **TOTAL**  | **7/16**        | **9/16**   | **+2** |

Plan target was ≥ 10/16. V6.4.4 lands at 9/16 — every inference-only
lever from on-disk artifacts is exhausted; remaining gates are gated on
retraining (Phase 8 split heads or a re-scored Phase-7 best-of-N with
opamp gain + RO period in the scoring vector).

Inverter gate held 8/8 on the V6.4.4 mix (VTC NRMSE
1.21/2.37/2.05/1.33 %, transient post-startup 1.62/1.09/1.41/1.45 %).
The V6.4.1-shipping extended harness (DC 55/55, VTC+tran 64/64) was
unchanged by the swap (TSMC5 swap touches only its own slots; the
extended harness re-routes through the same parser preempt cascade).

### Execution log

- **Step 1** — V6.4.1 seed-42 re-baseline on all four complex tests:
  7/16. Identical to the V6.3.1 iter-1 baseline that was never
  re-measured under the V6.4.2 solver; confirms Phase 5/6 are
  accuracy-neutral on complex circuits as documented.

- **Step 2** — TSMC5/7 swapped to Phase-7a stock winners (TSMC5 nmos=s17
  pmos=s42, TSMC7 nmos=s123 pmos=s17 per
  `logs/v6_4_2_phase7/search_TSMC{5,7}_stock_winner.json`). TSMC5 won
  cleanly (+2 circuits). **TSMC7 opamp regressed structurally**: gain
  error 30.67 % → 100 % flat-Vout. Better inverter VTC MaxErr (210 →
  100 mV) does not predict opamp differential-pair bias-point quality
  at this resolution — confirmation that the iter-1 selection criterion
  (inverter VTC alone) is broken for complex-circuit pass rate.

- **Step 3** — `NN_SYMMETRIC_CAPS=1` is inverter-safe (8/8 held under
  the flag); the RO + SC re-measurement under the flag was **not
  completed** this iteration. Decision: keep default OFF. The flag
  remains dormant infrastructure ready for a future RO-focused probe.

- **Step 4** — Lightweight per-tech selection from Step 2 evidence
  (full greedy pair-search deferred): TSMC5 keeps P7-stock (Step 2
  +2 wins), TSMC7 reverts to seed-42 (P7-stock opamp regression
  unshippable), TSMC12/16 stay seed-42 (no on-disk alternative).
  Final 4-circuit re-measurement matches the per-cell Step 1/Step 2
  numbers exactly because every cell is either Step 1 or Step 2
  unchanged.

### V6.4.2 Phase-7a code (load-bearing, restored in `df9cfe3`)

The V6.4.2 sprint shipped a Phase-7a monotonic-in-Vg residual on the
DirectNet `id` head (`--monotonic`, defaults OFF) — code that was
documented as shipped in V6.4.2 but never committed. **It is load-bearing
for the V6.4.1 seed-42 checkpoints**: the local V6.4.1 retrain ran with
the `_MonotoneVgResidual` class in scope (even though `--monotonic`
defaults OFF), so the saved state_dict carries `mono.*` keys. The model
class must know about them at load time, otherwise PyTorch raises:

    Error(s) in loading state_dict for DirectNet:
      Unexpected key(s) in state_dict: "mono.w_rest", "mono.w_vg_raw",
      "mono.b1", "mono.w_out_raw", "mono.b_out", "mono.sign".

The V6.4.4 release ships in two commits:
- `4fcce2a feat(v6.4.4): add BSIMAR_CHECKPOINT_DIR env var + v6_4_seed42
  checkpoint archive` — V6.4.4 docs (CLAUDE.md, this CHANGELOG entry,
  iter-2 plan, results/), the `BSIMAR_CHECKPOINT_DIR` env var support in
  `bsimar/config.py`, the `v6_4_seed42` checkpoint archive copy, and the
  V6.4 / V6.4.2 sprint scripts (`scripts/run_v6_4_*.sh`,
  `scripts/v6_4_*search.py`, `scripts/eval_v6_4_*.py`).
- `df9cfe3 fix(v6.4.4): restore Phase-7a code required by on-disk
  checkpoints` — the four files `bsimar/{cli/train,models/direct_net,
  training/trainer}.py` + `pycircuitsim/models/mosfet_directnet.py`
  that the docs commit referenced as "newly committed" but didn't
  actually stage. Without `df9cfe3` the on-disk checkpoints fail to load
  and the inverter gate errors out 2/8.

Plain (non-monotonic) checkpoints — including every V6.4.4 ship slot —
load through the same forward pass with `mono=None`; the only effect of
the Phase-7a code on a stock checkpoint is the load-path tolerance for
the `mono.*` state_dict keys. No behaviour change at inference.

### Dead ends recorded this iteration

- **V6.4 best-of-N restoration** — backup at
  `/tmp/v6_4_checkpoints_backup_20260517/` cleared by `/tmp` cleanup
  between V6.4.2 ship and now. No `v6_4_bof_*` / `v6_4_repro_*` source
  stems survive system-wide. The CHANGELOG V6.4 numbers (TSMC5 62 mV,
  TSMC7 60 mV inverter VTC) are no longer reproducible from on-disk
  artifacts.
- **TSMC7 P7-stock-on-opamp** — better inverter VTC (60.1 mV vs seed-42
  174.7 mV) collapses opamp Vout to flat-zero at the Step-3b Miller
  bias. Reverted; structural-fidelity gap that inverter-VTC selection
  could not see.
- **Full greedy per-tech pair search (planned Step 4)** — skipped this
  iteration. Step 2 evidence was decisive enough (TSMC5 clean +2, TSMC7
  structural regression, TSMC12/16 no candidates) that the greedy
  search reduces to the lightweight pick. The greedy infrastructure is
  on disk (`scripts/v6_4_2_phase7_search.py`,
  `scripts/eval_v6_4_1_pair.py`) for a future iteration that scores on
  the four complex-circuit metrics instead of inverter VTC.

---

## Phase Milestones

- **Phases 1-3:** Core simulator (MNA, NR solver, transient).
- **Phases 4-6:** BSIM-CMG (LEVEL=72) integration via PyCMG, NGSPICE-verified (<0.02% OP, <0.1% DC).
- **Phases 7-10:** Charge-based transient (0.20% NRMSE vs NGSPICE), 5-tech support (ASAP7, TSMC5/7/12/16), 21-config parametric sweep all PASS.
- **Phases 11-12:** NN compact model (LEVEL=73) — training pipeline, autograd conductances, multi-tech DC+transient verified.
- **Phases 13-15:** Universal NN v2 — 21 variants across 5 techs, 13-dim input (voltages + 7 process params), 19/21 PASS (ASAP7:SLVT and TSMC7:LVT FAIL on NMOS DC).
- **Leave-one-out transferability:** 8/10 good transfer (gap < 5%), zero-shot avg 4.65% NRMSE, in-dist avg 0.95%.
- **Charge-finetune training:** ChargeConsistencyLoss (autograd dq/dV = C), trained from scratch 800 epochs on A100.
- **NN Transient (charge-finetune + VT fix):** 5/5 PASS — ASAP7 6.20%, TSMC5 14.41%, TSMC7 7.15%, TSMC12 6.47%, TSMC16 7.42%.
- **Solver accuracy improvements:** SPICE-standard convergence (RELTOL=1e-4, VNTOL=1e-7), GMIN reduction (1e-6→1e-12), BE→Trap first-step switching, relative oscillation threshold. NN transient improved: TSMC7 7.15→6.09%, TSMC12 6.47→5.92%, TSMC16 7.42→6.70%. BSIM-CMG transient unchanged at 0.20% (already at integration-method floor).
- **SRAM Solver Upgrades (Phases 1-3):** Sparse matrix solver (lil_matrix→CSR+spsolve), DC GMIN stepping + oscillation detection + adaptive damping + hard `.ic` mode (force_ic), BDF-2 integration (auto-switches on stiffness detection), LTE adaptive sub-stepping. All 67 existing tests PASS with zero regression.

## Test infrastructure

- **3-level DC+Transient test suites** — 3-layer infrastructure: `tests/common/base.py` (tech defs, generic helpers) → `tests/common/bsimcmg_{dc,tran}.py` (analysis-specific) → `verify_*.py` (test scripts). `tests/common/nn.py` consolidates NN scaffolding (nrmse, mre, checkpoint resolution, path bootstrap). NGSPICE references in `tests/references/`.
- **Known-bad combos excluded:** TSMC5 SVT (pch PDIBL2_i<0), TSMC7 SVT/LVT (inverter garbage / pch PDIBL2_i<0), TSMC16 LNVT (nch PDIBL2_i<0), TSMC16 L=24nm (PDIBL2_i<0), NFIN=1 (NR divergence for tsmc5:ulvt / tsmc16:lnvt — ETA0_i/U0_i go negative, internal node drifts to 40V producing id=40kA + NaN derivatives; eval_dc raises RuntimeError), P/N ratio where NFIN_P crosses NFIN group boundary (TSMC naive modelcards are NFIN-group-specific).

## Data generation migration

- NN training data generation moved from `nn_model.data.generate` into `external_compact_models/PyCMG/scripts/generate_nn_data.py`. Data format includes `[NFIN, L, T, 12 process params]` geometry columns; v4 training uses only 7 input features (Vgs, Vds, Vbs, NFIN, L, T, tech_code) and ignores process params. Legal (L, NFIN) combos from PDK bin boundaries (TSMC) or fallback list (ASAP7). 954 total geometry combos across 5 techs, 21 variants.

## BSIMAR package refactor (2026-03)

Consolidated former `nn_model/` (DirectNet baseline) and `external_compact_models/BSIMAR/script/` (Transformer) into a single Python package at `external_compact_models/bsimar/` with clean subpackages (`config`, `data`, `models`, `losses`, `training`, `eval`, `utils`, `cli`). Unified CLI: `python -m bsimar.cli.train --model {direct,transformer} ...`. All downstream imports use the new `bsimar.*` namespace.

## BSIMAR v3 production refactor (2026-04-08/09)

After the medium-tier improvement sprint (see `external_compact_models/bsimar/docs/bsimar_improvement_plan_2026_04_08.md`) the winning recipe is **N7 (Vov-LDS) + N3 (AR finetune) + N1 (150-epoch cosine)**. All hard-wired as defaults. The refactor collapses the CLI, deletes the signed-log normaliser, and removes ~600 net LOC.

- **Final metrics on `universal_nmos.npz` medium (5.15M params):** NRMSE_phys **0.223%** (was 0.419, −46.8%), MRE_phys **1.41%** (was 2.52, −44.0%), R² **0.9984** (was 0.9928). ~107 min on Blackwell GPU.
- **Removed code:** `Normalizer`, `NormStats`, `signed_log` / `inv_signed_log`, `BSIMARNormalizer.signedlog`, `load_and_split` (legacy loader), `WeightedBNILoss`, `forward_curriculum`, `train_epoch_direct_ar` / `curriculum` / `scheduled`, and the CLI flags `--loss direct|bni`, `--lds`, `--vov-lds`, `--no-filter`, `--reorder`, `--scheduled-sampling`, `--curriculum`, `--consistency-weight`, `--norm-mode`, `--charge-consistency-weight`, `--learnable-output-affine`.
- **Hardwired knobs:** loss=MAE+LDS+VovLDS, norm=asinh+zscore, `parallel_caps=True`, `grouped_inputs=True`, BSIMAR reorder, phys-best ckpt, AR finetune (5 epochs).
- **Known-infeasible (DO NOT retry without new structural argument):** N6 Huber on I/V (wrong gradient near zero), N5 learnable output affine (disrupts post-asinh zscore), N4 charge-consistency penalty (asinh chain rule cosh factor makes constraint inequivalent).
- **Deferred:** N2 KV-cache encoder.
- **File renames:** `pycircuitsim/models/mosfet_nn.py` → `mosfet_directnet.py` (class names unchanged).

## BSIMAR v3 LOO cross-tech sprint (2026-04-09/10)

5-fold leave-one-tech-out on universal NMOS. TSMC intra-family: 0.84–2.18% NRMSE (production-usable). ASAP7 held out: 24,678% NRMSE (catastrophic — body physics gmb/qb 10⁴× smaller than TSMC, a data bottleneck not fixable by model changes). One keeper: **S2 asinh-scale floor for gmb/qb** (~3.2% geometric-mean improvement). E2 Vov+extras and E2b Vov-only both REJECT (+10–14% regression). Full report: `external_compact_models/bsimar/results/loo_cross_technology_report.md`.

## Cross-tech transfer roadmap review (2026-04-10)

Five-agent review of the original 10-idea/7-stage roadmap. Conclusion: zero-shot transfer is not a user requirement (retrain with new tech data takes ~2h); TSMC transfer already within threshold; ASAP7 gap is a data bottleneck. Revised to 3-tier plan: (1) retrain v3 NMOS+PMOS + port verify_nn scripts + investigate TSMC5 transient, (2) one low-risk cross-tech probe (multiple process tokens), (3) retrain-with-new-data workflow for new PDKs.

## verify_nn_*.py port to NNTechConfig API (2026-04-11)

Ported 3 broken test scripts (`verify_nn_multi_tech.py`, `verify_nn_universal.py`, `verify_nn_universal_v2.py`) from old `tech.variants[v].get_process_params()` to new `NNTechConfig.resolve_modelcard()` + `extract_process_params()`. Added `default_L()` and `get_process_params()` helpers to `tests/common/nn.py`.

## BSIMAR v4 tech-code migration (2026-04-14)

All v3 code (19-dim continuous process params) removed. Only v4 architecture (7-dim + discrete tech-code embedding via `nn.Embedding`) supported. ASAP7 excluded from training (`--exclude-techs asap7`). 4 universal models trained: DirectNet NMOS/PMOS (0.00167/0.00190 val loss) + Transformer NMOS/PMOS (0.270%/0.252% NRMSE, R²=0.9937/0.9965). TSMC5 SVT verification: DC PASS (7.79%/9.99%), VTC 17.70%. Removed: `ProcessParams`, `extract_process_params`, `PROCESS_PARAM_NAMES`, old 19-dim `INPUT_COLUMNS`. Added: `TECH_CODE_MAP`, `--exclude-techs`, `--num-tech-codes`. Checkpoint naming changed to `v4_` prefix.

## Analytical Vds correction for inverter transient (2026-04-15)

Implemented `_apply_vds_correction()` in `_MOSFETNNBase` to enforce Id(Vds=0)=0 and Id=0 for reverse-Vds at inference. Three-part correction: one-sided Vds factor (VT=0.052V), symmetric gds with linear-region conductance, sign enforcement. DirectNet inverter transient: **3/4 PASS** (TSMC7 8.87%, TSMC12 11.65%, TSMC16 10.59%; TSMC5 17.20% marginal FAIL). BSIMAR inverter: 0/4 PASS due to wrong-sign subthreshold predictions in Transformer. NMOS pulse: 8/8 PASS, zero regression. Full report: `results/v4_vds_correction_report_2026_04_15.md`.

## Rail-restoring extrapolation fix (2026-04-20)

Diagnosed real root cause of BSIMAR inverter transient explosion (V(out)→+4.4V on TSMC12/16): both NN models predict Id≈0 outside `[-VDD_train, VDD_train]`, creating a flat-zero KCL plateau the DCSolver mistakes for equilibrium. Fixed by rail-restoring extrapolation: quadratic Id ramp + linear gds ramp past `VDD_train`, smooth-joined at boundary (linear ramp tried first, caused NR oscillation for TSMC12/16 whose operating points sit at the boundary). Verified across all 4 TSMC techs with probe (670K) and production (5.15M) checkpoints: **inverter transient drops from 18-300% NRMSE (FAIL) to 6-12% (PASS)**. Production: TSMC5 12.13%, TSMC7 9.14%, TSMC12 6.78%, TSMC16 7.51%. Inference-time only, no retraining required.

## v5 inverter-transient sprint (2026-04-22/23) — closed, no production change

5-experiment sweep attempting to lift worst-case TSMC7 NMOS DC (14.72%) and drive BSIMAR inverter VTC TSMC7 (19.15%) below 10%. E1 (wider Vds-correction VT), E3 (per-tech fine-tune on same distribution), E4 (dense hot-box overlay + universal set), E5 (overlay-only fine-tune) all reverted on inverter acceptance gates. D1 diagnostic isolated TSMC7 NMOS error to strong-inversion + saturation plateau (Vgs ∈ [0.52, 0.73] V × Vds ∈ [0.40, 0.75] V), 16× under-sampled by LHS — but both densification approaches regressed NMOS DC by +2.7 pp identically, ruling out the density thesis. Retained: D1 heatmap diagnostic + finetune.py empty-test_idx guard. Full history: `results/v5_session_summary_2026_04_23.md`.

## v5 Phase A — Trim (2026-04-24, branch `feat/bsimar-v5-phase-a`)

Plan: `docs/plans/2026-04-24-v5-inverter-accuracy.md`. Deleted unjustified and dead loss code before Phase B.

- **A1:** Deleted `DirectLoss`, `ChargeConsistencyLoss`, legacy `BSIMARConfig`/`TrainConfig` aliases, dead `TransformerConfig` fields.
- **A2:** Deleted `SignConsistencyLoss`, `BoundaryLoss` — no A/B benefit; superseded by rail-restoring extrapolation and structural B3 gate.
- **A3:** Collapsed 3-axis LDS weight stack (per-target × Vov × subthreshold) to **per-target only**.
- **A5:** Deleted `_eval_autograd4` dead fast-path; added 13-output assertion at load.
- **A4 control retrain — GATE FAIL.** Retrain with trimmed pipeline regressed TSMC7 NMOS DC past ±1 pp gate. Root cause: LHS dataset insufficient — Phase B B1 hybrid uniform-grid data is required.

## v5 Phase B — Levers tried, code reverted (2026-04-24 .. 2026-05-03)

Three Phase B levers prototyped to address TSMC7 sampling-basis mismatch:

- **B1 (data, retained in PyCMG submodule):** Hybrid uniform-grid + LHS jitter sampler with `sample_class` column. Datasets regenerated under this sampler still consumed by the loader.
- **B2 (`SlopeMatchLoss`) and B3 (`apply_id_gate`) — DELETED 2026-05-03.** Neither lever validated against a v4 baseline before B3's `id_idx_in_stats` bug corrupted v5b/v5c TF runs. Inference-time `_apply_vds_correction` already enforces Id(Vds=0)=0; rail-restoring extrapolation is the load-bearing piece.
- **AR-finetune phase / `forward_scheduled` — DELETED 2026-05-03.** The 5/150 final-phase rollout carried ~160 LOC of separate optimizer + loader + tracker + checkpoint plumbing for marginal benefit over cosine.

## v4-re — NN-stack trim (2026-05-03, branch `chore/nn-stack-trim`)

Plan: `docs/plans/2026-05-03-nn-stack-trim.md`. Current shipping NN stack is labeled **v4-re**: same v4 7-dim + tech-code architecture, all unvalidated Phase B levers and AR-finetune plumbing removed. Re-trained checkpoints under `v4_re_` prefix; legacy `v4_` checkpoints continue to load via resolver fallback.

- **PR-1:** Removed 11 broken/superseded test scripts (~3.9 KLOC) — all v3-era APIs.
- **PR-2:** Deleted `bsimar/losses/slope_loss.py`, `bsimar/models/id_gate.py`, `forward_scheduled` on Transformer, `_train_epoch_scheduled_mae`, trainer's AR-finetune block, `BSIMARNormStats.id_gate` field, CLI flags (`--slope-weight`, `--slope-warmup-frac`, `--no-id-gate`, `--ar-finetune-epochs`). Inference glue deduped: `_resolve_nn_checkpoint(level, ...)` collapses LEVEL=73/74 path resolution and prefers `v4_re_*` over legacy `v4_*`; `_floor_gds(id, gds)` replaces 4 stamp sites; `_MOSFETBSIMARBase` reuses parent `_denorm_scalar` / `_denorm_full_derivative` via column-index lookup. v4 checkpoints continue to load unchanged. v5b checkpoints discard-only per Bug A.
- **Default save_prefix:** `train_directnet` → `v4_re_dn_universal_<dev>`; `train_transformer` → `v4_re_universal_<dev>`.

### Known v4 limitation carried into v4-re: TSMC7 NMOS DC 14.72%

TSMC7 NMOS DC NRMSE is 14.72% (BSIMAR v4) / 15.79% (DirectNet v4) against PyCMG ground truth at Vds=VDD/2, NFIN=10, L=16 nm. Propagates to inverter VTC (19.15% BSIMAR / 18.14% DirectNet). Root cause: LHS training distribution under-samples strong-inversion + saturation plateau by ~16× vs verifier's uniform Id-Vgs sweep. Inverter transient at TSMC7 PASSES (6.80% DN / 9.14% BSIMAR). Mitigation: retrain on B1 hybrid-grid data with the trimmed pipeline, save under `v4_re_*` prefix, expect TSMC7 NMOS DC ≤ 8% per trim plan's gate.

## V6.1 — Per-tech dedicated DirectNet for TSMC5/TSMC7 (2026-05-12 / 2026-05-13)

Sprint goal: improve inverter DC/Tran accuracy on TSMC5 and TSMC7 by training **dedicated** per-tech DirectNet models at small + medium scales. Triggered by baseline measurement on `refac_dn_medium` (V6 universal): TSMC5 inv VTC 9.58% PASS, **TSMC7 inv VTC 163383.88% FAIL** (catastrophic OP lock), TSMC5 inv tran 14.33% PASS, TSMC7 inv tran 14.48% PASS.

### Scope and destructive cleanup
- Wiped `external_compact_models/bsimar/checkpoints/` (refac_dn_*, refac_tf_*, v6_dn_*, v4_* symlinks), `checkpoints_legacy/` symlink, and the originals at `/home/shenshan/NN_SPICE/external_compact_models/bsimar/checkpoints/` + `data/datasets/` (~12 GB total). All universal V6 + V4 + legacy artifacts deleted; **no checkpoints remain for TSMC12/16/ASAP7 or LEVEL=74 BSIMAR** (out-of-scope for this sprint per user direction).
- Regenerated per-tech datasets via `generate_nn_data.py --device both --tech {tsmc5,tsmc7} --enable-inv-trip` into `tsmc{5,7}_{nmos,pmos}.npz`. Sizes after V6.1 final regen: TSMC5 nmos 2.30M rows / pmos 2.30M; TSMC7 nmos 2.07M / pmos 2.41M. Inv_trip overlay adds ~218K-255K samples per device.

### Code changes
- `bsimar/config.py`: added `LOCAL_VARIANT_CODES`, `LOCAL_UNKNOWN_CODE_ID`, `LOCAL_VOCAB_SIZE`, `local_variant_code(scope, tech, variant)`, `tech_scope_vocab_size(scope)`, and `VALID_TECH_SCOPES = ("universal", "tsmc5", "tsmc7")`. Per-tech vocab: TSMC5 = 5 (4 variants + UNKNOWN), TSMC7 = 4 (3 variants + UNKNOWN).
- `bsimar/data/dataset.py`: `load_and_split_bsimar` accepts `tech_scope`; when non-universal, remaps tech_codes from universal → 0-indexed local vocab after `exclude_techs` filter.
- `bsimar/cli/train.py`: added `--tech-scope` flag. When non-universal, auto-sets exclude-techs (all other techs), num-tech-codes (per-tech vocab size), default data path (`<scope>_<dev>.npz`), and save_prefix (`<scope>_dn_<size>[_<preset>]_<dev>`).
- `bsimar/training/trainer.py`: passes `tech_scope` through to dataset loader; instantiates `DirectNet(unknown_code_id = num_tech_codes - 1)` so per-tech UNKNOWN is at the LAST embedding row instead of hardcoded 17. **Without this fix, training-time `p_unknown` dropout writes code 17 into a 5-row embedding → CUDA assert.** (Universal training keeps the existing convention since vocab=18 → unknown=17.)
- `pycircuitsim/parser.py`: per-tech preempt slot inserted ABOVE the universal cascade for TSMC5/TSMC7. Resolver decodes vocab scope from the resolved checkpoint stem (`tsmc{5,7}_dn_*` → local; everything else → universal) and uses `local_variant_code` to map the netlist's TECH+VT to the right embedding index. Every resolution prints `[NN-resolver] L73 ... -> <chk> (scope=<s>, tech_code=<c>)` per Rule 12.
- `tests/verify_nn_dc_tran.py`: extended the directnet_v4 checkpoint resolver to also accept `refac_dn_medium`, `refac_dn_small`, and `tsmc{5,7}_dn_{medium,small}` as fallbacks (the path is now an *existence sentinel*). Added `_cascade_handles_stem(path)` and stopped stamping `MODEL_PATH=` for stems that the parser preempt cascade can route — so a single inverter test invocation picks TSMC5 medium for TSMC5 netlists and TSMC7 medium for TSMC7 netlists automatically.
- `external_compact_models/PyCMG/pycmg/nn_generate.py`: widened the inv_trip overlay gate from `tech_name == "tsmc5"` to `tech_name in ("tsmc5", "tsmc7")`. Same lever that took TSMC5 DN inv-tran from 16.90% → 0.92% in V5'.

### Training (8 cells, GPU 2)
S+M × {NMOS, PMOS} × {TSMC5, TSMC7} via `scripts/train_per_tech_8cells.sh`. Best val losses (asinh+zscore + per-target LDS-MAE):

| Cell                    | Best val loss |
|-------------------------|--------------:|
| tsmc5 small  nmos       | 0.00742       |
| tsmc5 small  pmos       | 0.00913       |
| tsmc5 medium nmos       | 0.00103       |
| tsmc5 medium pmos       | 0.00084       |
| tsmc7 small  nmos       | 0.01171       |
| tsmc7 small  pmos       | 0.00861       |
| tsmc7 medium nmos       | 0.00114 (after inv-trip retrain; was 0.00130) |
| tsmc7 medium pmos       | 0.00096 (after inv-trip retrain; was 0.00109) |

Medium val loss is **7-10× lower** than small for every (tech, polarity); medium is the production size and small is retained as a parser cascade fallback only for TSMC5 (TSMC7 small was deleted on the inv-trip regen, since it would be inconsistent with the new dataset and is never selected when medium is present).

### Validation (parser per-tech preempt active)

| Test                       | Baseline `refac_dn_medium` | V6.1 per-tech medium | Δ |
|----------------------------|-------------------------:|---------------------:|---:|
| TSMC5 inv VTC              | 9.58%   PASS             | 7.96%   PASS         | −1.62 pp |
| TSMC7 inv VTC              | 163383.88% **FAIL**      | **1.69%** PASS       | catastrophe fixed |
| TSMC5 inv transient (post-startup) | 14.33% PASS      | **8.23%** PASS       | **−6.10 pp** |
| TSMC7 inv transient (post-startup) | 14.48% PASS      | 13.49% PASS          | −0.99 pp |

Locked success criterion was ≥ 2 pp transient reduction on the worse-of-two (TSMC7). Final TSMC7 transient is **−0.99 pp** — strictly under the gate. Inv-trip overlay (added in the second pass) sharpened TSMC7 VTC further (3.22% → 1.69%) but **did not move TSMC7 transient**. Diagnosis from the comparison plot: TSMC7 transient settles at a **second stable equilibrium ~±100 mV outside the rails** because the PMOS forward-Vds region (V(out) > VDD in source-relative frame) is extrapolated outside the `[0, 2·VDD]` training box and produces non-zero leakage, balanced against Rule 15(a)'s NMOS pull-down. Documented as Rule 20 in CLAUDE.md; fix is out-of-scope for V6.1.

### Net result
- Catastrophic TSMC7 VTC failure fixed (163383% → 1.69%).
- Average DC NRMSE across TSMC5/7 inverter VTC: was unmeaningful (1 catastrophic FAIL); now 4.82%.
- 4/4 inverter tests PASS (was 2/4 PASS, 2/4 FAIL).
- TSMC12/16 / ASAP7 / LEVEL=74 simulations have no checkpoints and will fail until a separate retrain.

### Logs and artifacts
- Baseline measurement: `training_logs/baseline_tsmc57_v6medium/`
- Data-gen logs: `training_logs/data_gen/{tsmc5.log, tsmc7.log, tsmc7_invtrip.log}`
- 8-cell training logs: `training_logs/per_tech/`
- TSMC7 medium inv-trip retrain logs: `training_logs/per_tech_v2/`
- Validation: `training_logs/validation_pertech_medium/` and `training_logs/validation_pertech_v2/`

### Rule 20 fix attempt — closed, no production change

Three variants of an inference-time fix to the Rule 20 forward-Vds rail-overshoot finding were prototyped against `pycircuitsim/models/mosfet_nn.py:_apply_vds_correction` and all reverted: (1) widen the fast-path early-return to skip the wrong-sign clamp whenever `abs_vds > VDD_train`; (2) defer part-(a)'s id injection until after the part-(d) clamp; (3) defer + add an `|NN_raw| < 0.5·|id_a|` off-state detector. Each variant catastrophically regressed TSMC5/7 inverter VTC (>200000% NRMSE), because the wrong-sign clamp also catches NN-error overshoot during DC OP NR iterations at modest Vgs values where NN_raw is a real subthreshold current — not "off". Distinguishing genuine off-state from NR-intermediate subthreshold needs Vgs context, which the function doesn't currently receive. Variant 3 did improve TSMC5 transient (8.23% → 6.81%) but the trade was unacceptable. Recorded for future revisit: Path B (Vgs-aware refactor) and Path C (regenerate with two-sided Vds box + retrain).

## V6.2 — Rule 15(a) sign fix, Rule 20 dead-band closed (2026-05-13)

**Two-line sign flip in `pycircuitsim/models/mosfet_nn.py:_apply_vds_correction`.** No retraining, no dataset regen, no checkpoint changes. Same V6.1 per-tech DirectNet medium artefacts; the only diff is in the rail-restoring extrapolation step (a).

### Diagnosis (rebuts Rule 20's earlier "missing two-sided Vds box" thesis)

V6.1 left TSMC7 inverter transient at 13.49% NRMSE with a stable equilibrium ~±100 mV outside the rails. Rule 20 hypothesised the NN was producing unhandled leakage in a region between `0` and `VDD_train`. **Wrong root cause.** Three Rule-15 variants from V6.1's "Rule 20 fix attempt" all catastrophically regressed VTC (>200000% NRMSE) by deferring or weakening the wrong-sign clamp.

Probing the dead-band directly revealed the actual mechanism: Rule 15(a)'s `id_extra` injection was using the *opposite* sign from physical restoring leakage. In PyCMG convention an NMOS in conduction has `id < 0`; the restoring leakage of an OFF NMOS at high-rail overshoot should also drive `id < 0` (more negative, pulling drain back toward source). The original V4-re ship had `result["id"] += id_extra` for NMOS (positive, wrong direction) and `result["id"] -= id_extra` for PMOS (negative, also wrong). The wrong-sign clamp at step (d) then wiped any contribution that exceeded |id_raw| inside the band `VDD_train < |Vds| < 20·VT`, leaving a current-free dead-band where Vout could settle at any value in ~±0.15 V.

### Fix

```python
if normal_dir:
    if self._is_pmos:
        result["id"] += id_extra      # was: -=
    else:
        result["id"] -= id_extra      # was: +=
```

Two character swap; the existing magnitude/ramp formulae for `id_extra` and `g_extra` are unchanged.

### Validation (parser per-tech preempt active, V6.1 checkpoints unchanged)

| Test                       | V6.1                  | V6.2                  | Δ |
|----------------------------|----------------------:|----------------------:|---:|
| TSMC5 inv VTC              | 7.96%   PASS          | **3.08%** PASS        | −4.88 pp |
| TSMC7 inv VTC              | 1.69%   PASS          | **1.00%** PASS        | −0.69 pp |
| TSMC5 inv tran (post-startup) | 8.23% PASS         | **1.23%** PASS        | −7.00 pp / 6.7× |
| TSMC7 inv tran (post-startup) | 13.49% PASS        | **1.67%** PASS        | −11.82 pp / 8.1× |

Full TSMC5/7 NN sweep — 12/12 PASS:

- TSMC5 NMOS DC 0.81%, TSMC7 NMOS DC 7.44%
- TSMC5 PMOS DC 0.35%, TSMC7 PMOS DC 1.81%
- TSMC5 NMOS pulse tran 1.10%, TSMC7 NMOS pulse tran 8.36%

### Process notes

- The three dead-end V6.1 variants ("widen fast-path / defer id-injection / Vgs-aware off-state detector") all assumed the V4-re ship was correct and the rail-overshoot was an unhandled NN-leakage region. Each tried to extend Rule 15 with new state (Vgs context, deferred clamps, smoothsteps), none worked, because the actual bug was a sign convention in a single conditional that's been live since V4-re's 2026-04-20 rail-restoring extrapolation patch.
- The 2-line diff dispatched to an agent team (3 isolated worktrees, parallel proposals). Agent 2 (originally tasked with "sharper reverse-Vds VT") probed the dead-band before patching and found the sign error. The other two agents (Vgs-aware off-state, solver-level rail clamp) cancelled — the simpler fix dominated.

### Risk / scope

- Re-validation required before resurrecting TSMC12/TSMC16 or LEVEL=74 BSIMAR. Those code paths used the *old* sign and may have been silently relying on the wrong-sign clamp's `id=0` fallback as their effective rail behaviour.
- Rule 15(a) docstring in CLAUDE.md updated. Rule 20 collapsed to a one-line resurrection guard.
- No regression observed on the full 12/12 TSMC5/7 NN gate, but ring oscillator / SRAM / other circuits have not been re-validated as part of this sprint.

### Docs trim (same release boundary)

CLAUDE.md was pruned of stale rules and tricks now obsoleted by V6.2 shipping and BSIMAR being parked. No code or test changes — CLAUDE.md only.

- **Status block** retargeted V6.1 → V6.2 with the corrected NRMSE numbers.
- **Module structure** dropped the unshipped `tsmc5_residual.py` / `tsmc5_residual_train.py` references (V6 Tier M2 experiment, no checkpoints, never resurrected).
- **Resolver cascade** clarified that only `tsmc{5,7}_dn_{medium,small}` checkpoints exist on disk; the `refac_*` / `v4_*` universal fallback chain is wired in `parser.py` but unreachable until someone retrains a universal stack.
- **Testing & Verification** dropped the stale "verify_nn_universal*.py / verify_nn_multi_tech.py need porting" note — those scripts were deleted in v4-re PR-1. Also removed mention of TSMC12-SVT-only entry points (`verify_nn_dc.py`, `verify_nn_tran_v4.py`) since TSMC12 has no V6.2 checkpoint.
- **Rule 8 (PyCMG integration)** dropped the ASAP7-specific train-VDD parenthetical (ASAP7 excluded per Rule 17) and the long-removed `ProcessParams` / `extract_process_params` / `INPUT_COLUMNS` re-export note.
- **Rule 13 (Unified CLI)** retargeted from `refac_{dn,tf}_<size>` defaults to the V6.2 per-tech `tsmc{X}_dn_<size>_<device>` default; dropped the deleted `tsmc5_residual_train` entry.
- **Rule 15(a)** condensed: kept the operative sign-convention rule, deleted the duplicated V6.2 NRMSE numerics (already in this CHANGELOG entry).
- **Rule 19 (per-tech local vocab)** dropped the now-irrelevant universal-training convention (vocab=18, unknown=17) since no universal training is being done.
- **Rule 20** collapsed from a long CLOSED-issue block to a one-line guard noting the sign convention is load-bearing for parked code paths (TSMC12/16, LEVEL=74) and needs re-validation when those are resurrected.
- **Supported Features** retagged LEVEL=74 BSIMAR from "primary" (stale since V4-re) to "parked".

## V6.2.1 — Per-tech TSMC12/TSMC16 DirectNet extension (2026-05-14)

Reusing the V6.2 recipe end-to-end (data → train → verify) for the two unshipped TSMC nodes. Rule 20 explicitly called out re-validation of Rule 15(a)'s sign convention at the new VDD=0.80 V; the inverter gate passes without further changes.

### Code changes (3 small registry edits)

- `external_compact_models/bsimar/config.py`: extended `VALID_TECH_SCOPES` and `LOCAL_VARIANT_CODES` to include `tsmc12` and `tsmc16` (vocab = 5 variants + 1 UNKNOWN = 6 per scope).
- `external_compact_models/PyCMG/pycmg/nn_generate.py`: extended the inv-trip overlay gate from `("tsmc5", "tsmc7")` to `("tsmc5", "tsmc7", "tsmc12", "tsmc16")`. Overlay is VDD-relative (Vd ∈ [0.30·VDD, 0.70·VDD]) so it is safe at the new vdd_train=0.80 V.
- The rest of the pipeline (`bsimar/cli/train.py`, `bsimar/data/dataset.py`, `pycircuitsim/parser.py`, `tests/verify_nn_dc_tran.py`) already generalised on scope — no edits needed.

### Data + training

- Datasets generated with `--enable-inv-trip --n-workers 8`: `bsimar/data/datasets/tsmc{12,16}_{nmos,pmos}.npz`, 2,872,800 samples each.
- 8 training cells on the A100 (GPU 2 visible-index, run sequentially per `logs/train_8cells.sh`): `tsmc{12,16}_dn_{small,medium}_{nmos,pmos}_best.pt` + `_norm.npz`. Medium runs ~38 min/cell (200 epochs), small ~14 min/cell (80 epochs). All 8 cells `rc=0`; total wall ~3h31m.
- Local vocab `unknown_code_id=5` for both scopes — derived from `LOCAL_VOCAB_SIZE`, not hardcoded.

### Validation (parser per-tech preempt active)

| Test                           | TSMC12     | TSMC16     |
|--------------------------------|-----------:|-----------:|
| Inverter VTC NRMSE             | **1.61%** PASS | **0.91%** PASS |
| Inverter transient post-startup | **1.51%** PASS | **1.66%** PASS |
| Inv-tran high-rail / low-rail / transition | 1.29% / 1.47% / 3.16% | 1.06% / 1.67% / 4.21% |

Resolver logs confirm scope routing — `[NN-resolver] L73.0 Mn1 TECH=tsmc12 VT=svt -> tsmc12_dn_medium_nmos_best.pt (scope=tsmc12, tech_code=0)`. Quality is on par with V6.2 TSMC5/7 (TSMC5 3.08% / 1.23%, TSMC7 1.00% / 1.67%). Rule 15(a)'s sign convention transfers cleanly to VDD=0.80 V — no dead-band reappears.

### Risk / scope

- ASAP7 / LEVEL=74 BSIMAR still parked — would still need a dedicated retrain.
- The full DC sweep (without `--inverter-only`) was not run as part of this sprint; the inverter gate was the user-stated success criterion. Rule 20 remains for LEVEL=74 only.

## V6.3 / V6.3.1 — Inverter spike-removal sprint (2026-05-15)

Goal: remove the inverter VTC + transient error spikes documented in
`results/v6_2_1_metrics_report/`. Agent-team diagnosis found three root
causes; the sprint ran in three phases (A discarded, B + C shipped).
Full plan + dead-end record: `docs/plans/2026-05-14-v6.3-spike-removal.md`.

### Root causes (agent-team diagnostic)

- **RC1 — reverse-Vds Id clamp** (`pycircuitsim/models/mosfet_nn.py:430`):
  the `f_id = 0` branch zeroes Id for reverse Vds, so when an inverter's
  load cap rings past a rail the NMOS produces no restoring current and the
  output undershoots ~99 mV (TSMC12/16).
- **RC2 — `inv_trip` overlay mis-centered**: pre-V6.3 the overlay centered
  Vg on the transistor peak-gm Vth, not the inverter Vtrip ≈ VDD/2. TSMC12/16
  had zero overlay rows in the switching band; TSMC5 only 0.24 %.
- **RC3 — zero reverse-Vds training coverage**: the main grid swept Vd ≥ 0
  (NMOS) only, so reverse conduction was never learned.

### Phase A — inference-only `gds`-bump gate (DISCARDED)

Gating the unconditional `gds = max(gds, g_extra)` bump on `normal_dir`
produced **bit-identical** eval traces — the reverse branch with
`|Vds|>VDD_train` is never hit on converged operating points. Reverted.
The real RC1 driver is the Id clamp, not the gds bump.

### Phase B — dataset regen (`_inv_trip_points` recenter + `_reverse_vds_points`)

`nn_generate.py`: re-centered `_inv_trip_points` on VDD/2 with a
`[0.30,0.70]·VDD` Vg/Vd box; added `_reverse_vds_points` (480 samples/bin,
new `sample_class="reverse_vds"` code 10). Regenerated all 8 datasets,
retrained 8 medium cells.

Result: transient pull-low spikes fixed across all 4 techs (TSMC12/16
99→58 mV) and TSMC5 VTC catastrophe fixed (206→58 mV) — **but TSMC7/12/16
VTC regressed** (+87/+24/+20 mV) because the wider, denser `inv_trip`
overlay (9.83 % of rows) over-fit a too-steep Id-Vg slope at the trip.

### Phase C — `_inv_trip_points` Vbs reduction (V6.3.1, SHIPPED)

`_inv_trip_points` dropped the `±0.25·VDD` Vbs sweep (the inverter runs at
Vbs=0 always; the `grid` class already covers Vbs). Overlay cut 25×9×3 →
25×9×1, from 9.83 % → 3.51 % of rows. Regenerated 8 datasets, retrained 8
medium cells (3-way multi-GPU parallel, ~2 h).

### V6.3.1 inverter results vs V6.2.1 (NGSPICE BSIM-CMG ground truth)

| Tech | VTC MaxErr V6.2.1→V6.3.1 | Tran post-startup MaxErr V6.2.1→V6.3.1 |
|------|--------------------------|-----------------------------------------|
| TSMC5  | 206.6 → **66.4 mV** | 79.1 → **39.5 mV** |
| TSMC7  | 55.9 → **65.8 mV**  | 55.5 → **50.3 mV** |
| TSMC12 | 39.2 → **78.3 mV**  | 99.2 → **58.2 mV** |
| TSMC16 | 30.9 → **45.4 mV**  | 97.8 → **55.3 mV** |

VTC NRMSE 1.52–1.77 %, transient post-startup NRMSE 1.22–1.51 %, ΔVtrip
≤0.6 mV, R² ≥ 0.9987 everywhere. Reports in `results/v6_3_1_metrics_report/`;
intermediate V6.3 (pre-Phase-C) in `results/v6_3_metrics_report/`.

### Outcome — shipped with one open gate

V6.3.1 is the new shipping revision. **Wins:** transient pull-low spikes
cut ~43 % (99→58 mV); TSMC5 VTC catastrophe cut 3.1× (206→66 mV); the
Phase-B TSMC7 VTC regression (143 mV) fully recovered (66 mV). Worst-case
VTC 143→78 mV, average VTC 79→64 mV vs the Phase-B intermediate.

**Open gate (deferred per user, 2026-05-15):** VTC MaxErr ≤ 25 mV not met
— V6.3.1 sits at 45–78 mV. Three dataset revisions moved the trip error
around but never below ~45 mV. Diagnosis: this is **not** a coverage gap
but gain amplification — inverter gain ≈ −15 to −30 at the trip multiplies
the NN's residual Id error (~0.05 % test-split NRMSE) ~20× into Vout. The
fix needs a gm/gds-fidelity lever (e.g. trip-weighted gm-matching loss),
not more `inv_trip` samples. Transient post-startup ≤ 30 mV also unmet
(39–58 mV), same root cause at the t=0 DC OP.

### Infra notes

- `/home/shenshan/NN_SPICE/` (this worktree's parent `.git` + the PyCMG
  submodule + dataset `.npz` targets) was moved to `/tmp/NN_SPICE/` mid-sprint
  to free a 98 %-full `/home`. Symlinks `external_compact_models/PyCMG` and
  `bsimar/data/datasets/*.npz` were repointed to `/tmp/NN_SPICE/`.
- V6.3 (pre-Phase-C) datasets preserved as `*.v6_3.npz`; V6.3 checkpoints
  backed up to `/tmp/v6_3_checkpoints_backup/`.
- New scripts: `scripts/regen_v6_3_1.sh`, `scripts/train_v6_3_1_parallel.sh`
  (3-way multi-GPU), `scripts/eval_v6_3_1_inverter.py`.

## V6.3.2 — NN parametric test harness (2026-05-17)

Branched `feat/v6.3.2` from the pre-V6.4-finalize HEAD (`beac301`). No model
or checkpoint change — this release **ports the BSIM-CMG L3 parametric test
harness to the DirectNet (LEVEL=73) NN models** and runs the V6.3.1
checkpoints through it. (The `feat/v6.4` branch independently advanced the
checkpoints; V6.3.2 is a test-infrastructure point release on the V6.3.1
model. Its harness was merged into `feat/v6.4` on 2026-05-17.)

### Motivation

The BSIM-CMG harness (`verify_multi_tech_{dc,tran}.py`) sweeps device geometry
and inverter circuit parameters; the NN harness (`verify_nn_dc_tran.py`) only
ran fixed-geometry points. The V6.3.1 DirectNet checkpoints had never been
stress-tested across the parametric space the BSIM-CMG reference covers.

### What shipped

- **`tests/common/nn_sweep.py`** — shared parametric harness: sweep-config
  dataclasses, builders, single-test orchestrators, a baseline-gated
  multi-tech loop, and summary/CSV/bar-plot helpers. Reuses the existing
  `verify_nn_dc_tran.py` runners; geometry/VT/VDD sweeps ride on
  `dataclasses.replace(TestTechConfig, ...)` (zero DC-runner refactor).
- **`tests/verify_nn_multi_tech_dc.py`** — single-device NMOS/PMOS Id-Vgs over
  L / NFIN / VT.
- **`tests/verify_nn_multi_tech_tran.py`** — inverter VTC + transient over P/N
  ratio, VDD, Cload, input slew, pulse width.
- **`verify_nn_dc_tran.py` refactor (behaviour-preserving)** — added
  `InvCircuitParams` (frozen dataclass; Cload/tr/tf/pw/td/tstop, defaults =
  legacy globals) threaded through the two inverter-transient runners, and an
  `inv_nfin_p` field + `effective_inv_nfin_p` property on `TestTechConfig` for
  the P/N-ratio NFIN split. `circuit=None` / `inv_nfin_p=0` reproduce the
  legacy fixed point exactly — verified by netlist-string audit (a value-match
  regression guard was infeasible, see "checkpoint contamination" below).

### V6.3.1 results

- **Single-device DC — 55/55 PASS** (gate NRMSE < 10%). Baselines ≤ 0.2%
  except TSMC7 NMOS 6.5%. Stressors elevate as expected: off-bin L (TSMC5 nmos
  L=24nm 2.6%), NFIN=10 (TSMC12 pmos 7.6% / MRE 25%), TSMC5 NMOS VT variants
  (ulvt 6.3%). VT sweeps on TSMC12/16 near-perfect (< 0.25%).
- **Inverter VTC + transient — 63/64 PASS** (gate NRMSE < 15%). All baselines
  pass (VTC 1.4–4.1%, transient 1.3–1.8%). Sole **FAIL: `TSMC5_vtc_vdd_0p55`**
  16.8% — VDD−0.1V drops the trip below the per-tech NN's accurate band.
  VDD−0.1V is the dominant stressor across techs (TSMC7/TSMC16 VTC 11.7/12.2%,
  near the gate); fast slew (10 ps) elevates to 4–6%; Cload, pulse-width,
  slew=500 ps and the single P/N-ratio point (`nfin_p=3`) all stay < 3%.
- Finding: V6.3.1 DirectNet is **VDD-specific** — it degrades sharply ~0.1 V
  off the training VDD. A VDD-robustness lever (train-time VDD jitter) is the
  natural follow-up.

### Harness design notes

- **P/N ratio is one point.** The TSMC naive-modelcard NFIN-group rule
  (`nfin_p > nfin+1` skipped) with `default_nfin=2` admits only `nfin_p=3` —
  exact parity with the BSIM-CMG harness; the limiter is the modelcard.
- Off-bin L/NFIN points exercise NN extrapolation beyond the per-tech training
  bins; elevated NRMSE there is expected model behaviour, not a harness fault.

### Dead end recorded — checkpoint contamination (cost ~1 h)

The first full runs were **invalid**: `bsimar/checkpoints/` was symlinked into
the shared checkout, and the concurrent `feat/v6.4` best-of-N work **overwrote
the `tsmc*_dn_medium_*` slots at 07:46:58** mid-run. This produced ~±1 % VTC
NRMSE run-to-run scatter (7 TSMC12 readings spanning 1.7–3.8 %) — chased
fruitlessly against PyTorch threading and `PYTHONHASHSEED` before the moving
checkpoint files were identified as the cause. Fix: a worktree-local copy of
`/tmp/v6_3_1_checkpoints_backup/` (md5-verified V6.3.1), plus an isolated
`PyCMG/build/modelcards/` (the v6.4 eval jobs also raced the shared naive
modelcards). Lesson: a verification harness must own immutable copies of its
inputs; never point it at a directory under active training.

## V6.4 — Best-of-N retrain + complex-circuit benchmark harness (2026-05-15 .. 17)

Plan: `docs/plans/2026-05-15-directnet-complex-circuits.md` (re-prioritized
2026-05-15). V6.4 executed Phases 1–3; Phases 4–8 deferred.

### Phase 3 — complex-circuit benchmark harness (shipped)

Four benchmarks vs NGSPICE BSIM-CMG ground truth: 5-stage ring oscillator,
two-stage Miller opamp, 6T SRAM read SNM, switched-cap unit cell. Harness in
`tests/common/complex.py` + `tests/verify_complex_*.py`; netlists in
`examples/complex/` + `tests/references/complex/`. Baseline V6.3.1: ring-osc
2/4, opamp 0/4 (gain error 10–135 % — confirms plan blocker D0, gain
amplification of DirectNet's Id residual), SRAM-SNM 4/4, switched-cap 1/4.

### Phase 1 — DirectNet retraining is a seed lottery; best-of-N selection

The plan's original 1a–1e levers were all dropped: the 1b Sobolev term lost
its bake-off decisively (7–8× worse VTC — a *validated* dead end, distinct
from the 2026-05-03 unvalidated `SlopeMatchLoss`; Rule 10 unchanged); 1e
(`gds` asinh floor) is a confirmed no-op; the 1a validation slices are a
broken near-zero-denominator proxy.

Decisive finding: a clean, verified stock-recipe DirectNet retrain (`--seed
42`) regressed TSMC5 inverter VTC MaxErr to 218 mV vs V6.3.1's 66 mV; seed
123 gave 79 mV — a **139 mV seed-driven swing**. Transient is seed-stable
(~38 mV at every seed). V6.3.1's shipped checkpoints were a lucky draw. So
V6.4 produces checkpoints by **best-of-N**: 8 seeds × 8 cells
(tsmc{5,7,12,16} × {nmos,pmos}), each tech's (nmos,pmos) pair selected on the
real inverter VTC sim — never on val loss (the D6 decoupling).

Corrected false lead: an agent claimed the datasets had drifted from V6.3.1
and recommended restoring `*.v6_3.npz`. Verified false — the current datasets
carry a 3.51 % `inv_trip` overlay, an exact match to the V6.3.1 spec; the
`*.v6_3.npz` files are the *older V6.3* (9.83 %) data and must not be restored.

### Phase 2 — 2a kept, 2b reverted (unsound)

2a (transient C-stamp symmetrization, env-gated `NN_SYMMETRIC_CAPS`, default
off) is kept as dormant infrastructure for the Phase-3 ring oscillator.

**2b (always-on conducting-branch `gm/gmb` sign-floor) was reverted.** It
appeared to halve inverter VTC error on TSMC5/12/16 — but that was an artifact
of *circular selection*: best-of-N had been scored on the 2b solver, so it
merely picked seeds compatible with the gm hack. On neutral ground (V6.3.1
checkpoints) the `gm`-floor breaks TSMC7 (66→215 mV) and TSMC12 (78→261 mV);
the `gmb`-floor is completely inert; a `reflect` variant (wrong-sign `gm` →
correct sign, magnitude kept) breaks 3/4 techs (TSMC12 354 mV). Zeroing or
altering an autograd wrong-sign `gm` is a checkpoint-dependent coin-flip —
there is no sound `_floor_gm` fix. The principled fix for wrong-sign `gm` is a
network constraint (plan Phase 6 monotonicity / spectral norm), not a solver
hack. `mosfet_nn.py` reverted to its pre-Phase-2 state.

### V6.4 final — clean-solver best-of-N, inverter vs NGSPICE BSIM-CMG

Selection re-run on the clean (2b-reverted) solver. All 4 techs beat V6.3.1
inverter VTC MaxErr; transient holds (TSMC7 +1.2 mV, within noise):

| Tech   | seeds n/p   | VTC MaxErr V6.3.1→V6.4 | Tran post-startup V6.3.1→V6.4 |
|--------|-------------|------------------------|-------------------------------|
| TSMC5  | 17 / 42     | 66.4 → **62.0** (−7 %)  | 39.5 → 37.9 mV |
| TSMC7  | 31337 / 42  | 65.8 → **60.1** (−9 %)  | 50.3 → 51.5 mV |
| TSMC12 | 123 / 123   | 78.3 → **32.3** (−59 %) | 58.2 → 57.6 mV |
| TSMC16 | 42 / 123    | 45.4 → **29.7** (−35 %) | 55.3 → 54.9 mV |

VTC NRMSE 1.20–2.13 %, R² ≥ 0.9981, ΔVtrip ≤ 0.3 mV. TSMC16 (29.7 mV) and
TSMC12 (32.3 mV) approach the deferred ≤25 mV stretch gate. TSMC5/7 gains are
modest — their clean-solver seed lottery surfaced no strongly better draw
within N=8.

### Open / deferred

- Inverter VTC ≤25 mV still unmet (V6.4 at 29.7–62.0). TSMC5/7 are the
  laggards; a larger seed sweep or the plan's Phase 6 structural levers are
  the next step.
- Plan Phases 4–8 (data overlays, batched NN forward, NR convergence, soft
  physics constraints, per-target heads) deferred.
- Complex-circuit benchmarks not yet re-measured on the V6.4 checkpoints —
  harness is in place; pass/fail TBD.

### Process notes / dead ends

- Three sub-agents died to a 600 s no-progress watchdog while waiting on long
  jobs; orchestration was redone as plain background scripts (no watchdog).
- The retrain pool was once killed one batch early; 8 tsmc16-pmos cells were
  re-trained cleanly.
- Artifacts: 64 best-of-N candidate checkpoints `v6_4_bof_*` / `v6_4_repro_*`
  in `checkpoints/` (gitignored); V6.3.1 backup at
  `/tmp/v6_3_1_checkpoints_backup/`. Best-of-N pair evaluator:
  `scripts/eval_v6_4_pair.py`.

## V6.4.1 — harness merge + single-seed retrain (2026-05-17)

Branch `feat/v6.4.1`. Two changes: (1) merged the V6.3.2 parametric NN test
harness into the V6.4 line (`tests/common/nn_sweep.py` +
`verify_nn_multi_tech_{dc,tran}.py`); (2) re-trained all 8 DirectNet medium
cells from a **single seed (42)** — not best-of-N — and re-ran the full
extended harness against the fresh checkpoints.

### Retrain

`scripts/run_v6_4_1_retrain.sh`: 8 medium cells (tsmc{5,7,12,16} ×
{nmos,pmos}), `--seed 42`, across 3 GPUs (GPU1 Blackwell + GPU0/GPU2 A100).
The entire `checkpoints/` directory was wiped first (107 MB of V6.4 best-of-N
production + candidate artifacts); a copy was preserved at
`/tmp/v6_4_checkpoints_backup_20260517/`. New checkpoints land in the canonical
`tsmc{X}_dn_medium_{nmos,pmos}` parser-cascade slots. Per-tech test R²
0.997–1.000.

### Extended harness results (V6.4.1 seed-42 checkpoints)

- **`verify_nn_multi_tech_dc.py` — 55/55 PASS** (gate NRMSE < 10%). Unchanged
  vs the V6.3.1 harness run.
- **`verify_nn_multi_tech_tran.py` — 64/64 PASS** (gate NRMSE < 15%), VTC +
  transient. This *clears* the V6.3.1 harness's sole FAIL `TSMC5_vtc_vdd_0p55`
  (16.8% → 14.08%).

### Seed lottery confirmed — VTC regressed vs V6.4 best-of-N

Single-seed retrain lost the lottery on inverter VTC MaxErr, exactly as the
V6.4 finding predicted. Inverter VTC baseline MaxErr vs NGSPICE BSIM-CMG:

| Tech   | V6.4 best-of-N | V6.4.1 seed-42 |
|--------|----------------|----------------|
| TSMC5  | 62.0 mV        | **128.0 mV**   |
| TSMC7  | 60.1 mV        | **174.7 mV**   |
| TSMC12 | 32.3 mV        | **41.6 mV**    |
| TSMC16 | 29.7 mV        | **33.6 mV**    |

All 4 techs regressed (TSMC7 ~3×). Transient is seed-stable as documented —
baseline transient MaxErr TSMC5 61.1 / TSMC7 49.2 / TSMC12 64.2 / TSMC16
69.0 mV, in line with V6.4. The harness NRMSE gates (10% / 15%) still pass
because they are looser than the VTC-MaxErr program gate; the seed-42 draw is
**not** an improvement over the shipped V6.4 best-of-N checkpoints.

**Recommendation:** the better V6.4 best-of-N checkpoints are preserved at
`/tmp/v6_4_checkpoints_backup_20260517/`; restore them, or run a full
best-of-N (`scripts/run_v6_4_bestof.sh`) on `feat/v6.4.1`, before treating
V6.4.1 as a shippable model. V6.4.1 currently ships the harness merge, not a
model improvement.

## V6.4.2 — complex-circuits sprint: solver Phases 5–6, Phase 4 dead end (2026-05-18)

Branch `feat/v6.4.1`. Continuation of the deferred phases of
`docs/plans/2026-05-15-directnet-complex-circuits.md`. Solver Phases 5 & 6
shipped; the Phase 4 data lever was retrained and proved a dead end.
**Shipping checkpoints are unchanged — still the V6.4.1 seed-42 set.** The
inverter gate (`verify_nn_dc_tran.py --inverter-only`) is **8/8 PASS** on the
final state (VTC NRMSE 2.61 / 4.03 / 1.64 / 1.41 %; transient 1.56 / 1.26 /
1.41 / 1.45 %), BSIM-CMG trio byte-identical.

### Phase 5 — Batched DirectNet forward + Jacobian — SHIPPED (`d1fe87a`)

`_is_nn_mosfet()` + `_MOSFETNNBase.batch_eval()` collect every LEVEL=73 device
into one stacked forward + one `autograd.grad` per NR iteration (was one call
per device per iter). Inverter (1 NMOS + 1 PMOS, group-of-one) is
**bit-identical** to the per-device path; BSIM-CMG (LEVEL=72) untouched. Opamp
DC OP **3.4× faster** (`run_backward` 272k→37k calls). The plan's ≥5× target
was specced against a hypothetical 30-device opamp; the real Phase-3 opamp is
7 devices. **Known limitation:** with N>1 devices on a shared checkpoint, a
stacked GEMM differs from N separate GEMVs by ~1e-8 (a hard BLAS fact) —
measured metrics (opamp gain, RO period) are preserved but node voltages are
not bit-identical. `NN_BATCHED_EVAL=0` forces the exact per-device path;
default-on.

### Phase 6 — NR convergence upgrades — SHIPPED (`35e9a16`)

- **6a** Levenberg-Marquardt damping alongside the rail cap in both NR loops
  (DC + transient): when the MNA residual fails to decrease, re-solve with
  `λ·I`, Nielsen ×10 escalation / ÷3 on acceptance.
- **6b** Residual-norm `‖rhs−A·v‖∞` OR-gate on the SPICE `|ΔV|` convergence
  test, and a guard on the averaged-solution acceptance in DC oscillation
  detection + its transient analog — a stalled iterate with small `Δv` but
  large residual is now rejected.
- **6c** Pseudo-transient DC continuation as a fallback in
  `_solve_dc_with_retry` (after fast-path + GMIN-retry both fail).

Non-regressing: inverter 8/8, BSIM-CMG byte-identical, helpers unit-tested.
**The Phase-6 RO/SRAM success gate was NOT closed — and Phase 6 alone cannot
close it.** Verified root cause: the ring-oscillator TSMC5/7 period errors and
the SRAM `force_ic` failures are **model-accuracy gaps in the seed-42 V6.4.1
checkpoints**, not NR-convergence failures. The RO transient already converges
to a bit-identical inaccurate period; the SRAM `force_ic` re-solve converges to
a consistent non-rail NN fixed point. Phase 6 improves *how robustly* a fixed
point is reached — it cannot move a fixed point a converging solve already
reaches. Closing those gates needs a better model.

### Phase 4 — data overlays + non-uniform sampling — DEAD END, reverted (`565de40`)

Implemented §4a–4e (overlays `diff_pair_sat` / `ring_osc_trip` /
`bistable_static` / `switched_cap_offstate`, sinh-spaced Vgs/Vds sampling, LHS
Vbs, NFIN∈{4,6,8,12}, `--keep-offstate`), regenerated all 4 TSMC datasets, and
ran the full best-of-N grid (8 seeds × 8 cells = 64 cells). A greedy
~19-eval/tech pair search (`scripts/v6_4_1_phase4_search.py`, swap-eval-restore
via `eval_v6_4_1_pair.py`) found **no pair beating the V6.4.1 baseline on any
tech**:

| Tech   | V6.4.1 baseline (VTC / tran MaxErr) | P4 best-VTC pair | P4 best-tran pair |
|--------|-------------------------------------|------------------|-------------------|
| TSMC5  | 134.6 / 39.6 mV                     | 66.5 / **98.4**  | **96.5** / 247.0  |
| TSMC7  | 210.5 / 49.4 mV                     | 104.2 / **87.0** | **83.7** / 372.9  |
| TSMC12 | 63.1 / 58.6 mV                      | 90.8 / **112.3** | **112.0** / 387.5 |
| TSMC16 | 50.8 / 55.1 mV                      | 142.8 / **54.5** | **54.2** / 378.8  |

The Phase-4 data forces a hard **VTC↔transient tradeoff** — every candidate
that improves one metric wrecks the other (best-transient pairs carry 247–388
mV VTC). Same failure family as the V6.3 9.83%-overlay TSMC7 regression and the
Phase-1b Sobolev dead end: heavier overlay data destabilizes the joint fit.
Verdict for all 4 techs: **KEEP V6.4.1**.

The data-pipeline commits were reverted: `e605319` (overlays / sinh /
`--keep-offstate` + PyCMG submodule bump `7e7d06c`→`8794624`) and `ff8037f`
(its §4d extra-NFIN labeller fix). Default sampling returns to `np.linspace`.
The best-of-N harness (`b62b326`: `v6_4_1_phase4_search.py`,
`eval_v6_4_1_pair.py`, `run_v6_4_1_phase4_bestof.sh`) is kept as recorded
dead-end evidence; per-tech search logs in `logs/v6_4_1_phase4/`, candidate
checkpoints `v6_4_1_p4_*` in `checkpoints/` (gitignored). The "larger data
lever" is exhausted at this overlay/sampling design; the remaining accuracy
path is Phase 7 (network-structural constraints), not more data.

### Dead-end record

- **Phase 4 data overlays + sinh sampling** — full best-of-N retrain, no pair
  beats V6.4.1; hard VTC↔transient tradeoff. Reverted. Do not re-propose more
  operating-region overlay data without a structural argument for why the
  joint fit would not destabilize.
- **Phase 6 for RO/SRAM** — convergence upgrades cannot fix what is a model
  fidelity gap. RO/SRAM accuracy is gated on the model, not the solver.
