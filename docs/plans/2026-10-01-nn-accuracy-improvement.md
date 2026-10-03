# Improving NN compact model accuracy

Status: in progress for **V7.7.7**, revision 5 (2026-10-03). Targets are
DirectNet-Full LEVEL=75 and BSIM-AR-Full LEVEL=76. Review round 1 ran three
diagnostic retrains and read-only probes ([findings](#review-round-1-findings)).
Those reviews changed no model, runtime, or published result. Execution now
includes reference-bin and energy-metric repairs; see the progress below.

Round 2 found reference-bin drift, a severe near-origin regression in the
normalization arms, reference quantization, and holdout reassignment. Its
[diagnostic report](../accuracy/2026-10-01-nn-plan-review-round2.md) records
experiments and limitations. Artifacts are isolated under
`results/plan_review_20261001_round2/`. Existing controls and qualification
reports remain preserved; neither tested small-scale arm advances as-is.

Round 3 audited both documents against the raw artifacts and training data
([findings](#review-round-3-findings)). Both current references are quantized
near 0.11 pA, round 2's dead band sits in a sampling gap, the reference-bin
defect scores the NN against parameters its labels never used, and hot gate
current is a whole-surface regime. It ran no training.

Order of work:

1. Repair reference, split and metric prerequisites. Build the failure
   ledger, then decide from it whether leakage stays the first model target.
2. Measure seed spread.
3. Close the drain-origin sampling gap, then improve leakage without losing
   origin conductance.
4. Fix the training schedule, then checkpoint selection.
5. Close geometry and temperature coverage gaps.

Derivative supervision, the BSIM-AR strategy comparison and convergence work
are conditional and run later. Retraining and dataset regeneration are in
scope. Each arm makes one change against a control trained from the same
commit on the same GPU architecture.

## V7.7.7 execution progress (2026-10-03)

The [Phase 1 attribution report](../accuracy/2026-10-03-v777-phase1-attribution.md)
owns measurements and limitations. Reproducible scripts, exact launch settings,
source patches, cards, decks and traces are under
`results/nn_accuracy_20261001/phase1_reference/`. No checkpoint was retrained or
promoted. The published campaign remains V7.7.6.

- [x] Repair actual-NFIN bin selection in the circuit, LEVEL=72 DC and
  transient card builders; include both fin counts in the transient merged
  cache and filename, and in DC card filenames. Verify independent card-parameter identity, deck parity
  and LEVEL=72 fixed-bias currents against NGSPICE across the five technologies.
- [x] Build a lossless 7,232-row ledger from the preserved V7.7.6 flags-off
  rerun of the historical round inventories. Tag hardware, temperature,
  reference-bin exposure and distinct failure categories. Keep threshold-free
  diagnostic rows separate from qualification failures.
- [x] Re-baseline the DirectNet-medium TSMC5/12 SRAM gates; reproduce Miller
  AC, energy, beta multiplier, LDO (including non-`uic` load-step transient)
  and SRAM modes with explicit checkpoint pins. Add LEVEL=72 circuit controls
  to the four topology cases. These scoped runs do not replace a campaign.
- [x] Repair the inverter-energy measurement after the LEVEL=72 control
  reproduced the apparent NN error. Register signed, native-grid integration
  in the [methodology](../accuracy/methodology.md#inverter-supply-energy-v777),
  and re-run both pilots. Old energy numbers require re-baselining.
- [x] Run a 288-point temperature-onset screen and compare 32 generated cards
  with the merged PDK parameters. This checks serialization/cache fidelity;
  independent PDK parsing and source-wrapper equivalence remain unverified.
- [x] Run a scoped circuit-demand census (36 analyses, 204 device/analysis
  entries), retaining full NGSPICE voltages and voltage-source branch currents.
  Both leakage and the origin gap are exercised; keep O before N.
- [x] Compute the zero-training scalar residual/slope trade-off from the
  existing 45 fixed-bias normalization-arm probes. It is development evidence,
  not a prediction that an intermediate scalar will pass.
- [ ] Complete attribution on the remaining pilot families/cells, high-NFIN
  NN DC, full failure census, fresh-parse state re-stamping, Miller matched-bias
  I/G/C, polarity swaps and independent hot-card audit.
- [ ] Resolve the LEVEL=72 control failures before using the affected LDO and
  SRAM-mode rows as NN acceptance evidence. Keep their denominator slots and
  preserve failed-control diagnostics separately from candidate results.
- [x] Implement persistent parent split membership/order, raw float64 targets
  and row identities, and extend training completion provenance. The registered
  diagnostic screen freezes a conservative quantum; broader calibration is
  still required before leakage acceptance.
- [ ] Run Phase 0 and the registered training arms in the order below.

**Amendments from execution.** Re-baselining is required for non-default fin
counts starting at NFIN=3, not only NFIN≥4: nominal PMOS off-current changes
materially even where the on-state comparison agrees. The ledger flags the
40 aggregate SRAM rows in this nominal campaign; the already geometry-aware
`nn_gate` device/inverter paths are unaffected. Correcting the reference
substantially removes the pilot TSMC5 SRAM-lobe mismatch, but does not fix
the separate forced-state diagnostic. Energy is a harness correction and is
removed as evidence for an NN charge-loss arm until a new mismatch is shown.
The temperature-onset screen is too sharp and too sparsely sampled to justify
accepting the entire interpolation interval; it does not resolve the runtime
temperature-support decision.

**Completion condition.** This progress is not completion of the plan. Local
checkpoint commits are required before canonical data generation. The requested
final push to `main` follows completion and verification of the experiment and
comparison report; a prerequisite commit is not a completed accuracy result.

### Registered first model experiment

Execution source is local checkpoint `b385118`, isolated in
`PyCircuitSim-v777-pilot`; data, training logs and bundles are under
`results/nn_accuracy_20261001/pilot/`. The separate evaluation runtime includes
the DC logging fix and records its own source commit. The Phase 1 Miller
matched-bias matrices, fresh-parse candidate residuals, complete branch-current
snapshots and NMOS/PMOS LEVEL=72 swaps are retained under
`results/nn_accuracy_20261001/phase1_followup/`.

Use TSMC5 and TSMC12, both polarities, DirectNet medium, the preserved full
datasets and the existing seed-42 combo split. Freeze each parent row's
partition and order before appending any data. Arm O adds only the signed
drain-origin grid: 13 log-spaced magnitudes from 10 µV to 5 mV, 17 gate levels
from −0.25×VDD to 1.5×VDD and three body biases (−0.25, 0, +0.25)×VDD.
Skip exact coordinates already in each parent group; new points inherit that
group's partition. Keep original rows and labels unchanged. The isolated
control is the parent plus split/identity metadata, not a regenerated fit.

For the diagnostic screen, train control and O at seed 42, 200 epochs,
patience 40, batch 2048, LR 1e-3, EMA 0.999, AMP off, on UUID-pinned A100s.
Pair each control/candidate on the same physical GPU. Record the extra updates
from O's added rows; this is a coverage arm, not the equal-update schedule arm.
First verify that the control reproduces the preserved bundle. Extend the
control seed spread and replicate surviving arms before any acceptance or
promotion. This moves the single-seed rejection screen ahead of expensive
replication; it does not weaken the four-seed acceptance requirement.

Report the unchanged qualification tests plus the attributed development
sentinels for both technologies. Use raw float64 targets, signed origin
currents/conductances and all four terminal tails. The screen uses a
conservative fixed current quantum of `2^-42 A` inside its |Vds|≤2 V domain:
percent errors start at 200 quanta and log errors at 20. Report exact-zero
references/predictions and sign errors separately. An O arm with new negative
on-state origin conductance, lost physical fixed points, new qualification
failures or worse tails is rejected before scale search or replication.
Already inspected test rows and repeatedly used circuits are development
evidence. No scoped screen replaces the untouched confirmation inventory or
the full five-technology qualification campaign.

### Registered scalar screen N (after O)

The O screen's 648 NGSPICE-backed signed-origin points introduce no negative
on-state `gds`, but coverage alone leaves substantial weak-current error.
Test the declared constant `s_id=1e-7 A` on the same O datasets, both pilot
technologies/polarities, seed 42 and the unchanged medium recipe. The constant
is fixed across bundles, well above the reference quantum, and reduces the
origin slope demand by three orders of magnitude relative to the rejected
`1e-10 A` arm. No per-cell scale tuning or warm start is permitted.

Retrain O controls and N from the same source commit, paired on the same
physical A100, and verify the repeated O control before interpreting N.
Keep the 648-point origin screen, raw holdout inventory, current quantum,
92-job circuit inventory and vetoes fixed. O is an experimental data control,
not a promoted model; N must also be compared with the preserved baseline.
No derivative, runtime-transform, BSIM-AR transfer or seed-expansion arm is
launched on the strength of a failed single-seed screen.

## Evidence and priorities

The baseline is the V7.7.2 bundles scored by the
[V7.7.6 campaign](../accuracy/v776-evaluation.md) with the limiter off. The
[accuracy index](../accuracy/README.md) overrides status text in older plans.

| Evidence | Implication |
|---|---|
| [DirectNet medium](../accuracy/DirectNet-L75-clean.md): strict simple-v1 20/20, device DC 129/129, device AC 10/10, inverter 100/100. Miller AC is 0/5, but the Miller DC-gain gate passes 5/5 (0.69–1.73%). | First DirectNet control. Test operating-point displacement before slope error. |
| [BSIM-AR large](../accuracy/BSIM-AR-L76-clean.md): simple-v1 20/20, Miller AC 4/5. TSMC5 is a converged FAIL (13.9 dB); medium and XL TSMC5 are `ERROR` rows. | BSIM-AR control. |
| TSMC5 fails in both families: Miller AC in all 8 family×size cells, SRAM SNM NRMSE 6.1–7.1% (≤3.9% elsewhere), NMOS +125 °C surface error 19–22%. | TSMC5 is a technology-level target. |
| DirectNet large/XL device-DC misses are high-NFIN cells on TSMC12/16 ([round 1](../CHANGELOG.md#2026-09-14--v775-checkpoint-quick-round-1)). | A geometry gap ([finding 5](#review-round-1-findings)). Neither pilot control can observe it. |
| DirectNet L3/L4 results are non-monotone in size ([off matrix](../accuracy/v776-case-matrix-off.md)). | Size is confounded with schedule, early stopping and GPU architecture (findings 3–4). |
| 76 of 360 `ERROR` rows are in beta multiplier and 52 in SRAM modes ([campaign](../accuracy/v776-evaluation.md#models--test-circuits--technology)). The limiter changes none of them. | These failures are not support-limited. Keep cold-start and state checks. |
| Off-current exceeds the reference in every scoreable row in both families, by a median of about two decades; 25/80 rows are too flat to fit ([round 3](../CHANGELOG.md#2026-09-25--v775-checkpoint-round-3-and-round-report-verification)). | Normalization floor (finding 1). |
| Inverter energy error splits by technology regardless of size or family: high on TSMC5/6/7, low on TSMC12/16 ([round 2](../CHANGELOG.md#2026-09-15--v775-checkpoint-targeted-round-2)). DirectNet LDO NRMSE is 78–149% on TSMC6/7. DirectNet small switch-cap fails hold droop. | Attribute energy to reference or harness first. Make LDO an explicit target. Use droop as the circuit-level leakage check. |
| The [limiter](../accuracy/v776-evaluation.md#nn-limiter-re-gate) recovers 71 rows, some inaccurately, and adds 56 non-convergences and cost. | Keep the limiter off as the scored control. |

Compare errors on the intersection of converged rows and report recovered rows
separately. Read maximum voltage error together with period, trip shift,
droop, regulation slope, energy and settling. Near-flat references inflate
NRMSE.

## Review round 1 findings

Artifacts are under `results/plan_review_20261001/`. Unless stated otherwise,
measurements use held-out combo-split rows.

The first-round tables below are historical diagnostics, not acceptance
numbers. Their evaluator reconstructed reference outputs from float32
normalized tensors; round 2 measured substantial sub-pA quantization error.
Recompute low-current tables from original float64 rows before using them
to select a scale or freeze thresholds.

**1. Off-current error is a normalization floor.** The table is for the
preserved DirectNet medium TSMC12 NMOS bundle (`i_d`, test split).

| \|I_ref\| (A) | 1e-13 | 1e-12 | 1e-11 | 1e-10 | 1e-9 | 1e-8 | ≥1e-6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Median log10(pred/ref) | +4.25 | +3.20 | +2.10 | +1.04 | +0.20 | +0.01 | 0.00 |
| Sign wrong | 0.47 | 0.44 | 0.45 | 0.45 | 0.26 | 0.02 | 0.00 |

- **The floor.** Below about 1–10 nA, the error grows by one decade per decade
  and the sign is random.
- **Cause.** The asinh scale `s_id` is the geometric mean of \|I\| (5.5–16 µA),
  so sub-µA currents are in asinh's linear band, where a normalized error of
  1e-3 is about 3e-8 A.
- **What earlier loss arms tested.** V6.5 only tried auxiliary losses on top of
  this transform. `SubthresholdIdLoss` has the same floor: it is linear below
  1 nA (`s2=1e-9`), its OFF ceiling is about 10× the median off current, and
  its default λ=0.05 once swamped the base loss by 12–30×.
- **Data is not the gap.** The `subvt_off` overlay is already in the control
  data (346k TSMC12 NMOS rows, at Vd=VDD only).

**2. Lowering `s_id` moves the floor.** Three DirectNet medium TSMC12 NMOS
retrains differ only in `s_id`. All share data, split, seed 42, recipe and
A100.

| Test-split `i_d` metric | Control (`s_id`=7 µA) | `s_id`=1e-10 A | `s_id`=1e-12 A |
|---|---:|---:|---:|
| Off state (\|Vgs\|<20 mV, Vds>0.3 V): median / p95 \|log10 err\| | 0.68 / 3.96 | 0.009 / 0.45 | 0.009 / 0.26 |
| Off state: sign wrong | 29% | 2.8% | 0.9% |
| \|I\|<0.1 nA: median \|log10 err\| | 3.39 | 0.13 | 0.06 |
| 0.1–10 nA: median \|log10 err\| | 0.58 | 0.008 | 0.008 |
| On, \|Vds\|<10 mV: median relative error | 1630% | 2.9% | 2.7% |
| \|I\|≥1 µA: median relative error | 0.40% | 0.62% | 0.64% |
| `i_d` linear NRMSE | 0.034% | 0.078% | 0.065% |
| `qd` / `qg` / `qb` NRMSE | 0.011 / 0.010 / 0.004% | 0.012 / 0.010 / 0.004% | 0.013 / 0.011 / 0.003% |

- **Leakage improves; the origin needs separate treatment.** Lowering `s_id`
  greatly reduces the sampled leakage error. Round 2 contradicts the broad
  claim that it fixes the Vds≈0 region: NGSPICE-backed probes find nearly
  zero on-state `gds` at Vds=0 and approximately 99% current underprediction
  at Vds=1 mV / +125 °C in both smaller-scale arms. Separate exact-zero
  values, nonzero near-origin currents, and origin slopes before advancing N.
- **Trade-off.** On-state relative error rises about 1.6×, and linear NRMSE
  about 2×. Charges are unchanged.
- **Status.** One seed on one cell, so this is a diagnostic, not a gate. The
  N arm must resolve this trade-off.

**3. Training is deterministic per GPU architecture; TSMC6 is not a variance
repeat.**

- **TSMC6 duplicates TSMC7.** The datasets and seed are identical, and 13 of
  16 bundle pairs have bit-identical weights.
- **Divergence follows the Blackwell GPU.** The three differing pairs each had
  one member trained on the RTX Blackwell. In the DirectNet large NMOS pair,
  validation is 0.0113 against 0.0038, which yields the Miller AC split of
  15.9 dB FAIL against 2.7 dB PASS.
- **The original run reproduces on A100.** The review control, retrained from
  current source on a different A100, matches the preserved bundle
  bit-for-bit (maximum \|Δw\| = 0).
- **There is no measure of training noise.** No seed spread exists for the
  current families.
- **Completion markers miss the run settings.** They omit seed, recipe,
  epochs, selected epoch, GPU and training commit.

**4. Early stopping truncates the cosine schedule.** The cosine horizon is
`max_epochs`, but patience stops many runs before annealing.

- DirectNet large TSMC5 NMOS selected epoch 112 at learning rate 9.5e-4 and
  stopped at 262 of 800. Its validation is worse than medium's.
- DirectNet medium was still improving at epoch 186–189 of 200.
- BSIM-AR small and medium TSMC5 NMOS selected epochs 12 and 33.

**5. Gate NFIN values are untrained.** Training uses NFIN {2,3,4,6,20.888} on
TSMC12/16 and {2,3,4,6,12} on TSMC5. Device sweeps test NFIN 5 and 10
(`tests/common/nn_sweep.py`).

**6. LDS weighting is inert.** The `1/count` clip at 0.01 saturates every bin
with at least 100 rows, so all six weights are 1.00 on 5.2 M rows. Training
uses plain MAE.

**7. Drain-to-gate current dominates hot off-state labels.** These are NMOS
rows at 398 K with Vgs≈0 and Vds>0.3 V, and `i_g ≈ −i_d` in each:

| Technology | Rows with \|Id\| > 1 µA | Median \|Id\| |
|---|---:|---:|
| TSMC5 | 95% | 2.7 mA |
| TSMC12 | 35% | 0.10 µA |
| TSMC16 | 32% | 74 nA |

At 300 K the share is about 1%. Earlier NGSPICE screens saw similar hot
references, but these rows are unverified. This current inflates `s_ig` and
would dominate an unstratified leakage metric.

## Review round 3 findings

These are read-only probes of existing data and artifacts, plus one NGSPICE
repeat of round 2's bin probe at NFIN 3 and 4. Scripts and logs are under
`results/plan_review_20261002_round3/`. Each is a diagnostic on the cells
named, not a gate.

**1. Both current references are quantized near 0.11 pA.**

- **Labels.** Every stored `i_d` label below 10 pA is an integer multiple of
  `1000 × ulp(Vd)`: 0.111 pA for 0.5 ≤ \|Vd\| < 1 V (46,647 of 46,647 TSMC12
  NMOS rows, 32,619 of 32,619 TSMC5) and 0.222 pA above 1 V.
- **NGSPICE.** Round 2's cold and nominal off-state references are integer
  multiples of 2⁻⁴³ A (0.114 pA), from 4 to 138 quanta. Labels and NGSPICE
  differ by under 2 quanta, which is the 0.4–22% difference round 2 left
  unresolved.
- **Tolerances cannot expose it.** Round 2's tightened repeat shifted by
  exactly zero.
- **Reach.** 4.6% of TSMC12 NMOS rows are below 10 pA, 2.9% below 1 pA, and
  34,314 are exactly zero. Finding 1's two lowest columns score a fit to 1–90
  quanta. `s_id`=1e-12 places that staircase in asinh's nonlinear band.
- **Circuit floor.** Physical GMIN adds about 1 pA per node, and NGSPICE's
  default ABSTOL is 1 pA.

The cause is untraced; the factor 1000 suggests a 1 mΩ internal series element.

**2. Round 2's dead band sits in a sampling gap.** The TSMC12 NMOS pool has
165,744 rows at exactly Vds=0 and 9,753 (0.15%) in 0 < \|Vds\| ≤ 5 mV. The
first structured level is 8 mV (54,000 rows).

- **Transform.** At `s_id`=1e-10 the raw target rises from about 0 at Vds=0
  to 11.6 at 8 mV, as sign(Vds)·ln\|Vds\|. The control's target is linear
  across the gap (0.75 at 8 mV), so it interpolates correctly without samples.
- **Slope.** The origin slope the network must represent is
  `gds / (s·σ_u)`: 39 /V for the control and 7.8e5 /V at 1e-10. The arm
  delivered about zero.
- **Trade-off.** The floor scales with `s` and the required slope with `1/s`,
  so their product is fixed by the normalized residual and `gds`. A scalar
  moves along one curve.
- **Control defect.** At Vds=0 and Vgs=0.8 V the control predicts −60.7 nA at
  27 °C and −118 nA at +125 °C against −1.45 and −1.62 pA. Both small-scale
  arms are within 1 pA. This is in round 2's `arm_probe.json` but unreported.

Round 2 therefore confounds the transform with coverage.

**3. The bin defect penalizes the NN, and bins fix the NFIN grid.** At nominal
lengths the PDK bins are {2}, {3}, {4, 5} and {6–12} on TSMC5 or {6–21} on
TSMC12/16.

- **Direction.** `nn_generate.py` resolves the actual bin, so on affected rows
  the NN is scored against parameters its labels never used.
- **Scope.** At NFIN=3 the cards differ in 162–264 parameters, yet `Id` agrees
  to four digits in all six cells at the one saturation bias probed. At
  NFIN=4 it differs by 13.3% (TSMC5 NMOS), 6.7% (TSMC16 NMOS) and 0.6–2.9%
  elsewhere. Every `get_baked_modelcard` caller with NFIN ≥ 4 is affected: the
  `nfin_high` corner and the `circuit_sweep` levels 5 and 10.
- **Grid.** Gate NFIN=5 shares its bin with one trained level (4). NFIN=10
  lies between the two trained ends of its bin. Parameters are piecewise by
  bin, so NFIN=5 cannot be interpolated from 4 and 6.

**4. Hot gate current is a whole-surface regime.**

| NMOS rows with \|Ig\| > 1 µA | 248 K | 300 K | 398 K | Max \|Ig\| |
|---|---:|---:|---:|---:|
| TSMC5 | 0% | 0% | 89.0% | 93 mA |
| TSMC12 | 1.1% | 1.1% | 35.2% | 32 mA |

- Finding 7 counted only off-state rows.
- `s_ig` is 9.2e-10 A on TSMC5 against 2.2e-11 A on TSMC12.
- Training has three temperatures, and every gate runs at −25, 27 or +125 °C.
  The transition between 300 K and 398 K is neither trained nor gated, yet
  the runtime accepts any temperature inside the box.
- TSMC cards are generated in-house from the PDK. Whether the hot regime is
  the compact model or the generated card is untested.

**5. The source-current error is a metric-scale artifact in this basis.** At
round 2's hot TSMC5 NMOS point the 4.34 µA source error is 0.42% of the
1.04 mA terminal scale, the same order as the gate error (5.1 µA). Matching
the −0.96 nA reference needs about 1e-6 relative accuracy on two independent
heads. No loss term reaches that; only a different current basis could.

**6. Items with no owner in revision 3.**

- Finding 6 (inert LDS weights) has no action.
- The published matrix contains Blackwell-trained bundles, and the same data
  and seed gave FAIL against PASS. Markers do not record which.
- BSIM-AR replication (4 seeds × control and candidate × 2 polarities × 2
  technologies) is 32 bundles, or 320–704 GPU-h per arm at `large`. The whole
  V7.7.2 matrix took 658.
- Promotion requires opamp AC, where DirectNet is 0/5. The first model phase
  targets leakage, which has no gate.

## What previous work establishes

- **V7.6.4 rejections.** The
  [V7.6.4 loop](../CHANGELOG.md#post-v763--v764-closure-loop-and-cleanup-2026-08-29-to-2026-08-31)
  rejected all of the following:
  - Hermite supervision;
  - normalization anchoring;
  - exact OSDI current-Jacobian fine-tuning;
  - reference-seeded loss;
  - distillation;
  - local adapters.

  It bound tails at 1.02×.
- **Removed derivative losses.**
  [V7.7.0](../CHANGELOG.md#v770--retire-reduced-compact-model-families-2026-09-04)
  removed derivative heads and their losses.
- **Retired-family priors.** These results are priors, not qualification
  evidence:
  - id-Sobolev improved `gds` from 55.8% to 1.7% but collapsed the opamp on
    4/4 seeds;
  - the subthreshold loss fixed the weak band but collapsed SRAM bistability
    in 6/7 cells
    ([V6.4.7](https://github.com/ShenShan123/PyCircuitSim/blob/376b411/docs/plans/2026-06-10-directnet-v6.4.7-accuracy.md));
  - charge-Sobolev and the µA-band retune were killed, while autograd
    capacitances already matched OSDI to 0.3–2.5%
    ([V6.5](https://github.com/ShenShan123/PyCircuitSim/blob/7112f2c/docs/plans/2026-06-22-v6.5-accuracy-and-xl.md)).
- **Checkpoint selection.** A mean physical score once selected epoch ~10
  ([bug](https://github.com/ShenShan123/PyCircuitSim/blob/f834aae/docs/plans/2026-05-03-phys-best-tracker-bug.md)).
  Use per-surface medians, and replicate finalists.
- **Exposure bias.** The autoregressive/teacher-forced error ratio was
  1.02–1.05 at large/XL
  ([V7.4.2](https://github.com/ShenShan123/PyCircuitSim/blob/64ecd33/docs/plans/2026-08-10-v742-bsimar-capacity.md)).
- **Overlays.** An overdose regressed other technologies, overshoot labels
  caused Newton runaway, and overlays need the combo-split contract
  ([post-V7.6.4](../CHANGELOG.md#post-v764--level76-simple-circuit-recovery-2026-09-01)).
- **Instruments.** L0–L4 come from the
  [coverage plan](2026-09-02-v767-evaluation-coverage.md); provenance rules
  from [V7.7.2](2026-09-05-v772-full-retraining.md). The
  [fast subset](2026-09-28-nn-evaluation-fast-subset.md) is not a launch
  option yet.

## Phase 0 Reproducibility and noise floor

1. **Pin the hardware.** Pin one GPU architecture (A100, by UUID) and record
   torch, CUDA, driver, TF32 and deterministic-algorithm settings. First
   duplicate one same-seed run under those exact settings. One bit-identical
   reproduction does not establish determinism for every family and size.
   Changing precision/determinism policy is a separate control arm.
2. **Extend completion markers.** Record seed, preset and overrides, SWA/AMP,
   epochs run, selected epoch, early-stop epoch, GPU, training commit and
   dirty flag, optimizer-step count, split-map digest and normalization-rule
   version. Record generator, training and evaluation commits separately.
   Verify one new bundle through the evaluation manifest before launching
   the matrix; never alter an old dataset's source identity to bypass checks.
3. **Control rule.** The control is retrained from its arms' commit. Preserved
   V7.7.2 bundles remain the published baseline only; verify that the current
   source reproduces them. Evaluation pins every
   bundle through `PYCIRCUITSIM_NN_CHECKPOINT_{DNF,TFF}_{NMOS,PMOS}`, because
   automatic resolution prefers `large`.
4. **Measure seed spread before freezing thresholds:**
   - DirectNet medium: TSMC5 and TSMC12 × both polarities × 4 seeds;
   - BSIM-AR large: 2 seeds on TSMC5 NMOS.

   Score device metrics and pilot circuits, with ring and opamp at OMP 1/2/4.
   Use paired seeds 42/7/17/123 with the same persisted split; CLI training
   seed and the loader's default split seed (42) are different controls.
   Two BSIM-AR NMOS seeds are a feasibility screen, not a family noise bound.
   Before advancing a BSIM-AR arm, run its own four-seed control and candidate
   for both polarities and pilot technologies. Do not transfer DirectNet's
   spread to BSIM-AR or treat TSMC6/7 as independent technologies statistically.
   That rule costs 320–704 GPU-h per arm at `large`
   ([open decision 1](#open-decisions)); no BSIM-AR arm starts before it is
   budgeted.

## Phase 1 Attribution

**Pilot scope.**

- DirectNet medium and BSIM-AR large, both polarities, on TSMC5 and TSMC12.
- DirectNet large on TSMC12, to observe high-NFIN.
- Confirm finalists on TSMC7 and TSMC16.
- Keep TSMC6 in denominators, recorded as an identical retrain.

**Freeze per arm, under `results/nn_accuracy_20261001/`:** source commits,
split, normalization, recipe, seeds, GPU, cases, references and runtime flags.

1. **Failure ledger.** Start from the V7.7.5 round reports at `bde2c11`. Assign
   each row one kind: support rejection, non-convergence, converged mismatch,
   or unavailable metric. Keep infrastructure failures apart. Tag each row
   by code, before reading it, with:
   - the bin defect (a legacy default-bin `get_baked_modelcard` reference at
     non-default NFIN, including 3; independently resolved role cards are exempt);
   - the training GPU of each bundle, from the V7.7.2 logs, or `unknown`;
   - the temperature.

   A row carrying the bin defect or a Blackwell bundle is not attribution
   evidence until re-run.
2. **Reproduction.** Reproduce representative failures with deck parity and a
   LEVEL=72 control against NGSPICE. Cover:
   - Miller AC;
   - high-NFIN DC;
   - off-current;
   - inverter energy;
   - LDO regulation;
   - beta multiplier;
   - SRAM modes;
   - an L4 transient without `uic`.
   Audit the **resolved model parameters**, not just the rendered instance
   geometry. Before the V7.7.7 repair,
   `tests/common/base.py:TechProfile._resolve_tsmc_modelcard` used
   `default_nfin`; `circuit_benchmarks.get_baked_modelcard` subsequently baked
   the requested fin count into that card. Round 2 measured a 15.7% NGSPICE
   drain-current discrepancy at TSMC5 NMOS NFIN=10 versus resolving the card
   at NFIN=10 first, while existing physical deck parity passed. Repair and
   separately re-baseline affected circuit references before attribution.
   The `nn_gate.py` parametric device resolver already passes actual NFIN;
   this finding does not retract its high-NFIN misses. Training labels use
   the actual bin, so the defect counts against the NN
   ([round 3, finding 3](#review-round-3-findings)). The V7.7.7 linear/off
   extension finds an off-current mismatch at NFIN=3, so affected `pn_ratio`
   and `nfin_delta` cases require re-baselining too; saturation agreement
   does not clear them.
3. **Operating points.** Archive complete operating points, including branch
   currents, behind AC and non-`uic` transients, and re-stamp them on a fresh
   parse.
4. **Miller AC.** Compare terminal I/G/C at the NGSPICE operating point and at
   the candidate's own operating point. AC evaluated at a forced reference
   state is diagnostic only.
5. **Energy.** Check integration window, sign convention, load energy and step
   sensitivity in both engines. If LEVEL=72 reproduces the bias, route it to
   harness work.
6. **Hot labels.** Round 2 independently reproduced approximately 1.04 mA
   NMOS and 2.75 mA PMOS TSMC5 drain-to-gate leakage at +125 °C on the same
   OSDI model. Keep these rows in the existing scope; omitting them would
   redefine the task. This establishes model-reference agreement, not silicon
   realism or validation of every hot bin. Expand the audit by VT, L, NFIN
   and bias. Sweep temperature from 27 to +125 °C at fixed bias in NGSPICE to
   locate the gate-current onset. Compare the generated card's temperature
   parameters with the PDK source; a generation defect would be a reference
   repair, not a scope change.
7. **Measurement prerequisites.** Retain raw float64 outputs and stable row
   IDs through splitting; never build a reference by inverse-transforming
   float32 training tensors. Stratify by polarity, temperature, VT and
   geometry, and separate exact zero, signed nonzero near-origin, weak,
   reverse and on-state samples. Freeze reference-resolution floors in
   quanta of `1000 × ulp(Vd)`, not from tolerance repeats, which return a
   zero shift. Proposed values: percent metrics at 200 quanta or more (about
   22 pA at 0.5–1 V), log-decade metrics at 20 or more, and absolute error in
   quanta below that. Never use an arbitrary `1e-30` log denominator. Record
   the rule in the methodology before the leakage arm.
8. **Circuit demand census.** From the NGSPICE operating points and waveforms
   of every failing and `ERROR` row, tabulate per-device \|Id\| and \|Vds\|.
   Report the share of devices below the control's floor (about 30 nA) and
   inside \|Vds\| < 8 mV. This sets the floor leakage work must reach, and
   decides whether Phase 2 precedes the Miller AC work.
9. **Polarity swap.** Re-run representative failures with one polarity at
   LEVEL=72 and the other on the NN, if the renderer accepts mixed levels.
   This localizes a circuit miss without retraining.

Freeze weights, budgets, floors and stopping rules before each arm starts.
Unresolved attribution stays unresolved.

## Phase 2 Leakage through normalization

| Arm | Single change | Question |
|---|---|---|
| O | Add a log-spaced drain-origin overlay (±10 µV to ±5 mV, both signs) under the overlay split contract. | Does the control's 60–118 nA zero-bias offset fall while its origin conductance holds? O becomes the control data for N. |
| N | Set one scalar `i_d` asinh scale per bundle by a declared train-only rule, on O data. | Where on the floor-against-origin-slope curve does a scalar sit, and is any point acceptable? |
| T | Change the transform structure under its own runtime contract. Candidates: a bias-dependent scale, or a Vds-odd factor plus residual. | Runs only if N's curve has no acceptable point. |
| N-g | Apply N to `i_g` and `i_b`, only if their per-temperature errors justify it. | Does gate and body leakage improve? |
| L1 | Add `SubthresholdIdLoss` (λ ≲ 0.002) on top of N, only for a weak-inversion residual. | Does a value term still add anything? |

- **Before any training.** From the three existing arms, compute the
  normalized residual by region and the implied floor and origin slope for
  each scale. This costs no GPU time and predicts N's curve.
- **Scale bounds.** The census sets the floor to reach. Do not go below a
  scale that puts the 0.11 pA quantum in asinh's nonlinear band.
- **Precision.** Record AMP state. A bf16 forward resolves about three digits
  of normalized output, so it must be off or matched across N arms.

- **No runtime change for scalar N.** The runtime reads `asinh_scale` from
  the bundle. Per-fin or temperature-dependent scales are separate transform
  and runtime arms; existing sidecars store one constant per output. Train N
  from scratch. `--init-from` copies weights without reconciling normalization,
  so it is not a valid same-function warm start across changed transforms.
- **Compare physical test metrics, not validation MAE:**
  - log error by decade above the frozen reference-resolution floor;
  - sign error;
  - \|Vds\| < 10 mV and reverse-Vds error;
  - on-state error;
  - all four terminal-current and charge tails, including derived `i_s`/`qs`;
  - autograd `gds`/`gm` at Vds≈0 against OSDI.
  Resolve near-origin conductance independently with NGSPICE AC and signed
  DC sweeps. Exact-zero samples must not dominate a median that claims
  nonzero linear-region accuracy. Add log-spaced nonzero drain biases and
  report them separately from Vds=0. The normalization arm cannot advance
  on the existing aggregate near-zero metric alone.
- **Cancellation check.** Score the source current and source Jacobian row
  against reference values, not only KCL. Round 2 found sub-percent drain/gate
  errors but thousands-fold, wrong-sign source-current errors in hot TSMC5
  controls. Report cancellation-conditioned groups. Score each terminal
  error against the largest terminal current at that bias, not against the
  cancelled value; the thousands-fold figure is 0.42% on that scale. A
  derived-source loss cannot reach the 1e-6 head accuracy the cancelled
  value needs, so it is not registered. A current-basis arm is registered
  only if the census shows a circuit that depends on the cancelled source
  current.
- **Early veto.** Run the signed near-origin sweep before seed expansion or
  circuit campaigns. At minimum use Vds=0 and ±10 µV, ±100 µV, ±1 mV, ±5 mV
  at cold/nominal/hot, NMOS/PMOS, several gate biases, and trained/interpolated
  geometries. Reject negative-conductance regions where NGSPICE is positive.
  Current derivative **validation** is mandatory here even though derivative
  **supervision** remains conditional in Phase 5.
- **Binding vetoes inside the phase:** SRAM hold and modes, switch-cap droop,
  and inverter VTC. A more accurate weak band has removed SRAM bistability
  before.

## Phase 3 Schedule and selection

| Arm | Single change | Question |
|---|---|---|
| R | Cosine horizon equal to the actual budget: fixed epochs, or patience no shorter than the horizon. | Do annealed checkpoints remove size non-monotonicity and BSIM-AR early plateaus? |
| S | Physical-metric selection among per-epoch EMA snapshots from the R or control run. | Does selection alone improve leakage and worst-group error? |

- **Order.** Re-rank sizes only after R.
- **Budget isolation.** First compare shortened cosine annealing with the
  control at equal optimizer-update count; disabling early stopping for the
  original full horizon is a separate extra-compute arm. Record wall time,
  updates and EMA update count. Equal epochs cease to mean equal compute
  after a coverage expansion.
- **Bundles.** Every S candidate becomes a full bundle with a marker.
- **Selection data.** Select on validation only, never on qualification
  circuits. Freeze per-surface/group floors, tail constraints and tie-breaks;
  medians alone can hide the origin or cancellation failure. Already inspected
  test rows are development evidence; reserve untouched confirmation groups
  before further scale/selection search.
- **Separation.** Keep loss-arm selection identical to its control. Combine
  winners later.

## Phase 4 Coverage

- **NFIN.** Choose levels by bin
  ([round 3, finding 3](#review-round-3-findings)). Bins {2}, {3} and {4, 5}
  hold no unseen integer once 5 is trained, so add 5 and claim no
  generalization there. Inside the top bin, train 6, 10 and the maximum, and
  hold out 8 (and 16 on TSMC12/16) as untouched confirmation. Confirm the bin
  map at every trained L first; round 3 checked 16 and 20 nm. Keep the
  existing gates for compatibility. Never move existing gates onto a new grid
  or split nearby duplicate rows across partitions.
- **Temperature.** Either add a level and a gate between 27 and +125 °C, or
  make the runtime reject temperatures other than the three trained ones
  ([open decision 2](#open-decisions)). The onset sweep in Phase 1 informs
  the choice.
- **Voltage.** The drain-origin overlay is arm O in Phase 2. Extend other
  voltage regions only for defects that remain after N. `subvt_off` currently
  scans Vd=VDD only.
- **Bounds.** Do not widen normalization bounds for Newton excursions. Keep the
  NFIN=1 and invalid-OSDI exclusions.
- **Data hygiene:**
  - bump `generator_release`;
  - use the overlay split contract;
  - deduplicate;
  - keep (tech/VT, NFIN, L, T) groups intact;
  - persist the original group-to-partition map and row identities; assign
    new groups without reshuffling old ones, and retain an untouched
    confirmation set. A fixed RNG seed does not freeze the split: round 2's
    NFIN=8 expansion moved 553,454 of 646,830 original test rows to training;
    extra rows at existing NFIN=6 groups also changed assignments;
  - generate from a clean tree.

  Simple-v2 stays out of gradient training and checkpoint selection. Any
  circuit consulted repeatedly for arm acceptance is a development sentinel,
  not an untouched confirmation experiment. Preserve a separate, frozen
  confirmation inventory and describe this distinction in the report.

## Phase 5 Derivatives (conditional on attribution at matched bias)

- **Labels.** Add checksum-bound derivative labels for all six surfaces from
  `condense_last_jacobian()` and `condense_last_react()`.
- **Silent PyCMG fallbacks** must be rejected:
  - zero capacitances on a singular matrix;
  - an external-only Jacobian;
  - opvars defaulting to 0.0.
- **Validation.** Validate against NGSPICE with relative tolerances. The
  existing 1e-6 S absolute check leaves leakage derivatives untested.
- **G arm.** Add a current-Jacobian term, targeting Miller gain and bias
  shift.
- **C arm.** Add a charge-Jacobian term, targeting AC phase, energy, delay and
  droop.
- **Loss mechanics.**
  - Use physical-unit floors.
  - BSIM-AR needs rollout.
  - Use no bf16 AMP, because of double backward.
- **Stop rule.** Stop if circuit targets do not improve or value tails
  regress.

## Phase 6 BSIM-AR strategy and convergence

**Autoregressive comparison.** After R, compare six and three targets, each
with teacher forcing and with rollout. Keep data, budget, seed, selection and
GPU fixed. Three-target mode reads currents from the `qd` hidden state, so
report it as an architecture change. Record CPU cost.

**Solver arms.** Evaluate improved weights flags-off first, then with the
limiter on. Retain iteration count, support excursions, limiter activity,
residual trend, retries and final-state error. Any basin-changing arm must
pass the latch-basin contract. Physical GMIN, tolerances and support
acceptance stay intact.

## Experiment order and acceptance

1. Phase 1 reference, split and metric prerequisites, including the bin
   repair and the quantum floor rule.
2. The ledger, census and polarity swap. Decide from them whether leakage or
   Miller AC is the first model target.
3. Phase 0.
4. The zero-training trade-off calculation, then O, then a signed-origin and
   four-terminal screen of N on DirectNet medium. Replicate only survivors.
   Open T only if N has no acceptable point. Transfer winners to BSIM-AR
   large once its replication is budgeted.
5. R, then S.
6. Coverage arms.
7. Conditional G, C and the autoregressive comparison.
8. Convergence.
9. Combine winners last.

**Pilots and inference.** Each arm surviving the diagnostic screen runs both polarities with four seeds
against a same-commit, same-GPU control. Report every seed. Scored inference
runs on CPU with one OMP/MKL/Torch thread
([workflow](../../README.md#nn-workflow-from-the-project-root)).

Freeze the following objectives numerically from the Phase 0 spread before
training. They do not change published gates.

| Decision | Objective |
|---|---|
| Leakage arm advances | Median off-state log error below 0.5 decades per temperature above the frozen quantum floor, a reduction larger than the control seed range, and no worse p95. Zero-crossing, on-state and charge errors stay within the control spread. |
| Origin (O, N and T) | Origin `gds` within the control spread of NGSPICE at all three temperatures, no negative `gds` where NGSPICE is positive, and \|Id\| at Vds=0 no worse than the control. |
| R or S advances | Worst-group error and size ranking improve beyond the spread, with no tail regression. |
| G or C advances | Held-out derivative error falls beyond the spread, and the registered circuit error falls. |
| Every model arm | No new qualification failures (ring and opamp at OMP 1/2/4) and no new support or closure failures. Tails stay within 1.02× control plus a frozen floor unless the spread justifies more. Check per polarity and per technology. |
| Convergence arm | More converged rows, none lost, and recovered rows meet predeclared physical criteria. |
| Promotion | All declared qualification gates pass, including opamp AC. |

- **Diagnostics.** Leakage and derivative criteria are diagnostics; no such
  gate exists.
- **Denominators.** Record each pilot denominator with the
  `v710_regate_jobs.py` digest.
- **Negative results.** If a mechanism fails, publish the ledger and stop it;
  never relax thresholds after the fact.

**Cost.** The V7.7.2 training matrix took 658 GPU-h: DirectNet 0.1–3.1 h per
bundle, BSIM-AR large 10–22 h, and BSIM-AR XL 27–66 h. Evaluation takes 2,067
BSIM-AR cell-hours against 31 for DirectNet. Run each mechanism on DirectNet
medium first (under 15 GPU-h per four-seed arm), and move only winners to
BSIM-AR. One fully replicated BSIM-AR large arm costs 320–704 GPU-h.

## Final verification and deliverables

1. Run device, leakage-by-temperature, derivative, canary and inverter checks
   for both polarities. Follow with L1–L4.
2. Qualify each frozen candidate across five technologies, including OMP
   1/2/4 repeats and registered corners (high NFIN, temperature and body bias,
   asymmetric geometry, heavy loads).
3. Before replacing the clean reports, train the full matrix with a uniform
   recipe on one GPU architecture.
4. Report convergence and `ERROR` counts separately from MRE, R², NRMSE,
   maximum voltage error, leakage decades, derivative tails and runtime, with
   paired and recovered rows separated.
5. Preserve artifacts under `results/`, measurements in `docs/accuracy/`, and
   outcomes in `docs/CHANGELOG.md`.

Ownership:

| Change | Owner |
|---|---|
| Normalization | `data/normalize.py` |
| Labels | `pycmg/nn_generate.py` and `data/` |
| Losses | `losses/bni_mae.py` |
| Schedule, snapshots, selection and markers | `training/trainer.py` and the CLI |
| Comparisons | `tests/common/` |

Runtime edits need their own provenanced arm.

## Decisions after review round 2

1. Neither `s_id=1e-10` nor `1e-12` advances as-is. Choose an intermediate
   scalar rule only after a raw-label, signed-origin screen; runtime-dependent
   scaling is a separately budgeted proposal. No on-state regression budget
   has been approved by these diagnostics.
2. Hot leakage stays in scope against the stipulated OSDI reference. Add
   source-current cancellation as an explicit target; silicon realism remains
   outside what these experiments establish.
3. Pin and record the existing training numerics, verify same-seed repeats,
   and measure paired seed spread; do not infer universal determinism.
4. Choose geometry additions after correcting reference-bin resolution and
   freezing partition membership. Keep TSMC6 in the published denominator;
   independent seeds are separately named repeats, not extra technologies.
5. BSIM-AR needs its own replicated, both-polarity control spread. Schedule,
   selection, new geometry training and circuit-level benefits remain untested
   by round 2; do not call those proposals proven.

## Decisions after review round 3

These amend the round 2 decisions where they conflict.

1. Decision 1 changes. An intermediate scalar is no longer the default
   remedy. Close the origin sampling gap first (arm O), predict the scalar
   trade-off from existing arms, and open a structural transform (arm T) only
   if no scalar point is acceptable.
2. Decision 2 changes. Source-current cancellation is scored on the terminal
   scale and is not a loss target. Hot gate current stays in scope pending the
   temperature-onset and generated-card check.
3. Decision 4 changes. NFIN levels follow the bin map, and unseen-NFIN
   confirmation exists only inside the top bin.
4. Low-current metrics use a quantum floor. The sub-pA columns of round 1 are
   not targets.
5. Leakage keeps first place only if the census shows failing circuits that
   operate below the control's floor or inside the origin gap. Otherwise the
   Miller AC attribution leads.
6. LDS weighting stays inert and is not a leakage lever. Remove it only as a
   separately verified bit-identical cleanup.
7. The bin-defect repair, the generator change for arm O and any runtime
   temperature check are code changes outside this document.

## Open decisions

1. **BSIM-AR replication budget.** Fund 320–704 GPU-h per arm at `large`, or
   replicate at a smaller size and confirm one seed at `large`. Recommended:
   the smaller size, with the size named in every report.
2. **Temperature support.** Add an intermediate level and gate, or restrict
   the runtime to the three trained temperatures. Recommended: restrict now,
   and add a level only if the onset sweep shows a smooth transition.
3. **Hot TSMC5 scope.** If the generated card proves faithful to the PDK, keep
   the 89% hot gate-current rows as the stipulated reference, or train
   +125 °C TSMC5 as a separate bundle. Decide after the onset sweep.
