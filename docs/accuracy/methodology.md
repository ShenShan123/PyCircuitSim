# Accuracy methodology

This file defines the shared scoring contract for `docs/accuracy/`. Release
outcomes and retractions belong in [`docs/CHANGELOG.md`](../CHANGELOG.md);
family-specific measurements belong in the corresponding `*-clean.md` report.

## 1. Ground truth and scope

The reference is **NGSPICE using the identical BSIM-CMG LEVEL=72 OSDI model**
(`/usr/local/ngspice-45.2/bin/ngspice`, selected through `NGSPICE_BIN`). Reference
and candidate use the same netlist, modelcard, geometry, sources, options, and
analysis limits; only the MOSFET model changes.

Simplified equations, hand-written approximations, and PyCircuitSim output are
not independent references. LEVEL=72 is the yardstick, not a graded family.
The graded NN families are DirectNet-Full (LEVEL=75) and BSIM-AR-Full
(LEVEL=76). LEVEL=75 is the default runtime family. Both use
TSMC5/6/7/12/16 and the `small`, `medium`, `large`, and `xl` tiers. Retired
LEVEL=73/74 measurements remain historical evidence only. Their retired reports
are recoverable from the revision recorded in the
[V7.7.5 cleanup ledger](../CHANGELOG.md#v775--repository-cleanup).

## 2. Gates

The authoritative verdict is the verification script's exit code, not a
printed metric interpreted by eye.

| simple-v1 qualification gate | pass condition |
|---|---|
| `verify_circuit_ring_osc.py` | period error ≤ 5% |
| `verify_circuit_opamp.py` | open-loop DC gain error ≤ 10%; trip shift is diagnostic |
| `verify_circuit_sram_snm.py` | every lobe is positive and lobe NRMSE ≤ 10% at every NFIN corner |
| `verify_circuit_switchcap.py` | transfer error ≤ 5% of VDD and droop ≤ max(10% of reference droop, 0.1% of VDD) |

The simple-v1 score is 4 circuits × 5 technologies = **20 cells per tier**.
Reports before V7.3 used four electrically distinct technologies and `/16`;
rescale before comparing totals.

Campaign suite IDs use the canonical `verify_circuit_*` module names. Retired
aliases are not accepted because an implicit rename can hide a misspelled or
missing gate.

| device gate | pass condition or purpose |
|---|---|
| `verify_nn_dc.py` / `verify_nn_inverter.py` | resolver-path single-device and inverter checks |
| `verify_nn_multi_tech_dc.py` | Id–Vgs NRMSE < 10% for every L/NFIN/VT configuration |
| `verify_nn_multi_tech_tran.py` | inverter transient over the same sweep |
| `verify_nn_ac.py` | gain error ≤ 1.5 dB, f3dB ratio 0.7–1.43, magnitude NRMSE ≤ 10%; phase is diagnostic |
| `verify_circuit_opamp_ac.py` | gain error ≤ 3 dB, GBW ratio 0.6–1.67, PM error ≤ 15°, valid refined reference bias, converged NN operating point |
| `verify_nn_lifted_source_dc.py` | source-relative-frame canary, NRMSE ≤ 10% |
| `verify_device_integrity.py` | diagnostic output/subthreshold/linear-region and `gm`/`gds`/`gmb` accuracy |
| `verify_terminal_integrity.py` | diagnostic four-terminal current/KCL and 4x4 transcapacitance accuracy |
| `verify_nn_subckt.py` | diagnostic flat-versus-nested NN DC, transient, and AC equivalence against LEVEL=72 |

### Simple-v2 topology diagnostics

`verify_circuit_topologies.py` adds a held-out composition ladder covering
single-stage, logic, transmission-gate, differential, active-load, self-bias,
stateful, scale, and closed-feedback behavior. It runs OP, DC, transient, and
AC analyses from canonical templates in
`circuit_templates/`. Each case, its tier, signals, and domain metrics are
declared in `tests/common/simple_circuit_catalog.py`; the tiers are defined in
[`circuit_templates/README.md`](../../circuit_templates/README.md).

Simple-v2 rows are **diagnostics**, are held out from training, and do not
change the historical simple-v1 `/20` denominator. Numerical mismatches
remain diagnostic; an uncharacterizable requested cell is an explicit
`ERROR`.

- **Derived metrics.** CMRR and PSRR come from a pair of AC experiments, so a
  case naming `derived_metrics` emits one extra `analysis="derived"` row. If
  an input sweep errored, the derived row is itself an `ERROR`.
- **Unconverged rows.** A row that does not converge stays an `ERROR`. It may
  carry an `unconverged_diagnostic` payload from one rerun without the
  convergence requirement; no scoring path reads it.
- **Corners.** Nominal is the default. The declared stress matrix adds
  `temp_cold` (−25 °C), `temp_hot` (125 °C), `vdd_low` (0.85×), `vdd_high`
  (1.10×), `body_reverse` (0.10×VDD), the fin-ratio corners `pn_n3p2` and
  `pn_n2p3`, `joint_hot_lowvdd`, `vt_alternate`, `vt_asymmetric`, `ln_20`,
  `lp_16`, `nfin_high`, `slew_slow` and `load_heavy`. A corner that does not
  apply to an analysis, or renders a deck identical to nominal, creates no
  row.
- **AC phase.** Every AC analysis reports `phase_maxerr_deg` next to the
  magnitude NRMSE.

Promote simple-v2 into a new score version only when all of these hold:

1. LEVEL=72 produces three repeatable, complete traces at every proposed cell.
2. Each gate answers one declared question with a frozen threshold and unit.
3. The exact case/technology/corner/analysis denominator is immutable.
4. Topology parity, artifact completeness, checkpoint hashes, commit and CPU
   thread settings are recorded in one campaign.
5. No partial log or infrastructure failure counts as a pass or leaves the
   denominator.

## 3. Determinism and execution

Ring and opamp cells are run with

```text
OMP_NUM_THREADS = MKL_NUM_THREADS = PYCIRCUITSIM_TORCH_THREADS ∈ {1, 2, 4}
```

A strict pass must pass all three settings. A mixed result is a **FLIP** and
counts as failure. SRAM and switchcap use one pinned run. Gates also set
`OMP_WAIT_POLICY=passive` and `KMP_BLOCKTIME=0`.

The scored axis is CPU-only (`CUDA_VISIBLE_DEVICES=""`). CUDA and other
floating-point-perturbing optimizations require separate fidelity gates and
never replace the CPU score.

Every cell receives an isolated result directory below `results/`. Parallel
jobs must not share `PYCIRCUITSIM_SIMPLE_RESULTS` or
`PYCIRCUITSIM_NN_RESULTS`.

## 4. Reported metrics

Device AC scores `/10` (NMOS and PMOS × 5 technologies); opamp AC scores
`/5`. Reports include per-technology MRE, R², NRMSE, and maximum voltage error.
Charge-sensitive AC gates use the full autograd charge matrix and may move
independently of DC accuracy.

### Inverter supply energy (V7.7.7)

The `inverter_energy` diagnostic measures net energy delivered by the constant
VDD source: `E = -VDD × integral(i(Vdd), dt)`. Both engines define positive
voltage-source current into the source, so returned charge subtracts from
delivered energy. This is supply energy over the declared observation window,
including its initial state; it is not total dissipation or input-driver energy.

Integrate each engine's original samples with the trapezoidal rule over the
same overlapping time window. Interpolate only the two window boundaries.
Do not integrate the at-most-600-point trace-comparison grid: it can alias
narrow current pulses. Do not rectify current before integration. Preserve
step-sensitivity results separately from the scored/default step settings.
V7.7.6 and earlier energy numbers use the former rectified, resampled metric
and cannot be compared directly with this definition. The
[V7.7.7 attribution report](2026-10-03-v777-phase1-attribution.md) records the
correction; historical reports retain their original contract.

### Low-current diagnostic resolution

Use original float64 reference values and preserve their source row identities;
inverse-transforming float32 training targets is not a reference. Declare the
reference quantum `q` before a leakage arm and record it with each result.
The reviewed labels exhibit `1000 × ulp(Vd)` quantization and the audited
NGSPICE fixtures exhibit a roughly `2^-43 A` quantum. These measurements are
scoped to the audited biases, not a universal bound for every terminal or
geometry. Use the larger applicable measured quantum when comparing them;
calibrate uncovered regions before registering an arm.

- Report percent current error only for `|I_ref| >= 200q`.
- Report log-decade error only for `|I_ref| >= 20q`, with zero predictions
  counted explicitly as unresolved log errors, never hidden by `1e-30`.
- Below those floors, report absolute error in amperes and quanta. Separate
  exact zero, signed near-origin, reverse, weak and on-state groups; report
  sign mistakes separately.

At `q = 2^-43 A`, the percent and log thresholds are approximately 22.7 pA
and 2.27 pA. Temperature, polarity, VT and geometry remain separate strata.
These diagnostics do not change existing qualification gates.

Simple-circuit workers emit schema-stable `GateResult` JSON markers containing
case, technology, corner, analysis, role, convergence state, aggregate trace
metrics, and domain metrics. The collector consumes these markers before
falling back to legacy human-readable regexes. Multi-signal traces retain
signed source currents; transient comparisons may additionally report
phase-aligned NRMSE without replacing the unaligned metric.

## 5. Evidence validity

A report is publishable only when all of the following hold:

- One complete campaign supplies every denominator; partial passes are never
  combined.
- A recipe uses one uniform addendum across its declared technology, device,
  and tier matrix; per-cell tuning is a different experiment.
- The checkpoint manifest, gate commit, model family, tier, technology,
  device, and thread settings are recorded.
- Each checkpoint has its model, normalization, `*.complete` marker, and any
  family-required architecture/config sidecar. A lone `_best.pt` may be from
  an interrupted run.
- Explicit checkpoint pins resolve or fail loudly; automatic fallback cannot
  replace a missing pinned checkpoint.
- Invalid arguments, missing dependencies, Python tracebacks, and unavailable
  references are infrastructure failures, not scientific FAILs.
- An `ERROR` row remains in the denominator while numeric aggregates use only
  rows containing valid metrics.
- If a converged candidate SRAM never switches in a write-margin sweep where
  the reference does switch, its trip metric is absent. Keep a candidate
  `ERROR` with `candidate_converged=true`, no scored metrics and preserved raw
  traces. This is a scientific missing event (exit 1), not schema corruption.
  An absent reference trip still fails reference/metric validation; do not
  turn an uncharacterizable reference into a candidate verdict.
- A complete, converged reference may lack an identifiable diagnostic metric:
  for example, a hot device whose current does not rise from its off-state
  value has no window for the declared subthreshold-slope fit. These rows use
  `error_kind=reference_metric`, keep `ERROR` status and their denominator slot,
  and carry no scored metrics. Recovered values remain under
  `uncharacterized_diagnostic`. This is distinct from a failed reference solve,
  missing data, or a malformed metric payload, which remain infrastructure
  failures. Qualification rows cannot use this diagnostic-only category.
- Reference and candidate decks are rendered and compared before a numerical
  mismatch is attributed to the model.

The current clean pool contains DirectNet-Full and BSIM-AR-Full × 4 tiers × 5
technologies, with every catalog and device suite generated from one source.
`scripts/v710_regate_jobs.py` is the denominator source of truth; record its
job count and digest with every campaign. Older 480- and 600-job campaigns are
not interchangeable when their suite sets differ.
Report generation fails closed unless the applicable matrix and checkpoint
artifacts are complete. The shared builder emits clean LEVEL=75/76 reports;
named recipe evaluations remain separate experiments under their run roots.
Package maintenance versions do not change a report's campaign or source
identity and do not imply that an unfinished evaluation has completed.

## 6. Comparability

A gate result belongs to a specific checkpoint, solver commit, and gate
contract. Re-gating fixed weights is comparable only when those inputs match;
retraining the same recipe is stochastic.

Retired-family and pre-full-terminal results are not current-family evidence.
The detailed comparability history remains in the changelog and dated reports.

## 7. TSMC6 controlled repeat

TSMC6 is TSMC7 relabelled in this LEVEL=72 flow: their training arrays and
electrical response are identical. It remains in the five-tech denominator as
a controlled repeat, not an independent ground truth. Differences between its
NN verdicts and TSMC7's measure training and Newton-basin variability.

## 8. Measurement caveats and harness corrections

The retained solver and coverage-audit contracts are:

- Residual probes recover ideal-voltage-source branch currents and scale the
  tolerance from current-valued node rows, so they measure the complete MNA
  residual. The topology-stable fit is cached without changing the result.
- Opamp AC locates the maximum-gain bias with an NGSPICE physical fine sweep
  at 0.1 mV resolution, then uses that same input bias for both engines. It
  requires an off-rail reference and a converged NN operating point.
- Family labels and CLI validation fail closed; an invalid interpreter,
  technology, device, or analysis cannot silently shrink a denominator.
- DC/VTC gates reject every unconverged operating point or sweep point and
  retain signed terminal current.
- Every declared parametric cell remains in the denominator, including after
  baseline failure; the matrix directly covers temperature, body bias,
  reverse VDS, joint geometry/temperature corners, and three legal N/P ratios.
- Full-terminal dataset generation fails on missing rows/bins, records a hashed manifest and
  checksum-bound completion marker, and training rejects diagnostic, stale,
  dirty-source, or incomplete artifacts. The default training split holds out
  complete technology/VT/L/NFIN/temperature groups.
- Every worker log carries the digest of one immutable campaign
  manifest covering the source commit, jobs, NGSPICE/OSDI/PDKs, and every
  checkpoint sidecar. Collection and report generation reject mixed or
  missing provenance.
- An explicit older dataset/training source may accompany a newer harness
  commit only when the tracked numerical-source inventory is identical.
  The manifest records both commits and the shared inventory hash over
  compact models, generator, runtime, templates, PDKs, and environment inputs
  (Markdown excluded). Undeclared changes, mixed dataset commits, or differing
  numerical inputs are rejected. Original completion markers remain unchanged.

### 8.4 Run-to-run limits

The default validation/test split holds out complete geometry/variant/
temperature groups. It still does not reproduce full circuit trajectories, so
use broad family-level results for claims. The TSMC6 repeat measured about ±4 percentage
points of ring scatter and bimodal opamp basins; a single ring/opamp cell can
therefore change verdict between otherwise equivalent training runs.

## 9. Reproduction

Use the `pycircuitsim` conda environment and repository NGSPICE binary. The
authoritative launch, coverage, and report-build commands are in the
[README](../../README.md#run-the-complete-clean-checkpoint-matrix).

Historical raw trees are not mixed into a current pass. If complete local
evidence is absent, the builder may preserve an already committed report only
when its pinned SHA-256 matches; it must never synthesize a partial replacement.
