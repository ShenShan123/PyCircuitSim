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
The graded reduced NN families are DirectNet (LEVEL=73) and BSIM-AR (74).
The separately scored experimental full-terminal families are DirectNet-Full
(75) and BSIM-AR-Full (76). All use TSMC5/6/7/12/16 and the `small`, `medium`,
`large`, and `xl` tiers.

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
`circuit_templates/`.
The complete contract is in
[`simple-circuits-v2-topologies.md`](simple-circuits-v2-topologies.md).

Simple-v2 rows are **diagnostics**, are held out from training, and do not
change the historical simple-v1 `/20` denominator. Promotion requires a
separately versioned denominator, at least three stable LEVEL=72 repeats,
frozen thresholds, and a complete CPU-pinned campaign. Numerical mismatches
remain diagnostic; an uncharacterizable requested cell is an explicit
`ERROR`.

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
Charge-sensitive AC gates use autograd charge derivatives and may move
independently of DC accuracy.

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
- Reference and candidate decks are rendered and compared before a numerical
  mismatch is attributed to the model.

The frozen V7.5.17 clean matrix contains DirectNet and BSIM-AR × 4 tiers × 5
technologies × 12 gate invocations = **480 jobs**. V7.6.8 adds three diagnostic
invocations per group/technology, making a fresh current-family pool **600
jobs**; those denominators must not be merged or compared as if identical.
The V7.6.6 full-terminal matrix applies the same 480-job denominator to freshly
trained DirectNet-Full and BSIM-AR-Full models in one separate, non-backfilled
campaign.
Report generation fails closed unless the applicable matrix and checkpoint
artifacts are complete.

## 6. Comparability

A gate result belongs to a specific checkpoint, solver commit, and gate
contract. Re-gating fixed weights is comparable only when those inputs match;
retraining the same recipe is stochastic.

The V6.13 `gds` correction makes pre-fix device-DC results comparable because
the DC fixed point is invariant. Pre-fix AC, transient, and opamp results are
not comparable. The detailed history is in the changelog.

## 7. TSMC6 controlled repeat

TSMC6 is TSMC7 relabelled in this LEVEL=72 flow: their training arrays and
electrical response are identical. It remains in the five-tech denominator as
a controlled repeat, not an independent ground truth. Differences between its
NN verdicts and TSMC7's measure training and Newton-basin variability.

## 8. Measurement caveats and V7.5.17 corrections

V7.5.17 retains the V7.5.16 solver corrections and adds coverage-audit
contracts:

- Residual probes recover ideal-voltage-source branch currents and scale the
  tolerance from current-valued node rows, so they measure the complete MNA
  residual. The topology-stable fit is cached without changing the result.
- Opamp AC independently refines each simulator's bias at 0.1 mV resolution,
  validates the NGSPICE reference, and requires a converged NN operating point.
- Family labels and CLI validation fail closed; an invalid interpreter,
  technology, device, or analysis cannot silently shrink a denominator.
- DC/VTC gates reject every unconverged operating point or sweep point and
  retain signed terminal current.
- Every declared parametric cell remains in the denominator, including after
  baseline failure; the matrix directly covers temperature, body bias,
  reverse VDS, joint geometry/temperature corners, and three legal N/P ratios.
- Dataset generation fails on missing rows/bins, records a hashed manifest and
  checksum-bound completion marker, and training rejects diagnostic, stale,
  dirty-source, or incomplete artifacts. The default training split holds out
  complete technology/VT/L/NFIN/temperature groups.
- Every V7.5.17 worker log carries the digest of one immutable campaign
  manifest covering the source commit, jobs, NGSPICE/OSDI/PDKs, and every
  checkpoint sidecar. Collection and report generation reject mixed or
  missing provenance.

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

Raw V7.5.17 evidence is stored under `results/v7517_clean/`. Historical raw
trees are not mixed into the current pass. If complete local evidence is
absent, the builder may preserve an already committed report only when its
pinned SHA-256 matches; it must never synthesize a partial replacement.
