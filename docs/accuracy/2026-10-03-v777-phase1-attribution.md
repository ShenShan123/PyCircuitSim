# V7.7.7 NN accuracy work: Phase 1 attribution

Status: **partial Phase 1, diagnostic evidence**. The
[plan](../plans/2026-10-01-nn-accuracy-improvement.md) remains in progress.
No model was trained or promoted, and no full qualification campaign was run.
Published scores still describe V7.7.6 under its original reference and metric
contracts.

## Scope and provenance

Artifacts, executable probes, exact child commands/environments, source patches,
card hashes, decks, logs and traces are under
[`results/nn_accuracy_20261001/phase1_reference/`](../../results/nn_accuracy_20261001/phase1_reference/).
The source base is `1d7bdf9076e6faf7de5500ef5c58fdcedf59cc81` plus the recorded
local changes. This is dirty-source diagnostic work, not canonical training
or a publishable campaign. `provenance.json`, `pilots/provenance.json` and
`energy_corrected/provenance.json` distinguish the reference repair from the
later metric correction. Scored inference is CPU, OMP/MKL/Torch=1, numerical
experiment flags off. NGSPICE 45.2 loads the existing BSIM-CMG OSDI binary
(SHA-256 `f089f17d5d5b1178c48932ff699960dab3ab509c33b34c798421eccbbf14a78b`).

The circuit pilots pin both polarities of the preserved DirectNet-medium
TSMC5/12 bundles. Every bundle artifact is hashed in its `run.json`. They are
same-weight re-baselines, not same-commit retrained controls or a seed-spread
estimate. The 12-job inventory is two technologies × SRAM SNM, Miller AC,
inverter energy, beta multiplier, LDO and SRAM modes. It produces 38 rows;
it does not replace any release denominator.

## Reference-bin repair

The shared profile previously resolved at `default_nfin` and subsequently
baked the requested NFIN into that card. Baking cannot replace bin-dependent
process parameters. The circuit, LEVEL=72 DC and transient helpers now pass
the actual NMOS/PMOS fin counts to resolution. The transient merged-card
cache and filename also distinguish both fin counts. DC baked/merged filenames
include both fin counts so successive runs sharing a case label cannot reuse
or overwrite a different bin.

`bin_audit.py` checks five technologies × NFIN {2,3,4,5,10} × both polarities
× off/linear/saturation at nominal temperature: **150 bias rows**. It compares
corrected cards with independent direct calls to `resolve_modelcard`, checks
rendered physical parity, runs old and corrected cards in NGSPICE, and checks
the LEVEL=72 terminal adapter at the same fixed bias. All 50 polarity/card
parameter comparisons and 150 adapter checks pass. Maximum absolute
four-terminal adapter error is 0.231 pA. These are fixed-bias adapter checks,
not proof of circuit-solver correctness.

| NFIN | Largest linear-region Id change | Largest saturation Id change |
|---:|---:|---:|
| 2 | 0% | 0% |
| 3 | 0.00423% | 0.0000153% |
| 4 | 12.55% | 13.25% |
| 5 | 19.46% | 12.66% |
| 10 | 20.63% | 15.69% |

NFIN=3 is **not cleared**: TSMC6/7 PMOS off-current changes by 26.23%.
Only references above 200 quanta receive percent comparisons; this audit
uses `q = max(1000 × ulp(|Vd|), 2^-43 A)` at its nonzero drain biases. All
rows also retain absolute errors and signed four-terminal currents. Extend
the reference-resolution audit before applying this rule to uncovered regions.

The correction does not touch `nn_gate.py`, whose device/inverter resolver
already uses actual geometry. Its high-NFIN NN errors remain independent of
this defect.

## Preserved failure ledger

`ledger.py` consumes all three preserved V7.7.6 flags-off round inventories,
which supersede the historical V7.7.5 rows while retaining the same case
inventory. It preserves every source row and source-file hash.

| Category | Rows |
|---|---:|
| Qualification pass | 2,468 |
| Converged qualification mismatch | 32 |
| Non-convergence | 207 |
| Support rejection | 128 |
| Unavailable metric | 25 |
| Threshold-free diagnostic | 4,372 |
| **Total** | **7,232** |

The unavailable-metric rows are recognized by their preserved
`uncharacterized_diagnostic` payload; their historical status/error fields
remain intact. Numerical diagnostics are not relabeled as qualification
failures. No infrastructure failure is hidden in these categories.

Forty aggregate SRAM rows include the affected 5/10-fin references. Thirteen
of 80 bundles were trained on Blackwell, and 67 on A100, from the completed
training job records and their retained logs. This tags 2,354 rows with at
least one Blackwell bundle; the union with reference-bin exposure is 2,381
rows requiring re-run before attribution under the plan. No GPU is unknown.
The old completion markers do not bind hardware, so these are job-record
provenance tags, not a newly certified bundle contract. Temperature tags are
6,992 nominal, 80 cold and 160 hot rows. The bin tags apply to this nominal
campaign; unrun parametric circuit corners still need their own audit.

## Circuit re-baselines and controls

| DirectNet medium | TSMC5 | TSMC12 |
|---|---:|---:|
| SRAM worst lobe NRMSE, old reference | 6.5367% | 1.1360% |
| SRAM worst lobe NRMSE, corrected reference | 0.2865% | 2.3579% |
| SRAM gate, corrected reference | PASS | PASS |
| Miller AC gain error | 11.8636 dB FAIL | 4.6437 dB FAIL |

The SRAM comparison is matched to the medium model; 6.1012% was the small
model's old TSMC5 result. Reference repair materially reduces the medium
TSMC5 lobe mismatch, but its forced-state diagnostic still fails both states,
as it did before. The TSMC12 lobe error grows while remaining below the gate.
A reference correction is not uniformly an improvement in NN accuracy.

The four topology pilots include independent NGSPICE references and
PyCircuitSim LEVEL=72 controls. Their original 38 rows comprise 17 complete
diagnostics, two qualification PASSs, two qualification FAILs, three candidate
ERRORs and fourteen control/reference-stage ERRORs. There are no timeouts or
unhandled worker tracebacks. A control failure exits 2 and is not a scientific
NN failure:

- TSMC12 beta multiplier fails candidate convergence in OP and both supply
  sweeps; LEVEL=72 controls converge.
- Both LDO line-regulation controls fail the internal-node solve at large
  Newton trials. These rows cannot attribute error to the NN. Both LDO
  load-step transients run without `uic`, with converged controls. Candidate
  load-step voltage NRMSE is 23.11%/10.37% (TSMC5/12), with maximum errors
  12.28/6.16 mV.
- Twelve SRAM-mode controls exceed the harness's 1% voltage-NRMSE limit:
  hold, read and write-margin, both stored states, both technologies. Their
  NGSPICE references complete. These are unresolved solver/control differences;
  they cannot serve as NN-arm veto evidence until attributed. The write
  transients do complete in both engines.

Full per-row MRE, R², NRMSE, maximum error, convergence and domain metrics
remain in `pilots/results.json`. ERROR rows carry no scored numeric metrics.
These samples do not establish a per-family or per-technology aggregate.

## Inverter energy is a measurement defect

The former metric integrated rectified current on a resampled grid capped at
600 points. It both counts returned charge as consumed energy and aliases
narrow supply-current pulses. The LEVEL=72 control reproduced the apparent
technology-dependent NN error. The corrected definition lives in the
[methodology](methodology.md#inverter-supply-energy-v777).

| Energy error | TSMC5 | TSMC12 |
|---|---:|---:|
| Former metric, NN | 26.4915% | 3.6390% |
| Former metric, LEVEL=72 control | 26.9010% | 4.2409% |
| Corrected metric, NN | 0.7787% | 1.6241% |
| Corrected metric, LEVEL=72 control | 0.7994% | 1.6470% |

`energy_steps.py` repeats TSMC5 LEVEL=72 at 2 ps, 1 ps and 0.5 ps. The former
metric reports 26.90%, 12.76% and 46.44%; signed native-grid energy error is
0.7994%, 0.7949% and 0.7920%. This screen establishes measurement instability
and its correction, not a reason to change the simulator's default step or
integration scheme. It leaves visible supply-current ringing and a residual
sub-2% energy difference for further solver investigation.

`energy_corrected/` contains fresh runs on both technologies, archived NN and
NGSPICE traces, and corrected reanalysis of the saved step study. Original
pilot records and published reports are preserved. Energy provides no evidence
for a charge-supervision arm from this comparison.

## Temperature screen and circuit demand

`hot_onset.py` runs 288 NGSPICE off-bias points: TSMC5/12, NMOS/PMOS, two VTs,
NFIN 2/10, L 16/20 nm, and nine temperatures from 27 to 125 °C. All references
complete with deck parity. Thirty-two generated cards match every emitted
parameter from the PDK merge; omissions are geometry/bin bounds and the
existing sentinel exclusion. This checks cache/serialization fidelity using
the existing parser. Independent source parsing, ignored OSDI parameters and
PDK subcircuit-wrapper equivalence remain outside that check.

For the nominal two-fin TSMC5 PMOS, |Ig| rises from 5.05 nA at 40 °C to
103.5 µA at 55 °C and 2.507 mA at 70 °C, then reaches 2.752 mA at 125 °C.
The NMOS rises later, reaching 0.659 mA at 125 °C at its nominal 16 nm length.
These are smooth sampled values with a steep transition, not evidence that a
three-temperature NN interpolates accurately. Hot leakage remains in scope;
no runtime temperature restriction or separate hot bundle was implemented.

`census.py` re-runs 36 pilot reference analyses, retaining every device-node
voltage and ideal-voltage-source branch current. It uses NGSPICE's exported
physical drain conduction current `IDEFF`, after 24 source-biased checks
against `-i(Vd)`; it does not use channel `IDS` or PyCircuitSim as a reference.
Transient displacement current is excluded to match the NN current surfaces.

| Case | Device/analysis entries | Ever below 30 nA | Ever inside \|Vds\| < 8 mV |
|---|---:|---:|---:|
| Miller AC operating points | 14 | 0 | 0 |
| Beta multiplier | 30 | 20 | 20 |
| LDO | 56 | 0 | 8 |
| SRAM modes | 96 | 96 | 66 |
| Inverter energy | 8 | 8 | 8 |
| **Total** | **204** | **124** | **102** |

These are device/analysis entries, not unique physical devices or a full
failure census. Per-device sample fractions are in `census.json`; transient
fractions weight adaptive samples rather than elapsed time. This supports
keeping origin/leakage work early, but Miller AC needs separate matched-bias
attribution. `scalar_tradeoff.json` also records the normalized residual and
physical/normalized slopes for the existing 45 fixed-bias arm probes. Those
already inspected samples are diagnostic, not untouched selection data.

## Verification and remaining work

The bin regression test failed on the original five adapter paths and passes
after repair. Two new energy tests fail on the old metric; all three energy
contracts pass after repair, including unequal observation-window endpoints.
The full collected suite passes **1,183 tests, none skipped**. Five warnings
concern Torch pinned memory with no accelerator. `pytest_final.log` retains
the complete result. Early probe-script failures are retained as attempt logs
and were corrected before producing the complete diagnostic artifacts.

Still required: persistent split/raw-row identities, extended training
provenance, full-domain quantum calibration, same-hardware seed spread,
remaining Phase 1 cells and families, fresh-parse state re-stamping,
Miller matched-bias I/G/C and polarity swaps, control-failure attribution,
then the registered training arms. Neither a full campaign nor the plan is
complete. The final commit/push requested by the user has not been performed.
