# NN accuracy plan: review round 2

This is a diagnostic review of the [accuracy improvement plan](../plans/2026-10-01-nn-accuracy-improvement.md),
not a qualification campaign. No runtime, training implementation, preserved
bundle or published score changed. The tested normalization arms do **not**
advance as-is. Leakage improves, but a signed drain-origin sweep reveals a
much larger regression than the original on-state aggregate suggested.

## Evidence and scope

Source: `1d7bdf9` (full SHA in the artifact manifest). All new inference used
the `pycircuitsim` conda environment, CPU, OMP/MKL/Torch=1 and flags off.
The three TSMC12 NMOS DirectNet medium bundles are the existing round-1
control, `sid1e-10` and `sid1e-12`; no new training was run. The source-current
probe additionally uses preserved V7.7.2 TSMC5 DirectNet medium NMOS/PMOS.

Artifacts: [`results/plan_review_20261001_round2/`](../../results/plan_review_20261001_round2/).
`artifact_manifest.json` binds scripts, source, weights, sidecars, inputs and
outputs; `commands.txt` records reproduction commands. All decks use the
canonical `L0_devices/mosfet.spice.tmpl` through shared renderers. Reference
currents and conductances come from NGSPICE on the identical OSDI binary.
Fixed-bias NN evaluations are device-function diagnostics, not circuit solves
or evidence of circuit convergence. Every requested reference trace completed;
no failed reference was averaged into a result.

## 1. Smaller current scales improve leakage but damage conductance

At TSMC12 NMOS SVT, L=16 nm, NFIN=2, 27 °C, Vgs=0 and Vds=0.8 V, NGSPICE
Id is 34.67 pA. The control predicts −811.2 pA; the two smaller-scale arms
predict 33.95 and 33.54 pA (2.1% and 3.3% relative error). This independently
supports the leakage benefit at that point.

The same models at Vgs=0.8 V behave badly around Vds=0. A 201-point NGSPICE
sweep from −5 to +5 mV in 50 µV steps supplies an independent reference.
The table excludes the zero point from current-relative medians (200 points
per row). Conductance comes from runtime autograd; NGSPICE DC slopes and
separate AC drain excitation agree on the near-origin conductance.

| Temperature | Control median current error | `s_id=1e-10` | `s_id=1e-12` |
|---|---:|---:|---:|
| −25 °C | 0.81% | 99.95% | 100.00% |
| 27 °C | 2.04% | 99.93% | 100.00% |
| +125 °C | 6.79% | 99.96% | 100.00% |

At +125 °C, NGSPICE origin gds is approximately 0.609 mS. The control gives
0.610 mS; the smaller-scale arms give −4.48 nS and +1.44 nS. They introduce
14 and 61 negative-gds points, respectively, among the 201 sampled biases;
the control has none. At Vds=1 mV, their Id errors are 99.4% and 99.0%.
This is a local shape failure, not merely the earlier 1.6× on-state trade-off.

![Near-origin current and conductance](../../results/plan_review_20261001_round2/linear_probe.png)

**Implication:** require signed nonzero origin sweeps and Jacobian checks
before replication or circuit campaigns. Do not select a scale from the
existing near-zero median. In the full TSMC12 NMOS data pool, 69.4% of rows
selected by round 1's near-origin mask are exactly Vds=0. That pool count is
not a recount of the held-out subset, but demonstrates why the two populations
must be separated. The dense reference sweep directly disproves the broad
claim that the origin region was fixed.

A mechanism worth testing is the target transform's slope:
`d(asinh(I/s))/dV = (dI/dV)/sqrt(I²+s²)`. Reducing s can demand a very steep
normalized target through a current zero. The observed dead band and spikes
are consistent with inadequate local approximation; this experiment does not
identify the optimal transform, scale, capacity, or sampling remedy.

## 2. Round-1 reference reconstruction adds substantial error

`compare_arms.py` obtains truth by inverse-transforming the loader's float32
`te.outputs`. A round trip of the **original float64 data**, without any
network, isolates the resulting error under the control normalization:

| Raw |Id| band | Median relative error from round trip alone | Wrong sign |
|---|---:|---:|
| 0.01–0.1 pA | 597% | 51.0% |
| 0.1–1 pA | 56.8% | 14.8% |
| 1–10 pA | 6.81% | 0% |
| 10–100 pA | 0.67% | 0% |

These are full-pool representation diagnostics, not NN prediction errors.
The smaller-scale transforms greatly reduce this quantization, but comparing
them with a reference reconstructed through the control transform still
corrupts low-current evaluation. Preserve raw rows and IDs. Treat the original
sub-pA table as provisional and establish reference-resolution floors before
using log/sign metrics. NGSPICE/raw-label differences in our cold pA probes
remain unresolved; the hot mA agreement below does not validate pA accuracy.

## 3. Reference modelcard selection can silently use the wrong NFIN bin

`TechProfile._resolve_tsmc_modelcard` always resolves `default_nfin`.
`circuit_benchmarks.get_baked_modelcard` then inserts the requested NFIN into
that card. Correct instance geometry cannot undo the wrong bin's parameters.

We ran both paths through NGSPICE at Vgs=±VDD, Vds=±VDD/2, 27 °C, nominal
NMOS/PMOS lengths and VTs. Both rendered pairs pass the existing physical
parity check. NFIN=2 agrees exactly in all six technology/polarity controls.
For NFIN=10:

| Technology | NMOS Id difference | PMOS Id difference |
|---|---:|---:|
| TSMC5 | 15.69% | 7.36% |
| TSMC12 | 3.51% | 4.38% |
| TSMC16 | 1.01% | 2.03% |

Differences are relative to resolving the actual NFIN first. The 18-cell
experiment also includes NFIN=5. It establishes a reference defect in this
helper, not the magnitude of any circuit-score correction. The parametric
device resolver in `nn_gate.py` already supplies actual NFIN, so its reported
high-fin misses are not explained away. Repair affected references in a
separate change and re-run affected circuits before model attribution; do not
patch published totals with these device-level numbers.

## 4. Fixed seed does not preserve a holdout during data expansion

The production grouped splitter balances row counts after a size-based group
ordering. Its assignments depend on the full inventory and group sizes.
Using all 6,467,850 TSMC12 NMOS rows and seed 42:

| Split-only perturbation | Added rows | Old test → training | Share of old test |
|---|---:|---:|---:|
| Copy NFIN=6 group metadata into new NFIN=8 groups | 1,298,610 | 553,454 / 646,830 | 85.56% |
| Double rows in existing NFIN=6 groups | 1,298,610 | 488,186 / 646,830 | 75.47% |

No synthetic electrical labels were trained or used as ground truth. These
are deterministic splitter experiments on geometry metadata only. Each
individual split remains group-disjoint; the problem is cross-arm
contamination and comparability. Persist old group assignments and raw row
identities, assign new groups separately, and reserve untouched confirmation
data before coverage or overlay arms. Keep training seeds separate from the
split seed.

## 5. Hot leakage is reproduced, but KCL hides source-current error

Twelve stored off-state rows were checked against NGSPICE: TSMC5/12 ×
NMOS/PMOS × 248.15/300.15/398.15 K, NFIN=2 and the sampled L nearest 20 nm,
Vgs nearest zero, Vds nearest ±VDD and Vbs=Vs=0. At +125 °C, TSMC5 reference
|Id| is 1.035 mA (NMOS) and 2.752 mA (PMOS); stored Id agrees to better than
3e-10 relative. Do not discard these labels as evaluator errors. This is
agreement with the stipulated compact-model reference, not evidence of
silicon realism or every bin's validity.

The drain and gate terms nearly cancel. Independent terminal accuracy matters:

| Preserved TSMC5 medium model | Reference Is | Predicted Is | Id error | Ig error |
|---|---:|---:|---:|---:|
| NMOS | −0.961 nA | +4.342 µA | 0.073% | 0.492% |
| PMOS | +0.381 nA | −2.970 µA | 0.021% | 0.087% |

Source relative errors are about 4,517× and 7,805×, with opposite signs.
Candidate KCL residuals are below 6e-23 A. Repeating NGSPICE with tighter
RELTOL/ABSTOL/VNTOL tests reference sensitivity; the recorded shifts are tiny
relative to these µA errors. Exact algebraic closure alone is therefore an
insufficient acceptance condition. Score all four currents, source Jacobian
rows and derived source charge; stratify cancellation-heavy cases. Consider
a separate derived-source loss or current-basis experiment only after this
attribution, preserving physical terminal conventions.

## Planning changes and remaining limits

The plan now prioritizes reference/metric/split prerequisites, cheap origin
and terminal checks, then family-specific paired seed replication. It also:

- separates equal-update schedule experiments from extra-compute training;
- forbids transferring DirectNet variance bounds to BSIM-AR;
- treats repeated circuit feedback as development evidence;
- distinguishes scalar sidecar changes from per-fin/temperature transforms
  requiring a runtime contract;
- rules out weight-only warm starts across incompatible normalizations;
- distinguishes training gate NFIN values from unseen-geometry confirmation.

No new scale, schedule, selection method, coverage training, derivative loss,
BSIM-AR strategy, or circuit-level improvement was validated in this review.
No full qualification suite was run and no model is promoted. The code defects
identified here are recorded for follow-up; this task changed documentation
and added isolated diagnostic scripts/artifacts only.
