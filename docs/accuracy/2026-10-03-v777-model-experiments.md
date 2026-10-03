# V7.7.7 model-improvement experiments

Status: **O comparison complete; scalar N comparison running**. This report
separates actual retraining effects from the earlier
[reference and energy corrections](2026-10-03-v777-phase1-attribution.md).
The [plan](../plans/2026-10-01-nn-accuracy-improvement.md) defines advancement;
no candidate in this report has been promoted.

## Frozen experiment

- DirectNet-Full LEVEL=75, medium, TSMC5/12, both polarities, seed 42.
- C: isolated control with the preserved parent rows and frozen split/order.
- O: C plus log-spaced drain-origin rows, with geometry partitions preserved.
- N: O data with fixed drain-current asinh scale `1e-7 A`, trained from scratch
  against a freshly repeated O control on the same source commit and GPU.
- Existing medium recipe: 200 epochs, patience 40, batch 2048, LR 0.001,
  EMA 0.999, AMP off. Each polarity's control and candidate share one A100 UUID.
- CPU inference, one thread except the declared ring/opamp 1/2/4 repeats.
- Exactly 92 circuit jobs and 366 result rows per arm, generated from the
  authoritative clean/canary/simple-v2 inventory for these two technologies.
  This is a pilot denominator, not the five-technology release campaign.

Training/data source for C/O is `b385118`. The corrected comparison runtime and
N training source are `faf03cc`. The campaign manifests explicitly distinguish
training/data and evaluation source identities. All four C checkpoints
reproduce the preserved V7.7.2 weights bit-for-bit. All four repeated O controls
also reproduce the first O weights bit-for-bit. This verifies these runs; it
is not a multi-seed noise estimate or universal determinism claim.

The first O campaign exposed a metric-classification defect: a solved SRAM
that never switches had been marked as an infrastructure error. That pass is
preserved, not scored. Both complete C/O passes were rerun on the corrected
harness, which retains a candidate ERROR, convergence flag and denominator but
no numeric write-trip score. No archived verdict was manually relabelled.

## Complete C → O comparison

| Outcome | C | O |
|---|---:|---:|
| Jobs | 92 | 92 |
| Result rows | 366 | 366 |
| Converged candidate solves | 350 | 348 |
| Characterized rows | 347 | 344 |
| Qualification PASS rows | 128 | 129 |
| Qualification FAIL rows | 2 | 1 |
| Diagnostic rows | 217 | 214 |
| ERROR rows | 19 | 22 |

There are 332 common characterized rows, 12 newly characterized rows and
15 lost characterized rows. O loses 15 previously converged solves despite
recoveries elsewhere. No qualification PASS becomes FAIL/ERROR, but the
plan's no-lost-convergence condition still rejects O as a standalone advance.
A solved state without a required write event is distinct from non-convergence;
both remain ERROR rows with empty scored metrics.

| Test case / metric | TSMC5 C → O | TSMC12 C → O |
|---|---:|---:|
| Miller AC gain error (dB) | 11.8636 → 5.2020 | 4.6437 → 0.1117 |
| Miller AC verdict | FAIL → FAIL | FAIL → PASS |
| Device DC worst NRMSE (%) | 6.4398 → 6.4722 | 4.2496 → 2.9791 |
| Inverter VTC/transient worst NRMSE (%) | 4.3248 → 4.3098 | 4.0450 → 4.0385 |
| Ring worst unaligned NRMSE (%), OMP 1/2/4 | 1.9205 → 2.6794 | 12.1131 → 10.9278 |
| SRAM lobe worst NRMSE (%) | 0.2865 → 0.9788 | 2.3579 → 3.8755 |
| Switch-cap NRMSE (%) | 0.0473 → 0.1104 | 0.1134 → 0.1139 |
| Beta multiplier converged rows | 3/3 → 0/3 | 0/3 → 3/3 |

The listed numeric comparisons use the common characterized rows. Recovered
rows never enter that paired aggregate. Ring qualification uses period error,
not the unaligned voltage NRMSE in this descriptive table; both technologies
retain their ring PASS. SRAM and switch-cap retain PASS despite increased
errors. Full per-row MRE, R², NRMSE, voltage errors and convergence flags are
available in the [comparison artifacts](../../results/nn_accuracy_20261001/comparison/).

## Origin and leakage screen

A separate NGSPICE screen contains 648 points: both technologies/polarities,
three temperatures, trained/interpolated geometries, three gate biases, and
Vds=0 plus ±10 µV, ±100 µV, ±1 mV and ±5 mV. Both arms have zero new negative
on-state gds points. This does not mean all weak-band derivatives are accurate.

At 27 °C, worst zero-drain-current error over the two screened geometries
changes from 78.2 to 5.95 nA for TSMC12 PMOS and 60.7 to 24.5 nA for TSMC12
NMOS. TSMC5 NMOS origin on-state gds error worsens from 4.1% to 11.3%.

Raw float64 held-out labels show lower median off-current log error in every
technology/polarity/temperature slice, but O alone does not reach the declared
0.5-decade target in the cold TSMC5 NMOS/PMOS and TSMC12 NMOS slices. The hot
TSMC5 p95 errors increase slightly. The scalar N arm tests the remaining
normalization floor; it is not a promotion or an assertion that O's tails pass.

## Attribution limits

- Matched-bias Miller diagnostics reduce TSMC5 gain error from 11.86 to 3.91 dB
  at the forced NGSPICE state. The remaining derivative mismatch and the
  natural operating-point displacement both matter. Forced bias is diagnostic.
- NMOS/PMOS LEVEL=72 swaps localize the stronger Miller contribution to NMOS
  on TSMC5 and PMOS on TSMC12. Full LEVEL=72 controls agree with NGSPICE gain
  within 0.000022 dB. Fresh-parse candidate residuals are archived with complete
  node and voltage-source branch-current states.
- The largest circuit voltage errors are NAND/NOR internal-node startup
  differences, largely unchanged by O. NOR references leave the NN support
  box briefly. LEVEL=72 reproduces the large NOR discrepancy, so it cannot
  be attributed solely to training; the raw maximum remains reported.
- The preserved TSMC12-large device DC run still has one failure and no
  convergence ERRORs. The geometry-aware device resolver was not affected
  by the repaired reference-bin defect.

## Evidence locations

All paths below are under `results/nn_accuracy_20261001/`:

- `pilot/`: preparation, training, raw-float64 holdouts, 648-point reference
  screen, original campaigns and checkpoint-reproduction checks.
- `pilot_corrected/`: both complete C/O campaigns on `faf03cc`, manifests,
  raw traces, strict paired comparison and error/convergence transitions.
- `scalar/`: repeated O controls and N training/comparison, isolated bundles.
- `phase1_followup/`: Miller matrices, polarity swaps, fresh-state residuals,
  high-NFIN reproduction and initializer/logging diagnostics.
- `comparison/`: machine-readable full row and case tables; no ERROR is averaged
  into a numeric score. AC transfer/impedance errors are excluded from the
  voltage-error column, and inverter mV values are converted to V.

The current collected suite passes 1,195 tests, none skipped. N results,
advancement decisions, further plan work and the final requested push remain
pending; this file does not claim that the full improvement plan is complete.
