# V7.7.7 model-improvement experiments

Status: **registered C/O/N/R/S screens complete; no candidate promoted**. This report
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

![C/O/N testcase and derivative comparison](../../results/nn_accuracy_20261001/comparison/model_comparison.png)

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

## Complete N comparison

N passes all 130 qualification rows (including both Miller AC cells) and
converges in 360/366 rows. It has six ERROR rows and no qualification FAIL.
The repeated O campaign is bit-identical in every verdict, metric and domain
field to the first corrected O campaign.

| Metric | C | O | N |
|---|---:|---:|---:|
| Qualification PASS rows | 128 | 129 | 130 |
| Converged candidate solves | 350 | 348 | 360 |
| Characterized rows | 347 | 344 | 360 |
| ERROR rows | 19 | 22 | 6 |
| TSMC5 Miller gain error (dB) | 11.8636 | 5.2020 | 0.7607 |
| TSMC12 Miller gain error (dB) | 4.6437 | 0.1117 | 2.3633 |
| TSMC5 worst device DC NRMSE (%) | 6.4398 | 6.4722 | 5.4194 |
| TSMC12 worst device DC NRMSE (%) | 4.2496 | 2.9791 | 1.9032 |

N still **does not advance**. It loses three converged cascode-stack rows
relative to O. Against C, it recovers fifteen solves but loses five: the three
TSMC5 cascode-stack analyses and two TSMC5 beta-multiplier supply sweeps. Those
losses are retained even though the aggregate convergence count improves.

The derivative veto is also decisive. At TSMC12 NMOS, L=18 nm, NFIN=3,
125 °C, Vgs=0.8 V and Vds=1 mV, NGSPICE gives gds=+2.664 mS, O gives
+1.848 mS, and N gives **−5.362 mS**. Reference drain current is −469.94 µA;
N's value error is under 1%, yet its local slope has the wrong sign. Good
pointwise current accuracy is not sufficient for Newton or AC behavior.

### Per-technology metrics

The device columns summarize the same 26 TSMC5 / 30 TSMC12 DC configurations:
mean case MRE, median case R², and worst case NRMSE. Maximum voltage error
uses only the common characterized OP/DC/transient circuit rows; AC gains,
impedances and current errors are excluded. The maximum is an internal NOR
startup discrepancy reproduced by LEVEL=72, not a new N fitting error.

| Technology | Arm | Device MRE (%) | Device R² | Worst DC NRMSE (%) | Max voltage error (V) |
|---|---|---:|---:|---:|---:|
| TSMC5 | C | 2.5610 | 0.9999930 | 6.4398 | 1.460485 |
| TSMC5 | O | 2.9504 | 0.9999848 | 6.4722 | 1.460485 |
| TSMC5 | N | 2.4417 | 0.9999826 | 5.4194 | 1.460485 |
| TSMC12 | C | 1.2205 | 0.9999957 | 4.2496 | 1.064180 |
| TSMC12 | O | 1.4115 | 0.9999907 | 2.9791 | 1.064180 |
| TSMC12 | N | 0.9833 | 0.9999863 | 1.9032 | 1.064180 |

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
TSMC5 p95 errors increase slightly. N reduces the remaining floor, but worsens the hot TSMC5 NMOS p95 log error
from 1.217 to 2.142 decades. It does not satisfy the no-tail-regression rule.


### Off-current error by temperature

Median / p95 absolute log10 error, on identical original test rows with
|I_ref| ≥ 20 × 2^-42 A. These are development diagnostics, not qualification
scores. Each technology/polarity/temperature keeps its own denominator.

| Technology | Polarity | °C | C | O | N |
|---|---|---:|---:|---:|---:|
| TSMC5 | nmos | -25 | 1.595 / 2.733 | 1.258 / 2.447 | 0.3077 / 1.212 |
| TSMC5 | nmos | 27 | 1.053 / 2.82 | 0.7591 / 2.523 | 0.08471 / 1.198 |
| TSMC5 | nmos | 125 | 0.00139 / 1.171 | 0.001128 / 1.217 | 0.001618 / 2.142 |
| TSMC5 | pmos | -25 | 1.208 / 2.622 | 0.9722 / 2.353 | 0.2188 / 1.284 |
| TSMC5 | pmos | 27 | 0.9138 / 2.5 | 0.7411 / 2.335 | 0.09677 / 1.156 |
| TSMC5 | pmos | 125 | 0.001395 / 1.846 | 0.001069 / 1.889 | 0.001737 / 0.9452 |
| TSMC12 | nmos | -25 | 0.9831 / 2.612 | 0.7362 / 2.214 | 0.1199 / 1.339 |
| TSMC12 | nmos | 27 | 0.5145 / 2.621 | 0.2774 / 2.103 | 0.05866 / 1.061 |
| TSMC12 | nmos | 125 | 0.03926 / 2.149 | 0.0182 / 1.38 | 0.009478 / 0.7306 |
| TSMC12 | pmos | -25 | 0.639 / 2.656 | 0.446 / 2.263 | 0.07289 / 1.085 |
| TSMC12 | pmos | 27 | 0.5854 / 2.458 | 0.4186 / 2.188 | 0.05556 / 1.167 |
| TSMC12 | pmos | 125 | 0.06052 / 1.556 | 0.03237 / 1.207 | 0.01849 / 0.4821 |

## Independent large-model schedule and selection results

R0 and R use the same original frozen data, architecture, seed, physical GPU,
200 epochs and optimizer-update count. R0 truncates an 800-epoch cosine
schedule; R anneals over 200 epochs. These are fresh **large** controls and
candidates, separate from medium C/O/N. Both polarities and both technologies
complete the full 92-job / 366-row inventory.

| Outcome | R0 | R | S |
|---|---:|---:|---:|
| Converged candidate rows | 330 | 337 | 337 |
| Characterized rows | 330 | 335 | 335 |
| Qualification PASS rows | 128 | 128 | 128 |
| Qualification FAIL rows | 2 | 2 | 2 |
| ERROR rows | 36 | 31 | 31 |
| TSMC5 Miller gain error (dB) | 15.8813 | 0.2701 | 0.2701 |
| TSMC12 Miller gain error (dB) | 18.7093 | 11.4989 | 11.4989 |
| TSMC12 NMOS NFIN=10 DC NRMSE (%) | 8.4566 | 12.2992 | 12.2992 |

R recovers seventeen characterized rows but loses twelve, including ten lost
convergences. It also turns the TSMC12 NFIN=10 DC case from PASS to FAIL. Four
new negative on-state origin conductances appear in TSMC12 NMOS at Vds=−1 mV
across trained/interpolated geometries and cold/nominal/hot points. R therefore
fails advancement even though its aggregate convergence and TSMC5 Miller
accuracy improve.

S examines all 200 averaged snapshots per bundle using the frozen validation
subset, tail/sign guards and origin finite-difference checks. Only the ordinary
normalized-loss selected snapshot is eligible in each of the four bundles.
The selected epochs are 186/188 for TSMC5 NMOS/PMOS and 181/180 for TSMC12.
S changes no weights, and every testcase verdict, metric and domain field is
identical to R on its independent full campaign. This is a negative result for
this registered selector; it does not show that all physical selection rules
are ineffective. The guards were not relaxed to manufacture a different pick.

The 92-job R0, R and S manifests, snapshot indexes, validation row identities,
selection audits and complete comparisons are in `schedule/`. The separate
[large-model case table and per-row metrics](../../results/nn_accuracy_20261001/comparison_schedule/)
keep current/AC units separate from voltage errors.

### Large-model per-technology metrics

The device summaries and maximum-voltage scope use the same definitions as
the medium-model table; compare R0/R/S within this matched experiment.

| Technology | Arm | Device MRE (%) | Device R² | Worst DC NRMSE (%) | Max voltage error (V) |
|---|---|---:|---:|---:|---:|
| TSMC5 | R0 | 2.9828 | 0.9999824 | 7.7238 | 1.460486 |
| TSMC5 | R | 2.4878 | 0.9999927 | 6.8081 | 1.460485 |
| TSMC5 | S | 2.4878 | 0.9999927 | 6.8081 | 1.460485 |
| TSMC12 | R0 | 1.5535 | 0.9999952 | 8.4566 | 1.064180 |
| TSMC12 | R | 1.5616 | 0.9999986 | 12.2992 | 1.064180 |
| TSMC12 | S | 1.5616 | 0.9999986 | 12.2992 | 1.064180 |

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

## Cost and verification

The 24 completed training jobs used 11.694 assigned GPU wall-hours, including
11.335 hours inside the training loops. These are elapsed job times on shared
A100s, not exclusive GPU utilization or an inference-speed comparison. The
run-level accounting is in `training_cost.json`. Snapshot selection used
validation-only inference in addition to those training loops.

The final collected suite passes **1,204 tests, none skipped**; eight warnings
concern Torch pinned-memory settings in CPU smoke tests. The source-lineage,
row-count and checkpoint checks used by each campaign remain enabled. Raw
infrastructure attempts are preserved separately; none becomes scientific
PASS/FAIL evidence.

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

The current collected suite passes 1,204 tests, none skipped. The registered
V7.7.7 screening cycle is complete. O, N and R fail advancement vetoes; S makes
no model change. No candidate proceeds to seed expansion, BSIM-AR transfer,
five-technology qualification or full-matrix replacement. Those unexecuted
conditional steps are not claimed as evidence. Coverage, structural-transform
and derivative-supervision proposals remain future work, not completed results.

## Complete testcase inventory

Convergence is separate from characterization. NRMSE columns use the common
characterized intersection across C/O/N; a dash means that intersection has
no numeric NRMSE. No ERROR value is recovered into these columns.

| Technology | Case | C converged | O converged | N converged | Worst paired NRMSE (%) / C → O → N |
|---|---|---:|---:|---:|---:|
| TSMC5 | beta_multiplier | 3/3 | 0/3 | 1/3 | — → — → — |
| TSMC5 | bias_tree_fanout_17t | 1/1 | 1/1 | 1/1 | 0.007587 → 0.0002503 → 0.0008177 |
| TSMC5 | bias_tree_fanout_3t | 1/1 | 1/1 | 1/1 | 0.007598 → 0.0002529 → 0.0008139 |
| TSMC5 | bias_tree_fanout_5t | 1/1 | 1/1 | 1/1 | 0.007594 → 0.0002503 → 0.0008115 |
| TSMC5 | bias_tree_fanout_9t | 1/1 | 1/1 | 1/1 | 0.007593 → 0.0002514 → 0.0008182 |
| TSMC5 | cascode_stack | 3/3 | 3/3 | 0/3 | — → — → — |
| TSMC5 | common_gate | 2/2 | 2/2 | 2/2 | 0.09219 → 0.1287 → 0.1107 |
| TSMC5 | common_source_nn | 4/4 | 4/4 | 4/4 | 0.488 → 0.1893 → 1.443 |
| TSMC5 | current_mirror | 4/4 | 4/4 | 4/4 | 0.4369 → 0.5067 → 0.4275 |
| TSMC5 | device_derivative | 6/6 | 6/6 | 6/6 | 14.18 → 10.17 → 22.11 |
| TSMC5 | device_linear | 2/2 | 2/2 | 2/2 | 0.3225 → 0.474 → 0.3797 |
| TSMC5 | device_output | 8/8 | 8/8 | 8/8 | 2.052 → 2.002 → 1.159 |
| TSMC5 | device_subthreshold | 2/2 | 2/2 | 2/2 | — → — → — |
| TSMC5 | diffpair_active | 4/4 | 4/4 | 4/4 | 6.907 → 3.495 → 11.15 |
| TSMC5 | diffpair_active_load | 4/4 | 2/4 | 4/4 | 0.3367 → 0.318 → 0.3822 |
| TSMC5 | diffpair_ideal | 4/4 | 4/4 | 4/4 | 157.8 → 40.33 → 325.3 |
| TSMC5 | diode_load | 5/5 | 5/5 | 5/5 | 0.3694 → 0.1524 → 0.04364 |
| TSMC5 | inverter_chain | 1/1 | 1/1 | 1/1 | 0.7793 → 0.7792 → 0.7791 |
| TSMC5 | inverter_energy | 2/2 | 2/2 | 2/2 | 0.1147 → 0.08574 → 0.2184 |
| TSMC5 | ldo_regulator | 4/4 | 4/4 | 4/4 | 382.3 → 64.93 → 22.03 |
| TSMC5 | multistage_buffer_12t | 0/2 | 2/2 | 2/2 | — → — → — |
| TSMC5 | nand2 | 3/3 | 2/3 | 3/3 | 4.068 → 4.068 → 4.068 |
| TSMC5 | nn_ac | 2/2 | 2/2 | 2/2 | 2.552 → 1.447 → 2.735 |
| TSMC5 | nn_lifted_source_dc | 6/6 | 6/6 | 6/6 | 0.07746 → 0.0767 → 0.07623 |
| TSMC5 | nn_parametric_dc | 26/26 | 26/26 | 26/26 | 6.44 → 6.472 → 5.419 |
| TSMC5 | nn_parametric_inverter | 20/20 | 20/20 | 20/20 | 4.325 → 4.31 → 4.323 |
| TSMC5 | nn_subckt | 3/3 | 3/3 | 3/3 | 0.7629 → 0.1862 → 1.419 |
| TSMC5 | nor2 | 2/3 | 3/3 | 3/3 | 5.21 → 4.084 → 4.066 |
| TSMC5 | opamp | 3/3 | 3/3 | 3/3 | 0.1099 → 0.04027 → 0.00689 |
| TSMC5 | opamp_ac | 1/1 | 1/1 | 1/1 | 42.81 → 25.52 → 4.718 |
| TSMC5 | opamp_rejection | 4/4 | 4/4 | 4/4 | 9.583 → 3.265 → 5.806 |
| TSMC5 | ota_5t_buffer | 3/3 | 3/3 | 3/3 | 0.5833 → 0.5062 → 0.2087 |
| TSMC5 | ring_osc | 3/3 | 3/3 | 3/3 | 1.92 → 2.679 → 6.225 |
| TSMC5 | ring_osc_supply | 1/1 | 1/1 | 1/1 | 8.026 → 9.882 → 9.012 |
| TSMC5 | self_biased_cascode | 1/1 | 1/1 | 1/1 | 0.04161 → 0.08302 → 0.05682 |
| TSMC5 | self_biased_cascode_pmos | 1/1 | 1/1 | 1/1 | 0.03858 → 0.05408 → 0.0581 |
| TSMC5 | source_follower | 2/2 | 2/2 | 2/2 | 0.03443 → 0.02825 → 0.0414 |
| TSMC5 | sram6t_modes | 6/8 | 8/8 | 8/8 | 4.187 → 4.186 → 4.186 |
| TSMC5 | sram_snm | 1/1 | 1/1 | 1/1 | 0.2865 → 0.9788 → 0.415 |
| TSMC5 | switchcap | 1/1 | 1/1 | 1/1 | 0.04732 → 0.1104 → 0.05688 |
| TSMC5 | switchcap_multicycle | 1/1 | 1/1 | 1/1 | 0.0729 → 0.07022 → 0.04518 |
| TSMC5 | terminal_capacitance | 10/10 | 10/10 | 10/10 | 0.4401 → 0.2468 → 0.3629 |
| TSMC5 | terminal_currents | 8/8 | 8/8 | 8/8 | 0.07568 → 0.07849 → 0.1557 |
| TSMC5 | transmission_gate_dc | 2/2 | 2/2 | 2/2 | 0.08615 → 0.06423 → 0.06594 |
| TSMC5 | transmission_gate_hold | 1/1 | 1/1 | 1/1 | 9.583e-14 → 9.583e-14 → 9.583e-14 |
| TSMC5 | unity_gain_buffer | 1/3 | 3/3 | 3/3 | 2.285 → 0.8984 → 1.38 |
| TSMC12 | beta_multiplier | 0/3 | 3/3 | 3/3 | — → — → — |
| TSMC12 | bias_tree_fanout_17t | 1/1 | 1/1 | 1/1 | 0.01399 → 0.01573 → 0.02626 |
| TSMC12 | bias_tree_fanout_3t | 1/1 | 1/1 | 1/1 | 0.01399 → 0.01573 → 0.02626 |
| TSMC12 | bias_tree_fanout_5t | 1/1 | 1/1 | 1/1 | 0.01399 → 0.01573 → 0.02626 |
| TSMC12 | bias_tree_fanout_9t | 1/1 | 1/1 | 1/1 | 0.01399 → 0.01573 → 0.02626 |
| TSMC12 | cascode_stack | 0/3 | 3/3 | 3/3 | — → — → — |
| TSMC12 | common_gate | 2/2 | 2/2 | 2/2 | 0.1513 → 0.1098 → 0.06807 |
| TSMC12 | common_source_nn | 4/4 | 4/4 | 4/4 | 1.884 → 3.235 → 4.688 |
| TSMC12 | current_mirror | 4/4 | 4/4 | 4/4 | 0.5686 → 0.3483 → 0.4827 |
| TSMC12 | device_derivative | 6/6 | 6/6 | 6/6 | 16.9 → 27.37 → 43.47 |
| TSMC12 | device_linear | 2/2 | 2/2 | 2/2 | 0.1822 → 0.2193 → 0.4843 |
| TSMC12 | device_output | 8/8 | 8/8 | 8/8 | 1.841 → 1.09 → 0.6949 |
| TSMC12 | device_subthreshold | 2/2 | 2/2 | 2/2 | 0.1221 → 0.07144 → 0.02986 |
| TSMC12 | diffpair_active | 4/4 | 4/4 | 4/4 | 9.134 → 2.624 → 0.6814 |
| TSMC12 | diffpair_active_load | 4/4 | 2/4 | 4/4 | 0.9348 → 1.687 → 0.3461 |
| TSMC12 | diffpair_ideal | 4/4 | 4/4 | 4/4 | 3.587 → 2.104 → 1.673 |
| TSMC12 | diode_load | 5/5 | 5/5 | 5/5 | 0.3294 → 0.09409 → 0.03754 |
| TSMC12 | inverter_chain | 1/1 | 1/1 | 1/1 | 0.5764 → 0.5763 → 0.5764 |
| TSMC12 | inverter_energy | 2/2 | 2/2 | 2/2 | 0.1922 → 0.12 → 0.1198 |
| TSMC12 | ldo_regulator | 4/4 | 4/4 | 4/4 | 299.5 → 173.2 → 10.89 |
| TSMC12 | multistage_buffer_12t | 2/2 | 0/2 | 2/2 | — → — → — |
| TSMC12 | nand2 | 3/3 | 2/3 | 3/3 | 4.038 → 4.038 → 4.038 |
| TSMC12 | nn_ac | 2/2 | 2/2 | 2/2 | 1.744 → 2.353 → 4.023 |
| TSMC12 | nn_lifted_source_dc | 6/6 | 6/6 | 6/6 | 0.1479 → 0.04815 → 0.05157 |
| TSMC12 | nn_parametric_dc | 30/30 | 30/30 | 30/30 | 4.25 → 2.979 → 1.903 |
| TSMC12 | nn_parametric_inverter | 20/20 | 20/20 | 20/20 | 4.045 → 4.039 → 4.046 |
| TSMC12 | nn_subckt | 3/3 | 3/3 | 3/3 | 0.3694 → 1.208 → 0.433 |
| TSMC12 | nor2 | 2/3 | 2/3 | 2/3 | 4.085 → 4.079 → 4.036 |
| TSMC12 | opamp | 3/3 | 3/3 | 3/3 | 1.437 → 0.08575 → 0.1671 |
| TSMC12 | opamp_ac | 1/1 | 1/1 | 1/1 | 25.57 → 0.7701 → 14.61 |
| TSMC12 | opamp_rejection | 4/4 | 4/4 | 4/4 | 16.62 → 10.37 → 2.82 |
| TSMC12 | ota_5t_buffer | 3/3 | 0/3 | 3/3 | — → — → — |
| TSMC12 | ring_osc | 3/3 | 3/3 | 3/3 | 12.11 → 10.93 → 5.649 |
| TSMC12 | ring_osc_supply | 1/1 | 1/1 | 1/1 | 17.5 → 15.09 → 4.586 |
| TSMC12 | self_biased_cascode | 1/1 | 0/1 | 1/1 | — → — → — |
| TSMC12 | self_biased_cascode_pmos | 1/1 | 1/1 | 1/1 | 0.1077 → 0.06846 → 0.06698 |
| TSMC12 | source_follower | 2/2 | 2/2 | 2/2 | 0.06021 → 0.0496 → 0.04043 |
| TSMC12 | sram6t_modes | 6/8 | 6/8 | 8/8 | 4.935 → 4.87 → 4.872 |
| TSMC12 | sram_snm | 1/1 | 1/1 | 1/1 | 2.358 → 3.875 → 1.135 |
| TSMC12 | switchcap | 1/1 | 1/1 | 1/1 | 0.1134 → 0.1139 → 0.2626 |
| TSMC12 | switchcap_multicycle | 1/1 | 1/1 | 1/1 | 0.06424 → 0.0626 → 0.1439 |
| TSMC12 | terminal_capacitance | 10/10 | 10/10 | 10/10 | 0.3774 → 0.4532 → 0.5805 |
| TSMC12 | terminal_currents | 8/8 | 8/8 | 8/8 | 0.15 → 0.116 → 0.258 |
| TSMC12 | transmission_gate_dc | 2/2 | 2/2 | 2/2 | 0.1675 → 0.1543 → 0.0666 |
| TSMC12 | transmission_gate_hold | 1/1 | 1/1 | 1/1 | 9.597e-14 → 9.597e-14 → 9.597e-14 |
| TSMC12 | unity_gain_buffer | 3/3 | 3/3 | 3/3 | 2.637 → 0.3158 → 0.1958 |

## Complete large-model testcase inventory

R and S are identical in this complete comparison; all ERROR slots remain.

| Technology | Case | R0 converged | R/S converged | Worst paired NRMSE (%) R0 → R/S |
|---|---|---:|---:|---:|
| TSMC5 | beta_multiplier | 3/3 | 0/3 | — → — |
| TSMC5 | bias_tree_fanout_17t | 1/1 | 1/1 | 0.01162 → 0.01281 |
| TSMC5 | bias_tree_fanout_3t | 1/1 | 1/1 | 0.01162 → 0.01281 |
| TSMC5 | bias_tree_fanout_5t | 1/1 | 1/1 | 0.01163 → 0.01281 |
| TSMC5 | bias_tree_fanout_9t | 1/1 | 1/1 | 0.01163 → 0.01281 |
| TSMC5 | cascode_stack | 3/3 | 0/3 | — → — |
| TSMC5 | common_gate | 2/2 | 2/2 | 0.09735 → 0.05045 |
| TSMC5 | common_source_nn | 4/4 | 4/4 | 0.9647 → 0.668 |
| TSMC5 | current_mirror | 4/4 | 4/4 | 0.4509 → 0.1628 |
| TSMC5 | device_derivative | 6/6 | 6/6 | 11.38 → 12.21 |
| TSMC5 | device_linear | 2/2 | 2/2 | 0.4732 → 0.2623 |
| TSMC5 | device_output | 8/8 | 8/8 | 2.09 → 2.1 |
| TSMC5 | device_subthreshold | 2/2 | 2/2 | — → — |
| TSMC5 | diffpair_active | 4/4 | 4/4 | 28.33 → 7.569 |
| TSMC5 | diffpair_active_load | 2/4 | 4/4 | 0.605 → 0.04296 |
| TSMC5 | diffpair_ideal | 0/4 | 4/4 | — → — |
| TSMC5 | diode_load | 5/5 | 5/5 | 0.2894 → 0.1676 |
| TSMC5 | inverter_chain | 1/1 | 1/1 | 0.7897 → 0.7788 |
| TSMC5 | inverter_energy | 2/2 | 2/2 | 0.254 → 0.1143 |
| TSMC5 | ldo_regulator | 0/4 | 0/4 | — → — |
| TSMC5 | multistage_buffer_12t | 0/2 | 0/2 | — → — |
| TSMC5 | nand2 | 3/3 | 3/3 | 121 → 15.44 |
| TSMC5 | nn_ac | 2/2 | 2/2 | 2.976 → 3.223 |
| TSMC5 | nn_lifted_source_dc | 6/6 | 6/6 | 0.1213 → 0.05579 |
| TSMC5 | nn_parametric_dc | 26/26 | 26/26 | 7.724 → 6.808 |
| TSMC5 | nn_parametric_inverter | 20/20 | 20/20 | 4.333 → 4.323 |
| TSMC5 | nn_subckt | 3/3 | 3/3 | 24.73 → 13.92 |
| TSMC5 | nor2 | 3/3 | 3/3 | 99.66 → 11.86 |
| TSMC5 | opamp | 3/3 | 3/3 | 0.2131 → 0.06122 |
| TSMC5 | opamp_ac | 1/1 | 1/1 | 48.52 → 1.863 |
| TSMC5 | opamp_rejection | 4/4 | 4/4 | 26.33 → 20.27 |
| TSMC5 | ota_5t_buffer | 0/3 | 0/3 | — → — |
| TSMC5 | ring_osc | 3/3 | 3/3 | 4.153 → 1.612 |
| TSMC5 | ring_osc_supply | 1/1 | 1/1 | 4.333 → 3.173 |
| TSMC5 | self_biased_cascode | 0/1 | 0/1 | — → — |
| TSMC5 | self_biased_cascode_pmos | 1/1 | 1/1 | 0.3429 → 0.1134 |
| TSMC5 | source_follower | 2/2 | 2/2 | 0.07005 → 0.03594 |
| TSMC5 | sram6t_modes | 6/8 | 6/8 | 52.79 → 4.186 |
| TSMC5 | sram_snm | 1/1 | 1/1 | 0.7562 → 0.1544 |
| TSMC5 | switchcap | 1/1 | 1/1 | 0.09383 → 0.1001 |
| TSMC5 | switchcap_multicycle | 1/1 | 1/1 | 0.09598 → 0.07321 |
| TSMC5 | terminal_capacitance | 10/10 | 10/10 | 0.6453 → 0.3202 |
| TSMC5 | terminal_currents | 8/8 | 8/8 | 0.08583 → 0.08302 |
| TSMC5 | transmission_gate_dc | 2/2 | 2/2 | 0.1033 → 0.08893 |
| TSMC5 | transmission_gate_hold | 1/1 | 1/1 | 9.583e-14 → 9.583e-14 |
| TSMC5 | unity_gain_buffer | 3/3 | 3/3 | 7.392 → 1.199 |
| TSMC12 | beta_multiplier | 0/3 | 1/3 | — → — |
| TSMC12 | bias_tree_fanout_17t | 1/1 | 1/1 | 0.0133 → 0.02967 |
| TSMC12 | bias_tree_fanout_3t | 1/1 | 1/1 | 0.0133 → 0.02967 |
| TSMC12 | bias_tree_fanout_5t | 1/1 | 1/1 | 0.0133 → 0.02967 |
| TSMC12 | bias_tree_fanout_9t | 1/1 | 1/1 | 0.0133 → 0.02967 |
| TSMC12 | cascode_stack | 3/3 | 3/3 | 0.4282 → 2.948 |
| TSMC12 | common_gate | 2/2 | 2/2 | 0.09216 → 0.02893 |
| TSMC12 | common_source_nn | 4/4 | 4/4 | 2.112 → 0.9045 |
| TSMC12 | current_mirror | 4/4 | 4/4 | 0.5527 → 0.3725 |
| TSMC12 | device_derivative | 6/6 | 6/6 | 22.07 → 7.988 |
| TSMC12 | device_linear | 2/2 | 2/2 | 0.2203 → 0.1938 |
| TSMC12 | device_output | 8/8 | 8/8 | 2.06 → 1.098 |
| TSMC12 | device_subthreshold | 2/2 | 2/2 | 0.2044 → 0.1362 |
| TSMC12 | diffpair_active | 4/4 | 4/4 | 0.6534 → 5.047 |
| TSMC12 | diffpair_active_load | 2/4 | 2/4 | 2.396 → 1.389 |
| TSMC12 | diffpair_ideal | 4/4 | 0/4 | — → — |
| TSMC12 | diode_load | 5/5 | 5/5 | 0.1905 → 0.1355 |
| TSMC12 | inverter_chain | 1/1 | 1/1 | 0.586 → 0.5777 |
| TSMC12 | inverter_energy | 2/2 | 2/2 | 0.27 → 0.1756 |
| TSMC12 | ldo_regulator | 0/4 | 4/4 | — → — |
| TSMC12 | multistage_buffer_12t | 0/2 | 1/2 | — → — |
| TSMC12 | nand2 | 2/3 | 3/3 | 33.54 → 7.242 |
| TSMC12 | nn_ac | 2/2 | 2/2 | 5.368 → 3.368 |
| TSMC12 | nn_lifted_source_dc | 6/6 | 6/6 | 0.05675 → 0.0366 |
| TSMC12 | nn_parametric_dc | 30/30 | 30/30 | 8.457 → 12.3 |
| TSMC12 | nn_parametric_inverter | 20/20 | 20/20 | 4.029 → 4.034 |
| TSMC12 | nn_subckt | 3/3 | 3/3 | 90.85 → 43.07 |
| TSMC12 | nor2 | 2/3 | 3/3 | 36.49 → 5.795 |
| TSMC12 | opamp | 3/3 | 3/3 | 4.391 → 1.739 |
| TSMC12 | opamp_ac | 1/1 | 1/1 | 55.68 → 45.88 |
| TSMC12 | opamp_rejection | 4/4 | 4/4 | 75.03 → 9.265 |
| TSMC12 | ota_5t_buffer | 0/3 | 3/3 | — → — |
| TSMC12 | ring_osc | 3/3 | 3/3 | 7.11 → 12.36 |
| TSMC12 | ring_osc_supply | 1/1 | 1/1 | 7.533 → 34.18 |
| TSMC12 | self_biased_cascode | 1/1 | 1/1 | 0.08723 → 0.0568 |
| TSMC12 | self_biased_cascode_pmos | 1/1 | 1/1 | 0.2873 → 0.082 |
| TSMC12 | source_follower | 2/2 | 2/2 | 0.05847 → 0.01229 |
| TSMC12 | sram6t_modes | 6/8 | 6/8 | 8.79 → 4.87 |
| TSMC12 | sram_snm | 1/1 | 1/1 | 1.911 → 2.778 |
| TSMC12 | switchcap | 1/1 | 1/1 | 0.1097 → 0.06215 |
| TSMC12 | switchcap_multicycle | 1/1 | 1/1 | 0.1554 → 0.0543 |
| TSMC12 | terminal_capacitance | 10/10 | 10/10 | 0.3942 → 0.3731 |
| TSMC12 | terminal_currents | 8/8 | 8/8 | 0.1062 → 0.03844 |
| TSMC12 | transmission_gate_dc | 2/2 | 2/2 | 0.08753 → 0.07618 |
| TSMC12 | transmission_gate_hold | 1/1 | 1/1 | 9.597e-14 → 9.597e-14 |
| TSMC12 | unity_gain_buffer | 3/3 | 3/3 | 2.203 → 1.471 |
