# DirectNet-Full (LEVEL=75) — terminal-length follow-up

Date: 2026-08-29

Status: **experimental; no promotion**. This follow-up removed the dominant
terminal-length support gap and increased measurable AnalogGym metric coverage,
but it produced **no new fully passing complex-circuit deck**. All trial-only
runtime changes were reverted from the primary worktree.

NGSPICE 45.2 using the identical BSIM-CMG LEVEL=72 OSDI model remained ground
truth. CPU inference used one OpenMP, MKL, and Torch thread.

## Outcome

| arm | focused result | decision |
|---|---|---|
| `G-terminal-L`, unmatched regeneration | Ten clean datasets and checkpoints cover the PDK terminal length; the five-technology Song AC canary has zero input-5 errors but 0/5 completed decks | Geometry mechanism accepted; model not promoted |
| support-aware evaluator limiter | TSMC5 Song completes 221/221 AC points, with 1/8 metrics agreeing; the five-tech canary completes 1/5 and passes 0/5 | Rejected: no deck-pass gain and severe wall-time cost |
| `G-terminal-L-matched` | Exact V7.6.3 rows plus terminal-only tails lower TSMC5 NMOS held-out average NRMSE from 0.064% to 0.041%, but Song does not converge | Rejected: pointwise improvement did not select a usable circuit basin |
| `J-current`, 12 epochs | Current-Jacobian MAE falls 67% NMOS and 70% PMOS; Song does not converge | Rejected |
| `J-current`, one epoch | Smaller value drift and 33–43% lower Jacobian MAE; Song still does not converge | Rejected |

The fixed TSMC5 AnalogGym DC subset also remains **0/15 decks passing**. Seven
rows completed before the 300-second cap and eight timed out. Among completed
numeric cells, coverage improved from 2/14 agreeing in the V7.6.3 medium
control to 11/44 agreeing, but partial metric agreement is not a deck pass.

## Success criteria

The loop started from the medium V7.6.3 control in
[`DirectNet-L75-v763-targeted.md`](DirectNet-L75-v763-targeted.md). An arm had
to:

1. remove input-5 terminal-length errors without fabricating NFIN=1 support;
2. retain independently generated, checksum-bound LEVEL=72 data provenance;
3. increase fully passing complex-circuit decks, not merely suppress errors;
4. avoid regressions in convergence and wall time before a full simple-gate
   confirmation.

NFIN=1 remained explicitly unsupported throughout this work.

## Review → proposal → evaluation → analysis

### Loop 1 — terminal length

**Review.** V7.6.3 medium failed 184 AnalogGym rows at input 5 because the
saved data stopped at 107.754 nm for TSMC5 and 190.493 nm elsewhere. The
current parser enumerates the actual PDK terminal edges, 135.01 nm and
240.01 nm.

**Proposal.** Regenerate all ten full-terminal datasets from clean source,
then fine-tune the V7.6.3 medium checkpoints without manually widening a
normalizer.

**Evaluation.** Generation commit `3d38e2d` produced ten canonical datasets:

- 4.81–6.48 million rows per polarity/technology;
- 206 new geometry keys and 618 new temperature bins;
- 320,940–541,050 terminal-length rows per artifact;
- minimum NFIN=2; no duplicate geometry bins or unapproved rejection;
- trained normalization support ending at exactly 135.01 or 240.01 nm.

All ten training bundles were finite and completion-marker bound. The five
Song canaries no longer failed at input 5, but each immediately exposed a
voltage-domain escape instead. No deck completed.

**Analysis.** Terminal enumeration fixes a real static support defect, but
does not solve nonlinear trial states or model fidelity. This arm also was not
a strict one-factor experiment: inserting bins shifted the historical
`base_seed + counter` seed for 74.1% of common bins.

### Loop 2 — physical state versus Newton trial

**Review.** TSMC5 Song first failed at PMOS `M.Xop1.Mm61` with trial
`Vbs=+0.975 V`, outside `[-0.78,+0.78] V`.

An independent LEVEL=72 operating-point solve on the identical deck found the
same device at source-relative
`(Vds,Vgs,Vs,Vbs)=(-0.370089,-0.463079,0,+0.031040) V`. Topology and terminal
frame matched. The failing value was therefore a Newton trial, not a physical
state needing more data.

**Proposal.** First tighten the node trust region; then try a device evaluator
projection that linearizes at the nearest certified voltage point while the
existing `_nr_limited` gate forbids convergence on a projected stamp.

**Evaluation.** A 0.10 V cap moved the failure to `Vbs=0.8 V` in 2.95 s. A
0.05 V cap delayed it to 122.6 s and then failed at `Vds=0.7984 V`. Cap-only
globalization was rejected.

The support-aware evaluator limiter passed 18/18 focused tests, including
identity inside support and aliased-terminal preservation. Results were:

| technology | Song result | Py time |
|---|---|---:|
| TSMC5 | completed; 1/8 metrics agree; worst OP error 29.60 mV | 123.8 s |
| TSMC6 | late voltage-support error | 207.6 s |
| TSMC7 | late voltage-support error | 211.0 s |
| TSMC12 | late voltage-support error | 181.7 s |
| TSMC16 | DC did not converge | 239.7 s |

The fixed, non-cherry-picked TSMC5 subset comprised all six DC-source and all
nine DC-temperature decks. It ran from clean evaluation snapshot `fc85a8c`
with 15 workers and a 300-second per-deck cap:

| result | V7.6.3 medium | terminal-L + limiter |
|---|---:|---:|
| full deck pass | 0/15 | 0/15 |
| rows returning numeric comparisons before cap | 15/15, mostly partial | 7/15 |
| comparable metric cells agreeing | 2/14 | 11/44 |
| timed out | 0 | 8 |

The completed candidate rows still showed large state error: five rows with
comparable voltage samples aggregate to 13.74% symmetric MRE, 138.17% NRMSE,
R² -22.22, and 22.02 V maximum error.

**Analysis.** Projection is a useful diagnostic continuation mechanism, but
it trades fast, explicit failure for long solves without a deck-pass gain. It
was reverted rather than shipped.

### Loop 3 — matched terminal append

**Review.** The first regeneration confounded new geometry with changed common
samples.

**Proposal.** Preserve every selected V7.6.3 training row byte-for-byte and
append only terminal-length rows from the clean regeneration.

**Evaluation.** Builder commit `b6ea582` merged the exact base manifest with
new terminal-bin entries and recorded both parent hashes. A chunked audit
proved exact base prefixes and exact regenerated tails for all ten artifacts.
Only the TSMC5 polarity pair was trained as a screening gate; the other eight
jobs were deferred after TSMC5 failed.

The matched TSMC5 NMOS held-out average NRMSE improved from the V7.6.3
control's 0.064% to 0.041%. Nevertheless, Song did not converge after 246.0 s.

**Analysis.** Lower pointwise loss is anti-correlated with this circuit-basin
result. The matched construction is the correct future data A/B mechanism,
but the selected value-only objective is insufficient.

### Loop 4 — full-terminal current Jacobians

**Review.** Existing Sobolev loss targets the older reduced contract and
cannot supervise the six-surface full-terminal model. Its stored reverse-Vds
`gds` label is also known to be sign-corrupted, so it was not reused.

**Proposal.** Generate exact current Jacobians from
`condense_last_jacobian()`—the same matrix used by LEVEL=72—and fine-tune the
three independent current surfaces with broad value replay.

**Evaluation.** Overlay commit `565150c` generated 15,360 rows per polarity
over 60 TSMC5 terminal geometry/temperature/variant bins. The stored current
rows use the solver-positive leaving convention. Per-bin central differences
passed with worst error 0.102 times the established NGSPICE-backed tolerance.

Fine-tune commit `3c55429` used 65,536 replay rows per polarity. The selected
12-epoch candidate changed validation metrics as follows:

| polarity | value MAE | current-Jacobian MAE |
|---|---:|---:|
| NMOS | 0.001159 → 0.001695 | 1.6299 → 0.5436 |
| PMOS | 0.001057 → 0.002181 | 5.4260 → 1.6366 |

Song did not converge after 234.2 s. A one-epoch lower-drift checkpoint also
failed after 238.8 s.

**Analysis.** Terminal-only Jacobian fine-tuning changes the nonlinear basin
without preserving circuit behavior. A future Jacobian arm needs all-geometry
labels, a constrained value budget, and design-held-out circuit selection; a
lower overlay loss alone is not promotion evidence.

## Decision and next experiment

No model or solver arm is accepted, and LEVEL=75 remains experimental. Before
recording this evidence, the primary tracked source diff was restored to its
pre-follow-up SHA-256
`94be04b828c0288a34b7a0b5501634b931cabe9905009b93ca478e5575937797`.

The next justified model experiment is a from-scratch, all-geometry
full-terminal value-plus-current-Jacobian training run with:

- coordinate-derived bin seeds or the validated matched-append construction;
- exact condensed OSDI Jacobian labels, not reduced `gds` opvars;
- an explicit maximum value-error regression budget;
- model selection on held-out device derivatives and held-out circuit
  designs, not the training circuits;
- the five-tech Song canary before any 248-deck campaign.

NFIN=1 certification remains an independent prerequisite for complete corpus
coverage. Solver residual probes should also treat an out-of-domain averaged
iterate as non-acceptable rather than raising, but that is robustness work and
does not address the measured model error.

## Provenance and skipped checks

- Terminal generation source: `3d38e2d9a7de546f9f7365c3775a0c1a8aa4666e`.
- Matched builder source: `b6ea5825d16009d666315c4e83f729296c916226`;
  sorted completion-marker aggregate `fb70b61d94984355ad369f8f4a07eff2d333ce1ecf94eceae043ab75ede023e7`.
- Jacobian overlay source: `565150c63876e2587840fc8b9debadc8ecc3091b`;
  marker aggregate `a6f7ac03831a027ac816248bff8bc7cbe4ae9b6fef4a2cf42c9b159afe8e4456`.
- Jacobian fine-tune source: `3c55429dfe6187303a541ed671c698823cc75ddf`;
  selected-marker aggregate `ee126587e3d879a530d4fd8e1e52eea8402b5667357a80cc5b6748b43e5b74b8`.
- Clean limiter evaluation source: `fc85a8c6ec31d98ad4665acb6091de1f2f6427bf`;
  completed-row JSON aggregate `3ca28974e24b476920452f2e63feabac97091e27da05e0eb6045f4c65b4408c0`.
- The rejected local payloads under `results/v764_terminal_l*` were purged
  after the closure loop. The hashes and measurements in this report and the
  [condensed closure ledger](../plans/2026-08-29-v764-complex-circuit-closure-loop.md)
  are the retained record; raw payload recovery requires Git/history backups.
- Generator tests: 28 passed; full-terminal dataset tests: 19 passed. The
  temporary limiter path passed 18 focused tests. After its reversion, the
  focused primary suite passed 16/16. PyTorch deprecation messages were
  warnings, not skipped tests.
- A first clean-campaign attempt failed before simulation because ignored
  generated model includes were absent in the worktree. The same 12 include
  files were linked and all 15 rows rerun; only the corrected rerun counts.
- The full 248-deck campaign, all-technology matched training, and simple-gate
  confirmation were intentionally not run after their screening gates failed.
  This report therefore does not replace a clean qualification campaign.
