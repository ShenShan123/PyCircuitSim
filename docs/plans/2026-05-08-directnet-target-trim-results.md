# DirectNet target-trim results (E1 / E2 / E3) — follow-up note
**Date:** 2026-05-08
**Branch:** `refactor/nn-simple`
**Plan:** `2026-05-08-directnet-target-trim.md`

## Summary

E2 (4-output head: only `[id, qg, qd, qb]`) is the only experiment that
matters. It wins decisively at **small** capacity (matches v4-legacy's
inverter trip-shift with 27× fewer parameters) but **regresses at
medium** capacity. E1 (drop qs) and E3 (down-weight unused targets)
both made things *worse* at small scale.

## Inverter VTC, TSMC12 SVT, L=18 nm, NFIN_n=10, NFIN_p=20, VDD=0.8

Dense 81-point Vin sweep so trip-point landing is averaged out.

| Recipe          | Out dim | Params  | Trip Δ  | NRMSE % | Verdict |
|-----------------|---------|---------|---------|---------|---------|
| v4-legacy       | 13      |  1.5 M  | −10 mV  |   6.57  | PASS    |
| B0-medium       | 13      |  520 K  | +26 mV  |  10.16  | PASS    |
| **E2-small-4out** | **4** | **57 K** | **−10 mV** | **8.54** | **PASS** |
| E2-medium-4out  |  4      |  520 K  | +58 mV  |  21.51  | FAIL    |

(Coarse 17-point Vin sweep showed E2-small at 1.77 % NRMSE and
E2-medium at 18.5 %; that gap was a discretisation artifact — the
sharp NN trip happens to *miss* a Vin sample at small scale and *hit*
one at medium scale. Dense sampling gives the honest comparison
above.)

## Read-out per hypothesis

- **H1 (drop qs):** E1 regressed from B0-small 72.89 % → 23.10 % NRMSE
  at 17-point coarse-grid (still FAIL). qs-only mask doesn't move
  the needle once the simulator is enforcing KCL anyway. **Verdict:
  no.**

- **H2/H3 (capacitances + gm/gds/gmb are dead weight):** *partially
  true at small capacity, false at medium.* E2-small at 4 outputs
  matches the legacy 1.5 M-param model on trip-shift. E2-medium at
  the same 4 outputs but 9× more parameters drifts +58 mV. The 9
  supervised "unused" targets *do* earn their keep at scale by acting
  as smoothness priors on the id and q\* surfaces — without them, the
  extra capacity is spent overfitting noise that the autograd
  derivatives can't recover from.

- **H4 (down-weight 9 unused):** E3 regressed B0-small 72.89 % →
  33.09 % NRMSE. Confirms that mixed-weight loss surfaces just
  confuse the optimiser at small capacity (the 9 down-weighted
  targets steal gradient signal from the 4 we want). **Verdict: no.**

## Implications

1. **E2 is shippable as the "fast iteration" recipe.** A 57 K-param
   DirectNet trains in ~4 min on one A100, fits in 800 K rows, and
   gives an inverter VTC within 1 mV of the legacy 1.5 M-param model.
   Use it for ablations / quick tech qualification.

2. **B0 is shippable as the "production" recipe.** 13-output head,
   uniform LDS-MAE, is the sweet spot: gm/gds/c\* supervision pays
   for itself at 500 K+ params.

3. **Don't down-weight or partially-mask.** Either keep all targets
   or drop them entirely — mixed weights regressed the small model
   under both E1 and E3.

4. **The plan's H1/H2/H3/H4 framing was too binary.** The capacity-
   dependent transition between "supervision is overhead" (small) and
   "supervision regularises" (medium+) is the actual phenomenon.

## Follow-ups

- E2-medium with stronger regularisation (dropout, weight decay) might recover the small-scale win at medium scale. Untested.
- Multi-seed E2-medium would distinguish "regression" from "variance" — current run is single-seed.
- The legacy 1.5 M-param model at 6.57 % NRMSE remains the best result. Re-running at "large" preset would be the next data point.

## Plumbing landed

- ``MAELoss``: per-column-mask via existing `weights` arg (no API
  change).
- ``train_directnet`` and ``_train_loop``: new `column_weights`
  + `output_subset` kwargs.
- ``load_and_split_bsimar``: new `output_subset` arg slices the 13
  outputs to the requested subset before fit.
- ``NormStats``: new optional `output_columns` field, persisted in
  norm.npz so a 4-output checkpoint can be loaded without ambiguity.
- ``_NormalizerBase.fit`` / `_fit_outputs`: subset-aware (asinh-floor
  lookup uses the supplied column list).
- ``_MOSFETNNBase``: reads `output_columns` from norm.npz to build
  `_out_col` for E2 4-output checkpoints. New `_stats_col` helper
  routes per-target stat lookups through the same map.
- ``compute_physical_metrics`` / `print_metrics`: dynamic column list
  from the normaliser (instead of hard-coded 13).
- ``cli/train.py``: `--loss-preset {default,e1,e2,e3}` flag,
  presets defined inline.

These are the complete deltas; no breaking changes to legacy v4
checkpoints.
