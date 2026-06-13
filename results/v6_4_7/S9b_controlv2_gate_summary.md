# V6.4.7 S9b — control-v2 gate summary + data-change go/no-go

**Date:** 2026-06-14 · **Verdict: PROCEED** (data change sound; do not rewind)

## Environment rebuild (this session)

S9b executed on a machine that was a **bare source checkout** — all runtime
infrastructure was missing and had to be rebuilt before any data work:

- PyCMG submodule restored via working proxy (pinned commit gone from remote →
  on `feat/v6`, the V6.3/V6.3.1 generator lineage with `inv_trip`+`reverse_vds`).
- OpenVAF 23.5.0 + BSIM-CMG OSDI binary built (`build/osdi/bsimcmg.osdi`),
  validated (PyCMG API 20/20).
- conda env `pycircuitsim` + torch 2.6.0+cu124 (CUDA on 3× RTX 4090).
- NGSPICE 45.2 + OSDI built from source (`tools/ngspice-45.2/bin/ngspice`),
  OSDI load verified vs PyCMG; harness made `NGSPICE_BIN`-configurable.
- TSMC PDKs supplied by the user.

## S9b generator reconstruction (lost-commit code) + bug fixes

Reconstructed on `feat/v6` (patch: `results/v6_4_7/s9b_pycmg_patch/`):
- `NN_DC_SOLVE_TOL` floor fix (`model.py`): exact-zero rows 10.0% → 1.3%;
  (1e-12,1e-9] band 0 → 537/bin.
- `subvt_off` sample class (code 11) + `--enable-subvt-off`/`--dc-solve-tol`.
- **Bug 1 — parallel modelcard-cache write race:** T-sibling workers shared one
  on-the-fly naive-card file → partial reads → degenerate cards (only
  PHIG/TOXP non-zero) → tech-variant labeller assert. Fixed with atomic
  temp-file + `os.replace`.
- **Bug 2 — NFIN<2 inclusion** (feat/v6 vs main CLAUDE.md Rule 9): excluded in
  `enumerate_bins`.

## Regen-v2 acceptance (all PASS)

- 8 datasets (tsmc{5,7,12,16}_v2_{nmos,pmos}.npz), 1.8–3.1M rows each.
- **Decade gate PASS 8/8** — 40k–200k rows/decade in 1e-12..1e-6 A (gate 1k).
- **asinh audit:** `drift_id = 1.0000` (no s_id pinning needed); gm/gds drift
  0.73–0.96 (P4-relevant).
- **tech-variant labeller: 0 misses** on all 8 (race+NFIN fix verified).

## control-v2 retrain

32 cells = 4 seeds {42,17,7,31} × 8 (tech×dev), stock medium recipe,
`--apply-filter off` (unfiltered small-current data), EMA, v2 data. 31/32 on
first pass (1 CUDA-contention FAIL re-run clean). Checkpoints
`v6_4_7_ctlv2_s<seed>_<tech>_<dev>` (inert; not resolver-matching).

## Full multi-tech gate (per-cell best seed vs baseline_v6_4_7_pre.json)

| tech | ring_osc | opamp | switchcap | inverter |
|------|----------|-------|-----------|----------|
| tsmc5  | **REGR** 5.80% (base 2.61 pass; all seeds 5.8–7.85) | pass 0.79% (s7) | fail 11.5% (base fail) | pass 0.93% (s7) |
| tsmc7  | fail 8.66% (base fail 8.28) | fail 10.46% (base fail) | pass 1.60% (s17) | pass 1.39% (s31) |
| tsmc12 | pass 2.04% (s7) | **REGR** collapsed (base 5.21 pass; all 4 seeds flat) | pass 4.01% (s17) | pass 1.09% (s17) |
| tsmc16 | pass 2.18% (s17) | fail 10.58% (base fail) | **NEW-PASS** 3.17% (base fail 13.1) | pass 1.06% (s17) |

**2 protected-gate regressions** (tsmc5 RO, tsmc12 opamp); **1 new-pass**
(tsmc16 SC); **inverters hold on all 4 techs**.

## Go/no-go: PROCEED — regressions are fresh-retrain variance, not data defects

- **EMA ruled out:** EMA-off ablation (tsmc5 s7) gives RO 7.23% ≈ EMA-on 7.21%
  (RO-neutral; EMA stays as a free default).
- **tsmc5 RO regression = lost best-of-8 cherry-pick.** Confirmed by tsmc7:
  where the baseline RO was *not* a cherry-pick (8.28%, systematic bias),
  control-v2 matches it (8.66%); where it was (tsmc5 2.61%), fresh retrains
  land at the un-selected ~6%.
- **tsmc12 opamp regression = 4-seed spontaneous-collapse lottery.**
  P(all-4-collapse) ≈ (13/16)⁴ ≈ 44%; tsmc5 s7 proves a seed *can* pass at
  0.79%. Recoverable with more seeds + the P4 collapse-resistance arm.
- **Data is sound:** decade/asinh/labeller pass, inverters hold everywhere
  (no s_id drift), EMA RO-neutral, and the v2 data **improves** tsmc16 SC
  (13.1% → 3.17%).

**Decision:** the data change (regen-v2 + filter-off + subvt_off + NFIN≥2) is
NOT the cause of the regressions → **do not rewind**. control-v2 is the
fresh-retrain attribution baseline the S10+ arms A/B against; the S8
`baseline_v6_4_7_pre.json` remains the promotion gatekeeper. **tsmc5 ring_osc
and tsmc12 opamp join the arms' recover-set** (P4 collapse-resistance; P5/P8a
RO). Optional follow-up: 4 more control-v2 seeds on tsmc12 to confirm the
opamp lottery and complete a clean attribution mix.

## Gate files

`results/v6_4_7/S9b_controlv2_gate_{tsmc5,tsmc7,tsmc12_16}.md` (raw per-seed
RESULT vectors inline). Gate driver: `scripts/v6_4_7_s9b_gate_controlv2.py`
(GPU-serial `--workers 1`, or `--cpu`; deriv-fidelity optional via `--deriv`).
