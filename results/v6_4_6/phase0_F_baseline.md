# Phase 0 — P0-F: V6.4.4 baseline + Goodhart/holdout freeze

**Date:** 2026-06-01 • **Branch:** `feat/v6.4.6` • **Git HEAD:** `54c475948dfeb168a38f4a98563036d3c9c58721`
**Mode:** INSTRUMENTATION-ONLY (no retrain, no checkpoint modification). Ran the existing
`tests/verify_complex_*.py` harness on the canonical V6.4.4 checkpoints. Ground truth = NGSPICE
BSIM-CMG (LEVEL=72), as the verify scripts already use.

**Env:** conda `pycircuitsim`, every command prefixed `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`
(the ~20× VTC trip gain makes inverter/opamp scatter ~±1 %, so thread-pinning matters).
`BSIMAR_CHECKPOINT_DIR` was **not** set — runs used the canonical read-only slots in
`external_compact_models/bsimar/checkpoints/`. `P0A_RESIDUAL` was unset, so the sibling P0-A
solver instrumentation (env-gated, dormant) did not alter measured V6.4.4 behaviour.

## Canonical checkpoints (sha256, all 8 verified present)

| slot | sha256 |
|------|--------|
| tsmc5_dn_medium_nmos  | `22eef03e…acd125c3` |
| tsmc5_dn_medium_pmos  | `a6a09be0…8441bd71` |
| tsmc7_dn_medium_nmos  | `d8f91418…30d7057f` |
| tsmc7_dn_medium_pmos  | `395e451f…b88b4900` |
| tsmc12_dn_medium_nmos | `4a045557…9e35f166` |
| tsmc12_dn_medium_pmos | `88dc5f91…26ebaaa9` |
| tsmc16_dn_medium_nmos | `05ae0ba8…047aa0a50` |
| tsmc16_dn_medium_pmos | `127e53a8…398984b` |

(Full untruncated digests in `baseline_v6_4_4.json → meta.checkpoint_sha256`.)

## Commands (tee'd to `results/v6_4_6/phase0_logs/`)

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run -n pycircuitsim python tests/verify_complex_ring_osc.py --tech TSMC5,TSMC7,TSMC12,TSMC16   # -> p0f_ring_osc.log
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run -n pycircuitsim python tests/verify_complex_opamp.py    --tech TSMC5,TSMC12             # -> p0f_opamp.log
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run -n pycircuitsim python tests/verify_complex_sram_snm.py --tech TSMC7 --nfin 3           # -> p0f_sram_tsmc7_nfin3.log
```

## Measured vs expected (every cell run)

| Benchmark | Tech | Role | Metric | Measured | Expected (plan/CHANGELOG) | Δ | PASS/FAIL | Match? |
|-----------|------|------|--------|---------:|--------------------------:|---:|:---------:|:------:|
| ring_osc | TSMC12 | blind_veto | period err % | **3.01** | 3.01 | 0.00 | PASS | ✅ exact |
| ring_osc | TSMC16 | blind_veto | period err % | **2.88** | 2.88 | 0.00 | PASS | ✅ exact |
| ring_osc | TSMC7  | target | period err % | **8.97** (NG 46.64 / DN 50.82 ps) | 8.97 (NG 46.64 / DN 50.82) | 0.00 | FAIL | ✅ exact |
| ring_osc | TSMC5  | protected_no_worsen | period err % | **2.98** | (RO row, PASS) | — | PASS | ✅ |
| opamp | TSMC5  | protected_no_worsen | gain err % | **2.64** (gain 164.2 vs NG 160.0) | 2.64 | 0.00 | PASS | ✅ exact |
| opamp | TSMC12 | protected_no_worsen | gain err % | **10.94** (gain 167.8 vs NG 188.4) | 10.94 | 0.00 | FAIL | ✅ exact |
| sram_snm | TSMC7 NFIN=3 | blind_sram_corner | force_ic states | **0/2** | 0/8 grid (q≈0.87/qb≈0.20 attractor) | — | FAIL | ✅ consistent |

**Rule-16 quartet per cell** (MRE / R² / NRMSE / MaxErr) — full numbers in `baseline_v6_4_4.json`:

- RO TSMC12: MRE 232.94 %, R² 0.3405, NRMSE 33.83 %, MaxErr 811.55 mV.
- RO TSMC16: MRE 232.67 %, R² 0.4917, NRMSE 29.49 %, MaxErr 809.71 mV.
- RO TSMC7:  MRE 346.77 %, R² −0.6525, NRMSE 54.37 %, MaxErr 790.76 mV.
- RO TSMC5:  MRE 205.89 %, R² 0.6402, NRMSE 25.65 %, MaxErr 607.16 mV.
- opamp TSMC5:  Vout-curve MRE 97.81 %, R² −0.9399, NRMSE 69.41 %, MaxErr 662.88 mV; trip shift −148.0 mV.
- opamp TSMC12: Vout-curve MRE 48.22 %, R² 0.3300, NRMSE 40.74 %, MaxErr 737.05 mV; trip shift −72.0 mV.
- SRAM TSMC7 NFIN=3 butterfly: NG SNM 185.7 mV, DN SNM 371.0 mV, SNMerr 99.8 %, DN min(qb) 524.7 mV (positive ⇒ butterfly sub-gate PASS), NRMSE 54.96 %.

> Note on the RO/opamp waveform NRMSE/R²: these are large (RO TSMC7 R²<0) because the metric
> compares the full transient/Vout waveform sample-by-sample, where any phase/trip offset blows
> up pointwise error. The **gate** is the scalar (period err % for RO, gain err % for opamp) and
> those are what determine PASS/FAIL — they match expected exactly. This is baseline behaviour,
> not a regression.

### Blind SRAM corner (TSMC7 NFIN=3) detail

NFIN=3 is **off** the default `{2,5,10}` selection grid, so it is a genuine held-out corner.
- Butterfly lobe **positive** (min(qb)=524.7 mV ≥ 0) → SNM positivity sub-gate PASSES (this is the part counted in the 4/4 butterfly headline).
- `force_ic` **state1**: q=0.815, qb=0.226 → FAILED (lands on the inboard attractor, not the q=VDD rail).
- `force_ic` **state0**: q=0.226, qb=0.815 → FAILED (mirror).
- Attractor q≈0.82 / qb≈0.23 (×VDD=0.75 → q≈0.61 V / qb≈0.17 V) is consistent with the documented q≈0.87 / qb≈0.20 NN basin (D3). This is the **blind reference** for Phase-1 (`force_ic` solver-recovery) and Phase-3 (off-leakage) SRAM work.

## Committed baseline JSON

`results/v6_4_6/baseline_v6_4_4.json` — 7 cells, validated parseable. Each cell records
`{benchmark, tech, (nfin), role, metric, value, gate, passfail, detail{Rule-16 quartet + raw}}`,
plus `meta{git_head_sha, commands, log paths, all-8 checkpoint sha256, role glossary}`.

## Holdout role assignment — which cells are valid vetoes vs no-worsening-only

| Cell | Role | Usable as… |
|------|------|------------|
| **TSMC12 RO (PASS 3.01 %)** | `blind_veto` | **Valid PASS/FAIL veto.** Currently PASS → a Phase-2/3 candidate that drops it to FAIL is rejected. Never used to pick adapters/seeds/λ. |
| **TSMC16 RO (PASS 2.88 %)** | `blind_veto` | **Valid PASS/FAIL veto.** Currently PASS → same. |
| **TSMC5 opamp (PASS 2.64 %)** | `protected_no_worsen` | In-scope protection: must stay PASS (no-worsening). Currently PASS, so effectively also a veto-style guard on the protected gate. |
| **TSMC12 opamp (FAIL 10.94 %)** | `protected_no_worsen` | **No-worsening NUMERIC DIFF only — NOT a PASS/FAIL veto** (already FAILS). Gate that DN gain 167.8 does not drop further / gain stays ≥10 (flat-flag). |
| **TSMC7 RO (FAIL 8.97 %)** | `target` | The gate to close. Phase 2 must beat DN 50.82 ps (need ≤48.97 ps, ≤5 %). |
| **TSMC7 SRAM NFIN=3 (FAIL 0/2 force_ic)** | `blind_sram_corner` | Blind held-out corner; Phase-1/3 SRAM fix must rail it without it having been used in selection. |

## Discrepancies flagged (Rule 12)

**None.** Every measured number matches the plan's expected value to the recorded precision:
TSMC12 RO 3.01 = 3.01, TSMC16 RO 2.88 = 2.88, TSMC5 opamp 2.64 = 2.64, TSMC12 opamp 10.94 = 10.94,
TSMC7 RO 8.97 with NG 46.64 / DN 50.82 exactly. No "PASS" cell failed to actually pass; no metric
drifted >0.5 %. The two expected-PASS blind vetoes (TSMC12/TSMC16 RO) genuinely PASS; the two
expected-FAIL cells (TSMC7 RO, TSMC12 opamp) genuinely FAIL at the recorded magnitudes.

## DECISION

**The holdout set is FROZEN and usable for Phase-2/3 promotion gating.** Confirmed from actual
NGSPICE-referenced sims (not markdown), pinned in `baseline_v6_4_4.json` against git HEAD
`54c4759` with all 8 checkpoint sha256s. Two valid selection-blind PASS/FAIL vetoes (TSMC12 RO,
TSMC16 RO), two protected-no-worsening opamp diffs (TSMC5 PASS, TSMC12 FAIL-but-no-worsen), the
TSMC7 RO target (8.97 %), and one blind SRAM corner (TSMC7 NFIN=3, force_ic 0/2). No discrepancy.
Promotion in later phases must hold both blind RO vetoes PASS and must not worsen the protected
opamp gains, on top of beating the recorded target.
