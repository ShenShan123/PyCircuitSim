# v3 Wider-Voltage-Box Retrain — Post-Mortem

**Date:** 2026-04-21
**Branch:** `feat/bsimar-v4-tech-code`
**Commits:** `381bbfc` (rail-fix), `fc35daf` (DataLoader perf + diagnostic)
**Status:** Experiment rolled back. Rail-restoring inference fix (`381bbfc`) is the shipping solution.

## TL;DR

Generated a wider-voltage training dataset (`voltage_box_factor=3.0` vs the previous `2.0`) and retrained both NMOS and PMOS production-size BSIMAR Transformers on it, with sign+boundary losses enabled. Hypothesis: wider training data would reduce the "transition zone" where the rail-restoring inference correction engages, improving inverter accuracy.

**Result: the v3 retrain is uniformly worse than the prior production** on the inverter across all 4 TSMC techs. Inverter transient went from 6.78–12.13 % NRMSE (prior production + rail-fix) to 18.17–25.61 % NRMSE (v3 + rail-fix). The wider training distribution diluted model capacity on the actual operating range without producing any extrapolation benefit that the rail-fix wasn't already providing analytically.

**Decision:** roll back v3 checkpoints. Keep `datasets_v3/` on disk as a research artifact (in case we revisit with a larger model or different sampling strategy). Prior production checkpoints + rail-fix remain the production solution.

## Motivation

After diagnosing the BSIMAR inverter transient explosion as out-of-range Vds extrapolation (see `v4_vds_correction_report_2026_04_15.md` and commit `381bbfc`), the rail-restoring fix in `_apply_vds_correction()` resolved transients on all 4 techs. VTC accuracy was still ~14–20 % on some techs, driven by DC NRMSE on the inverter's specific slice (Vds=VDD/2, NFIN=10, L=16nm).

Two hypotheses:
- **H1 (training data):** the NN was trained on Vd ∈ [0, 2·VDD] = [0, 1.6 V] at TSMC12/16. Training on Vd ∈ [0, 3·VDD] = [0, 2.4 V] would give the model more samples near the rail-restoring boundary, yielding smoother accuracy at those points.
- **H2 (loss):** the existing production had no sign-consistency or Vds=0 boundary loss. Adding them (per commit `f3e5f56`) might reduce subthreshold sign errors.

v3 combined both hypotheses plus a fresh 100-epoch run.

## What was done

### Dataset generation

```
python external_compact_models/PyCMG/scripts/generate_nn_data.py \
    --device both --universal \
    --voltage-box-factor 3.0 \
    --n-workers 16 \
    --data-dir external_compact_models/bsimar/data/datasets_v3
```

- Runtime: ~45 min each for NMOS and PMOS (parallel, 16 workers)
- Output: `universal_nmos.npz` (2.9 GB, 12,082,757 samples), `universal_pmos.npz` (2.9 GB, 12,082,757 samples)
- Distribution: 50 % more samples at out-of-range Vds vs the 2.0× box

### Training runs

Production architecture, 100 epochs, sign+boundary loss, batch=8192 on A100 / Blackwell:

```
python -m bsimar.cli.train --model transformer --device-type {nmos,pmos} \
    --data external_compact_models/bsimar/data/datasets_v3/universal_{nmos,pmos}.npz \
    --exclude-techs asap7 --num-tech-codes 18 --cuda \
    --d-model 256 --nhead 8 --num-layers 6 --dim-feedforward 1024 --dropout 0.2 \
    --epochs 100 --batch-size 8192 --patience 100 --ar-finetune-epochs 5 \
    --sign-weight 5.0 --boundary-weight 2.0 \
    --exp-name v4_universal_signfix_v3 --overwrite
```

- NMOS v3 trained on Blackwell (96 GB, shared with other user): 23,427 s = 6.5 hr for 100 TF epochs
- PMOS v3 trained on A100 (40 GB): 24,822 s = 6.9 hr
- Both crashed in the 5-epoch AR-finetune phase with `torch.OutOfMemoryError` (`forward_scheduled` uses more memory than normal TF mode). Phys-best checkpoints were saved at epoch 100 before the crash, so both runs produced usable artifacts.

**Incidental speedup for future work:** added `num_workers=8 + pin_memory=True + persistent_workers=True` to the three DataLoader calls in `trainer.py` (commit `fc35daf`). GPU utilisation went from 10 % → 95–99 %. Per-epoch time dropped from 22 min (batch=1024, num_workers=0) to 4.3 min (batch=2048) / 4.8 min (batch=8192). This change is kept — it does not affect training math.

### Test-set NRMSE (BSIMARNormStats asinh phys space)

| Run | Final NRMSE | R² | TF val | AR val |
|-----|-------------|-----|--------|--------|
| **NMOS v3** | 0.818 % | 0.951 | 0.02447 | 0.02557 |
| **PMOS v3** | 0.966 % | 0.958 | 0.02533 | 0.02637 |
| NMOS prior production | 0.270 % | 0.994 | n/a | n/a |
| PMOS prior production | 0.252 % | 0.997 | n/a | n/a |

v3 is ~3× worse on aggregate test NRMSE. First clue that wider training hurt accuracy.

### Inverter verify (LEVEL=74 BSIMAR with rail-fix enabled)

Using `tests/verify_bsimar_v4_inverter.py --bsimar-prefix v4_universal_signfix_v3 --tech <name>` with the committed rail-restoring correction in `_apply_vds_correction()`.

#### Transient (post-startup NRMSE % of VDD)

| Tech | VDD | v3 + rail-fix | Prior production + rail-fix | Δ |
|------|-----|---------------|-----------------------------|---|
| TSMC5 | 0.65 V | **25.61 %** FAIL | 12.13 % PASS | **+13.5 (worse)** |
| TSMC7 | 0.75 V | **22.15 %** FAIL | 9.14 %  PASS | +13.0 |
| TSMC12 | 0.80 V | **18.23 %** FAIL | 6.78 %  PASS | +11.5 |
| TSMC16 | 0.80 V | **18.17 %** FAIL | 7.51 %  PASS | +10.7 |

#### VTC (static sweep NRMSE % of VDD)

| Tech | v3 AR | Prior AR | Δ |
|------|-------|----------|---|
| TSMC5 | **25.26 %** FAIL | 13.96 % | +11.3 |
| TSMC7 | **21.00 %** FAIL | 19.15 % | +1.9 |
| TSMC12 | **4.79 %** PASS | 4.10 % | +0.7 |
| TSMC16 | **3.71 %** PASS | 3.40 % | +0.3 |

TSMC12/16 VTC is essentially unchanged — they sit at the centre of the training distribution (VDD=0.80 V = universal training max, the dense middle of the data). TSMC5/7 VTC is much worse — they're at low VDD where the 3.0× box spreads density away from the operating range.

#### Single-device DC (Id-Vgs sweep)

| Tech | NMOS DC AR v3/prior | PMOS DC AR v3/prior |
|------|---------------------|---------------------|
| TSMC5 | 5.43 / 4.59 | 11.07 / 5.97 |
| TSMC7 | 18.10 / 14.72 | 5.63 / 3.06 |
| TSMC12 | 12.61 / 9.95 | 14.07 / 13.72 |
| TSMC16 | 11.50 / 8.96 | 14.30 / 13.48 |

Every single-device DC entry is worse in v3.

#### V(out) range during transient (V)

| Tech | NGSPICE (truth) | Prior BSIMAR | **v3 BSIMAR** |
|------|-----------------|--------------|----------------|
| TSMC5 | [−0.005, 0.655] | [−0.012, 0.894] | [−0.009, **1.158**] |
| TSMC12 | [−0.004, 0.803] | [−0.040, 0.916] | [−0.032, **1.246**] |
| TSMC16 | [−0.003, 0.803] | [−0.064, 0.931] | [−0.043, **1.244**] |

v3 produces larger rail overshoots — counter to the hypothesis that wider training would smooth the rail-fix transition.

## Diagnosis

Two compounding effects make v3 worse:

### 1. Capacity dilution

The 5.15 M-parameter Transformer has a fixed capacity. With `voltage_box_factor=3.0`, roughly 50 % of samples are at out-of-range Vds (|Vds| > VDD). In the 2.0× regime the model concentrated capacity on [0, VDD] where the inverter operates. Spreading capacity onto out-of-range data made in-range predictions uniformly worse.

### 2. The rail-fix obviates the need to train for extrapolation

The analytical rail-restoring term in `_apply_vds_correction()` already provides a smooth, physically-motivated continuation past `VDD_train`. There is no accuracy benefit from the NN having seen those voltages during training — the inference correction would override the NN's predictions there anyway (via `gds_extra = max(gds_nn, g_rail · overshoot / x_ref)`).

So v3 paid the full cost of capacity dilution for zero benefit.

### Why sign+boundary loss alone also didn't help

Pure comparison (sign+boundary loss, 2.0× data, 100 epochs, batch=2048) wasn't run — we combined both changes in v3. Priors from earlier diagnostic work: the 670 K probe *with* sign+boundary loss on 2.0× data hit similar aggregate NRMSE (0.72–0.84 %) and similar inverter transient (6.88–9.76 %) as the v3 production, which hints that sign+boundary loss doesn't move the inverter needle much on its own. The rail-fix is doing the load-bearing work.

## Retained artifacts

- **`external_compact_models/bsimar/data/datasets_v3/`** (5.8 GB on disk, git-ignored): kept per request. Potential future use: larger model capacity (e.g. 10 M parameters) might amortise the wider distribution, or sampling with voltage-dependent weighting could spend capacity only where it matters.
- **`tests/diag_tsmc7_nmos_coverage.py`**: new diagnostic that slices the training `.npz` by tech-variant. Commit `fc35daf`.
- **DataLoader perf fix** in `trainer.py`: `num_workers=8 + pin_memory + persistent_workers`. Kept — it's a generic speedup unaffected by this rollback.

## What was rolled back

- `v4_universal_signfix_v3_{nmos,pmos}_*` checkpoints (10 files, ~100 MB): **deleted**.

## Forward guidance

For anyone retrying a "train for the rails" strategy:

1. **Don't widen the training box further without widening the model.** 5 M params on 3.0× data is worse than 5 M params on 2.0× data. If you want coverage out to 3× VDD, budget 10–20 M params.
2. **Tempered sampling beats uniform box widening.** The 3.0× dataset has uniform LHS sampling. A better approach: sample densely in [0, 1.2·VDD] (the plausible NR trajectory near the true rail) and sparsely from [1.2·VDD, 3·VDD] (the runaway region the rail-fix handles).
3. **Per-tech or per-VDD sampling.** TSMC5 (VDD=0.65) suffered most because training went up to 0.80 V. Consider normalising the Vd grid per-device so each tech gets the same relative coverage, rather than one absolute grid across all techs.
4. **Test the inverter, not just test-set NRMSE.** Aggregate NRMSE only differed by ~3× between v3 and prior production, but inverter transient differed by ~10 percentage points. The inverter is the harder downstream test.

The rail-restoring fix (`381bbfc`) with the prior-production checkpoints remains the shipping solution.

## Timeline (UTC+8)

- 2026-04-20 10:28–11:13 — 3.0× dataset generation (45 min total for NMOS+PMOS, 16 workers)
- 2026-04-20 13:13–04:55+1 — PMOS retrain attempts on 2.0× data (killed, CPU-starved; fixed DataLoader)
- 2026-04-21 04:55–11:36 — NMOS v3 training on Blackwell (6.5 hr, TF complete, AR OOM)
- 2026-04-21 04:55–12:00 — PMOS v3 training on A100 (6.9 hr, TF complete, AR OOM)
- 2026-04-21 12:05–12:45 — Inverter verify on all 4 TSMC techs (v3 rolled back based on results)
- 2026-04-21 13:00+ — v3 checkpoints deleted, trainer + diag commits pushed to origin
