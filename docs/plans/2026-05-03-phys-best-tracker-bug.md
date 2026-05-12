# Bugs — BSIMAR phys-best tracker rewards untrained checkpoints (TWO independent bugs)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans`.
> Each step is checkbox-tracked (`- [ ]`).

**Date:** 2026-05-03
**Branch target:** `feat/bsimar-v5-phase-a` (or successor)
**Status:** PLAN — v5c trainings killed; merged 2026-05-03 PM after a second
independent bug was discovered in `apply_id_gate`
**Severity:** HIGH — Bug B contaminates v4/v5a/v5b/v5c BSIMAR shipped
checkpoints; Bug A additionally corrupts every BSIMAR v5b/v5c run that
used the structural id-gate (`--no-id-gate` not set).

---

## 0. One-paragraph summary

Two independent bugs both manifest as "BSIMAR `*_best.phys.pt` is the
wrong checkpoint", and they compound on v5b/v5c.

* **Bug A — `apply_id_gate` index mismatch (v5b/v5c only).** The
  structural Vds gate added in v5 Phase B (B3) reads id's
  normalisation stats by the *model-output* column index. For DirectNet
  that index is 0 (matches `OUTPUT_COLUMN_ORDER` where id is at 0). For
  the BSIMAR Transformer the trainer passes index 4 (id's slot in
  `BSIMAR_COLUMN_ORDER`), but `normalizer.stats` always live in
  `OUTPUT_COLUMN_ORDER` where index 4 is **qg**. The gate therefore
  denormalises id using qg's asinh_scale (~1e-16 vs id's ~5.5e-5), a
  ~10⁹× magnitude error. PMOS Transformer training was hit hardest:
  `*_best.phys.pt` froze early while `*_best.pt` and `*_best.ar.pt`
  kept improving.
* **Bug B — phys-score is mean-over-outputs and explodes on AR-rollout
  id drift (v4 onwards).** `phys_score = mean(NRMSE_phys) + 0.1 ·
  (1 − mean(R²_phys))` is computed in AR mode. The id column's wide
  dynamic range plus the asinh+zscore inverse `s · sinh(σ · id_norm +
  μ)` lets a single AR-rollout overflow row drive id-NRMSE to
  millions of percent, dominating the mean and rewarding near-untrained
  weights.

Both bugs are real and both need fixing. Bug A explains why v5c PMOS's
phys-best plateaued at epoch ~10 even though Bug B alone would *also*
cause that plateau. The fixes are independent and both small. The
post-fix story:

* **v4/v5a** — only Bug B applies (no id-gate). Rename `.phys.pt`
  aside, let the simulator load `_best.pt`. No retraining required.
* **v5b** — both bugs apply. v5b checkpoints discarded; either retrain
  or roll back.
* **v5c** — both bugs apply. Retrain with both fixes after C1.

---

## 1. Evidence

### 1A. Bug A (id_gate index mismatch) — `v5c_universal_pmos_norm.npz`

Dumped 2026-05-03 PM directly from the on-disk checkpoint:

```
OUTPUT_COLUMN_ORDER (storage order in normalizer.stats):
  [0] id : scale=5.499e-05  mean=+1.213e+00  std=+2.076e+00   ← real id stats
  [1] gm : scale=1.002e-04
  [2] gds: scale=5.893e-06
  [3] gmb: scale=1.049e-06
  [4] qg : scale=1.128e-16  mean=-1.030e+00  std=+1.287e+00   ← read by gate (WRONG)
  [5] qd : scale=4.725e-17
  ...

BSIMAR_COLUMN_ORDER (model-output order):
  [0] qg  [1] qb  [2] qd  [3] qs  [4] id  [5] gm  ...
```

Trainer passes `_TF_ID_IDX = 4` (`bsimar/training/trainer.py:44`) as
`id_idx_in_output` to `apply_id_gate`. The gate then does (`id_gate.py:113-131`):

```python
out_mean_id = float(normalizer.stats.output_mean[id_idx_in_output])  # = qg's mean
out_std_id  = float(normalizer.stats.output_std[id_idx_in_output])   # = qg's std
s_id        = float(normalizer.stats.asinh_scale[id_idx_in_output])  # = qg's scale
```

A ~10⁹× magnitude error. PMOS data confirms id has the opposite
sign of Vds in the conducting regime (Vds ∈ [−1.6, 0]V, ~90% of
samples have id > 0), so the gate's `tanh(Vds/0.04)` factor
(approaches −1) compounds with the wrong-stats reconstruction.

Symptom in checkpoint timestamps (`v5c_universal_pmos_*`):

| File              | Timestamp           | Stage                         |
|-------------------|---------------------|-------------------------------|
| `_best.phys.pt`   | May 2 14:04         | Froze ~12 h into training     |
| `_best.ar.pt`     | May 3 01:55         | AR-best kept advancing        |
| `_best.pt`        | May 3 01:57         | TF-best kept advancing        |

The TF MAE in normalised space kept improving on the 12 well-formed
columns while the gate-corrupted id slot was structurally driven into
junk, so the phys score (which depends on the id column) plateaued.
DirectNet was unaffected (`_DN_ID_IDX = 0` matches both orderings by
coincidence).

Unit tests in `tests/test_nn_id_gate.py` did not catch this because
the synthetic stats use arbitrary values at every index — the
test-stats `[0]` and `[4]` are uncorrelated with real id/qg semantics,
so swapping them produces a numerically valid (just wrong) result.

### 1B. Bug B (mean-aggregated phys-score) — diagnostic captured 2026-05-03 12:30

Diagnostic captured against `v5c_universal_pmos` checkpoints:

| Output | `_best.phys.pt` (epoch ~10) | `_best.pt` (TF-best, ~epoch 500) |
|--------|-----------------------------|----------------------------------|
| id     | NRMSE=**667 422 %**, R²=−4.5×10¹⁰ | NRMSE=**13 660 563 %**, R²=−1.9×10¹³ |
| gm     | 0.56 %, R²=0.998 | 0.33 %, R²=0.999 |
| gds    | 0.58 %, R²=0.990 | 0.41 %, R²=0.995 |
| gmb    | 0.28 %, R²=0.999 | 0.08 %, R²=1.000 |
| qg     | 0.20 %, R²=0.999 | 0.05 %, R²=1.000 |
| qd     | 0.20 %, R²=0.999 | 0.05 %, R²=1.000 |
| qs     | 0.18 %, R²=1.000 | 0.04 %, R²=1.000 |
| qb     | 0.33 %, R²=0.995 | 0.34 %, R²=0.995 |
| cgg/cgd/cgs/cdg/cdd | ≤0.27 % | ≤0.07 % |
| **phys_score** | **3.46 × 10⁸** | **1.45 × 10¹¹** |
| **TF val MAE (norm)** | 0.01194 | **0.00362** |

`val MAE` improved 3× during training (genuine convergence in
normalised space), but `phys_score` got 420× **worse** because the id
NRMSE blew up under AR rollout. Note the late-checkpoint id NRMSE is
**still catastrophic** (13 M %) — that is Bug A's footprint: even the
TF-best id slot is contaminated. The 12 non-id columns improve as
expected.

Diagnostic script (kept for re-runs): `tests/diag_phys_best_explosion.py` (TODO — see step P1).

---

## 2. Root cause

### 2A. Bug A — `apply_id_gate` index reuse

File: `external_compact_models/bsimar/models/id_gate.py:113-131`.

The function takes a single `id_idx_in_output` argument and uses it
*twice*: once to slice the model output (`out_norm[:, id_idx_in_output]`)
and once to look up id's normalisation stats from `normalizer.stats`.
This is correct only when both indices coincide. For DirectNet they
do (id is at 0 in both orderings, since DirectNet's output is in
`OUTPUT_COLUMN_ORDER`). For the BSIMAR Transformer they do not — the
output is in `BSIMAR_COLUMN_ORDER` (id at 4), but `normalizer.stats`
were fit on `OUTPUT_COLUMN_ORDER`-ordered targets and never
re-permuted. The trainer applies `reorder_outputs` to the *targets*
(`trainer.py:714-719`) but does not re-permute the stats.

Mathematical effect for the Transformer's asinh path:

```
intended:  id_raw_phys = s_id · sinh(σ_id · id_raw_norm + μ_id)         # s≈5.5e-5
actual:    id_raw_phys = s_qg · sinh(σ_qg · id_raw_norm + μ_qg)         # s≈1.1e-16
```

with `σ_qg ≈ 1.29`, `μ_qg ≈ −1.03`, `s_qg ≈ 1.13e-16`. For typical
post-asinh-zscore range `id_raw_norm ∈ [−2, 2]`, the actual
reconstruction lands in `[1e-16 · sinh(−3.6), 1e-16 · sinh(1.55)] ≈
[−1.8e-15, 2.4e-16]` — i.e. a charge-like magnitude. The gate then
multiplies by `tanh(Vds/0.04) ∈ [−1, 1]`, asinh-re-encodes through
`asinh(value/s_qg)`, and the loss compares this junk to the true id
target. The model adapts by emitting weird id_raw_norm values, but
the AR-conditioning chain (rule 21: AR sees `id_raw`) propagates the
junk into downstream tokens and stalls phys-best.

Why PMOS is hit harder than NMOS: PMOS qg's mean is signed
(`μ_qg ≈ −1.03`) while NMOS qg's mean is comparable in magnitude but
opposite sign. PMOS's id distribution (mostly positive, since
Vds<0 and the convention here makes id>0 in conduction) interacts
adversely with the `tanh(Vds/0.04) ≈ −1` factor: the gate forces the
model to learn an opposite-sign id_raw_norm AND that target lives in
a regime where `sinh(σ_qg · …)` is already saturating.

### 2B. Bug B — mean-aggregated phys-score under AR-rollout

#### 2B.1 The mathematical core

The asinh+zscore normaliser stores per-target `s_id`, `μ`, `σ` and
inverts via:

```
id_phys = s_id · sinh(σ · id_norm + μ)
```

`sinh` grows exponentially. A small drift `Δ` in normalised space
amplifies as

```
Δ_phys ≈ s_id · cosh(σ · id_norm + μ) · σ · Δ_norm
       ≈ |id_phys| · σ · Δ_norm                 (for |id_phys/s| ≫ 1)
```

For PMOS where `|id_phys|` reaches ~1 mA in the saturation plateau,
`Δ_norm = 0.01` (well within trained accuracy) becomes
`Δ_phys = 0.02 mA = 2 × 10⁻⁵ A`. That is benign. **The pathology
appears when the AR rollout occasionally emits an extreme `id_norm`**:
if the autoregressive chain produces `id_norm = 50` (well outside the
training distribution), then

```
id_phys = s_id · sinh(σ · 50 + μ)
```

`sinh(100) ≈ 1.3 × 10⁴³`. Multiplied by any non-zero `s_id`, the
prediction is plain numerical overflow on the order of 10³⁴ A.
NRMSE = `RMSE / data_range` divides by a finite range (~mA), so a
single overflow row drives NRMSE per output to millions of percent.

#### 2B.2 Why AR exposes it but TF does not

`test_model()` in `bsimar/training/trainer.py:233` uses **AR mode**
(no teacher forcing). The id token sits at index 4 in
`BSIMAR_COLUMN_ORDER`, conditioned on the model's own predictions
for `qg, qb, qd, qs`. Tiny errors in those four upstream tokens
compound: by the time the id head fires, the conditioning context is
out-of-distribution, and the head emits an extreme `id_norm`.

`_validate_epoch_tf()` uses TF mode with ground-truth conditioning
tokens. It never sees the AR drift — which is why the TF-val MAE
keeps improving while the phys metric explodes.

#### 2B.3 Why mean-over-outputs is the fragile part

`phys_score = mean(NRMSE) + 0.1 · (1 − mean(R²))` averages across all
13 outputs. **A single outlier output with NRMSE = 10⁷ % destroys the
average.** The 12 well-behaved outputs (sub-1 % NRMSE, R² ≥ 0.99)
contribute negligibly. The 0.1 weight on `(1 − R²)` is supposed to
discourage huge negative R², but the negative R² on id (−10¹³) makes
that term *also* dominate. So both terms point the same wrong way.

#### 2B.4 Why id and not the other outputs

The 12 non-id outputs (qg, qb, qd, qs, gm, gds, gmb, cgg, cgd, cgs,
cdg, cdd) all sit in saturation regimes where their physical
magnitudes are bounded and their asinh scales are tighter relative to
data range. The id column has the widest dynamic range
(subthreshold 10⁻¹⁵ A → saturation 10⁻³ A spans 12 decades) and the
asinh scale `s_id` sits at the geometric mean. This combination
makes id the most exposed to sinh overflow on any AR drift.

#### 2B.5 Why PMOS gets stuck and NMOS recovers

Empirically v5c TF NMOS phys-best advanced after B2 slope warmup
(`08:33 today`), PMOS did not. The B2 slope-match loss penalises
∂Id/∂Vg shape — for NMOS the AR rollout must respect a smooth
shape, so the upstream tokens stabilise. PMOS' source-relative
frame and wider current range mean the same penalty did not stabilise
the AR rollout enough to push PMOS out of the epoch-10 minimum. The
mean-over-outputs phys-score then locked PMOS in place. **Note:**
With Bug A also in play, the PMOS plateau has *two* reinforcing
causes; Bug A alone suffices to cause the plateau because it
structurally corrupts the id slot regardless of AR drift.

### 2C. How the two bugs interact

Bug A makes the trained id slot itself bad (the model learns weights
that are best-effort against a corrupted gate target). Bug B makes
the *selection* metric pick the wrong checkpoint among the (already
bad) candidates. v5b/v5c suffer both: the candidates are bad and the
ranker is broken. v4/v5a only suffer Bug B: the candidates are
intrinsically OK but the ranker still picks an early one. This split
shapes the re-train scope in §5.

---

## 3. Blast radius — what else is contaminated

### 3A. Bug A — version scope

Bug A entered the codebase with v5 Phase B B3 (`feat(v5): Phase B
B2+B3 — SlopeMatchLoss + structural Vds id-gate`,
commit `a1aa fc`-class). Affected:

* **v5b** — `external_compact_models/bsimar/checkpoints/v5b_universal_*`.
  All four BSIMAR Transformer checkpoints (NMOS+PMOS) trained with
  the gate active and therefore against a corrupted id target.
  DirectNet v5b checkpoints (`v5b_dn_universal_*`) are unaffected
  (DirectNet's index-0 lookup happens to match).
* **v5c** — same as v5b. All four BSIMAR Transformer runs corrupted.
  DirectNet v5c unaffected.

Pre-B3 versions (v4, v5a) **never used `apply_id_gate`** and are
therefore Bug-A-clean. They still suffer Bug B.

### 3B. Bug B — version scope

`mosfet_bsimar.py:158-161`:
```python
prefix = os.environ.get("BSIMAR_PREFIX", "v4_universal")
v4_phys = CHECKPOINT_DIR / f"{prefix}_{device_type}_best.phys.pt"
v4_plain = CHECKPOINT_DIR / f"{prefix}_{device_type}_best.pt"
# loads v4_phys preferentially when it exists
```

The simulator has loaded `_best.phys.pt` for every BSIMAR
verification since v4 (`feat(train+sim): sign/boundary losses + …`,
2026-04-13). Implication:

- **v4 BSIMAR baseline** (CLAUDE.md "Phase A4 control retrain… AR side
  has wins") used the bug-affected `.phys.pt` weights. Numbers in
  `results/v5_baseline_2026_04_22.md` may understate v4 BSIMAR's
  intrinsic capability.
- **v5a control retrain** (`a1ba677`) was gated against the same
  contaminated baseline. Its "FAIL" verdict on §A4 stands, but the
  per-cell deltas should be re-read with the caveat that both sides
  used `.phys.pt`.
- **v5b** (`results/v5b_sdata_gate_2026_05_02.md`) compared
  `.phys.pt` to `.phys.pt`. The B1 sampler verdict (FAIL) is robust
  *as a relative comparison* because both arms had the same Bug B,
  but **both arms also had Bug A**, so the absolute numbers are
  doubly contaminated.
- **v5c** (just killed) — same as v5b.

This is potentially the most expensive bug class in this codebase's
history — it gates the entire compact-model accuracy story and may
have caused us to chase phantom failures (e.g., the TSMC7 NMOS
"sampling-basis" thesis in v5 §17).

### 3C. DirectNet (LEVEL=73) is fully unaffected

* DirectNet is single-shot (no AR rollout) → no Bug B exposure (no
  phys-best tracker; the trainer uses TF-only val loss).
* DirectNet's `_DN_ID_IDX = 0` matches both `OUTPUT_COLUMN_ORDER` and
  the model output order → no Bug A exposure.

DirectNet v4/v5a/v5b/v5c checkpoints can be trusted; only the
simulator-side "prefer phys.pt" loader logic is a no-op for DirectNet.

### 3D. Side issue (separate, smaller) — DirectNet `norm_mode` default

`train_directnet` (`bsimar/training/trainer.py:482-490`) calls
`load_and_split_bsimar` without `norm_mode="zscore"`, so DirectNet
trains under `asinh+zscore` outputs even though CLAUDE.md rule 6
specifies plain zscore for DirectNet. Not a cause of the phys-best
plateau, but a separate drift to fix while we're in the file. Adds a
single keyword argument; no retraining strictly required (existing
checkpoints are still consistent — they use asinh stats end-to-end —
but the design intent and the inference-time chain rule comment in
`mosfet_directnet.py` need to align).

---

## 4. Acceptance criteria

| # | Criterion | Bug | How verified |
|---|-----------|-----|--------------|
| 1 | `apply_id_gate` reads id stats from `OUTPUT_COLUMN_ORDER` index 0 regardless of where id sits in the model output. Unit test exercises both DirectNet (idx 0/0) and BSIMAR (idx 4 model / 0 stats) layouts with **distinct** synthetic stats at indices 0 and 4 to prove the lookup is decoupled. | A | `tests/test_nn_id_gate.py` extended; `pytest tests/test_nn_id_gate.py -v` passes |
| 2 | The phys-best score rewards the **late-trained** v5c PMOS weights, not epoch-10 | B | Re-run the diagnostic in §1B; new score on `_best.pt` < new score on `_best.phys.pt` (TODO P2) |
| 3 | A 5-epoch BSIMAR retrain on the existing data produces a `_best.phys.pt` whose val MAE is monotonically below the eventual `_best.pt`'s val MAE | A+B | Smoke test (TODO C0) |
| 4 | All four v5c checkpoints retrain under both fixes, and `_best.phys.pt` agrees with `_best.pt` on which run is "best" within 1 % NRMSE on the validation slice | A+B | Full v5c retrain (TODO C1) |
| 5 | Inverter verifier on the **fixed** v5c checkpoints meets §2 of `docs/plans/2026-04-24-v5-inverter-accuracy.md` | A+B | TODO C2 |
| 6 | A re-baseline of v4 against the **fixed** simulator-side loader (force `_best.pt`) produces a TSMC5/7/12/16 table that we can audit against the existing `v5_baseline_2026_04_22.md` to scope how much of the v5b "B1 failure" was actually the bug class | B | TODO D1 |
| 7 | DirectNet v5c retrain with `norm_mode="zscore"` matches or beats the asinh-defaulted v5b DirectNet on per-tech NRMSE on the same data | side | TODO C3 |

---

## 5. Plan — Phase P, Phase B (small-fix), Phase C (re-train), Phase D (re-baseline)

### Phase P — Pin down both bugs (no production change)

- [ ] **P1.** Add the `apply_id_gate` index-mismatch reproducer
      `tests/diag_id_gate_index_mismatch.py`. Loads `v5c_universal_pmos_norm.npz`,
      builds a synthetic single-row input with Vds = −0.6 V, runs
      `apply_id_gate(id_idx_in_output=4)` against both the buggy and
      fixed code paths, prints `id_gated_phys` for each. Asserts the
      buggy path produces a value with a charge-scale magnitude (~1e-16)
      and the fixed path produces a value with a current-scale magnitude
      (~1e-5).
- [ ] **P2.** Move the diagnostic from `/tmp/probe_phys_best.py`
      (used during discovery) into `tests/diag_phys_best_explosion.py`
      so Bug B is reproducible. Take `--prefix` as a CLI flag so it
      runs against any checkpoint set.
- [ ] **P3.** Add a unit test
      `tests/test_phys_score_robustness.py` that asserts the
      median-based phys-score correctly ranks `_best.pt` above
      `_best.phys.pt` for v5c PMOS (and equivalently for the
      synthetic single-output-blowup case).
- [ ] **P4.** Audit `compute_physical_metrics` (`bsimar/eval/metrics.py`)
      to confirm the **per-output** NRMSE/R² values are correctly
      computed even when one output overflows. The mask
      `|y_t| > 0.1 % of peak` is per-output, so an overflow in id
      doesn't break gm/gds/etc — confirmed by §1B evidence. No
      change needed here.

### Phase B — Smallest fixes that hold the gate

#### B-A. `apply_id_gate` — decouple the model-output index from the stats index

File: `external_compact_models/bsimar/models/id_gate.py`.

- [ ] **B-A1.** Add a new keyword arg `id_idx_in_stats: int = 0` to
      `apply_id_gate`. Default 0 because `OUTPUT_COLUMN_ORDER` puts
      id at 0 regardless of model layout.
- [ ] **B-A2.** Replace every read of `normalizer.stats.<field>[id_idx_in_output]`
      with `[id_idx_in_stats]`. Concretely the four reads on
      `id_gate.py:113, 114, 131` and the asinh chain at `id_gate.py:131-138`.
      The four `out_norm[:, id_idx_in_output]` slicing calls
      (`id_gate.py:123, 146-160`) keep using `id_idx_in_output` because
      that's about column position in the *model output*, not stats.
- [ ] **B-A3.** Update the docstring (`id_gate.py:74-94`) to call out
      that `id_idx_in_stats` is **independent of layout** and
      defaults to 0 because `BSIMARNormStats` always lives in
      `OUTPUT_COLUMN_ORDER`.
- [ ] **B-A4.** Trainer (`bsimar/training/trainer.py`): all six
      Transformer call sites that pass `id_idx_in_output=_TF_ID_IDX`
      need an explicit `id_idx_in_stats=0` (or equivalently, rely on
      the new default). DirectNet call sites (passing `_DN_ID_IDX=0`)
      need no change. Touch lines 106, 149, 179, 249, 291, 319, 364, 400.
- [ ] **B-A5.** Extend `tests/test_nn_id_gate.py`: in the existing
      asinh+BSIMAR test, set `_asinh_stats` so that index 0 has
      `asinh_scale = 5.5e-5` (id-like) and index 4 has
      `asinh_scale = 1.1e-16` (qg-like). Assert that the gated id
      magnitude is in the current scale, not the charge scale. Also
      add a regression test with the **buggy** call form
      (`id_idx_in_stats=id_idx_in_output=4`) confirming it would
      produce charge-scale output — guards against silent re-introduction.

#### B-B. Trainer-side phys-score fix (median over outputs)

File: `external_compact_models/bsimar/training/trainer.py`.

- [ ] **B-B1.** Replace `np.nanmean` with `np.nanmedian` in the
      phys-score block (the two occurrences inside the main loop and
      the AR-finetune loop, ~lines 887-901 and 996-1010 in the
      current trainer):
      ```python
      nrmse_med = float(np.nanmedian(nrmse_arr))
      r2_med    = float(np.nanmedian(r2_arr))
      phys_score = float("inf") if (np.isnan(nrmse_med) or np.isnan(r2_med)) \
                                 else nrmse_med + 0.1 * (1.0 - r2_med)
      ```
      Median is robust to a single-column blowup: 12 well-behaved
      outputs dominate the score regardless of how badly id explodes.
      The 0.1 weight on `(1 − R²)` stays sensible because median R²
      lives in `[−1, 1]` for any reasonable model.
- [ ] **B-B1-alt.** *(fallback if C0 smoke check inverts)* Keep mean
      but **clip per-output NRMSE to 100 %** before averaging:
      ```python
      nrmse_clipped = np.clip(nrmse_arr, 0.0, 100.0)
      r2_clipped    = np.clip(r2_arr, -1.0, 1.0)
      ```
      Slightly more conservative; loses information when outputs
      naturally have NRMSE > 100 % under asinh. Document the choice
      inline.

Recommend **B-B1 (median)**.

#### B-B2. Simulator-side loader fix

File: `pycircuitsim/models/mosfet_bsimar.py:158-180`.

After B-B1, `_best.phys.pt` is reliable for **future** trainings. The
**legacy v4/v5a/v5b checkpoints already on disk** were saved with
the buggy tracker (Bug B) and — for v5b — an additionally
gate-corrupted id slot (Bug A). We need to keep both paths usable.

- [ ] **B-B2.** Add a normaliser flag `phys_best_metric: str` to
      `BSIMARNormStats` (default `"legacy_mean"`, set to `"median"`
      by the fixed trainer). Loader logic:
      ```python
      stats = BSIMARNormStats.load(norm_path)
      if stats.phys_best_metric == "median" and v_phys.exists():
          load(v_phys)
      else:  # legacy or missing — prefer plain best.pt for safety
          load(v_plain if v_plain.exists() else v_phys)
      ```
      Lets the simulator run safely on v5c-onwards with the
      median-tracked phys-best, and falls back to plain best.pt for
      every legacy checkpoint where the bug applied.

- [ ] **B-B3.** Manually rename
      `external_compact_models/bsimar/checkpoints/v{4,5a,5b}_universal_*best.phys.pt`
      to `*best.phys.bug.pt` so the loader's existence check picks
      the plain `_best.pt`. Pure operational hygiene; doesn't
      affect the loader logic but makes it impossible to silently
      regress to the bug if the loader rule is later softened.

#### B-side. DirectNet `norm_mode="zscore"` default fix

File: `external_compact_models/bsimar/training/trainer.py:482-490`.

- [ ] **B-side1.** In `train_directnet`, pass
      `norm_mode="zscore"` explicitly to `load_and_split_bsimar`.
      Existing v4/v5a/v5b/v5c DirectNet checkpoints stay valid (they
      are end-to-end consistent under asinh), but new runs will
      adhere to CLAUDE.md rule 6.
- [ ] **B-side2.** Update CLAUDE.md rule 6 to clarify that the
      Transformer uses asinh+zscore on outputs and DirectNet uses
      plain zscore — already documented but the trainer drifted.

### Phase C — Re-train under both fixes

- [ ] **C0.** 5-epoch smoke test on TSMC4 PMOS (or any cheap subset)
      after B-A and B-B1 land. Assert: (i) gate's id_phys lands in
      current scale (sanity vs. P1); (ii) `_best.phys.pt` val-MAE in
      norm space is monotone non-increasing over the run; (iii) at
      run end, `_best.phys.pt` and `_best.pt` agree on the same
      epoch (or differ by ≤ 5 %).
- [ ] **C1.** Apply B-A1..B-A5 + B-B1 + B-B2 + B-B3 + B-side1. Re-run
      the four v5c trainings (DirectNet NMOS/PMOS + Transformer
      NMOS/PMOS) with the same hyperparams and the existing
      B1-regenerated dataset
      (`external_compact_models/bsimar/data/datasets/`). Use
      `conda run --no-capture-output -n pycircuitsim …` so the logs
      flush in real time (this session lost ~24 h of log content to
      conda's stdout capture; see §11). Each run launches with
      `CUDA_VISIBLE_DEVICES=N` pinning the model to a specific GPU;
      remember **PyTorch's default CUDA_DEVICE_ORDER=FASTEST_FIRST**
      maps `0 → Blackwell`, `1/2/3 → A100s` on this host
      (verified empirically 2026-05-02).
- [ ] **C2.** When all four v5c retrains finish, run the inverter
      verifier on TSMC5/7/12/16 (`SKIP_VTC=1` for tsmc7) per
      `docs/plans/2026-04-24-v5-inverter-accuracy.md` §2,
      using `--checkpoint-prefix v5c_universal --directnet-prefix
      v5c_dn_universal`. Compare to v4 baseline and v5b in
      `results/v5c_b2b3_gate_report.md` (NEW).
- [ ] **C3.** Smoke check: assert `_best.phys.pt`'s normalised val
      MAE is within 5 % of `_best.pt`'s normalised val MAE for each
      of the four runs. If not, B-B1's median-based score is still
      picking the wrong checkpoint and we fall back to B-B1-alt
      (clipped mean).
- [ ] **C4.** *(optional)* DirectNet rerun under
      `norm_mode="zscore"` to validate B-side. Compare per-tech
      NRMSE to v5b DirectNet. Skip if C2 has already pushed
      verification numbers below the gate.

### Phase D — Re-baseline v4 (read-only sanity)

- [ ] **D1.** Re-run the verifier against the **legacy v4** BSIMAR
      checkpoints with the fixed simulator-loader (which now falls
      back to `_best.pt` because the legacy norm.npz lacks the
      `phys_best_metric` flag). Capture
      `results/v4_rebaseline_post_phys_fix.md`. Note: v4 has only
      Bug B, not Bug A — its `_best.pt` is intrinsically a usable
      checkpoint.
- [ ] **D2.** Compare the new v4 numbers to
      `results/v5_baseline_2026_04_22.md`. If the per-cell NRMSE
      changes by > 1 pp on any cell, the v5b S-DATA failure verdict
      has to be re-evaluated:
      - If v4 (rebased) ≈ v5b on TSMC7 NMOS DC → B1 sampler still
        didn't help → keep current B2/B3 hypothesis.
      - If v4 (rebased) ≪ v5b on TSMC7 NMOS DC → at least one of the
        two bugs was masking v5b's improvement; B1 might already
        have closed F1 and we wasted GPU days on B2/B3.

This re-baseline is **mandatory** before signing off the v5 plan,
because it determines whether the v5b "B1 failed" verdict survives.

---

## 6. Risks and mitigations

| Risk | Likelihood | Blast radius | Mitigation |
|------|:----------:|--------------|------------|
| The `id_idx_in_stats` default of 0 silently masks future per-device permutations of `OUTPUT_COLUMN_ORDER` | Low | Reintroduces a class-A bug | B-A5 regression test on the *buggy* call form; CLAUDE.md rule 21 amended to call out the stats-vs-output-index distinction |
| Median-based score still rewards bad checkpoints in some pathology we haven't seen | Low | One retrain | C3 smoke check — if the ranking inverts, fall back to B-B1-alt (clipped mean) |
| Legacy v4/v5a/v5b verifications can't be reproduced because the simulator now loads `_best.pt` instead of `_best.phys.pt` | High by design | Apparent "regression" that's actually a correction | Document the change in CLAUDE.md rule #16; D1 captures the corrected v4 baseline as the new reference |
| The id-column AR-rollout instability is the **real** Bug B and median just hides it | Medium | We ship structurally fragile models that work on validation but blow up on out-of-distribution simulator queries | Add a runtime check in `mosfet_bsimar.py` that clamps `id_norm` to ±10 zscore units before sinh; warn if clamped (hint of AR drift). Track clamp rate during inverter-VTC verification |
| Re-baseline (Phase D) shows v4 was actually fine and v5b's "FAIL" was a bug-induced phantom | Medium | Wasted v5c retrain + B2/B3 implementation effort, but **B2/B3 still ship** because they're independently justified | Phase D is read-only; outcome shapes the next sprint, doesn't break Phase C |
| Bug A's fix changes the gate's loss landscape enough that v5c retrains require hyperparameter retuning | Low-Medium | One re-tuned retrain | C0 smoke test catches divergence before committing to the full C1 |

---

## 7. Open questions — RESOLVED

1. **Which Bug B fix?** B-B1 (median). One line, robust, preserves
   physical-space intent.
2. **Touch legacy checkpoints?** Yes — rename `_best.phys.pt`
   aside (B-B3) so existing simulator paths fall back to the plain
   `_best.pt`. No retraining of v4/v5a required because v4/v5a
   `_best.pt` are intrinsically clean (only Bug B applied).
3. **Re-baseline v4?** Yes (D1) — it's the only way to know whether
   v5b's negative result was real.
4. **Block v5c retrain on D1 outcome?** No — C and D run in
   parallel.
5. **Are v5b checkpoints recoverable without retraining?** No —
   Bug A means v5b TF Transformer weights were fitted against a
   gate that was reconstructing id from qg's stats. Even loading
   `_best.pt` will give a corrupted id slot. v5b is discard-only
   for the Transformer; v5b DirectNet is fine.

---

## 8. File-by-file change manifest

| File | Phase | Action | Why |
|------|-------|--------|-----|
| `external_compact_models/bsimar/models/id_gate.py` | B-A | Add `id_idx_in_stats: int = 0` kwarg; replace stats-lookup index with this kwarg; update docstring | Decouple model-output index from stats index |
| `external_compact_models/bsimar/training/trainer.py` | B-A | All six TF Transformer `apply_id_gate` calls add `id_idx_in_stats=0` (explicit) | Make the new contract obvious at every site |
| `external_compact_models/bsimar/training/trainer.py` | B-B1 | Replace `np.nanmean` with `np.nanmedian` in the phys-score block (both main loop and AR-finetune loop) | Mean dominated by id-column blowup |
| `external_compact_models/bsimar/training/trainer.py` | B-side | Pass `norm_mode="zscore"` to `load_and_split_bsimar` in `train_directnet` | CLAUDE.md rule 6 alignment |
| `external_compact_models/bsimar/data/normalize.py` | B-B2 | Add `phys_best_metric: str = "legacy_mean"` field to `BSIMARNormStats`; preserved in `save()`/`load()` with backward-compat default | Loader needs a flag to know which checkpoint is trustworthy |
| `external_compact_models/bsimar/training/trainer.py` | B-B2 | When trainer constructs the normaliser's stats, set `phys_best_metric="median"` after the fix | Forward-flag the new behaviour |
| `pycircuitsim/models/mosfet_bsimar.py` | B-B2 | Loader logic: `_best.phys.pt` only when `phys_best_metric == "median"`, else fall back to `_best.pt` | Don't trust legacy phys-best |
| `pycircuitsim/models/mosfet_directnet.py` | (none) | DirectNet has no AR rollout, no phys-best tracker, and `_DN_ID_IDX=0` matches the stats order, so no fix needed | DirectNet not affected by either bug |
| `external_compact_models/bsimar/checkpoints/v{4,5a,5b}_universal_*best.phys.pt` | B-B3 | Rename to `*best.phys.bug.pt` | Belt-and-suspenders against silent regression |
| `tests/test_nn_id_gate.py` | B-A5 | Extend asinh+BSIMAR test with id-vs-qg-shaped distinct stats; assert gated id is in current scale; add regression guard exercising the buggy `id_idx_in_stats=4` form | Catch class-A regressions |
| `tests/diag_id_gate_index_mismatch.py` | P1 | NEW — reproducer loading `v5c_universal_pmos_norm.npz`, comparing buggy vs fixed gate output | Reproducibility for Bug A |
| `tests/diag_phys_best_explosion.py` | P2 | NEW — diagnostic that loads any prefix's `_best.pt` + `_best.phys.pt`, computes per-output NRMSE/R² in AR-rollout mode, prints the comparison table from §1B | Reproducibility for Bug B |
| `tests/test_phys_score_robustness.py` | P3 | NEW — pytest unit test ensuring median-based score correctly ranks v5c PMOS late > epoch-10 | Regression guard for Bug B |
| `CLAUDE.md` | release | Update rule #16 (checkpoint files) — note that `_best.phys.pt` is only trustworthy under `phys_best_metric == "median"`. Update rule #21 (B3 structural id-gate) — call out the `id_idx_in_stats=0` invariant | Documentation |
| `results/v5c_b2b3_gate_report.md` | C2 | NEW — gate report for the fixed v5c retrain | Verification record |
| `results/v4_rebaseline_post_phys_fix.md` | D1 | NEW — re-baseline of v4 BSIMAR with the corrected loader | Re-anchor v5b's negative verdict |

---

## 9. What this plan deliberately does NOT propose

- **No re-training of v4/v5a.** Their Transformer training never
  touched `apply_id_gate` (Bug A absent). Just rename their
  `.phys.pt` aside (B-B3) and let the simulator use `_best.pt`,
  which is the TF-best checkpoint and intrinsically usable. D1 verifies.
- **No retraining of v5b/v5c DirectNet** — DirectNet is unaffected by
  both bugs. The B-side change is forward-only.
- **No model architecture change.** The id AR-rollout instability is
  real but addressing it (e.g., clamping id_norm during AR) is a
  separate sprint. The Bug-A fix removes the structural id corruption;
  the Bug-B fix removes the metric-side amplifier. That is sufficient
  for the immediate gating problem.
- **No replacement of `compute_physical_metrics`.** It is correct
  per-output. Only the aggregation logic in the trainer needs
  changing.
- **No removal of the `_best.phys.pt` checkpoint** in the long run.
  Once B-B1 ships, phys-best is meaningful and should keep being
  saved alongside `_best.pt` and `_best.ar.pt`.
- **No migration of `BSIMARNormStats` to `BSIMAR_COLUMN_ORDER`.**
  Keeping stats in `OUTPUT_COLUMN_ORDER` is the more stable choice —
  it's the canonical column order shared by DirectNet, the dataset
  on disk, the analysis scripts, and the simulator's autograd-
  derivative indexing. The Transformer's `BSIMAR_COLUMN_ORDER` is a
  model-internal AR convenience; reordering stats to match it would
  ripple through every consumer. The B-A1 kwarg is the right
  abstraction boundary.

---

## 10. Definition of Done

- All §4 acceptance criteria satisfied.
- `apply_id_gate` unit tests assert independent stats lookup; pytest passes.
- v5c retrains complete; `results/v5c_b2b3_gate_report.md` exists.
- v4 re-baseline complete; `results/v4_rebaseline_post_phys_fix.md`
  exists; v5b "FAIL" verdict either confirmed or reversed in the
  re-baseline.
- `CLAUDE.md` rules #16 and #21 updated.
- This file's status header changes from "PLAN — v5c trainings killed"
  to "DONE — fixed in commit `<sha>`" with both bug fixes referenced.

---

## 11. Operational note: conda-run stdout buffering

While monitoring the v5c training over ~23 h, every training log
file under `results/v5c_train_logs/` stayed at 0 bytes. Root cause:
`conda run -n env python …` captures stdout/stderr of the wrapped
python process and only flushes on conda-run exit. The shell `>` redirect
catches conda-run's output, not python's. When the processes were
killed, conda-run's buffer was discarded — so we never recovered the
training logs for forensic inspection.

For future training launches use:

```bash
conda run --no-capture-output -n pycircuitsim python -u -m bsimar.cli.train ...
```

The `--no-capture-output` flag was the missing piece. This is
unrelated to either phys-best bug but caused real diagnostic pain
during this session and should be the new default in any
training-launch helper.

---

## 12. Discovery trail (for future archaeologists)

* **2026-05-03 AM** — Bug B (mean phys-score blowup) discovered while
  monitoring v5c trainings; v5c killed. Original plan drafted.
* **2026-05-03 PM** — User asks Claude to audit the trainer for
  PMOS-specific bugs. Audit traces the PMOS phys-best plateau to
  `apply_id_gate` reading qg's asinh_scale (~1e-16) instead of id's
  (~5.5e-5). Bug A confirmed independently of Bug B by dumping
  `v5c_universal_pmos_norm.npz`. Plan merged — this file.
* **Lesson** — `_TF_ID_IDX = 4` was the critical clue. Any future
  refactor that introduces a model output ordering distinct from the
  storage ordering must add a separate stats-side index (or
  re-permute the stats at load time). The `BSIMARNormStats` schema
  should grow a `column_order` field documenting which order its
  arrays live in, so future consumers can assert at load time.
