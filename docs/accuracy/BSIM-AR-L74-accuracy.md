# BSIM-AR Transformer (LEVEL=74) — family report

**What it is:** a decode-only **autoregressive Transformer** compact model that
emits the 13 outputs as a token sequence, netlist `LEVEL=74`. The validated
**higher-fidelity** option — not production, because AR inference is ~40× a
DirectNet evaluation on CPU.

**Status:** `corroft@medium` (1.9 M params) is **16/16 strict** across
OMP ∈ {1,2,4} with zero flips, device DC 44/44.

Cross-cutting numbers live in the axis files — [`by-tech.md`](by-tech.md),
[`by-scale.md`](by-scale.md), [`by-recipe.md`](by-recipe.md) — and the gate
definitions in [`methodology.md`](methodology.md). This file carries only what is
specific to the AR transformer.

**Covers:** V6.8.0 (un-parking + scale × recipe campaign) → V6.8.1 (xl fill) →
V6.11.0 (TSMC6 sweep) → V6.13.0 (post-gds-fix re-gate) → V7.1.0 (device/AC
re-gate).

---

## 1. Architecture and cost

Per-tech checkpoints `tsmc{5,6,7,12,16}_tf_{small,medium,large,xl}_{nmos,pmos}`
(+ recipe variants; `tsmc6` is the TSMC7 repeat, `by-tech.md` §5), sharing DirectNet's data / normalization / loss / training /
eval pipeline through the `bsimar` package. Tech identity enters as an
`nn.Embedding` code with a **local per-scope vocab** (Rule 16).

| tier | shape | params |
|---|---|---|
| small | — | 0.67 M |
| **medium** | — | **1.94 M** |
| large | — | 5.02 M |
| xl | 384 × 8L × ff1536 | 14.81 M |

**Inference is an 8-step sequential AR loop** — each step a full encoder forward
— at 5.5× DirectNet's parameter count. On the CPU-pinned gates (1 thread) that
is **61.5 ms/eval at medium vs DirectNet-large's 1.5 ms**; the heavy transient
cells (ring / SRAM / switchcap) run ~1 h each at `small` and 2–4 h at `large`.

Two mitigations exist:

* **Batched eval** (V6.8.0) — the solver groups a whole netlist's devices into
  one per-checkpoint forward + Jacobian pre-warm, so a 10-device ring collapses
  to 2 grouped forwards. This is the single biggest gate-time lever.
  `NN_BATCHED_EVAL=0` opts out.
* **AR prefix cache** (V7.0.4, `PYCIRCUITSIM_NN_AR_CACHE=1`, **default off**) —
  the AR loop re-encoded the whole growing prefix once per step (60 token-passes
  for 11 hidden states); the cache keeps per-layer K/V and encodes each token
  once. 1.60× DC / 1.56× tran / 1.21× AC, 4.3× on `no_grad` batch-2048 eval,
  deviation ≤1.6 µV on solved nodes. It is exact in real arithmetic but **not**
  in float32 — `F.linear` is not row-stable on CPU (0/96 shapes), so no
  incremental form can ever be bit-identical. It stays default-off pending a
  16-gate `MODEL=transformer` re-gate; `tests/verify_ar_cache.py` (10 checks)
  guards it meanwhile.

**For production use the Transformer is a fidelity option, not a speed option.**

## 2. What un-parking cost (V6.8.0)

The Transformer already shared the data pipeline but was not a first-class
recipe citizen. Nine changes, all default-off / behaviour-preserving for
existing paths:

1. **`unknown_code_id` parameter** — was hardcoded to the universal `17`, so a
   per-tech local vocab would CUDA-assert on the first `p_unknown` dropout. Now
   derived as `num_tech_codes - 1`.
2. **`init_from` warm-start** for `train_transformer` — the precondition for
   every curriculum recipe.
3. **Aux losses for the Transformer** (`--sobolev` / `--charge-sobolev` /
   `--subthresh`), previously guarded DirectNet-only. The math is model-agnostic;
   the fix was permuting the output-side norm stats into BSIMAR column order.
   The column-sum autograd trick was **verified** valid here — attention mixes
   token positions within a sample, never batch rows (max |colsum − perrow| grad
   diff 1.3e-7). One real blocker surfaced: fused SDPA has no double-backward, so
   aux losses force the MATH attention backend at forward time.
4. **`xl` preset** (384×8L, 14.8 M).
5. **`--amp`** (bf16 autocast) — 1.3× at large, unused in the campaign for
   comparability.
6. **Parser LEVEL=74 per-tech preempt cascade** (`tsmc{X}_tf_{size}_{dev}`,
   large-first) + `_tf_` local-vocab scope detection.
7. **`PYCIRCUITSIM_NN_FORCE_LEVEL=74`** — retargets every LEVEL=73 model card at
   parse time, so the entire complex / sweep / AC gate infrastructure runs the
   Transformer with **zero deck changes**.
8. **Driver parameterization** — `MODEL={direct,transformer,tabpfn}` across
   `recipe_train.sh`, `gate_matrix_iso.sh`, `recipe_eval.sh`,
   `benchmark_run_tests.sh`, `recipe_multirun_gate.sh`.
9. **Solver batched-eval** extended to LEVEL=74.

### The latent integration bug worth remembering

The first real checkpoint scored **389 % device-DC NRMSE through the netlist
path** despite 0.38 % on the trainer's own test set.
`_MOSFETNNBase._out_col` read the model outputs through
`norm_stats.output_columns` — the canonical order, which describes the *stats
arrays* — **before** the BSIMAR model-output layout, so `qg` was denormalized as
`id` (~5× current). Historical BSIM-AR norm files predate the `output_columns`
field, so the bug had never fired. Fixed by ranking the `output_layout ==
"bsimar"` branch first; the probe then matched the trainer path bit for bit.

*Lesson: a model that is excellent on its own test set and catastrophic through
the simulator is an* output-layout *bug until proven otherwise.*

## 3. Current state

`TSMC6 ⚠` is the repeat column — the same recipe retrained on bit-identical
TSMC7 rows, scored /4, never inside the /16.

| config | params | complex single-run | complex strict | TSMC6 ⚠ /4 | device DC |
|---|---|---|---|---|---|
| **`corroft@medium`** | **1.9 M** | **16/16** | **16/16, zero flips** | — | 44/44 |
| `corro15@xl` | 14.8 M | **16/16** | **16/16, zero flips** | — | 55/55 |
| `corro15@medium` | 1.9 M | **16/16** | not swept | — | 44/44 |
| `corroft` / `crit15m` / `crit30` @xl | 14.8 M | **16/16** each | not swept | — | 55/55 |
| `corroft` / `crit15m` / `crit30` @large | 5.0 M | 15/16 each | not swept | — | 44/44 |
| `clean` @ {small, medium, large, xl} | 0.67–14.8 M | **14/16 at every tier** | 14/16 (large), zero flips | **3/4 at every tier** | 44/44 |

Margins at `corroft@medium` are not marginal: opamp 2.52 / 6.73 / 5.32 / 5.82 %
against a 10 % gate, ring 3.33 / 2.25 / 2.13 / 2.19 % against 5 %, switchcap
1.99–4.15 % against 5 %, worst SRAM lobe 6.11 % against 10 % (tsmc5/7/12/16).

**Recommendation: `corroft@medium`.** `xl` reaches the same 16/16 at 7.7× the
parameters and ~11 days of training. Its AC is *not* the collapse the pre-fix
campaign recorded (§4), but it is no better than medium's either — so xl
corroborates the medium result rather than beating it.

## 4. Family-specific findings

**The ceiling moved opamp → ring.** For clean BSIM-AR the only open cells at any
tier are `tsmc5-ring` and `tsmc7-ring`, and they fail *deterministically* —
identical period error at OMP 1, 2 and 4 — against a 5 % gate, worsening with
capacity (tsmc7-ring 5.97 % at small → 12.55 % at xl). Rings are gds-invariant,
so this is a genuine value-surface gap, and the corridor curriculum is the lever
that closes it (`by-recipe.md` §3).

**BSIM-AR is the most reproducible family under retraining.** The TSMC6
controlled repeat (`by-tech.md` §5) retrained the clean recipe on bit-identical
rows: all **16 of BSIM-AR's verdicts reproduce**, and its opamps land within
4 pp of their TSMC7 counterparts without ever railing, where DirectNet
reproduces 11/16 and PFN 10/12 with bimodal opamps. On top of the 16/16 score,
that stability is the second argument for BSIM-AR as the fidelity option.

**BSIM-AR's device surface is excellent and capacity-insensitive** — 44/44
configs at every clean tier, 55/55 at `corroft@xl`, with a single systematic
outlier: **tsmc7-NMOS DC NRMSE grows with capacity** (3.37 → 4.07 → 4.77 %
small→large). Every BSIM-AR failure at any tier is a *circuit-level* value/bias
problem, never a device-surface one. Per-tech numbers: `by-tech.md`.

**The xl fill (V6.8.1) is a documented negative result.** 48 checkpoints, full
300/120-epoch fidelity, an ~11-day run on a heavily shared box, asking two
questions:

* *Does the Transformer basin-shuffle at xl the way DirectNet-xl did?* No —
  pre-fix it held the same 15/16 basket; post-fix every corridor recipe at xl
  simply sweeps. What looked like DirectNet-style basin-shuffling was largely
  the gds bug.
* *Does AC hold at xl?* Pre-fix the answer was **"no — AC collapses"**:
  opamp-AC 0/4 for every recipe (tsmc5/tsmc16 rail; tsmc12 good GBW 1.03 /
  PM 4.4° but magNRMSE 102 %), device AC 4/8 `corroft`, 4/8 `csob`, 2/8
  `clean`, and `tsmc7-opamp-AC` **not converging at all** — its DC OP spun
  non-convergent for ~6 h with no `recipe_eval` timeout.
  **The V7.1.0 re-gate overturns most of that.** `clean@xl` banks TSMC16 on
  the opamp-AC gate (0.97 dB), and `tsmc7-opamp-AC` now *completes* and
  returns 3.86 dB — the non-convergence was the railed operating point, i.e.
  the gds bug, not a tier property. Current per-tier numbers: `by-scale.md`
  §5. The pre-fix conclusion that "`csob`'s charge-Sobolev does not recover
  AC at xl" was measured on the same broken OP and should be re-tested before
  it is relied on.

**Selection-vs-deployment mismatch (audit C6n, open).** LEVEL=74 selects
checkpoints on **teacher-forced** validation loss while deployment is
**free-running AR**; the audit measured `gds` 33 % worse under AR than under
teacher forcing. Wave 2 adds an opt-in `--val-mode ar`, which changes nothing for
existing weights — realizing it means retraining the family, and it should run
against the post-wave-2 baseline so the effects are not confounded.

## 5. Historical context

The parked v4 **universal** BSIM-AR (2026-04) scored device NRMSE 0.27 / 0.26 %
and was 6–8× worse than DirectNet at 5.5× the parameters — dominated, hence
parked. This campaign's **per-tech** BSIM-AR closes most of that gap (small-tier
device AVG 0.38 % NRMSE) and at the circuit level matches DirectNet's best.

## 6. Reproduction

```bash
# train a recipe wave
MODEL=transformer RECIPES=corroft SIZES=medium GPUS="0 1 2" NSTREAMS=6 bash scripts/recipe_train.sh
# gate it
MODEL=transformer SIZE=medium bash scripts/gate_matrix_iso.sh
# any LEVEL=73 deck, retargeted to BSIM-AR
PYCIRCUITSIM_NN_FORCE_LEVEL=74 python main.py examples/bsimar_inverter_dc.sp
```

* **Datasets:** `external_compact_models/bsimar/data/datasets/{tech}_{dev}.npz`
  (2026-06-23, the same sets DirectNet production trained on) + `_corro_`
  corridor sets. No regeneration.
* **Checkpoints:** `tsmc{5,7,12,16}_tf_{small,medium,large,xl}_{nmos,pmos}` +
  `_{crit30,corroft,crit15m,invtrip}_large_*` + `_{corroft,corro15}_medium_*` +
  the V6.8.1 xl recipe mirror. A finished run carries `*_best.pt.complete`.
* **Gate logs:** `results/bsimar_bench/`, `results/recipe_bench/gate_iso_xl_tf/`,
  `results/recipe_bench/xl_campaign/`, `results/a3_regate/tf_*`,
  `results/v710_regate/tf/`.
* **Plans:** `docs/plans/2026-07-05-bsimar-transformer-recipe-campaign.md`,
  `docs/plans/2026-07-07-bsimar-transformer-xl-fill.md`.
* **Operational gotcha:** never reuse a `gate_iso_*` results directory across
  model families — the per-cell verdict files collide silently.
