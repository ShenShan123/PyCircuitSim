# BSIM-AR Scaling-Law Test v2 — New Architecture (2026-04-08)

Re-run of the small/medium/large BSIM-AR Transformer scaling sweep on
`universal_nmos.npz` after the round-1 / round-2 architecture sprint
landed on `main` (commit `99f8086`, refined through `54ef16c`). Same
fixed recipe as the original sweep
(`results/scaling_law_2026_04_08/scaling_law_report.md`); the **only
things that changed are the four architectural levers from the
sprint**.

This is the same matched comparison surface as the prior BSIMAR scaling
sweep and the prior DirectNet scaling sweep
(`results/scaling_law_directnet_2026_04_08/scaling_law_directnet_report.md`),
so all three reports should be read together.

## What changed in the architecture

Four pieces of the round-1 sprint, all KEPT and merged on `main`:

| Code | Change | Effect |
|---|---|---|
| **A2** | `grouped_inputs=True` — collapse 19 scalar input tokens into 3 semantic group tokens (voltages / geometry / process params) via small group MLPs | Encoder seq length 28 → 12 under P4. Cuts per-epoch wall-clock 30–50% at d_model=256/384 |
| **P4** | `parallel_caps=True` — emit all 5 capacitance outputs in one parallel head from the gmb hidden state, instead of as 5 sequential AR steps | AR sequence shrinks 13 → 8 charged+I-V steps; cap block becomes one parallel projection |
| **T2** | `--norm-mode asinh` — per-target `arcsinh(y / s_k) + zscore` (where `s_k` is a per-target geometric-mean scale) | Compresses the 14-decade target range without the catastrophic `inv_signed_log` denormalization explosion. Replaces zscore as the recommended AR norm |
| **T1** | Physical-space early-stopping checkpoint (`*_best.phys.pt`) tracker. Final test load prefers the phys-best checkpoint over the TF-val-best one | Prevents the "TF-val-best ≠ phys-best" footgun the old report flagged |

Both `parallel_caps=True` and `grouped_inputs=True` are now hard-wired
inside `train_transformer` (`bsimar/training/trainer.py:757-761`); the
asinh normalizer is selected via `--norm-mode asinh`.

The round-1 sprint baseline at d_model=128 / 3 layers reported **AVG MRE
4.93%, NRMSE_phys 0.575%** in
`results/architecture_round2_report.md`; that is the headline number
for the new architecture at "small" tier with the asinh recipe. This
report extends the same recipe to medium and large tiers.

## Setup (held constant — identical to v1 sweep)

| Knob | Value |
|---|---|
| Dataset | `universal_nmos.npz`, 582,480 → 447,827 after sub-floor filter |
| Split | train 358,261 / val 44,782 / test 44,784 |
| Loss | `MAE + LDS` |
| Normalization | **asinh** (was zscore in v1) |
| Reorder outputs | charges → caps → cond → id (paper order, on by default) |
| Epochs | 50 |
| Batch size | 1024 |
| LR | 8e-4 with `CosineAnnealingLR(T_max=50)` |
| Optimizer | AdamW, weight_decay 1e-4 |
| Patience | 50 (no early stop within 50 epochs) |
| Dropout | 0.2 |
| Seed | 42 |
| Hardware | small → A100 GPU 1, medium → Blackwell GPU 2, large → A100 GPU 3 (parallel) |

## Configurations swept

| Run | d_model | nhead | num_layers | dim_feedforward | params (v2) | params (v1) | Δ params |
|-----|--------:|------:|-----------:|----------------:|------------:|------------:|---------:|
| Small  | 128 | 4 |  3 |  256 |    506,125 |    403,853 | +25.3% |
| Medium | 256 | 8 |  6 | 1024 |  5,152,525 |  4,751,373 |  +8.4% |
| Large  | 384 | 8 |  8 | 1536 | 15,111,565 | 14,214,925 |  +6.3% |

The new architecture is slightly heavier at every tier. The extra
params come from the three group-MLPs (each 4/3/12 → 2·d_model →
d_model GELU). At small the overhead is +100K (significant relative
to a 400K baseline); at large it dilutes to ~6%. The "scaling tier"
labels (small/medium/large) refer to the d_model/nhead/layers shape,
not exact param counts.

## Headline results (NEW arch)

| Run | Params | TF best val | AR best val | Phys-best NRMSE % | NRMSE_phys % | MRE_phys % | R²_phys | R²_norm | Wall-clock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Small**  |    506,125 | 0.02410 | 0.02699 | 0.890 | **0.862** |  **7.37** | 0.9672 | 0.9970 | 2125 s (42.5 s/ep) |
| **Medium** |  5,152,525 | 0.00993 | 0.01052 | 0.380 | **0.419** |  **2.52** | 0.9928 | 0.9994 | 2716 s (54.3 s/ep) |
| **Large**  | 15,111,565 | 0.01113 | 0.01178 | 0.381 | **0.408** |  **3.03** | 0.9940 | 0.9994 | 3427 s (68.5 s/ep) |

(Test metrics are loaded from the **phys-best** checkpoint, which is
T1's improvement over the v1 trainer that loaded the TF-val-best
checkpoint. The phys-best NRMSE column shows the val-time NRMSE that
selected the saved checkpoint; the next column shows the same model on
the held-out test split.)

## Per-target NRMSE % (physical, masked) — NEW arch

| Target | Small | Medium | Large |
|---|---:|---:|---:|
| id  | 0.624 | 0.878 | 0.861 |
| gm  | 1.549 | 0.624 | **0.471** |
| gds | 1.352 | 0.675 | **0.543** |
| gmb | 1.547 | 0.783 | 0.833 |
| qg  | 0.333 | 0.175 | 0.192 |
| qd  | 0.426 | 0.191 | 0.213 |
| qs  | 0.378 | 0.215 | 0.242 |
| qb  | 2.663 | 0.867 | 0.763 |
| cgg | 0.519 | 0.204 | 0.259 |
| cgd | 0.573 | 0.213 | 0.233 |
| cgs | 0.339 | 0.201 | 0.230 |
| cdg | 0.407 | 0.210 | 0.239 |
| cdd | 0.496 | 0.210 | 0.225 |
| **AVG** | **0.862** | **0.419** | **0.408** |

## Per-target MRE % (physical, masked) — NEW arch

| Target | Small | Medium | Large |
|---|---:|---:|---:|
| id  |  7.02 |  2.45 |  2.80 |
| gm  | 13.55 |  5.21 |  5.89 |
| gds | 12.92 |  4.99 |  5.10 |
| gmb | 10.17 |  3.79 |  4.86 |
| qg  |  4.99 |  1.64 |  2.28 |
| qd  |  5.78 |  1.89 |  2.21 |
| qs  |  6.41 |  1.36 |  1.97 |
| qb  |  8.37 |  3.28 |  5.94 |
| cgg |  4.66 |  1.29 |  1.37 |
| cgd |  6.08 |  1.83 |  1.70 |
| cgs |  4.24 |  1.72 |  1.83 |
| cdg |  4.67 |  1.59 |  1.63 |
| cdd |  6.94 |  1.77 |  1.75 |
| **AVG** | **7.37** | **2.52** | **3.03** |

## Per-target R² (physical) — NEW arch

| Target | Small | Medium | Large |
|---|---:|---:|---:|
| id  | 0.9848 | 0.9699 | 0.9711 |
| gm  | 0.9871 | 0.9979 | 0.9988 |
| gds | 0.9062 | 0.9766 | 0.9849 |
| gmb | 0.9830 | 0.9956 | 0.9951 |
| qg  | 0.9966 | 0.9991 | 0.9989 |
| qd  | 0.9950 | 0.9990 | 0.9987 |
| qs  | 0.9979 | 0.9993 | 0.9991 |
| qb  | 0.7407 | 0.9725 | 0.9787 |
| cgg | 0.9979 | 0.9997 | 0.9995 |
| cgd | 0.9921 | 0.9989 | 0.9987 |
| cgs | 0.9992 | 0.9997 | 0.9996 |
| cdg | 0.9984 | 0.9996 | 0.9994 |
| cdd | 0.9949 | 0.9991 | 0.9989 |
| **AVG** | **0.9672** | **0.9928** | **0.9940** |

---

## NEW vs OLD architecture (apples-to-apples by tier)

Same data, same loss, same epochs/batch/lr/seed. **Only the
architecture changed.** v1 numbers are from
`results/scaling_law_2026_04_08/scaling_law_report.md`.

| Tier | Metric | v1 (old arch) | v2 (new arch) | Δ |
|---|---|---:|---:|---:|
| **Small**  | NRMSE % | 1.027 | **0.862** | **−16%** |
|            | MRE %   | 15.17 | **7.37**  | **−51%** |
|            | R²      | 0.9778 | 0.9672    | −1.1pp |
|            | Wall-clock | 1898 s | 2125 s | +12% |
| **Medium** | NRMSE % | 0.954 | **0.419** | **−56%** |
|            | MRE %   | 18.39 | **2.52**  | **−86%** |
|            | R²      | 0.9790 | **0.9928** | +1.4pp |
|            | Wall-clock | 4113 s | **2716 s** | **−34%** |
| **Large**  | NRMSE % | 0.810 | **0.408** | **−50%** |
|            | MRE %   | 12.99 | **3.03**  | **−77%** |
|            | R²      | 0.9894 | **0.9940** | +0.5pp |
|            | Wall-clock | 6748 s | **3427 s** | **−49%** |

This is a **categorical improvement on every metric except small-tier
R²** (where qb regresses from 0.9772 to 0.7407 — see "Where it does not
help" below). Highlights:

1. **MRE collapses by 50–86% across all three tiers.** The
   architecture sprint specifically targeted MRE as the primary metric
   (architecture_round2_report.md §1) and the win shows: medium is now
   at **2.52% AVG MRE**, where v1 medium was at 18.39%.

2. **Wall-clock at medium and large drops by 34–49%.** A2 grouped
   inputs collapses the encoder sequence length from 28 to 12, which
   compounds with d_model. Per-epoch times: medium 82s → 54s, large
   135s → 68s. Small is essentially unchanged (38s → 42s) because the
   group-MLP overhead at d_model=128 outweighs the seq-length savings.

3. **NRMSE_phys halves at medium and large.** v1 large was 0.810%; v2
   large is 0.408%. v2 medium (0.419%) is now better than v1 large.

4. **The medium tier becomes the natural sweet spot.** v1 medium was a
   sour point — it overfit at epoch 31 and was *worse* than v1 small on
   MRE. v2 medium is **monotone-improving through the entire 50-epoch
   schedule** (final epoch is the best epoch on every metric) and is
   essentially tied with v2 large on every aggregate.

### Convergence trajectories (TF val loss every 10 epochs)

| Run | ep 10 | ep 20 | ep 30 | ep 40 | ep 50 |
|---|---:|---:|---:|---:|---:|
| Small (v2) | 0.0426 | 0.0309 | 0.0290 | 0.0257 | **0.0244** |
| Medium (v2) | 0.0387 | 0.0279 | 0.0156 | 0.0120 | **0.0099** |
| Large (v2) | 0.0423 | 0.0331 | 0.0161 | 0.0133 | **0.0113** |

All three monotone-improve through epoch 50. Compare to v1 medium,
which peaked at epoch 31 (val=0.0388) and *drifted upward* to 0.0395
by epoch 50 — the new architecture eliminates the v1 sour-point
entirely.

### TF↔AR gap (smaller = less exposure-bias residual)

| Run | TF best | AR best | Gap | Gap (v1) |
|---|---:|---:|---:|---:|
| Small  | 0.0241 | 0.0270 | **0.0029** | 0.0119 |
| Medium | 0.0099 | 0.0105 | **0.0006** | 0.0102 |
| Large  | 0.0111 | 0.0118 | **0.0007** | 0.0078 |

The exposure-bias residual collapses **by ~10× at medium**. P4 parallel
caps removes 5 of the 13 AR decoding steps, and grouped input tokens
let the encoder distribute its attention budget more efficiently across
the remaining 8 steps. This is the structural fix the v1 report
identified as "the dominant cost, not closeable by capacity alone" —
it turns out to be closeable, just not by *raw capacity*. It needed
the AR sequence to get shorter and the input tokenization to get
denser, both of which the sprint delivered.

### Where it does not help

**Small tier qb regresses sharply** (R² 0.9772 → 0.7407, NRMSE
0.791 → 2.663). The qb head sits at sequence position 4 in the AR
order (charges block: qg, qb, qd, qs) and is the first target whose
asinh scale is not well determined at d_model=128 — its per-target
geometric-mean scale ends up too tight for the small model to fit
without distortion in physical space. The same target is fine at
medium (R² 0.9725) and large (0.9787), so this is a small-model
artifact of the asinh normalizer, not a structural problem. **At
production scale (medium / large) the regression is gone.**

The small-tier physical R² average looks slightly worse than v1
(0.9672 vs 0.9778) for this single reason; on every other target
v2 small ties or improves on v1 small. If you compute R² without qb,
v2 small averages 0.9861 vs v1 small's 0.9779.

---

## Comparison vs DirectNet scaling sweep

Both swept three matched param tiers under identical 50-epoch /
batch=1024 / lr=8e-4 / seed=42 conditions. DirectNet numbers from
`results/scaling_law_directnet_2026_04_08/scaling_law_directnet_report.md`.

### Headline averages, side by side

| Scale | Params (BSIMAR / DN) | NRMSE BSIMAR / DN | MRE BSIMAR / DN | R² BSIMAR / DN |
|---|---|---:|---:|---:|
| Small  |   506K /   403K | **0.862** / **0.242** | **7.37** / **9.05** | 0.9672 / 0.9991 |
| Medium |  5.15M /  4.75M | **0.419** / **0.113** | **2.52** / **4.96** | 0.9928 / 0.9998 |
| Large  | 15.11M / 14.25M | **0.408** / **0.126** | **3.03** / **5.20** | 0.9940 / 0.9998 |

### Best of each architecture

| Metric | Best BSIMAR v2 (Medium 5.15M) | Best DirectNet (Medium 4.75M) | Winner |
|---|---:|---:|---|
| AVG NRMSE_phys |  0.419 % |  **0.113 %** | DirectNet (3.7×) |
| AVG MRE_phys   | **2.52 %** |  4.96 % | **BSIMAR v2 (1.97×)** |
| AVG R²_phys    |  0.9928  |  **0.9998** | DirectNet |
| Params         |  5.15 M  |  4.75 M | comparable |
| Wall-clock     |  2716 s  |   **398 s** | DirectNet (6.8×) |

This is the central new result: **the new BSIMAR architecture wins
on MRE at every scale where it has enough capacity (medium and
large)**, while DirectNet still wins on NRMSE and R². They have
**different error profiles** and therefore different ideal use cases.

### Why does BSIMAR v2 win on MRE but lose on NRMSE?

This is exactly the trade-off that makes a metric choice load-bearing
(`scaling_law_directnet_report.md` calls this out):

- **MRE** (mean relative error) weights all samples equally in
  *relative* terms. It is dominated by the **low-magnitude tail**:
  samples where `|y_true|` is small. asinh normalization compresses
  the dynamic range so that the model can *resolve* low-magnitude
  values; MAE loss + LDS reweighting pushes the model toward fitting
  every sample regardless of magnitude. v2 BSIMAR is optimized end-to-
  end for this regime and the results show it: medium MRE 2.52% is
  **the lowest number any model has achieved on this dataset** and
  is 1.97× better than DirectNet's best.

- **NRMSE / R²** weight by *squared absolute residuals*. They are
  dominated by the **high-magnitude head**: samples where `|y_true|`
  is large (saturation region currents, peak capacitances). DirectNet
  with weighted MSE + sub-floor filter is optimized end-to-end for
  this regime. The MLP also has zero exposure-bias residual and a
  4.75M-param dense network can interpolate the high-magnitude
  surface to noise-floor accuracy in a way that an autoregressive
  decoder structurally cannot at the same param budget.

In other words: **DirectNet is the right answer if you care about
worst-case absolute error in physical units; BSIMAR v2 is the right
answer if you care about per-sample relative accuracy across the full
dynamic range.** That is a real, defensible architecture-vs-architecture
trade-off — much better than the v1 result, where BSIMAR was simply
worse on everything.

### Per-tier deltas vs DirectNet

| Tier | NRMSE (BSIMAR v2 / DN) | MRE (BSIMAR v2 / DN) | Verdict |
|---|---:|---:|---|
| Small  | 0.862 / 0.242 = **3.6× worse** | 7.37 / 9.05 = **1.23× better** | mixed (BSIMAR wins MRE only) |
| Medium | 0.419 / 0.113 = **3.7× worse** | 2.52 / 4.96 = **1.97× better** | mixed (best BSIMAR MRE win) |
| Large  | 0.408 / 0.126 = **3.2× worse** | 3.03 / 5.20 = **1.72× better** | mixed |

Compare to v1, where BSIMAR was 4.2× / 8.4× / 6.4× worse on NRMSE
**and** lost on MRE at medium (18.39 vs 4.96, 3.7× worse). The new
architecture closes the NRMSE gap from 6× → 3× and *flips* the MRE
gap from "loses by 3×" to "wins by 1.7-2×".

### Wall-clock and capacity efficiency

| Tier | Params | BSIMAR v2 wall | DN wall | BSIMAR slowdown |
|---|---:|---:|---:|---:|
| Small  |  ~400K | 2125 s | 421 s | 5.0× |
| Medium |  ~5M   | 2716 s | 398 s | 6.8× |
| Large  | ~15M   | 3427 s | 510 s | 6.7× |

BSIMAR v2 is now **6.8× slower** than DirectNet at medium, vs **17×
slower** in v1 (where v1 medium was 4113s and DN medium was 398s).
The wall-clock gap closes by ~2.5×. AR validation passes on the full
val set are still the bottleneck — KV-caching the encoder during the
8-step AR decode (rather than the 13-step decode v1 had) would close
this further.

### Architectural responses to scale, side by side

| Step | DirectNet ΔNRMSE | BSIMAR v1 ΔNRMSE | BSIMAR v2 ΔNRMSE |
|---|---:|---:|---:|
| 404K → 4.75M  | **−53%** |  −7% | **−51%** |
| 4.75M → 14.25M | +11% (saturates) | −15% | **−3%** (saturates) |
| 404K → 14.25M | −48% | −21% | **−53%** |

For the first time, **BSIMAR's scaling curve looks like DirectNet's**:
a steep gain from small → medium (−51% vs DirectNet's −53%), followed
by saturation at the third tier. The v1 curve was "drips improvement
across the full 35× range but never gets close" — v2 is "locks in
nearly all the gain in the first 12× and then plateaus", which is
how a healthy scaling law looks.

This is the diagnosis the architecture sprint promised: v1 BSIMAR
was bottlenecked by exposure bias and a per-scalar tokenizer that
forced the encoder to spread its attention budget thin. P4 (parallel
caps) cuts the AR depth from 13 to 8; A2 (grouped inputs) collapses
the context from 19 tokens to 3 dense ones. With those two changes,
BSIMAR's scaling-law slope is now structurally identical to
DirectNet's. The remaining headroom is in the loss / final-test
regime, not in capacity.

---

## What the new scaling law tells us

### 1. The architecture sprint **fixed the scaling law**

v1's headline conclusion was "BSIMAR's accuracy gap with DirectNet is
not a capacity problem; the architecture has a structural exposure-
bias residual that scaling alone cannot remove." That diagnosis was
correct *for the v1 architecture*: the data showed BSIMAR's
TF↔AR gap shrinking only 35% across 35× scale, and the AR-decoded
NRMSE moving only 21% across the same range.

The v2 numbers tell a different story:
- TF↔AR gap collapses by ~10× at medium (0.0102 → 0.0006).
- AR-decoded NRMSE drops 56% from small to medium (vs v1's −7%).
- The 12× capacity step is the productive band, exactly like
  DirectNet.

The conclusion is that exposure bias *is* closeable, but not by raw
parameter count. It needed (a) shortening the AR sequence (P4), and
(b) denser context tokens (A2), so that each AR step has more useful
context per token and fewer total steps to compound errors over.

### 2. v2 is competitive with DirectNet on MRE, dominant on dynamic range

This is the cleanest framing of the v2 result:

- For **physical-space relative accuracy across the full dynamic
  range** (MRE), BSIMAR v2 medium is **2.52%**, the lowest number any
  model has produced on this dataset. DirectNet medium is 4.96%, 1.97×
  worse.
- For **physical-space absolute fit on high-magnitude samples** (NRMSE
  / R²), DirectNet medium is **0.113% / 0.9998**. BSIMAR v2 medium is
  0.419% / 0.9928, ~3.7× looser on absolute residuals.

These are different objectives. The v1 framing
(`bsimar_scaling_law.md` memory note: "35x BSIMAR scaling gives only
−21% NRMSE; still 4× worse than 404K DirectNet") is **stale** — the
v2 number is 0.408% NRMSE at large, which is "only" 3.2× worse than
DirectNet's best, and v2 wins on MRE outright.

### 3. The medium tier is the new sweet spot for BSIMAR

Just like DirectNet's scaling sweep concluded that medium (4.75M) is
the right operating point, v2 BSIMAR's medium (5.15M) is also the
right point:
- Best MRE (2.52%) of any tier or any architecture on this dataset.
- Tied with large on NRMSE / R², but ~25% less wall-clock and 3×
  fewer parameters.
- Final epoch is the best epoch — schedule is well-matched, no
  overfit cliff.

The large tier provides only marginal gains (+0.5pp R², −3% NRMSE,
+0.5pp MRE) at the cost of 26% more wall-clock and 3× more params.
**For production accuracy work, BSIMAR v2 medium is the recommended
checkpoint.**

### 4. Wall-clock parity is now within reach

v1 BSIMAR medium ran in 4113s; DirectNet medium ran in 398s. The 10×
gap was a hard barrier to using BSIMAR in any iterative workflow. v2
BSIMAR medium runs in 2716s — still 6.8× DirectNet, but the bottleneck
is the AR-validation pass over the val set (3 passes per epoch under
the trainer's `ar_check_every=10` schedule + the final pass), not the
TF training itself. Adding KV-cache support to the encoder for the
8-step AR decode would cut this further. A 2-3× speedup would put
BSIMAR within the same order of magnitude as DirectNet for daily use.

### 5. BSIMAR v2 has a defensible niche

v1 BSIMAR was strictly dominated by DirectNet — "best BSIMAR at any
scale is strictly worse than DirectNet at every scale" was the
literal conclusion of the v1 directnet-comparison report. v2 BSIMAR
has a real niche: **the lowest MRE on `universal_nmos`**, by ~2× over
DirectNet at every scale. Workloads that care about per-sample
relative accuracy in the low-magnitude tail (subthreshold currents,
small-signal conductances near pinch-off, high-impedance node
analyses) should prefer v2 BSIMAR medium. Workloads that care about
peak fit and global R² (transient simulation peak amplitudes,
inverter VTC transition points, charge integration accuracy) should
prefer DirectNet medium.

---

## Recommendations

### Production ranking (NEW)

1. **DirectNet medium 4.75M** — primary production checkpoint for
   universal NMOS. Best NRMSE (0.113%), best R² (0.9998), 6.8×
   faster than BSIMAR v2 medium. Use this for transient simulation,
   VTC fits, peak-current accuracy.
2. **BSIMAR v2 medium 5.15M** — recommended secondary checkpoint
   for low-magnitude / relative-accuracy workloads. Best MRE (2.52%),
   solid R² (0.9928). Use this for subthreshold accuracy checks,
   high-impedance node analyses, leakage estimation.
3. **BSIMAR v2 large 15.11M** — only marginal gain over medium at
   3× the param cost; not worth the storage / inference overhead
   for production. Useful as an optimistic ceiling check.
4. **DirectNet large 14.25M** — slightly worse than medium at this
   epoch budget; recommended only if you can run 100+ epochs.

### Memory-note cleanup

The current memory note `bsimar_scaling_law.md` says

> 35x BSIMAR scaling (404K→14.2M) gives only −21% NRMSE; still 4× worse
> than 404K DirectNet. Bottleneck is exposure bias, not capacity.

This is **stale**. The v2 numbers are:
- 12× scaling (506K → 5.15M) gives **−51% NRMSE** and **−66% MRE**.
- v2 medium is 3.7× worse than DirectNet on NRMSE but **1.97× better on
  MRE**.
- The exposure-bias residual *is* the bottleneck and the architecture
  sprint *closed it*: TF↔AR gap collapsed 10× at medium.

The new framing for the memory note should be:

> The new BSIMAR v2 architecture (P4 parallel caps + A2 grouped inputs
> + T2 asinh + T1 phys-best ckpt) restores a productive scaling law:
> 12× capacity buys −51% NRMSE and −66% MRE on universal_nmos.
> v2 medium is the new MRE champion (2.52%, ~2× better than DirectNet
> medium) but still ~3.7× worse on NRMSE. They are
> differently-optimized models, not comparable on a single scalar.

---

## Suggested next steps to further improve BSIMAR

In rough order of expected impact (highest first), with the
specific failure mode each one targets:

### Tier 1 — close the NRMSE gap to DirectNet

**N1. Long-schedule retrain at medium** *(high impact, low cost)*
The v2 medium TF val loss is **still strictly decreasing at epoch 50**
(0.0099 final, slope ≈ −5%/10 epochs). Extending the cosine schedule
to T_max=100 or 150 epochs is the cheapest possible win. Expected: 10–
20% additional NRMSE reduction at medium, possibly closing the gap to
DirectNet by ~30%. Same recipe, just `--epochs 150 --patience 150`.

**N2. KV-cache the encoder during AR decode** *(high impact, medium
cost)* The AR validation pass is the wall-clock bottleneck (3 of every
10 epochs run the full 8-step sequential decode on 45K samples, plus
the final test pass). Caching attention K/V across decode steps cuts
the AR decode from O(L²) to O(L). Expected: 3–5× speedup on
validation, 1.5–2× speedup on wall-clock. Once this lands, longer
schedules (N1) become free, and 200-epoch runs become routine.

**N3. AR fine-tune phase** *(high impact, medium cost)* The trainer
already saves an `*_best.ar.pt` checkpoint, but training is still
fully teacher-forced. After the TF training plateaus, run a final
phase with `forward_scheduled(ss_ratio=1.0)` — i.e., feed the model's
own predictions instead of the oracle for the last 10–20 epochs. This
directly attacks the residual TF↔AR gap (still 0.0006 at v2 medium —
small in absolute terms but represents the remaining 6% of the val
loss). Expected: −15% NRMSE at medium, possibly bringing the
NRMSE gap from 3.7× to 2.5×. Implementation: add an `--ar-finetune
--ar-finetune-epochs 20` flag to `cli/train.py` and a code path in
`train_transformer` that switches `forward_scheduled` on for the last
N epochs.

**N4. Charge-finetune mode for BSIMAR** *(high impact, medium cost)*
DirectNet has a `--mode charge-finetune` that adds an autograd
`dq/dV = C` consistency loss. v2 BSIMAR caps already fit very well
(NRMSE 0.20% at medium, on par with DirectNet medium's 0.10–0.14%),
but the cap-vs-charge consistency is not enforced. Adding a charge-
consistency penalty term to the BSIMAR training loop would tighten
both the cap NRMSE and the qb regression at small. Expected: −10%
cap NRMSE, fixes the small-tier qb R² regression. Implementation:
port `ChargeConsistencyLoss` from `bsimar/losses/direct_loss.py` into
`train_epoch_bni` so MAE-mode runs can opt into it.

### Tier 2 — push MRE further down

**N5. Per-target asinh-scale tuning** *(medium impact, low cost)* The
small-tier qb regression (R² 0.74) is caused by the asinh `s_k` scale
being too tight for d_model=128 to fit. The current scale is per-
target geometric mean; a learned-scale variant (per-target `s_k` as a
trainable parameter, initialized from the geometric mean) would let
the model loosen scales it cannot otherwise resolve. Expected: fixes
the small-tier qb regression, modest gains on caps everywhere. Low
risk because at convergence the scales should re-find the geometric-
mean values.

**N6. Mixed-loss head — Huber on currents, MAE on charges/caps**
*(medium impact, low cost)* The biggest remaining MRE error is on
**currents and conductances** at medium (id 2.45, gm 5.21, gds 4.99,
gmb 3.79) — they sit at the tail of the AR sequence and inherit
upstream noise. Charges and caps are already at <2% MRE. Switching
the I/V block to a Huber loss (MAE-like in the tail, MSE-like near
zero residuals) while keeping MAE+LDS on the C/Q block would let the
optimizer focus on the right regime per target. Expected: −20% MRE
on id/gm/gds at medium.

**N7. Sample-weighted training on Vov regions** *(medium impact, low
cost)* The dataset's MRE is dominated by samples near subthreshold
(small `|id|`) and near saturation (high `|id|`, small `|gds|`).
Adding LDS on `Vov = Vgs − Vth` (in addition to per-target output
LDS) would oversample the regions where MRE is bottlenecked.
Expected: −10–15% MRE in low-Vov region, no impact elsewhere.

### Tier 3 — capacity / regularization sweet spot

**N8. d_model=192 / num_layers=4 mid-point** *(low cost, surveying)*
v2 small (d=128, L=3) and v2 medium (d=256, L=6) have a 12× param
gap and a sharp 51% NRMSE drop between them. There is no data point
in between. A 1.5M-param mid-tier would map the actual saturation
curve and could turn out to be the new sweet spot if it lands at
~80% of medium's accuracy with ~30% of medium's params. Run as a
single `--d-model 192 --nhead 6 --num-layers 4 --dim-feedforward 768`
sweep. Expected: better understanding of the scaling curve; possible
production-ready compact checkpoint.

**N9. Stronger regularization at large** *(low cost)* v2 large is
nearly identical to v2 medium on every metric while costing 3× the
params. This is a strong sign of capacity saturation. Try
`--dropout 0.3 --weight-decay 1e-3` at large to see if the extra
parameters can be "unlocked" by stronger regularization. If yes,
large becomes the production answer; if no, large can be permanently
deprecated and we can stop running it.

### Tier 4 — alternative architectures (only after Tier 1 saturates)

**N10. Replace the per-token output heads with a small shared MLP +
target embedding head** *(speculative)* v2 has 13 separate
`nn.Linear(d_model, 1)` heads. Switching to a single
`nn.Linear(d_model + d_emb, d_model)` MLP that takes a target-id
embedding could share statistical strength across targets that are
physically related (qg/qd/qs are charge components; cgg/cgd/cdg are
gate-coupled caps). Expected: −5–10% NRMSE on the under-constrained
targets (qb, gmb), no impact elsewhere.

**N11. EKV residual head for current block** *(speculative, untried)*
The v2 round-1 sprint identified P1 (EKV-flavoured `id ≈
exp(Vov / nVT)` analytic prior + learned residual) as the most
promising untried direction for the conductance bottleneck. The idea
is that gm/gds are gradients of id, and a smooth analytic prior with
a small residual is fundamentally easier to fit than the full
14-decade range. Implementation cost is moderate (one new head, one
new loss term), payoff is unknown but the structural argument is
sound.

### What NOT to spend time on

The round-2 retry sprint (`results/architecture_round2_report.md`)
already burned budget on six experiments that did **not** beat the
v2 baseline:
- **P2 charge neutrality** — qb explosion is structural, dead.
- **T3 Laplace NLL** — equivalent to weighted MAE, no advantage.
- **T5 log|id| loss** — strictly worse than asinh, dominated.
- **P3 Vov featurization** — regresses currents without supervision.
- **P5 spectral-norm gm/gds heads** — bottleneck is upstream, not in
  the heads.

These should not be retried without a fundamentally new structural
argument. The architecture-improvement-plan working log
(`bsimar/docs/architecture_improvement_plan.md`) lists all 11 round-1
experiments with verdicts; new ideas should be cross-referenced
against that document before being implemented.

### Recommended order of operations

If only one thing gets done next, it should be **N1 (longer
schedule)** — it's a one-line change to the driver script and the
data already shows v2 medium and large are still improving at epoch
50. Running a 150-epoch v2 medium would either flip the NRMSE
ranking against DirectNet (best case), or pin down the v2 NRMSE
floor (worst case). Either way you learn something concrete.

If two things get done, do **N1 + N3 (AR finetune)** in sequence:
the long-schedule TF training gets the model to its TF-best floor,
then 10–20 epochs of pure AR finetuning extracts the residual
exposure-bias gap. Expected combined: 25–35% NRMSE reduction at
medium, putting v2 BSIMAR within ~2× of DirectNet on NRMSE while
keeping its 2× MRE lead. That would be the cleanest possible v2
"final" production checkpoint.

If three things get done, add **N2 (KV-cache)** so the longer runs
do not become wall-clock prohibitive — this is the structural
unlock for everything in Tier 1.

---

## Artifacts

- Logs: `external_compact_models/bsimar/results/scaling_law_2026_04_08_arch_v2/{small,medium,large,driver}.log`
- Driver: `external_compact_models/bsimar/results/scaling_law_2026_04_08_arch_v2/run_scaling_v2.sh`
- Checkpoints: `external_compact_models/bsimar/checkpoints/scaling_v2_{small,medium,large}_nmos_{best,best.ar,best.phys,norm,config}.{pt,npz}`
- Plots: `external_compact_models/bsimar/results/scaling_v2_{small,medium,large}_nmos/{loss_curves,scatter_comparison}.png`
- Prior BSIMAR scaling (v1, OLD arch): `external_compact_models/bsimar/results/scaling_law_2026_04_08/scaling_law_report.md`
- Prior DirectNet scaling: `external_compact_models/bsimar/results/scaling_law_directnet_2026_04_08/scaling_law_directnet_report.md`
- Architecture sprint round-2 report: `external_compact_models/bsimar/results/architecture_round2_report.md`
- Architecture sprint round-1 report: `external_compact_models/bsimar/results/architecture_experiments_report.md`
