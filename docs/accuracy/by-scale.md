# Accuracy by scale — what small / medium / large / xl actually buy

Cross-family view along the **capacity** axis. Companion pivots:
[`by-tech.md`](by-tech.md) (technology), [`by-recipe.md`](by-recipe.md)
(training recipe). Gate definitions and the code-state ladder:
[`methodology.md`](methodology.md).

---

## 1. The tiers

| tier | DirectNet | BSIM-AR | PFN |
|---|---|---|---|
| **small** | 128 × 3 ≈ **0.06 M** | 0.67 M | 0.69 M |
| **medium** | 256 × 5 ≈ **0.40 M** | 1.94 M | 2.03 M |
| **large** | 384 × 6 ≈ **0.92 M** | 5.02 M | 4.65 M |
| **xl** | 512 × 8 ≈ **2.13 M** | 384 × 8L × ff1536 = **14.81 M** | 192 × (4+4+9) blocks = **14.86 M** |

The three families do **not** share a parameter scale: BSIM-AR `small` is bigger
than DirectNet `large`, and PFN `small` is bigger than DirectNet `medium`. Tier
names compare a family with itself, never across families.

PFN's `xl` tier is new in V7.1.0 — the family had only three scales until then.
Its 14.86 M mirrors BSIM-AR xl's 14.81 M and its ICL width
(embed_dim × n_cls_tokens = 384) equals BSIM-AR xl's `d_model`, so the top of
the capacity axis is comparable across families. Training in progress; the rows
below fill in as the checkpoints and their gates land.

CPU cost (1 thread, the gate configuration): DirectNet large **1.5 ms/eval**,
DirectNet xl 3.4 ms, PFN small 15.6 ms, BSIM-AR medium 61.5 ms. Cost is
**memory-bandwidth-bound on the weights**, so it tracks parameter count times
the number of times the checkpoint is streamed per evaluation — which is why
BSIM-AR's 8-step autoregressive loop costs ~40× DirectNet rather than ~5×
(`docs/plans/2026-07-25-v700-nn-perf.md`).

## 2. Complex gates by tier (post-fix, V6.13.0, single-run OMP=1)

### Clean recipe — the honest capacity curve

| family | small | medium | large | xl |
|---|---|---|---|---|
| **DirectNet** | 10/16 | 10/16 | **13/16** | 12/16 |
| *(pre-fix)* | *7* | *10* | *13* | *10* |
| **BSIM-AR** | **14/16** | **14/16** | **14/16** | **14/16** |
| *(pre-fix)* | *12* | *14* | *13* | *13* |
| **PFN** | **11/16** | **11/16** | 9/16 | — |
| *(pre-fix)* | *11* | *10* | *8* | — |

DirectNet's `large` row is the archived **`v660clean_large`**, not the
`tsmc{X}_dn_large_*` slot — that slot has carried the `crit30f` curriculum
weights since V6.6.4 and appears in the curriculum table below.

Three different shapes, and two of the three changed when the `gds` sign bug was
fixed:

* **DirectNet peaks at `large`** — the one curve that survived intact. Beyond
  `large` the model fits the training distribution ~10× tighter (val loss ~2e-4
  vs medium's ~2e-3) and generalizes *worse* off-nominal.
* **BSIM-AR is flat — 14/16 at every tier, failing the same two cells
  (`tsmc5-ring`, `tsmc7-ring`) every time.** The pre-fix reading "capacity peaks
  at medium (12 → 14 → 13)" was **largely a gds artifact**; what the fix bought
  was the entire opamp column at all four sizes.
* **PFN declines** past `medium`, and the decline survives the fix. Its `large`
  tier is also optimization-unstable (8 divergence-collapse events —
  `PFN-L75-accuracy.md` §2).

### With the curriculum recipes

| family | small | medium | large | xl |
|---|---|---|---|---|
| DirectNet | — | — | **15/16** (`crit30f`, production) | **16/16** (`crit15m`), 15 (`corroft`), 14 (`crit10`) |
| BSIM-AR | — | **16/16** (`corroft`, `corro15`) | 15/16 (`corroft`, `crit15m`, `crit30`) | **16/16** (all four corridor recipes) |
| PFN | *(never trained)* | | | |

The curriculum **inverts DirectNet's capacity story**: clean@xl is 12/16 and one
identical fine-tune adds up to +4. And it removes BSIM-AR's flatness in the
opposite direction — `large` is the *worst* corridor tier (15/16), with `medium`
and `xl` both sweeping.

## 3. Strict determinism by tier

Strict = PASS at OMP ∈ {1,2,4}. Ten groups were strict-swept post-fix; **all ten
are flip-free**, which is the durable result (`methodology.md` §3).

| group | strict | flips |
|---|---|---|
| `dn/crit15m/xl` | **16/16** | 0 |
| `tf/corro15/xl` | **16/16** | 0 |
| `tf/corroft/medium` | **16/16** | 0 |
| `dn/clean/large` (production) | 15/16 | 0 |
| `dn/corroft/xl` | 15/16 | 0 |
| `dn/crit10/xl` | 14/16 | 0 |
| `tf/clean/large` | 14/16 | 0 |
| `dn/v660clean/large` | 13/16 | 0 |
| `dn/clean/xl` | 12/16 | 0 |
| `pfn/clean/small` | 11/16 | 0 |

Groups not in this list have no post-fix strict measurement; their single-run
counts are in `by-recipe.md` §2 and must not be read as strict.

### Strict sweeps added in V7.1.0

The V7.1.0 pass re-runs opamp and ring at OMP ∈ {1,2,4} for every group it
touches, which extends strict coverage beyond the ten groups above. A group with
unmeasured cells is shown against its measured denominator, not 16.

| group | strict PASS | FLIPs | cells not yet measured |
|---|---|---|---|
| `dn/corroft_xl` | 11/12 measured | 0 | 4 |
| `dn/crit10_xl` | 1/1 measured | 0 | 15 |
| `dn/crit30f_large` | 15/16 | 0 | — |
| `dn/csob_large` | 9/13 measured | 0 | 3 |
| `dn/large` | 15/16 | 0 | — |
| `dn/medium` | 10/16 | 0 | — |
| `dn/small` | 10/16 | 0 | — |
| `dn/v660clean_large` | 13/16 | 0 | — |
| `dn/xl` | 10/12 measured | 0 | 4 |
| `pfn/large` | 4/4 measured | 0 | 12 |
| `pfn/small` | 8/11 measured | 0 | 5 |

**Zero FLIPs, everywhere, again.** The V6.13.0 result — that the OMP
multistability was a wrong-signed Jacobian and not a property of high-gain
circuits — now holds across the newly swept groups too, including the two
DirectNet tiers (`small`, `medium`) that had never been strict-swept at all.
`dn/crit30f_large` re-measures the production slot independently at 15/16
strict, matching `dn/large`.

Pre-fix, the tiers differed sharply in how *stable* they were: `xl` basins were
OMP-deterministic (strict ≈ single-run for every recipe; the sole FLIP in the
whole tier was `corft`'s tsmc5-ring sitting at 4.6/4.9/5.1 % around the 5 %
gate), while at `large` opamp FLIPs were endemic. That difference is gone —
it was the wrong-signed Jacobian entry, not a tier property.

## 4. Device-level fidelity by tier

Parametric DC is **gds-invariant** (`methodology.md` §6), so DC numbers are
comparable across the whole campaign history; transient and AC are not, and are
re-measured below.

**Parametric DC — `verify_nn_multi_tech_dc`, 55 configs**

| family | tier | pass | mean NRMSE % — TSMC5 / 7 / 12 / 16 |
|---|---|---|---|
| DirectNet | small | 55/55 | 2.39 / 1.81 / 0.57 / 0.62 |
| DirectNet | medium | 55/55 | 1.48 / 1.46 / 0.56 / 0.58 |
| DirectNet | large | 54/55 | 1.91 / 1.21 / 1.69 / 1.01 |
| DirectNet | xl | 53/55 | 2.91 / 2.35 / 2.73 / 1.52 |
| BSIM-AR | small | 54/55 | 2.54 / 1.24 / 0.82 / 1.61 |
| BSIM-AR | medium | 53/55 | 1.77 / 1.46 / 1.34 / 1.57 |
| BSIM-AR | large | 52/55 | 1.80 / 1.21 / 1.67 / 1.58 |
| BSIM-AR | xl | 55/55 | 1.94 / 2.92 / 1.08 / 1.07 |
| PFN | small | 54/55 | 2.14 / 1.58 / 0.56 / 1.12 |

**Parametric transient — `verify_nn_multi_tech_tran`, 64 configs**

| family | tier | pass | mean NRMSE % — TSMC5 / 7 / 12 / 16 |
|---|---|---|---|
| DirectNet | small | 64/64 | 2.99 / 1.53 / 2.07 / 1.62 |
| DirectNet | medium | 64/64 | 1.90 / 1.48 / 1.52 / 1.47 |
| DirectNet | large | 64/64 | 1.67 / 1.46 / 1.49 / 1.47 |
| DirectNet | xl | 64/64 | 1.66 / 1.45 / 1.52 / 1.48 |
| BSIM-AR | small | 64/64 | 2.54 / 1.47 / 1.53 / 1.60 |
| BSIM-AR | medium | 64/64 | 1.80 / 1.52 / 1.52 / 1.50 |
| PFN | small | 64/64 | 1.88 / 1.44 / 1.50 / 1.48 |

**Non-tier (recipe) stems measured in the same pass**

| stem | device AC | opamp AC | DC | tran |
|---|---|---|---|---|
| `dn/v660clean_large` | 7/8 | 0/4 | 54/55 | 64/64 |
| `dn/csob_large` | 8/8 | 1/4 | 55/55 | 64/64 |
| `dn/corroft_xl` | 6/8 | 0/4 | — | — |
| `dn/crit10_xl` | 2/2 | 0/2 | — | — |
| `dn/crit15m_xl` | 2/2 | 0/0 | — | — |
| `tf/corroft_medium` | 6/6 | 2/4 | — | — |

Reading the DC column: **`medium` is the best device fit for DirectNet**, and
`large`'s slightly worse device numbers are the price of the curriculum
fine-tune it carries in production. Device fidelity has never been the bind —
`large` wins the *circuit* gates while `medium` wins the *device* metrics, which
is the cleanest statement of the central result that circuit gates measure the
value surface at fixed points, not pointwise device accuracy.

## 5. AC by tier — re-measured in V7.1.0

**This is the axis the `gds` fix invalidated most, and it had never been
re-measured per tier.** The V6.13.0 campaign re-ran the AC suites only for each
family's resolver-default stem. Every per-tier AC number in these reports was a
pre-fix measurement until now.

> **Coverage.** The denominator in each row is what has been *measured*, not the
> full 8 (device AC) or 4 (opamp AC) cells — the BSIM-AR and PFN rows fill in as
> their (much slower) cells complete. A row reading `1/1` is one measured cell,
> not a complete tier. Raw per-cell data: `results/v710_regate/REPORT.md`.

**Device CS-amp AC (gate: gain0 ≤1.5 dB, f3db ratio ∈[0.7,1.43], magNRMSE ≤10 %)**

| family | tier | pass | per-tech (n = NMOS, p = PMOS) |
|---|---|---|---|
| DirectNet | small | **7/8** | TSMC5: n✓ p✓ · TSMC7: n✗ p✓ · TSMC12: n✓ p✓ · TSMC16: n✓ p✓ |
| DirectNet | medium | **8/8** | TSMC5: n✓ p✓ · TSMC7: n✓ p✓ · TSMC12: n✓ p✓ · TSMC16: n✓ p✓ |
| DirectNet | large | **8/8** | TSMC5: n✓ p✓ · TSMC7: n✓ p✓ · TSMC12: n✓ p✓ · TSMC16: n✓ p✓ |
| DirectNet | xl | **7/8** | TSMC5: n✗ p✓ · TSMC7: n✓ p✓ · TSMC12: n✓ p✓ · TSMC16: n✓ p✓ |
| BSIM-AR | small | **8/8** | TSMC5: n✓ p✓ · TSMC7: n✓ p✓ · TSMC12: n✓ p✓ · TSMC16: n✓ p✓ |
| BSIM-AR | medium | **8/8** | TSMC5: n✓ p✓ · TSMC7: n✓ p✓ · TSMC12: n✓ p✓ · TSMC16: n✓ p✓ |
| BSIM-AR | large | **4/4** | TSMC5: n✓ p✓ · TSMC7: — · TSMC12: — · TSMC16: n✓ p✓ |
| BSIM-AR | xl | **0/2** | TSMC5: — · TSMC7: n✗ p✗ · TSMC12: — · TSMC16: — |
| PFN | small | **8/8** | TSMC5: n✓ p✓ · TSMC7: n✓ p✓ · TSMC12: n✓ p✓ · TSMC16: n✓ p✓ |
| PFN | medium | **2/2** | TSMC5: n✓ p✓ · TSMC7: — · TSMC12: — · TSMC16: — |
| PFN | large | **4/4** | TSMC5: n✓ p✓ · TSMC7: n✓ p✓ · TSMC12: — · TSMC16: — |

**Opamp open-loop AC (gate: DC-gain err ≤3 dB, GBW ratio ∈[0.6,1.67], PM err ≤15°, and a non-railed NN OP; magNRMSE reported, not gated). The number shown is the DC-gain error**

| family | tier | pass | TSMC5 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|---|
| DirectNet | small | **1/4** | FAIL 2.41 dB | FAIL 19.50 dB | FAIL 42.27 dB | PASS 1.78 dB |
| DirectNet | medium | **0/4** | FAIL 22.02 dB | FAIL 29.46 dB | FAIL 27.69 dB | FAIL 35.11 dB |
| DirectNet | large | **0/4** | FAIL 3.31 dB | FAIL 31.88 dB | FAIL 6.44 dB | FAIL 4.98 dB |
| DirectNet | xl | **0/4** | FAIL 14.72 dB | FAIL 31.48 dB | FAIL 208.73 dB | FAIL 3.73 dB |
| BSIM-AR | small | **1/4** | FAIL 0.39 dB | FAIL 11.33 dB | FAIL 10.27 dB | PASS 0.54 dB |
| BSIM-AR | medium | **1/4** | FAIL 4.63 dB | FAIL 11.13 dB | FAIL 0.17 dB | PASS 2.62 dB |
| BSIM-AR | large | **2/4** | FAIL 2.54 dB | PASS 0.12 dB | FAIL 4.34 dB | PASS 0.33 dB |
| BSIM-AR | xl | **1/4** | FAIL 24.95 dB | FAIL 3.86 dB | FAIL 5.18 dB | PASS 0.97 dB |
| PFN | small | **0/4** | FAIL 16.04 dB | FAIL 30.04 dB | FAIL 8.46 dB | FAIL 4.44 dB |
| PFN | medium | **1/4** | FAIL 15.61 dB | PASS 0.84 dB | FAIL 3.29 dB | FAIL 33.44 dB |
| PFN | large | **0/2** | FAIL 5.32 dB | FAIL 30.99 dB | — | — |

### What this changes

1. **"AC peaks at SMALL" is retracted.** DirectNet's device CS-amp AC is
   **7/8 · 8/8 · 8/8 · 7/8** across small → xl: essentially saturated at every
   capacity, with the only two misses at the *ends* of the range. Pre-fix the
   same suite read 5/12 · 4/12 · 4/12 · 4/12 — a pass fraction near a third,
   declining with capacity. Both the level and the shape were artifacts of the
   wrong-signed `gds`. (The /12 denominator is pre-TSMC6-retire; see
   `methodology.md` §5. The comparison that matters is the fraction, not the
   count.)

2. **The two surviving device-AC misses are the documented failure classes, not
   new ones.** DirectNet `small` on TSMC7-NMOS fails on *gain* (2.03 dB against
   a 1.5 dB gate) and magnitude (24.9 %) — an under-capacity value surface on
   the steepest tech. DirectNet `xl` on TSMC5-NMOS fails on *pole placement*
   (f3db ratio 2.51 against [0.7, 1.43]) with gain and magnitude comfortably
   in-gate — output-capacitance under-prediction, exactly the class
   `DirectNet-L73-accuracy.md` §4 names.

3. **The production curriculum improved the charge surface, not just the current
   surface.** At the same tier and the same architecture, `v660clean@large`
   fails TSMC5-NMOS (f3db 1.78) while the `crit30f` weights that replaced it in
   the production slot pass 8/8. The corridor curriculum was designed to fix
   ring *period* error — a value-surface property — and it also moved the
   output-cap pole on that cell.

4. **"The opamp open-loop AC gate is 0/4 at every tier for every family" is
   false — by a wide margin.** Seven cells pass, and they are clean passes with
   un-railed operating points, not threshold-grazing:
   * **BSIM-AR is the strongest here (5 of 16)** — TSMC16 at *every* tier
     (0.54 / 2.62 / 0.33 / 0.97 dB) and **TSMC7 at `large` at 0.12 dB**, the
     tech this project spent two campaigns calling unreachable.
   * DirectNet passes 1 of 16: TSMC16 at `small`
     (1.78 dB / GBW 0.962 / PM err 0.32°, NN OP 0.649 V of a 0.8 V supply).
   * PFN passes TSMC7 at `medium` (0.84 dB).

   It also retires the V6.8.1 reading that **"AC collapses at xl"**: BSIM-AR
   `xl` banks TSMC16 at 0.97 dB, and `tsmc7-opamp-AC` — recorded there as not
   converging *at all* after ~6 h — now completes and returns 3.86 dB. That
   pathology was the railed OP, i.e. the gds bug.

   What remains true is that **no family passes more than half** of the gate,
   and the next subsection shows part of that denominator is unreachable by
   construction.

### The opamp open-loop AC gate has a bias-resolution defect

Worth reading before drawing conclusions from the opamp-AC row.
`verify_complex_opamp_ac` linearizes both sides about their own **peak-gain
bias**, found as `argmax |dVout/dVin|` over a DC sweep with a **2 mV step**. A
two-stage Miller opamp with 33–48 dB of gain has a transition only ~3–14 mV
wide, so that grid samples the transition with a handful of points and the
"peak-gain" sample lands off-centre.

Measured consequence — the **NGSPICE reference's own** output at the bias the
harness picks:

| tech | reference Vout at peak-gain bias | as % of VDD | inside the gate's 15–85 % validity window? |
|---|---|---|---|
| TSMC5 | 0.021 V | **3.2 %** | no |
| TSMC7 | 0.584 V | 77.9 % | yes |
| TSMC12 | 0.722 V | **90.2 %** | no |
| TSMC16 | 0.685 V | **85.6 %** | no (marginally) |

The `op_valid` check (0.15·VDD < vout < 0.85·VDD) is then applied **to the NN
only**. So on three of four techs a model that faithfully reproduces the
reference operating point is scored `FAIL [OP-MISBIAS: NN opamp output railed]`.

The clearest demonstration is BSIM-AR `small` on TSMC5: NN OP 0.042 V against
the reference's 0.021 V, DC-gain error **0.39 dB**, GBW ratio **1.19**, PM error
**14.3°** — inside all three *gated* criteria — scored FAIL purely on
`op_valid`.

**This is a gate-construction defect, not an NN result, and it is not fixed
here** — changing an accuracy gate changes the accuracy record, which is a
deliberate decision, not a side effect of a documentation pass. The fix has two
parts: refine the sweep near the transition (local bisection or a 10× finer
grid) so the peak-gain sample is the actual steepest point, and judge `op_valid`
against the *reference's* operating point rather than an absolute band. Until
then, treat the opamp-AC row as a lower bound, and TSMC5's cell as unreachable
by construction.

## 6. Recommendation by tier

| family | tier | why |
|---|---|---|
| **DirectNet** | **`large`** | Peak of the clean capacity curve *and* the tier the production curriculum was tuned on; 0.92 M params at 1.5 ms/eval. `xl` reaches 16/16 with `crit15m` but costs 2.3× per eval with no device-fidelity gain. |
| **BSIM-AR** | **`medium`** | 16/16 strict at 1.9 M. `large` is the *worst* corridor tier (15/16), and `xl` ties `medium` at 7.7× the parameters, ~11 days of training and collapsed AC. |
| **PFN** | **`small`** | 11/16 at 0.69 M — its capacity curve declines and its `large` tier is optimization-unstable. |

The general law, three times over: **beyond each family's sweet spot, extra
capacity buys a tighter fit to the training distribution and *loses* circuit
fixed points.** Where the families differ is only in where that spot sits.
