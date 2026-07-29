# Accuracy reports

Ground truth is **always** NGSPICE on the *identical* BSIM-CMG (LEVEL=72) OSDI
model — never a simplified or self-defined reference.

## Start here

| file | what it answers |
|---|---|
| **[`methodology.md`](methodology.md)** | What a gate is, the thresholds, the strict-OMP rule, and **which code state produced which number**. Read before comparing any two numbers. |
| [`DirectNet-L73-clean.md`](DirectNet-L73-clean.md) | The production family under one training run — per tech, per scale, per testcase. |
| [`DirectNet-L73-recipes.md`](DirectNet-L73-recipes.md) | Its curriculum arms, the universal-scope study, and the dead ends. |
| [`BSIM-AR-L74-clean.md`](BSIM-AR-L74-clean.md) | The autoregressive Transformer — the reproducible family. |
| [`BSIM-AR-L74-recipes.md`](BSIM-AR-L74-recipes.md) | Its corridor arms, and why the `inv_trip` anchor is inert here. |
| [`PFN-L75-clean.md`](PFN-L75-clean.md) | The in-context research family. |
| [`PFN-L75-recipes.md`](PFN-L75-recipes.md) | Its first curriculum arm, new in V7.3.0. |
| [`archive-pre-gds-fix.md`](archive-pre-gds-fix.md) | The register of retracted claims. |

**Two files per family: one clean, one recipes.** The clean report is the
control — one training run, no addendum — and answers *per tech, per scale, per
testcase* in one place. The recipes report carries the training addenda
measured against that control. Anything cross-cutting — the `gds` fix, the
corridor law, the TSMC6 repeat, the noise floor — lives in `methodology.md` so
it is stated once.

## Scoreboard

| LEVEL | family | role | best clean tier | best recipe | CPU cost |
|---|---|---|---|---|---|
| 73 | **DirectNet** | **production** | `large` **16/20** | `crit15m`@xl **16/16** | 1.5 ms @ `large` |
| 74 | **BSIM-AR** | higher fidelity | `small` **17/20** | `corroft`@medium **16/16** | 61.5 ms @ `medium` |
| 75 | **PFN** | research | `small` **14/20** | `corroft`@small **14/20** | 15.6 ms @ `small` |

Strict = passes at OMP ∈ {1, 2, 4}. Totals are **/16–/20**, against the /20 these reports target (4 circuits × 5 techs): some groups have no TSMC6 checkpoint measured yet. **Compare the fractions, not the counts.**

## The finding that governs how to read everything else

The TSMC6 controlled repeat retrained one recipe on **bit-identical rows** and
compared strict verdicts (`methodology.md` §7, §8.4). Gate *counts* reproduced
at three of four tiers — but **which** cells passed swapped: `ring_osc` carries
**±4 pp** of run-to-run scatter across a **5 %** gate, and `opamp` is
**bimodal** (a 1.8–7.1 % basin, or a 100 % rail). `sram_snm` and `switchcap`
reproduce to ≤0.3 pp and never flip.

**A recipe promoted on a single ring or opamp margin is inside the noise.** The
same claim resting on SRAM or switchcap is not, and neither are the
family-level counts above, which aggregate twenty cells.

Put beside the re-gate result, the picture completes: **the variance is in
training, not in evaluation.** Re-gating *the same weights* is deterministic —
223/223 complex cells agreed across two passes on different days. Retraining
*the same recipe on the same rows* is not. A gate result is a reproducible
property of a checkpoint, not of a recipe.

## Denominators changed in V7.3.0

TSMC6 now counts toward the headline, so complex totals are **/20**, device AC
**/10** and opamp AC **/5**. Every report before V7.3.0 scored /16, /8 and /4
and quoted TSMC6 separately. **No total here is comparable to a total in an
older document without rescaling** — a V7.3.0 "16/20" and a V7.1.0 "13/16" can
be the same measurement. TSMC6 is still TSMC7 relabelled; what changed is the
denominator, not the finding.

## Provenance

Every number carries one of four code states — **pre-fix** (`gds` sign bug
present), **V6.13.0** (fix shipped, complex re-gate), **V7.1.0** (device / AC /
strict re-gate), **V7.3.0** (this campaign). `methodology.md` §6 says what each
is comparable to.

Tables in these files are **generated** from the gate logs, so they cannot
drift from the evidence:

```bash
python scripts/v730_coverage.py            # what is measured, and what is not
python scripts/v730_docs_build.py --check  # fail if any file is stale
```
