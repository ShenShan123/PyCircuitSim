# Accuracy reports

Ground truth is **always** NGSPICE on the *identical* BSIM-CMG (LEVEL=72) OSDI
model — never a simplified or self-defined reference.

## Start here

| file | what it answers |
|---|---|
| **[`methodology.md`](methodology.md)** | What a gate is, the thresholds, the strict-OMP rule, and **which code state produced which number**. Read before comparing any two numbers. |
| [`DirectNet-L73-clean.md`](DirectNet-L73-clean.md) | The production family under one training run — per tech, per scale, per testcase. |
| [`DirectNet-L73-AnalogGym.md`](DirectNet-L73-AnalogGym.md) | The 2026-08-19 `large` retrain, strict gates, and LEVEL=73 accuracy on the 255-row AnalogGym basket. |
| [`DirectNet-L73-recipes.md`](DirectNet-L73-recipes.md) | Its curriculum arms, the universal-scope study, and the dead ends. |
| [`BSIM-AR-L74-clean.md`](BSIM-AR-L74-clean.md) | The autoregressive Transformer — the reproducible family. |
| [`BSIM-AR-L74-recipes.md`](BSIM-AR-L74-recipes.md) | Its corridor arms, and why the `inv_trip` anchor is inert here. |
| [`PFN-L75-clean.md`](PFN-L75-clean.md) | The in-context research family. |
| [`PFN-L75-recipes.md`](PFN-L75-recipes.md) | Its first curriculum arm, new in V7.3.0. |
| [`archive-pre-gds-fix.md`](archive-pre-gds-fix.md) | The register of retracted claims. |

**Two baseline files per family: one clean, one recipes.** The clean report is
the control — one training run, no addendum — and answers *per tech, per scale,
per testcase* in one place. The recipes report carries the training addenda
measured against that control. Cross-benchmark qualification reports are
indexed separately above. Anything cross-cutting — the `gds` fix, the
corridor law, the TSMC6 repeat, the noise floor — lives in `methodology.md` so
it is stated once.

## Scoreboard

| LEVEL | family | role | current / best clean | historical best recipe | CPU cost |
|---|---|---|---|---|---|
| 73 | **DirectNet** | **production** | V7.4 `large` **14/20** served; `xl` **15/20** best | V7.3 `crit15m`@xl **19/20** | 1.5 ms @ `large` |
| 74 | **BSIM-AR** | higher fidelity | V7.4 `small` **18/20** | V7.3 `corroft`@medium **20/20** | 61.5 ms @ `medium` |
| 75 | **PFN** | research | V7.3 `small` **14/20** | V7.3 `corroft`@small **14/20** | 15.6 ms @ `small` |

Strict = passes at OMP ∈ {1, 2, 4}. Totals are **/20** — 4 circuits × 5 techs, TSMC6 included (`methodology.md` §2). Earlier reports scored /16 over four techs, so a /20 total here and a /16 total there can be the same measurement.

V7.4.0 rebuilt the DirectNet and BSIM-AR clean matrices from scratch on the
new hardware. Recipe columns and both PFN columns are the latest available
evidence but remain V7.3.0 measurements; those recipe deltas belong to their
V7.3.0 controls and are not direct deltas against the V7.4.0 clean rows.
The 2026-08-19 DirectNet retrain is reported separately because it was scored
at the one-thread production contract but not repeated at OMP ∈ {2, 4}; it
therefore does not replace the strict scoreboard above.

## The finding that governs how to read everything else

The TSMC6 controlled repeat retrained one recipe on **bit-identical rows** and
compared strict verdicts (`methodology.md` §7, §8.4). Gate *counts* reproduced
at three of four tiers — but **which** cells passed swapped: `ring_osc` carries
**±4 pp** of run-to-run scatter across a **5 %** gate, and `opamp` is
**bimodal** (a 1.8–7.1 % basin, or a 100 % rail). `sram_snm` and `switchcap`
reproduce to ≤0.3 pp and never flip.

The fresh V7.4 clean controls confirm the family dependence: BSIM-AR repeats
15/16 TSMC6/TSMC7 verdicts (one ring split), while DirectNet repeats 14/16
(two opamp splits). PFN's latest repeat remains V7.3 at 10/12.

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

Every number carries one of five code states — **pre-fix** (`gds` sign bug
present), **V6.13.0** (fix shipped, complex re-gate), **V7.1.0** (device / AC /
strict re-gate), **V7.3.0** (five-tech recipes and PFN), or **V7.4.0**
(new-hardware DirectNet/BSIM-AR clean rebuild). `methodology.md` §6 says what
each is comparable to. GPU acceleration is a separate fidelity axis and never
replaces the CPU-pinned accuracy scoreboard.

The V7.4 GPU axis is closed: the 48-run T3 bundle reproduces DirectNet
clean-`large` at 12/16 strict with both binding suites 24/24 and zero flips;
the full-bundle T4 latch gate is 8/8 with zero basin flips. See
[`DirectNet-L73-clean.md`](DirectNet-L73-clean.md) §8.

Tables in these files are **generated** from the gate logs, so they cannot
drift from the evidence:

```bash
python scripts/v730_coverage.py --tag dn --set clean --require-complete
python scripts/v730_coverage.py --tag tf --set clean --require-complete
python scripts/v730_docs_build.py --check
```

The V7.3.0 raw recipe/PFN trees were not copied to the new machine. Their
rendered reports are the durable evidence. The builder pins every report to
one complete campaign pass; when that source is incomplete locally, it accepts
the preserved report only if its committed SHA-256 matches. Partial evidence
therefore cannot mix campaign passes or replace measured historical tables.
