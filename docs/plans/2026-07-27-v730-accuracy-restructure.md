# V7.3.0 — accuracy reports: condense, restructure, re-gate

**Status:** IN FLIGHT (started 2026-07-27). Live routing document — update on
every change. Executed history goes to `docs/CHANGELOG.md`, not here.

---

## 1. What is being built

Nine files under `docs/accuracy/` become **two per model family**, plus the
shared contract and an index:

| new file | contents |
|---|---|
| `DirectNet-L73-clean.md` | the clean recipe only: **per tech × per scale × per testcase** |
| `DirectNet-L73-recipes.md` | the surviving recipes, filtered |
| `BSIM-AR-L74-clean.md` | ditto |
| `BSIM-AR-L74-recipes.md` | ditto |
| `PFN-L75-clean.md` | ditto |
| `PFN-L75-recipes.md` | ditto |
| `methodology.md` | kept, condensed — the one place cross-cutting material lives |
| `README.md` | index + scoreboard |

Retired: `by-tech.md`, `by-scale.md`, `by-recipe.md`, and the three
`*-accuracy.md` family reports. `archive-pre-gds-fix.md` shrinks to the
register of retracted claims.

**Why two axes per family rather than three shared pivots.** The V7.1.0
structure sliced by tech / scale / recipe *across* families, which forced every
question to be answered by joining three files. The clean-vs-recipe split is
the one cut that matches how the checkpoints are actually produced: one control
and a set of addenda measured against it.

Cross-cutting material — the `gds` fix, the corridor law, the TSMC6 repeat, the
run-to-run noise floor — stays in `methodology.md`. It must not be triplicated
across the six family files.

## 2. Two decisions taken up front

**TSMC6 folds into the headline.** Every per-tech table carries all five techs
and complex totals become **/20**, device AC **/10**, opamp AC **/5**. This
reverses the V7.1.0 rule (`/16` + a separate `/4` repeat column). TSMC6 remains
TSMC7 relabelled and `methodology.md` keeps saying so; what changes is only the
denominator. **Consequence: no total in the new reports is comparable to a
total in the old ones** without rescaling, and that must be stated wherever a
historical number is quoted.

**Recipes are filtered.** A recipe earns a row only if it is production, best
in its family, or carries a durable law. Everything else drops to a one-line
dead-end table. See §5.

## 3. What was actually stale — measured, not assumed

The audit that opened this campaign found the docs in better shape than the
brief assumed, and one real hole:

* **Complex + device + AC for DirectNet (all 4 tiers) and PFN (small, medium,
  large)** — fresh, all five techs, OMP {1,2,4}, from the V7.1.0 pass. Reused.
* **PFN clean@`xl` on TSMC5/7/12/16** — never gated. A prior session's pool was
  still running it when this campaign started (Wave A).
* **BSIM-AR clean, all four tiers** — complex numbers come from the V6.13.0
  `a3_regate` campaign, single-run for `small`/`medium`/`xl` and strict only
  for `large`. Re-gated (Wave B).
* **BSIM-AR recipe groups** — strict OMP and device/AC coverage are partial.
  Re-gated for the survivors only (Wave C).
* **`by-recipe.md` §4's device-DC-by-recipe table** — pre-fix, and its
  checkpoints (`sob`, `ekv`, `csobekv`, `cor`, `corft`, seeds) no longer exist.
  Cannot be re-run; these are exactly the recipes the filter drops, so the
  table goes rather than gets refreshed.

**Correction to an early finding in this campaign:** the training datasets were
first reported deleted. They are not — all 20 GB sit in
`external_compact_models/bsimar/data/datasets/` (the initial search used too
shallow a `find` depth). Base data exists for all five techs; the ring-only
`corro` corridor exists for TSMC5/7/12/16 and was **missing only for TSMC6**.

## 4. Waves

| wave | work | state (2026-07-27 22:0x) |
|---|---|---|
| **A** | PFN clean@`xl` × TSMC5/7/12/16, full suite (48 jobs) | inherited pool, **40/48** |
| **B** | BSIM-AR clean × 4 tiers × 4 techs — ring+opamp at OMP {1,2,4}, sram+switchcap (128 jobs) | **24/128**, PAR=12 |
| **C** | surviving-recipe gates: 9 BSIM-AR groups, complex strict + device/AC (320 jobs) | **0/320**, PAR=8, longest-job-first so the device sweeps are all in flight |
| **D** | TSMC6 corridor harvest | ✅ **done** — 6762 rows/device in 125 s, and `array_equal` to TSMC7's |
| **E** | PFN corridor `corroft@small` × **5** techs × N/P (10 ckpts) | training, ~10/120 epochs, 3× RTX 4090 at ~95 % |
| **F** | consistency control, final doc generation, retire old files, push | after A–E |

Throughput is set by the shared host, not by the work: loadavg ~1380 on 192
cores, almost all of it other users' Xyce. Wave B is landing ~14 jobs/hour.
Expect A+B complete within a day, C within two, E within one.

**Deliverables already landed:** `scripts/v730_coverage.py` (coverage/gap map
over all three evidence passes), `scripts/v730_docs_build.py` (generates all six
reports + the README scoreboard), the six templates, the rewritten
`methodology.md`, the condensed archive, and the two DirectNet reports — whose
data was already complete, so they are final.

**Held back deliberately:** the rendered BSIM-AR / PFN reports and the README,
because their tables would show partial gate coverage; and the deletion of
`by-tech.md` / `by-scale.md` / `by-recipe.md` / the three `*-accuracy.md`, so
the repo never has a window with no valid accuracy documentation. Both land in
wave F.

All gating goes through `scripts/v710_regate.sh` (resumable: a job whose log
carries `===V710_DONE` is skipped) with `V710_OUT=results/v730_regate` so the
new campaign's provenance stays separate from V7.1.0's.

**Host caveat.** The box is shared and heavily oversubscribed by other users
(loadavg ~1225 on 192 cores, mostly Xyce). Pools run at PAR≈12, not higher;
wall-clock estimates are inflated accordingly and single-job timings are not
comparable with the V7.1.0 campaign's.

## 5. The recipe filter

**Kept.** DirectNet `crit30f@large` (production), `crit15m@xl` (16/16),
`corroft@xl`, `csob@large` (device/AC alternate only). BSIM-AR
`corroft@medium` + `corro15@medium` (both sweep), `corroft`/`crit15m`/`crit30`
@`large`, and the four corridor recipes at `xl`. PFN `corroft@small` — new in
this campaign.

**Dropped to the dead-end table.** `crit10@xl` (withdrawn — fails the very cell
it was documented to hold), `invtrip@large` and `csob@xl` (both land exactly on
clean's failure set — no lever), `sob`, `ekv`, `ekvhi`, `csobekv`, the seed
sweeps, `csobcrit`, `crit30a1`, and full-corridor at `xl`.

**Filtering rule, stated so it can be checked:** a one-cell difference on
`ring_osc` or `opamp` is inside the pipeline's measured run-to-run noise
(±4 pp on a 5 % ring gate; opamp bimodal). A recipe is kept only on a margin
that clears that floor, or on many cells at once.

## 6. Open questions

1. **PFN has no recipe history at all.** Wave E gives its recipes file one real
   row. If `corroft@small` moves PFN's tsmc5/tsmc7 cells, a second tier is
   worth training; if it does not, the file documents a negative result.
2. **The README's "every family is flip-free" claim is already contradicted**
   by `by-scale.md` §3, which records 1 FLIP each for `pfn/medium` and
   `pfn/large`. Resolve against the re-gate rather than restating either.
3. **The opamp-AC bias-resolution defect** (`by-scale.md` §5) is a gate
   construction bug, not a model result. It stays unfixed in this campaign —
   changing a gate changes the accuracy record and is a separate decision — but
   the new reports must carry the caveat wherever an opamp-AC number appears.
