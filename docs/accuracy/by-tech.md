# Accuracy by technology — TSMC5 / 7 / 12 / 16, and TSMC6

Cross-family, cross-scale view along the **technology** axis. Companion pivots:
[`by-scale.md`](by-scale.md) (capacity), [`by-recipe.md`](by-recipe.md)
(training recipe). Gate definitions and the code-state ladder:
[`methodology.md`](methodology.md).

---

## 1. The technologies

| tech | VDD | VT variants | local vocab | gate VT | what makes it hard |
|---|---|---|---|---|---|
| **TSMC5** | **0.65 V** | svt, lvt, ulvt, elvt | 5 (4 + UNKNOWN) | lvt | Lowest supply, steepest transfer curve. The ring's switching edge sits where the NN under-drives, and the opamp bias is the most fragile of the four. |
| **TSMC7** | **0.75 V** | svt, lvt, ulvt | 4 (3 + UNKNOWN) | ulvt | The other steep low-VDD tech. Carries the single hardest cell in the project (`tsmc7-opamp` at DirectNet `large`) and the worst device-DC NMOS surface of the four. |
| **TSMC12** | 0.80 V | svt, lvt, ulvt, hvt, lnvt | 6 (5 + UNKNOWN) | svt | Comfortable on rings and opamps; its weak cell is **switchcap** (the charge/off-state surface), which is where PFN loses it. Off-grid NFIN=10 also probes its 6→21 sampling gap. |
| **TSMC16** | 0.80 V | svt, lvt, ulvt, hvt, lnvt | 6 (5 + UNKNOWN) | svt | The easiest of the four on rings; its opamp basin is tier-dependent and was the last cell production banked (V6.13.0, 7.69 %). |
| **TSMC6** ⚠ | 0.75 V | svt, lvt, ulvt | 4 (3 + UNKNOWN) | ulvt | **Not a fifth technology — TSMC7 relabelled** (§5). Carried as a deliberate repeat experiment, scored in its own /4 column and never folded into the /16. |

The split that matters is **VDD**: TSMC5/7 at 0.65–0.75 V behave as one class and
TSMC12/16 at 0.80 V as another, on every axis in this file. TSMC6 sits, by
construction, exactly on top of TSMC7.

## 2. Which cells are actually hard

Every post-fix single-run verdict, over the 27 checkpoint groups re-gated in
V6.13.0 (all three families, every recipe and tier on disk). TSMC6 is excluded
here — it has no post-fix groups yet, and including a duplicate of TSMC7 would
bias the census (§5).


| tech | groups | ring_osc | opamp | sram_snm | switchcap | all cells |
|---|---|---|---|---|---|---|
| **TSMC5** | 27 | 13/27 | 20/27 | 27/27 | 27/27 | **87/108** |
| **TSMC7** | 27 | 14/27 | 18/27 | 27/27 | 26/27 | **85/108** |
| **TSMC12** | 27 | 27/27 | 23/27 | 27/27 | 23/27 | **100/108** |
| **TSMC16** | 27 | 27/27 | 21/27 | 27/27 | 25/27 | **100/108** |
| **TSMC6** ⚠ | 12 | 3/12 | 5/12 | 12/12 | 11/12 | 31/48 |

⚠ TSMC6 is TSMC7 relabelled (§5) and never enters a /16 total. Its 12 groups are **clean-only**, where the other columns' 27 include the curriculum recipes that close rings — so do not compare its ring and opamp fractions with theirs. The like-for-like comparison, clean against clean at matching tiers, is §5.

Read column-wise:

* **`sram_snm` is solved.** 27/27 in every tech — 108/108 cells across every
  family, recipe and tier.
  (Caveat: the `force_ic` half of that probe is currently a no-op —
  `methodology.md` §8.2 — but the butterfly/positivity half is real.)
* **Rings are a VDD story, not a tech story.** TSMC12 and TSMC16 pass 27/27;
  TSMC5 and TSMC7 pass roughly half — and **every single ring failure, in all
  three families, is a recipe without the corridor**. The converse is nearly
  exact too: of the 27 ring passes on TSMC5/7, exactly one comes from a
  non-corridor recipe (DirectNet `v660clean@large` on TSMC7, 4.82 % against a
  5 % gate). See `by-recipe.md` §3.
* **Opamps are the hard class everywhere** (18–23 of 27) but no longer
  concentrated on one tech — that concentration was the gds bug
  (`methodology.md` §6).
* **Switchcap has only 7 failures in 108 cells, and they are two different
  faults.** Three are PFN's TSMC12 **charge-half** near-misses (5.10–5.32 %
  against a 5.0 % gate — the closest open cells in the project). The other four
  are **droop-half** failures whose charge error is comfortably inside gate:
  DirectNet `clean@small` on TSMC7 / TSMC12 / TSMC16 (TSMC12: charge 4.09 %,
  droop 1.036 mV = 130 % of a 0.800 mV allowance) and PFN `large` on TSMC16
  (4.99 %). TSMC5 never fails a switchcap cell.

## 3. Per-tech detail — every re-gated checkpoint group

Post-fix (V6.13.0, `d2ea720`), single-run OMP=1, verdict = gate exit code.
The number is the gate's headline metric: ring period error %, opamp DC-gain
error %, worst SRAM lobe NRMSE %, switchcap charge error % of VDD. Reproduced
verbatim from `results/a3_regate/REPORT.md`.


### TSMC5

**TSMC5 in one line:** the ring is the wall — 13/27 groups pass it, and every
pass has a corridor in its recipe. The opamp is *reachable* at every capacity
(DirectNet `crit30f@large` 9.54 %, `crit15m@xl` 3.41 %, BSIM-AR 2.4–4.2 % at
every tier), but DirectNet's 9.54 % production margin against a 10 % gate is the
thinnest number in the whole matrix. SRAM and switchcap never fail.

| checkpoint group | ring_osc | opamp | sram_snm | switchcap |
|---|---|---|---|---|
| `dn/clean/small` | FAIL 8.06% | PASS 0.88% | PASS 2.85% | PASS 1.72% |
| `dn/clean/medium` | FAIL 5.89% | FAIL 100.00% | PASS 0.81% | PASS 1.54% |
| `dn/v660clean/large` | FAIL 11.47% | PASS 5.08% | PASS 1.55% | PASS 3.48% |
| `dn/crit30f/large` | PASS 4.04% | PASS 9.54% | PASS 0.90% | PASS 2.04% |
| `dn/csob/large` | FAIL 10.31% | FAIL 100.00% | PASS 1.04% | PASS 2.90% |
| `dn/clean/xl` | FAIL 13.11% | FAIL 100.00% | PASS 1.39% | PASS 3.18% |
| `dn/corroft/xl` | PASS 4.05% | FAIL 100.00% | PASS 0.89% | PASS 1.66% |
| `dn/crit10/xl` | PASS 4.17% | FAIL 100.00% | PASS 0.92% | PASS 2.20% |
| `dn/crit15m/xl` | PASS 4.05% | PASS 3.41% | PASS 0.92% | PASS 2.25% |
| `tf/clean/small` | FAIL 6.53% | PASS 3.07% | PASS 1.26% | PASS 2.05% |
| `tf/clean/medium` | FAIL 5.55% | PASS 1.48% | PASS 1.33% | PASS 2.11% |
| `tf/clean/large` | FAIL 7.38% | PASS 3.00% | PASS 0.99% | PASS 2.46% |
| `tf/clean/xl` | FAIL 7.61% | PASS 2.73% | PASS 0.97% | PASS 2.57% |
| `tf/corroft/medium` | PASS 3.33% | PASS 2.52% | PASS 0.73% | PASS 1.99% |
| `tf/corro15/medium` | PASS 3.40% | PASS 2.38% | PASS 0.71% | PASS 2.01% |
| `tf/corroft/large` | PASS 3.88% | PASS 2.89% | PASS 0.77% | PASS 1.69% |
| `tf/crit15m/large` | PASS 3.93% | PASS 3.68% | PASS 0.78% | PASS 1.74% |
| `tf/crit30/large` | PASS 3.85% | PASS 3.14% | PASS 0.78% | PASS 1.68% |
| `tf/invtrip/large` | FAIL 6.84% | PASS 2.76% | PASS 1.25% | PASS 2.24% |
| `tf/corroft/xl` | PASS 3.83% | PASS 4.06% | PASS 0.78% | PASS 1.74% |
| `tf/crit15m/xl` | PASS 3.87% | PASS 3.76% | PASS 0.78% | PASS 1.74% |
| `tf/crit30/xl` | PASS 3.88% | PASS 4.17% | PASS 0.79% | PASS 1.75% |
| `tf/corro15/xl` | PASS 3.86% | PASS 3.51% | PASS 0.77% | PASS 1.74% |
| `tf/csob/xl` | FAIL 7.28% | PASS 3.57% | PASS 1.02% | PASS 2.52% |
| `pfn/clean/small` | FAIL 8.15% | FAIL 100.00% | PASS 1.83% | PASS 2.14% |
| `pfn/clean/medium` | FAIL 8.72% | FAIL 100.00% | PASS 1.06% | PASS 2.77% |
| `pfn/clean/large` | FAIL 9.06% | PASS 6.73% | PASS 1.27% | PASS 2.84% |

### TSMC7

**TSMC7 in one line:** the only tech where a cell is open at production —
`tsmc7-opamp` fails at DirectNet `large` and passes at DirectNet `small`
(1.81 %), DirectNet `xl` (4.20 %) and *every* BSIM-AR tier (0.55–7.3 %), so it
is a basin property of that one tier, not a fidelity wall. Its ring behaves like
TSMC5's. It also owns the worst device-DC surface: BSIM-AR's tsmc7-NMOS NRMSE
*grows* with capacity (3.37 → 4.07 → 4.77 % small→large).

| checkpoint group | ring_osc | opamp | sram_snm | switchcap |
|---|---|---|---|---|
| `dn/clean/small` | FAIL 5.94% | PASS 1.81% | PASS 1.91% | FAIL 2.34% |
| `dn/clean/medium` | FAIL 10.86% | FAIL 100.00% | PASS 2.22% | PASS 2.81% |
| `dn/v660clean/large` | PASS 4.82% | FAIL 100.00% | PASS 1.28% | PASS 2.45% |
| `dn/crit30f/large` | PASS 2.40% | FAIL 100.00% | PASS 0.98% | PASS 2.17% |
| `dn/csob/large` | FAIL 5.09% | FAIL 100.00% | PASS 1.26% | PASS 2.42% |
| `dn/clean/xl` | FAIL 13.59% | PASS 4.20% | PASS 1.92% | PASS 2.67% |
| `dn/corroft/xl` | PASS 2.42% | PASS 6.90% | PASS 1.00% | PASS 2.26% |
| `dn/crit10/xl` | PASS 2.54% | PASS 7.91% | PASS 1.02% | PASS 2.29% |
| `dn/crit15m/xl` | PASS 2.42% | PASS 7.56% | PASS 1.02% | PASS 2.27% |
| `tf/clean/small` | FAIL 5.97% | PASS 0.55% | PASS 1.75% | PASS 2.46% |
| `tf/clean/medium` | FAIL 7.41% | PASS 4.12% | PASS 1.26% | PASS 2.62% |
| `tf/clean/large` | FAIL 8.63% | PASS 5.39% | PASS 1.47% | PASS 2.64% |
| `tf/clean/xl` | FAIL 12.55% | PASS 4.26% | PASS 1.95% | PASS 2.73% |
| `tf/corroft/medium` | PASS 2.25% | PASS 6.73% | PASS 1.08% | PASS 2.33% |
| `tf/corro15/medium` | PASS 2.31% | PASS 6.81% | PASS 1.09% | PASS 2.35% |
| `tf/corroft/large` | PASS 2.31% | FAIL 100.00% | PASS 1.01% | PASS 2.27% |
| `tf/crit15m/large` | PASS 2.34% | FAIL 100.00% | PASS 1.00% | PASS 2.26% |
| `tf/crit30/large` | PASS 2.32% | FAIL 100.00% | PASS 1.00% | PASS 2.25% |
| `tf/invtrip/large` | FAIL 6.98% | PASS 5.21% | PASS 1.28% | PASS 2.51% |
| `tf/corroft/xl` | PASS 2.21% | PASS 7.21% | PASS 1.03% | PASS 2.23% |
| `tf/crit15m/xl` | PASS 1.91% | PASS 7.06% | PASS 1.07% | PASS 2.22% |
| `tf/crit30/xl` | PASS 2.24% | PASS 7.07% | PASS 1.03% | PASS 2.24% |
| `tf/corro15/xl` | PASS 1.98% | PASS 7.27% | PASS 1.07% | PASS 2.21% |
| `tf/csob/xl` | FAIL 12.06% | PASS 4.65% | PASS 1.83% | PASS 2.72% |
| `pfn/clean/small` | FAIL 9.72% | FAIL 100.00% | PASS 1.54% | PASS 3.07% |
| `pfn/clean/medium` | FAIL 8.92% | PASS 5.45% | PASS 1.03% | PASS 2.59% |
| `pfn/clean/large` | FAIL 8.32% | FAIL 100.00% | PASS 1.35% | PASS 3.08% |

### TSMC12

**TSMC12 in one line:** rings are free (27/27) and opamps are the most reliable
of the four (23/27), but it owns the **switchcap** weakness — PFN misses it at
all three tiers by 0.1–0.3 pp, which is the single closest open cell in the
project. It is also the tech whose off-grid NFIN=10 case exposes PFN's
context-anchored geometry interpolation (`PFN-L75-accuracy.md` §3).

| checkpoint group | ring_osc | opamp | sram_snm | switchcap |
|---|---|---|---|---|
| `dn/clean/small` | PASS 1.95% | FAIL 100.00% | PASS 1.51% | FAIL 4.09% |
| `dn/clean/medium` | PASS 2.26% | FAIL 100.00% | PASS 0.94% | PASS 4.19% |
| `dn/v660clean/large` | PASS 2.14% | PASS 6.26% | PASS 0.89% | PASS 4.14% |
| `dn/crit30f/large` | PASS 2.68% | PASS 6.26% | PASS 0.89% | PASS 4.17% |
| `dn/csob/large` | PASS 2.12% | PASS 5.84% | PASS 0.92% | PASS 4.08% |
| `dn/clean/xl` | PASS 2.18% | FAIL 100.00% | PASS 0.91% | PASS 4.19% |
| `dn/corroft/xl` | PASS 2.68% | PASS 6.25% | PASS 0.91% | PASS 4.22% |
| `dn/crit10/xl` | PASS 3.49% | PASS 6.22% | PASS 0.91% | PASS 4.22% |
| `dn/crit15m/xl` | PASS 2.68% | PASS 6.23% | PASS 0.91% | PASS 4.22% |
| `tf/clean/small` | PASS 1.92% | PASS 8.53% | PASS 0.95% | PASS 4.50% |
| `tf/clean/medium` | PASS 1.52% | PASS 4.78% | PASS 1.12% | PASS 4.22% |
| `tf/clean/large` | PASS 1.54% | PASS 5.81% | PASS 0.87% | PASS 4.15% |
| `tf/clean/xl` | PASS 1.98% | PASS 5.81% | PASS 0.89% | PASS 4.19% |
| `tf/corroft/medium` | PASS 2.13% | PASS 5.32% | PASS 1.01% | PASS 4.15% |
| `tf/corro15/medium` | PASS 2.16% | PASS 5.29% | PASS 1.03% | PASS 4.12% |
| `tf/corroft/large` | PASS 2.08% | PASS 5.69% | PASS 0.86% | PASS 4.14% |
| `tf/crit15m/large` | PASS 2.07% | PASS 5.63% | PASS 0.86% | PASS 4.15% |
| `tf/crit30/large` | PASS 2.08% | PASS 5.64% | PASS 0.86% | PASS 4.14% |
| `tf/invtrip/large` | PASS 1.46% | PASS 6.11% | PASS 0.90% | PASS 4.17% |
| `tf/corroft/xl` | PASS 2.57% | PASS 5.91% | PASS 0.88% | PASS 4.25% |
| `tf/crit15m/xl` | PASS 2.54% | PASS 6.08% | PASS 0.89% | PASS 4.25% |
| `tf/crit30/xl` | PASS 2.55% | PASS 5.85% | PASS 0.87% | PASS 4.25% |
| `tf/corro15/xl` | PASS 2.53% | PASS 5.98% | PASS 0.88% | PASS 4.25% |
| `tf/csob/xl` | PASS 1.90% | PASS 5.72% | PASS 0.89% | PASS 4.15% |
| `pfn/clean/small` | PASS 3.89% | PASS 6.23% | PASS 0.89% | FAIL 5.32% |
| `pfn/clean/medium` | PASS 2.32% | PASS 5.33% | PASS 0.90% | FAIL 5.10% |
| `pfn/clean/large` | PASS 2.97% | FAIL 100.00% | PASS 0.92% | FAIL 5.32% |

### TSMC16

**TSMC16 in one line:** the easiest tech on rings (27/27, margins 1.5–3.2 %) and
the cell production banked last — `tsmc16-opamp` closed in V6.13.0 at 7.69 %
strict, after being the cell that every pre-fix recipe study fought over. Two
alternates (`csob@large`, `crit10@xl`) were documented *because* they held this
basin and both now fail it, which is the cleanest illustration of how much the
gds bug distorted recipe rankings.

| checkpoint group | ring_osc | opamp | sram_snm | switchcap |
|---|---|---|---|---|
| `dn/clean/small` | PASS 1.47% | PASS 6.60% | PASS 1.01% | FAIL 2.76% |
| `dn/clean/medium` | PASS 2.22% | FAIL 100.00% | PASS 0.91% | PASS 3.22% |
| `dn/v660clean/large` | PASS 2.23% | FAIL 100.00% | PASS 0.91% | PASS 3.32% |
| `dn/crit30f/large` | PASS 2.78% | PASS 7.69% | PASS 0.87% | PASS 3.32% |
| `dn/csob/large` | PASS 2.18% | FAIL 100.00% | PASS 0.90% | PASS 3.35% |
| `dn/clean/xl` | PASS 3.17% | PASS 6.24% | PASS 0.91% | PASS 3.42% |
| `dn/corroft/xl` | PASS 2.76% | PASS 6.69% | PASS 0.90% | PASS 3.48% |
| `dn/crit10/xl` | PASS 2.89% | FAIL 100.00% | PASS 0.90% | PASS 3.47% |
| `dn/crit15m/xl` | PASS 2.76% | PASS 6.48% | PASS 0.90% | PASS 3.48% |
| `tf/clean/small` | PASS 2.06% | PASS 6.11% | PASS 1.56% | PASS 3.68% |
| `tf/clean/medium` | PASS 1.59% | PASS 6.79% | PASS 1.00% | PASS 3.43% |
| `tf/clean/large` | PASS 1.92% | PASS 5.74% | PASS 0.87% | PASS 3.33% |
| `tf/clean/xl` | PASS 2.19% | PASS 5.87% | PASS 0.97% | PASS 3.36% |
| `tf/corroft/medium` | PASS 2.19% | PASS 5.82% | PASS 1.02% | PASS 3.36% |
| `tf/corro15/medium` | PASS 2.13% | PASS 6.08% | PASS 1.01% | PASS 3.38% |
| `tf/corroft/large` | PASS 2.53% | PASS 5.28% | PASS 0.84% | PASS 3.36% |
| `tf/crit15m/large` | PASS 2.52% | PASS 5.39% | PASS 0.84% | PASS 3.37% |
| `tf/crit30/large` | PASS 2.54% | PASS 5.56% | PASS 0.85% | PASS 3.38% |
| `tf/invtrip/large` | PASS 1.63% | PASS 7.00% | PASS 0.91% | PASS 3.35% |
| `tf/corroft/xl` | PASS 2.77% | PASS 6.00% | PASS 0.96% | PASS 3.39% |
| `tf/crit15m/xl` | PASS 2.76% | PASS 5.81% | PASS 0.97% | PASS 3.38% |
| `tf/crit30/xl` | PASS 2.75% | PASS 5.83% | PASS 0.97% | PASS 3.37% |
| `tf/corro15/xl` | PASS 2.77% | PASS 5.92% | PASS 0.98% | PASS 3.39% |
| `tf/csob/xl` | PASS 1.99% | PASS 5.57% | PASS 1.00% | PASS 3.35% |
| `pfn/clean/small` | PASS 2.78% | PASS 6.53% | PASS 1.00% | PASS 3.80% |
| `pfn/clean/medium` | PASS 3.36% | FAIL 100.00% | PASS 0.94% | PASS 4.19% |
| `pfn/clean/large` | PASS 2.87% | FAIL 100.00% | PASS 0.84% | FAIL 4.99% |


### TSMC6 ⚠

**TSMC6 in one line:** not a fifth technology — TSMC7 relabelled (§5), carried as
a *repeat* of it. Only the clean recipe exists here, at all four scales for all
three families, so this matrix has 12 groups where the others have 27; read it
against TSMC7's clean rows, never against its curriculum rows.

| checkpoint group | ring_osc | opamp | sram_snm | switchcap |
|---|---|---|---|---|
| `dn/clean/small` | PASS 4.32% | FAIL 11.15% | PASS 1.68% | FAIL 2.27% |
| `dn/clean/medium` | FAIL 9.37% | FAIL 100.00% | PASS 2.10% | PASS 2.75% |
| `dn/clean/large` | FAIL 9.04% | PASS 7.12% | PASS 1.58% | PASS 2.52% |
| `dn/clean/xl` | FAIL 15.05% | FAIL 100.00% | PASS 1.86% | PASS 2.65% |
| `tf/clean/small` | FAIL 6.14% | PASS 4.49% | PASS 1.40% | PASS 2.72% |
| `tf/clean/medium` | FAIL 6.66% | PASS 5.29% | PASS 1.70% | PASS 2.60% |
| `tf/clean/large` | FAIL 11.61% | PASS 4.72% | PASS 2.32% | PASS 2.69% |
| `tf/clean/xl` | FAIL 11.99% | PASS 4.21% | PASS 1.95% | PASS 2.74% |
| `pfn/clean/small` | PASS 4.83% | FAIL 100.00% | PASS 1.59% | PASS 2.64% |
| `pfn/clean/medium` | FAIL 7.21% | FAIL 100.00% | PASS 1.77% | PASS 3.74% |
| `pfn/clean/large` | FAIL 9.12% | FAIL 100.00% | PASS 1.88% | PASS 2.83% |
| `pfn/clean/xl` | PASS 0.08% | FAIL 100.00% | PASS 1.69% | PASS 2.65% |

## 4. Device-level fidelity by technology

**Parametric DC — mean Id-Vgs NRMSE % per tech (config fails in brackets)**

| family / tier | TSMC5 | TSMC7 | TSMC12 | TSMC16 | TSMC6 |
|---|---|---|---|---|---|
| DirectNet `small` | 2.39 | 1.81 | 0.57 | 0.62 | 2.06 |
| DirectNet `medium` | 1.48 | 1.46 | 0.56 | 0.58 | 2.21 |
| DirectNet `large` | 1.91 | 1.21 | 1.69 (17/18) | 1.01 | 2.25 |
| DirectNet `xl` | 2.91 | 2.35 | 2.73 (16/18) | 1.52 | 3.18 (13/14) |
| DirectNet `v660clean_large` | 2.60 | 1.31 | 1.72 (17/18) | 0.93 | — |
| DirectNet `csob_large` | 2.29 | 1.58 | 0.43 | 1.15 | — |
| DirectNet `corroft_xl` | 2.39 | 1.52 | — | 1.39 | — |
| DirectNet `crit10_xl` | 2.25 | 1.40 | 2.73 (16/18) | 1.49 | — |
| DirectNet `crit15m_xl` | 2.35 | 1.43 | 2.75 (16/18) | 1.48 | — |
| BSIM-AR `small` | 2.54 | 1.24 | 0.82 | 1.61 (13/14) | 2.38 |
| BSIM-AR `medium` | 1.77 | 1.46 | 1.34 (17/18) | 1.57 (13/14) | 2.24 (13/14) |
| BSIM-AR `large` | 1.80 | 1.21 | 1.67 (16/18) | 1.58 (13/14) | 2.93 (13/14) |
| BSIM-AR `xl` | 1.94 | 2.92 | 1.08 | 1.07 | 3.16 |
| BSIM-AR `corroft_medium` | — | 1.00 | 1.15 (17/18) | 1.54 (13/14) | — |
| PFN `small` | 2.14 | 1.58 | 0.56 | 1.12 (13/14) | 1.46 |
| PFN `medium` | 2.36 | 1.62 | 1.95 (17/18) | 2.65 (13/14) | 2.28 (13/14) |
| PFN `large` | 2.65 | 1.52 | 1.04 | 1.10 | 1.93 |
| PFN `xl` | — | — | — | — | 2.53 |

**Parametric transient — mean NRMSE % per tech**

| family / tier | TSMC5 | TSMC7 | TSMC12 | TSMC16 | TSMC6 |
|---|---|---|---|---|---|
| DirectNet `small` | 2.99 | 1.53 | 2.07 | 1.62 | 1.57 |
| DirectNet `medium` | 1.90 | 1.48 | 1.52 | 1.47 | 1.45 |
| DirectNet `large` | 1.67 | 1.46 | 1.49 | 1.47 | 1.45 |
| DirectNet `xl` | 1.66 | 1.45 | 1.52 | 1.48 | 1.46 |
| DirectNet `v660clean_large` | 1.68 | 1.46 | 1.50 | 1.47 | — |
| DirectNet `csob_large` | 1.69 | 1.46 | 1.50 | 1.47 | — |
| DirectNet `corroft_xl` | 1.68 | 1.46 | — | 1.46 | — |
| DirectNet `crit10_xl` | 1.66 | 1.45 | 1.49 | 1.48 | — |
| DirectNet `crit15m_xl` | 1.67 | 1.45 | 1.50 | 1.48 | — |
| BSIM-AR `small` | 2.54 | 1.47 | 1.53 | 1.60 | 1.48 |
| BSIM-AR `medium` | 1.80 | 1.52 | 1.52 | 1.50 | 1.50 |
| BSIM-AR `large` | 1.66 | 1.48 | 1.51 | 1.49 | 1.50 |
| BSIM-AR `xl` | 1.62 | 1.48 | 1.51 | — | 1.48 |
| BSIM-AR `corroft_medium` | — | 1.53 | 1.50 | 1.49 | — |
| PFN `small` | 1.88 | 1.44 | 1.50 | 1.48 | 1.43 |
| PFN `medium` | 1.71 | 1.48 | 1.50 | 1.49 | 1.53 |
| PFN `large` | 2.23 | 1.49 | 1.50 | 1.51 | 1.43 |
| PFN `xl` | — | — | — | — | 1.56 |

**Device CS-amp AC — NMOS / PMOS verdicts**

| family / tier | TSMC5 | TSMC7 | TSMC12 | TSMC16 | TSMC6 |
|---|---|---|---|---|---|
| DirectNet `small` | ✓ / ✓ | ✗ gain 2.026 dB, mag 24.89 % / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |
| DirectNet `medium` | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |
| DirectNet `large` | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |
| DirectNet `xl` | ✗ f3db 2.51 / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |
| DirectNet `v660clean_large` | ✗ f3db 1.78 / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | — |
| DirectNet `csob_large` | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | — |
| DirectNet `corroft_xl` | ✓ / ✓ | ✓ / ✓ | ✗ f3db nan / ✗ f3db nan | ✓ / ✓ | — |
| DirectNet `crit10_xl` | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | — |
| DirectNet `crit15m_xl` | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | — |
| BSIM-AR `small` | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |
| BSIM-AR `medium` | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |
| BSIM-AR `large` | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |
| BSIM-AR `xl` | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |
| BSIM-AR `corroft_medium` | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | — |
| PFN `small` | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |
| PFN `medium` | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |
| PFN `large` | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ |
| PFN `xl` | — | — | — | — | ✓ / ✓ |

## 5. TSMC6 — the controlled repeat

**TSMC6 is TSMC7 relabelled.** That is settled, and restoring it does not
reopen it: `tsmc6_{nmos,pmos}.npz` are `array_equal` to `tsmc7_*` over
1.8 M / 2.2 M rows, every differing PDK key is a TSMC TMI extension with zero
occurrences in the BSIM-CMG Verilog-A, and two LEVEL=72 Id-Vgs sweeps match to
the last printed digit (`methodology.md` §7).

It was deleted in V6.13.0 for exactly that reason, and **restored in V7.1.0 by
explicit decision** — because a duplicate technology is the one thing this
project has never been able to buy any other way: **a controlled repeat.** Same
data, same recipe, same code, different training run. Every "recipe A beats
recipe B by one cell" claim in this repo has an unmeasured run-to-run variance
underneath it, and TSMC6 is the only instrument that measures it.

`assert_tech_is_distinct()` still flags the collision; `tsmc6`↔`tsmc7` is the
sole entry in `ACKNOWLEDGED_DUPLICATE_TECHS`, so the guard prints loudly and
continues instead of raising. TSMC6 holds tail codes 22-24, so its presence
renumbers nothing.

**Scoring rule: TSMC6 is a /4 column of its own, never part of the /16.**
Folding a duplicate into the headline denominator would inflate every total.

The V6.11.0 run against the V6.13.0 re-gate of TSMC7 is the sharpest evidence in
the project that pre-fix rankings were unreliable: the two disagreed by **68.2 %
vs 2.0 %** on SRAM SNM error at NFIN=2, which the project had been reading as
per-tech difficulty. After the gds fix the same comparison lands at 5.2 % vs
6.2 % — a 66.2 pp gap collapsing to **1.0 pp**. *Part of the "training lottery"
variance that recipes were ranked against was the wrong Jacobian, not
stochasticity.*

The other casualty is the recorded claim *"BSIM-AR is the only family to bank
the tsmc6 opamp (9.83 %) while tsmc7-opamp is the universal ceiling"*: 9.83 % is
the *same number* as BSIM-AR's tsmc7-opamp clean-medium pass, because it was the
same measurement. PFN, by contrast, read flat 2/4 across all three sizes on the
duplicated data, agreeing with its TSMC7 rows — the reassuring case, and the
family whose gates were already flip-free.

Both runs are the **clean** recipe at the matching tier, on identical rows, with
the same code — so every difference below is run-to-run variance of the whole
pipeline (weight init, GPU nondeterminism, then NR basin selection), not tech
fidelity. Verdicts are strict (OMP ∈ {1,2,4}) for opamp and ring_osc.

**The complex gates are genuinely controlled; the device suites are not.**
Verified against the harness config: for the four complex circuits TSMC6 and
TSMC7 are gated identically — same VT (`ulvt`), VDD 0.75 V, L = 16 n / 20 n,
NFIN 2, TFIN 6 n, same model cards. The **device** suites are not comparable,
because the two profiles differ: TSMC6 carries `l_values = [16, 20, 24] nm` and
all three VT pairs, while TSMC7's SVT/LVT bins were empirically pruned to
`[20, 24] nm` + `ulvt` only. `verify_nn_multi_tech_dc` therefore runs **14
configs for TSMC6 against 9 for TSMC7**, and TSMC6's set contains the harder
unpruned bins. TSMC6's higher device NRMSE (2.06 / 2.21 / 2.25 / 3.18 % by tier,
DC 14/14 · 14/14 · 14/14 · 13/14, transient 16/16 throughout, device AC 2/2 at
every tier) is that config difference, **not** run-to-run scatter — do not read
it as variance.

| run | /4 | ring_osc | opamp | sram_snm | switchcap |
|---|---|---|---|---|---|
| DirectNet `small` | **2/4** | PASS 4.32% | FAIL 11.15% | PASS 1.68% | FAIL 2.27% |
| ↳ TSMC7 same tier | 2/4 | FAIL 5.94% | PASS 1.81% | PASS 1.91% | FAIL 2.34% |
| DirectNet `medium` | **2/4** | FAIL 9.37% | FAIL 100.00% | PASS 2.10% | PASS 2.75% |
| ↳ TSMC7 same tier | 2/4 | FAIL 10.86% | FAIL 100.00% | PASS 2.22% | PASS 2.81% |
| DirectNet `large` | **3/4** | FAIL 9.04% | PASS 7.12% | PASS 1.58% | PASS 2.52% |
| ↳ TSMC7 same tier | 3/4 | PASS 4.82% | FAIL 100.00% | PASS 1.28% | PASS 2.45% |
| DirectNet `xl` | **2/4** | FAIL 15.05% | FAIL 100.00% | PASS 1.86% | PASS 2.65% |
| ↳ TSMC7 same tier | 3/4 | FAIL 13.59% | PASS 4.20% | PASS 1.92% | PASS 2.67% |
| BSIM-AR `small` | **3/4** | FAIL 6.14% | PASS 4.49% | PASS 1.40% | PASS 2.72% |
| ↳ TSMC7 same tier | 3/4 | FAIL 5.97% | PASS 0.55% | PASS 1.75% | PASS 2.46% |
| BSIM-AR `medium` | **3/4** | FAIL 6.66% | PASS 5.29% | PASS 1.70% | PASS 2.60% |
| ↳ TSMC7 same tier | 3/4 | FAIL 7.41% | PASS 4.12% | PASS 1.26% | PASS 2.62% |
| BSIM-AR `large` | **3/4** | FAIL 11.61% | PASS 4.72% | PASS 2.32% | PASS 2.69% |
| ↳ TSMC7 same tier | 3/4 | FAIL 8.63% | PASS 5.39% | PASS 1.47% | PASS 2.64% |
| BSIM-AR `xl` | **3/4** | FAIL 11.99% | PASS 4.21% | PASS 1.95% | PASS 2.74% |
| ↳ TSMC7 same tier | 3/4 | FAIL 12.55% | PASS 4.26% | PASS 1.95% | PASS 2.73% |
| PFN `small` | **3/4** | PASS 4.83% | FAIL 100.00% | PASS 1.59% | PASS 2.64% |
| ↳ TSMC7 same tier | 2/4 | FAIL 9.72% | FAIL 100.00% | PASS 1.54% | PASS 3.07% |
| PFN `medium` | **2/4** | FAIL 7.21% | FAIL 100.00% | PASS 1.77% | PASS 3.74% |
| ↳ TSMC7 same tier | 3/4 | FAIL 8.92% | PASS 5.45% | PASS 1.03% | PASS 2.59% |
| PFN `large` | **2/4** | FAIL 9.12% | FAIL 100.00% | PASS 1.88% | PASS 2.83% |
| ↳ TSMC7 same tier | 2/4 | FAIL 8.32% | FAIL 100.00% | PASS 1.35% | PASS 3.08% |
| PFN `xl` | **3/4** | PASS 0.08% | FAIL 100.00% | PASS 1.69% | PASS 2.65% |
| ↳ TSMC7 same tier | 0/4 | — | — | — | — |

**Cells whose verdict differs between the two runs on identical data:** DirectNet `small`: ring_osc, opamp; DirectNet `large`: ring_osc, opamp; DirectNet `xl`: opamp; PFN `small`: ring_osc; PFN `medium`: opamp.

Read the pairs, not the totals — and read them **per family**, because that is
where the result lives.

| family | cells compared | verdicts agreeing | worst ring gap | opamp behaviour |
|---|---|---|---|---|
| DirectNet | 16 | 11 (69 %) | **4.2 pp** | bimodal — rails in one run, 7.12 % in the other |
| **BSIM-AR** | 16 | **16 (100 %)** | 3.0 pp, never crossing | **never rails in either run** (4.21–5.39 %) |
| PFN | 12 | 10 (83 %) | **4.9 pp** | bimodal |

* **`ring_osc` carries up to ±4.9 pp of run-to-run scatter across a 5 % gate** —
  DirectNet 4.82 % vs 9.04 % at `large`, PFN 9.72 % vs 4.83 % at `small`. **The
  gate sits inside the scatter.**
* **`opamp` is bimodal for DirectNet and PFN** — a good basin (1.81–7.12 %) or a
  100 % rail, unpredictably — **and stable for BSIM-AR**, whose four opamps land
  within 4 pp of their TSMC7 counterparts and never rail.
* **`sram_snm` and `switchcap` reproduce tightly** (≤0.9 pp) and their verdicts
  never differ, in any family.

Two conclusions follow. First, the pipeline's run-to-run variance is
**concentrated in exactly the two cells this project has always ranked recipes
on**: every "recipe A beats recipe B by one cell" claim in `by-recipe.md` that
turns on a ring or an opamp is within noise, while the same claim resting on a
SRAM or switchcap cell is not. Second, **reproducibility is family-dependent**,
and BSIM-AR is the stable one — all 16 of its verdicts reproduce across two
independent trainings, where production DirectNet reproduces 11. That is a
practical argument for BSIM-AR as the fidelity option on top of its 16/16
score, and a caution about DirectNet's.

None of this touches the family-level conclusions, which rest on many cells at
once (production DirectNet 15/16 strict, BSIM-AR `corroft@medium` 16/16), nor
the corridor law, which moves rings by ~8 pp against a ~4 pp noise floor. What
it retires is promoting a recipe on a single-cell margin.

Datasets regenerated from the kept vendor PDK and **verified `array_equal` to
`tsmc7_*`** — 1,816,830 nmos / 2,187,292 pmos rows matching on `inputs`,
`geometry`, `outputs` and `sample_class`. That is both the precondition for the
repeat being controlled and a fresh, independent reproduction of the V6.13.0
duplicate finding: the audit compared files generated in June, this re-derives
the same bytes from the PDK today.

The campaign refuses to train unless that check passes
(`scripts/tsmc6_restore_campaign.sh`) — it stopped one wave already, when the
first regeneration followed a documented recipe that omitted
`--enable-subvt-off` and silently produced a set 4.7 % smaller. All **24
checkpoints** (3 families × 4 scales × 2 devices, one clean recipe) trained and
were gated by `scripts/tsmc6_gate_campaign.sh`; the table above is that result.
The superseded V6.11.0 pre-fix TSMC6 tables are in
[`archive-pre-gds-fix.md`](archive-pre-gds-fix.md).

