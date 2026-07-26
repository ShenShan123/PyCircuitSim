# Accuracy reports

Ground truth is **always** NGSPICE on the *identical* BSIM-CMG (LEVEL=72) OSDI
model — never a simplified or self-defined reference.

## Start here

| file | what it answers |
|---|---|
| **[`methodology.md`](methodology.md)** | What a "gate" is, the thresholds, strict-OMP discipline, isolation, and **which code state produced which number**. Read before comparing any two numbers. |
| **[`by-tech.md`](by-tech.md)** | *"How does TSMC5 / 7 / 12 / 16 behave, across families and scales?"* — plus TSMC6, retired. |
| **[`by-scale.md`](by-scale.md)** | *"What does small / medium / large / xl buy?"* — the capacity laws, and which ones survived the re-gate. |
| **[`by-recipe.md`](by-recipe.md)** | *"What does each training recipe do?"* — the catalogue, the corridor/anchor levers, the dead ends. |
| [`DirectNet-L73-accuracy.md`](DirectNet-L73-accuracy.md) | The production family: architecture, production state, universal-scope study. |
| [`BSIM-AR-L74-accuracy.md`](BSIM-AR-L74-accuracy.md) | The high-fidelity family: AR transformer, its cost, its ceiling. |
| [`PFN-L75-accuracy.md`](PFN-L75-accuracy.md) | The research family: TabPFN port, in-context inference. |
| [`archive-pre-gds-fix.md`](archive-pre-gds-fix.md) | Frozen pre-fix data tables + the register of retracted claims. |

The three axis files are the **single source of truth** for cross-cutting
numbers; the family files carry only what is specific to one family.

## Scoreboard

Strict = passes at OMP ∈ {1, 2, 4}. Complex matrix = 4 circuits × 4 techs.

| LEVEL | family | role | best config | params | complex (strict) | device AC | CPU ms/eval |
|---|---|---|---|---|---|---|---|
| 73 | **DirectNet** | **production** | `crit30f@large` | 0.92 M | **15/16**, 0 flips | 8/8 | **1.5** |
| 73 | DirectNet | best any tier | `crit15m@xl` | 2.13 M | **16/16**, 0 flips | see `by-scale.md` | 3.4 |
| 74 | **BSIM-AR** | higher fidelity | `corroft@medium` | 1.9 M | **16/16**, 0 flips | 8/8 | 61.5 |
| 75 | **PFN** | research | `clean@small` | 0.69 M | 11/16, 0 flips | 8/8 at `large` | 15.6 |

* **Production stays DirectNet.** 15/16 strict at 0.92 M params and ~40× BSIM-AR's
  speed; the single open cell is `tsmc7-opamp` **at `large` only** — DirectNet
  passes it at `small` (1.81 %) and `xl` (4.20 %).
* **Two independent families now sweep 16/16 strict** with ordinary uniform data
  recipes (`crit15m@xl`, `corroft@medium`), so the matrix no longer separates
  them; inference cost and device-suite breadth do.
* **Every family is flip-free.** The OMP multistability that plagued every
  earlier campaign was a wrong-signed Jacobian entry, not a property of
  high-gain circuits (`methodology.md` §6).

## What is actually open

1. **The low-VDD rings** (`tsmc5-ring`, `tsmc7-ring`) for every *clean* recipe —
   deterministic failures at 5.5–13 % against a 5 % gate, closed only by the
   corridor curriculum (`by-recipe.md` §3).
2. **`tsmc7-opamp` at DirectNet `large`** — the sole cell keeping production off
   a full sweep, and reachable at other tiers, so a basin problem rather than a
   fidelity wall.
3. **The opamp open-loop AC gate** — see `by-scale.md` §5 for the V7.1.0
   re-measurement of a claim that stood as "0/4 everywhere".
4. **PFN's clean-recipe ceiling** — the corridor curriculum has never been run on
   PFN, and it is the lever that closes exactly the cells PFN fails.

## Provenance in one line

Numbers carry one of three code states — **pre-fix** (≤ `a96112a`, `gds` sign
bug present), **V6.13.0** (`d2ea720`, fix shipped, full complex re-gate),
**V7.1.0** (HEAD, device/AC/strict re-gate). `methodology.md` §6 says exactly
what each is comparable to; every table here is labelled.
