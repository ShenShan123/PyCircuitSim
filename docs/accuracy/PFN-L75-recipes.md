# PFN (LEVEL=75) — recipe variants

Until V7.3.0 this file could not have existed. PFN had only ever been trained
`clean`: the curriculum recipes were fully wired — every driver accepts
`MODEL=tabpfn` — but none had been run, so the family's entire accuracy record
rested on a single training configuration.

This campaign runs the one arm that is aimed squarely at the cells PFN fails.

> **Read the noise floor first.** `ring_osc` carries ±4 pp of run-to-run
> scatter across a 5 % gate and PFN's `opamp` is bimodal — a good basin or a
> 100 % rail (`methodology.md` §7). With a single recipe and one training run
> per cell, a one-cell difference below is **not** a result. What would be a
> result is a consistent move across many ring cells, which is the size of
> effect the corridor produces in the other two families.

> **Denominators.** Totals are **/20** — 4 circuits × 5 techs, TSMC6 included (`methodology.md` §2). Earlier reports scored /16 over four techs, so a /20 total here and a /16 total there can be the same measurement.

---

## 1. The arm

| recipe | tier | class weights | warm start | data |
|---|---|---|---|---|
| **`corroft`** | `small` | `traj_corridor=3.0` | own clean `small` | ring-only `corro` |

120-epoch fine-tune at lr 3e-4, patience 40, `--amp`, seed 42 — the identical
addendum the other two families use, applied to every (tech × device) pair. No
per-tech special.

`small` is the tier because it is PFN's best clean tier: its capacity curve
declines past `medium`, `large` is optimization-unstable and `xl` mostly banked
early (`PFN-L75-clean.md` §4). Fine-tuning the strongest base is the
mechanical choice, not a tuned one.

**Why this arm and not another.** The corridor is the only lever in this
project that has ever closed a low-VDD ring, in either other family, at any
tier — and low-VDD rings plus opamps are exactly what PFN fails. It is also
`gds`-invariant, so the result cannot be an artifact of the bug that distorted
the pre-fix recipe rankings.

The corridor dataset for TSMC6 did not exist before this campaign and was
harvested for it. It came out `array_equal` to TSMC7's — a third independent
reproduction of the TSMC6-is-TSMC7 finding, and the first at the corridor
level rather than on base data: the corridor is harvested by *running the ring
oscillator* under LEVEL=72, so identical rows mean the two techs' circuits
follow the same trajectory, not merely that their datasets match.

## 2. Gates by recipe

| group | strict /20 | ring_osc | opamp | sram_snm | switchcap | flips | open cells |
|---|---|---|---|---|---|---|---|
| `corroft`@small | **14/20** | 5/5 | 0/5 | 5/5 | 4/5 | 0 | tsmc5-opamp, tsmc6-opamp, tsmc7-opamp, tsmc12-opamp, tsmc16-opamp, tsmc12-switchcap |

## 3. What the corridor changed against clean

| recipe | tier | cells gained vs clean | cells lost vs clean | net |
|---|---|---|---|---|
| `corroft` | small | tsmc5-ring_osc, tsmc7-ring_osc | tsmc12-opamp, tsmc16-opamp | **+0** |

### The result: the corridor closes every ring, and pays for it in opamps

**The total does not move. The failure set is replaced.** `corroft` takes the
ring column from 3/5 to **5/5** — every ring, including the two low-VDD cells
(`tsmc5`, `tsmc7`) that no PFN checkpoint at any tier had ever passed — and
takes the opamp column from 2/5 to **0/5**. Net zero.

Read the two halves differently, because the evidence for them is not equal:

* **The ring result is real.** It is not one cell but the whole column, and the
  corridor's effect on rings (~8 pp) is about twice the measured ring noise
  floor (±4 pp). It also **reproduces the law across a third architecture**: the
  corridor was already the only lever that closes a low-VDD ring on DirectNet
  (an MLP) and on BSIM-AR (an autoregressive Transformer), and it now does the
  same on an in-context transformer with a frozen context. Three architectures,
  one data lever, same effect — that is a property of the *training
  distribution*, not of any model.
* **The opamp loss is at the edge of what this project can resolve.** PFN's
  opamp cells are bimodal — a good basin or a 100 % rail — and two cells is
  within that scatter. What raises it above a coin-flip is the direction and the
  company it keeps: it is the same trade DirectNet's `csobcrit` and `crit30a1`
  arms produced, and the same one BSIM-AR shows between its clean and corridor
  tiers.

So this is a third instance of the standing law that **curricula relocate
basins rather than composing them** — and the cleanest instance yet, because
here the relocation is exact: two cells gained, two cells lost, on one shared
`id` head whose capacity the two circuit classes compete for.

**What it does not settle.** Whether the trade is avoidable. The `inv_trip`
anchor is what makes the corridor safe on DirectNet — it composes there and is
inert on BSIM-AR — and it has never been run on PFN. If the anchor holds the
opamp basins while the corridor holds the rings, `crit30`@small would be
16–20/20 rather than 14. That is the single highest-value experiment this file
points at, and §6 lists it first.

## 4. Per testcase

#### Ring oscillator

*Verdict is the gate's exit code; the number is the period error %, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| `corroft`@small | **PASS** 4.33% | **PASS** 2.54% | **PASS** 3.66% | **PASS** 3.22% | **PASS** 2.47% |

#### Two-stage Miller opamp (DC)

*Verdict is the gate's exit code; the number is the open-loop gain error %, gate ≤10 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| `corroft`@small | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% | FAIL 100.00% | FAIL 10.16% |

#### 6T SRAM read SNM

*Verdict is the gate's exit code; the number is the worst lobe NRMSE %, gate ≤10 % and all lobes positive.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| `corroft`@small | **PASS** 6.62% | **PASS** 1.55% | **PASS** 1.57% | **PASS** 1.57% | **PASS** 1.41% |

#### Switched-capacitor cell

*Verdict is the gate's exit code; the number is the charge error % of VDD, gate ≤5 %.*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 |
|---|---|---|---|---|---|
| `corroft`@small | **PASS** 2.25% | **PASS** 2.13% | **PASS** 2.48% | FAIL 5.81% | **PASS** 3.97% |

## 5. Device fidelity and AC

The corridor is a *data* lever — it adds a bias tube along the circuits'
own trajectories and upweights it — so its risk to the device suites is that
concentrating loss mass on 0.3 % of the rows degrades the rest of the surface.
The device tables below are the check on that.

**Parametric DC — `verify_nn_multi_tech_dc`** *(mean Id-Vgs NRMSE %, config fails in brackets)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| `corroft`@small | 1.80 | 2.02 (13/14) | 1.00 | 0.41 | 0.51 | 68/69 |

**Parametric transient — `verify_nn_multi_tech_tran`** *(mean NRMSE %)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass |
|---|---|---|---|---|---|---|
| `corroft`@small | 1.72 | 1.51 | 1.52 | 1.56 | 1.57 | 80/80 |

**Device CS-amp AC** — NMOS / PMOS *(gate: gain0 ≤1.5 dB, f3db ratio ∈[0.7, 1.43], magNRMSE ≤10 %)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /10 |
|---|---|---|---|---|---|---|
| `corroft`@small | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ | **10/10** |

**Opamp open-loop AC** — DC-gain error *(gate: ≤3 dB, GBW ratio ∈[0.6, 1.67], PM err ≤15°, non-railed OP)*

| group | TSMC5 | TSMC6 | TSMC7 | TSMC12 | TSMC16 | pass /5 |
|---|---|---|---|---|---|---|
| `corroft`@small | FAIL 16.29 dB | FAIL 31.58 dB | FAIL 32.10 dB | FAIL 31.05 dB | **PASS** 1.72 dB | **1/5** |

## 6. Untried arms

Recorded so the next campaign does not have to rediscover the list. All are
wired and none has been run on PFN:

| arm | why it might matter |
|---|---|
| `crit30` / `crit15m` (corridor + `inv_trip`) | The anchor composes with the corridor on DirectNet and is inert on BSIM-AR. PFN would be the third data point, and the first that could distinguish "MLP vs everything else" from "architecture-specific". |
| `csob` (charge-Sobolev) | Supervises the ∂q/∂V surface. PFN's one distinctive failure — `tsmc12-switchcap` by 0.1–0.3 pp — is a charge/off-state miss, which is what this lever targets. The closest open cell in the project meeting the lever aimed at it is the highest-value untried combination here. |
| `corroft` at other tiers | Would separate "the corridor helps PFN" from "the corridor helps PFN at `small`", and the tier axis is where PFN behaves least like the others. |
