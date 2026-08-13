# AnalogGym on TSMC16 — BSIM-CMG FinFET via PyCMG

AnalogGym's circuits re-targeted from sky130 (planar, 1.8 V) to **TSMC16
FinFET at 0.8 V**, using the BSIM-CMG compact model that
`PyCircuitSim/external_compact_models/PyCMG` wraps.

Each design is a directory with a netlist, the model cards it names, the design
vector that produced it, and runnable testbenches:

```
designs_tsmc16/
  tools/                       the port: parsers, emitters, sizing loops
  models/cache/TSMC16/         PyCMG-generated cards, keyed (device, L, NFIN)
  amplifier/<design>/          netlist.spice  tsmc16_models.spice  design.json
                               tb_gain.cir tb_cmrr.cir tb_psrrp.cir
                               tb_psrrn.cir tb_dc.cir tb_tran.cir  result.json
  ldo/<design>/                netlist.spice + 8 benches
  sensing_front_end/<design>/  netlist.spice  tb_dc.cir
  voltage_reference/<design>/  netlist.spice  tb_dc.cir
  charge_pump/chargepump/      netlist.spice  tb_tran.cir
  ../run_all.py                runs everything, writes results/summary.csv
  RESULTS.md                   the numbers
```

## Running

```sh
python3 ../run_all.py                 # everything
python3 ../run_all.py amplifier ldo   # selected categories
NGSPICE=/path/to/ngspice python3 ../run_all.py
```

Requirements: **ngspice 45.2+** (needs OSDI) and the OSDI binary at
`PyCircuitSim/external_compact_models/PyCMG/build/osdi/bsimcmg.osdi`.
`PYCMG_DIR` overrides the PyCMG location.

A deck cannot be handed to ngspice directly — the OSDI binary must be loaded
before the netlist is parsed, so everything runs through a generated runner:

```
.control
set num_threads=1
osdi <.../bsimcmg.osdi>
source <deck>.cir
ac dec 20 0.1 10G
.endc
```

## How the port works

### Model cards are generated, not hand-written

`tools/pycmg_lib.py` calls `pycmg.tech.resolve_modelcard`, which picks the right
(L-bin × NFIN-group) variant out of the TSMC16 PDK and writes a single `.model`
card. Cards are cached under `models/cache/TSMC16/`.

**ngspice's OSDI binding rejects instance parameters.** `L=20n` on a device line
aborts the parse with `unknown parameter (l)`. Every distinct geometry therefore
needs its own `.model` with L / NFIN / TFIN baked in, which is why a design
carries a `tsmc16_models.spice` naming geometries like `nsvt_l60_f4`. The device
multiplier `m=` *is* accepted (verified: `m=3` gives exactly 3× the current), so
a library is keyed on (device, Vt, L, NFIN) only and multiplicity stays on the
instance line.

### Geometry mapping

FinFET width is quantised: one fin is `2·HFIN + TFIN` = 72 nm of channel. A
device's width becomes a fin count times a multiplier.

* **Amplifiers** are sized by *role*. AnalogGym tags every transistor
  (`gm1_PMOS`, `BIASCM_NMOS`, `LOAD2_NMOS`, …), and devices sharing a role share
  a geometry — that tagging is what makes a topology-agnostic re-design
  possible. The integer factor in `m='VAR*4'` is a mirror ratio and is carried
  across unchanged.
* **Everything else** states W and L directly. There, L is preserved in absolute
  terms (clamped into TSMC16's [16, 240] nm) and W follows it so W/L holds, with
  one fin count chosen per design so ratios between devices are exact integers.

### The circuits are re-designed, not transcribed

Connectivity and mirror ratios are the shipped ones. Geometry, bias and
compensation are not: a 1.8 V planar design point does not survive a move to
0.8 V FinFET, and several sources use a 20 µm channel as a high-impedance
element — TSMC16 stops at 240 nm, so holding W/L there would ask for a 2.6 nm
wide device, a thirtieth of one fin.

Each design is therefore re-sized by a deterministic pattern search over a small
set of physically meaningful knobs (compensation strength, bias level, per-stage
drive, channel length, common mode), scored against per-category targets. The
resulting design vector is saved as `design.json`, and `result.json` records the
measurements and which targets were met.

### The polish pass

The reduced knob sets make 37 designs tractable but under-parameterise a few
of them; the designs the first pass left partial got a second, per-design pass
(`tools/polish_amp.py`, `polish_ldo.py`, `polish_sfe.py`, `polish_vref.py`).
It warm-starts from the shipped `design.json` and adds exactly the freedom the
first pass lacked:

* **Full per-role vectors** for the amplifier (every role and passive on its
  own axis) — the 10-knob search files Ramos_PFC's output *pull-up* under its
  "mid" group and welds both compensation capacitors to one multiplier.  The
  amplifier polish also searches on the dec-20 AC sweep the report uses: the
  dec-8 search sweep can misread phase-at-0dB by the whole margin when the
  phase wraps near the crossing.
* **Per-device geometry and Vt flavor** for the sensors and references built
  from AnalogGym templates that ship every device at the same placeholder
  W/L.  "Matched groups" derived from that geometry weld unrelated devices
  together (a reference stack to the amplifier tail that happens to share its
  W/L), and identical stacked devices develop no |Delta|Vgs at all — sizing per
  device, with per-device Vt flavors, is what the templates' own design space
  (per-device variables in the RL environment) already implies.
* **Compensation resistors, the bias current, and the divider impedance** for
  the LDOs — all fixed constants in the first pass, all first-order levers on
  the phase margins three of the four partials failed on.
* A `--fine` mode (small steps, no reseeding) for designs within a fraction
  of a dB of a bar, where coarse moves fall off the ridge.

A polish result replaces the shipped directory only when the full-bench score
improves; `results/*_polish.json` records each attempt either way.

## Deviations from the AnalogGym testbenches

Every `.meas` statement is AnalogGym's. What changed, and why:

1. **One amplifier instance per deck.** AnalogGym runs the ADM, CMRR, PSRR± and
   DC benches from a single file. They share only ideal supply rails, so they
   are electrically independent and splitting them changes no measurement — but
   five 90 dB loops in one matrix does not solve. gmin stepping, source stepping
   and the transient op all fail, and which groupings happen to converge is luck
   of the Newton basin, not a rule.

2. **`.nodeset` at the unity-feedback solution.** Each instance is wired in
   unity feedback, so its output sits at VCM by construction. Seeding that is
   what lets the operating point solve at all: the 1 TH feedback inductor is a
   bare 0 V branch at DC, and with 90 dB of loop gain behind it Newton cannot
   find the point unaided.

3. **Literals in `.meas param` expressions.** A `.meas` param expression cannot
   read a `.PARAM`; `vos25 = 'vout25-VCM'` fails with *Cannot compute
   substitute*. The values are inlined.

4. **The LDO transient and PSRR benches close the loop.** A 1 TH inductor is a
   short at DC but an *open* in transient, so a load step measured through it
   runs with no feedback and the output collapses. PSRR is likewise a
   closed-loop property. Only the loop-gain bench keeps the inductor.

5. **A real LDO output capacitor.** AnalogGym's mim instance is about 2 pF,
   which cannot hold a 50 mA load step — its own sky130 run reports 1.1 V to
   18 V of undershoot on a 1.8 V output. The regulators get 100 nF.

6. **AnalogGym's `sr_fall` bug is fixed.** The shipped fall-edge block names
   `t_fall` twice and the second definition reads a `t_fall_` that never exists,
   so `sr_fall` lands 1e6× off `sr_rise`. The rise block is correct and is
   mirrored. (The sky130 port in `../designs` carries the same fix.)

7. **Sub-threshold tolerances.** The voltage references run on picoamp
   currents — below ngspice's default 1 pA `ABSTOL`, which made the temperature
   sweep abort before it reached 25 °C. Those decks set `abstol=1e-16`.

8. **`set num_threads=1`.** ngspice parallelises device evaluation with OpenMP
   and sets the thread count itself; `OMP_NUM_THREADS` and `.option numthreads`
   are both ignored. Unpinned, one AC deck took 1.70 s at 319 % CPU; pinned,
   0.19 s at 60 %. The threads were fighting each other rather than the work.

9. **Transient-op aids on the LDO load-step deck.** ngspice's transient
   operating point is a different algorithm from the DC op the other decks
   use, and it fails on the bias-diode node of the large-pass designs.  The
   tran deck sets `cshunt=1e-15` (1 fF from every node to ground — far below
   any device capacitance here), and two designs additionally seed their
   input-pair tail node with a `.nodeset`.  The DC and AC decks are untouched.

10. **The voltage-reference suite sweeps cold-to-hot** (`dc temp -40 125`),
    matching the direction the category's own bench uses — the per-device
    three_output design solves from -40 °C by continuation but not from a
    125 °C first point.  (The amplifier DC bench sweeps hot-to-cold for the
    mirror-image reason; each category keeps the direction its designs
    actually solve in.)

## What is not here

* **Phase-Locked Loop** — ships no netlist in any text form; the connectivity
  exists only inside a Cadence OpenAccess database. Blocked in the sky130 port
  too, for the same reason.
* **`bandgap_vref` and `subthreshold_vref`** — need an NPN and a PNP. BSIM-CMG
  models a FinFET; there is no bipolar in this TSMC16 view to bind them to.
* **The rest of the charge pump's PLL** — the shipped netlist is a whole PLL,
  but its transient bench reaches exactly one block (`PLL_CHARGEPUMP` and the
  `PLL_QUENCH_v33` inside it). Only that is ported; nothing was invented to fill
  the gap.

## Provenance

Ported from `../designs` (the sky130 tree), which is itself derived from
`../AnalogGym`. Both are untouched. `tools/` regenerates everything here.

## Other technology nodes

`tools/` is tech-parametric: the node is auto-detected from the tree's
directory name (override with `AG_TECH`), and every per-tech constant --
supply, fin geometry, L bins, NFIN groups, Vt flavors -- comes from the PyCMG
tech registry plus a scanned-PDK table in `tools/pycmg_lib.py`.  The sibling
trees `../designs_tsmc5`, `../designs_tsmc6`, `../designs_tsmc7` and
`../designs_tsmc12` carry the same circuits on those nodes, produced by

```sh
rsync -a --exclude __pycache__ tools ../designs_<tech>/
cd ../designs_<tech> && sh ../tools/pipeline.sh
```

The pipeline ports this tree's design vectors (`tools/port_tech.py`: sizes
shared, voltages scaled to the rail, geometry snapped into the PDK envelope,
Vt fallbacks hvt->svt / lnvt->lvt), measures them against the same gates
(`tools/finalize.py`), and re-tunes only the designs a gate fails
(`tools/retune.py`).  Cross-node comparison: `../RESULTS_TSMC.md`.
