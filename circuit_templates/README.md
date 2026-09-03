# Circuit templates

`circuit_templates/` is the authoritative source for every circuit topology
exercised by the test harness. Each topology exists once as a parameterized
`*.spice.tmpl` file; runnable candidate and NGSPICE reference decks are
rendered from that same file.

Materialized `.sp`, `.cir`, simulator logs, traces, plots, and reports do not
belong here. The harness writes them below `results/`.

## Layout — ordered by what a circuit demands of a compact model

The directories are a difficulty ladder, not an application taxonomy. What
makes a circuit hard for a fitted device model is not how many transistors it
has but how much of the answer the deck already supplies. Each rung removes
one more crutch:

| tier | what the deck still supplies | what the model must supply |
|---|---|---|
| `L0_devices/` | every terminal voltage | one current |
| `L1_primitives/` | every bias; one passive load | an operating point on a load line |
| `L2_stages/` | every gate rail, from an ideal source | several coupled device currents |
| `L3_blocks/` | the supply only | internally generated bias, or internal state |
| `L4_systems/` | the supply and a reference | a closed negative-feedback solution |

`subcircuits/` sits outside the ladder. Those are flat/hierarchical fixture
pairs that verify subcircuit expansion, parameter passing, and initial
conditions — they test the parser, not the compact model.

### Reading a failure by tier

The ladder exists to localize. A model that passes `L2_stages` and fails
`L3_blocks` has not got "worse at big circuits" — it has failed to select an
operating point that no source is pinning for it. That is a different defect
from a pointwise current error, and it needs a different fix.

The smallest instance of that defect is `L1_primitives/diode_load.spice.tmpl`:
a diode-connected device fed through a resistor, where the operating point is
the intersection of a load line with the model's own surface. Every other
diode-connected device in this tree is fed by an ideal current source, which
pins the current and leaves only the voltage to solve.

## Template contract

Placeholders use uppercase angle-bracket tokens such as `<VDD>` and `<NFN>`.
Rendering is strict: a missing value or a supplied value unused by the template
is an error. Keep the topology, element names, sources, loads, and analysis
slot in the template; keep sweep selection and pass/fail policy in `tests/`.

The common parameter groups are:

| Group | Representative tokens | Purpose |
|---|---|---|
| Simulator adapter | `MODEL_SETUP`, `N_PREFIX`, `P_PREFIX`, `N_DEVICE`, `P_DEVICE` | Render the same topology for an NN model or LEVEL=72 OSDI reference |
| Technology and VT | `TECH`, `NVT`, `PVT`, `LEVEL` | Select technology, threshold variants, and compact-model family |
| Geometry and P/N ratio | `LN`, `LP`, `NFN`, `NFP` | Vary channel lengths and independent NMOS/PMOS fin counts |
| PVT and body bias | `VDD`, `TEMP`, `BODY_N`, `BODY_P` | Apply supply, temperature, and body-bias corners |
| Stimulus | `INPUT_RISE`, `INPUT_FALL`, timing and bias tokens | Vary slew, period, duty cycle, common mode, and DC bias |
| Loading | `OUTPUT_LOAD` and topology-specific load tokens | Add or change output capacitance, resistance, and fanout |
| Analysis | `ANALYSIS` | Insert the gate-owned `.op`, `.dc`, `.tran`, or `.ac` card |

A token exists so that one topology can serve several experiments. The opamp's
source specs (`VDD_SPEC`, `VINP_SPEC`, `VINN_SPEC`) are the worked example:
their defaults render byte-identically to the pre-token deck, which is what
keeps the frozen simple-v1 opamp cell unchanged while the same file also
serves the CMRR and PSRR experiments.

## Ownership

`tests/common/simple_circuit_catalog.py` owns the topology inventory, the
analysis metadata, and each case's declared `tier`.
`tests/common/simple_circuit_harness.py` owns corners and the two simulator
adapters. `tests/common/base.py` owns strict rendering and `template_deck()`,
which resolves a bare template name to its tier and rejects a name that two
tiers both claim.

When adding a circuit: add one template to the tier whose crutches it removes,
register it in the catalog with that `tier`, and extend the catalog contract
check. Never add a paired candidate/reference deck or copy a netlist into a
test module.

Environment setup and executable commands are maintained in the repository
[README](../README.md).
