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

Two directories sit outside the ladder. `subcircuits/` owns flat/hierarchical
fixture pairs for expansion, parameter passing, initial conditions, and NN
model resolution. `controls/` owns passive solver controls such as the RC
low-pass; a circuit with no compact-model device cannot honestly occupy a
compact-model difficulty tier.

### Subcircuit fixture seam

`subcircuits/` is deliberately separate because it classifies a different
question: whether two netlist representations become the same physical
circuit. The L0–L4 ladder classifies how much circuit behavior a compact model
must determine. Mixing those axes would make a parser-expansion failure look
like a model-fidelity failure and would scatter one harness across several
tiers.

| Fixtures | Counterpart | Contract |
|---|---|---|
| `rc_ladder_{flat,hierarchical}` | local pair | transient flattening, parameter override, and expression evaluation |
| `rc_lowpass_hierarchical` | `controls/rc_lowpass` | hierarchical AC equivalence without duplicating the canonical flat control |
| `resistor_tree_{flat,hierarchical}` | local pair | nested instances, quoted/braced expressions, and DC operating point |
| `ic_hierarchical` | parsed hierarchical state | internal-node naming, parameterized `.ic`, and `uic` pinning |
| `inverter_hierarchical` | `L2_stages/inverter` | MOS model, L/NFIN, and port propagation through one instance |
| `inverter_buffer_{flat,hierarchical}` | local pair plus NGSPICE | nested X-in-X expansion, internal `.ic`, NN family resolution, and DC/transient/AC physical parity |

Only representation-equivalence fixtures belong here. A test whose primary
question is device or circuit behavior belongs in `controls/` or L0–L4, even
if one adapter happens to use `.subckt`. Flat fixtures are retained when they
are the independent equivalence oracle; they are not catalog cases and never
enter a compact-model denominator.

### Reading a failure by tier

The ladder exists to localize. A model that passes `L2_stages` and fails
`L3_blocks` has not got "worse at big circuits" — it has failed to select an
operating point that no source is pinning for it. That is a different defect
from a pointwise current error, and it needs a different fix.

The smallest instance of that defect is `L1_primitives/diode_load.spice.tmpl`:
a diode-connected device fed through a resistor, where the operating point is
the intersection of a load line with the model's own surface. The PMOS-only
`L3_blocks/self_biased_cascode_pmos.spice.tmpl` extends that load-line question
to a two-device generated rail. Current-source-fed diode branches instead pin
the current and leave only the voltage to solve.

## Template contract

Placeholders use uppercase angle-bracket tokens such as `<VDD>` and `<NFN>`.
Rendering is strict: a missing value or a supplied value unused by the template
is an error. Keep the topology, element names, sources, loads, and analysis
slot in the template; keep sweep selection and pass/fail policy in `tests/`.
Different device counts are different topologies: the bias-fanout scale ladder
therefore owns explicit 3T, 5T, 9T, and 17T files instead of injecting devices
from harness code. The ring stage-count sweep likewise owns explicit 3-, 5-,
7-, and 9-stage templates.

The common parameter groups are:

| Group | Representative tokens | Purpose |
|---|---|---|
| Simulator adapter | `MODEL_SETUP`, `N_PREFIX`, `P_PREFIX`, `N_DEVICE`, `P_DEVICE` | Render the same topology for an NN model or LEVEL=72 OSDI reference |
| Technology and VT | `TECH`, `NVT`, `PVT`, `LEVEL` | Select technology, threshold variants, and compact-model family |
| Geometry and P/N ratio | `LN`, `LP`, `NFN`, `NFP`, role-specific `*_DEVICE` | Vary global or named-role L/NFIN/VT independently |
| PVT and body bias | `VDD`, `TEMP`, `BODY_N`, `BODY_P` | Apply supply, temperature, and body-bias corners |
| Stimulus | `INPUT_RISE`, `INPUT_FALL`, timing and bias tokens | Vary slew, period, duty cycle, common mode, and DC bias |
| Loading | `OUTPUT_LOAD` and topology-specific load tokens | Add or change output capacitance, resistance, and fanout |
| Analysis | `ANALYSIS` | Insert the gate-owned `.op`, `.dc`, `.tran`, or `.ac` card |

### What a template may not contain

A template may only use cards and value syntax that NGSPICE and PyCircuitSim
read identically. This is not covered by candidate/reference parity: both decks
render from the same file, so a card that one engine honours and the other
drops produces two different problems and a clean parity report.
`tests/test_deck_engine_compatibility.py` enforces the rule against the
parser's real support surface in `tests/common/parser_support.py`. Two limits
it exists to catch:

- `.options`, `.nodeset`, `.param` and the other directives
  `Parser.parse_line` does not implement are dropped. Those that change the
  circuit are named in `Parser.PHYSICAL_DIRECTIVES` and warn when dropped, but
  a template must not carry one at all. `.op` is the one card that looks
  unhandled and is not: it leaves `analysis_type` unset, which the runner reads
  as "single DC operating point", so both engines agree.
- Values use SPICE scale factors and V7.6.9 corrected the parser to match
  NGSPICE, so `1m` is milli, `1meg` is mega, and `2.2kohm` is 2200 in both
  engines. Before that fix `m` meant mega here, a 1e9 divergence on a
  byte-identical deck.

A token exists so that one topology can serve several experiments. The opamp's
source specs (`VDD_SPEC`, `VINP_SPEC`, `VINN_SPEC`) are the worked example:
their defaults render byte-identically to the pre-token deck, which is what
keeps the frozen simple-v1 opamp cell unchanged while the same file also
serves the CMRR and PSRR experiments.

## Ownership

`tests/common/simple_circuit_catalog.py` owns the topology inventory, the
analysis metadata, and each case's declared `tier`.
`tests/common/simple_circuit_harness.py` owns corners and the NN, NGSPICE
LEVEL=72, and PyCircuitSim LEVEL=72-control adapters. `tests/common/base.py`
owns strict rendering, `template_deck()`, and `control_deck()`,
which resolves a bare template name to its tier and rejects a name that two
tiers both claim.

When adding a circuit: add one template to the tier whose crutches it removes,
register it in the catalog with that `tier`, and extend the catalog contract
check. Never add a paired candidate/reference deck or copy a netlist into a
test module.

Environment setup and executable commands are maintained in the repository
[README](../README.md).
