# Circuit templates

`examples/` is the authoritative source for every circuit topology exercised by
the test harness. Each topology exists once as a parameterized
`*.spice.tmpl` file; runnable candidate and NGSPICE reference decks are rendered
from that same file.

Materialized `.sp`, `.cir`, simulator logs, traces, plots, and reports do not
belong here. The harness writes them below `results/`.

## Layout

- `single_devices/` contains the canonical four-terminal MOSFET template.
- `simple_circuits/` contains inverter, amplifier, logic, memory, switched-cap,
  and other compact-model qualification or diagnostic topologies.
- `subcircuits/` contains flat and hierarchical forms used to verify subcircuit
  expansion, parameter passing, initial conditions, and simulator parity.

## Template contract

Placeholders use uppercase angle-bracket tokens such as `<VDD>` and `<NFN>`.
Rendering is strict: a missing value or a supplied value unused by the template
is an error. Keep the topology, element names, sources, loads, and analysis slot
in the template; keep sweep selection and pass/fail policy in `tests/`.

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

`tests/common/simple_circuit_catalog.py` owns the topology inventory and analysis
metadata. `tests/common/simple_circuit_harness.py` owns corners and the two
simulator adapters. `tests/common/base.py` owns strict rendering.

When adding a circuit, add one template, register it in the catalog when it is
part of the simple-circuit matrix, and extend the catalog contract check. Never
add a paired candidate/reference deck or copy a netlist into a test module.

Environment setup and executable commands are maintained in the repository
[README](../README.md).
