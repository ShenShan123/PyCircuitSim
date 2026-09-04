#!/usr/bin/env python3
"""NGSPICE-validated gate for the BSIM-CMG instance multiplier ``m=`` (C1).

``m=N`` means N IDENTICAL devices in parallel. It multiplies the residual and
the resistive AND reactive Jacobian, and changes no device physics — measured
on ngspice-45.2 (ASAP7 nmos_rvt, L=30nm, NFIN=10): i(vd) at m=1/4/16 is
-1.27341475e-04 / -5.09365899e-04 / -2.03746360e-03, i.e. exactly x4 and x16.

Checks (all against NGSPICE 45.2 on the identical OSDI model, except where
noted as an internal consistency check that has no ngspice observable):

  1. NMOS OP, m = 1 / 4 / 16: |Id| vs NGSPICE (1% as in verify_bsimcmg_op) and
     the PyCircuitSim m-ratio exact to 1e-9 relative.
  2. PMOS OP, same.
  3. Raw-vs-scaled: the eval cache stays RAW (gm/cgg inside the device are
     m-independent) while get_conductance/get_capacitances/get_charges scale.
     This is what keeps the NR Jacobian the derivative of the stamped residual.
  4. DC sweep (Vgs 0..0.7 at Vds=0.5), m=4, vs NGSPICE.
  5. m=4 versus FOUR devices in parallel: solved node voltages must agree to
     solver tolerance (proves current + all three conductances scale together).
  6. AC: common-source amplifier at m=4, complex V(out) vs NGSPICE at three
     frequencies (proves the capacitances scale too).

Run CPU-pinned (AGENTS.md gate methodology):

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        NGSPICE_BIN=/usr/local/ngspice-45.2/bin/ngspice \\
        python tests/single_devices/verify_cmg_multiplier.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.core_gates import (
    ASAP7_MODELCARD,
    OSDI_PATH,
    bake_asap7,
    ngspice_probe,
    rel_err,
    report,
)
from tests.common.base import (
    DEVICE_DECKS, parse_no_options, template_deck, render_template,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "tests" / "cmg_multiplier"

L = 30e-9
NFIN = 10.0
MULTS = (1, 4, 16)
REL_TOL_I = 0.01        # 1% vs NGSPICE (same as verify_bsimcmg_op)
ABS_TOL_I = 1e-9
RATIO_TOL = 1e-9        # PyCircuitSim's own m-scaling must be exact
PARALLEL_TOL_V = 1e-6   # m=N vs N parallel devices, in volts
AC_MAG_TOL = 0.01       # 1% of |V(out)|
AC_PHASE_TOL = 1.0      # degrees


def _render_device(
    *, model_setup: str, drain_bias: str, gate_bias: str,
    source_bias: str, bulk_bias: str, device_name: str,
    nodes: Tuple[str, str, str, str], device: str, load: str = "",
    extra_devices: str = "", analysis: str = ".op",
) -> str:
    """Render a multiplier experiment from the canonical MOSFET template."""
    drain, gate, source, bulk = nodes
    return render_template(DEVICE_DECKS / "mosfet.spice.tmpl", {
        "MODEL_SETUP": model_setup, "TEMP": "27",
        "DRAIN_BIAS": drain_bias, "GATE_BIAS": gate_bias,
        "SOURCE_BIAS": source_bias, "BULK_BIAS": bulk_bias,
        "DEVICE_NAME": device_name, "DRAIN_NODE": drain,
        "GATE_NODE": gate, "SOURCE_NODE": source, "BULK_NODE": bulk,
        "DEVICE": device, "EXTRA_DEVICES": extra_devices,
        "LOAD": load, "ANALYSIS": analysis,
    })


# ---------------------------------------------------------------------------
# PyCircuitSim helpers
# ---------------------------------------------------------------------------
def _parse(deck_text: str, label: str, modelcard: Path):
    """Parse a deck with an explicit modelcard; return (circuit, parser)."""
    from pycircuitsim.parser import Parser

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"ps_{label}.sp"
    path.write_text(deck_text)
    parser = Parser(modelcard_path=str(modelcard),
                    model_name_map={"NMOS": "nmos_rvt", "PMOS": "pmos_rvt"})
    parser.parse_file(str(path))
    return parser.circuit, parser


def _solve_dc(circuit) -> Dict[str, float]:
    from pycircuitsim.solver import DCSolver

    solver = DCSolver(circuit, use_source_stepping=False,
                      use_gmin_stepping=False, max_iterations=200)
    return solver.solve()


def _device(circuit, name: str):
    for comp in circuit.components:
        if comp.name.lower() == name.lower():
            return comp
    raise ValueError(f"device {name} not in circuit")


# ---------------------------------------------------------------------------
# 1 + 2: single-device OP at m = 1 / 4 / 16
# ---------------------------------------------------------------------------
def test_single_device_op(device: str) -> bool:
    """|Id| vs NGSPICE at three multipliers, plus exactness of the ratio."""
    is_n = device == "nmos"
    model = "nmos_rvt" if is_n else "pmos_rvt"
    baked = bake_asap7(RESULTS_DIR, model,
                       {"L": L, "NFIN": NFIN, "DEVTYPE": 1 if is_n else 0},
                       tag="op")

    print(f"\n  --- {device.upper()} OP, m = {MULTS} ---")
    ng_i: Dict[int, float] = {}
    ps_i: Dict[int, float] = {}
    ok = True

    for m in MULTS:
        # NGSPICE: N-prefix OSDI device, m on the instance line.
        if is_n:
            drain_bias = "Vd d 0 0.5"
            gate_bias = "Vg g 0 0.5"
            source_bias = bulk_bias = ""
            nodes = ("d", "g", "0", "0")
        else:
            # PMOS: source/bulk at 0.7, gate low so the device is ON.
            drain_bias = "Vd d 0 0"
            gate_bias = "Vg g 0 0.2"
            source_bias = "Vs s 0 0.7"
            bulk_bias = ""
            nodes = ("d", "g", "s", "s")
        deck = _render_device(
            model_setup=f'.include "{baked}"', drain_bias=drain_bias,
            gate_bias=gate_bias, source_bias=source_bias,
            bulk_bias=bulk_bias, device_name="N1", nodes=nodes,
            device=f"{model} m={m}",
        )
        data = ngspice_probe(RESULTS_DIR, f"{device}_op_m{m}", deck, "op",
                             ["i(vd)"])
        ng_i[m] = abs(float(data[0, 1]))

        # PyCircuitSim: M-prefix, same m.
        if is_n:
            ps_deck = _render_device(
                model_setup=".model nmos1 NMOS (LEVEL=72)",
                drain_bias="Vd 1 0 0.5", gate_bias="Vg 2 0 0.5",
                source_bias="", bulk_bias="", device_name="M1",
                nodes=("1", "2", "0", "0"),
                device=f"nmos1 L=30n NFIN=10 m={m}",
            )
        else:
            ps_deck = _render_device(
                model_setup=".model pmos1 PMOS (LEVEL=72)",
                drain_bias="Vd 1 0 0", gate_bias="Vg 2 0 0.2",
                source_bias="Vs 3 0 0.7", bulk_bias="", device_name="M1",
                nodes=("1", "2", "3", "3"),
                device=f"pmos1 L=30n NFIN=10 m={m}",
            )
        circuit, _ = _parse(ps_deck, f"{device}_op_m{m}", ASAP7_MODELCARD)
        sol = _solve_dc(circuit)
        ps_i[m] = abs(_device(circuit, "M1").calculate_current(sol))

        err = rel_err(ps_i[m], ng_i[m], ABS_TOL_I)
        ok &= report(f"{device} m={m:<2d} |Id| vs NGSPICE",
                     err <= REL_TOL_I,
                     f"ng={ng_i[m]:.6e} ps={ps_i[m]:.6e} rel={err*100:.4f}%")

    for m in MULTS[1:]:
        r_ps = ps_i[m] / ps_i[1]
        r_ng = ng_i[m] / ng_i[1]
        ok &= report(f"{device} ratio m={m}/m=1 exact (PyCircuitSim)",
                     abs(r_ps - m) <= RATIO_TOL * m, f"ratio={r_ps:.12f}")
        ok &= report(f"{device} ratio m={m}/m=1 exact (NGSPICE)",
                     abs(r_ng - m) <= 1e-6 * m, f"ratio={r_ng:.9f}")
    return ok


# ---------------------------------------------------------------------------
# 3: raw cache vs scaled accessors
# ---------------------------------------------------------------------------
def test_raw_cache_is_unscaled() -> bool:
    """gm/cgg INSIDE the device are m-independent; the accessors scale."""
    print("\n  --- raw eval cache vs scaled accessors ---")
    ok = True
    ref: Dict[str, float] = {}
    for m in MULTS:
        deck = _render_device(
            model_setup=".model nmos1 NMOS (LEVEL=72)",
            drain_bias="Vd 1 0 0.5", gate_bias="Vg 2 0 0.5",
            source_bias="", bulk_bias="", device_name="M1",
            nodes=("1", "2", "0", "0"),
            device=f"nmos1 L=30n NFIN=10 m={m}",
        )
        circuit, _ = _parse(deck, f"raw_m{m}", ASAP7_MODELCARD)
        dev = _device(circuit, "M1")
        volt = {"1": 0.5, "2": 0.5, "0": 0.0}
        raw = dev._eval_dc(volt)
        g_ds, g_m, g_mb = dev.get_conductance(volt)
        caps = dev.get_capacitances(volt)
        charges = dev.get_charges(volt)
        if m == 1:
            ref = {"gm": raw["gm"], "cgg": raw["cgg"], "qg": raw["qg"]}
        ok &= report(f"m={m:<2d} raw gm/cgg/qg unchanged by m",
                     raw["gm"] == ref["gm"] and raw["cgg"] == ref["cgg"]
                     and raw["qg"] == ref["qg"],
                     f"gm={raw['gm']:.6e} cgg={raw['cgg']:.6e}")
        ok &= report(f"m={m:<2d} accessors scaled by m",
                     g_m == raw["gm"] * m
                     and g_ds == abs(raw["gds"]) * m
                     and g_mb == raw["gmb"] * m
                     and caps["cgg"] == raw["cgg"] * m
                     and caps["cgd"] == raw["cgd"] * m
                     and charges["qg"] == raw["qg"] * m,
                     f"gm*m={g_m:.6e} cgg*m={caps['cgg']:.6e}")
    return ok


# ---------------------------------------------------------------------------
# 4: DC sweep at m=4
# ---------------------------------------------------------------------------
def test_dc_sweep() -> bool:
    """Vgs sweep at Vds=0.5, m=4, PyCircuitSim vs NGSPICE."""
    print("\n  --- NMOS DC sweep (Vgs 0..0.7 step 0.05), m=4 ---")
    m = 4
    baked = bake_asap7(RESULTS_DIR, "nmos_rvt",
                       {"L": L, "NFIN": NFIN, "DEVTYPE": 1}, tag="dc")
    deck = _render_device(
        model_setup=f'.include "{baked}"', drain_bias="Vd d 0 0.5",
        gate_bias="Vgs g 0 0.5", source_bias="", bulk_bias="",
        device_name="N1", nodes=("d", "g", "0", "0"),
        device=f"nmos_rvt m={m}", analysis="",
    )
    data = ngspice_probe(RESULTS_DIR, "nmos_dc_m4", deck,
                         "dc Vgs 0 0.7 0.05", ["i(vd)"])
    vg_ng, i_ng = data[:, 0], np.abs(data[:, 1])

    from pycircuitsim.solver import DCSolver

    ps_deck = _render_device(
        model_setup=".model nmos1 NMOS (LEVEL=72)",
        drain_bias="Vd 1 0 0.5", gate_bias="Vgs 2 0 0.5",
        source_bias="", bulk_bias="", device_name="M1",
        nodes=("1", "2", "0", "0"),
        device=f"nmos1 L=30n NFIN=10 m={m}",
    )
    circuit, _ = _parse(ps_deck, "nmos_dc_m4", ASAP7_MODELCARD)
    dev = _device(circuit, "M1")
    vgs_src = _device(circuit, "Vgs")

    i_ps: List[float] = []
    for vg in vg_ng:
        vgs_src.voltage = float(vg)
        solver = DCSolver(circuit, use_source_stepping=False,
                          use_gmin_stepping=False, max_iterations=200)
        sol = solver.solve()
        i_ps.append(abs(dev.calculate_current(sol)))
    i_ps_arr = np.array(i_ps)

    errs = np.abs(i_ps_arr - i_ng) / np.maximum(i_ng, ABS_TOL_I)
    worst = float(np.max(errs))
    return report(f"DC sweep {len(vg_ng)} points vs NGSPICE",
                  worst <= REL_TOL_I,
                  f"max rel err={worst*100:.4f}% at Vgs="
                  f"{vg_ng[int(np.argmax(errs))]:.2f}V")


# ---------------------------------------------------------------------------
# 5: m=N versus N devices in parallel
# ---------------------------------------------------------------------------
def test_parallel_equivalence() -> bool:
    """A resistor-loaded NMOS: m=4 must solve identically to 4 devices."""
    print("\n  --- m=4 vs 4 parallel devices (resistor-loaded NMOS) ---")
    common = {
        "model_setup": ".model nmos1 NMOS (LEVEL=72)",
        "drain_bias": "Vdd 1 0 0.7", "gate_bias": "Vg 2 0 0.4",
        "source_bias": "", "bulk_bias": "", "nodes": ("3", "2", "0", "0"),
        "load": "Rload 1 3 20k",
    }
    deck_m = _render_device(
        **common, device_name="M1", device="nmos1 L=30n NFIN=10 m=4",
    )
    deck_p = _render_device(
        **common, device_name="M1", device="nmos1 L=30n NFIN=10",
        extra_devices="\n".join(
            f"M{i} 3 2 0 0 nmos1 L=30n NFIN=10" for i in range(2, 5)
        ),
    )

    c_m, _ = _parse(deck_m, "par_m4", ASAP7_MODELCARD)
    c_p, _ = _parse(deck_p, "par_x4", ASAP7_MODELCARD)
    s_m = _solve_dc(c_m)
    s_p = _solve_dc(c_p)
    dv = max(abs(s_m[n] - s_p[n]) for n in ("1", "2", "3"))
    i_m = abs(_device(c_m, "M1").calculate_current(s_m))
    i_p = sum(abs(_device(c_p, f"M{i}").calculate_current(s_p))
              for i in range(1, 5))
    ok = report("node voltages agree", dv <= PARALLEL_TOL_V,
                f"max|dV|={dv:.3e} V  V(3)={s_m['3']:.6f}/{s_p['3']:.6f}")
    ok &= report("total drain current agrees",
                 rel_err(i_m, i_p, ABS_TOL_I) <= 1e-6,
                 f"m=4: {i_m:.6e} A  4x: {i_p:.6e} A")
    return ok


# ---------------------------------------------------------------------------
# 6: AC at m=4 (capacitances scale)
# ---------------------------------------------------------------------------
def test_ac() -> bool:
    """Common-source amplifier at m=4: complex V(out) vs NGSPICE."""
    print("\n  --- AC common-source amplifier, m=4 ---")
    m = 4
    baked = bake_asap7(RESULTS_DIR, "nmos_rvt",
                       {"L": L, "NFIN": NFIN, "DEVTYPE": 1}, tag="ac")
    # Rload chosen so the OP sits mid-rail with m=4 (4x the current).
    cs_common = {
        "TEMP": "27", "VDD": "0.7", "INPUT_SPEC": "DC=0.4 AC=1 0",
        "LOAD_NETWORK": "Rload vdd out 5k", "BULK_NETWORK": "",
        "SOURCE_NODE": "0", "BULK_NODE": "0",
        "OUTPUT_LOAD": "Cload out 0 10f",
    }
    deck = render_template(template_deck("common_source.spice.tmpl"), {
        **cs_common, "MODEL_SETUP": f'.include "{baked}"',
        "DEVICE_PREFIX": "N", "DEVICE": f"nmos_rvt m={m}",
        "ANALYSIS": "",
    })
    data = ngspice_probe(RESULTS_DIR, "cs_ac_m4", deck,
                         "ac dec 1 1e6 1e9", ["v(out)"])
    freq = data[:, 0]
    v_ng = data[:, 1] + 1j * data[:, 2]

    from pycircuitsim.solver import ACSolver, DCSolver

    ps_deck = render_template(template_deck("common_source.spice.tmpl"), {
        **cs_common, "MODEL_SETUP": ".model nmos1 NMOS (LEVEL=72)",
        "DEVICE_PREFIX": "M", "DEVICE": f"nmos1 L=30n NFIN=10 m={m}",
        "ANALYSIS": ".ac dec 1 1e6 1e9",
    })
    circuit, _ = _parse(ps_deck, "cs_ac_m4", ASAP7_MODELCARD)
    op = DCSolver(circuit, use_source_stepping=False, use_gmin_stepping=False,
                  max_iterations=200).solve()
    ac = ACSolver(circuit, dc_solution=op).solve(freq)
    v_ps = ac["out"]

    mag_err = float(np.max(np.abs(np.abs(v_ps) - np.abs(v_ng))
                           / np.maximum(np.abs(v_ng), 1e-30)))
    ph_err = float(np.max(np.abs(np.degrees(
        np.angle(v_ps) - np.angle(v_ng)))))
    ok = report("AC |V(out)| vs NGSPICE", mag_err <= AC_MAG_TOL,
                f"max rel={mag_err*100:.4f}%  |V|(1GHz)={abs(v_ps[-1]):.6e}")
    ok &= report("AC phase(V(out)) vs NGSPICE", ph_err <= AC_PHASE_TOL,
                 f"max err={ph_err:.4f} deg")
    return ok


def main() -> int:
    print("=" * 78)
    print("BSIM-CMG instance multiplier (m=) verification vs NGSPICE 45.2")
    print(f"  ASAP7 TT, L={L*1e9:.0f}nm, NFIN={NFIN:.0f}, OSDI={OSDI_PATH.name}")
    print("=" * 78)

    results: List[Tuple[str, bool]] = [
        ("NMOS OP m=1/4/16", test_single_device_op("nmos")),
        ("PMOS OP m=1/4/16", test_single_device_op("pmos")),
        ("raw cache unscaled", test_raw_cache_is_unscaled()),
        ("DC sweep m=4", test_dc_sweep()),
        ("m=4 == 4 parallel", test_parallel_equivalence()),
        ("AC m=4", test_ac()),
    ]

    print("\n" + "=" * 78)
    n_pass = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {name:28s}: {'PASS' if ok else 'FAIL'}")
    print(f"\nRESULT: {n_pass}/{len(results)} groups PASS")
    print("=" * 78)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    parse_no_options(__doc__ or "")
    sys.exit(main())
