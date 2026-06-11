#!/usr/bin/env python3
"""S6 artifact-mechanism probe: native LEVEL=72 device vs the S6/P0-I
injection dict, key by key, AT THE CLASS-METHOD LEVEL the solver consumes.

The native L72 RO reproduces NGSPICE (46.64 vs 46.65 ps) while the
exact-OSDI injection through the monkeypatched LEVEL=73 device gives
93.01 ps (ratio 1.994). Both feed the SAME stamps, so the guilty quantity
must show up as a sign flip / factor in what the solver calls:
calculate_current, get_conductance, get_charges, get_capacitances.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "external_compact_models" / "PyCMG" / "tests",
          ROOT / "external_compact_models" / "PyCMG",
          ROOT / "external_compact_models",
          ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    from tests.common.complex import BENCH, render_directnet_netlist, \
        parse_netlist
    from pycircuitsim.solver import _is_mosfet

    ctl = load_mod("s6ctl", ROOT / "scripts" / "v6_4_7_s6_l72_ro_control.py")
    swp = load_mod("s6swp", ROOT / "scripts" / "v6_4_7_s6_p1_swap_matrix.py")

    bt = BENCH["TSMC7"]
    work = ROOT / "results" / "v6_4_7" / "s6_logs" / "artifact_probe"
    work.mkdir(parents=True, exist_ok=True)

    # --- native L72 components (the 46.64 ps path) ---
    card = ctl.build_merged_card(bt, work)
    merged = card[0] if isinstance(card, tuple) else card
    l72_parse = ctl.make_l72_parse(merged, bt)
    l72_net = ctl.render_l72_netlist(bt, 2e-12, 0.6e-9,
                                     work / "ro_l72_probe.sp")
    l72 = l72_parse(l72_net)
    l72_devs = {c.name: c for c in l72.circuit.components if _is_mosfet(c)}

    # --- injected LEVEL=73 components (the 93.01 ps path) ---
    dn_net = render_directnet_netlist(
        ROOT / "examples" / "complex" / "ring_osc_5stage_directnet.sp",
        bt, work / "ro_dn_probe.sp")
    dn = parse_netlist(dn_net)
    dn_devs = {c.name: c for c in dn.circuit.components if _is_mosfet(c)}
    cache = swp.OsdiCache(bt)
    patch = swp.make_patched_eval(cache, swp.SwapStats(0), {"n": 0}) \
        if hasattr(swp, "make_patched_eval") else None

    # biases: (label, vd, vg, vs, vb) absolute — NMOS frame; PMOS mirrored
    vdd = bt.vdd
    biases = [
        ("off",      0.0,     0.0,   0.0, 0.0),
        ("on-sat",   vdd,     vdd,   0.0, 0.0),
        ("out-hi",   vdd,     0.0,   0.0, 0.0),
        ("mid",      vdd / 2, vdd / 2, 0.0, 0.0),
        ("pulling",  0.2,     vdd,   0.0, 0.0),
    ]

    # pick one NMOS + one PMOS from each circuit (stage-1 devices)
    pairs = [("NMOS", "Mn1"), ("PMOS", "Mp1")]
    for dtype, mname in pairs:
        l72_m = l72_devs[mname]
        dn_m = dn_devs[mname]
        is_p = dtype == "PMOS"
        print(f"\n{'=' * 90}\n### {dtype} ({mname})  L72-native vs "
              f"injected-OSDI-dict (solver-consumed forms)")
        for label, vd0, vg0, vs0, vb0 in biases:
            if is_p:  # mirror into PMOS frame: source/bulk at VDD
                vd, vg, vs, vb = vdd - vd0, vdd - vg0, vdd, vdd
            else:
                vd, vg, vs, vb = vd0, vg0, vs0, vb0
            volt_l72 = {l72_m.nodes[0]: vd, l72_m.nodes[1]: vg,
                        l72_m.nodes[2]: vs, l72_m.nodes[3]: vb}
            volt_dn = {dn_m.nodes[0]: vd, dn_m.nodes[1]: vg,
                       dn_m.nodes[2]: vs, dn_m.nodes[3]: vb}
            l72_m.clear_cache()
            i_l72 = l72_m.calculate_current(volt_l72)
            g_l72 = l72_m.get_conductance(volt_l72)
            q_l72 = l72_m.get_charges(volt_l72)
            c_l72 = l72_m.get_capacitances(volt_l72)

            op = cache.consistent_op(is_p, dn_m.L, vd, vg, vs, vb)
            if op is None:
                print(f"  [{label}] OSDI eval FAILED")
                continue

            print(f"  [{label:8s}] bias d/g/s/b = "
                  f"{vd:.3f}/{vg:.3f}/{vs:.3f}/{vb:.3f}")
            print(f"    id     : L72 calculate_current={i_l72:+.4e}   "
                  f"inj dict id={op['id']:+.4e}")
            print(f"    gds    : L72={g_l72[0]:+.4e}  inj={op['gds']:+.4e}"
                  f"   | gm: L72={g_l72[1]:+.4e}  inj={op['gm']:+.4e}"
                  f"   | gmb: L72={g_l72[2]:+.4e}  inj={op['gmb']:+.4e}")
            for k in ("qg", "qd", "qs", "qb"):
                r = (op[k] / q_l72[k]) if abs(q_l72[k]) > 1e-25 else float("nan")
                print(f"    {k:3s}    : L72={q_l72[k]:+.4e}  inj={op[k]:+.4e}"
                      f"  ratio={r:+.3f}")
            for k in ("cgg", "cgd", "cgs", "cdg", "cdd"):
                r = (op[k] / c_l72[k]) if abs(c_l72[k]) > 1e-25 else float("nan")
                print(f"    {k:3s}    : L72={c_l72[k]:+.4e}  inj={op[k]:+.4e}"
                      f"  ratio={r:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
