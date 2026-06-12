#!/usr/bin/env python3
"""S7/P2 self-check: corrected-id continuity across Vds=0, exact zero at
Vds=0, reverse-sign correctness, gds positivity, taper roll-off, and the
SRAM restoring-device recovery spot-check (vs the S7 probe's raw/OSDI rows).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "external_compact_models" / "PyCMG" / "tests",
          ROOT / "external_compact_models" / "PyCMG",
          ROOT / "external_compact_models",
          ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tests.common.complex import BENCH, render_directnet_netlist, \
    parse_netlist  # noqa: E402
from pycircuitsim.solver import _is_mosfet  # noqa: E402

FAIL = 0


def check(cond: bool, msg: str) -> None:
    global FAIL
    if not cond:
        FAIL += 1
        print(f"  FAIL: {msg}")


def main() -> int:
    bt = BENCH["TSMC7"]
    work = ROOT / "results" / "v6_4_7" / "s7_logs" / "selfcheck"
    work.mkdir(parents=True, exist_ok=True)
    net = render_directnet_netlist(
        ROOT / "examples" / "complex" / "ring_osc_5stage_directnet.sp",
        bt, work / "ro_probe.sp")
    devs = {c.name: c for c in
            parse_netlist(net).circuit.components if _is_mosfet(c)}

    for mname, is_p in (("Mn1", False), ("Mp1", True)):
        m = devs[mname]
        vdd = bt.vdd
        print(f"\n### {mname} ({'PMOS' if is_p else 'NMOS'}), "
              f"VDD_train={m._vdd_estimate:.3f}")
        for vgs_f in (0.3, 0.6, 0.9):
            vg_eff = vgs_f * vdd
            vds_grid = np.linspace(-0.5 * vdd, 0.5 * vdd, 501)
            ids, gdss = [], []
            for vds_eff in vds_grid:
                # absolute terminals: source/bulk at 0 (NMOS) or VDD (PMOS)
                if is_p:
                    volt = {m.nodes[0]: vdd + vds_eff * -1.0,
                            m.nodes[1]: vdd - vg_eff,
                            m.nodes[2]: vdd, m.nodes[3]: vdd}
                    # PMOS device frame: vds_dev = vd - vs = -vds_eff... use
                    # direct mapping: choose vd so that device vds = vds_eff
                    volt[m.nodes[0]] = vdd + vds_eff
                else:
                    volt = {m.nodes[0]: vds_eff, m.nodes[1]: vg_eff,
                            m.nodes[2]: 0.0, m.nodes[3]: 0.0}
                m.clear_cache()
                r = m._eval(volt)
                ids.append(r["id"])
                gdss.append(r["gds"])
            ids_a = np.array(ids)
            gds_a = np.array(gdss)
            i0 = len(vds_grid) // 2          # vds = 0
            check(ids_a[i0] == 0.0,
                  f"vgs={vgs_f}: id(vds=0) = {ids_a[i0]:.3e} != 0")
            jumps = np.abs(np.diff(ids_a))
            scale = max(np.max(np.abs(ids_a)), 1e-12)
            check(float(np.max(jumps)) < 0.05 * scale,
                  f"vgs={vgs_f}: id jump {np.max(jumps):.3e} "
                  f"(>{0.05 * scale:.3e}) at vds="
                  f"{vds_grid[int(np.argmax(jumps))]:.4f}")
            check(bool(np.all(gds_a > 0.0)),
                  f"vgs={vgs_f}: non-positive gds (min {np.min(gds_a):.3e})")
            # reverse-sign correctness: NMOS reverse id >= 0, PMOS <= 0
            rev = vds_grid < 0 if not is_p else vds_grid > 0
            rev_ids = ids_a[rev]
            bad = (rev_ids < -1e-12).sum() if not is_p \
                else (rev_ids > 1e-12).sum()
            check(int(bad) == 0,
                  f"vgs={vgs_f}: {bad} wrong-sign reverse points")
            # taper: beyond 0.40*VDD_train reverse, id must be ~0
            deep = np.abs(vds_grid) > 0.42 * m._vdd_estimate
            deep_rev = deep & rev
            if deep_rev.any():
                check(float(np.max(np.abs(ids_a[deep_rev]))) < 1e-12,
                      f"vgs={vgs_f}: taper leak "
                      f"{np.max(np.abs(ids_a[deep_rev])):.3e}")
            n_rev_on = int((np.abs(rev_ids) > 1e-6).sum())
            print(f"  vgs={vgs_f:.1f}: id(0)=0 ok, max|id|="
                  f"{scale * 1e6:.1f}uA, reverse pts conducting "
                  f"(|id|>1uA): {n_rev_on}, gds range "
                  f"[{np.min(gds_a):.2e}, {np.max(gds_a):.2e}] S")

    # SRAM restoring-device recovery spot check (S7 probe row, TSMC7 Mpr:
    # device vds=+34.6 mV reverse for PMOS, raw -10.18 uA, OSDI -13.71 uA)
    m = devs["Mp1"]
    vdd = bt.vdd
    vds_dev = +0.0346
    volt = {m.nodes[0]: vdd + vds_dev, m.nodes[1]: 0.0,
            m.nodes[2]: vdd, m.nodes[3]: vdd}
    m.clear_cache()
    r = m._eval(volt)
    print(f"\n### TSMC7 Mpr-like recovery: corrected id={r['id'] * 1e6:+.2f}uA"
          f" (raw was -10.18, OSDI -13.71), gds={r['gds']:.3e} S")
    check(r["id"] < -1e-6, "recovery: corrected reverse id not conducting")
    check(r["gds"] > 5e-5, "recovery: gds below the NR-fold cure scale")

    print(f"\nRESULT: {'PASS' if FAIL == 0 else f'{FAIL} CHECKS FAILED'}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
