"""Carry the TSMC16 design points into this tree's tech.

The unified sizing strategy across TSMC5/6/7/12/16: device sizes are shared.
Every design's TSMC16 vector (``design.json`` in ``../designs_tsmc16``) is
re-expressed on this tree's tech --

* voltages (supply, common mode, references, bias voltages) scale with the
  rail, so an operating point at a fraction of VDD stays at that fraction;
* channel lengths snap into the tech's binned range (identity on TSMC12,
  a 240 -> 135 nm clamp on TSMC5);
* fin counts clamp to the tech's NFIN groups (20 -> 12 on TSMC5);
* Vt flavors the tech does not ship fall back (hvt -> svt, lnvt -> lvt);
* mirror ratios, multiplicities, capacitors, resistors and load ranges carry
  over unchanged --

and the result is emitted as a full runnable design directory.  ``finalize.py``
then measures everything against the same gates; only designs that come out
unhealthy get a per-tech re-tune (the ``polish_*`` / ``size_*`` loops).

Run from a per-tech tree:  python3 tools/port_tech.py [category ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pycmg_lib
from pycmg_lib import TECH, VDD, VT_FLAVORS, snap_l, snap_nfin

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT.parent / "designs_tsmc16"

# Vt fallback chains for flavors TSMC16 has and this tech may not.
_VT_FALLBACK = {"hvt": ("svt",), "lnvt": ("lvt", "svt"), "elvt": ("ulvt",)}


def map_vt(vt: str) -> str:
    if vt in VT_FLAVORS:
        return vt
    for alt in _VT_FALLBACK.get(vt, ()):
        if alt in VT_FLAVORS:
            return alt
    return "svt"


def snap_l_nm(l_nm: float) -> float:
    return round(snap_l(float(l_nm) * 1e-9) * 1e9)


def snap_geom(g: List) -> Tuple[float, int, int]:
    """(L[nm], NFIN, m) -> snapped into this tech's envelope, m unchanged."""
    return (snap_l_nm(g[0]), snap_nfin(g[1]), int(g[2]))


def vscale(ref_vdd: float) -> float:
    return VDD / ref_vdd


# ---------------------------------------------------------------------------
# Per-category translators
# ---------------------------------------------------------------------------
def port_amplifier(name: str) -> None:
    from build_amp import AmpDesign, RoleGeom, write_design
    from skyparse import parse_netlist

    d = AmpDesign.from_json((REF / "amplifier" / name /
                             "design.json").read_text())
    k = vscale(d.vdd)
    roles = {r: RoleGeom(vt=map_vt(g.vt), l_nm=snap_l_nm(g.l_nm),
                         nfin=snap_nfin(g.nfin), m=g.m)
             for r, g in d.roles.items()}
    new = AmpDesign(vdd=VDD, vcm=d.vcm * k, cload=d.cload,
                    gbw_ideal=d.gbw_ideal, roles=roles,
                    passives=dict(d.passives))
    topo = parse_netlist(ROOT.parent / "designs" / "amplifier" / name /
                         "netlist.spice")
    write_design(ROOT / "amplifier" / name, topo, new)


def port_ldo(name: str) -> None:
    from geom_port import parse_generic
    from ldo import LdoDesign, emit
    from size_ldo import _GND, _OUT, decks, has_resistive_divider, wiring

    d = json.loads((REF / "ldo" / name / "design.json").read_text())
    k = vscale(d["vdd"])
    passives = {}
    for key, val in d["passives"].items():
        if key == "vref" or key.startswith("bias_"):
            passives[key] = float(val) * k
        else:
            passives[key] = float(val)
    design = LdoDesign(
        vdd=VDD, vref=float(d["vref"]) * k, vt=map_vt(d["vt"]),
        geoms={key: snap_geom(g) for key, g in d["geoms"].items()},
        passives=passives,
        vt_overrides={key: map_vt(v)
                      for key, v in d.get("vt_overrides", {}).items()})

    subs, top = parse_generic(ROOT.parent / "designs" / "ldo" / name /
                              "netlist.spice")
    sub = next(s for s in subs if s.mos)
    out_nodes = [p.lower() for p in sub.ports if _OUT.match(p)]
    has_div = has_resistive_divider(subs, out_nodes)

    out = ROOT / "ldo" / name
    emit(out, subs, top, design, sub.name, sub.ports, out_nodes)
    for deck, (text, _control) in decks(sub, design, has_div).items():
        (out / deck).write_text(text)


def _keyed(geoms: Dict, dev_key: str, group_key: str) -> str:
    if dev_key in geoms:
        return dev_key
    return group_key


def port_sfe(name: str) -> None:
    import sfe_amp
    if name == sfe_amp.NAME:
        # The amplifier shipped in this category: voltages scale with the
        # rail, geometry snaps into this tech's envelope, and the mirror
        # ratios, bias current and RC compensation carry over unchanged.
        from dataclasses import replace
        d = sfe_amp.SfeAmpDesign.from_json(
            (REF / "sensing_front_end" / name / "design.json").read_text())
        k = vscale(d.vdd)
        groups = {g: replace(geom, vt=map_vt(geom.vt),
                             l_nm=snap_l_nm(geom.l_nm),
                             nfin=snap_nfin(geom.nfin))
                  for g, geom in d.groups.items()}
        sfe_amp.emit(ROOT / "sensing_front_end" / name,
                     replace(d, vdd=VDD, vcm=d.vcm * k, groups=groups))
        return

    from geom_port import parse_generic
    from sfe import SfeDesign, emit, group_key

    d = json.loads((REF / "sensing_front_end" / name /
                    "design.json").read_text())
    design = SfeDesign(
        vdd=VDD, vt=map_vt(d["vt"]),
        geoms={key: snap_geom(g) for key, g in d["geoms"].items()},
        vts={key: map_vt(v) for key, v in d.get("vts", {}).items()})

    subs, _ = parse_generic(ROOT.parent / "designs" / "sensing_front_end" /
                            name / "netlist.spice")
    sub = next(s for s in subs if s.mos)
    key_fn = lambda m: _keyed(design.geoms, m.name.lstrip("xX").lower(),
                              group_key(m))
    emit(ROOT / "sensing_front_end" / name, subs, design, sub.ports, sub.name,
         key_fn=key_fn)


def port_vref(name: str) -> None:
    from geom_port import parse_generic
    from size_vref import VrefDesign, emit, outputs_of, supply_nets

    d = json.loads((REF / "voltage_reference" / name /
                    "design.json").read_text())
    design = VrefDesign(
        vdd=VDD, vt=map_vt(d["vt"]),
        geoms={key: snap_geom(g) for key, g in d["geoms"].items()},
        vts={key: map_vt(v) for key, v in d.get("vts", {}).items()})

    subs, _ = parse_generic(ROOT.parent / "designs" / "voltage_reference" /
                            name / "netlist.spice")
    vdd_net, gnd_net = supply_nets(subs)
    outs = outputs_of(subs)
    key_fn = lambda m: _keyed(design.geoms, m.name.lstrip("xX").lower(),
                              m.group)
    emit(ROOT / "voltage_reference" / name, subs, design, name, vdd_net,
         gnd_net, outs, key_fn=key_fn)
    # finalize.py recovers the output list from result.json when present;
    # seed it so the verdict never sees an empty output list.
    rj = ROOT / "voltage_reference" / name / "result.json"
    if not rj.exists():
        rj.write_text(json.dumps({"design": name, "outputs": outs}, indent=2))


def port_cp(name: str) -> None:
    from size_cp import CpDesign, emit, load

    d = json.loads((REF / "charge_pump" / name / "design.json").read_text())
    design = CpDesign(
        vdd=VDD, vcp=0.5 * VDD, vt=map_vt(d["vt"]),
        geoms={key: snap_geom(g) for key, g in d["geoms"].items()})
    subs, _params = load()
    emit(ROOT / "charge_pump" / name, subs, design)


PORTERS = {
    "amplifier": port_amplifier,
    "ldo": port_ldo,
    "sensing_front_end": port_sfe,
    "voltage_reference": port_vref,
    "charge_pump": port_cp,
}


def main() -> None:
    if TECH == "TSMC16" and REF == ROOT:
        raise SystemExit("port_tech.py runs in a per-tech tree, not the "
                         "TSMC16 reference tree")
    wanted = sys.argv[1:] or list(PORTERS)
    n_ok = n_fail = 0
    for cat in wanted:
        cdir = REF / cat
        if not cdir.is_dir():
            continue
        for d in sorted(cdir.iterdir()):
            if not (d / "design.json").exists():
                continue
            try:
                PORTERS[cat](d.name)
                n_ok += 1
                print(f"  ported {cat}/{d.name}", flush=True)
            except Exception as exc:
                n_fail += 1
                print(f"  FAIL   {cat}/{d.name}: {type(exc).__name__}: {exc}",
                      flush=True)
    print(f"\n{TECH}: {n_ok} ported, {n_fail} failed")


if __name__ == "__main__":
    main()
