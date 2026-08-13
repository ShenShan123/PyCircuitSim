"""Port and size the AnalogGym charge pump for this tree's tech.

The shipped netlist is a whole PLL -- VCO, PFD, dividers, loop filter, lock
detector -- but its transient bench reaches exactly one block: ``xi0``, an
instance of ``PLL_CHARGEPUMP`` driven by ideal current references and pulsed
up/dn inputs, with the output node clamped by a 1 GF capacitor so the ammeters
read the source and sink currents directly.  Only that block and the
``PLL_QUENCH_v33`` it instantiates are ported; the rest is unreachable from this
measurement and is not invented.

The source runs on 3.3 V I/O devices (nod33ll/pod33ll).  The TSMC PDK as PyCMG
exposes it here has core transistors only, so the pump is re-designed for the
core rail:
the clamp voltage moves to mid-rail and the up/dn swing follows the supply.
What the measurement is about -- how well the source and sink currents match --
is preserved.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geom_port import GSubckt, parse_generic, parse_params
from meas import SimError, run_deck
import pycmg_lib
from pycmg_lib import (L_CHOICES_LONG, MODELS_FILE, TECH, ModelLibrary,
                       scale_l)

ROOT = Path(__file__).resolve().parents[1]
SKY = ROOT.parent / "designs" / "charge_pump" / "chargepump"

KEEP = ("PLL_CHARGEPUMP", "PLL_QUENCH_v33")
VDD = pycmg_lib.VDD
VCP = 0.5 * VDD            # output clamp, mid-rail
IREF10, IREF5 = 10e-6, 5e-6
L_CHOICES = list(L_CHOICES_LONG)
NFIN_CHOICES = list(pycmg_lib.NFIN_CHOICES)
BAD = 1e3


@dataclass
class CpDesign:
    vdd: float = VDD
    vcp: float = VCP
    vt: str = "svt"
    geoms: Dict[str, Tuple[float, int, int]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"vdd": self.vdd, "vcp": self.vcp, "vt": self.vt,
                           "iref_10u": IREF10, "iref_5u": IREF5,
                           "geoms": {k: list(v) for k, v in self.geoms.items()}},
                          indent=2, sort_keys=True)


def load() -> Tuple[List[GSubckt], Dict[str, float]]:
    params = parse_params(SKY / "param.spice")
    subs, _top = parse_generic(SKY / "netlist.spice", params)
    keep = [s for s in subs if s.name in KEEP]
    if len(keep) != len(KEEP):
        raise RuntimeError(f"missing charge-pump blocks: "
                           f"{set(KEEP) - {s.name for s in keep}}")
    return keep, params


def initial_design(subs: List[GSubckt]) -> CpDesign:
    geoms: Dict[str, Tuple[float, int, int]] = {}
    for s in subs:
        for m in s.mos:
            if m.group in geoms:
                continue
            # W/L ratio carried across, expressed as fin count x multiplicity.
            ratio = max(0.1, m.w / m.l)
            geoms[m.group] = (scale_l(120), 4,
                              max(1, min(500, int(round(ratio)))))
    return CpDesign(geoms=geoms)


def emit(out_dir: Path, subs: List[GSubckt], design: CpDesign) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lib = ModelLibrary()
    lines = [f"* PLL_CHARGEPUMP -- AnalogGym charge pump on {TECH} BSIM-CMG",
             "* Only the block the transient bench reaches is ported; the rest",
             "* of the shipped PLL is unreachable from this measurement."]
    for s in subs:
        lines.append(f".subckt {s.name} {' '.join(s.ports)}")
        for m in s.mos:
            l_nm, nfin, mult = design.geoms[m.group]
            model = lib.model_name(m.kind, design.vt, l_nm * 1e-9, nfin)
            lines.append(f"N{m.name.lstrip('xX')} {' '.join(m.nodes)} "
                         f"{model} m={mult}")
        for p in s.passives:
            lines.append(f"{p.name} {' '.join(p.nodes)} {p.value}")
        for name, nodes, line in s.insts:
            lines.append(line)
        lines.append(f".ends {s.name}")

    lib.write(out_dir / MODELS_FILE)
    (out_dir / "netlist.spice").write_text("\n".join(lines) + "\n")
    (out_dir / "design.json").write_text(design.to_json())

    v, cp = design.vdd, design.vcp
    (out_dir / "tb_tran.cir").write_text(f"""\
* PLL_CHARGEPUMP -- AnalogGym charge-pump transient bench on {TECH}
* Measurement set follows the shipped tran_27corner.sp nominal corner: average
* source (up) and sink (down) current and their range over the pulse train.
* vcp_net is clamped by the original 1 GF load so the ammeters inside the block
* read the pump currents directly.
.include ./{MODELS_FILE}
.include ./netlist.spice

.param vdd = {v:g}
.param vcp = {cp:g}
.temp 25

xi0 vcp_net dn_net dnb_net i10u_net i5u_net 0 vdd_net up_net upb_net vdd_net 0 PLL_CHARGEPUMP
i10 vdd_net i10u_net DC={IREF10:g}
i5  vdd_net i5u_net  DC={IREF5:g}
v0  vdd_net 0 DC='vdd'
c0  vcp_net 0 1e9
.ic v(vcp_net)='vcp'

vdn  dn_net  0 PULSE 0 '{v:g}' 10e-9 10e-12 10e-12 10e-9 20e-9
vdnb dnb_net 0 PULSE 0 '{v:g}' 0     10e-12 10e-12 10e-9 20e-9
vupb upb_net 0 PULSE 0 '{v:g}' 0     10e-12 10e-12 10e-9 20e-9
vup  up_net  0 PULSE 0 '{v:g}' 10e-9 10e-12 10e-12 10e-9 20e-9

.measure tran up_imin min i(v.xi0.vupper) from=20e-9 to=100e-9
.measure tran up_iavg avg i(v.xi0.vupper) from=20e-9 to=180e-9
.measure tran up_imax max i(v.xi0.vupper) from=20e-9 to=100e-9
.measure tran lo_imin min i(v.xi0.vlower) from=20e-9 to=100e-9
.measure tran lo_iavg avg i(v.xi0.vlower) from=20e-9 to=180e-9
.measure tran lo_imax max i(v.xi0.vlower) from=20e-9 to=100e-9
.end
""")


CONTROL = "tran 2p 200n"
CONTROL_FAST = "tran 10p 200n"


def cp_score(m: Dict) -> float:
    """Penalty: both currents must flow, and match each other."""
    up, lo = m.get("up_iavg"), m.get("lo_iavg")
    if up is None or lo is None:
        return 50.0
    up, lo = abs(up), abs(lo)
    if up < 1e-7 or lo < 1e-7:
        return 20.0 + (1e-7 - min(up, lo)) * 1e7
    mismatch = abs(up - lo) / max(up, lo)
    pen = mismatch / 0.05                       # target: within 5 %
    # Keep the current in a range a PLL would actually use.
    for i in (up, lo):
        if i < 2e-6:
            pen += (2e-6 - i) / 2e-6
        if i > 200e-6:
            pen += (i - 200e-6) / 200e-6
    return max(0.0, pen)


def cp_report(m: Dict) -> Dict[str, bool]:
    up, lo = m.get("up_iavg"), m.get("lo_iavg")
    if up is None or lo is None:
        return {"currents": False, "match": False, "in_range": False}
    up, lo = abs(up), abs(lo)
    mism = abs(up - lo) / max(up, lo) if max(up, lo) > 0 else 1.0
    return {"currents": up > 1e-7 and lo > 1e-7,
            "match": mism <= 0.05,
            "in_range": 2e-6 <= up <= 200e-6 and 2e-6 <= lo <= 200e-6}


def _vec(d: CpDesign, keys: List[str]) -> List[float]:
    v: List[float] = []
    for k in keys:
        l_nm, nfin, m = d.geoms[k]
        v += [l_nm, float(nfin), float(m)]
    return v


def _unvec(v: List[float], d: CpDesign, keys: List[str]) -> CpDesign:
    geoms = {}
    for i, k in enumerate(keys):
        geoms[k] = (float(min(L_CHOICES, key=lambda c: abs(c - v[3 * i]))),
                    int(min(NFIN_CHOICES, key=lambda c: abs(c - v[3 * i + 1]))),
                    max(1, min(2000, int(round(v[3 * i + 2])))))
    return replace(d, geoms=geoms)


def size(max_evals: int = 150) -> dict:
    subs, _params = load()
    keys = list(dict.fromkeys(m.group for s in subs for m in s.mos))
    work = ROOT / "work" / "charge_pump" / "chargepump"
    out = ROOT / "charge_pump" / "chargepump"
    work.mkdir(parents=True, exist_ok=True)

    def evaluate(d: CpDesign, where: Path, control: str, timeout: float):
        try:
            emit(where, subs, d)
            m = run_deck(where / "tb_tran.cir", control, where, "tran",
                         timeout=timeout)
        except Exception:
            return BAD, {}
        return cp_score(m), m

    best = initial_design(subs)
    best_score, best_m = evaluate(best, work, CONTROL_FAST, 60)
    evals = 1
    step = [2.0, 1.6, 2.5] * len(keys)
    t0 = time.time()

    while evals < max_evals and best_score > 0 and max(step) > 1.05:
        improved = False
        vec = _vec(best, keys)
        for i in range(len(vec)):
            if evals >= max_evals:
                break
            for direction in (+1, -1):
                cv = list(vec)
                cv[i] *= step[i] ** direction
                cand = _unvec(cv, best, keys)
                if _vec(cand, keys) == _vec(best, keys):
                    continue
                s, m = evaluate(cand, work, CONTROL_FAST, 60)
                evals += 1
                if s < best_score - 1e-9:
                    best, best_score, best_m = cand, s, m
                    improved = True
                    vec = _vec(best, keys)
                    break
        if not improved:
            step = [1 + (s - 1) / 2 for s in step]

    elapsed = time.time() - t0
    emit(out, subs, best)
    err = ""
    try:
        final = run_deck(out / "tb_tran.cir", CONTROL, out, "tran", timeout=1800)
    except SimError as exc:
        final, err = {}, str(exc).splitlines()[0]

    up, lo = final.get("up_iavg"), final.get("lo_iavg")
    result = {"design": "chargepump", "evals": evals,
              "evals_seconds": round(elapsed, 1),
              "score": cp_score(final) if final else BAD,
              "mismatch_pct": (abs(abs(up) - abs(lo)) / max(abs(up), abs(lo)) * 100
                               if up and lo else None),
              "metrics": final, "pass": cp_report(final), "error": err}
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    r = size(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
    print(f"chargepump score={r['score']:.3f} pass={sum(r['pass'].values())}/3 "
          f"up={r['metrics'].get('up_iavg')} lo={r['metrics'].get('lo_iavg')} "
          f"mismatch={r['mismatch_pct']}")
    outp = ROOT / "results" / "cp_sizing.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(r, indent=2, default=str))
