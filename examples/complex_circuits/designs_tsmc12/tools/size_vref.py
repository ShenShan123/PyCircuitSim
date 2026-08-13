"""Port and size the MOS-only AnalogGym voltage references for TSMC FinFET.

Two of AnalogGym's four references are all-transistor sub-threshold designs and
port cleanly.  The other two do not and are not attempted: ``bandgap_vref``
needs an NPN and ``subthreshold_vref`` a PNP, and BSIM-CMG models a FinFET --
there is no bipolar in this TSMC view to bind them to.

A reference wants the opposite of the PTAT sensors: the sub-threshold
temperature terms have to cancel rather than add, so the objective is the
temperature coefficient in ppm/C, with the output held in a usable window.
"""

from __future__ import annotations

import json
import re
import sys
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from geom_port import GMos, GSubckt, parse_generic
from meas import SimError, run_deck
import pycmg_lib
from pycmg_lib import (L_CHOICES_LONG, L_MAX_NM, MODELS_FILE, TECH, VT_FLAVORS,
                       ModelLibrary)

ROOT = Path(__file__).resolve().parents[1]
SKY = ROOT.parent / "designs" / "voltage_reference"

# V7.5.6: three_output_vref removed from the basket (its MOS core is
# byte-identical to the dual-output core; the third output was an ideal
# 1e18-ohm divider onto a node never qualified for load drive).
PORTABLE = ["dual_output_subthreshold_vref"]
BLOCKED = {"bandgap_vref": "needs an NPN (vnpn_0p54x2_sm062)",
           "subthreshold_vref": "needs a PNP (pbhvnwpsub2_ga)"}

VDD = pycmg_lib.VDD
TMIN, TMAX = -40, 125
L_CHOICES = list(L_CHOICES_LONG)
NFIN_CHOICES = list(pycmg_lib.NFIN_CHOICES)
BAD = 1e3


@dataclass
class VrefDesign:
    vdd: float = VDD
    vt: str = "svt"
    geoms: Dict[str, Tuple[float, int, int]] = field(default_factory=dict)
    # Optional per-group override of the global vt.  Sub-threshold references
    # trade on Vt *differences*, so letting device groups mix flavors is a
    # first-class design knob; empty means the global vt everywhere.
    vts: Dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"vdd": self.vdd, "vt": self.vt,
                           "vts": self.vts,
                           "geoms": {k: list(v) for k, v in self.geoms.items()}},
                          indent=2, sort_keys=True)


def outputs_of(subs: List[GSubckt]) -> List[str]:
    """Reference output nets, by name."""
    nets = set()
    for s in subs:
        for m in s.mos:
            nets.update(n.lower() for n in m.nodes)
        for p in s.passives:
            nets.update(n.lower() for n in p.nodes)
    return sorted(n for n in nets if re.fullmatch(r"vref\d*", n))


def supply_nets(subs: List[GSubckt]) -> Tuple[str, str]:
    nets = set()
    for s in subs:
        for m in s.mos:
            nets.update(n.lower() for n in m.nodes)
    vdd = next((n for n in nets if n in ("vdda", "vdd")), "vdd")
    gnd = next((n for n in nets if n in ("gnda", "gnd", "vss")), "0")
    return vdd, gnd


def initial_design(mos: List[GMos]) -> VrefDesign:
    geoms: Dict[str, Tuple[float, int, int]] = {}
    for m in mos:
        if m.group in geoms:
            continue
        # Longest available channel by default: sub-threshold slope and output
        # resistance are what these circuits trade on.
        geoms[m.group] = (float(L_MAX_NM), 4,
                          max(1, min(200, int(round(m.mult)))))
    return VrefDesign(geoms=geoms)


def emit(out_dir: Path, subs: List[GSubckt], design: VrefDesign,
         name: str, vdd_net: str, gnd_net: str, outs: List[str],
         key_fn=None) -> None:
    """*key_fn* maps a device to its geometry/vt key (default: matched group).

    The polish pass keys per device: these references come from templates
    where unrelated devices share one placeholder W/L, so the matched group
    welds e.g. a reference stack to the amplifier tail that happens to share
    its source geometry."""
    key_fn = key_fn or (lambda m: m.group)
    out_dir.mkdir(parents=True, exist_ok=True)
    lib = ModelLibrary()
    lines = [f"* {name} -- AnalogGym voltage reference on {TECH} BSIM-CMG",
             f"* Topology is the shipped one; geometry is re-designed for "
             f"{design.vdd:g} V."]
    for sub in subs:
        # The design's ground net is renamed to node 0 rather than tied to it
        # with a 0 V source -- ngspice rejects that as a shorted VSRC.
        def gnd(n: str) -> str:
            """Map ground and export the stable first tap of the 3-output core."""
            low = n.lower()
            if low == gnd_net:
                return "0"
            if name == "three_output_vref":
                if low == "vref1":
                    return "vtop"
                if low == "net3":
                    return "vref1"
            return n
        for m in sub.mos:
            key = key_fn(m)
            l_nm, nfin, mult = design.geoms[key]
            model = lib.model_name(m.kind, design.vts.get(key, design.vt),
                                   l_nm * 1e-9, nfin)
            lines.append(f"N{m.name.lstrip('xX')} "
                         f"{' '.join(gnd(n) for n in m.nodes)} "
                         f"{model} m={mult}")
        for p in sub.passives:
            lines.append(f"{p.name} {' '.join(gnd(n) for n in p.nodes)} "
                         f"{p.value}")

    lib.write(out_dir / MODELS_FILE)
    (out_dir / "netlist.spice").write_text("\n".join(lines) + "\n")
    (out_dir / "design.json").write_text(design.to_json())

    meas = []
    for o in outs:
        meas += [
            f".measure dc {o}_at25 FIND v({o}) AT=25",
            f".measure dc {o}_max  MAX  v({o}) from={TMIN} to={TMAX}",
            f".measure dc {o}_min  MIN  v({o}) from={TMIN} to={TMAX}",
            f".measure dc {o}_pp   PP   v({o}) from={TMIN} to={TMAX}",
            f".measure dc {o}_avg  AVG  v({o}) from={TMIN} to={TMAX}",
            f".measure dc {o}_tc   param='{o}_pp/{o}_avg/{TMAX - TMIN}*1e6'",
        ]
    gnd_line = ""
    (out_dir / "tb_dc.cir").write_text(
        f"* {name} -- AnalogGym voltage reference temperature bench on {TECH}\n"
        f"* Measurement set is the shipped one; TC is in ppm/C.\n"
        f".include ./{MODELS_FILE}\n"
        f".include ./netlist.spice\n\n"
        f"* These references run on picoamp sub-threshold currents; ngspice's\n"
        f"* default 1 pA ABSTOL is larger than the currents themselves, and the\n"
        f"* temperature sweep aborts before it reaches 25 C.\n"
        f".options abstol=1e-16 gmin=1e-15 reltol=1e-4\n"
        f".nodeset {' '.join(f'V({o})={0.375 * VDD:g}' for o in outs)}\n\n"
        f".PARAM supply_voltage = {design.vdd:g}\n"
        f"V{vdd_net} {vdd_net} 0 'supply_voltage'\n{gnd_line}\n"
        + "\n".join(meas) + "\n.end\n"
    )


DC_CONTROL = f"dc temp {TMIN} {TMAX} 0.5"
# Step must land exactly on 25 C: the at25 measurements read that point.
DC_CONTROL_FAST = f"dc temp {TMIN} {TMAX} 5"

TC_TARGET = 500.0          # ppm/C
VREF_LO, VREF_HI = 0.08, 0.875 * VDD


def vref_score(m: Dict, outs: List[str]) -> float:
    pen = 0.0
    for o in outs:
        tc, v25 = m.get(f"{o}_tc"), m.get(f"{o}_at25")
        if tc is None or v25 is None:
            pen += 20.0
            continue
        pen += max(0.0, (abs(tc) - TC_TARGET) / TC_TARGET)
        if v25 < VREF_LO:
            pen += (VREF_LO - v25) / 0.05
        if v25 > VREF_HI:
            pen += (v25 - VREF_HI) / 0.05
    return pen


def vref_report(m: Dict, outs: List[str]) -> Dict[str, bool]:
    return {
        "outputs": all(m.get(f"{o}_at25") is not None for o in outs),
        "tc": all(m.get(f"{o}_tc") is not None and abs(m[f"{o}_tc"]) <= TC_TARGET
                  for o in outs),
        "in_range": all(m.get(f"{o}_at25") is not None
                        and VREF_LO <= m[f"{o}_at25"] <= VREF_HI for o in outs),
    }


def _vec(d: VrefDesign, keys: List[str]) -> List[float]:
    v: List[float] = []
    for k in keys:
        l_nm, nfin, m = d.geoms[k]
        v += [l_nm, float(nfin), float(m)]
    return v


def _unvec(v: List[float], d: VrefDesign, keys: List[str]) -> VrefDesign:
    geoms: Dict[str, Tuple[float, int, int]] = {}
    for i, k in enumerate(keys):
        geoms[k] = (float(min(L_CHOICES, key=lambda c: abs(c - v[3 * i]))),
                    int(min(NFIN_CHOICES, key=lambda c: abs(c - v[3 * i + 1]))),
                    max(1, min(4000, int(round(v[3 * i + 2])))))
    return replace(d, geoms=geoms)


def _seeded(base: VrefDesign, seed: int) -> VrefDesign:
    """Perturbed starting point -- see the note in size_sfe._seeded."""
    if seed == 0:
        return base
    rng = random.Random(seed)
    geoms = {k: (float(rng.choice(L_CHOICES)), int(rng.choice(NFIN_CHOICES)),
                 max(1, min(4000, int(m * rng.choice([0.25, 0.5, 1, 2, 4])))))
             for k, (l, n, m) in base.geoms.items()}
    return VrefDesign(vdd=base.vdd, geoms=geoms,
                      vt=rng.choice([v for v in ("svt", "lvt", "ulvt", "hvt")
                                     if v in VT_FLAVORS]))


def size(name: str, max_evals: int = 250, restarts: int = 1) -> dict:
    subs, _top = parse_generic(SKY / name / "netlist.spice")
    mos = [m for s in subs for m in s.mos]
    outs = outputs_of(subs)
    vdd_net, gnd_net = supply_nets(subs)
    keys = list(dict.fromkeys(m.group for m in mos))

    work = ROOT / "work" / "voltage_reference" / name
    out = ROOT / "voltage_reference" / name
    work.mkdir(parents=True, exist_ok=True)

    def evaluate(d: VrefDesign, where: Path, control: str, timeout: float):
        try:
            emit(where, subs, d, name, vdd_net, gnd_net, outs)
            m = run_deck(where / "tb_dc.cir", control, where, "dc",
                         timeout=timeout)
        except Exception:
            return BAD, {}
        return vref_score(m, outs), m

    base0 = initial_design(mos)
    best, best_score, best_m = None, float("inf"), {}
    for seed in range(restarts):
        start = _seeded(base0, seed)
        s0, m0 = evaluate(start, work, DC_CONTROL_FAST, 30)
        if s0 < best_score:
            best, best_score, best_m = start, s0, m0
        if best_score <= 0:
            break
    evals = restarts
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
                s, m = evaluate(cand, work, DC_CONTROL_FAST, 30)
                evals += 1
                if s < best_score - 1e-9:
                    best, best_score, best_m = cand, s, m
                    improved = True
                    vec = _vec(best, keys)
                    break
        for vt in [v for v in ("svt", "lvt", "ulvt", "hvt")
                   if v in VT_FLAVORS]:
            if evals >= max_evals or vt == best.vt:
                continue
            cand = replace(best, vt=vt)
            s, m = evaluate(cand, work, DC_CONTROL_FAST, 30)
            evals += 1
            if s < best_score - 1e-9:
                best, best_score, best_m = cand, s, m
                improved = True
        if not improved:
            step = [1 + (s - 1) / 2 for s in step]

    elapsed = time.time() - t0
    emit(out, subs, best, name, vdd_net, gnd_net, outs)
    err = ""
    try:
        final = run_deck(out / "tb_dc.cir", DC_CONTROL, out, "dc", timeout=900)
    except SimError as exc:
        final, err = {}, str(exc).splitlines()[0]

    result = {"design": name, "outputs": outs, "evals": evals,
              "evals_seconds": round(elapsed, 1),
              "score": vref_score(final, outs) if final else BAD,
              "metrics": final, "pass": vref_report(final, outs), "error": err}
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def _one(args):
    name, evals = args
    try:
        return size(name, evals, restarts=12)
    except Exception as exc:
        return {"design": name, "error": f"{type(exc).__name__}: {exc}",
                "pass": {}, "metrics": {}}


def main() -> None:
    evals = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    results = []
    with ProcessPoolExecutor(max_workers=len(PORTABLE)) as pool:
        futs = {pool.submit(_one, (n, evals)): n for n in PORTABLE}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            p = r.get("pass", {})
            print(f"{r['design']:34s} score={r.get('score', -1):8.3f} "
                  f"pass={sum(p.values())}/{len(p) or 1} "
                  f"{r.get('error', '')[:60]}", flush=True)
    for name, why in BLOCKED.items():
        print(f"{name:34s} BLOCKED -- {why}", flush=True)
    outp = ROOT / "results" / "vref_sizing.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"sized": results, "blocked": BLOCKED},
                               indent=2, default=str))
    print(f"\n-> {outp}")


if __name__ == "__main__":
    main()
