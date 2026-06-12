import sys
from pathlib import Path
PROJECT_ROOT = Path("/data2/home/shenshan/NN_SPICE-refactor-nn")
sys.path.insert(0, str(PROJECT_ROOT))
from tests.common.complex import BENCH, parse_netlist
from tests.verify_complex_sram_snm import _directnet_6t_netlist
from pycircuitsim.solver import _is_mosfet
import pycircuitsim.models.mosfet_nn as mnn

SCRATCH = PROJECT_ROOT / "results/v6_4_7/s7_logs/scratch"
SCRATCH.mkdir(parents=True, exist_ok=True)


def get_devs(tech):
    bt = BENCH[tech]
    netlist = _directnet_6t_netlist(bt, bt.vdd, 0.0, SCRATCH / f"s7sc_{tech}.sp")
    parser = parse_netlist(netlist)
    devs = {c.name: c for c in parser.circuit.components if _is_mosfet(c)}
    return bt, devs["Mnr"], devs["Mpr"]


def ev(m, vd, vg, vs=0.0, vb=0.0):
    volt = {m.nodes[0]: vd, m.nodes[1]: vg, m.nodes[2]: vs, m.nodes[3]: vb}
    m.clear_cache()
    r = dict(m._eval(volt))
    m.clear_cache()
    return r


orig_floor = mnn._MOSFETNNBase._floor_gds
captured = []


def cap_floor(id_phys, gds_phys):
    captured.append((id_phys, gds_phys))
    return orig_floor(id_phys, gds_phys)


mnn._MOSFETNNBase._floor_gds = staticmethod(cap_floor)

for tech in ("TSMC7",):
    bt, mn, mp = get_devs(tech)
    print(f"{tech}: VDD={bt.vdd}, nmos VDD_train={mn._vdd_estimate:.4f}, "
          f"pmos VDD_train={mp._vdd_estimate:.4f}")
    for m, lbl, sgn in ((mn, "nmos", 1.0), (mp, "pmos", -1.0)):
        for frac_d, frac_g in ((1.0, 1.0), (0.5, 0.6), (0.05, 0.6),
                               (-0.15, 0.6), (-0.35, 0.6)):
            vd, vg = sgn * frac_d * bt.vdd, sgn * frac_g * bt.vdd
            h = 1e-5
            captured.clear()
            r = ev(m, vd, vg)
            ncap = list(captured)
            rp = ev(m, vd + h, vg)
            rm = ev(m, vd - h, vg)
            fd_d = (rp["id"] - rm["id"]) / (2 * h)
            gp = ev(m, vd, vg + h)
            gm_ = ev(m, vd, vg - h)
            fd_g = (gp["id"] - gm_["id"]) / (2 * h)
            print(f"  {lbl} vds={vd:+.4f} vgs={vg:+.4f}: id={r['id']:+.4e} "
                  f"gm={r['gm']:+.4e} gds={r['gds']:+.4e} | "
                  f"FD d(id)/dvd={fd_d:+.4e} FD d(id)/dvg={fd_g:+.4e} | "
                  f"rawfloor={[(f'{a:+.3e}', f'{b:+.3e}') for a, b in ncap]}")
