#!/usr/bin/env python3
"""S6 follow-up: run the PURE-NN ring oscillator with the off-diagonal cap
convention flipped to the L72/PyCMG one (negate cgd/cgs/cdg at the
get_capacitances boundary), all 4 techs.

Why: the artifact probe (s6_artifact_probe) showed the NN device class hands
the solver off-diagonal caps with OPPOSITE sign to the native LEVEL=72 device
at every bias (charges identical, +1.000). The L72 convention reproduces
NGSPICE through `_stamp_mosfet_transient` exactly (46.64 vs 46.65 ps); the
NN convention with exact-OSDI id/charges gives 93.01 ps (2x). This isolates
the convention on the production NN model: if TSMC7 moves from ~50.8 ps
toward NG ~46.6 ps, the cap-sign convention owns (a large part of) the
long-standing RO period error. The three passing techs are the blind veto.

Monkeypatch-only — no production code modified.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "external_compact_models" / "PyCMG" / "tests",
          ROOT / "external_compact_models" / "PyCMG",
          ROOT / "external_compact_models",
          ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tests.common.complex import BENCH  # noqa: E402
from tests.verify_complex_ring_osc import (  # noqa: E402
    run_directnet_ro, run_ngspice_ro, _period_from_wave)
from pycircuitsim.models.mosfet_nn import _MOSFETNNBase  # noqa: E402

SETTLE = 0.3e-9


def main() -> int:
    orig_caps = _MOSFETNNBase.get_capacitances

    def l72_convention_caps(self, voltages):
        c = orig_caps(self, voltages)
        return {"cgg": c["cgg"], "cdd": c["cdd"],
                "cgd": -c["cgd"], "cgs": -c["cgs"], "cdg": -c["cdg"]}

    print("S6 cap-sign convention experiment — pure NN RO, off-diagonal "
          "caps negated to the L72 convention", flush=True)
    for tech in ("TSMC7", "TSMC5", "TSMC12", "TSMC16"):
        bt = BENCH[tech]
        mid = bt.vdd / 2
        ng = run_ngspice_ro(bt, Path(tempfile.mkdtemp(prefix="s6cap_ng_")))
        ng_per = _period_from_wave(ng["time"], ng["v(n5)"], mid, SETTLE)
        rows = {}
        for label, flip in (("nn-conv", False), ("l72-conv", True)):
            _MOSFETNNBase.get_capacitances = (
                l72_convention_caps if flip else orig_caps)
            try:
                work = Path(tempfile.mkdtemp(prefix=f"s6cap_{tech}_{label}_"))
                dn, partial, err = run_directnet_ro(bt, work)
                per = _period_from_wave(
                    np.asarray(dn["time"]), np.asarray(dn["v(n5)"]),
                    mid, SETTLE)
                rows[label] = (per, partial)
            finally:
                _MOSFETNNBase.get_capacitances = orig_caps
        nn_ps = rows["nn-conv"][0] * 1e12
        l72_ps = rows["l72-conv"][0] * 1e12
        ng_ps = ng_per * 1e12
        err_nn = abs(nn_ps - ng_ps) / ng_ps * 100
        err_l72 = abs(l72_ps - ng_ps) / ng_ps * 100
        print(f"{tech:7s}: NG {ng_ps:6.2f} ps | NN-conv {nn_ps:6.2f} ps "
              f"(err {err_nn:5.2f}%) | L72-conv {l72_ps:6.2f} ps "
              f"(err {err_l72:5.2f}%) | partial nn/l72 = "
              f"{rows['nn-conv'][1]}/{rows['l72-conv'][1]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
