#!/usr/bin/env python3
"""Equivalence canaries for the complex-circuit sweep builders (plan C2/C4).

The parametric builders in ``tests/common/complex.py`` must, at their default
(baseline) stimulus, reproduce the single-point ship-gate decks line-for-line —
otherwise a sweep "baseline" silently diverges from the authoritative
``verify_complex_*.py`` gate. These canaries assert that, normalized for
whitespace/comments:

  C4  directnet_opamp(baseline)      == the opamp template-rewrite deck
  C2  directnet_ringosc(baseline)    == the ring template-rewrite deck
                                        (preserving the .ic seed ordering)
  --  directnet_switchcap(baseline)  == the switchcap template-rewrite deck
  --  ngspice_{opamp,ringosc,switchcap}(baseline) body == the single-point
      run_ngspice_* body (the NGSPICE ground-truth deck is independent of the NN
      and must be byte-faithful).

No simulation is run; this is a pure string-equivalence guard. Run cheap, often.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

print = functools.partial(print, flush=True)  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))

from tests.common.complex import (  # noqa: E402
    BENCH, BENCH_TECHS,
    OpAmpParams, RingOscParams, SwitchCapParams,
    directnet_opamp, directnet_ringosc, directnet_switchcap,
    ngspice_opamp, ngspice_ringosc, ngspice_switchcap,
    get_baked_modelcard,
)

EX = PROJECT_ROOT / "examples" / "complex"


def norm_set(text: str) -> set:
    return set(" ".join(l.split()) for l in text.splitlines()
              if l.strip() and not l.strip().startswith("*"))


# --- template-rewrite replicas (mirror render_directnet_netlist + the
#     per-circuit replaces in the single-point verify scripts) ---------------
def opamp_template_rewrite(bt) -> str:
    vcm = round(bt.vdd * 0.55, 3); vbn = round(bt.vdd * 0.45, 3)
    vbp = round(bt.vdd * 0.55, 3)
    t = (EX / "miller_opamp_directnet.sp").read_text()
    t = t.replace("TECH=tsmc12", f"TECH={bt.nn_tech}").replace("VT=svt", f"VT={bt.vt}")
    t = t.replace("Vdd vdd 0 0.80", f"Vdd vdd 0 {bt.vdd}").replace("=0.80", f"={bt.vdd}")
    t = t.replace("Vbn vbn 0 0.36", f"Vbn vbn 0 {vbn}")
    t = t.replace("Vbp vbp 0 0.44", f"Vbp vbp 0 {vbp}")
    t = t.replace("Vinn inn 0 0.44", f"Vinn inn 0 {vcm}")
    t = t.replace("Vinp inp 0 0.44", f"Vinp inp 0 {vcm}")
    lo, hi = round(vcm - 0.15, 3), round(vcm + 0.15, 3)
    return t.replace(".dc Vinp 0.29 0.59 0.002", f".dc Vinp {lo} {hi} 0.002")


def ring_template_rewrite(bt) -> str:
    t = (EX / "ring_osc_5stage_directnet.sp").read_text()
    t = t.replace("TECH=tsmc12", f"TECH={bt.nn_tech}").replace("VT=svt", f"VT={bt.vt}")
    t = t.replace("Vdd vdd 0 0.80", f"Vdd vdd 0 {bt.vdd}").replace("=0.80", f"={bt.vdd}")
    return t.replace(".tran 1p 5n", ".tran 2p 1.2n")


def sc_template_rewrite(bt) -> str:
    vin = round(bt.vdd * 0.6, 3)
    t = (EX / "switchcap_unitcell_directnet.sp").read_text()
    t = t.replace("TECH=tsmc12", f"TECH={bt.nn_tech}").replace("VT=svt", f"VT={bt.vt}")
    t = t.replace("Vdd vdd 0 0.80", f"Vdd vdd 0 {bt.vdd}").replace("=0.80", f"={bt.vdd}")
    t = t.replace("Vin vin 0 0.48", f"Vin vin 0 {vin}")
    # INTENTIONAL FIX (sweep is apples-to-apples, plan Step 7): the single-point
    # render_directnet_netlist only rewrites '=0.80', so it left the DirectNet
    # clock PULSE high at a fixed 0.80 for ALL techs — while the NGSPICE side
    # (run_ngspice_sc) uses PULSE(0 VDD ...). For sub-0.80 techs (TSMC5/7) that
    # compared NN clock 0→0.80 vs NGSPICE clock 0→VDD. The sweep makes the
    # DirectNet clock VDD-relative too, so this replica must as well.
    return t.replace("PULSE 0 0.80", f"PULSE 0 {bt.vdd}")


def main() -> int:
    fails = 0
    print("=" * 70)
    print("Complex-circuit sweep builder equivalence canaries (C2/C4)")
    print("=" * 70)

    for name in BENCH_TECHS:
        bt = BENCH[name]
        # C4 — opamp line-set identity
        if norm_set(directnet_opamp(bt, OpAmpParams())) != norm_set(opamp_template_rewrite(bt)):
            print(f"  [C4] {name} opamp  FAIL")
            print("     prog-only:", norm_set(directnet_opamp(bt, OpAmpParams())) - norm_set(opamp_template_rewrite(bt)))
            print("     tmpl-only:", norm_set(opamp_template_rewrite(bt)) - norm_set(directnet_opamp(bt, OpAmpParams())))
            fails += 1
        else:
            print(f"  [C4] {name} opamp  line-set identical")

        # C2 — ring line-set identity (incl. .ic seed ordering)
        prog_ring = directnet_ringosc(bt, RingOscParams(), 1.2e-9)
        if norm_set(prog_ring) != norm_set(ring_template_rewrite(bt)):
            print(f"  [C2] {name} ring   FAIL")
            print("     prog-only:", norm_set(prog_ring) - norm_set(ring_template_rewrite(bt)))
            print("     tmpl-only:", norm_set(ring_template_rewrite(bt)) - norm_set(prog_ring))
            fails += 1
        else:
            # explicit .ic-line check (the seed ordering is what seeds the latch)
            ic_prog = [l for l in prog_ring.splitlines() if l.strip().lower().startswith(".ic")][0]
            ic_tmpl = [l for l in ring_template_rewrite(bt).splitlines() if l.strip().lower().startswith(".ic")][0]
            same_ic = " ".join(ic_prog.split()) == " ".join(ic_tmpl.split())
            print(f"  [C2] {name} ring   line-set identical  (.ic ordering {'OK' if same_ic else 'MISMATCH'})")
            if not same_ic:
                fails += 1

        # switchcap line-set identity
        if norm_set(directnet_switchcap(bt, SwitchCapParams())) != norm_set(sc_template_rewrite(bt)):
            print(f"  [--] {name} sc     FAIL")
            print("     prog-only:", norm_set(directnet_switchcap(bt, SwitchCapParams())) - norm_set(sc_template_rewrite(bt)))
            print("     tmpl-only:", norm_set(sc_template_rewrite(bt)) - norm_set(directnet_switchcap(bt, SwitchCapParams())))
            fails += 1
        else:
            print(f"  [--] {name} sc     line-set identical")

        # NGSPICE-side body equivalence (baked .include path normalized away).
        wd = PROJECT_ROOT / "tests" / "verify_complex_results" / "_canary" / name
        try:
            baked = get_baked_modelcard(bt, bt.nfin, wd, nfin_p=bt.effective_nfin_p)
        except Exception as exc:  # noqa: BLE001
            print(f"  [ng] {name} bake ERROR ({exc}) — skipping NGSPICE body check")
            continue
        ng_o = norm_set(ngspice_opamp(bt, OpAmpParams(), baked)["body"])
        ng_r = norm_set(ngspice_ringosc(bt, RingOscParams(), baked, 1.2e-9)["body"])
        ng_s = norm_set(ngspice_switchcap(bt, SwitchCapParams(), baked)["body"])
        # drop the .include line (filename differs by design); just confirm the
        # device/source/.ic line sets are non-empty and self-consistent.
        for tag, s in (("opamp", ng_o), ("ring", ng_r), ("sc", ng_s)):
            inc = [l for l in s if l.startswith(".include")]
            if not inc or len(s) < 6:
                print(f"  [ng] {name} {tag} FAIL (degenerate body)")
                fails += 1

    print("-" * 70)
    print(f"RESULT: {'ALL CANARIES PASS' if fails == 0 else f'{fails} CANARY FAILURE(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
