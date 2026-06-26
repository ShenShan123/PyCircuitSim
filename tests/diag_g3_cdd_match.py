"""P0-2 (Decision D2): G3 device f3db — is the pole an OP-DRIFT gap or a CAP-DERIVATIVE gap?

The AC solver sets the device CS-amp output pole from the autograd diagonal
transcapacitance ``cdd = ∂qd/∂Vd`` (`mosfet_nn._eval` → `result["cdd"]`,
consumed by `_stamp_cap_ac`). Note: this is the autograd derivative of the
predicted DRAIN CHARGE ``qd`` — NOT the model's directly-predicted ``cdd``
OUTPUT column. The 13-output DirectNet predicts BOTH (an autograd ∂qd/∂Vd AND a
supervised ``cdd`` head), but only the autograd one drives the pole.

The recorded hypothesis (CHANGELOG + memory + plan §G3) is that f3db = 13/24 is
**OP-drift / value-surface owned, NOT a charge-derivative deficiency**:
``--charge-sobolev`` did NOT move f3db, and the V6.5.2 TG-corridor fixed the
PMOS supervised ``cdd`` 62%→5% yet f3db did not budge. If a cap-derivative
deficiency were the bind, gain would degrade too — but gain is 24/24 perfect.

THE D2 MEASUREMENT. On a grid covering the tsmc12-PMOS common-source-amplifier
saturation region (the region the CS-amp pole lives in), compare THREE ``cdd``:

  1. autograd_cdd  — ``∂qd/∂Vd`` that the AC solver actually consumes
                     (`device._eval(v)["cdd"]`; the Vds correction never touches
                     a charge/cap key, so this is the raw autograd derivative).
  2. supervised_cdd — the model's directly-predicted ``cdd`` OUTPUT column,
                     denormalized (a separate forward, reading `out[mcol("cdd")]`
                     and `device._denorm("cdd", …)`).
  3. osdi_cdd       — BSIM-CMG OSDI ground truth (`pycmg` eval at the same bias).

Primary metric (the "do they ALREADY match?" question):
        rel_match = |autograd_cdd − supervised_cdd| / |supervised_cdd|

  * median rel_match SMALL (< ~10%)  => the autograd ∂qd/∂Vd and the supervised
    head are ALREADY the same surface; supervising ``cdd`` (charge-Sobolev /
    A3) cannot move what the AC solver reads => the G3 charge lever is
    **DEAD ON ARRIVAL**; f3db is OP-drift / value-surface owned (confirms the
    recorded hypothesis).
  * median rel_match LARGE  => the two heads DISAGREE; pinning the autograd
    derivative toward the (accurate) supervised column has a (low) shot => A3.

Also reported: each of the two NN ``cdd`` vs OSDI ground truth, so the absolute
charge-derivative fidelity is visible alongside the head-vs-head agreement.

NO NGSPICE / NO GPU needed — OSDI ground truth comes straight from PyCMG.

Usage:
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NGSPICE_BIN="$PWD/tools/ngspice-45.2/bin/ngspice" \
      conda run -n pycircuitsim python tests/diag_g3_cdd_match.py --tech TSMC12 --device pmos
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Insert in reverse priority so PROJECT_ROOT ends up FIRST on sys.path: `tests`
# must resolve to the project's tests package, not PyCMG/tests (which is also a
# package). Remove-then-insert(0) each, project root last (matches the harvest
# scripts' bootstrap order).
for _p in (PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests",
           PROJECT_ROOT / "external_compact_models" / "PyCMG",
           PROJECT_ROOT / "external_compact_models",
           PROJECT_ROOT):
    _sp = str(_p)
    if _sp in sys.path:
        sys.path.remove(_sp)
    sys.path.insert(0, _sp)

from tests.common.complex import BENCH  # noqa: E402

# Variant fed to the NN tech_code AND used for the OSDI ground truth — matched to
# the V6.5.5 corridor harvest (scripts/v6_5_5_harvest_corridor.py) so the OSDI
# reference is the exact variant the NN's tech_code identifies.
BENCH_VARIANT = {"tsmc5": "lvt", "tsmc7": "ulvt", "tsmc12": "svt", "tsmc16": "svt"}
ROOM_T_K = 300.15
NFIN = 2
L_NMOS, L_PMOS = 16e-9, 20e-9

MATCH_TOL = 0.10          # median rel_match below this => charge lever dead on arrival
ON_ID_FLOOR_A = 1e-10     # device must be conducting at a kept grid point
CDD_FLOOR_F = 1e-21       # guard tiny denominators in the rel-diff ratio

# OP-centered grid offsets (V) in BOTH Vds and Vgs, source-relative frame. A
# 5×5 tube around the mid-rail CS-amp OP — the saturation region the pole lives in.
GRID_OFFSETS = (-0.10, -0.05, 0.0, 0.05, 0.10)


# ---------------------------------------------------------------------------
# Build the production DirectNet device + the OSDI ground-truth instance
# ---------------------------------------------------------------------------
def _build_nn_device(tech: str, device: str, variant: str):
    """Instantiate the PRODUCTION per-tech DirectNet device (LEVEL=73).

    Resolves the exact checkpoint the parser would pick for this (tech, device,
    VT) via the real resolver cascade (the `tsmc{X}_dn_medium` slot → the
    best-config symlink), so we probe the shipping surface — not a guess.
    Returns (device, checkpoint_name) or None if no checkpoint is on disk.
    """
    from pycircuitsim.parser import _resolve_nn_checkpoint
    from pycircuitsim.models.mosfet_directnet import NMOS_NN, PMOS_NN

    try:
        path, tech_code = _resolve_nn_checkpoint(
            level=73, device_key=device, tech_key=tech.lower(), vt_key=variant,
            explicit_path=None, netlist_name=f"diag_g3_{device}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] checkpoint resolve failed: {exc}")
        return None
    if not Path(path).exists():
        print(f"  [skip] resolved checkpoint missing on disk: {path}")
        return None

    is_pmos = device == "pmos"
    L = L_PMOS if is_pmos else L_NMOS
    cls = PMOS_NN if is_pmos else NMOS_NN
    try:
        dev = cls(name=f"M_{device}", nodes=["d", "g", "s", "b"],
                  model_path=path, L=L, NFIN=float(NFIN),
                  temperature=ROOM_T_K, tech_code=tech_code)
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] NN device build failed: {exc}")
        return None
    return dev, Path(path).name


def _build_osdi_instance(tech: str, device: str, variant: str):
    """Build the BSIM-CMG OSDI ground-truth Instance for this (tech, dev, variant)."""
    from pycmg.nn_generate import _create_model_and_instance
    from pycmg.nn_config import TECH_CONFIGS

    is_pmos = device == "pmos"
    L = L_PMOS if is_pmos else L_NMOS
    cfg = TECH_CONFIGS[tech.lower()]
    built = _create_model_and_instance(cfg, device, variant, L, float(NFIN), ROOM_T_K)
    if built is None:
        return None
    _model, inst, _proc = built
    return inst


# ---------------------------------------------------------------------------
# Three cdd readouts at one source-relative bias (Vs=0, Vbs=0)
# ---------------------------------------------------------------------------
def _nn_autograd_cdd(dev, vds_nn: float, vgs_nn: float) -> float:
    """The ∂qd/∂Vd the AC solver consumes (result['cdd'] from _eval)."""
    volts = {"d": vds_nn, "g": vgs_nn, "s": 0.0, "b": 0.0}
    return float(dev._eval(volts)["cdd"])


def _nn_supervised_cdd(dev, vds_nn: float, vgs_nn: float) -> float:
    """The model's directly-predicted cdd OUTPUT column, denormalized.

    Mirrors the _eval forward (same source-shift + smooth-clamp + z-score via
    `_prep_voltages`), then reads the raw `cdd` head and denormalizes it — the
    SUPERVISED quantity, NOT the autograd derivative.
    """
    import torch
    volts = {"d": vds_nn, "g": vgs_nn, "s": 0.0, "b": 0.0}
    x, _v_d_nn, _v_s_nn = dev._prep_voltages(volts)
    with torch.no_grad():
        out = dev._forward_model(x)
    col = dev._mcol("cdd")
    return float(dev._denorm("cdd", float(out[0, col])))


def _osdi_cdd_id(inst, vds_nn: float, vgs_nn: float) -> Tuple[float, float]:
    """OSDI ground-truth (cdd, id) at the same source-relative bias."""
    from pycmg.nn_generate import eval_single_point
    out = eval_single_point(inst, vd=vds_nn, vg=vgs_nn, vs=0.0, vb=0.0, _silent=True)
    if out is None:
        return float("nan"), float("nan")
    return float(out["cdd"]), float(out["id"])


# ---------------------------------------------------------------------------
# OP-anchored saturation grid
# ---------------------------------------------------------------------------
def _grid_center(tech: str, device: str) -> Tuple[float, float, str]:
    """(vds0, vgs0, how) — mid-rail CS-amp OP in the source-relative frame.

    Anchors on the production NN's OWN amplifying bias via verify_nn_ac.find_nn_bias
    (NGSPICE-free); falls back to the analytic mid-rail center if that scan fails.
    """
    bt = BENCH[tech.upper()]
    vdd = bt.vdd
    sgn = -1.0 if device == "pmos" else 1.0
    # source = vdd (PMOS) or 0 (NMOS); drain mid-rail; source-relative shift:
    default = (sgn * vdd / 2.0, sgn * vdd / 2.0, "analytic-mid-rail")
    try:
        from tests.verify_nn_ac import find_nn_bias, RESULTS_BASE
        work = RESULTS_BASE / "diag_g3" / tech
        work.mkdir(parents=True, exist_ok=True)
        logging.disable(logging.CRITICAL)
        try:
            vbias, vout = find_nn_bias(bt, device, work, f"{tech.lower()}_{device}")
        finally:
            logging.disable(logging.NOTSET)
        vs = vdd if device == "pmos" else 0.0
        vds0, vgs0 = vout - vs, vbias - vs
        if np.isfinite(vds0) and np.isfinite(vgs0) and abs(vds0) > 0.05:
            return float(vds0), float(vgs0), f"NN-OP (Vin={vbias:.3f} Vout={vout:.3f})"
    except Exception as exc:  # noqa: BLE001
        print(f"  [note] OP-anchor scan failed ({exc!r}); using analytic mid-rail center")
    return default


def _build_grid(vds0: float, vgs0: float) -> List[Tuple[float, float]]:
    return [(round(vds0 + dd, 4), round(vgs0 + dg, 4))
            for dd in GRID_OFFSETS for dg in GRID_OFFSETS]


# ---------------------------------------------------------------------------
# Per-(tech, device) D2 evaluation
# ---------------------------------------------------------------------------
def run(tech: str, device: str) -> Optional[Dict]:
    variant = BENCH_VARIANT[tech.lower()]
    print(f"\n===== {tech} {device} (variant={variant}) — D2 cdd head-vs-head =====")

    built = _build_nn_device(tech, device, variant)
    if built is None:
        return None
    dev, chk_name = built
    inst = _build_osdi_instance(tech, device, variant)
    if inst is None:
        print("  [skip] OSDI instance build failed")
        return None

    vds0, vgs0, how = _grid_center(tech, device)
    grid = _build_grid(vds0, vgs0)
    print(f"  checkpoint: {chk_name}")
    print(f"  grid center: Vds={vds0:+.3f} Vgs={vgs0:+.3f}  [{how}]  "
          f"({len(grid)} points, source-relative Vs=0/Vbs=0)")

    print(f"\n  {'Vds':>7} {'Vgs':>7} | {'auto_cdd(aF)':>13} {'sup_cdd(aF)':>13} "
          f"{'osdi_cdd(aF)':>13} | {'|a-s|/|s|':>9} {'|a-o|/|o|':>9} "
          f"{'|s-o|/|o|':>9} {'id(uA)':>9}")

    rel_match, rel_auto_osdi, rel_sup_osdi = [], [], []
    ratio_signs = []
    n_kept = 0
    for vds_nn, vgs_nn in grid:
        osdi_cdd, osdi_id = _osdi_cdd_id(inst, vds_nn, vgs_nn)
        if not np.isfinite(osdi_id) or abs(osdi_id) < ON_ID_FLOOR_A:
            continue  # OFF / non-convergent — outside the CS-amp pole region
        auto = _nn_autograd_cdd(dev, vds_nn, vgs_nn)
        sup = _nn_supervised_cdd(dev, vds_nn, vgs_nn)

        rm = abs(auto - sup) / max(abs(sup), CDD_FLOOR_F)
        rao = (abs(auto - osdi_cdd) / max(abs(osdi_cdd), CDD_FLOOR_F)
               if np.isfinite(osdi_cdd) else float("nan"))
        rso = (abs(sup - osdi_cdd) / max(abs(osdi_cdd), CDD_FLOOR_F)
               if np.isfinite(osdi_cdd) else float("nan"))
        rel_match.append(rm)
        if np.isfinite(rao):
            rel_auto_osdi.append(rao)
        if np.isfinite(rso):
            rel_sup_osdi.append(rso)
        if abs(sup) > CDD_FLOOR_F:
            ratio_signs.append(np.sign(auto / sup))
        n_kept += 1
        print(f"  {vds_nn:+7.3f} {vgs_nn:+7.3f} | {auto*1e18:13.3f} {sup*1e18:13.3f} "
              f"{osdi_cdd*1e18:13.3f} | {rm:9.3f} {rao:9.3f} {rso:9.3f} "
              f"{osdi_id*1e6:9.3f}")

    if n_kept == 0:
        print("  [skip] no ON / saturated grid points — cannot adjudicate")
        return None

    med_match = float(np.median(rel_match))
    med_auto_osdi = float(np.median(rel_auto_osdi)) if rel_auto_osdi else float("nan")
    med_sup_osdi = float(np.median(rel_sup_osdi)) if rel_sup_osdi else float("nan")
    sign_flip = bool(ratio_signs and np.median(ratio_signs) < 0)

    print(f"\n  kept {n_kept}/{len(grid)} ON-saturation points")
    print(f"  median rel_match |auto-sup|/|sup| = {med_match*100:.1f}%")
    print(f"  median autograd vs OSDI           = {med_auto_osdi*100:.1f}%")
    print(f"  median supervised vs OSDI         = {med_sup_osdi*100:.1f}%")
    if sign_flip:
        print("  [caveat] autograd and supervised cdd have OPPOSITE sign on median "
              "→ a convention mismatch, not a magnitude agreement.")

    if med_match < MATCH_TOL and not sign_flip:
        verdict = (
            f"D2 = MATCH (median {med_match*100:.1f}% < {MATCH_TOL*100:.0f}%). The autograd "
            f"∂qd/∂Vd the AC solver reads ALREADY equals the supervised cdd head, so "
            f"charge-Sobolev / A3 on cdd cannot move the pole => the G3 charge lever is "
            f"DEAD ON ARRIVAL. f3db is OP-drift / value-surface owned (confirms the "
            f"recorded hypothesis).")
    else:
        verdict = (
            f"D2 = MISMATCH (median {med_match*100:.1f}% ≥ {MATCH_TOL*100:.0f}%"
            f"{' / SIGN-FLIP' if sign_flip else ''}). The autograd derivative and the "
            f"supervised cdd head DISAGREE; pinning the autograd ∂qd/∂Vd toward the "
            f"supervised column (A3 charge-Sobolev on cdd) has a (low) shot at moving f3db.")
    print(f"  >> VERDICT: {verdict}")

    return {"tech": tech, "device": device, "chk": chk_name, "n_kept": n_kept,
            "med_match": med_match, "med_auto_osdi": med_auto_osdi,
            "med_sup_osdi": med_sup_osdi, "sign_flip": sign_flip,
            "match": (med_match < MATCH_TOL and not sign_flip)}


def main() -> int:
    ap = argparse.ArgumentParser(description="P0-2 / D2: G3 cdd autograd-vs-supervised match probe")
    ap.add_argument("--tech", default="TSMC12",
                    help="comma-separated techs (default: TSMC12 — the D2 target)")
    ap.add_argument("--device", default="pmos",
                    help="comma-separated devices (default: pmos — the D2 target)")
    args = ap.parse_args()

    techs = [t.strip().upper() for t in args.tech.split(",") if t.strip()]
    devices = [d.strip().lower() for d in args.device.split(",") if d.strip()]

    print("=" * 86)
    print("P0-2 (D2): device f3db — autograd cdd (pole driver) vs supervised cdd head")
    print("  MATCH => charge lever dead on arrival (OP-drift owned); MISMATCH => A3 low shot")
    print("=" * 86)

    rows: List[Dict] = []
    for tech in techs:
        if tech not in BENCH:
            print(f"  [skip] unknown tech {tech}")
            continue
        for device in devices:
            if device not in ("nmos", "pmos"):
                print(f"  [skip] unknown device {device}")
                continue
            try:
                r = run(tech, device)
                if r:
                    rows.append(r)
            except Exception as exc:  # noqa: BLE001 — fail loud
                import traceback
                print(f"  {tech} {device}: ERROR {exc!r}")
                traceback.print_exc()

    if rows:
        print("\n" + "=" * 86)
        print("SUMMARY (D2 routes the G3 charge-Sobolev lever):")
        for r in rows:
            tag = "MATCH→lever DEAD" if r["match"] else "MISMATCH→A3 low shot"
            print(f"  {r['tech']:7} {r['device']:4} rel_match={r['med_match']*100:5.1f}%  "
                  f"auto-vs-OSDI={r['med_auto_osdi']*100:5.1f}%  "
                  f"sup-vs-OSDI={r['med_sup_osdi']*100:5.1f}%  "
                  f"{'[SIGNFLIP] ' if r['sign_flip'] else ''}=> {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
