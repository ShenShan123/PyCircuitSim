"""V6.4 best-of-N pair evaluator.

Evaluates an inverter for ONE tech given an explicit
(nmos-checkpoint-stem, pmos-checkpoint-stem) pair. The inverter VTC
depends on the (nmos-seed, pmos-seed) PAIR, so this driver lets the
greedy search mix any nmos seed with any pmos seed.

It physically swaps the two candidate checkpoints (+_norm.npz) into the
canonical ``tsmc{X}_dn_medium_{dev}`` slots the parser preempt cascade
resolves, runs ``eval_v6_3_1_inverter.evaluate_inverter``, prints a
machine-parseable metrics line, and ALWAYS restores the canonical slots
from the V6.3.1 backup at ``/tmp/v6_3_1_checkpoints_backup/``.

A "stem" is a checkpoint basename without ``_best.pt`` / ``_norm.npz``,
e.g. ``v6_4_bof_tsmc5_s42_nmos`` or ``tsmc5_dn_medium_nmos`` (the
V6.3.1 baseline itself).

Usage:
    python scripts/eval_v6_4_pair.py --tech TSMC5 \
        --nmos v6_4_bof_tsmc5_s7_nmos --pmos v6_4_repro_tsmc5_dn_medium_pmos
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

CKPT_DIR = PROJECT_ROOT / "external_compact_models" / "bsimar" / "checkpoints"
BACKUP_DIR = Path("/tmp/v6_3_1_checkpoints_backup")


def _files(stem: str) -> tuple[Path, Path]:
    return CKPT_DIR / f"{stem}_best.pt", CKPT_DIR / f"{stem}_norm.npz"


def _canonical(tech: str, dev: str) -> tuple[Path, Path]:
    s = f"{tech.lower()}_dn_medium_{dev}"
    return CKPT_DIR / f"{s}_best.pt", CKPT_DIR / f"{s}_norm.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", required=True)
    ap.add_argument("--nmos", required=True, help="nmos checkpoint stem")
    ap.add_argument("--pmos", required=True, help="pmos checkpoint stem")
    ap.add_argument("--json", action="store_true",
                    help="emit one JSON line for machine parsing")
    args = ap.parse_args()
    tech = args.tech

    for dev, stem in (("nmos", args.nmos), ("pmos", args.pmos)):
        pt, nz = _files(stem)
        if not pt.exists() or not nz.exists():
            sys.exit(f"missing candidate: {pt} / {nz}")

    # Backup integrity guard.
    n_bk = len(list(BACKUP_DIR.glob("*.pt")))
    if n_bk < 14:
        sys.exit(f"V6.3.1 backup looks incomplete ({n_bk} .pt) — abort")

    swapped: list[Path] = []
    try:
        for dev, stem in (("nmos", args.nmos), ("pmos", args.pmos)):
            cpt, cnz = _canonical(tech, dev)
            spt, snz = _files(stem)
            for src, dst in ((spt, cpt), (snz, cnz)):
                if src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)
                swapped.append(dst)

        import scripts.eval_v6_3_1_inverter as ev
        ev.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="v6_4_pair_") as tmp:
            res = ev.evaluate_inverter(tech, Path(tmp))

        v = res["vtc"]
        tp = res["tran_post"]
        out = {
            "tech": tech, "nmos": args.nmos, "pmos": args.pmos,
            "vtc_maxerr_mv": v["MaxErr(V)"] * 1e3,
            "vtc_nrmse_pct": v["NRMSE_vdd(%)"],
            "vtc_mre_pct": v["MRE(%)"],
            "vtc_r2": v["R2"],
            "vtc_dvtrip_mv": v.get("Vtrip_err(mV)", float("nan")),
            "tran_post_maxerr_mv": tp["MaxErr(V)"] * 1e3,
            "tran_post_nrmse_pct": tp["NRMSE_vdd(%)"],
            "tran_post_mre_pct": tp["MRE(%)"],
            "tran_post_r2": tp["R2"],
        }
        if args.json:
            print("RESULT " + json.dumps(out))
        else:
            print(f"\n=== {tech}  nmos={args.nmos}  pmos={args.pmos} ===")
            print(f"  VTC      MaxErr={out['vtc_maxerr_mv']:.1f}mV  "
                  f"NRMSE={out['vtc_nrmse_pct']:.3f}%  "
                  f"MRE={out['vtc_mre_pct']:.2f}%  R2={out['vtc_r2']:.4f}  "
                  f"dVtrip={out['vtc_dvtrip_mv']:+.1f}mV")
            print(f"  TranPost MaxErr={out['tran_post_maxerr_mv']:.1f}mV  "
                  f"NRMSE={out['tran_post_nrmse_pct']:.3f}%  "
                  f"MRE={out['tran_post_mre_pct']:.2f}%  "
                  f"R2={out['tran_post_r2']:.4f}")
    finally:
        for f in swapped:
            bk = BACKUP_DIR / f.name
            if bk.exists():
                shutil.copy2(bk, f)
        print(f"Restored {len(swapped)} canonical slots from {BACKUP_DIR}")


if __name__ == "__main__":
    main()
