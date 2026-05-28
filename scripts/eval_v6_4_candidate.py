"""V6.4 bake-off evaluator.

Swaps a candidate's per-tech DirectNet checkpoints into the canonical
``tsmc{X}_dn_medium_{dev}`` slots that the parser preempt cascade
resolves, runs the inverter VTC + transient reproduction, prints a
metrics table, and restores the canonical slots from the V6.3.1 backup.

Usage:
    python scripts/eval_v6_4_candidate.py --exp v6_4_a --techs TSMC5,TSMC7
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

CKPT_DIR = PROJECT_ROOT / "external_compact_models" / "bsimar" / "checkpoints"
BACKUP_DIR = Path("/tmp/v6_3_1_checkpoints_backup")


def _canonical(tech: str, dev: str) -> tuple[Path, Path]:
    stem = f"{tech.lower()}_dn_medium_{dev}"
    return CKPT_DIR / f"{stem}_best.pt", CKPT_DIR / f"{stem}_norm.npz"


def _candidate(exp: str, tech: str, dev: str) -> tuple[Path, Path]:
    stem = f"{exp}_{tech.lower()}_{dev}"
    return CKPT_DIR / f"{stem}_best.pt", CKPT_DIR / f"{stem}_norm.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True,
                    help="Candidate exp-name prefix, e.g. v6_4_a")
    ap.add_argument("--techs", default="TSMC5,TSMC7,TSMC12,TSMC16")
    args = ap.parse_args()
    techs = [t.strip() for t in args.techs.split(",")]
    devs = ("nmos", "pmos")

    # Verify candidate checkpoints exist for every tech/dev.
    for t in techs:
        for d in devs:
            pt, nz = _candidate(args.exp, t, d)
            if not pt.exists() or not nz.exists():
                sys.exit(f"missing candidate checkpoint: {pt} / {nz}")

    swapped: list[tuple[Path, Path]] = []
    try:
        # Swap candidates into the canonical slots.
        for t in techs:
            for d in devs:
                cpt, cnz = _canonical(t, d)
                spt, snz = _candidate(args.exp, t, d)
                shutil.copy2(spt, cpt)
                shutil.copy2(snz, cnz)
                swapped.append((cpt, cnz))
        print(f"Swapped {len(swapped)*2} files for exp={args.exp}, "
              f"techs={techs}")

        from tests.verify_nn_dc_tran import ALL_TEST_TECHS  # noqa: F401
        import scripts.eval_v6_3_1_inverter as ev

        ev.TECHS = tuple(techs)
        results = {}
        with tempfile.TemporaryDirectory(prefix="v6_4_eval_") as tmp:
            wd = Path(tmp)
            for t in techs:
                results[t] = ev.evaluate_inverter(t, wd)

        print(f"\n=== V6.4 candidate {args.exp} — inverter metrics ===")
        print(f"{'Tech':>7s} | {'VTC NRMSE%':>10s} {'VTC MRE%':>9s} "
              f"{'VTC MaxErr':>11s} {'VTC R2':>8s} | "
              f"{'TrPost NRMSE%':>13s} {'TrPost MRE%':>11s} "
              f"{'TrPost MaxErr':>13s} {'TrPost R2':>9s}")
        for t in techs:
            v = results[t]["vtc"]
            tp = results[t]["tran_post"]
            print(f"{t:>7s} | "
                  f"{v['NRMSE_vdd(%)']:>10.3f} {v['MRE(%)']:>9.2f} "
                  f"{v['MaxErr(V)']*1e3:>9.1f}mV {v['R2']:>8.4f} | "
                  f"{tp['NRMSE_vdd(%)']:>13.3f} {tp['MRE(%)']:>11.2f} "
                  f"{tp['MaxErr(V)']*1e3:>11.1f}mV {tp['R2']:>9.4f}")
    finally:
        # Restore canonical slots from the V6.3.1 backup.
        for cpt, cnz in swapped:
            for f in (cpt, cnz):
                bk = BACKUP_DIR / f.name
                if bk.exists():
                    shutil.copy2(bk, f)
        print(f"\nRestored {len(swapped)*2} canonical slots from "
              f"{BACKUP_DIR}")


if __name__ == "__main__":
    main()
