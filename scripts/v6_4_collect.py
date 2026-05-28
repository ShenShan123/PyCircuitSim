"""V6.4 best-of-N collection + greedy pair search helper.

Reads training logs to tabulate per-cell val loss, lists which seeds
have completed checkpoints, and emits the greedy-search command plan.
Does NOT itself run inverter evals (that is eval_v6_4_pair.py) — it
prepares the bookkeeping so the search is reproducible.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "external_compact_models" / "bsimar" / "checkpoints"
LOGDIR = ROOT / "logs" / "v6_4_bestof"
REPRO_LOGDIR = ROOT / "logs" / "v6_4_repro"

TECHS = ("tsmc5", "tsmc7", "tsmc12", "tsmc16")
DEVS = ("nmos", "pmos")
SEEDS = (42, 123, 7, 17, 99, 256, 2024, 31337)

# Pre-existing v6_4_repro_* stock checkpoints -> (stem, log).
REPRO = {
    ("tsmc5", "nmos", 42): ("v6_4_repro_tsmc5_dn_medium_nmos",
                            REPRO_LOGDIR / "tsmc5_nmos.log"),
    ("tsmc5", "pmos", 42): ("v6_4_repro_tsmc5_dn_medium_pmos",
                            REPRO_LOGDIR / "tsmc5_pmos.log"),
    ("tsmc7", "nmos", 42): ("v6_4_repro_tsmc7_dn_medium_nmos",
                            REPRO_LOGDIR / "tsmc7_nmos.log"),
    ("tsmc7", "pmos", 42): ("v6_4_repro_tsmc7_dn_medium_pmos",
                            REPRO_LOGDIR / "tsmc7_pmos.log"),
    ("tsmc5", "nmos", 123): ("v6_4_repro_seed123_tsmc5_dn_medium_nmos",
                             REPRO_LOGDIR / "seed123_tsmc5_nmos.log"),
}

_BESTVAL = re.compile(r"Best val=([0-9.]+)")


def stem(tech: str, dev: str, seed: int) -> str:
    """Checkpoint stem for a (tech, dev, seed) cell."""
    if (tech, dev, seed) in REPRO:
        return REPRO[(tech, dev, seed)][0]
    return f"v6_4_bof_{tech}_s{seed}_{dev}"


def val_loss(tech: str, dev: str, seed: int) -> float | None:
    """Best val loss parsed from the cell's training log."""
    if (tech, dev, seed) in REPRO:
        log = REPRO[(tech, dev, seed)][1]
    else:
        log = LOGDIR / f"{tech}_{dev}_s{seed}.log"
    if not log.exists():
        return None
    m = None
    for line in log.read_text(errors="ignore").splitlines():
        mm = _BESTVAL.search(line)
        if mm:
            m = mm
    return float(m.group(1)) if m else None


def ckpt_ok(tech: str, dev: str, seed: int) -> bool:
    s = stem(tech, dev, seed)
    return (CKPT / f"{s}_best.pt").exists() and \
           (CKPT / f"{s}_norm.npz").exists()


def main() -> None:
    print(f"{'tech':>7} {'dev':>5} {'seed':>6} {'val_loss':>10} {'ckpt':>6}")
    print("-" * 40)
    n_ok = n_total = 0
    for tech in TECHS:
        for dev in DEVS:
            for seed in SEEDS:
                n_total += 1
                vl = val_loss(tech, dev, seed)
                ok = ckpt_ok(tech, dev, seed)
                n_ok += ok
                vls = f"{vl:.6f}" if vl is not None else "—"
                print(f"{tech:>7} {dev:>5} {seed:>6} {vls:>10} "
                      f"{'yes' if ok else 'NO':>6}")
    print("-" * 40)
    print(f"checkpoints ready: {n_ok}/{n_total}")
    if n_ok < n_total:
        print("INCOMPLETE — wait for training to finish.")
        sys.exit(1)
    print("ALL READY — greedy pair search can proceed.")


if __name__ == "__main__":
    main()
