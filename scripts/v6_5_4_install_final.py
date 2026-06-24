#!/usr/bin/env python3
"""Install the V6.5.4 FINAL best-config-per-tech mix into the resolver's medium
slots (medium-first preempt), reading results/v6_5_4_retrain/final_mix.json.

Each chosen config is a freshly-trained checkpoint prefix (e.g. `tsmc5_dn_large`
or `tsmc7_dn_lgs7`). We symlink `tsmc{X}_dn_medium_{dev}_{best.pt,norm.npz}` ->
the chosen file so the resolver (which reads the local tech_code from the slot
NAME) picks it with the correct per-tech vocab. If the chosen config already IS
`tsmc{X}_dn_medium`, nothing to do for that tech.

`--restore` removes the medium symlinks (leaving the real fresh medium files).
Idempotent.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / "external_compact_models" / "bsimar" / "checkpoints"
MIX = json.loads((ROOT / "results" / "v6_5_4_retrain" / "final_mix.json").read_text())
DEVS = ("nmos", "pmos")
SUF = ("best.pt", "norm.npz")


def install():
    for tech, info in MIX.items():
        cand = info["cand"]                 # e.g. tsmc5_dn_large or tsmc7_dn_lgs7
        if cand == f"{tech}_dn_medium":
            print(f"  {tech}: already medium (no-op)")
            continue
        for dev in DEVS:
            for suf in SUF:
                target = CK / f"{cand}_{dev}_{suf}"
                link = CK / f"{tech}_dn_medium_{dev}_{suf}"
                if not target.exists():
                    print(f"  !! MISSING {target.name}"); continue
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(target.name)
                print(f"  {link.name} -> {target.name}")
    print("\nInstalled V6.5.4 FINAL mix into the medium slots.")


def restore():
    # remove medium symlinks; if a real fresh medium file is needed, retrain or
    # copy from the size checkpoints. (We keep all fresh size ckpts on disk.)
    for tech in MIX:
        for dev in DEVS:
            for suf in SUF:
                link = CK / f"{tech}_dn_medium_{dev}_{suf}"
                if link.is_symlink():
                    link.unlink(); print(f"  removed symlink {link.name}")
    print("Removed medium symlinks.")


if __name__ == "__main__":
    restore() if "--restore" in sys.argv else install()
