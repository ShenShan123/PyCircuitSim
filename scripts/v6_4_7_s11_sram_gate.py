#!/usr/bin/env python3
"""V6.4.7 S11 (P3) — combined SRAM gate: force_ic + subthreshold probe.

The single arbiter for the ship-required SRAM force_ic gate on a candidate
checkpoint set. For one (prefix, seed):

  1. isolate the candidate into a temp dir under the per-tech production name
     ``{tech}_dn_medium_{dev}_best.pt`` (fires Rule-19 local-vocab scope), set
     BSIMAR_CHECKPOINT_DIR before any project import;
  2. run ``force_ic_probe`` for each tech (state1 + state0) -> N/8 railed;
  3. run the subthreshold/weak-inversion id-fidelity probe (NN vs OSDI) ->
     the kill-gate leading indicator (≥10x improvement in the weak band).

Emits a RESULT json line with per-tech force_ic + probe bands.

Run (CPU; CUDA hidden to avoid GPU contention with training):
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      NGSPICE_BIN=tools/ngspice-45.2/bin/ngspice \
      PYTHONPATH=external_compact_models:external_compact_models/PyCMG \
      conda run -n pycircuitsim python scripts/v6_4_7_s11_sram_gate.py \
        --prefix v6_4_7_ctlv2 --seed 17 --techs TSMC5,TSMC7,TSMC12,TSMC16 \
        --out results/v6_4_7/s11_force_ic_ctlv2_s17.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_CKPT = PROJECT_ROOT / "external_compact_models" / "bsimar" / "checkpoints"


def _setup_isolation(prefix: str, seed: str, techs: List[str]) -> List[str]:
    """Copy candidate checkpoints into a temp dir under the production names
    and point BSIMAR_CHECKPOINT_DIR at it. MUST run before project imports."""
    iso = Path(tempfile.mkdtemp(prefix="s11_ckpt_"))
    have: List[str] = []
    for tech in techs:
        tl = tech.lower()
        ok = True
        for dev in ("nmos", "pmos"):
            stem = f"{prefix}_s{seed}_{tl}_{dev}"
            spt = SRC_CKPT / f"{stem}_best.pt"
            snz = SRC_CKPT / f"{stem}_norm.npz"
            if not (spt.exists() and snz.exists()):
                print(f"[skip] {tech}: missing {stem}", file=sys.stderr)
                ok = False
                break
            slot = f"{tl}_dn_medium_{dev}"
            shutil.copy2(spt, iso / f"{slot}_best.pt")
            shutil.copy2(snz, iso / f"{slot}_norm.npz")
        if ok:
            have.append(tech)
    os.environ["BSIMAR_CHECKPOINT_DIR"] = str(iso)
    os.environ.setdefault(
        "NGSPICE_BIN",
        str(PROJECT_ROOT / "tools" / "ngspice-45.2" / "bin" / "ngspice"))
    return have, iso


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--techs", default="TSMC5,TSMC7,TSMC12,TSMC16")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the subvt probe (force_ic only)")
    args = ap.parse_args()
    techs = [t.strip() for t in args.techs.split(",") if t.strip()]

    have, iso = _setup_isolation(args.prefix, args.seed, techs)
    label = f"{args.prefix}_s{args.seed}"

    # Heavy project imports AFTER BSIMAR_CHECKPOINT_DIR is set.
    sys.path.append(str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))
    if str(PROJECT_ROOT) in sys.path:
        sys.path.remove(str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT))
    from tests.common.complex import BENCH  # noqa: E402
    from tests.verify_complex_sram_snm import force_ic_probe  # noqa: E402
    sys.path.append(str(PROJECT_ROOT / "scripts"))
    from v6_4_7_s11_subvt_probe import probe_device  # noqa: E402

    work = iso / "work"
    work.mkdir(parents=True, exist_ok=True)

    try:
        force_ic: Dict[str, Dict[str, bool]] = {}
        n_pass = 0
        for tech in have:
            print(f"\n=== {tech} force_ic ===")
            res = force_ic_probe(BENCH[tech], work)
            force_ic[tech] = res
            n_pass += int(bool(res.get("state1"))) + int(bool(res.get("state0")))

        probe: List[Dict] = []
        if not args.no_probe:
            print(f"\n=== {label} subthreshold/weak-inversion probe ===")
            for tech in have:
                for dev in ("nmos", "pmos"):
                    r = probe_device(tech, dev, work)
                    probe.append(r)
                    wl = r["bands"].get("weak_lo", {})
                    wh = r["bands"].get("weak_hi", {})
                    off = r["bands"].get("off", {})
                    print(f"{tech:7s} {dev:4s} | "
                          f"weak_lo |log10|={wl.get('median_abs_log10', float('nan')):.3f} "
                          f"(r {wl.get('median_ratio', float('nan')):.2f}) | "
                          f"weak_hi |log10|={wh.get('median_abs_log10', float('nan')):.3f} "
                          f"(r {wh.get('median_ratio', float('nan')):.2f}) | "
                          f"off |log10|={off.get('median_abs_log10', float('nan')):.3f}")

        result = {
            "label": label, "prefix": args.prefix, "seed": args.seed,
            "techs": have,
            "force_ic_n_pass": n_pass, "force_ic_n_total": 2 * len(have),
            "force_ic": force_ic, "probe": probe,
        }
        print(f"\nforce_ic: {n_pass}/{2 * len(have)} railed  ({label})")
        print("RESULT " + json.dumps(result))
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=1))
            print(f"[gate] wrote {args.out}", file=sys.stderr)
    finally:
        shutil.rmtree(iso, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
