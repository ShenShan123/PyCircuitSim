#!/usr/bin/env python3
"""V6.4.7 S10 (P4) — score the fine-tune λ-screen candidates + control-v2.

Runs eval_v6_4_5_candidate.py (with --deriv-fidelity, v2/filter-off) for each
(label, nmos, pmos) pair SERIALLY on one GPU (or --cpu), parses the RESULT
lines, and prints a compact A/B table vs the control-v2 s17 warm-start source.

The screen kill gate (plan S10): the best config must cut TSMC7 opamp gain err
below ~15% with the inverter held. The P4 promotion read additionally wants
deriv fwd_inrail (esp PMOS gds) strictly below control-v2 and the RO not
regressed.

Usage:
    NGSPICE_BIN=... conda run -n pycircuitsim python scripts/v6_4_7_s10_score.py \
        --tech tsmc7 --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CKPT = PROJECT_ROOT / "external_compact_models" / "bsimar" / "checkpoints"


def run_scorer(tech: str, nmos: str, pmos: str, gpu: Optional[str],
               timeout_s: int = 3000) -> dict:
    cmd = ["conda", "run", "-n", "pycircuitsim", "python",
           str(PROJECT_ROOT / "scripts" / "eval_v6_4_5_candidate.py"),
           "--tech", tech.upper(), "--nmos", nmos, "--pmos", pmos, "--json",
           "--deriv-fidelity", "--deriv-data-suffix", "v2",
           "--no-deriv-apply-filter"]
    env = dict(os.environ)
    env.setdefault("NGSPICE_BIN",
                   str(PROJECT_ROOT / "tools" / "ngspice-45.2" / "bin" / "ngspice"))
    env["PYTHONPATH"] = (f"{PROJECT_ROOT/'external_compact_models'}:"
                         f"{PROJECT_ROOT/'external_compact_models'/'PyCMG'}:"
                         + env.get("PYTHONPATH", ""))
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else gpu
    env["OMP_NUM_THREADS"] = "4"
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired:
        return {"_error": "timeout"}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    return {"_error": "no RESULT", "stderr_tail": p.stderr[-500:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", default="tsmc7")
    ap.add_argument("--gpu", default=None, help="GPU id; omit/empty for CPU")
    ap.add_argument("--src-seed", default="17")
    ap.add_argument("--ft-prefix", default="v6_4_7_s10sob",
                    help="stem prefix for the candidate configs "
                         "(v6_4_7_s10sob = from-scratch screen2)")
    ap.add_argument("--configs", default="a,b,c")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "v6_4_7"
                                         / "S10_screen_score.json"))
    args = ap.parse_args()
    tech = args.tech
    gpu = args.gpu if args.gpu else None

    pairs = [("ctlv2_s%s" % args.src_seed,
              f"v6_4_7_ctlv2_s{args.src_seed}_{tech}_nmos",
              f"v6_4_7_ctlv2_s{args.src_seed}_{tech}_pmos")]
    for c in args.configs.split(","):
        c = c.strip()
        if not c:
            continue
        pairs.append((f"sob_{c}",
                      f"{args.ft_prefix}_{c}_{tech}_nmos",
                      f"{args.ft_prefix}_{c}_{tech}_pmos"))

    results = {}
    for label, nmos, pmos in pairs:
        if not (CKPT / f"{nmos}_best.pt").exists():
            print(f"[skip] {label}: {nmos} missing", file=sys.stderr)
            continue
        print(f"[score] {label} ...", file=sys.stderr)
        results[label] = run_scorer(tech, nmos, pmos, gpu)

    def g(r, k):
        return r.get(k, float("nan"))

    hdr = (f"\n=== S10 screen scores ({tech}) ===\n"
           f"{'label':10s} {'opamp%':>8s} {'gain':>8s} {'flat':>4s} "
           f"{'RO%':>7s} {'invVTC':>7s} {'invTr':>6s} "
           f"{'gm_fwd':>7s} {'gds_fwd':>8s} {'gmb_fwd':>8s} {'off_exc':>9s}")
    print(hdr)
    for label, _, _ in pairs:
        r = results.get(label)
        if not r or "_error" in (r or {}):
            print(f"{label:10s}  ERROR {r.get('_error') if r else 'missing'}")
            continue
        print(f"{label:10s} {g(r,'opamp_gain_err'):8.2f} "
              f"{g(r,'opamp_gain'):8.1f} {str(g(r,'opamp_flat_flag')):>4s} "
              f"{g(r,'ring_osc_period_err'):7.2f} "
              f"{g(r,'inv_vtc_nrmse'):7.2f} {g(r,'inv_tran_post_nrmse'):6.2f} "
              f"{g(r,'deriv_gm_nrmse_fwd'):7.1f} {g(r,'deriv_gds_nrmse_fwd'):8.1f} "
              f"{g(r,'deriv_gmb_nrmse_fwd'):8.1f} {g(r,'offstate_id_excess_max'):9.2e}")
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"\n[score] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
