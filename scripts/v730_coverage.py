#!/usr/bin/env python3
"""V7.3.0 — one coverage map over every accuracy measurement on disk.

The accuracy campaign has run in three passes that each wrote their own tree:

  results/a3_regate/    V6.13.0, complex matrix only, REPORT.md + OMP_REPORT.md
  results/v710_regate/  V7.1.0, device + AC + strict OMP, data.json
  results/v730_regate/  V7.3.0 (this campaign), same layout as v710

Nothing so far could answer "is cell X measured, and by which pass?" without
reading all three by eye. This tool answers it: it merges the passes with
newest-wins precedence, compares the result against the coverage the new
reports require, and emits the missing cells as a job file the gate driver
(`scripts/v710_regate.sh`) consumes directly.

    python scripts/v730_coverage.py                      # coverage report
    python scripts/v730_coverage.py --emit-jobs jobs.txt # what is still missing
    python scripts/v730_coverage.py --emit-jobs j.txt --tag tf --set recipes

A cell with no checkpoint on disk is reported as NO-CKPT, never as a gap: it is
work that cannot be run, not work that was forgotten.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
from typing import Dict, List, Optional, Tuple

if __package__:
    from .v710_regate_collect import is_verdict, rc_of
else:
    from v710_regate_collect import is_verdict, rc_of

ROOT = pathlib.Path(__file__).resolve().parents[1]
CKPT = pathlib.Path(os.environ.get(
    "BSIMAR_CHECKPOINT_DIR",
    ROOT / "external_compact_models" / "neural_network" / "checkpoints",
))

TECHS = ["TSMC5", "TSMC6", "TSMC7", "TSMC12", "TSMC16"]
FAM = {"dn": "DirectNet", "tf": "BSIM-AR", "pfn": "PFN"}

# Suites, and the OMP settings each must be measured at. ring_osc and opamp sit
# on multistable fixed points, so their verdict is only bankable if it holds at
# every thread count (methodology.md §3); the rest are deterministic under the
# thread pin and are taken from a single run.
STRICT_OMP = ("1", "2", "4")
SUITES: Dict[str, Tuple[str, ...]] = {
    "verify_complex_ring_osc": STRICT_OMP,
    "verify_complex_opamp": STRICT_OMP,
    "verify_complex_sram_snm": ("1",),
    "verify_complex_switchcap": ("1",),
    "verify_nn_ac": ("1",),
    "verify_complex_opamp_ac": ("1",),
    "verify_nn_multi_tech_dc": ("1",),
    "verify_nn_multi_tech_tran": ("1",),
}

# The clean control, per family. V7.4.0 retrained every tier of DirectNet and
# BSIM-AR from scratch on the clean recipe, straight into the production slots,
# so clean@large is now simply `large` for both — the v660clean archive detour
# (needed while the DN `large` slot carried the crit30f curriculum) is gone.
CLEAN: Dict[str, Dict[str, str]] = {
    "dn": {"small": "small", "medium": "medium", "large": "large", "xl": "xl"},
    "tf": {"small": "small", "medium": "medium", "large": "large", "xl": "xl"},
    "pfn": {"small": "small", "medium": "medium", "large": "large", "xl": "xl"},
}
CLEAN_TECH_OVERRIDE: Dict[Tuple[str, str, str], str] = {}

# Recipes that survive the V7.3.0 filter (docs/plans/2026-07-27-...md §5).
# Everything else is archive-only or demoted to the dead-end table.
RECIPES: Dict[str, List[str]] = {
    "dn": ["crit30f_large", "crit15m_xl", "corroft_xl", "csob_large"],
    "tf": ["corroft_medium", "corro15_medium",
           "corroft_large", "crit15m_large", "crit30_large",
           "corroft_xl", "corro15_xl", "crit15m_xl", "crit30_xl"],
    "pfn": ["corroft_small"],
}

# Newest pass wins: a cell re-measured in V7.3.0 supersedes its V7.1.0 value,
# which supersedes the V6.13.0 one.
PASSES = [("a3", ROOT / "results" / "a3_regate"),
          ("v710", ROOT / "results" / "v710_regate"),
          ("v730", ROOT / "results" / "v730_regate"),
          ("v740", ROOT / "results" / "v740_regate"),
          ("v742", ROOT / "results" / "v742_regate"),
          ("simple-recheck", ROOT / "results" / "simple_recheck_24c181a")]

CellKey = Tuple[str, str, str, str, str]


def load_json_pass(root: pathlib.Path) -> Dict:
    p = root / "data.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def scan_logs(root: pathlib.Path) -> Dict[CellKey, Optional[str]]:
    """Observed raw logs and their optional scientific verdicts.

    data.json is only rewritten when the collector runs; during a campaign the
    logs are ahead of it. Returning invalid and unfinished observations as
    ``None`` lets callers suppress a stale JSON verdict for the same cell.
    """
    out: Dict[CellKey, Optional[str]] = {}
    for log in root.glob("*/*/*/*.omp*.log"):
        suite, _, omp = log.name[:-4].partition(".omp")
        key = (log.parent.parent.parent.name, log.parent.parent.name,
               log.parent.name.upper(), suite, omp)
        try:
            txt = log.read_text(errors="replace")
        except OSError:
            out[key] = None
            continue
        rc = rc_of(txt)
        out[key] = rc if is_verdict({"rc": rc}) else None
    return out


def build_index(
    only: Optional[List[str]] = None,
) -> Dict[CellKey, str]:
    """(tag, variant, TECH, suite, omp) -> which pass measured it.

    ``only`` restricts which passes count as coverage. A rebuild campaign
    needs that: its cells were all measured by an EARLIER pass, so merging
    every pass would report full coverage and emit zero jobs. Default
    (``None``) merges everything, newest last, as before.
    """
    idx: Dict[CellKey, str] = {}
    for name, root in PASSES:
        if only is not None and name not in only:
            continue
        pass_idx: Dict[CellKey, str] = {}
        data = load_json_pass(root)
        for tag, variants in data.items():
            for variant, suites in variants.items():
                for suite, techs in suites.items():
                    for tech, omps in techs.items():
                        for omp in omps:
                            if not omp.startswith("omp"):
                                continue
                            if not is_verdict(omps[omp]):
                                continue
                            pass_idx[(tag, variant, tech, suite, omp[3:])] = name
        for key, rc in scan_logs(root).items():
            pass_idx.pop(key, None)
            if rc is not None:
                pass_idx[key] = name
        idx.update(pass_idx)
    return idx


def ckpt_exists(tag: str, variant: str, tech: str,
                require_complete: bool = False) -> bool:
    """Both devices present. `require_complete` also demands the done marker.

    A bare `_best.pt` may be a run that was killed mid-training — the trainer
    writes it at every val improvement — so gating one silently produces a
    number for a checkpoint nobody finished. The marker is the discipline
    (AGENTS.md); it is opt-in here because checkpoints predating the marker are
    genuinely complete and would otherwise be excluded.
    """
    t = tech.lower()
    suffixes = ("_best.pt.complete",) if require_complete else ("_best.pt",)
    return all((CKPT / f"{t}_{tag}_{variant}_{d}{sfx}").exists()
               for d in ("nmos", "pmos") for sfx in suffixes)


def variant_for(tag: str, group: str, tech: str, is_clean: bool) -> str:
    if not is_clean:
        return group
    return CLEAN_TECH_OVERRIDE.get((tag, group, tech), CLEAN[tag][group])


def groups(tag: str, which: str) -> List[Tuple[str, bool]]:
    out: List[Tuple[str, bool]] = []
    if which in ("clean", "all"):
        out += [(g, True) for g in CLEAN[tag]]
    if which in ("recipes", "all"):
        out += [(g, False) for g in RECIPES[tag]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="V7.3.0 accuracy coverage map")
    ap.add_argument("--emit-jobs", type=pathlib.Path, default=None,
                    help="Write missing cells as a v710_regate.sh job file")
    ap.add_argument("--tag", default=None, choices=sorted(FAM),
                    help="Restrict to one family")
    ap.add_argument("--set", default="all", choices=["clean", "recipes", "all"])
    ap.add_argument("--techs", default=None,
                    help="Comma list, e.g. TSMC5,TSMC7 (default: all five)")
    ap.add_argument("--passes", default=None,
                    help="Comma list of pass names that count as coverage "
                         "(default: all of "
                         + ",".join(n for n, _ in PASSES) + "). A rebuild "
                         "campaign passes its OWN name so cells measured by "
                         "an earlier pass are re-gated rather than inherited.")
    ap.add_argument("--require-complete", action="store_true",
                    help="Only count a checkpoint that carries its "
                         "*_best.pt.complete marker — use when a training "
                         "wave for these stems is still running")
    ap.add_argument("--fail-on-gaps", action="store_true",
                    help="Exit nonzero when a requested measurement is "
                         "missing or a requested checkpoint group is "
                         "unavailable")
    args = ap.parse_args()

    techs = ([t.strip().upper() for t in args.techs.split(",")]
             if args.techs else TECHS)
    tags = [args.tag] if args.tag else ["dn", "tf", "pfn"]
    only = ([p.strip() for p in args.passes.split(",")]
            if args.passes else None)
    if only is not None:
        known = {n for n, _ in PASSES}
        bad = [p for p in only if p not in known]
        if bad:
            raise SystemExit(
                f"--passes: unknown pass name(s) {bad}; known: {sorted(known)}")
    idx = build_index(only)

    jobs: List[str] = []
    print(f"{'group':30s} {'tech':7s} {'measured':>9s} {'missing':>8s}  by")
    print("-" * 72)
    tot_have = tot_miss = tot_nockpt = 0
    for tag in tags:
        for group, is_clean in groups(tag, args.set):
            label = f"{tag}/{'clean' if is_clean else 'recipe'}/{group}"
            for tech in techs:
                variant = variant_for(tag, group, tech, is_clean)
                if not ckpt_exists(tag, variant, tech, args.require_complete):
                    tot_nockpt += 1
                    print(f"{label:30s} {tech:7s} {'':>9s} {'':>8s}  NO-CKPT")
                    continue
                have, miss, by = 0, [], set()
                for suite, omps in SUITES.items():
                    for omp in omps:
                        src = idx.get((tag, variant, tech, suite, omp))
                        if src:
                            have += 1
                            by.add(src)
                        else:
                            miss.append((suite, omp))
                            jobs.append(f"{tag} {variant} {tech} {suite} {omp}")
                tot_have += have
                tot_miss += len(miss)
                flag = "" if not miss else "  <-- gap"
                print(f"{label:30s} {tech:7s} {have:>9d} {len(miss):>8d}  "
                      f"{','.join(sorted(by)) or '-'}{flag}")

    n_expect = sum(len(v) for v in SUITES.values())
    print("-" * 72)
    print(f"cells measured {tot_have}, missing {tot_miss}, "
          f"no-checkpoint groups {tot_nockpt}  ({n_expect} runs per group-tech)")

    if args.emit_jobs:
        args.emit_jobs.write_text("".join(j + "\n" for j in jobs))
        print(f"wrote {len(jobs)} jobs -> {args.emit_jobs}")
    if args.fail_on_gaps and (tot_miss or tot_nockpt):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
