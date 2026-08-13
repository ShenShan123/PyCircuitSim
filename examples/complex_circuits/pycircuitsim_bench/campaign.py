"""Campaign driver: every deck of one tech through ``run_compare``, in parallel.

The V7.5.x pilot scored 7 hand-picked decks; this module is the scripted
category-by-category expansion (open issue #2 of the 2026-08-10 notes). It
enumerates the same corpus ``run_compare`` sees (**51 decks per tech since
the V7.5.9 curation**: 28 AC, 9 dc_temp, 6 dc_source, 8 transient; V7.5.6 was
75 = 41/14/9/11 and the original corpus 159 = 89/32/15/23), applies the
pilot's per-deck stride policy, fans the decks out over N worker
subprocesses, and aggregates the per-deck JSON rows into one summary table.

The basket is the tree: there is no selection flag here, because the
curations removed the redundant designs from ``designs_tsmc*/`` outright.
What is in the basket, and why each survivor is there, is RESULTS_TSMC.md
§"The curated core basket".

**``tsmc6`` is a duplicate on THIS axis.** Under LEVEL=72 it is an exact
simulation copy of ``tsmc7`` — measured again at V7.5.9 HEAD, 75/75 decks
identical in verdict AND in every miss's relative error to four decimals. It
is still a tech directory because the NN families train separate checkpoints
on it, so it is the training-variance control there; but running it here buys
nothing and costs a fifth of the campaign. Run ``--tech tsmc5,tsmc7,tsmc12,
tsmc16`` unless you are re-validating the duplication itself.

One deck = one ``run_compare`` subprocess (crash isolation: a diverging deck
cannot take the campaign down) with its own work directory (two decks of the
same design run concurrently; ``run_compare`` keys scratch space by design
only). A deck whose JSON already exists is skipped, so an interrupted
campaign resumes by re-running the same command.

Usage::

    python -m examples.complex_circuits.pycircuitsim_bench.campaign \
        --tech tsmc5 --families ac,dc_source,dc_temp --jobs 12 \
        --out <dir> [--refine]

``--refine`` sets ``PYCIRCUITSIM_BENCH_TRAN_REFINE=1`` for the transient
family (and pins the trapezoid for the charge pump, matching the pilot row);
it does not affect AC/DC decks. ``--summarize-only`` rebuilds the summary
from existing JSONs without running anything.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .run_compare import BENCH_ROOT, _decks_of, _designs

#: (category, deck) -> measurement family. Anything not listed is a bug in
#: the corpus enumeration, not a deck to guess about.
FAMILIES: Dict[Tuple[str, str], str] = {
    ("amplifier", "tb_gain.cir"): "ac",
    ("amplifier", "tb_cmrr.cir"): "ac",
    ("amplifier", "tb_psrrp.cir"): "ac",
    ("amplifier", "tb_psrrn.cir"): "ac",
    ("ldo", "tb_loop_max.cir"): "ac",
    ("ldo", "tb_loop_min.cir"): "ac",
    ("ldo", "tb_psrr_max.cir"): "ac",
    ("ldo", "tb_psrr_min.cir"): "ac",
    ("sensing_front_end", "tb_ac.cir"): "ac",
    ("amplifier", "tb_dc.cir"): "dc_temp",
    ("sensing_front_end", "tb_dc.cir"): "dc_temp",
    ("voltage_reference", "tb_dc.cir"): "dc_temp",
    ("ldo", "tb_load.cir"): "dc_source",
    ("ldo", "tb_line_max.cir"): "dc_source",
    ("ldo", "tb_line_min.cir"): "dc_source",
    ("amplifier", "tb_tran.cir"): "tran",
    ("ldo", "tb_tran.cir"): "tran",
    ("charge_pump", "tb_tran.cir"): "tran",
}

#: Pilot stride policy (RESULTS_TSMC.md): full grids where they are cheap,
#: the pilot's measured strides where they are not.
STRIDES: Dict[Tuple[str, str], int] = {
    ("amplifier", "tb_dc.cir"): 25,
    ("amplifier", "tb_tran.cir"): 4,
    ("ldo", "tb_tran.cir"): 4,
    ("charge_pump", "tb_tran.cir"): 20,
}

CATEGORIES = ("amplifier", "ldo", "sensing_front_end", "voltage_reference",
              "charge_pump")


def corpus(tech: str, families: List[str]) -> List[Tuple[str, str, str]]:
    """(category, design, deck) triples of one tech, in stable order."""
    decks: List[Tuple[str, str, str]] = []
    for cat in CATEGORIES:
        for design_dir in _designs(BENCH_ROOT, tech, cat):
            for deck in _decks_of(cat, design_dir):
                fam = FAMILIES.get((cat, deck))
                if fam is None:
                    raise KeyError(f"unclassified deck: {cat}/{deck}")
                if fam in families:
                    decks.append((cat, design_dir.name, deck))
    return decks


def run_deck(tech: str, cat: str, design: str, deck: str, out: Path,
             work: Path, refine: bool, timeout: float) -> Dict[str, Any]:
    """One run_compare subprocess; returns a status record."""
    stem = Path(deck).stem
    stride = STRIDES.get((cat, deck), 1)
    cmd = [sys.executable, "-m",
           "examples.complex_circuits.pycircuitsim_bench.run_compare",
           "--tech", tech, "--category", cat, "--design", design,
           "--deck", deck, "--stride", str(stride), "--out", str(out),
           "--work", str(work / f"{design}_{stem}")]
    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    if refine and FAMILIES[(cat, deck)] == "tran":
        env["PYCIRCUITSIM_BENCH_TRAN_REFINE"] = "1"
        if cat == "charge_pump":
            env["PYCIRCUITSIM_BENCH_TRAN_METHOD"] = "trap"
    log = work / f"{design}_{stem}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        try:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                env=env, timeout=timeout,
                                cwd=str(BENCH_ROOT.parents[1])).returncode
        except subprocess.TimeoutExpired:
            rc = -9
            fh.write(f"\nCAMPAIGN: killed after {timeout:.0f}s\n")
    return {"category": cat, "design": design, "deck": deck, "rc": rc,
            "log": str(log)}


def summarize(tech: str, out: Path, families: List[str]) -> str:
    """Markdown summary over the JSON rows present in ``out``."""
    lines = [f"# AnalogGym campaign — {tech}", ""]
    fam_tot: Dict[str, List[int]] = {}
    for fam in families:
        rows = []
        for cat, design, deck in corpus(tech, [fam]):
            stem = Path(deck).stem
            path = out / f"{tech}_{cat}_{design}_{stem}.json"
            if not path.exists():
                rows.append((cat, design, stem, None))
                continue
            rows.append((cat, design, stem,
                         json.loads(path.read_text())["verdict"]))
        lines += [f"## {fam} ({len(rows)} decks)", "",
                  "| deck | agree | engine | op worst (V) | py / ng s |",
                  "|---|:--:|:--:|--:|--:|"]
        full = 0
        for cat, design, stem, v in rows:
            name = f"{cat}/{design}/{stem}"
            if v is None:
                lines.append(f"| {name} | MISSING | | | |")
                continue
            ok = v["ran"] and v["measured"] > 0 and \
                v["agree"] == v["measured"] and v["missing_py"] == 0
            full += ok
            op = v.get("op_worst_abs")
            missing = ("" if v["missing_py"] == 0
                       else f" (+{v['missing_py']} unmeasured)")
            lines.append(
                f"| {name} | {v['agree']}/{v['measured']}{missing} "
                f"| {'ok' if v['engine_ok'] else 'FAIL'} "
                f"| {'' if op is None else format(op, '.2e')} "
                f"| {v['py_seconds']:.1f} / {v['ng_seconds']:.1f} |")
        fam_tot[fam] = [full, len(rows)]
        lines += ["", f"**{fam}: {full}/{len(rows)} decks fully agree**", ""]
    lines += ["## Totals", ""] + [
        f"- {fam}: {n}/{d}" for fam, (n, d) in fam_tot.items()]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--tech", default="tsmc5")
    ap.add_argument("--families", default="ac,dc_source,dc_temp,tran",
                    help="comma list of ac,dc_source,dc_temp,tran")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True,
                    help="JSON output directory (rows + summary.md)")
    ap.add_argument("--work", type=Path, default=None,
                    help="scratch root (default: <out>/work)")
    ap.add_argument("--refine", action="store_true",
                    help="run the tran family with output refinement "
                         "(charge pump additionally pins the trapezoid)")
    ap.add_argument("--force", action="store_true",
                    help="re-run decks whose JSON already exists")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--deck-timeout", type=float, default=3600.0)
    args = ap.parse_args(argv)

    tech = args.tech.lower()
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    out = args.out
    work = args.work or (out / "work")
    out.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        todo = []
        for cat, design, deck in corpus(tech, families):
            path = out / f"{tech}_{cat}_{design}_{Path(deck).stem}.json"
            if args.force or not path.exists():
                todo.append((cat, design, deck))
        print(f"[campaign] {tech}: {len(todo)} decks to run "
              f"({args.jobs} workers)", flush=True)
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
            futs = {pool.submit(run_deck, tech, cat, design, deck, out, work,
                                args.refine, args.deck_timeout):
                    (cat, design, deck) for cat, design, deck in todo}
            done = 0
            for fut in concurrent.futures.as_completed(futs):
                cat, design, deck = futs[fut]
                rec = fut.result()
                done += 1
                print(f"[campaign] {done}/{len(todo)} rc={rec['rc']} "
                      f"{cat}/{design}/{deck}", flush=True)

    text = summarize(tech, out, families)
    (out / "summary.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
