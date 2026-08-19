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

from .run_compare import BENCH_ROOT, _decks_of, _designs, _sha256

_NN_PARENT = BENCH_ROOT.parents[1] / "external_compact_models"
if str(_NN_PARENT) not in sys.path:
    sys.path.insert(0, str(_NN_PARENT))
from neural_network.config import CHECKPOINT_DIR


def _directnet_stems(tech: str, size: str) -> Dict[str, str]:
    """Checkpoint stems for one explicitly pinned DirectNet campaign."""
    return {
        device: f"{tech.lower()}_dn_{size}_{device}"
        for device in ("nmos", "pmos")
    }


def _require_directnet_checkpoints(tech: str, size: str) -> None:
    """Reject missing or interrupted checkpoints before campaign fan-out."""
    missing: List[Path] = []
    for stem in _directnet_stems(tech, size).values():
        checkpoint = CHECKPOINT_DIR / f"{stem}_best.pt"
        norm = CHECKPOINT_DIR / f"{stem}_norm.npz"
        complete = checkpoint.with_suffix(checkpoint.suffix + ".complete")
        missing.extend(path for path in (checkpoint, norm, complete)
                       if not path.is_file())
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "DirectNet campaign requires completed, explicitly pinned "
            f"checkpoints:\n{rendered}")


def _model_matches_row(row: object, tech: str, model_level: int,
                       checkpoint_size: str) -> bool:
    """Whether one decoded row belongs to this exact model selection."""
    if not isinstance(row, dict):
        return False
    model = row.get("py_model")
    if model_level == 72:
        return model is None or model == {
            "family": "bsim_cmg", "level": 72, "tech": tech,
        }
    if not isinstance(model, dict) or model.get("level") != model_level:
        return False
    checkpoints = model.get("checkpoints")
    if not isinstance(checkpoints, dict):
        return False
    expected = _directnet_stems(tech, checkpoint_size)
    for device, stem in expected.items():
        info = checkpoints.get(device)
        checkpoint = CHECKPOINT_DIR / f"{stem}_best.pt"
        norm = CHECKPOINT_DIR / f"{stem}_norm.npz"
        if not (
            isinstance(info, dict)
            and info.get("stem") == stem
            and bool(info.get("complete"))
            and checkpoint.is_file()
            and norm.is_file()
            and info.get("checkpoint_sha256") == _sha256(checkpoint)
            and info.get("norm_sha256") == _sha256(norm)
        ):
            return False
    return True


def _read_row(path: Path) -> Optional[Dict[str, Any]]:
    """Decode one campaign row, returning ``None`` for an invalid file."""
    try:
        row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return row if isinstance(row, dict) else None


def _row_matches_model(path: Path, tech: str, model_level: int,
                       checkpoint_size: str) -> bool:
    """Whether a resumable row belongs to this exact model selection."""
    row = _read_row(path)
    return row is not None and _model_matches_row(
        row, tech, model_level, checkpoint_size)


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

#: Stride policy: full grids everywhere except the one sweep that is
#: genuinely expensive and genuinely subsampled.
#:
#: **The transient family carries no stride, and must not** (V7.5.11). On a
#: sweep a stride subsamples the abscissa and both simulators are then scored
#: on the shared grid, so it costs resolution and nothing else. On a TRANSIENT
#: it multiplies the solver's timestep (``run_compare``: ``dt = t_step *
#: stride``) — it is a fidelity knob wearing a cost knob's name. The pilot's
#: ``tb_tran @4`` therefore marched PyCircuitSim at 20 ns while NGSPICE marched
#: the deck's own 5 ns, and 15 of the 20 transient disagreements V7.5.10
#: reported were that 4x, not the simulator: at stride 1, ``Leung_NMCNR``
#: goes 0/11 -> 11/11 on tsmc7 and 1/11 -> 11/11 on tsmc12, ``Fan_SMC`` 5/11
#: -> 11/11 on tsmc5, and ``Song_DACFC``/``Peng_IAC``/``Qu2017_AZC`` close on
#: five more tech cells. It cost 4x the transient runtime to learn that.
#:
#: **The charge pump is the exception, and it is an exception BECAUSE it was
#: measured.** Its ``@20`` was validated across strides when it was set
#: (V7.5.3: ``up_imin`` 1.49 % at stride 100, 1.84 % at stride 20), and
#: V7.5.11 re-measured it at stride 1 — 102k committed pieces against
#: NGSPICE's 100k, i.e. matched resolution — where it gets **worse**, not
#: better: 6/6 -> 4/6 on tsmc7 and tsmc16, 5/6 -> 4/6 on tsmc12, with
#: ``up_imin`` blowing from ~1 % to 6-61 %, at 15x the cost (2.2 ks/deck
#: against 0.15 ks). A +-4 uA 10 ps current-reversal spike whose amplitude
#: does not converge as dt approaches the deck's own tstep is an open item in
#: our step controller (RESULTS_TSMC.md), not a resolution deficit — so this
#: entry stays, and stays quoted with that measurement next to it.
STRIDES: Dict[Tuple[str, str], int] = {
    ("amplifier", "tb_dc.cir"): 25,
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
             work: Path, refine: bool, timeout: float, model_level: int,
             checkpoint_size: str) -> Dict[str, Any]:
    """One run_compare subprocess; returns a status record."""
    stem = Path(deck).stem
    stride = STRIDES.get((cat, deck), 1)
    cmd = [sys.executable, "-m",
           "examples.complex_circuits.pycircuitsim_bench.run_compare",
           "--tech", tech, "--category", cat, "--design", design,
           "--deck", deck, "--stride", str(stride), "--out", str(out),
           "--work", str(work / f"{design}_{stem}"),
           "--model-level", str(model_level)]
    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYCIRCUITSIM_TORCH_THREADS="1")
    if model_level == 73:
        pins = _directnet_stems(tech, checkpoint_size)
        env["PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS"] = pins["nmos"]
        env["PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS"] = pins["pmos"]
        env["PYCIRCUITSIM_NN_STRICT_TECH_CODE"] = "1"
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


def _aggregate_voltage_error(
    rows: Dict[Path, Dict[str, Any]],
) -> Optional[Dict[str, float]]:
    """Combine per-row voltage sufficient statistics exactly."""
    n = decks = 0
    relative_error_sum = ss_res = truth_sum = truth_sum_squared = 0.0
    truth_min = float("inf")
    truth_max = float("-inf")
    max_error = 0.0
    for row in rows.values():
        stats = (row.get("op_delta") or {}).get("error_stats")
        if not isinstance(stats, dict) or not stats.get("n"):
            continue
        count = int(stats["n"])
        n += count
        decks += 1
        relative_error_sum += float(stats["relative_error_sum"])
        ss_res += float(stats["sum_squared_error"])
        truth_sum += float(stats["truth_sum"])
        truth_sum_squared += float(stats["truth_sum_squared"])
        truth_min = min(truth_min, float(stats["truth_min"]))
        truth_max = max(truth_max, float(stats["truth_max"]))
        max_error = max(max_error, float(stats["max_error"]))
    if n == 0:
        return None
    ss_total = truth_sum_squared - truth_sum * truth_sum / n
    scale = truth_max - truth_min
    if scale <= 0.0:
        scale = max(abs(truth_min), abs(truth_max), 1e-12)
    return {
        "decks": float(decks), "n": float(n),
        "mre": relative_error_sum / n,
        "r2": 1.0 - ss_res / ss_total if ss_total > 0.0 else float("nan"),
        "nrmse": (ss_res / n) ** 0.5 / scale,
        "max_error": max_error,
    }


def summarize(tech: str, out: Path, families: List[str], model_level: int = 72,
              checkpoint_size: str = "large") -> str:
    """Markdown summary over the JSON rows present in ``out``."""
    model_name = "BSIM-CMG" if model_level == 72 else "DirectNet"
    lines = [
        f"# AnalogGym campaign — {tech} — {model_name} LEVEL={model_level}",
        "",
    ]
    fam_tot: Dict[str, List[int]] = {}
    matching_rows: Dict[Path, Dict[str, Any]] = {}
    for fam in families:
        rows = []
        for cat, design, deck in corpus(tech, [fam]):
            stem = Path(deck).stem
            path = out / f"{tech}_{cat}_{design}_{stem}.json"
            if not path.exists():
                rows.append((cat, design, stem, None))
                continue
            row = _read_row(path)
            if row is None or not _model_matches_row(
                row, tech, model_level, checkpoint_size
            ):
                rows.append((cat, design, stem, "PROVENANCE_MISMATCH"))
                continue
            matching_rows[path] = row
            rows.append((cat, design, stem, row["verdict"]))
        lines += [f"## {fam} ({len(rows)} decks)", "",
                  "| deck | agree | quarantined | engine | op worst (V) "
                  "| py / ng s |",
                  "|---|:--:|:--:|:--:|--:|--:|"]
        full = scored = quarantined = missing = mismatched = 0
        for cat, design, stem, v in rows:
            name = f"{cat}/{design}/{stem}"
            if v is None:
                missing += 1
                lines.append(f"| {name} | MISSING | | | | |")
                continue
            if v == "PROVENANCE_MISMATCH":
                mismatched += 1
                lines.append(
                    f"| {name} | PROVENANCE_MISMATCH | | | | |")
                continue
            # A deck whose every metric is quarantined asks no question of
            # this version, so it is neither a pass nor a fail: it leaves the
            # denominator and is counted in `quarantined` instead.
            whole = v["measured"] == 0 and v.get("not_comparable", 0) > 0
            scored += not whole
            quarantined += whole
            ok = v["ran"] and v["measured"] > 0 and \
                v["agree"] == v["measured"] and v["missing_py"] == 0
            full += ok
            op = v.get("op_worst_abs")
            missing_suffix = ("" if v["missing_py"] == 0
                              else f" (+{v['missing_py']} unmeasured)")
            nc = v.get("not_comparable", 0)
            score = "--" if whole else f"{v['agree']}/{v['measured']}"
            lines.append(
                f"| {name} | {score}{missing_suffix} "
                f"| {nc if nc else ''} "
                f"| {'ok' if v['engine_ok'] else 'FAIL'} "
                f"| {'' if op is None else format(op, '.2e')} "
                f"| {v['py_seconds']:.1f} / {v['ng_seconds']:.1f} |")
        fam_tot[fam] = [full, scored, quarantined, missing, mismatched]
        lines += ["", f"**{fam}: {full}/{scored} decks fully agree**"
                  + (f" ({quarantined} quarantined as invalid test examples "
                     f"— see run_compare.NOT_COMPARABLE)" if quarantined
                     else "")
                  + (f" ({missing} missing)" if missing else "")
                  + (f" ({mismatched} provenance mismatch)"
                     if mismatched else ""), ""]
    lines += ["## Totals", ""] + [
        f"- {fam}: {n}/{s}"
        + (f" ({q} quarantined)" if q else "")
        + (f" ({m} missing)" if m else "")
        + (f" ({p} provenance mismatch)" if p else "")
        for fam, (n, s, q, m, p) in fam_tot.items()]
    voltage = _aggregate_voltage_error(matching_rows)
    lines += ["", "## Voltage-state error", ""]
    if voltage is None:
        lines.append("No comparable operating-point/DC-sweep voltage samples.")
    else:
        lines += [
            "Computed over comparable operating-point and DC-sweep node "
            "voltages; MRE uses a symmetric denominator and NRMSE uses the "
            "technology campaign's NGSPICE voltage range.",
            "",
            "| rows | samples | MRE | R² | NRMSE | max abs(ΔV) |",
            "|--:|--:|--:|--:|--:|--:|",
            f"| {int(voltage['decks'])} | {int(voltage['n'])} "
            f"| {100.0 * voltage['mre']:.6g}% | {voltage['r2']:.8g} "
            f"| {100.0 * voltage['nrmse']:.6g}% "
            f"| {voltage['max_error']:.8g} V |",
        ]
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
    ap.add_argument("--model-level", type=int, choices=(72, 73), default=72,
                    help="PyCircuitSim compact-model level: 72=BSIM-CMG, "
                         "73=DirectNet (default: %(default)s)")
    ap.add_argument("--checkpoint-size",
                    choices=("small", "medium", "large", "xl"),
                    default="large",
                    help="DirectNet per-tech checkpoint tier "
                         "(default: %(default)s)")
    args = ap.parse_args(argv)

    tech = args.tech.lower()
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    out = args.out
    work = args.work or (out / "work")
    out.mkdir(parents=True, exist_ok=True)

    if args.model_level == 73:
        _require_directnet_checkpoints(tech, args.checkpoint_size)

    failed: List[Dict[str, Any]] = []
    if not args.summarize_only:
        todo = []
        for cat, design, deck in corpus(tech, families):
            path = out / f"{tech}_{cat}_{design}_{Path(deck).stem}.json"
            if path.exists() and not args.force and not _row_matches_model(
                path, tech, args.model_level, args.checkpoint_size
            ):
                raise SystemExit(
                    f"Existing row has different or incomplete model "
                    f"provenance: {path}. Use --force or a fresh --out.")
            if args.force or not path.exists():
                todo.append((cat, design, deck))
        print(f"[campaign] {tech}: {len(todo)} decks to run "
              f"({args.jobs} workers)", flush=True)
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
            futs = {pool.submit(run_deck, tech, cat, design, deck, out, work,
                                args.refine, args.deck_timeout,
                                args.model_level, args.checkpoint_size):
                    (cat, design, deck) for cat, design, deck in todo}
            done = 0
            for fut in concurrent.futures.as_completed(futs):
                cat, design, deck = futs[fut]
                rec = fut.result()
                if rec["rc"] != 0:
                    failed.append(rec)
                done += 1
                print(f"[campaign] {done}/{len(todo)} rc={rec['rc']} "
                      f"{cat}/{design}/{deck}", flush=True)

    text = summarize(
        tech, out, families, args.model_level, args.checkpoint_size
    )
    (out / "summary.md").write_text(text)
    print(text)
    if failed:
        print(
            f"[campaign] FAILED: {len(failed)} deck subprocesses returned "
            "nonzero; inspect the recorded logs",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
