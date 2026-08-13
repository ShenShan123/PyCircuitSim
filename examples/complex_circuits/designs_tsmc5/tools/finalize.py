"""Re-measure every built design with the reporting deck set and rewrite result.json.

The sizing loops evaluate against a fast, coarse deck set; this pass runs the
full one and is the single source of the numbers in RESULTS.md.  Keeping it
separate means a bench fix does not require re-running a search.

This module also owns the measurement semantics shared by every artifact:
``run_all.py`` produces its rows by calling :func:`finalize`, so summary.csv /
run_all.json and result.json / RESULTS.md can never disagree.  The AC loop
decks run through ``acstab.run_deck_auto`` (wrap-aware true phase margin, GBW
as the 0 dB crossover of the dumped sweep) and the PTAT sensor DC decks
through ``sfe.run_sfe_dc`` (full-sweep monotonicity/smoothness metrics plus
the pseudo-transient fallback flag).
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from acstab import run_deck_auto                                 # noqa: E402
from meas import SimError, run_deck                              # noqa: E402
from sfe import run_sfe_dc                                       # noqa: E402

from amp_spec import report as amp_report, score as amp_score    # noqa: E402
from sfe import sfe_report, sfe_score                            # noqa: E402

# category -> {deck: ngspice control block}.  One table for every runner.
CONTROLS: Dict[str, Dict[str, str]] = {
    "amplifier": {
        "tb_gain.cir": "ac dec 20 0.1 10G",
        "tb_cmrr.cir": "ac dec 20 0.1 10G",
        "tb_psrrp.cir": "ac dec 20 0.1 10G",
        "tb_psrrn.cir": "ac dec 20 0.1 10G",
        "tb_dc.cir": "dc temp 125 -40 -0.1",   # hot-to-cold: see build_amp.py
        "tb_tran.cir": None,                    # filled from design.json
    },
    "ldo": {
        "tb_load.cir": None,
        "tb_line_max.cir": None,
        "tb_line_min.cir": None,
        "tb_loop_max.cir": "ac dec 20 0.1 1G",
        "tb_loop_min.cir": "ac dec 20 0.1 1G",
        "tb_psrr_max.cir": "ac dec 20 0.1 1G",
        "tb_psrr_min.cir": "ac dec 20 0.1 1G",
        "tb_tran.cir": "tran 20n 44u",
    },
    # Cold-to-hot, matching the source bench and sizing flow's qualified startup.
    # tb_ac.cir exists only for SMCNR_SE_2st_AMP, the amplifier shipped in this
    # category (see tools/sfe_amp.py); the PTAT sensors skip it.
    "sensing_front_end": {"tb_dc.cir": "dc temp -20 120 0.5",
                          "tb_ac.cir": "ac dec 20 0.1 1G"},
    # Cold-to-hot, matching the category's own bench (size_vref.DC_CONTROL):
    # the per-device three_output design solves from -40 C by continuation but
    # not from a 125 C first point.
    "voltage_reference": {"tb_dc.cir": "dc temp -40 125 0.5"},
    "charge_pump": {"tb_tran.cir": "tran 2p 200n"},
}


def _ldo_controls(design_dir: Path) -> Dict[str, str]:
    """LDO sweep limits come from the design's own reference and load range."""
    from size_ldo import line_control
    d = json.loads((design_dir / "design.json").read_text())
    vdd = d["vdd"]
    imin, imax = d["iload_min"], d["iload_max"]
    return {
        "tb_load.cir": f"dc Iload {imin:g} {imax:g} {(imax - imin) / 100:g}",
        "tb_line_max.cir": line_control("max", vdd),
        "tb_line_min.cir": line_control("min", vdd),
    }


def _amp_tran_control(design_dir: Path) -> str:
    from build_amp import TECH, TRAN_MAXSTEP
    d = json.loads((design_dir / "design.json").read_text())
    step_time = 10.0 / d.get("gbw_ideal", 1e6)
    ctl = f"tran {step_time / 2000:.6g} {2.2 * step_time:.6g}"
    cap = TRAN_MAXSTEP.get((TECH, design_dir.name.lower()))
    return f"{ctl} 0 {cap}" if cap else ctl


def _verdict(category: str, metrics: Dict, design_dir: Path) -> Tuple[Dict, float]:
    if category == "amplifier":
        return amp_report(metrics), amp_score(metrics)
    if category == "sensing_front_end":
        # SMCNR_SE_2st_AMP is an amplifier shipped in this category; it is the
        # only sensing front end with an AC bench and has its own gates.
        if (design_dir / "tb_ac.cir").exists():
            from sfe_amp import sfe_amp_report, sfe_amp_score
            return sfe_amp_report(metrics), sfe_amp_score(metrics)
        return sfe_report(metrics), sfe_score(metrics)
    if category == "ldo":
        from size_ldo import ldo_report, ldo_score
        return ldo_report(metrics), ldo_score(metrics)
    if category == "voltage_reference":
        from size_vref import SKY, outputs_of, vref_report, vref_score
        from geom_port import parse_generic
        outs: List[str] = []
        rj = design_dir / "result.json"
        if rj.exists():
            try:
                outs = json.loads(rj.read_text()).get("outputs", [])
            except json.JSONDecodeError:
                outs = []
        if not outs:
            # Derive from the source rather than fall back to an empty list:
            # every vref_report check is an all() and all([]) is True, so an
            # empty output list would silently report a perfect pass.
            subs, _ = parse_generic(SKY / design_dir.name / "netlist.spice")
            outs = outputs_of(subs)
        return vref_report(metrics, outs), vref_score(metrics, outs)
    if category == "charge_pump":
        from size_cp import cp_report, cp_score
        return cp_report(metrics), cp_score(metrics)
    return {}, 0.0


def finalize(args: Tuple[str, Path]) -> Dict:
    category, design_dir = args
    controls = dict(CONTROLS[category])
    if category == "ldo":
        controls.update(_ldo_controls(design_dir))
    if category == "amplifier":
        controls["tb_tran.cir"] = _amp_tran_control(design_dir)

    # PTAT sensors get the full-sweep dump so the verdict sees every solved
    # point; SMCNR_SE_2st_AMP (the amplifier in this category) is gated on its
    # AC bench instead and its tb_dc runs plain.
    is_ptat = (category == "sensing_front_end"
               and not (design_dir / "tb_ac.cir").exists())

    metrics: Dict = {}
    errors: List[str] = []
    for deck, control in controls.items():
        path = design_dir / deck
        if not path.exists():
            continue
        tag = deck.replace(".cir", "")
        try:
            if is_ptat and deck == "tb_dc.cir":
                part, sweep_warnings = run_sfe_dc(path, control, design_dir,
                                                  tag, timeout=1800)
                metrics.update(part)
                errors.extend(sweep_warnings)
            else:
                metrics.update(run_deck_auto(path, control, design_dir,
                                             tag, timeout=1800))
        except SimError as exc:
            # The amplifier slew bench ships a second nodeset variant; some
            # designs' transient op only solves from one of the two seeds.
            alt = design_dir / "tb_tran_altns.cir"
            if category == "amplifier" and deck == "tb_tran.cir" and alt.exists():
                try:
                    metrics.update(run_deck(alt, control, design_dir,
                                            "tb_tran_altns", timeout=1800))
                    continue
                except SimError as exc2:
                    errors.append(f"{deck}: {str(exc2).splitlines()[0]}")
                    continue
            errors.append(f"{deck}: {str(exc).splitlines()[0]}")

    # Some high-gain loops lose their Newton branch in one monolithic 165 C
    # sweep.  Sweep outward from the qualified 25 C operating point in both
    # directions, then recombine the two contiguous ranges.  This still covers
    # every temperature point; it only changes the continuation path.
    if category == "amplifier" and metrics.get("tc") is None:
        segmented: Dict = {}
        segment_errors: List[str] = []
        for tag, control in (("hot", "dc temp 25 125 0.1"),
                             ("cold", "dc temp 25 -40 -0.1")):
            try:
                part = run_deck(design_dir / "tb_dc.cir", control, design_dir,
                                f"tb_dc_{tag}", timeout=900)
                segmented.update({k: v for k, v in part.items() if v is not None})
            except SimError as exc:
                segment_errors.append(f"tb_dc.cir ({tag}): "
                                      f"{str(exc).splitlines()[0]}")
        needed = ("max_hot", "min_hot", "avg_hot",
                  "max_cold", "min_cold", "avg_cold")
        if all(segmented.get(k) is not None for k in needed):
            metrics.update(segmented)
            metrics["maxval"] = max(segmented["max_hot"], segmented["max_cold"])
            metrics["minval"] = min(segmented["min_hot"], segmented["min_cold"])
            metrics["avgval"] = ((100 * segmented["avg_hot"]
                                  + 65 * segmented["avg_cold"]) / 165)
            metrics["ppavl"] = metrics["maxval"] - metrics["minval"]
            metrics["tc"] = metrics["ppavl"] / metrics["avgval"] / 165
            for key in ("power", "vos25", "vout25", "ivdd25"):
                if segmented.get(key) is not None:
                    metrics[key] = segmented[key]
            errors = [e for e in errors if not e.startswith("tb_dc.cir:")]
        else:
            errors.extend(segment_errors)

    # A temperature sweep whose first point fails yields no data at all, and
    # then every measurement in the deck reports "out of interval" -- including
    # Power and the offset, which have nothing to do with temperature.  A few
    # designs fail at both ends of -40..125 C.  Re-run those over a narrow band
    # around 25 C, which is all Power / vos25 / vout25 actually need; the
    # temperature-coefficient measurements stay unavailable and are reported so.
    if category == "amplifier" and metrics.get("power") is None:
        try:
            narrow = run_deck(design_dir / "tb_dc.cir", "dc temp 30 20 -5",
                              design_dir, "tb_dc_narrow", timeout=600)
            for key in ("power", "vos25", "vout25", "ivdd25"):
                if narrow.get(key) is not None:
                    metrics[key] = narrow[key]
            metrics["tc_unavailable"] = 1.0
        except SimError as exc:
            errors.append(f"tb_dc.cir (narrow): {str(exc).splitlines()[0]}")

    prev = {}
    rj = design_dir / "result.json"
    if rj.exists():
        try:
            prev = json.loads(rj.read_text())
        except json.JSONDecodeError:
            prev = {}

    verdict, score = _verdict(category, metrics, design_dir)
    out = dict(prev)
    out.update({"design": design_dir.name, "category": category,
                "metrics": metrics, "pass": verdict, "score": score,
                "errors": errors})
    # The sizing/polish passes write a legacy single-string "error" field;
    # this pass re-derives the authoritative errors[] above, so a carried
    # copy would go stale and contradict it (report.has_errors reads both).
    out.pop("error", None)
    if category == "charge_pump":
        up, lo = metrics.get("up_iavg"), metrics.get("lo_iavg")
        out["mismatch_pct"] = (abs(abs(up) - abs(lo)) / max(abs(up), abs(lo))
                               * 100) if up and lo else None
    rj.write_text(json.dumps(out, indent=2, default=str))
    return out


def main() -> None:
    wanted = sys.argv[1:] or list(CONTROLS)
    jobs: List[Tuple[str, Path]] = []
    for cat in wanted:
        cdir = ROOT / cat
        if not cdir.is_dir():
            continue
        for d in sorted(cdir.iterdir()):
            if d.is_dir() and (d / "netlist.spice").exists():
                jobs.append((cat, d))
    if not jobs:
        print("nothing built yet")
        return

    done = []
    workers = int(os.environ.get("FINALIZE_WORKERS", "8"))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(finalize, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            done.append(r)
            p = r["pass"]
            flag = f" ERR:{len(r['errors'])}" if r["errors"] else ""
            print(f"[{len(done):2d}/{len(jobs)}] {r['category']:18s} "
                  f"{r['design']:36s} pass={sum(p.values())}/{len(p) or 1}"
                  f"{flag}", flush=True)


if __name__ == "__main__":
    main()
