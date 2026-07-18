#!/usr/bin/env python3
"""Collect the TSMC6 NN gate results (results/tsmc6_gate/<model>/<size>/*.log)
into a compact per-(model,size) scoreboard + per-cell metric tables, reusing the
existing collector parsers. Verdict = the gate's own exit code recorded on the
``===GATE_DONE rc=N===`` line my driver appends (rc==0 => PASS), which is exactly
the semantics gate_matrix_iso uses. Usage: python scripts/tsmc6_collect.py
"""
from __future__ import annotations
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from recipe_retest_collect import PARSERS                     # noqa: E402
from benchmark_collect import (parse_device_log,              # noqa: E402
                               parse_ac_device_log, parse_ac_complex_log,
                               resolver_line)

BASE = ROOT / "results" / "tsmc6_gate"


def resolved_ckpt(text: str) -> str:
    """The resolved checkpoint stem — skip the 'FORCE LEVEL 73->75' lines that
    precede it for L74/L75 (they also contain '->')."""
    m = re.search(r"->\s*(tsmc6_\w+_best\.pt)", text)
    return m.group(1) if m else (resolver_line(text) or "?")
CIRCS = ["ring_osc", "opamp", "sram_snm", "switchcap"]
MODELS = [("direct", "dn", ["small", "medium", "large", "xl"]),
          ("transformer", "tf", ["small", "medium", "large", "xl"]),
          ("tabpfn", "pfn", ["small", "medium", "large"])]

KEYMETRIC = {  # (parser key, label, unit)
    "ring_osc":  ("period_err_pct", "period_err", "%"),
    "opamp":     ("gain_err_pct",   "gain_err",   "%"),
    "sram_snm":  ("nrmse_max_pct",  "lobeNRMSEmax", "%"),
    "switchcap": ("charge_err_pct", "charge_err", "%"),
}


_DEV_ROW = re.compile(
    r"TSMC6\s*\|\s*\S+\s*\|\s*(\w+)\s*\|\s*([\d.]+|N/A)\s*\|\s*([\d.]+|N/A)\s*\|"
    r".*?\|\s*(PASS|FAIL)")


def parse_dc_tran(text: str) -> dict:
    """verify_nn_dc_tran summary table: rows keyed by test type
    (dc=NMOS DC, pmos_dc, vtc, tran, inv_tran) -> nrmse%, mre%, pass."""
    out = {}
    for m in _DEV_ROW.finditer(text):
        test, nrmse, mre, st = m.group(1), m.group(2), m.group(3), m.group(4)
        out[test] = {"nrmse": None if nrmse == "N/A" else float(nrmse),
                     "mre": None if mre == "N/A" else float(mre),
                     "passed": st == "PASS"}
    r = re.search(r"RESULT:\s*ALL\s+(\d+)\s+tests\s+PASSED\s+out\s+of\s+(\d+)", text)
    if not r:
        r = re.search(r"RESULT:.*?(\d+)\s*/\s*(\d+)", text)
    if r:
        out["_n_pass"], out["_n"] = int(r.group(1)), int(r.group(2))
    return out


def rc_of(log: Path):
    if not log.exists():
        return None
    m = re.search(r"===GATE_DONE rc=(-?\d+)===", log.read_text(errors="replace"))
    return int(m.group(1)) if m else None


def cell(model, size):
    d = BASE / model / size
    out = {"complex": {}, "n_complex": 0, "ckpt": None}
    # complex 4 cells
    for c in CIRCS:
        log = d / f"verify_complex_{c}.log"
        rc = rc_of(log)
        info = {"rc": rc, "verdict": None, "metric": None}
        if rc is not None:
            info["verdict"] = ("PASS" if rc == 0 else
                               "TIMEOUT" if rc == 124 else "FAIL")
            txt = log.read_text(errors="replace")
            out["ckpt"] = out["ckpt"] or resolved_ckpt(txt)
            try:
                parsed = PARSERS[c](txt)
                k = KEYMETRIC[c][0]
                info["metric"] = parsed.get(k)
                info["status_parsed"] = parsed.get("status") or parsed.get("gate")
            except Exception as e:
                info["metric"] = f"parse-err:{e}"
        out["complex"][c] = info
        if info["verdict"] == "PASS":
            out["n_complex"] += 1
    # device DC + inverter (verify_nn_dc_tran)
    dev_log = d / "verify_nn_dc_tran.log"
    if dev_log.exists():
        txt = dev_log.read_text(errors="replace")
        out["device_rc"] = rc_of(dev_log)
        out["device"] = parse_dc_tran(txt)
        out["ckpt"] = out["ckpt"] or resolved_ckpt(txt)
    # device AC (verify_nn_ac)
    ac_log = d / "verify_nn_ac.log"
    if ac_log.exists():
        out["ac"] = parse_ac_device_log(ac_log.read_text(errors="replace"))
        out["ac_rc"] = rc_of(ac_log)
    # opamp AC (verify_complex_opamp_ac)
    oac_log = d / "verify_complex_opamp_ac.log"
    if oac_log.exists():
        out["opamp_ac"] = parse_ac_complex_log(oac_log.read_text(errors="replace"))
        out["opamp_ac_rc"] = rc_of(oac_log)
    return out


def fmt(v, nd=2):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def main():
    print("# TSMC6 NN gate scoreboard (clean recipe, vs NGSPICE BSIM-CMG L72)\n")
    print("## Complex 4-cell matrix (ring / opamp / sram / switchcap) — PASS count + key metric\n")
    hdr = "| model | size | complex | ring period_err% | opamp gain_err% | sram lobeNRMSE% | switchcap chg_err% |"
    print(hdr); print("|" + "---|" * 7)
    data = {}
    for model, tag, sizes in MODELS:
        for size in sizes:
            c = cell(model, size)
            data[(model, size)] = c
            if not any(c["complex"][x]["rc"] is not None for x in CIRCS):
                continue
            def pc(cc):
                info = c["complex"][cc]
                if info["rc"] is None:
                    return "·"
                mark = {"PASS": "✓", "TIMEOUT": "⏱", "FAIL": "✗"}.get(info["verdict"], "✗")
                return f"{fmt(info['metric'])} {mark}"
            print(f"| {tag} | {size} | **{c['n_complex']}/4** | "
                  f"{pc('ring_osc')} | {pc('opamp')} | {pc('sram_snm')} | {pc('switchcap')} |")
    # device + AC. NMOS/PMOS DC as nrmse%/mre%; inverter VTC + tran nrmse%.
    print("\n## Device + inverter (verify_nn_dc_tran) NRMSE% / MRE%  +  AC pass\n")
    print("| model | size | NMOS DC nrmse/mre% | PMOS DC nrmse/mre% | inv VTC% | inv tran% | dev pass | AC dev | opamp AC |")
    print("|" + "---|" * 9)
    for model, tag, sizes in MODELS:
        for size in sizes:
            c = data[(model, size)]
            dev = c.get("device", {})
            if not dev and c.get("device_rc") is None:
                continue
            def dccol(k):
                r = dev.get(k)
                if not r:
                    return "—"
                return f"{fmt(r.get('nrmse'))} / {fmt(r.get('mre'))}"
            def ncol(k):
                r = dev.get(k)
                return fmt(r.get("nrmse")) if r else "—"
            devpass = (f"{dev.get('_n_pass')}/{dev.get('_n')}" if dev.get('_n')
                       else ("PASS" if c.get('device_rc') == 0
                             else "FAIL" if c.get('device_rc') is not None else "—"))
            ac = c.get("ac", {})
            acs = f"{ac.get('n_pass','—')}/{ac.get('n','—')}" if ac else "—"
            oac = c.get("opamp_ac", {}).get("gate", "—")
            print(f"| {tag} | {size} | {dccol('dc')} | {dccol('pmos_dc')} | "
                  f"{ncol('vtc')} | {ncol('inv_tran')} | {devpass} | {acs} | {oac} |")
    # checkpoints seen
    print("\n## Resolved checkpoints (sanity)\n")
    for (model, size), c in data.items():
        if c.get("ckpt"):
            print(f"- {model}/{size}: `{c['ckpt']}`")


if __name__ == "__main__":
    main()
