#!/usr/bin/env python3
"""Benchmark Phase E — aggregate the per-(size,tech) NN test logs into a report.

Parses results/benchmark_sml/<size>/<tech>/<suite>.log (captured stdout of the 6
NN suites) and emits:
  - results/benchmark_sml/REPORT.md   (human-facing tables)
  - results/benchmark_sml/benchmark_data.json  (machine-readable)

Device suites (verify_nn_multi_tech_{dc,tran}) print a SUMMARY TABLE with
NRMSE%/MRE%/R2/MaxErr per config; we aggregate per device. Complex suites print a
gate line (period/gain/charge/SNM err -> PASS/FAIL) plus a waveform metric line.
Regexes are tolerant; raw gate lines are preserved in the report.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "benchmark_sml"
SIZES = ["small", "medium", "large"]
TECHS = ["tsmc5", "tsmc7", "tsmc12", "tsmc16"]
DEV_SUITES = ["verify_nn_multi_tech_dc", "verify_nn_multi_tech_tran"]
CPX_SUITES = ["verify_complex_ring_osc", "verify_complex_opamp",
              "verify_complex_sram_snm", "verify_complex_switchcap"]
# AC (small-signal) suites — V6.5. Device CS-amp (per-checkpoint) + opamp
# open-loop. RO/SRAM excluded (no stable amplifying OP → no .ac ground truth).
AC_DEV_SUITE = "verify_nn_ac"
AC_CPX_SUITE = "verify_complex_opamp_ac"
AC_SUITES = [AC_DEV_SUITE, AC_CPX_SUITE]

NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

# ── parametric summary-table row: "label | sweep | nrmse | mre | r2 | maxerrUNIT | status"
def parse_device_log(text: str) -> dict:
    rows = []
    for ln in text.splitlines():
        if ln.count(" | ") < 5:
            continue
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) < 7:
            continue
        label, sweep = parts[0], parts[1]
        m = re.match(rf"({NUM})", parts[2])
        if not m:                       # header / separator row
            continue
        try:
            nrmse = float(parts[2]); mre = float(parts[3]); r2 = float(parts[4])
        except ValueError:
            continue
        me = re.match(rf"({NUM})\s*([a-zA-Z%]*)", parts[5])
        maxerr = float(me.group(1)) if me else float("nan")
        unit = me.group(2) if me else ""
        status = parts[6]
        dev = "nmos" if "nmos" in label.lower() else ("pmos" if "pmos" in label.lower() else "all")
        rows.append(dict(label=label, sweep=sweep, dev=dev, nrmse=nrmse, mre=mre,
                         r2=r2, maxerr=maxerr, unit=unit,
                         passed=status.upper().startswith("PASS")))
    out = {}
    for dev in ("nmos", "pmos", "all"):
        drows = [r for r in rows if r["dev"] == dev]
        if not drows:
            continue
        base = next((r for r in drows if "base" in r["label"].lower()), drows[0])
        out[dev] = dict(
            n=len(drows), n_pass=sum(r["passed"] for r in drows),
            nrmse_mean=round(mean(r["nrmse"] for r in drows), 2),
            nrmse_med=round(median(r["nrmse"] for r in drows), 2),
            nrmse_max=round(max(r["nrmse"] for r in drows), 2),
            mre_mean=round(mean(r["mre"] for r in drows), 2),
            r2_min=round(min(r["r2"] for r in drows), 4),
            maxerr_max=round(max(r["maxerr"] for r in drows), 3),
            unit=drows[0]["unit"],
            base_nrmse=round(base["nrmse"], 2), base_mre=round(base["mre"], 2),
            base_r2=round(base["r2"], 5), base_maxerr=round(base["maxerr"], 3),
        )
    res = re.search(r"RESULT:\s*(.+)", text)
    return {"by_dev": out, "result": res.group(1).strip() if res else "?"}

WAVE = re.compile(rf"MRE=({NUM})%\s+R2=({NUM})\s+NRMSE=({NUM})%\s+MaxErr=({NUM})\s*([a-zA-Z]*)")

def _wave(text: str):
    m = WAVE.search(text)
    if not m:
        return None
    return dict(mre=float(m.group(1)), r2=float(m.group(2)),
                nrmse=float(m.group(3)), maxerr=float(m.group(4)), unit=m.group(5))

def parse_complex_log(suite: str, text: str) -> dict:
    d = {"wave": _wave(text), "gate": None, "headline": "", "raw": []}
    pat = {
        "verify_complex_ring_osc": rf"period error\s*=\s*({NUM})%\s*->\s*(PASS|FAIL)",
        "verify_complex_opamp": rf"gain error\s*=\s*({NUM})%\s+trip shift\s*=\s*({NUM})mV\s*->\s*(PASS|FAIL)",
        "verify_complex_switchcap": rf"charge err\s*=\s*({NUM})%.*?->\s*(PASS|FAIL)",
    }.get(suite)
    for ln in text.splitlines():
        if "->" in ln and ("PASS" in ln or "FAIL" in ln):
            d["raw"].append(ln.strip())
        if "SNM" in ln and ("DN" in ln or "NG" in ln):
            d["raw"].append(ln.strip())
    if pat:
        m = re.search(pat, text)
        if m:
            d["gate"] = m.groups()[-1]
            if suite == "verify_complex_opamp":
                d["headline"] = f"gain_err={m.group(1)}% trip_shift={m.group(2)}mV"
            elif suite == "verify_complex_ring_osc":
                d["headline"] = f"period_err={m.group(1)}%"
            elif suite == "verify_complex_switchcap":
                d["headline"] = f"charge_err={m.group(1)}%"
    if suite == "verify_complex_sram_snm":
        # Authoritative gate (verify_complex_sram_snm.main): "both butterfly lobes
        # positive across NFIN corners" == the `all-positive: yes/NO` summary flag.
        # force_ic is an informational probe, NOT part of the gate.
        snm = re.search(rf"NG SNM=({NUM})mV\s+DN SNM=({NUM})mV", text)
        fic = re.search(r"force_ic:\s*state1=(\w+)\s*state0=(\w+)", text)
        ap = re.search(r"all-positive:\s*(yes|NO)", text)
        hl = []
        if snm:
            hl.append(f"NG_SNM={snm.group(1)}mV DN_SNM={snm.group(2)}mV")
        if fic:
            hl.append(f"force_ic={fic.group(1)}/{fic.group(2)}")
        d["headline"] = "  ".join(hl) if hl else "—"
        d["gate"] = ("PASS" if ap and ap.group(1) == "yes"
                     else ("FAIL" if ap else "?"))
    # fallback overall gate from any explicit verdict
    if d["gate"] is None:
        vs = re.findall(r"->\s*(PASS|FAIL)", text)
        d["gate"] = ("PASS" if vs and all(v == "PASS" for v in vs) else ("FAIL" if vs else "?"))
    return d

# ── AC device gate: rows "AC | tech_device | gain0_err | f3db_ratio | nrmse | phase | status"
def parse_ac_device_log(text: str) -> dict:
    by_dev = {}
    for ln in text.splitlines():
        s = ln.strip()
        if not s.startswith("AC |"):
            continue
        parts = [p.strip() for p in s.split("|")]
        if len(parts) < 7:
            continue
        label = parts[1]
        dev = "nmos" if "nmos" in label else ("pmos" if "pmos" in label else "?")
        status = parts[6].upper()
        try:
            row = dict(
                gain0_err_db=float(parts[2]), f3db_ratio=float(parts[3]),
                mag_nrmse_pct=float(parts[4]), phase_deg=float(parts[5]),
                passed=status.startswith("PASS"), status=status)
        except ValueError:
            row = dict(passed=False, status=status, gain0_err_db=float("nan"),
                       f3db_ratio=float("nan"), mag_nrmse_pct=float("nan"),
                       phase_deg=float("nan"))
        by_dev[dev] = row
    res = re.search(r"RESULT:\s*(.+)", text)
    return {"by_dev": by_dev, "result": res.group(1).strip() if res else "?",
            "n": len(by_dev), "n_pass": sum(r["passed"] for r in by_dev.values())}


# ── AC opamp gate: "opamp AC <tech>: dc_gain_err=..dB gbw_ratio=.. pm_err=..deg magNRMSE=..% -> PASS"
_AC_OPAMP = re.compile(
    rf"opamp AC \w+:\s*dc_gain_err=({NUM})dB\s+gbw_ratio=(\S+)\s+"
    rf"pm_err=(\S+)deg\s+magNRMSE=({NUM})%\s*->\s*(PASS|FAIL)")

def parse_ac_complex_log(text: str) -> dict:
    m = _AC_OPAMP.search(text)
    if not m:
        vs = re.findall(r"->\s*(PASS|FAIL)", text)
        return {"gate": ("PASS" if vs and all(v == "PASS" for v in vs)
                         else ("FAIL" if vs else "?")), "headline": "—"}
    dc_err, gbw, pm, nrmse, gate = m.groups()
    return {"gate": gate, "dc_gain_err_db": float(dc_err), "gbw_ratio": gbw,
            "pm_err_deg": pm, "mag_nrmse_pct": float(nrmse),
            "headline": f"dc_gain_err={dc_err}dB gbw_ratio={gbw} pm_err={pm}deg"}


def resolver_line(text: str) -> str:
    m = re.search(r"\[NN-resolver\].*?->\s*(\S+)", text)
    return m.group(1) if m else "?"

def load():
    data = {}
    for size in SIZES:
        data[size] = {}
        for tech in TECHS:
            d = {"dev": {}, "cpx": {}, "ac": {}, "ckpt": None}
            for suite in DEV_SUITES:
                p = BASE / size / tech / f"{suite}.log"
                if p.exists():
                    t = p.read_text(errors="replace")
                    d["dev"][suite] = parse_device_log(t)
                    d["ckpt"] = d["ckpt"] or resolver_line(t)
            for suite in CPX_SUITES:
                p = BASE / size / tech / f"{suite}.log"
                if p.exists():
                    t = p.read_text(errors="replace")
                    d["cpx"][suite] = parse_complex_log(suite, t)
                    d["ckpt"] = d["ckpt"] or resolver_line(t)
            p = BASE / size / tech / f"{AC_DEV_SUITE}.log"
            if p.exists():
                d["ac"][AC_DEV_SUITE] = parse_ac_device_log(
                    p.read_text(errors="replace"))
            p = BASE / size / tech / f"{AC_CPX_SUITE}.log"
            if p.exists():
                d["ac"][AC_CPX_SUITE] = parse_ac_complex_log(
                    p.read_text(errors="replace"))
            data[size][tech] = d
    return data

def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)

CPX_SHORT = {"verify_complex_ring_osc": "ring_osc", "verify_complex_opamp": "opamp",
             "verify_complex_sram_snm": "sram_snm", "verify_complex_switchcap": "switchcap"}


def _fmtnum(x, nd=2):
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return "—"


def _ac_section(data) -> list:
    """Top-level AC small-signal accuracy section (V6.5)."""
    L = [
        "## AC small-signal accuracy (V6.5)",
        "",
        "First-ever NGSPICE-gated evaluation of DirectNet (LEVEL=73) AC fidelity. "
        "The NN's small-signal capacitances are autograd derivatives of its predicted "
        "terminal charges (cgd=∂qg/∂Vd, cdd=∂qd/∂Vd, …) — a quantity no prior gate "
        "measured. Ground truth = NGSPICE `.ac` on the identical BSIM-CMG (LEVEL=72) "
        "OSDI model. Two circuit classes (AC needs a stable amplifying OP, so the "
        "free-running ring oscillator and bistable SRAM are out of scope):",
        "",
        "- **Device CS-amp** — per-checkpoint NMOS/PMOS common-source amplifier, no "
        "external load cap so the device's own Cgd/Cdd set the pole; gates gain0 err "
        "≤1.5 dB, f3db ratio ∈[0.7,1.43], mag NRMSE ≤10%. The passband phase is "
        "reported (not gated): deep in-band it matches (<7°), but at/beyond the −3 dB "
        "corner NG carries a strong Cgd-feedforward RHP-zero phase lag the NN does not "
        "reproduce — a distinct limitation from the (excellent) cap-driven pole.",
        "- **Opamp open-loop** — two-stage Miller opamp; gates DC-gain err ≤3 dB, "
        "GBW ratio ∈[0.6,1.67], phase-margin err ≤15° (linear mag NRMSE reported, "
        "not gated — dominated by the 40 dB passband plateau).",
        "",
        "**Findings.** (1) AC **gain** fidelity is excellent everywhere — device gain0 "
        "err <1.5 dB in 24/24 cells (mean 0.55–0.86 dB) — so the autograd gm/gds the NN "
        "feeds the AC stamp are accurate. (2) The dominant **cap-driven pole** is mostly "
        "faithful (f3db ratio ≈1.0 for the well-fit cells) but capacity/tech-variable: "
        "tsmc5 NMOS and tsmc12/16 PMOS under-predict the output cap (ratio 1.1–1.6), so "
        "13/24 clear the magnitude gate. (3) The high-frequency **phase** (Cgd-feedforward "
        "RHP zero) is not reproduced — a clean, specific transcapacitance limitation. "
        "(4) The **opamp** AC inherits the DC value-surface fragility (0/12): the gain "
        "collapses or over-predicts at most cells, BUT where the OP lands in the good "
        "basin (tsmc12-large) the NN reproduces GBW to 0.97× and phase margin to 1.3° — "
        "the dynamics are right, the DC-gain *level* is the value-surface-owned miss. "
        "**No retraining is warranted:** the gaps are value-surface- and feedforward-"
        "owned, not a charge-derivative (dQ/dV) deficiency (which would have shown as bad "
        "gain *and* bad pole everywhere, the opposite of what is measured).",
        "",
    ]
    # cross-size tally
    srows = []
    for size in SIZES:
        dev_pass = dev_tot = op_pass = op_tot = 0
        g0errs, nrmses = [], []
        for tech in TECHS:
            ac = data[size][tech].get("ac", {})
            dv = ac.get(AC_DEV_SUITE)
            if dv:
                for r in dv["by_dev"].values():
                    dev_tot += 1
                    dev_pass += 1 if r["passed"] else 0
                    if r["gain0_err_db"] == r["gain0_err_db"]:  # not NaN
                        g0errs.append(r["gain0_err_db"])
                    if r["mag_nrmse_pct"] == r["mag_nrmse_pct"]:
                        nrmses.append(r["mag_nrmse_pct"])
            op = ac.get(AC_CPX_SUITE)
            if op:
                op_tot += 1
                op_pass += 1 if op.get("gate") == "PASS" else 0
        srows.append([
            size, f"{dev_pass}/{dev_tot}" if dev_tot else "—",
            f"{op_pass}/{op_tot}" if op_tot else "—",
            round(mean(g0errs), 2) if g0errs else "—",
            round(mean(nrmses), 2) if nrmses else "—"])
    L += ["### Cross-size AC summary", "",
          md_table(["Size", "Device CS-amp PASS", "Opamp PASS",
                    "Device mean gain0-err dB", "Device mean magNRMSE%"], srows), ""]

    # device CS-amp gate matrix
    L += ["### Device CS-amp AC gate by capacity", ""]
    rows = []
    for tech in TECHS:
        for dev in ("nmos", "pmos"):
            cells = []
            for size in SIZES:
                dv = data[size][tech].get("ac", {}).get(AC_DEV_SUITE)
                r = dv["by_dev"].get(dev) if dv else None
                cells.append(r["status"] if r else "—")
            rows.append([tech, dev] + cells)
    L += [md_table(["Tech", "Dev", "small", "medium", "large"], rows), ""]

    # device CS-amp gain0-err dB by capacity (headline accuracy number)
    L += ["### Device CS-amp gain0 error (dB) by capacity (lower = better)", ""]
    rows = []
    for tech in TECHS:
        for dev in ("nmos", "pmos"):
            cells = []
            for size in SIZES:
                dv = data[size][tech].get("ac", {}).get(AC_DEV_SUITE)
                r = dv["by_dev"].get(dev) if dv else None
                cells.append(_fmtnum(r["gain0_err_db"]) if r else "—")
            rows.append([tech, dev] + cells)
    L += [md_table(["Tech", "Dev", "small", "medium", "large"], rows), ""]

    # opamp AC gate matrix
    L += ["### Opamp open-loop AC gate by capacity (DC-gain err dB / GBW ratio / PM err°)", ""]
    rows = []
    for tech in TECHS:
        cells = []
        for size in SIZES:
            op = data[size][tech].get("ac", {}).get(AC_CPX_SUITE)
            if not op or op.get("gate") in (None, "?"):
                cells.append("—")
            else:
                cells.append(
                    f"{op['gate']} ({_fmtnum(op.get('dc_gain_err_db'))}/"
                    f"{op.get('gbw_ratio', '—')}/{op.get('pm_err_deg', '—')})")
        rows.append([tech] + cells)
    L += [md_table(["Tech", "small", "medium", "large"], rows), ""]
    return L


def _ac_size_detail(data, size) -> list:
    """Per-size AC detail tables (device CS-amp + opamp)."""
    L = ["### AC small-signal (per tech)", ""]
    hdr = ["Tech", "Dev", "gain0_err dB", "f3db ratio", "magNRMSE%",
           "phase(inband)°", "Gate"]
    rows = []
    for tech in TECHS:
        dv = data[size][tech].get("ac", {}).get(AC_DEV_SUITE)
        for dev in ("nmos", "pmos"):
            r = dv["by_dev"].get(dev) if dv else None
            if not r:
                rows.append([tech, dev, "—", "—", "—", "—", "—"]); continue
            rows.append([tech, dev, _fmtnum(r["gain0_err_db"]),
                         _fmtnum(r["f3db_ratio"]), _fmtnum(r["mag_nrmse_pct"]),
                         _fmtnum(r["phase_deg"]), r["status"]])
    L += [md_table(hdr, rows), ""]
    # opamp
    ohdr = ["Tech", "DC-gain err dB", "GBW ratio", "PM err°", "magNRMSE%", "Gate"]
    orows = []
    for tech in TECHS:
        op = data[size][tech].get("ac", {}).get(AC_CPX_SUITE)
        if not op:
            orows.append([tech, "—", "—", "—", "—", "—"]); continue
        orows.append([tech, _fmtnum(op.get("dc_gain_err_db")),
                      op.get("gbw_ratio", "—"), op.get("pm_err_deg", "—"),
                      _fmtnum(op.get("mag_nrmse_pct")), op.get("gate", "—")])
    L += ["Opamp open-loop:", "", md_table(ohdr, orows), ""]
    return L

def report(data) -> str:
    L = ["# DirectNet (LEVEL=73) capacity benchmark — small / medium / large",
         "",
         "All checkpoints trained on ONE identical clean recipe "
         "(`--apply-filter off --swa-mode ema --seed 42`); capacity is the only "
         "variable. Datasets = full Vth + geometry grid per tech "
         "(`--variants all`, inv-trip + subvt-off overlays). Ground truth = NGSPICE "
         "BSIM-CMG (LEVEL=72), repo ngspice-45.2, CPU-pinned.",
         "",
         "Sizes: small=128x3 (~0.06M p) / medium=256x5 (~0.4M p) / large=384x6 (~0.9M p).",
         "",
         "> **V6.5 update — AC small-signal accuracy.** The DC/transient capacity "
         "study below is unchanged; V6.5 adds the first NGSPICE-gated evaluation of "
         "DirectNet **AC** (`.ac`) fidelity across all 24 checkpoints — see the "
         "[AC small-signal accuracy](#ac-small-signal-accuracy-v65) section. Headline: "
         "AC **gain** fidelity (gm/gds via autograd) is excellent everywhere "
         "(gain0 err <1.5 dB, 24/24); the **dominant cap-driven pole / bandwidth** is "
         "good but capacity/tech-variable (device CS-amp gate 13/24); the **opamp** AC "
         "inherits the DC value-surface fragility (0/12, though tsmc12-large reproduces "
         "GBW to 0.97× and phase margin to 1.3°). No retraining warranted (the gaps are "
         "value-surface- and feedforward-owned, not a charge-derivative deficiency).",
         "",
         "## Key findings",
         "",
         "1. **Circuit pass-rate rises monotonically with capacity: 6/16 -> 9/16 -> 12/16** "
         "(small -> medium -> large). The gains come from charge-transfer and timing circuits, "
         "not from device-curve accuracy (already excellent everywhere).",
         "2. **Device-level Id-Vgs / inverter accuracy is excellent at every size** (mean NRMSE "
         "<4%, most <2%; inverter VTC+transient 16/16 PASS at all sizes, all techs). Capacity "
         "barely moves it, and at the finer nodes the large net slightly *overfits* the device "
         "surface (e.g. tsmc12 DC mean NRMSE 0.36% medium -> 1.25% large). Device fidelity is NOT "
         "the bind.",
         "3. **The opamp is the hardest, value-surface-fragile gate.** Open-loop gain collapses to "
         "~0 (NRMSE ~70%) at small AND medium for all four techs, and recovers to PASS only at "
         "**large**, only for tsmc5 and tsmc12 (tsmc12 large: gain err 6.25%, locus NRMSE 1.0%). "
         "tsmc7/tsmc16 never pass on this clean single recipe -- matching project history that the "
         "*shipping* tsmc7/tsmc16 needed special recipes (pivcor / s12cor) to keep the opamp alive. "
         "Refines V6.4.8-S1: the high-gain basin is capacity- AND tech-sensitive, not a clean "
         "capacity win or loss.",
         "4. **Switched-cap needs capacity:** 0/4 (small) -> 3/4 (medium and large). tsmc5 never "
         "passes (~11-12% charge error) -- its known micro-amp-band loss-compression over-conduction "
         "(V6.4.8-S3), independent of capacity.",
         "5. **Ring-osc** passes for the higher-VDD nodes (tsmc12/16) at every size and tsmc7 at "
         "large; tsmc5 never (period err 6-13%). **SRAM butterfly** (all-lobes-positive gate) "
         "passes 4/4 at every size; force_ic is reported as an informational probe.",
         "",
         "**Bottom line:** larger capacity helps circuit-level behaviour overall (12/16 at large) "
         "but does NOT close the two recipe-sensitive gaps (tsmc7/tsmc16 opamp, tsmc5 switchcap) "
         "that the V6.4.x campaigns already attributed to value-surface / loss-compression, not "
         "capacity.",
         ""]

    # ── cross-size headline ──
    L += ["## Cross-size summary", ""]
    srows = []
    for size in SIZES:
        gates = npass = 0
        nrmses = []
        for tech in TECHS:
            d = data[size][tech]
            for suite in CPX_SUITES:
                c = d["cpx"].get(suite)
                if c:
                    gates += 1
                    npass += 1 if c["gate"] == "PASS" else 0
            for suite in DEV_SUITES:
                dv = d["dev"].get(suite)
                if dv:
                    for dev in dv["by_dev"].values():
                        nrmses.append(dev["nrmse_mean"])
        dev_nrmse = round(mean(nrmses), 2) if nrmses else None
        srows.append([size, f"{npass}/{gates}" if gates else "—",
                      dev_nrmse if dev_nrmse is not None else "—"])
    L += [md_table(["Size", "Complex gates PASS", "Device mean-NRMSE% (all sweeps)"], srows), ""]

    # ── capacity comparison: complex gate matrix (tech x circuit, S/M/L) ──
    L += ["## Capacity comparison (the headline view)", "",
          "### Complex-circuit gate verdict by capacity", ""]
    rows = []
    for tech in TECHS:
        for suite in CPX_SUITES:
            cells = []
            for size in SIZES:
                c = data[size][tech]["cpx"].get(suite)
                cells.append(c["gate"] if c else "—")
            rows.append([tech, CPX_SHORT[suite]] + cells)
    L += [md_table(["Tech", "Circuit", "small", "medium", "large"], rows), ""]

    # ── device-level mean NRMSE% by size (dynamic dev keys) ──
    L += ["### Device-level mean NRMSE% by capacity (lower = better fit)", "",
          "DC = Id-Vgs (NMOS/PMOS); INV = inverter VTC+transient (combined).", ""]
    rows = []
    for tech in TECHS:
        for suite in DEV_SUITES:
            tag = "DC" if suite.endswith("dc") else "INV"
            devkeys = []
            for size in SIZES:
                dv = data[size][tech]["dev"].get(suite)
                if dv:
                    for k in dv["by_dev"]:
                        if k not in devkeys:
                            devkeys.append(k)
            for dev in devkeys:
                cells = []
                for size in SIZES:
                    dv = data[size][tech]["dev"].get(suite)
                    v = dv["by_dev"].get(dev) if dv else None
                    cells.append(v["nrmse_mean"] if v else "—")
                rows.append([tech, f"{tag}/{dev}"] + cells)
    L += [md_table(["Tech", "Suite/Dev", "small", "medium", "large"], rows), ""]

    # ── AC small-signal accuracy (V6.5) ──
    L += _ac_section(data)

    # ── per-size detail ──
    for size in SIZES:
        L += [f"## Size = {size}", ""]
        # device table
        L += ["### Device-level parametric sweeps (per tech)", "",
              "DC = Id-Vgs over L/NFIN/VT; INV = inverter VTC+transient. "
              "Values: baseline NRMSE% / mean NRMSE% over all sweep configs; "
              "MRE% (mean); R2 (min); pass-rate.", ""]
        hdr = ["Tech", "Suite", "Dev", "base NRMSE%", "mean NRMSE%", "mean MRE%",
               "min R2", "max MaxErr", "Pass"]
        rows = []
        for tech in TECHS:
            d = data[size][tech]
            for suite in DEV_SUITES:
                tag = "DC" if suite.endswith("dc") else "INV"
                dv = d["dev"].get(suite)
                if not dv:
                    rows.append([tech, tag, "—", "—", "—", "—", "—", "—", "—"]); continue
                for dev, s in dv["by_dev"].items():
                    rows.append([tech, tag, dev, s["base_nrmse"], s["nrmse_mean"],
                                 s["mre_mean"], s["r2_min"],
                                 f"{s['maxerr_max']}{s['unit']}",
                                 f"{s['n_pass']}/{s['n']}"])
        L += [md_table(hdr, rows), ""]
        # complex table
        L += ["### Complex circuits (per tech)", "",
              "Gate verdict + headline + waveform NRMSE%/R2.", ""]
        chdr = ["Tech", "Circuit", "Gate", "Headline", "NRMSE%", "R2", "MaxErr"]
        crows = []
        for tech in TECHS:
            d = data[size][tech]
            for suite in CPX_SUITES:
                c = d["cpx"].get(suite)
                short = CPX_SHORT[suite]
                if not c:
                    crows.append([tech, short, "—", "—", "—", "—", "—"]); continue
                w = c["wave"]
                crows.append([tech, short, c["gate"], c["headline"] or "—",
                              w["nrmse"] if w else "—", w["r2"] if w else "—",
                              (f"{w['maxerr']}{w['unit']}" if w else "—")])
        L += [md_table(chdr, crows), ""]
        # AC small-signal detail (per tech)
        L += _ac_size_detail(data, size)
        # checkpoints used
        L += ["<details><summary>checkpoints resolved</summary>", ""]
        for tech in TECHS:
            L.append(f"- {tech}: `{data[size][tech]['ckpt']}`")
        L += ["", "</details>", ""]
    return "\n".join(L)

def main():
    data = load()
    (BASE / "benchmark_data.json").write_text(json.dumps(data, indent=1))
    (BASE / "REPORT.md").write_text(report(data))
    print(f"Wrote {BASE/'REPORT.md'} and benchmark_data.json")
    # quick console tally
    for size in SIZES:
        g = p = 0
        for tech in TECHS:
            for suite in CPX_SUITES:
                c = data[size][tech]["cpx"].get(suite)
                if c:
                    g += 1; p += 1 if c["gate"] == "PASS" else 0
        print(f"  {size:7s}: complex gates {p}/{g}")

if __name__ == "__main__":
    main()
