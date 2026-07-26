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
# Canonical capacity order; the report includes only the tiers that have
# result dirs on disk, so adding/removing a tier (e.g. V6.5.1 `xl`) needs no
# further edits — every table header derives from SIZES.
# audit B4: NO `or _SIZE_ORDER[:3]` fallback. On an empty tree that fabricated
# three tier names and produced a full report of 0/0 ratios; main() now refuses
# to collect instead.
_SIZE_ORDER = ["small", "medium", "large", "xl"]
SIZES = [s for s in _SIZE_ORDER if (BASE / s).is_dir()]
TECHS = ["tsmc5", "tsmc6", "tsmc7", "tsmc12", "tsmc16"]
DEV_SUITES = ["verify_nn_multi_tech_dc", "verify_nn_multi_tech_tran"]
CPX_SUITES = ["verify_complex_ring_osc", "verify_complex_opamp",
              "verify_complex_sram_snm", "verify_complex_switchcap"]
# AC (small-signal) suites — V6.5. Device CS-amp (per-checkpoint) + opamp
# open-loop. RO/SRAM excluded (no stable amplifying OP → no .ac ground truth).
AC_DEV_SUITE = "verify_nn_ac"
AC_CPX_SUITE = "verify_complex_opamp_ac"
AC_SUITES = [AC_DEV_SUITE, AC_CPX_SUITE]

# nan|n/a included so diverged rows are parsed (and surface as nan) instead
# of being silently dropped from the aggregates (device_retest_collect parity)
NUM = r"(?:[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|nan|n/a)"

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


def _missing_cpx(reason: str) -> dict:
    """audit B4 — placeholder for a complex cell that reached no verdict.

    A cell that is simply absent from `data` drops out of BOTH sides of every
    pass ratio, so a half-run tier reports e.g. 15/15 instead of 15/16. Storing
    an explicit non-PASS entry keeps it in the denominator.
    """
    return {"wave": None, "gate": "MISSING", "headline": reason, "raw": []}


def load() -> dict:
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
                if not p.exists():
                    d["cpx"][suite] = _missing_cpx("no log")
                    continue
                t = p.read_text(errors="replace")
                d["ckpt"] = d["ckpt"] or resolver_line(t)
                c = parse_complex_log(suite, t)
                # audit B4: a log with no parseable verdict AND no completion
                # marker is a killed worker or a `no-ckpt` stub — absence, not a
                # result. A log that DID reach a verdict is kept verbatim
                # (marker or not), so no cell that already reports a number moves.
                if c["gate"] == "?" and "===BENCH_DONE rc=" not in t:
                    d["cpx"][suite] = _missing_cpx("no verdict in log")
                else:
                    d["cpx"][suite] = c
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
        # audit B4 — the narrative below (through the V6.5.1 paragraph) is a
        # frozen string literal; only the tables that follow it come from the
        # logs this invocation parsed.
        "> **The commentary in this section is the frozen V6.5 write-up — historical, "
        "NOT re-derived from this run.** The tables after it are generated from this "
        "invocation's logs.",
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
        "**V6.5.1 — XL adds 8 checkpoints (32 total) and does NOT change the AC story.** "
        "Device CS-amp gate holds at 4/12 for XL (17/32 overall), opamp stays 0/4 (0/16 "
        "overall), and device mean gain0-err drifts marginally worse (0.86→0.91 dB L→XL) — "
        "consistent with the XL device-surface over-fit seen in DC. AC fidelity is "
        "capacity-saturated by `large`.",
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
    L += [md_table(["Tech", "Dev"] + SIZES, rows), ""]

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
    L += [md_table(["Tech", "Dev"] + SIZES, rows), ""]

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
    L += [md_table(["Tech"] + SIZES, rows), ""]
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
    L = ["# DirectNet (LEVEL=73) capacity benchmark — small / medium / large / xl",
         "",
         "All checkpoints trained on ONE identical clean recipe "
         "(`--apply-filter off --swa-mode ema --seed 42`); capacity is the only "
         "variable. Datasets = full Vth + geometry grid per tech "
         "(`--variants all`, inv-trip + subvt-off overlays). Ground truth = NGSPICE "
         "BSIM-CMG (LEVEL=72), repo ngspice-45.2, CPU-pinned.",
         "",
         "Sizes: small=128x3 (~0.06M p) / medium=256x5 (~0.4M p) / large=384x6 (~0.9M p) / "
         "**xl=512x8 (~2.13M p)**.",
         "",
         # audit B4 — everything down to "## Cross-size summary" is a fixed
         # string literal from the V6.5.1 write-up. It was previously emitted
         # unlabelled, so a report generated from *any* tree asserted V6.5.1's
         # measurements as if this run had reproduced them.
         "## Frozen V6.5.1 narrative (historical — NOT re-derived from this run)",
         "",
         "> The prose and the pass-rate counts quoted in this section were written for "
         "the V6.5.1 capacity study and are reproduced verbatim as context. Every "
         "**table** below this section is generated from the logs this invocation "
         "parsed; every **number quoted in prose here** is not.",
         "",
         "> **V6.5.1 update — XL capacity tier + µA-band loss lever (KILLED).** This revision "
         "adds the **XL** tier (512×8, 2.13M p) on the identical clean recipe and a tested-"
         "but-reverted accuracy lever. **Headline: the capacity curve PEAKS at `large` and "
         "DECLINES at XL — 6 → 9 → 12 → 9 / 16.** XL fits the device surface ~10× tighter "
         "(val loss 2e-4 vs medium's ~2e-3) yet has the *worst* off-nominal parametric NRMSE "
         "(2.17% mean, the highest of any tier) and *loses every value-surface-fragile gate "
         "`large` had won* (tsmc5 opamp, tsmc12 opamp, tsmc7 ring-osc all PASS→FAIL). This is "
         "the cleanest possible confirmation of V6.4.8-S1: **capacity is not the bind; beyond "
         "`large` it over-fits the value-surface and collapses the high-gain basins.** The "
         "**µA-band loss de-compression** lever (the V6.4.8 roadmap's named next step — retune "
         "the default-off `SubthresholdIdLoss` to the µA band, `s2=1e-7 upper=3e-5`) was A/B-"
         "tested on tsmc5 switchcap and **KILLED**: charge_err 11.84% → 12.05%/11.69% "
         "(unchanged) with a slight DC regress. It *refutes* the 'loss-compression-owned' "
         "hypothesis — the tsmc5 switchcap over-charge survives direct µA-band loss de-"
         "compression (and survived S3's EKV prior), so it is **charge/transient (sample-and-"
         "hold) owned, not DC-current-band owned**. The V6.5 AC study below is unchanged "
         "(XL does not move AC: opamp 0/4, device CS-amp 4/12, gain0-err marginally worse).",
         "",
         "### Key findings (V6.5.1, frozen)",
         "",
         "1. **Circuit pass-rate is NON-monotonic in capacity: 6/16 → 9/16 → 12/16 → 9/16** "
         "(small → medium → large → **xl**). It rises through `large`, then **regresses at XL**. "
         "The S→L gains come from charge-transfer and timing circuits; the L→XL loss comes from "
         "value-surface over-fit collapsing the opamp / high-VDD-RO high-gain basins.",
         "2. **Device-level Id-Vgs / inverter accuracy is excellent at every size, but XL over-fits.** "
         "Mean parametric NRMSE 1.63 → 1.29 → 1.7 → 2.17% (S→M→L→XL): it is *best at medium* and "
         "*worst at XL*, even though XL's validation loss is ~10× lower than medium's. XL fits the "
         "training distribution tightest but generalizes worst to off-nominal geometry/VT sweeps "
         "(e.g. tsmc7 DC/pmos 0.51% medium → 2.65% XL; tsmc12 DC/pmos 0.76% → 3.57%). Inverter "
         "VTC+transient stays 16/16 PASS at all four sizes. **Device fidelity is NOT the bind; "
         "more capacity past `large` hurts generalization.**",
         "3. **The opamp is the hardest, value-surface-fragile gate.** Open-loop gain collapses to "
         "~0 (NRMSE ~70%) at small AND medium for all four techs, recovers to PASS only at **large** "
         "and only for tsmc5/tsmc12 (tsmc12 large: gain err 6.25%, locus NRMSE 1.0%), then **collapses "
         "again at XL** (tsmc5/tsmc12 opamp PASS→FAIL). tsmc7/tsmc16 never pass on this clean recipe "
         "(the *shipping* tsmc7/tsmc16 need the pivcor / s12cor recipes). The high-gain basin is "
         "capacity- AND tech-sensitive with a sweet spot at `large` — bigger is not better.",
         "4. **Switched-cap needs capacity up to a point:** 0/4 (small) → 3/4 (medium, large, **xl**). "
         "tsmc5 never passes (~11-12% charge error) at ANY size — and V6.5.1 proves it is **not** "
         "µA-band-loss-compression-owned (the lever moved it <0.2%): it is sample-and-hold "
         "charge/transient over-charge, independent of both capacity and µA-band DC loss weighting.",
         "5. **Ring-osc** passes for the higher-VDD nodes (tsmc12/16) at every size; tsmc7 passes "
         "ONLY at `large` (PASS at L, FAIL at S/M/**XL** — another L→XL over-fit regression); tsmc5 "
         "never (period err 6-14%). **SRAM butterfly** (all-lobes-positive gate) passes 4/4 at every "
         "size; force_ic is an informational probe.",
         "",
         "**Bottom line:** capacity helps circuit-level behaviour up to **`large` (12/16), which is "
         "the sweet spot**; **XL over-fits and regresses to 9/16**. Neither more capacity (XL) nor "
         "µA-band loss de-compression closes the recipe-sensitive gaps (tsmc7/tsmc16 opamp, tsmc5 "
         "switchcap) — the V6.4.x attribution to value-surface / charge-transient ownership (not "
         "capacity, not µA-band DC loss) is confirmed and sharpened. `large` remains the production "
         "capacity; XL is retained as the empirical over-fit boundary.",
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
    L += [md_table(["Tech", "Circuit"] + SIZES, rows), ""]

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
    L += [md_table(["Tech", "Suite/Dev"] + SIZES, rows), ""]

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

def main() -> int:
    # audit B4: refuse to manufacture a report out of an empty tree. Without
    # this the fabricated SIZES fallback wrote a 17 kB REPORT.md of 0/0 ratios
    # (whose frozen prose still asserted measurements) and exited 0.
    if not SIZES:
        sys.exit(f"[collect] no size dirs under {BASE} — nothing to collect")
    data = load()
    missing = [(size, tech, suite)
               for size in SIZES for tech in TECHS for suite in CPX_SUITES
               if data[size][tech]["cpx"][suite]["gate"] == "MISSING"]
    expected = len(SIZES) * len(TECHS) * len(CPX_SUITES)

    text = report(data)
    if missing:
        # Banner first, so no reader reaches a ratio before learning it is partial
        # (scripts/a3_regate_collect.py uses the same "(incomplete)" discipline).
        text = ("> **INCOMPLETE: {} of {} complex cells have no log — every ratio "
                "below is partial.**\n>\n> Missing: {}\n\n{}").format(
                    len(missing), expected,
                    "; ".join(f"`{s}/{t}/{CPX_SHORT[q]}`" for s, t, q in missing),
                    text)

    (BASE / "benchmark_data.json").write_text(json.dumps(data, indent=1))
    (BASE / "REPORT.md").write_text(text)
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
    if missing:
        print(f"[collect] INCOMPLETE: {len(missing)}/{expected} complex cells have "
              f"no log — REPORT.md ratios are partial", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
