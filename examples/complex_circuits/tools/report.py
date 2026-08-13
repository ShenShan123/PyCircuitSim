"""Build RESULTS.md from the per-design result.json files this tree produces."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pycmg_lib import ROOT                              # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pycmg_lib import TECH, VDD  # noqa: E402


def eng(v: Optional[float], unit: str = "", digits: int = 3) -> str:
    """Engineering notation, or 'n/a' when the measurement did not resolve."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    a = abs(v)
    for scale, suf in ((1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""),
                       (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"),
                       (1e-12, "p"), (1e-15, "f")):
        if a >= scale or scale == 1e-15:
            if a == 0:
                return f"0 {unit}".strip()
            return f"{v / scale:.{digits}g} {suf}{unit}".strip()
    return f"{v:.3g} {unit}".strip()


def num(v: Optional[float], digits: int = 1) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:.{digits}f}"


def load(category: str) -> List[Dict]:
    out = []
    cdir = ROOT / category
    if not cdir.is_dir():
        return out
    for d in sorted(cdir.iterdir()):
        f = d / "result.json"
        if f.exists():
            try:
                out.append(json.loads(f.read_text()))
            except json.JSONDecodeError:
                pass
    return out


def has_errors(result: Dict) -> bool:
    """Return whether a result contains a simulator/evaluator failure."""
    return bool(result.get("error") or result.get("errors"))


def verdict(result: Dict) -> str:
    """Format passed gates and flag analyses that did not complete."""
    passed = result.get("pass") or {}
    suffix = " !" if has_errors(result) else ""
    return f"{sum(bool(value) for value in passed.values())}/{len(passed) or 1}{suffix}"


def amp_table(rows: List[Dict]) -> str:
    # PM is the true (unwrapped) stability margin from tools/acstab.py; the
    # raw principal-value crossover phase stays in result.json as
    # ``phase_in_deg`` / ``ph_xover`` for auditing.
    head = ("| design | A_v (dB) | GBW | PM (deg) | P | CMRR (dB) "
            "| PSRR+ (dB) | PSRR- (dB) | Vos | SR+ (V/us) | SR- (V/us) | pass |\n"
            "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|\n")
    body = ""
    for r in rows:
        m, p = r.get("metrics", {}), r.get("pass", {})
        body += (f"| {r['design']} | {num(m.get('dcgain'))} "
                 f"| {eng(m.get('gain_bandwidth_product'), 'Hz')} "
                 f"| {num(m.get('pm_true'))} "
                 f"| {eng(m.get('power'), 'W')} "
                 f"| {num(m.get('cmrrdc'))} | {num(m.get('dcpsrp'))} "
                 f"| {num(m.get('dcpsrn'))} | {eng(m.get('vos25'), 'V')} "
                 f"| {num(m.get('sr_rise'), 3)} | {num(m.get('sr_fall'), 3)} "
                 f"| {verdict(r)} |\n")
    return head + body


def ldo_table(rows: List[Dict]) -> str:
    head = ("| design | Vout max load | Vout min load | line reg | load reg "
            "| P max load | A_v (dB) | GBW | PM (deg) | PSRR (dB) "
            "| undershoot | pass |\n"
            "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|\n")
    body = ""
    for r in rows:
        m, p = r.get("metrics", {}), r.get("pass", {})
        body += (f"| {r['design']} | {eng(m.get('vout_max'), 'V')} "
                 f"| {eng(m.get('vout_min'), 'V')} "
                 f"| {num(m.get('lnrmax'), 4)} | {num(m.get('lr'), 4)} "
                 f"| {eng(m.get('power_max'), 'W')} "
                 f"| {num(m.get('dcgain_max'))} "
                 f"| {eng(m.get('gbw_max'), 'Hz')} "
                 f"| {num(m.get('pm_true_max'))} | {num(m.get('psrr_max'))} "
                 f"| {eng(m.get('undershoot'), 'V')} "
                 f"| {verdict(r)} |\n")
    return head + body


def sfe_table(rows: List[Dict]) -> str:
    # SMCNR_SE_2st_AMP is an amplifier shipped in this category (see
    # tools/sfe_amp.py); it reports AC-bench metrics rather than PTAT ones.
    amps = [r for r in rows if "dcgain" in (r.get("metrics") or {})]
    sensors = [r for r in rows if r not in amps]
    head = ("| design | Vout @25 C | sensitivity 25-75 C | spread -20..120 C "
            "| Vout @0 C | Vout @100 C | pass |\n"
            "|---|--:|--:|--:|--:|--:|:--:|\n")
    body = ""
    for r in sensors:
        m, p = r.get("metrics", {}), r.get("pass", {})
        body += (f"| {r['design']} | {eng(m.get('vout25'), 'V')} "
                 f"| {eng(m.get('lsb_25_75c'), 'V/C')} "
                 f"| {eng(m.get('ppval'), 'V')} "
                 f"| {eng(m.get('vout0'), 'V')} "
                 f"| {eng(m.get('vout100'), 'V')} "
                 f"| {verdict(r)} |\n")
    out = head + body
    if amps:
        out += ("\nThe amplifier shipped in this category runs its own AC "
                "bench (gain, GBW, true phase margin, power):\n\n"
                "| design | A_v (dB) | GBW | PM (deg) | P | pass |\n"
                "|---|--:|--:|--:|--:|:--:|\n")
        for r in amps:
            m = r.get("metrics", {})
            out += (f"| {r['design']} | {num(m.get('dcgain'))} "
                    f"| {eng(m.get('gain_bandwidth_product'), 'Hz')} "
                    f"| {num(m.get('pm_true'))} "
                    f"| {eng(m.get('power'), 'W')} "
                    f"| {verdict(r)} |\n")
    return out


def vref_table(rows: List[Dict]) -> str:
    head = ("| design | output | Vref @25 C | spread | TC (ppm/C) | pass |\n"
            "|---|---|--:|--:|--:|:--:|\n")
    body = ""
    for r in rows:
        m, p = r.get("metrics", {}), r.get("pass", {})
        outs = r.get("outputs", [])
        row_verdict = verdict(r)
        for i, o in enumerate(outs):
            body += (f"| {r['design'] if i == 0 else ''} | {o} "
                     f"| {eng(m.get(f'{o}_at25'), 'V')} "
                     f"| {eng(m.get(f'{o}_pp'), 'V')} "
                     f"| {num(m.get(f'{o}_tc'), 0)} "
                     f"| {row_verdict if i == 0 else ''} |\n")
    return head + body


def cp_table(rows: List[Dict]) -> str:
    head = "| measurement | value |\n|---|--:|\n"
    body = ""
    for r in rows:
        m = r.get("metrics", {})
        body += (f"| source (up) current, average | "
                 f"{eng(m.get('up_iavg'), 'A')} |\n"
                 f"| sink (down) current, average | "
                 f"{eng(m.get('lo_iavg'), 'A')} |\n"
                 f"| up/down mismatch | "
                 f"{num(r.get('mismatch_pct'), 2)} % |\n"
                 f"| up current range | {eng(m.get('up_imin'), 'A')} ... "
                 f"{eng(m.get('up_imax'), 'A')} |\n"
                 f"| down current range | {eng(m.get('lo_imin'), 'A')} ... "
                 f"{eng(m.get('lo_imax'), 'A')} |\n")
    return head + body


TABLES = {
    "amplifier": amp_table,
    "ldo": ldo_table,
    "sensing_front_end": sfe_table,
    "voltage_reference": vref_table,
    "charge_pump": cp_table,
}


def coverage(all_rows: Dict[str, List[Dict]]) -> str:
    head = "| category | designs | fully passing | partial |\n|---|--:|--:|--:|\n"
    body = ""
    tot = full = part = 0
    for cat, rows in all_rows.items():
        # An empty verdict dict is not a pass -- all({}) is True.
        f = sum(1 for r in rows
                if r.get("pass") and all(r["pass"].values())
                and len(r["pass"]) > 0 and not has_errors(r))
        body += f"| {cat} | {len(rows)} | {f} | {len(rows) - f} |\n"
        tot += len(rows)
        full += f
        part += len(rows) - f
    body += f"| **total** | **{tot}** | **{full}** | **{part}** |\n"
    return head + body


NOTES = f"""

## Reading these numbers

**These are not comparable to the sky130 port.** Different process, different
supply ({VDD:g} V against 1.8 V), and a different design point: the topology and
mirror ratios carry over, the sizing does not. `tools/verify_port.py` is what
checks the port itself — same devices, nodes, channel types and mirror ratios as
the sky130 source.

**Targets.** Amplifier: gain >= 60 dB, GBW >= 300 kHz into 500 pF, true
(unwrapped) phase margin >= 45 deg, power <= 2 mW, |Vos| <= 20 mV, CMRR and
PSRR+ <= -30 dB. Sensor: monotonic rise over every solved sweep point in
0-100 C, 0.3-6 mV/C, staircase-free (local slope >= 0.1 mV/C over 25-75 C, no
single sweep step carrying more than half the 25-75 C rise), output inside the
rail. Reference: |TC| <= 500 ppm/C. LDO: regulates within 50 mV of nominal at
both load extremes, loop gain >= 40 dB, true PM >= 45 deg, GBW >= 100 kHz,
power <= 80 mW. Charge pump: both currents flow, matched within 5 %, in the
2-200 uA range.

**Phase margin is wrap-aware.** The AC benches read phase through ngspice's
`vp()`, which reports the principal value: a loop whose true phase fell
through -180 deg before crossover wraps back into (+90..+180] and is
indistinguishable from a lead-recovered stable loop. Every stability deck is
therefore run with a full-sweep dump; `tools/acstab.py` unwraps the phase from
the lowest frequency up and reports the true margin at the 0 dB crossover
(negative when the unwrapped phase crossed the instability point while
|T| >= 1). A lead-recovered loop may legitimately report a PM above 90 deg,
up to ~180. The raw principal-value crossover phase is kept in result.json
(`phase_in_deg`, `ph_xover*`, `pm_max`/`pm_min`) for auditing; GBW everywhere
is the unity-gain (0 dB) crossover frequency of the measured sweep.
"""

TECH_NOTES = {
    "TSMC5": """
**TSMC5 specifics.**  0.65 V rail, L binned at 6-135 nm, NFIN <= 12 — the
hardest carry-over target.  The error amplifiers of `Basic_LDO` and
`ldo_folded_cascode` cannot bias with svt devices at this rail; the re-tune
lands both on lower-Vt flavors (the polish seeds every flavor the PDK ships).
`ptat_3`'s temperature swing exceeds the scaled window at the shipped flavors;
mixing flavors across the stack (elvt leakage pull-ups over ulvt diode
pull-downs) compresses the swing under the bar without giving up sensitivity.
`ldo_1` — the design whose missing compensation fails the phase-margin gate on
sky130, TSMC16 and every other node here — *passes* on TSMC5: the rail and
geometry shift moves the loop poles apart on their own.  `ldo_2` regulates
with large margins (PM ~175 deg, GBW 8 MHz) but its replica-feedback loop
tops out at 37.8 dB against the 40 dB gate — intrinsic gain at 0.65 V is
the limit.
""",
    "TSMC6": """
**TSMC6 specifics.**  0.75 V rail, L binned at 8-240 nm.  `Qu_LEC`'s
common-mode deck needs the 1 fF `cshunt` operating-point aid (measured CMRR
is unchanged where the deck converges without it).  The two partials are the
cross-node structural ones: `ldo_1` (no compensation element; fails the same
gate on sky130 and TSMC16) and `three_output_vref` (vref1 rides on vref2
plus a stack drop; partial on every node including TSMC16).
""",
    "TSMC7": """
**TSMC7 specifics.**  0.75 V rail, L binned at 8-240 nm.  Identical partials
to TSMC6 — `ldo_1` and `three_output_vref`, both structural across nodes —
and the same `cshunt` aid on `Qu_LEC`'s common-mode deck.  The TSMC6 and
TSMC7 PDK views are close relatives and deliver near-identical numbers at
the shared design points.
""",
    "TSMC12": """
**TSMC12 specifics.**  Same 0.8 V rail and geometry envelope as TSMC16, so
the ported sizes carry with the least disturbance.  The two partials are the
cross-node structural ones: `ldo_1` (no compensation element) and
`three_output_vref` (output-stack ceiling), the same pair TSMC16 reports.
""",
}

TSMC16_NOTES = """

**PM is the true, wrap-aware margin** (`pm_true` in result.json, from the
unwrapped AC sweep -- see "Phase margin is wrap-aware" above); AnalogGym's raw
`phase_in_deg` crossover reading is kept alongside it for auditing.

**Negative and n/a slew rates.** AnalogGym measures the first crossing of the
step midpoint. On an output that rings or settles inside the edge, that crossing
happens before the step ends and the computed interval goes negative; where no
crossing falls in the window the measurement reports nothing. Both appear in the
sky130 port as well, for the same reason. The rate itself is fine — this is the
measurement definition, not the circuit.

**The polish pass.** Ten designs landed partial on the first sizing pass; a
second pass (`tools/polish_*.py`) re-searched them warm-started from the
shipped design vector, with the knobs the first pass lacked. What each fix
actually needed is instructive:

* **Ramos_PFC_Pin_3** (was -3.7 deg at 0 dB): the reduced 10-knob amplifier
  search files its `gmf1` device under the "mid" group, but in this topology
  gmf1 *is* the output pull-up — the shared multipliers drove the push-pull
  output 1000:1 apart, and both compensation capacitors sat welded to one knob
  at a frozen 2:1 ratio.  A full per-role search (every role and passive on
  its own axis, searched on the dec-20 sweep the report uses rather than the
  dec-8 one, whose phase-at-0dB reading can be off by the whole margin when
  the phase wraps near the crossing) lands 7/7 at 124 dB / 176 deg / 34 uW.
* **Basic_LDO** (was 298 mV at max load): drive, then gain.  Its level
  shifter cannot push the pass gate under ~0.40 V, and the shipped-size pass
  device sources ~12 mA there against a 55 mA load (measured — `work/iv`), so
  the loop rails and the small-signal numbers collapse with it.  A 3x-larger
  pass device seed plus the new bias-current knob recovers regulation, and a
  wider pass fin count trades 16 deg of spare phase margin for the last
  1.4 dB of loop gain.  5/5.
* **ldo_2** (was 28 deg PM at max load) reaches 5/5 once the compensation
  resistor — fixed at 100 k in the first pass — becomes a knob.
* **ldo_folded_cascode** (was 39.96 dB against a 40 dB bar, on a ridge where
  every coarse move loses more than it gains): a stronger input pair
  (m 96 -> 120) buys the gain and `vref` re-centres the output.  5/5.
* **Sensing front ends** (three partial): AnalogGym's sensor templates ship
  every device at the same placeholder W/L, and grouping "matched" devices by
  source geometry welded each sensor into one three-knob group — identical
  stacked devices develop almost no delta-Vgs, which is precisely what the
  partials measured.  Sized per device with per-device Vt flavors (a Vt
  difference between stacked devices is a designable level/slope term),
  all twelve now pass 3/3.
* **dual_output_subthreshold_vref** (was 518 ppm/C and 75 mV): per-group Vt
  mixing rebalances the cancellation to 3/3 (335 / 72 ppm/C, both outputs in
  range).

**What is still partial, and why.**

* **ldo_1** (4/5, phase margin 4.9 deg at min load): the shipped netlist is
  three high-impedance gain stages with *no* compensation capacitor, into
  this bench's 100 nF output capacitor.  Sizing moves the poles and the
  crossing together — strengthening the pass-gate driver raises the crossing
  as fast as it raises the pole (measured 12 deg at best, with PM at max load
  collapsing) — and three independent searches (48 seeds, divider-impedance
  and bias-current knobs included) end in the same place.  Stabilising it at
  both load extremes needs a compensation element the topology does not have;
  the sky130 port fails the same criterion on the same design.
* **three_output_vref** (2/3): every temperature coefficient now passes
  (349 / 122 / 124 ppm/C after the per-device pass), but vref1 sits at
  790 mV on a 0.8 V rail against the 700 mV ceiling.  That level is
  structural: vref1 rides on top of vref2 (vref1 = vref2 + a three-segment
  gate-tied stack drop), and the one flavor that pulls the drop under the
  ceiling (ulvt, 686 mV) destroys the sub-threshold cancellation that the TC
  numbers live on — re-searching from that corner does not recover it.

**LDO PSRR and line-regulation readings.** `ldo_folded_cascode` reports
*positive* PSRR because the bench supplies its cascode bias from
ground-referenced sources that cannot track supply ripple — the sky130 port
shows the same effect (-8.6 dB, the worst of its five).  Where the table shows
n/a, that PSRR deck's operating point did not converge.  The line-regulation
sweep runs the supply +-10 %, and its bottom end (0.72 V) leaves 120 mV over
the 0.6 V output — designs whose pass device needs more dropout than that
(`Basic_LDO`, `ldo_folded_cascode`) fall out of regulation inside the sweep
window and report meaningless line-reg figures.  Neither PSRR nor line
regulation is one of the five pass criteria.
"""


def main() -> None:
    rows = {cat: load(cat) for cat in TABLES}
    parts = [f"# Results -- AnalogGym on {TECH} (BSIM-CMG via PyCMG)\n",
             "Every number below comes from running the decks in this tree "
             "under ngspice-45.2 with the\n"
             "`bsimcmg.osdi` binary from "
             "`PyCircuitSim/external_compact_models/PyCMG`. Nothing is copied "
             "from\nthe AnalogGym paper or from the sky130 port.\n",
             "\n## Coverage\n\n" + coverage(rows)]
    titles = {"amplifier": "Amplifier", "ldo": "Low Dropout Regulator",
              "sensing_front_end": "Sensing Front End",
              "voltage_reference": "Voltage Reference",
              "charge_pump": "Charge Pump"}
    for cat, fn in TABLES.items():
        if rows[cat]:
            parts.append(f"\n## {titles[cat]}\n\n" + fn(rows[cat]))
    parts.append(f"""

## Audit criteria

`!` means that one or more requested analyses did not complete, so the design
is partial even if all available numeric gates pass. Amplifiers include gain,
GBW, true (unwrapped) phase margin, power, offset, CMRR, both PSRR polarities,
temperature drift, and both slew directions. LDOs include regulation, loop
gain/GBW/true phase margin, power, line and load regulation, both-load PSRR,
and load-step excursion. Sensors require in-rail output that rises
monotonically over every solved sweep point, 0.3-6 mV/C sensitivity, and a
staircase-free characteristic (local slope and single-step share gates in
tools/sfe.py); references require every output in range with <=500 ppm/C; the
charge pump requires both current directions, 2-200 uA magnitude, and <=5%
mismatch. Simulations use the {VDD:g} V core rail for {TECH}.
""")
    out = ROOT / "RESULTS.md"
    out.write_text("\n".join(parts))
    print(f"-> {out}")
    print(coverage(rows))


if __name__ == "__main__":
    main()
