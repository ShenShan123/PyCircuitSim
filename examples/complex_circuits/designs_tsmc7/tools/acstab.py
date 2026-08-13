"""Wrap-aware AC loop-stability measurement shared by every AC bench.

The AC benches read the loop phase through ngspice's ``vp()``, which reports
the PRINCIPAL VALUE: a loop whose true phase has fallen through -180 deg
before the 0 dB crossover wraps back into (+90..+180] and becomes
indistinguishable from a lead-recovered stable loop.  ``.meas`` cannot
unwrap, so every stability deck is run with a ``wrdata`` dump of the full
sweep appended to its control block and the margin is computed here:

* the phase is unwrapped across the sweep from the lowest frequency up;
* the low-frequency phase anchors the reference, snapped to the nearest
  multiple of 180 deg -- where an ideal loop's phase sits at DC under either
  sign convention (these benches measure -T, so the ideal is +/-180);
* GBW is the unity-gain (0 dB) crossover frequency of the measured sweep --
  the ONE definition every artifact uses -- interpolated in log-frequency;
* the true margin is 180 deg minus the unwrapped deviation from the
  reference at crossover.  A lead-recovered loop may legitimately report up
  to ~180 deg.  A loop whose unwrapped deviation reaches 180 deg anywhere at
  or below crossover has crossed the instability point while |T| >= 1 and
  reports a NEGATIVE margin, wrap or no wrap.

The raw principal-value phase at crossover is stored next to the true margin
(``ph_xover*``) so the tables stay auditable against the ``.meas`` output.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from meas import run_deck

# deck file name -> (output vector, GBW metric key, stability key suffix).
# The runner appends the wrdata dump for these decks and computes the
# unwrapped margin; every other deck runs untouched.
STABILITY_DECKS: Dict[str, Tuple[str, str, str]] = {
    "tb_gain.cir": ("opout", "gain_bandwidth_product", ""),      # amplifier
    "tb_ac.cir": ("vout", "gain_bandwidth_product", ""),         # SMCNR amp
    "tb_loop_max.cir": ("vo", "gbw_max", "_max"),                # LDO loop
    "tb_loop_min.cir": ("vo", "gbw_min", "_min"),
}


def read_ac_sweep(path: Path) -> Tuple[List[float], List[float], List[float]]:
    """Parse a ``wrdata f vdb(x) vp(x)`` dump: (freq, gain_db, phase_deg)."""
    freqs: List[float] = []
    gains: List[float] = []
    phases: List[float] = []
    if not path.exists():
        return freqs, gains, phases
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            f, g, p = float(parts[0]), float(parts[1]), float(parts[3])
        except ValueError:
            continue
        if not (math.isfinite(f) and math.isfinite(g) and math.isfinite(p)):
            continue
        freqs.append(f)
        gains.append(g)
        phases.append(math.degrees(p))     # vp() reports radians
    return freqs, gains, phases


def unwrap_deg(phases: List[float]) -> List[float]:
    """Unwrap principal-value phases (deg) into a continuous sequence."""
    if not phases:
        return []
    out = [phases[0]]
    for p in phases[1:]:
        d = p - out[-1]
        d -= 360.0 * round(d / 360.0)
        out.append(out[-1] + d)
    return out


def true_margin(freqs: List[float], gains_db: List[float],
                phases_deg: List[float]) -> Dict[str, Optional[float]]:
    """Unity-gain crossover and wrap-aware stability margin of one sweep.

    Returns ``gbw`` (Hz), ``pm_true`` (deg, negative when the unwrapped phase
    has crossed the instability point at or below crossover), ``ph_xover``
    (raw principal-value phase at crossover, deg), ``phase_ref`` (the
    snapped low-frequency anchor, deg) and ``phase_maxdev`` (largest
    unwrapped deviation from the anchor seen at or below crossover, deg).
    All ``None`` when the gain never crosses 0 dB inside the sweep.
    """
    none: Dict[str, Optional[float]] = {
        "gbw": None, "pm_true": None, "ph_xover": None,
        "phase_ref": None, "phase_maxdev": None,
    }
    if len(freqs) < 2:
        return none
    unwrapped = unwrap_deg(phases_deg)
    idx = next((i for i in range(len(gains_db) - 1)
                if gains_db[i] >= 0.0 > gains_db[i + 1]), None)
    if idx is None:
        return none
    # Interpolate the crossover in log-frequency; phase linearly alongside.
    t = gains_db[idx] / (gains_db[idx] - gains_db[idx + 1])
    lf = (math.log10(freqs[idx])
          + t * (math.log10(freqs[idx + 1]) - math.log10(freqs[idx])))
    ph_c = unwrapped[idx] + t * (unwrapped[idx + 1] - unwrapped[idx])
    ref = 180.0 * round(unwrapped[0] / 180.0)
    maxdev = max([abs(u - ref) for u in unwrapped[:idx + 1]]
                 + [abs(ph_c - ref)])
    dev_c = abs(ph_c - ref)
    # 180 deg of deviation anywhere with |T| >= 1 is the instability point;
    # past it the margin is negative no matter where the phase sits at
    # crossover (a dip through -180 that recovers is still a Nyquist
    # crossing, not a margin).
    pm = 180.0 - dev_c if maxdev < 180.0 else 180.0 - maxdev
    principal = ((ph_c + 180.0) % 360.0) - 180.0
    return {"gbw": 10.0 ** lf, "pm_true": pm, "ph_xover": principal,
            "phase_ref": ref, "phase_maxdev": maxdev}


def run_ac_stability(deck: Path, control: str, work: Path, tag: str,
                     timeout: float = 300.0) -> Dict[str, Optional[float]]:
    """Run a stability deck, returning ``.meas`` results plus the true margin.

    The GBW key is OVERWRITTEN with the crossover of the dumped sweep so both
    artifact paths share one definition; if the dump cannot be parsed the
    ``.meas`` value is kept and the ``pm_true*`` keys report ``None`` -- a
    missing true margin fails the gate rather than falling back to the
    wrapped reading.
    """
    vec, gbw_key, suffix = STABILITY_DECKS[deck.name]
    sweep = (work / f"{tag}.sweep").resolve()
    sweep.unlink(missing_ok=True)
    ctl = f"{control}\nwrdata {sweep} vdb({vec}) vp({vec})"
    metrics = run_deck(deck, ctl, work, tag, timeout=timeout)
    stab = true_margin(*read_ac_sweep(sweep))
    if stab["gbw"] is not None or sweep.exists():
        metrics[gbw_key] = stab["gbw"]
    for key in ("pm_true", "ph_xover", "phase_ref", "phase_maxdev"):
        metrics[f"{key}{suffix}"] = stab[key]
    return metrics


def run_deck_auto(deck: Path, control: str, work: Path, tag: str,
                  timeout: float = 300.0) -> Dict[str, Optional[float]]:
    """``run_deck``, upgraded to the stability runner for the AC loop decks."""
    if deck.name in STABILITY_DECKS:
        return run_ac_stability(deck, control, work, tag, timeout=timeout)
    return run_deck(deck, control, work, tag, timeout=timeout)
