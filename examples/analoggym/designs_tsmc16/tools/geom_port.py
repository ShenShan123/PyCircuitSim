"""Port netlists that carry explicit W/L geometry (LDO, sensing front end, refs).

The amplifier netlists name their geometry through AnalogGym design variables;
everything else states W and L directly, in the units of a process this repo
does not have.  This module reads those forms and re-expresses them on TSMC16.

Geometry rule
-------------
* **L is preserved in absolute terms**, clamped into the TSMC16 binned range
  [16, 240] nm.  These circuits -- PTAT sensors, sub-threshold references, LDO
  error amplifiers -- work because their devices sit in weak inversion with a
  clean sub-threshold slope, and that is a long-channel property.  Scaling L
  down by the node ratio would hand them DIBL instead.
* **W follows L** so W/L, and therefore every current-density ratio the circuit
  depends on, is unchanged: ``W16 = W * (L16 / L)``.
* One NFIN is chosen per design -- the fin count that makes the *narrowest*
  device come out at m=1 -- and every other device expresses its width through
  ``m`` alone.  Ratios between devices are then exact integers rather than two
  independently rounded fin counts, and matched pairs stay matched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pycmg_lib import W_PER_FIN, snap_l, snap_nfin, L_MIN, L_MAX

# Foreign device name -> channel type.  Everything AnalogGym's ngspice ports
# bind to a sky130 FET, plus the raw Spectre/HSPICE names they came from.
NMOS_MODELS = {
    "nmos", "nch_mac", "nmos_1p8", "n11ll_ckt", "nod33ll_ckt", "nmos_6p0",
    "sky130_fd_pr__nfet_01v8", "sky130_fd_pr__nfet_01v8_lvt",
    "sky130_fd_pr__nfet_g5v0d10v5",
}
PMOS_MODELS = {
    "pmos", "pch_mac", "pmos_1p8", "p11ll_ckt", "pod33ll_ckt",
    "sky130_fd_pr__pfet_01v8", "sky130_fd_pr__pfet_01v8_lvt",
    "sky130_fd_pr__pfet_g5v0d10v5",
}

_NUM_RE = re.compile(r"^([-+0-9.eE]+)\s*([a-zA-Z]*)$")
_SUFFIX = {"t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3, "": 1.0, "m": 1e-3,
           "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15, "a": 1e-18}


def parse_value(text: str) -> Optional[float]:
    """Parse a SPICE number with an optional engineering suffix."""
    text = text.strip().strip("'\"")
    m = _NUM_RE.match(text)
    if not m:
        try:
            return float(text)
        except ValueError:
            return None
    mantissa, suffix = m.group(1), m.group(2).lower()
    try:
        val = float(mantissa)
    except ValueError:
        return None
    if suffix.startswith("meg"):
        return val * 1e6
    # 'n' in '1.5n' is nano, but a bare trailing letter that is not a known
    # suffix (e.g. '180nm') keeps only its first character.
    for key in (suffix, suffix[:1]):
        if key in _SUFFIX:
            return val * _SUFFIX[key]
    return val


@dataclass
class GMos:
    """A transistor with explicit geometry."""
    name: str
    nodes: List[str]
    kind: str                  # "n" | "p"
    w: float                   # [m]
    l: float                   # [m]
    mult: float                # m * multi combined
    # Raw geometry text.  The LDO netlists name their W/L through design
    # variables rather than stating numbers, and those names are what says which
    # devices are meant to track each other -- so they are kept for grouping.
    w_expr: str = ""
    l_expr: str = ""
    m_expr: str = ""

    @property
    def group(self) -> str:
        """Key that matched devices share."""
        if self.l_expr and parse_value(self.l_expr) is None:
            return f"{self.kind}:{self.l_expr}|{self.w_expr}|{self.m_expr}"
        return f"{self.kind}:w{self.w:.4g}|l{self.l:.4g}|x{self.mult:g}"


@dataclass
class GPassive:
    name: str
    nodes: List[str]
    value: str                 # verbatim SPICE value or expression


@dataclass
class GSubckt:
    name: str
    ports: List[str]
    mos: List[GMos] = field(default_factory=list)
    passives: List[GPassive] = field(default_factory=list)
    insts: List[Tuple[str, List[str], str]] = field(default_factory=list)


def _kv(tokens: List[str]) -> Dict[str, str]:
    """Collect ``key=value`` tokens, tolerating ``key = value`` spacing."""
    out: Dict[str, str] = {}
    joined = " ".join(tokens)
    for m in re.finditer(r"([A-Za-z_]\w*)\s*=\s*('[^']*'|\S+)", joined):
        out[m.group(1).lower()] = m.group(2).strip("'")
    return out


def parse_params(path: Path) -> Dict[str, float]:
    """Read ``.param name = value`` lines into a dict."""
    out: Dict[str, float] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        s = raw.split("$")[0].strip()
        if not s.lower().startswith(".param"):
            continue
        for m in re.finditer(r"([A-Za-z_]\w*)\s*=\s*(\S+)", s):
            v = parse_value(m.group(2))
            if v is not None:
                out[m.group(1).lower()] = v
    return out


def parse_generic(path: Path,
                  params: Optional[Dict[str, float]] = None
                  ) -> Tuple[List[GSubckt], List[str]]:
    """Parse a netlist into its subcircuits plus any top-level lines.

    Returns (subckts, top_level_lines).  Lines that are neither a transistor nor
    an R/C are kept verbatim so hierarchy and unknown devices survive.

    *params* resolves geometry given as a design-variable name (the charge pump
    states every W and L that way).
    """
    params = {k.lower(): v for k, v in (params or {}).items()}
    subs: List[GSubckt] = []
    top: List[str] = []
    stack: List[GSubckt] = []
    pending = ""

    def flush(line: str) -> None:
        if not line:
            return
        tokens = line.split()
        head = tokens[0].lower()
        target = stack[-1] if stack else None

        # transistor: name n1 n2 n3 n4 model [k=v ...]
        if len(tokens) >= 6:
            model = tokens[5].lower()
            if model in NMOS_MODELS or model in PMOS_MODELS:
                kv = _kv(tokens[6:])
                resolve = lambda s, d: (
                    parse_value(s) if parse_value(s) is not None
                    else params.get(s.strip().lower(), d))
                w = resolve(kv.get("w", "1u"), None)
                l = resolve(kv.get("l", "1u"), None)
                mult = (resolve(kv.get("m", "1"), 1.0) or 1.0) * \
                       (resolve(kv.get("multi", "1"), 1.0) or 1.0)
                mos = GMos(name=tokens[0], nodes=tokens[1:5],
                           kind="n" if model in NMOS_MODELS else "p",
                           w=w if w else 1e-6, l=l if l else 1e-6, mult=mult,
                           w_expr=kv.get("w", ""), l_expr=kv.get("l", ""),
                           m_expr=kv.get("m", ""))
                (target.mos if target else _implicit_top(subs).mos).append(mos)
                return

        if head[0] in "rc" and len(tokens) >= 4:
            pas = GPassive(name=tokens[0], nodes=tokens[1:3], value=tokens[3])
            (target.passives if target else _implicit_top(subs).passives
             ).append(pas)
            return

        if target is not None:
            target.insts.append((tokens[0], tokens[1:], line))
        else:
            top.append(line)

    for raw in path.read_text().splitlines():
        line = raw.split("$")[0].rstrip()
        s = line.strip()
        if not s or s.startswith("*"):
            continue
        if s.startswith("+"):
            pending += " " + s[1:].strip()
            continue
        if pending:
            flush(pending)
            pending = ""
        low = s.lower()
        if low.startswith(".subckt"):
            t = s.split()
            stack.append(GSubckt(name=t[1], ports=t[2:]))
            subs.append(stack[-1])
            continue
        if low.startswith(".ends"):
            if stack:
                stack.pop()
            continue
        if s.startswith("."):
            continue
        pending = s
    if pending:
        flush(pending)

    return subs, top


_TOP_NAME = "__top__"


def _implicit_top(subs: List[GSubckt]) -> GSubckt:
    """Container for devices that sit outside any .subckt."""
    for s in subs:
        if s.name == _TOP_NAME:
            return s
    s = GSubckt(name=_TOP_NAME, ports=[])
    subs.append(s)
    return s


# ---------------------------------------------------------------------------
# Geometry mapping
# ---------------------------------------------------------------------------
def map_length(l_src: float) -> float:
    """Preserve L absolutely, clamped into the TSMC16 binned range."""
    return snap_l(l_src)


def choose_nfin(devices: List[GMos], *, l_scale: bool = True) -> int:
    """Pick the single fin count that puts the narrowest device at m=1."""
    widths = []
    for d in devices:
        l16 = map_length(d.l)
        w16 = d.w * (l16 / d.l) if l_scale else d.w
        widths.append(w16)
    if not widths:
        return 4
    return snap_nfin(min(widths) / W_PER_FIN)


def device_m(d: GMos, nfin: int, *, l_scale: bool = True) -> int:
    """Multiplicity that reproduces this device's width at the chosen NFIN."""
    l16 = map_length(d.l)
    w16 = d.w * (l16 / d.l) if l_scale else d.w
    fins = w16 * d.mult / W_PER_FIN
    return max(1, int(round(fins / nfin)))
