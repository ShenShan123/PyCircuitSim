"""Parse the sky130 AnalogGym netlists in ../designs into a structured topology.

The amplifier netlists are uniform: every transistor is

    xm<N> d g s b sky130_fd_pr__{n,p}fet_01v8 l='<Lvar>' w='<Wvar>*1' m='<expr>'

where ``<Lvar>``/``<Wvar>``/``<expr>`` name design variables of the form
``MOSFET_<i>_<j>_{L,W,M}_<role>_{NMOS,PMOS}``.  Devices that share a *role*
share a geometry -- that is how AnalogGym keeps matched pairs matched -- and the
integer factor in ``m='VAR*4'`` / ``m='4*VAR'`` is the mirror ratio, which must
survive the port unchanged.

Passives are ``C``/``R``/``I`` lines carrying a single quoted variable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SKY_NFET = "sky130_fd_pr__nfet_01v8"
SKY_PFET = "sky130_fd_pr__pfet_01v8"

_SUBCKT_RE = re.compile(r"^\.subckt\s+(\S+)\s+(.*)$", re.IGNORECASE)
_MOS_RE = re.compile(
    r"^(x?m\S*)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)$", re.IGNORECASE
)
_PASSIVE_RE = re.compile(
    r"^([CRIVcriv]\S*)\s+(\S+)\s+(\S+)\s+(.*)$"
)
_KV_RE = re.compile(r"(\w+)\s*=\s*'([^']*)'|(\w+)\s*=\s*(\S+)")
_ROLE_RE = re.compile(
    r"^MOSFET_(\d+)_(\d+)_([LWM])_(.+)_(NMOS|PMOS)$", re.IGNORECASE
)


@dataclass
class Mos:
    """One transistor instance."""
    name: str
    d: str
    g: str
    s: str
    b: str
    kind: str              # "n" or "p"
    role: str              # e.g. "gm1_PMOS"
    mult: int              # integer factor multiplying the role's M variable


@dataclass
class Passive:
    """One C / R / I instance whose value is a design variable."""
    name: str
    n1: str
    n2: str
    var: str               # variable name, or a literal if not parameterised


@dataclass
class Topology:
    subckt: str
    ports: List[str]
    mos: List[Mos] = field(default_factory=list)
    passives: List[Passive] = field(default_factory=list)

    @property
    def roles(self) -> List[str]:
        """Distinct device roles, in first-appearance order."""
        seen: List[str] = []
        for m in self.mos:
            if m.role not in seen:
                seen.append(m.role)
        return seen

    def role_kind(self, role: str) -> str:
        for m in self.mos:
            if m.role == role:
                return m.kind
        raise KeyError(role)

    @property
    def passive_vars(self) -> List[str]:
        seen: List[str] = []
        for p in self.passives:
            if p.var not in seen:
                seen.append(p.var)
        return seen


def _strip_quotes(s: str) -> str:
    return s.strip().strip("'\"").strip()


def _role_of(var: str) -> Optional[str]:
    m = _ROLE_RE.match(var.strip())
    if not m:
        return None
    return f"{m.group(4)}_{m.group(5).upper()}"


def _mult_of(expr: str) -> Tuple[Optional[str], int]:
    """Split ``m='VAR*4'`` / ``m='4*VAR'`` / ``m='VAR'`` into (var, factor)."""
    expr = _strip_quotes(expr)
    parts = [p.strip() for p in expr.split("*") if p.strip()]
    var: Optional[str] = None
    factor = 1
    for p in parts:
        if _ROLE_RE.match(p):
            var = p
        else:
            try:
                factor *= int(float(p))
            except ValueError:
                pass
    return var, max(1, factor)


def parse_netlist(path: Path) -> Topology:
    """Read one sky130 ``netlist.spice`` and return its topology."""
    topo: Optional[Topology] = None
    pending = ""

    for raw in path.read_text().splitlines():
        line = raw.split("$")[0].rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("*"):
            continue
        if stripped.startswith("+"):
            pending += " " + stripped[1:].strip()
            continue
        if pending:
            _consume(pending, topo)
            pending = ""

        sub = _SUBCKT_RE.match(stripped)
        if sub:
            topo = Topology(subckt=sub.group(1), ports=sub.group(2).split())
            continue
        if stripped.lower().startswith(".ends"):
            continue
        if stripped.startswith("."):
            continue
        pending = stripped

    if pending:
        _consume(pending, topo)

    if topo is None:
        raise ValueError(f"no .subckt found in {path}")
    return topo


def _consume(line: str, topo: Optional[Topology]) -> None:
    if topo is None:
        return

    mos = _MOS_RE.match(line)
    if mos and mos.group(6).lower() in (SKY_NFET, SKY_PFET):
        name, d, g, s, b, model, tail = mos.groups()
        kv: Dict[str, str] = {}
        for m in _KV_RE.finditer(tail):
            key = (m.group(1) or m.group(3)).lower()
            kv[key] = m.group(2) if m.group(2) is not None else m.group(4)
        role = _role_of(kv.get("l", ""))
        if role is None:
            raise ValueError(f"un-tagged transistor geometry in {topo.subckt}: {line}")
        _, mult = _mult_of(kv.get("m", "1"))
        topo.mos.append(Mos(
            name=name, d=d, g=g, s=s, b=b,
            kind="n" if model.lower() == SKY_NFET else "p",
            role=role, mult=mult,
        ))
        return

    pas = _PASSIVE_RE.match(line)
    if pas and line[0].upper() in "CRI":
        name, n1, n2, val = pas.groups()
        topo.passives.append(Passive(name=name, n1=n1, n2=n2,
                                     var=_strip_quotes(val)))


def parse_params(path: Path) -> Dict[str, str]:
    """Read a ``params.spice`` ``.PARAM`` block into {name: value-string}."""
    text = path.read_text()
    out: Dict[str, str] = {}
    for m in re.finditer(r"(\w+)\s*=\s*([^\s=]+)", text):
        key, val = m.group(1), m.group(2)
        if key.upper() == "PARAM":
            continue
        out[key] = val
    return out
