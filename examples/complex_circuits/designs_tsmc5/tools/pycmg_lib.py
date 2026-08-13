"""TSMC BSIM-CMG model libraries for ngspice, built through the PyCMG interface.

Tech-parametric: the technology node is derived from the name of the tree this
``tools/`` directory sits in (``designs_tsmc16`` -> TSMC16, ``designs_tsmc7`` ->
TSMC7, ...), overridable with the ``AG_TECH`` environment variable.  Every
per-tech constant -- supply, fin geometry, L bins, NFIN range, Vt flavors --
comes from the PyCMG tech registry plus a scanned-PDK geometry table, so the
same tools source runs unchanged in each per-tech tree.

ngspice's OSDI binding rejects instance parameters on the device line -- ``L=``
and ``NFIN=`` on an ``N...`` instance abort the parse with ``unknown parameter``.
Every distinct geometry therefore needs its own ``.model`` card with L / NFIN /
TFIN baked in.  The device multiplier ``m=`` *is* accepted (verified: m=3 gives
exactly 3x the current), so a library is keyed on (device, vt, L, NFIN) only and
multiplicity stays on the instance line.

The cards themselves come from PyCMG: ``pycmg.tech.resolve_modelcard`` selects the
right (L-bin, NFIN-group) variant out of the tech's PDK and writes a naive single
``.model`` card.  This module renames that card to a per-geometry alias so many
geometries can coexist in one file, and bakes the instance parameters in.

Usage::

    lib = ModelLibrary()
    name = lib.model_name("n", "svt", 60e-9, 4)   # -> "nsvt_l60_f4"
    lib.write(Path(MODELS_FILE))
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Vendored in-repo since V7.4.1: examples/complex_circuits/designs_tsmc*/tools ->
# four parents up is the repository root.  PYCMG_DIR overrides.
PYCMG_DIR = Path(
    os.environ.get(
        "PYCMG_DIR",
        str(Path(__file__).resolve().parents[4]
            / "external_compact_models" / "PyCMG"),
    )
)
if str(PYCMG_DIR) not in sys.path:
    sys.path.insert(0, str(PYCMG_DIR))

from pycmg.tech import get_tech_config, resolve_modelcard  # noqa: E402

OSDI_PATH: Path = PYCMG_DIR / "build" / "osdi" / "bsimcmg.osdi"

_ROOT: Path = Path(__file__).resolve().parents[1]

# Per-tech geometry envelope, scanned once from the PDK files (see the numbers
# in tools/README notes): svt-device L bins, NFIN groups and fin height.
_TECH_GEOM: Dict[str, Dict[str, float]] = {
    #            hfin     l_min   l_max    nfin_max
    "TSMC5":  dict(hfin=42e-9, l_min=6e-9,  l_max=135e-9, nfin_max=12),
    "TSMC6":  dict(hfin=42e-9, l_min=8e-9,  l_max=240e-9, nfin_max=20),
    "TSMC7":  dict(hfin=42e-9, l_min=8e-9,  l_max=240e-9, nfin_max=20),
    "TSMC12": dict(hfin=33e-9, l_min=16e-9, l_max=240e-9, nfin_max=20),
    "TSMC16": dict(hfin=33e-9, l_min=16e-9, l_max=240e-9, nfin_max=20),
}


def _detect_tech() -> str:
    """Tech from AG_TECH, else from the tree name (designs_tsmc7 -> TSMC7)."""
    env = os.environ.get("AG_TECH")
    if env:
        return env.upper()
    name = _ROOT.name.lower()
    for t in _TECH_GEOM:
        if name.endswith(t.lower()):
            return t
    return "TSMC16"


TECH: str = _detect_tech()
if TECH not in _TECH_GEOM:
    raise RuntimeError(f"unknown tech {TECH!r}; have {sorted(_TECH_GEOM)}")

_CFG = get_tech_config(TECH)
VDD: float = _CFG.vdd      # nominal supply, from the PyCMG tech registry
TFIN: float = _CFG.tfin    # fin thickness
HFIN: float = _TECH_GEOM[TECH]["hfin"]  # fin height, from the PDK modelcards
W_PER_FIN: float = 2 * HFIN + TFIN   # effective channel width per fin

# One model library file per tree, named for the tech: tsmc16_models.spice etc.
MODELS_FILE: str = f"{TECH.lower()}_models.spice"

# Cache of generated naive cards.  Absolute so PyCMG writes into this tree
# rather than into the PyCircuitSim checkout.
CACHE_DIR: Path = _ROOT / "models" / "cache"

# L-bin boundaries of the tech's PDK.  A geometry's L must land inside one of
# these; anything outside is clamped by snap_l().
L_MIN: float = _TECH_GEOM[TECH]["l_min"]
L_MAX: float = _TECH_GEOM[TECH]["l_max"]

# NFIN groups defined by the PDK (TSMC16: [1,2] [2,3] [3,4] [4,6] [6,20.888]).
NFIN_MIN: int = 1
NFIN_MAX: int = int(_TECH_GEOM[TECH]["nfin_max"])

# Vt flavors the PyCMG registry exposes for this tech.
VT_FLAVORS: Tuple[str, ...] = tuple(sorted(
    {k.split("_", 1)[1] for k in _CFG.devices}))

L_MIN_NM: float = round(L_MIN * 1e9)
L_MAX_NM: float = round(L_MAX * 1e9)


def scale_l(l_nm: float) -> float:
    """Re-express a channel length chosen on TSMC16's [16, 240] nm ladder in
    this tech's range: proportional to L_MAX, clamped to the binned range,
    whole nanometres.  Identity on TSMC16 and TSMC12."""
    return round(max(L_MIN_NM, min(L_MAX_NM, l_nm * L_MAX_NM / 240.0)))


# Discrete channel-length ladders for the coordinate searches, per tech.  On
# TSMC16 they reproduce the historical lists exactly ([16,20,36,...,240] and
# [36,60,...,240]); on the other techs the long-channel rungs scale with L_MAX
# and the short end starts at the tech's own L_MIN.
L_CHOICES_LONG: List[float] = sorted({scale_l(l)
                                      for l in (36, 60, 90, 120, 180, 240)})
L_CHOICES_FULL: List[float] = sorted({L_MIN_NM, round(1.25 * L_MIN_NM),
                                      *L_CHOICES_LONG})

# Fin-count ladder, clipped to the tech's NFIN range.
NFIN_CHOICES: List[int] = sorted({n for n in (2, 3, 4, 6, 10, 16, 20)
                                  if n <= NFIN_MAX} | {NFIN_MAX})


def snap_l(l_m: float) -> float:
    """Clamp a channel length into the TSMC16 PDK's binned range [16, 240] nm.

    Returns L rounded to a whole nanometre so the model alias and the cache
    filename (which PyCMG keys on ``int(L*1e9)``) stay in step.
    """
    l_nm = round(max(L_MIN, min(L_MAX, l_m)) * 1e9)
    return l_nm * 1e-9


def snap_nfin(nfin: float) -> int:
    """Clamp a fin count into the PDK's supported range."""
    return int(max(NFIN_MIN, min(NFIN_MAX, round(nfin))))


def split_width(w_total: float, nfin_max: int = NFIN_MAX) -> Tuple[int, int]:
    """Factor a target channel width into (NFIN, m).

    FinFET width is quantised: one fin contributes ``W_PER_FIN``.  A device of
    total width *w_total* therefore needs ``round(w_total / W_PER_FIN)`` fins,
    which is split across ``m`` parallel devices once it exceeds *nfin_max*.

    Returns (NFIN, m) with ``NFIN * m`` fins in total, both at least 1.
    """
    fins = max(1, round(w_total / W_PER_FIN))
    if fins <= nfin_max:
        return fins, 1
    m = -(-fins // nfin_max)          # ceil
    nfin = max(1, round(fins / m))
    return snap_nfin(nfin), m


def width_of(nfin: int, m: int = 1) -> float:
    """Effective channel width [m] of an (NFIN, m) device."""
    return nfin * m * W_PER_FIN


# ---------------------------------------------------------------------------
# Model card generation
# ---------------------------------------------------------------------------
_MODEL_RE = re.compile(r"^(\s*)\.model\s+(\S+)\s+(\S+)\s*\(", re.IGNORECASE)
_EOTACC_RE = re.compile(r"(eotacc\s*=\s*)([0-9eE+\-.]+)", re.IGNORECASE)
_PARAM_RE_CACHE: Dict[str, re.Pattern] = {}


def _param_re(key: str) -> re.Pattern:
    if key not in _PARAM_RE_CACHE:
        _PARAM_RE_CACHE[key] = re.compile(
            rf"(?i)^(\s*\+\s*){re.escape(key)}\s*=\s*[0-9eE+\-.]+\s*$"
        )
    return _PARAM_RE_CACHE[key]


def model_name(kind: str, vt: str, l_m: float, nfin: int) -> str:
    """Alias for one baked geometry, e.g. ``nsvt_l60_f4``.

    *kind* is ``"n"`` or ``"p"``; *vt* one of VT_FLAVORS.
    """
    return f"{kind}{vt}_l{round(snap_l(l_m) * 1e9)}_f{snap_nfin(nfin)}"


def _bake(card_text: str, pdk_device: str, alias: str,
          inst_params: Dict[str, float]) -> str:
    """Rename a naive PyCMG card to *alias* and bake instance parameters in.

    Mirrors ``PyCMG/tests/helpers.py:bake_inst_params`` -- including the EOTACC
    floor, which OSDI rejects below 1e-10 -- but renames the model so that many
    geometries can share one library file, and only ever touches the block whose
    name matches *pdk_device*.
    """
    card_text = _EOTACC_RE.sub(
        lambda m: m.group(1) + ("1.1e-10" if float(m.group(2)) <= 1.0e-10
                                else m.group(2)),
        card_text,
    )

    out: List[str] = []
    in_target = False
    seen: Set[str] = set()

    for raw in card_text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()
        m = _MODEL_RE.match(line)
        if m:
            in_target = m.group(2).lower() == pdk_device.lower()
            if in_target:
                line = f"{m.group(1)}.model {alias} bsimcmg ("
        elif in_target:
            if stripped == ")":
                for key, val in inst_params.items():
                    if key not in seen:
                        out.append(f"  + {key} = {val:g}")
                in_target = False
                seen.clear()
            else:
                for key, val in inst_params.items():
                    if _param_re(key).match(line):
                        line = f"  + {key} = {val:g}"
                        seen.add(key)
        out.append(line)

    if in_target:                      # card without a closing paren
        for key, val in inst_params.items():
            if key not in seen:
                out.append(f"  + {key} = {val:g}")

    return "\n".join(out) + "\n"


class ModelLibrary:
    """Collects the geometries a deck uses and emits one ngspice model file."""

    def __init__(self, tech: str = TECH) -> None:
        self.tech = get_tech_config(tech)
        self._specs: Dict[str, Tuple[str, str, float, int]] = {}

    def model_name(self, kind: str, vt: str, l_m: float, nfin: int) -> str:
        """Register a geometry and return the ``.model`` name to instantiate."""
        if kind not in ("n", "p"):
            raise ValueError(f"kind must be 'n' or 'p', got {kind!r}")
        if vt not in VT_FLAVORS:
            raise ValueError(f"unknown Vt flavor {vt!r}; have {VT_FLAVORS}")
        l_m = snap_l(l_m)
        nfin = snap_nfin(nfin)
        name = model_name(kind, vt, l_m, nfin)
        self._specs[name] = (kind, vt, l_m, nfin)
        return name

    def card(self, name: str) -> str:
        """Return the baked ``.model`` text for one registered alias."""
        kind, vt, l_m, nfin = self._specs[name]
        dev_key = f"{'nmos' if kind == 'n' else 'pmos'}_{vt}"
        device = self.tech.get_device(dev_key)
        path = Path(resolve_modelcard(device, self.tech, l_m, float(nfin),
                                      cache_dir=str(CACHE_DIR)))
        return _bake(
            path.read_text(),
            device.pdk_device,
            name,
            {"L": l_m, "NFIN": float(nfin), "TFIN": TFIN},
        )

    def write(self, out_path: Path) -> Path:
        """Write every registered geometry into one ngspice-includable file."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        chunks = [
            f"* {self.tech.name} BSIM-CMG (LEVEL=72) model library -- generated by "
            "tools/pycmg_lib.py via PyCMG resolve_modelcard().\n"
            "* L / NFIN / TFIN are baked in: ngspice's OSDI binding takes no "
            "instance parameters.\n"
            "* Multiplicity stays on the instance line as m=.\n"
        ]
        for name in sorted(self._specs):
            chunks.append(self.card(name))
        out_path.write_text("\n".join(chunks))
        return out_path

    def __len__(self) -> int:
        return len(self._specs)


def runner_deck(body_path: Path, control: str, title: str = "tsmc run") -> str:
    """Wrap a circuit deck in the ``.control`` block that loads the OSDI binary.

    ngspice needs ``osdi <path>`` executed before the netlist is sourced, so the
    circuit body lives in its own file and the runner sources it.
    """
    return (
        f"* {title}\n"
        f".control\n"
        f"osdi {OSDI_PATH}\n"
        f"source {body_path}\n"
        f"{control}\n"
        f".endc\n"
        f".end\n"
    )
