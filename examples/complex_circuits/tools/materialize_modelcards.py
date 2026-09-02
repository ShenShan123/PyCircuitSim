"""Materialize ignored per-design BSIM-CMG libraries from tracked geometry.

AnalogGym's translated ``netlist.spice`` files retain every device's VT,
channel length, and fin count in generated model aliases such as
``nsvt_l32_f7``.  The corresponding ``*_models.spice`` files are intentionally
ignored because they contain PDK-derived model data.  This tool reconstructs
only those private libraries; it never rewrites the tracked design corpus, so
deck regeneration and its upstream-source preflight remain separate.

Run one technology tree per process::

    conda run -n pycircuitsim python -m \
      examples.complex_circuits.tools.materialize_modelcards \
      --tree examples/complex_circuits/designs_tsmc5
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple


ModelSpec = Tuple[str, str, float, int]
_MODEL_ALIAS = re.compile(
    r"^([np])([^_\s]+)_l(\d+)_f(\d+)$", re.IGNORECASE
)


def model_specs(netlist_text: str) -> Dict[str, ModelSpec]:
    """Extract unique ``model -> (kind, vt, L_m, NFIN)`` specifications."""
    specs: Dict[str, ModelSpec] = {}
    for raw in netlist_text.splitlines():
        tokens = raw.split()
        if len(tokens) < 6 or not tokens[0].lower().startswith("n"):
            continue
        name = tokens[5]
        match = _MODEL_ALIAS.fullmatch(name)
        if match is None:
            raise ValueError(
                f"Cannot recover geometry from model alias {name!r}: {raw}")
        kind, vt, length_nm, nfin = match.groups()
        spec = (
            kind.lower(), vt.lower(), int(length_nm) * 1e-9, int(nfin)
        )
        previous = specs.setdefault(name, spec)
        if previous != spec:
            raise ValueError(
                f"Conflicting geometry specifications for {name}: "
                f"{previous} vs {spec}")
    return specs


def materialize_tree(tree: Path) -> int:
    """Write every private model library needed by one generated tree."""
    tree = tree.resolve()
    if not tree.is_dir():
        raise FileNotFoundError(f"AnalogGym technology tree not found: {tree}")

    os.environ["AG_TREE"] = str(tree)
    tools_dir = Path(__file__).resolve().parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from pycmg_lib import MODELS_FILE, ModelLibrary  # noqa: PLC0415

    netlists = sorted(tree.glob("*/*/netlist.spice"))
    if not netlists:
        raise RuntimeError(f"No generated design netlists found under {tree}")

    written = 0
    for netlist in netlists:
        specs = model_specs(netlist.read_text())
        if not specs:
            raise RuntimeError(f"No MOSFET model aliases found in {netlist}")
        library = ModelLibrary()
        for expected, (kind, vt, length_m, nfin) in specs.items():
            actual = library.model_name(kind, vt, length_m, nfin)
            if actual != expected:
                raise RuntimeError(
                    f"Model alias round-trip changed {expected!r} to "
                    f"{actual!r} in {netlist}")
        library.write(netlist.parent / MODELS_FILE)
        written += 1
        print(f"[modelcards] {netlist.parent.relative_to(tree)}")
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=Path, required=True)
    args = parser.parse_args(argv)
    count = materialize_tree(args.tree)
    print(f"[modelcards] wrote {count} private libraries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
