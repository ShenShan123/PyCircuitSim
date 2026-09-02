"""Shared NGSPICE probe helpers for the V7.5 core-feature gates.

Used by ``verify_cmg_multiplier.py``, ``verify_inductor.py``,
``verify_current_source_ngspice.py``, ``verify_cmg_set_temperature.py``,
``verify_tran_branch_current.py`` and ``verify_tran_gear2.py`` — the gates for
the compact-model core additions (instance multiplier ``m=``, Inductor, NGSPICE
current-source sign / PULSE, in-place ``set_temperature``, transient branch
currents, Gear-2 integration).

Everything here is a thin wrapper over ``tests.common.base``: one function
writes a deck + a ``.control`` runner, runs NGSPICE and returns the ``wrdata``
matrix. Column layout is the caller's business, because it depends on the
analysis: ``wrdata`` writes an x column in front of EVERY vector for real
data (x, y1, x, y2, ...) and (x, real, imag) for a complex (AC) vector.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from tests.common.base import (  # noqa: F401  (re-exported for the gates)
    MODELCARDS_DIR,
    NGSPICE_BIN,
    OSDI_PATH,
    PROJECT_ROOT,
    bake_inst_params,
    run_ngspice_subprocess,
)

#: ASAP7 TT modelcard — the same card every existing BSIM-CMG gate uses.
ASAP7_MODELCARD = MODELCARDS_DIR / "ASAP7" / "7nm_TT_160803.pm"


def bake_asap7(work_dir: Path, model_name: str,
               inst_params: Dict[str, float], tag: str = "") -> Path:
    """Bake instance params into the ASAP7 card for NGSPICE OSDI.

    NGSPICE's OSDI binding rejects instance parameters on the device line, so
    L / NFIN / DEVTYPE have to live in the ``.model`` block.

    Args:
        work_dir: Directory for the generated card
        model_name: Target ``.model`` name (e.g. "nmos_rvt")
        inst_params: Params to bake (L, NFIN, DEVTYPE, ...)
        tag: Optional filename discriminator

    Returns:
        Path of the baked modelcard
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    dst = work_dir / f"baked_{model_name}{('_' + tag) if tag else ''}.lib"
    bake_inst_params(ASAP7_MODELCARD, dst, model_name, inst_params)
    return dst


def ngspice_probe(
    work_dir: Path,
    label: str,
    deck_text: str,
    analysis: str,
    vectors: Sequence[str],
    extra_control: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Run one NGSPICE analysis and return its ``wrdata`` matrix.

    Args:
        work_dir: Directory for deck / runner / csv / log artifacts
        label: Filename discriminator (must be unique per probe)
        deck_text: The netlist (no ``.control`` block, no analysis card)
        analysis: The analysis line to execute (e.g. "op",
            "dc Vgs 0 0.7 0.05", "ac dec 10 1e3 1e7", "tran 1p 20n")
        vectors: Vectors to dump, e.g. ["v(out)", "i(vd)"]
        extra_control: Extra ``.control`` lines executed before the analysis
            (e.g. "set num_threads=1")

    Returns:
        2-D float array of the CSV rows (header stripped)
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    deck_path = work_dir / f"ng_{label}.cir"
    runner_path = work_dir / f"ng_{label}_runner.cir"
    csv_path = work_dir / f"ng_{label}.csv"
    log_path = work_dir / f"ng_{label}.log"

    deck_path.write_text(deck_text)

    control: List[str] = [
        "* NGSPICE runner (line 1 is the title — keep this comment)",
        ".control",
        "set noaskquit",
        "set num_threads=1",
        f"osdi {OSDI_PATH}",
        f"source {deck_path}",
        "set filetype=ascii",
        "set wr_vecnames",
    ]
    control.extend(extra_control or [])
    control.append(analysis)
    control.append(f"wrdata {csv_path} {' '.join(vectors)}")
    control.append(".endc")
    control.append(".end")
    runner_path.write_text("\n".join(control) + "\n")

    lines = run_ngspice_subprocess(runner_path, log_path, csv_path)

    rows: List[List[float]] = []
    for line in lines[1:]:  # skip the `set wr_vecnames` header
        s = line.strip()
        if s:
            rows.append([float(x) for x in s.split()])
    if not rows:
        raise RuntimeError(f"NGSPICE wrote no data rows: {csv_path}")
    return np.array(rows, dtype=float)


def rel_err(measured: float, reference: float, abs_floor: float) -> float:
    """Relative error with an absolute floor (avoids blowing up near zero)."""
    return abs(measured - reference) / max(abs(reference), abs_floor)


def report(name: str, ok: bool, detail: str = "") -> bool:
    """Print one gate line in the repo's `[PASS] ...` style and return ``ok``."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:52s} {detail}")
    return ok
