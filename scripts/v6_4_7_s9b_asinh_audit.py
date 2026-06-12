"""V6.4.7 S9b — asinh per-target scale audit (plan P3 rev-3 (vii)).

For each dataset .npz, fit the asinh per-target scales s_k twice using the
exact training-time fit (``AsinhNormalizer._fit_outputs`` with
``_OUTPUT_LOG_FLOORS``):

  (a) filter-ON rows  — |id| > 1e-15 (the legacy loader filter), and
  (b) ALL rows        — what every V6.4.7 ``--apply-filter off`` arm trains on,

and report s_id (+ gm, gds, qg) both ways with the drift ratio
``s_all / s_filt`` as a markdown table at ``--out``.

Background (plan rev 3, ruling 3): ~6-8 % of rows in every per-tech dataset
have id == 0.0 exactly; removing the loader filter changes the rows the
geometric-mean s_id fit sees. NOTE the fit MASKS rows with |y| <= floor
(1e-18 for id) out of the geometric mean entirely — exact zeros never
contribute — so on pre-regen data only the (1e-18, 1e-15] band can move
s_id; post-regen (sub-nA densification) the drift is expected to be large.
This script MEASURES it so the campaign can pin s_id if the control-v2
inverter VTC canary degrades.

Usage:
    conda run -n pycircuitsim python scripts/v6_4_7_s9b_asinh_audit.py \
        external_compact_models/bsimar/data/datasets/tsmc7_nmos.npz \
        external_compact_models/bsimar/data/datasets/tsmc7_pmos.npz \
        --out /tmp/s9b_asinh_audit.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from bsimar.data.dataset import DEFAULT_FILTER_THRESHOLDS  # noqa: E402
from bsimar.data.normalize import (  # noqa: E402
    OUTPUT_COLUMN_ORDER,
    _OUTPUT_LOG_FLOORS,
    AsinhNormalizer,
)

# Targets reported in the table (id + 3 representative others).
_REPORT_TARGETS = ["id", "gm", "gds", "qg"]
_ID_FILTER_THR = DEFAULT_FILTER_THRESHOLDS["id"]  # 1e-15 (legacy loader)
_ID_FLOOR = _OUTPUT_LOG_FLOORS["id"]              # 1e-18 (fit mask floor)


def _fit_scales(outputs: np.ndarray) -> np.ndarray:
    """Per-target asinh scales via the exact training-time fit path."""
    _, _, asinh_scale = AsinhNormalizer()._fit_outputs(outputs)
    assert asinh_scale is not None
    return asinh_scale


def _audit_one(npz_path: Path) -> Dict[str, object]:
    data = np.load(npz_path, allow_pickle=True)
    outputs = np.asarray(data["outputs"], dtype=np.float64)
    if outputs.shape[1] != len(OUTPUT_COLUMN_ORDER):
        raise ValueError(
            f"{npz_path}: outputs has {outputs.shape[1]} cols, expected "
            f"{len(OUTPUT_COLUMN_ORDER)} ({OUTPUT_COLUMN_ORDER})")

    id_col = OUTPUT_COLUMN_ORDER.index("id")
    abs_id = np.abs(outputs[:, id_col])
    keep = abs_id > _ID_FILTER_THR

    n_total = len(outputs)
    n_filt = int(keep.sum())
    n_zero = int((abs_id == 0.0).sum())
    # Rows that the filter drops but the fit mask still counts — the only
    # rows that can actually move s_id between the two fits.
    n_band = int(((abs_id > _ID_FLOOR) & ~keep).sum())

    s_filt = _fit_scales(outputs[keep])
    s_all = _fit_scales(outputs)

    row: Dict[str, object] = {
        "cell": npz_path.stem,
        "n_total": n_total,
        "n_filt": n_filt,
        "n_zero": n_zero,
        "n_band": n_band,
    }
    for t in _REPORT_TARGETS:
        i = OUTPUT_COLUMN_ORDER.index(t)
        row[f"s_{t}_filt"] = float(s_filt[i])
        row[f"s_{t}_all"] = float(s_all[i])
        row[f"drift_{t}"] = float(s_all[i] / s_filt[i])
    return row


def _markdown(rows: List[Dict[str, object]]) -> str:
    lines = [
        "# V6.4.7 S9b — asinh scale audit (filter-ON vs ALL rows)",
        "",
        f"Fit: `AsinhNormalizer._fit_outputs` (geometric mean over rows "
        f"with |y| > floor; id floor = {_ID_FLOOR:g}). Filter-ON = "
        f"|id| > {_ID_FILTER_THR:g} (legacy loader). drift = s_all/s_filt.",
        "",
        "n_zero = exact-zero id rows (masked out of BOTH fits); "
        "n_band = rows in (floor, 1e-15] — the only rows that move s_id.",
        "",
    ]
    hdr = ["cell", "n_total", "n_filt", "n_zero", "n_band"]
    for t in _REPORT_TARGETS:
        hdr += [f"s_{t}_filt", f"s_{t}_all", f"drift_{t}"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    for r in rows:
        cells = []
        for h in hdr:
            v = r[h]
            if isinstance(v, float):
                cells.append(f"{v:.4e}" if "s_" in h else f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="S9b asinh per-target scale audit (filter-ON vs ALL)")
    ap.add_argument("datasets", nargs="+", type=Path,
                    help="Dataset .npz paths (per-tech per-device)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Markdown report output path")
    args = ap.parse_args()

    rows = []
    for p in args.datasets:
        if not p.exists():
            raise SystemExit(f"Dataset not found: {p}")
        print(f"Auditing {p} …")
        rows.append(_audit_one(p))

    md = _markdown(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(md)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
