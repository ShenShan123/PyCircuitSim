#!/usr/bin/env python
"""TSMC PDK parse-coverage audit (V6.9.0).

Audits, for every registered TSMC tech (TSMC5/6/7/12/16), every core
device (nch/pch x vt flavor) and every L/NFIN bin, that the PyCMG
naive-modelcard pipeline (`parse_tsmc_pdk` -> `generate_naive_tsmc_modelcard`
-> `parse_modelcard`) extracts the raw PDK `.model` blocks faithfully,
and quantifies every simplification it applies:

  A. Raw-block assignment coverage — every `name = value` token inside the
     `.global` + numbered `.model` blocks is classified as: numeric
     (captured), or non-numeric/expression (silently skipped by
     `_ASSIGN_RE`). Skipped names are reported so physically load-bearing
     parameters can't vanish unnoticed.
  B. Mid-line `*` hazard — `_extract_model_params` only skips lines that
     START with `*`; a mid-line comment containing `x = 3` tokens would be
     parsed as a parameter. Reports any block line with content before `*`.
  C. Bin selection ambiguity — `_find_length_variant` uses inclusive
     bounds on both ends; adjacent bins share boundaries, so boundary
     (L, NFIN) points can match >1 bin (first-in-file-order wins, HSPICE
     uses lmin <= L < lmax). Counts multi-match grid points per device.
  D. Naive-card round trip — for every bin (at L=lmin, NFIN=nfinmin) the
     naive card must equal merged(global, bin) minus instance params minus
     TSMC ±999e+n sentinels, modulo the documented EOTACC clamp and
     devtype injection.
  E. OSDI-unknown params — parameters in the naive card that the OSDI
     descriptor does not define (TMI hooks etc.); `apply_param` silently
     ignores these on both the PyCMG and NGSPICE sides.

Usage:
    python scripts/audit_pdk_parse.py [--techs TSMC5,TSMC6,...] [--jobs N]
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pycmg.parser import (  # noqa: E402
    _ASSIGN_RE, _scan_all_variants, _find_length_variant,
    parse_modelcard, parse_tsmc_pdk, scan_pdk_geometry_combos,
)
from pycmg.tech import (  # noqa: E402
    TECH_REGISTRY, _INSTANCE_PARAMS, generate_naive_tsmc_modelcard,
)

# Any `name = <rhs>` where rhs is NOT purely numeric — quoted expressions,
# agauss(...), parameter references. These are what _ASSIGN_RE skips.
_ANY_ASSIGN_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*('[^']*'|\"[^\"]*\"|[^\s=]+)")

_DEFAULT_TECHS = ["TSMC5", "TSMC6", "TSMC7", "TSMC12", "TSMC16"]


def _is_sentinel(fval: float) -> bool:
    """Mirror of the sentinel filter in generate_naive_tsmc_modelcard."""
    if abs(fval) <= 1e6:
        return False
    try:
        exp = int(math.log10(abs(fval)))
        mantissa = abs(fval) / (10.0 ** (exp - 2))
        return abs(mantissa - 999.0) < 0.5
    except (ValueError, OverflowError):
        return False


def _iter_model_blocks(path: str, base_name: str):
    """Yield (block_name, [raw_block_lines]) for `.model {base_name}.*`."""
    prefix = f"{base_name.lower()}."
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    idx = 0
    while idx < len(lines):
        trimmed = lines[idx].strip()
        if not trimmed or trimmed.startswith("*"):
            idx += 1
            continue
        if trimmed.lower().startswith(".model"):
            parts = trimmed.split()
            if len(parts) >= 3 and parts[1].lower().startswith(prefix):
                block = [trimmed]
                idx += 1
                while idx < len(lines):
                    cont = lines[idx].strip()
                    if not cont or cont.startswith("*"):
                        idx += 1
                        continue
                    if cont.startswith("+"):
                        block.append(cont)
                        idx += 1
                        continue
                    break
                yield parts[1], block
                continue
        idx += 1


def _osdi_param_names() -> Set[str]:
    """All parameter names the OSDI descriptor defines (lowercase)."""
    from pycmg.core import OsdiLibrary
    osdi = ROOT / "build" / "osdi" / "bsimcmg.osdi"
    if not osdi.exists():
        return set()
    lib = OsdiLibrary(str(osdi))
    desc = lib.descriptor(0)
    names: Set[str] = set()
    for i in range(desc.num_params):
        p = desc.param_opvar[i]
        if p.name:
            names.add(p.name[0].decode("utf-8", errors="replace").lower())
    return names


def audit_device(args: Tuple[str, str]) -> Dict:
    """Audit one (tech_name, base_name); returns a result record."""
    tech_name, base_name = args
    tech = TECH_REGISTRY[tech_name]
    pdk = str(ROOT / tech.pdk_path)
    model_type, device_type = base_name.split("_", 1)
    rec: Dict = {
        "tech": tech_name, "base": base_name, "bins": 0,
        "assign_total": 0, "assign_numeric": 0,
        "skipped_names": set(), "midline_star_lines": 0,
        "midline_star_assign_risk": 0,
        "ambiguous_grid_points": 0, "grid_points": 0,
        "roundtrip_bins": 0, "roundtrip_mismatches": [],
        "sentinels_dropped": 0, "errors": [],
    }

    # A + B: raw block scan
    for name, block in _iter_model_blocks(pdk, base_name):
        for raw in block:
            body = raw[1:] if raw.startswith("+") else raw
            star = body.find("*")
            if star > 0 and body[:star].strip():
                rec["midline_star_lines"] += 1
                if _ANY_ASSIGN_RE.search(body[star + 1:]):
                    rec["midline_star_assign_risk"] += 1
                body = body[:star]
            for m in _ANY_ASSIGN_RE.finditer(body):
                rec["assign_total"] += 1
                if _ASSIGN_RE.match(m.group(0)):
                    rec["assign_numeric"] += 1
                else:
                    rec["skipped_names"].add(m.group(1).lower())

    # C: bin scan + boundary ambiguity
    variants = _scan_all_variants(pdk, base_name)
    rec["bins"] = len(variants)
    combos = scan_pdk_geometry_combos(pdk, base_name)
    rec["grid_points"] = len(combos)
    for (L, NFIN) in combos:
        matches = [v for v in variants
                   if v.lmin <= L <= v.lmax
                   and v.nfinmin is not None and v.nfinmax is not None
                   and v.nfinmin <= NFIN <= v.nfinmax]
        if len(matches) > 1:
            rec["ambiguous_grid_points"] += 1

    # D + E: round trip per bin at (lmin, nfinmin)
    cache_dir = ROOT / "build" / "modelcards" / f"{tech_name}_audit"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for v in variants:
        L, NFIN = v.lmin, v.nfinmin
        try:
            merged = parse_tsmc_pdk(pdk, model_type, device_type, L, NFIN).params
            out = cache_dir / f"{base_name}_bin{v.variant_num}.l"
            generate_naive_tsmc_modelcard(
                pdk, model_type, device_type, L, str(out), tech_name, NFIN)
            naive = parse_modelcard(str(out), base_name).params
        except Exception as exc:  # pragma: no cover - report, don't die
            rec["errors"].append(f"bin {v.variant_num}: {exc!r}")
            continue
        rec["roundtrip_bins"] += 1

        expected: Dict[str, float] = {}
        for k, val in merged.items():
            if k in _INSTANCE_PARAMS:
                continue
            if _is_sentinel(float(val)):
                rec["sentinels_dropped"] += 1
                continue
            expected[k] = float(val)
        # parse_modelcard forces nf/nfin to 1.0 and injects devtype;
        # nf/nfin are instance params (excluded above), devtype is in both.
        for k in set(expected) | set(naive):
            if k in ("nf", "nfin"):
                continue
            ev, nv = expected.get(k), naive.get(k)
            if ev is None or nv is None or not math.isclose(
                    ev, nv, rel_tol=1e-12, abs_tol=0.0):
                rec["roundtrip_mismatches"].append(
                    f"bin {v.variant_num} {k}: merged={ev!r} naive={nv!r}")

    rec["skipped_names"] = sorted(rec["skipped_names"])
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--techs", default=",".join(_DEFAULT_TECHS))
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 4))
    args = ap.parse_args()

    techs = [t.strip() for t in args.techs.split(",") if t.strip()]
    jobs_list: List[Tuple[str, str]] = []
    for t in techs:
        tech = TECH_REGISTRY[t]
        seen = set()
        for dev in tech.devices.values():
            if dev.pdk_device and dev.pdk_device not in seen:
                seen.add(dev.pdk_device)
                jobs_list.append((t, dev.pdk_device))

    osdi_names = _osdi_param_names()

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(audit_device, jobs_list))

    hard_fail = False
    skipped_union: Dict[str, Set[str]] = defaultdict(set)
    print(f"{'tech':7s} {'device':14s} {'bins':>4s} {'assigns':>8s} "
          f"{'numeric':>8s} {'expr-skip':>9s} {'star':>4s} {'amb':>4s} "
          f"{'rt-bins':>7s} {'rt-miss':>7s} {'sent':>5s}")
    for r in sorted(results, key=lambda r: (r["tech"], r["base"])):
        print(f"{r['tech']:7s} {r['base']:14s} {r['bins']:4d} "
              f"{r['assign_total']:8d} {r['assign_numeric']:8d} "
              f"{r['assign_total']-r['assign_numeric']:9d} "
              f"{r['midline_star_assign_risk']:4d} "
              f"{r['ambiguous_grid_points']:4d} "
              f"{r['roundtrip_bins']:7d} {len(r['roundtrip_mismatches']):7d} "
              f"{r['sentinels_dropped']:5d}")
        for n in r["skipped_names"]:
            skipped_union[n].add(r["tech"])
        if r["roundtrip_mismatches"]:
            hard_fail = True
            for m in r["roundtrip_mismatches"][:5]:
                print(f"    MISMATCH {m}")
        if r["errors"]:
            hard_fail = True
            for e in r["errors"][:5]:
                print(f"    ERROR {e}")

    if skipped_union:
        print("\nExpression-valued (skipped) parameter names "
              "[name: techs] — verify none are load-bearing BSIM-CMG params:")
        for n in sorted(skipped_union):
            osdi_flag = " **OSDI-KNOWN**" if n in osdi_names else ""
            print(f"  {n}: {','.join(sorted(skipped_union[n]))}{osdi_flag}")

    # E: naive-card params unknown to OSDI (sample: first device per tech)
    if osdi_names:
        print("\nNaive-card params not defined in the OSDI descriptor "
              "(silently ignored by apply_param on BOTH PyCMG and NGSPICE):")
        seen_tech = set()
        for r in sorted(results, key=lambda r: (r["tech"], r["base"])):
            if r["tech"] in seen_tech:
                continue
            seen_tech.add(r["tech"])
            cache_dir = ROOT / "build" / "modelcards" / f"{r['tech']}_audit"
            cards = sorted(cache_dir.glob(f"{r['base']}_bin*.l"))
            if not cards:
                continue
            params = parse_modelcard(str(cards[0]), r["base"]).params
            unknown = sorted(k for k in params if k not in osdi_names)
            print(f"  {r['tech']}/{r['base']}: {len(unknown)} unknown: "
                  f"{', '.join(unknown[:12])}{' ...' if len(unknown) > 12 else ''}")

    print(f"\nAUDIT {'FAIL' if hard_fail else 'PASS'} "
          f"({len(results)} device audits)")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
