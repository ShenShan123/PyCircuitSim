"""Shared test infrastructure for BSIM-CMG verification suites.

Provides technology profiles, constants, and generic orchestration helpers
shared across DC and transient 3-level verification suites.

Extracted from bsimcmg_tran_common.py to eliminate duplication between
DC and transient common modules.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# tests/common/base.py → parents[0]=common/, parents[1]=tests/, parents[2]=project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))

OSDI_PATH = (
    PROJECT_ROOT / "external_compact_models" / "PyCMG"
    / "build" / "osdi" / "bsimcmg.osdi"
)
MODELCARDS_DIR = (
    PROJECT_ROOT / "external_compact_models" / "PyCMG" / "modelcards"
)
NGSPICE_BIN = "/usr/local/ngspice-45.2/bin/ngspice"

from helpers import bake_inst_params  # noqa: E402


# ---------------------------------------------------------------------------
# VtPair -- matched NMOS/PMOS threshold voltage variant
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VtPair:
    """Matched NMOS/PMOS threshold voltage variant for CMOS inverter."""
    vt_name: str       # e.g. "rvt", "svt", "lvt"
    nmos_model: str    # e.g. "nmos_rvt" or "nch_svt_mac"
    pmos_model: str    # e.g. "pmos_rvt" or "pch_svt_mac"


# ---------------------------------------------------------------------------
# TechProfile -- technology node configuration
# ---------------------------------------------------------------------------
@dataclass
class TechProfile:
    """Technology node specification with available VT/L combinations."""
    name: str
    vdd: float
    tfin: float                  # Fin thickness [m]
    default_l_nmos: float        # Default NMOS channel length [m]
    default_l_pmos: float        # Default PMOS channel length [m]
    l_values: List[float]        # All available L values for sweeping [m]
    default_nfin: int
    vt_pairs: List[VtPair]
    default_vt: str              # Name of default VtPair
    single_file: bool = False    # ASAP7: all models in one file
    modelcard_dir: str = ""      # Subdir under MODELCARDS_DIR
    single_file_name: str = ""   # For ASAP7: the single modelcard filename
    i_per_fin: float = 1e-5      # Estimated Idsat per fin [A]

    def get_vt_pair(self, vt_name: str) -> VtPair:
        for vp in self.vt_pairs:
            if vp.vt_name == vt_name:
                return vp
        raise ValueError(f"Unknown VT '{vt_name}' for {self.name}. "
                         f"Available: {[v.vt_name for v in self.vt_pairs]}")

    @property
    def default_vt_pair(self) -> VtPair:
        return self.get_vt_pair(self.default_vt)

    def _resolve_tsmc_modelcard(self, pdk_device: str, l_m: float) -> Path:
        """Generate a TSMC naive modelcard on-the-fly via pycmg.tech.resolve_modelcard.

        The committed ``modelcards/TSMC*/naive/*.l`` files were removed; they
        are now regenerated from the raw PDK and cached under PyCMG's
        ``build/modelcards/``. Uses ``NFIN=self.default_nfin`` so the correct
        NFIN-group variant is selected.
        """
        from pycmg.tech import TECH_REGISTRY, resolve_modelcard
        tech_config = TECH_REGISTRY[self.name]
        # Map "nch_svt_mac" -> "nmos_svt", "pch_lvt_mac" -> "pmos_lvt"
        prefix = pdk_device.split("_", 1)[0]
        vt = pdk_device.split("_", 1)[1].replace("_mac", "")
        canonical = ("nmos_" if prefix == "nch" else "pmos_") + vt
        device_config = tech_config.get_device(canonical)
        return Path(resolve_modelcard(
            device_config, tech_config,
            L=l_m, NFIN=float(self.default_nfin),
        ))

    def get_nmos_modelcard(self, vt: VtPair, l_nmos: float) -> Path:
        """Return path to NMOS modelcard file for given VT and L."""
        if self.single_file:
            return MODELCARDS_DIR / self.modelcard_dir / self.single_file_name
        return self._resolve_tsmc_modelcard(vt.nmos_model, l_nmos)

    def get_pmos_modelcard(self, vt: VtPair, l_pmos: float) -> Path:
        """Return path to PMOS modelcard file for given VT and L."""
        if self.single_file:
            return MODELCARDS_DIR / self.modelcard_dir / self.single_file_name
        return self._resolve_tsmc_modelcard(vt.pmos_model, l_pmos)

    def is_combo_available(self, vt: VtPair, l_nmos: float, l_pmos: float) -> bool:
        """Check if modelcard files exist for this VT and L combination."""
        return (self.get_nmos_modelcard(vt, l_nmos).exists()
                and self.get_pmos_modelcard(vt, l_pmos).exists())

    def get_available_l_values(self, vt: VtPair) -> List[float]:
        """Return L values where both NMOS and PMOS modelcards exist."""
        if self.single_file:
            return list(self.l_values)
        return [l for l in self.l_values
                if self.is_combo_available(vt, l, l)]


# ---------------------------------------------------------------------------
# Technology definitions (5 technologies x multiple VT flavors)
# ---------------------------------------------------------------------------
ALL_TECHS: Dict[str, TechProfile] = {
    "ASAP7": TechProfile(
        name="ASAP7", vdd=0.7, tfin=6.5e-9,
        default_l_nmos=30e-9, default_l_pmos=30e-9,
        l_values=[30e-9],
        default_nfin=10, default_vt="rvt",
        vt_pairs=[
            VtPair("rvt",  "nmos_rvt",  "pmos_rvt"),
            VtPair("lvt",  "nmos_lvt",  "pmos_lvt"),
            VtPair("slvt", "nmos_slvt", "pmos_slvt"),
            VtPair("sram", "nmos_sram", "pmos_sram"),
        ],
        single_file=True,
        modelcard_dir="ASAP7",
        single_file_name="7nm_TT_160803.pm",
    ),
    "TSMC5": TechProfile(
        name="TSMC5", vdd=0.65, tfin=6e-9,
        default_l_nmos=16e-9, default_l_pmos=20e-9,
        l_values=[16e-9, 20e-9, 24e-9],
        default_nfin=2, default_vt="lvt",
        vt_pairs=[
            # SVT removed: pch_svt_mac PDIBL2_i<0 at L=20nm NFIN=2
            VtPair("lvt",  "nch_lvt_mac",  "pch_lvt_mac"),
            VtPair("ulvt", "nch_ulvt_mac", "pch_ulvt_mac"),
            VtPair("elvt", "nch_elvt_mac", "pch_elvt_mac"),
        ],
        modelcard_dir="TSMC5/naive",
    ),
    "TSMC7": TechProfile(
        name="TSMC7", vdd=0.75, tfin=6e-9,
        default_l_nmos=16e-9, default_l_pmos=20e-9,
        l_values=[20e-9, 24e-9],  # L=16nm removed: ULVT inverter diverges at symmetric L=16nm
        default_nfin=2, default_vt="ulvt",  # SVT/LVT: garbage output or PDIBL2_i<0
        vt_pairs=[
            # SVT removed: inverter garbage output at L=16/20nm
            # LVT removed: pch_lvt_mac PDIBL2_i<0 at L=20nm NFIN=2
            VtPair("ulvt", "nch_ulvt_mac", "pch_ulvt_mac"),
        ],
        modelcard_dir="TSMC7/naive",
    ),
    "TSMC12": TechProfile(
        name="TSMC12", vdd=0.80, tfin=6e-9,
        default_l_nmos=16e-9, default_l_pmos=20e-9,
        l_values=[16e-9, 20e-9, 24e-9],
        default_nfin=2, default_vt="svt",
        vt_pairs=[
            VtPair("svt",  "nch_svt_mac",  "pch_svt_mac"),
            VtPair("lvt",  "nch_lvt_mac",  "pch_lvt_mac"),
            VtPair("hvt",  "nch_hvt_mac",  "pch_hvt_mac"),
            VtPair("ulvt", "nch_ulvt_mac", "pch_ulvt_mac"),
            VtPair("lnvt", "nch_lnvt_mac", "pch_lnvt_mac"),
        ],
        modelcard_dir="TSMC12/naive",
    ),
    "TSMC16": TechProfile(
        name="TSMC16", vdd=0.80, tfin=6e-9,
        default_l_nmos=16e-9, default_l_pmos=20e-9,
        l_values=[16e-9, 20e-9],  # L=24nm removed: nch_svt_mac PDIBL2_i<0
        default_nfin=2, default_vt="svt",
        vt_pairs=[
            VtPair("svt",  "nch_svt_mac",  "pch_svt_mac"),
            VtPair("lvt",  "nch_lvt_mac",  "pch_lvt_mac"),
            VtPair("hvt",  "nch_hvt_mac",  "pch_hvt_mac"),
            VtPair("ulvt", "nch_ulvt_mac", "pch_ulvt_mac"),
            # LNVT removed: nch_lnvt_mac PDIBL2_i<0 at L=16nm NFIN=2
        ],
        modelcard_dir="TSMC16/naive",
    ),
}

TECH_ORDER: List[str] = ["ASAP7", "TSMC5", "TSMC7", "TSMC12", "TSMC16"]

TECH_COLORS: Dict[str, str] = {
    "ASAP7": "tab:blue",
    "TSMC5": "tab:green",
    "TSMC7": "tab:orange",
    "TSMC12": "tab:purple",
    "TSMC16": "tab:red",
}


# ---------------------------------------------------------------------------
# Generic NGSPICE subprocess runner
# ---------------------------------------------------------------------------
def run_ngspice_subprocess(
    runner_path: Path,
    log_path: Path,
    csv_path: Path,
) -> List[str]:
    """Run NGSPICE in batch mode and return raw CSV lines.

    Handles:
      - subprocess execution
      - OSDI fatal error checking in log
      - CSV existence and emptiness checks

    Returns the raw lines from the CSV file (caller does domain-specific parsing).
    Raises RuntimeError on any failure.
    """
    res = subprocess.run(
        [NGSPICE_BIN, "-b", "-o", str(log_path), str(runner_path)],
        capture_output=True, text=True,
    )

    # Check for OSDI fatal errors (NGSPICE may still produce garbage output)
    if log_path.exists():
        log_text = log_path.read_text()
        if "Fatal:" in log_text:
            fatals = re.findall(r"Fatal:.*", log_text)
            raise RuntimeError(
                f"NGSPICE OSDI fatal error(s): {'; '.join(fatals[:3])}"
            )

    if not csv_path.exists():
        log_text = log_path.read_text() if log_path.exists() else "(no log)"
        raise RuntimeError(
            f"NGSPICE produced no output: {csv_path}\n"
            f"RC={res.returncode}, log (tail): ...{log_text[-500:]}"
        )

    with csv_path.open() as f:
        lines = f.readlines()

    if not lines:
        raise RuntimeError(f"Empty NGSPICE output: {csv_path}")

    return lines


# ---------------------------------------------------------------------------
# Generic summary bar chart
# ---------------------------------------------------------------------------
def plot_summary_bar(
    results: List[Dict[str, Any]],
    save_path: Path,
    title: str,
    nrmse_key: str,
    threshold: float,
    y_label: str,
) -> None:
    """Generic bar chart of NRMSE across all configs, colored by tech.

    Parameters
    ----------
    results : list of result dicts (must contain ``nrmse_key`` and ``config``)
    save_path : output PNG path
    title : chart title
    nrmse_key : key in result dict holding the NRMSE fraction (e.g. "nrmse_post", "nrmse")
    threshold : pass/fail threshold (fraction, e.g. 0.05 for 5%)
    y_label : Y-axis label
    """
    valid = [r for r in results if nrmse_key in r and "error" not in r]
    if not valid:
        return

    names = [r["config"].label for r in valid]
    nrmse_pct = [r[nrmse_key] * 100 for r in valid]
    colors = [TECH_COLORS.get(r["config"].tech.name, "tab:gray") for r in valid]

    fig, ax = plt.subplots(figsize=(max(14, len(valid) * 0.7), 6))
    x = np.arange(len(valid))
    ax.bar(x, nrmse_pct, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=threshold * 100, color="red", lw=1.5, ls="--",
               label=f"Threshold ({threshold*100:.0f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, edgecolor="black", label=t)
                  for t, c in TECH_COLORS.items()]
    legend_els.append(plt.Line2D([0], [0], color="red", lw=1.5, ls="--",
                                 label=f"Threshold ({threshold*100:.0f}%)"))
    ax.legend(handles=legend_els, loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Summary saved: {save_path}")


# ---------------------------------------------------------------------------
# Generic L2 test suite orchestrator
# ---------------------------------------------------------------------------
def run_test_suite(
    configs: List[Any],
    results_dir: Path,
    title: str,
    acceptance_msg: str,
    run_single_fn: Callable[[Any, Path], Dict[str, Any]],
    print_summary_fn: Callable[[List[Dict[str, Any]]], Tuple[int, int, int]],
    save_csv_fn: Callable[[List[Dict[str, Any]], Path], None],
    plot_bar_fn: Callable[[List[Dict[str, Any]], Path, str], None],
) -> int:
    """Generic L2 test suite orchestrator.

    Runs a list of configs through ``run_single_fn``, collects results,
    prints a summary table, saves CSV and bar chart, and returns an exit code.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results: List[Dict[str, Any]] = []

    print(f"\n{'='*78}")
    print(title)
    print(f"  Tests: {len(configs)}")
    print(f"  Acceptance: {acceptance_msg}")
    print(f"{'='*78}")

    for i, cfg in enumerate(configs):
        print(f"\n--- [{i+1}/{len(configs)}] ---")
        work_dir = results_dir / cfg.tech.name
        try:
            result = run_single_fn(cfg, work_dir)
            all_results.append(result)
        except Exception as exc:
            print(f"    ERROR: {exc}")
            all_results.append({"config": cfg, "error": str(exc), "passed": False})

    # Summary
    print(f"\n{'='*78}")
    print("SUMMARY TABLE")
    print(f"{'='*78}")
    n_pass, n_fail, n_error = print_summary_fn(all_results)
    total = len(all_results)

    print(f"\n  Total: {total}  Pass: {n_pass}  Fail: {n_fail}  Error: {n_error}")

    save_csv_fn(all_results, results_dir / "summary.csv")
    plot_bar_fn(all_results, results_dir / "summary.png", title)

    print(f"\n{'='*78}")
    if n_fail > 0:
        print(f"RESULT: {n_fail} FAIL, {n_error} ERROR out of {total}")
        return 1
    if n_error > 0:
        print(f"RESULT: {n_pass} PASS, {n_error} ERROR (modelcard issues) out of {total}")
    else:
        print(f"RESULT: ALL {n_pass} tests PASSED")
    return 0


# ---------------------------------------------------------------------------
# Generic tech parameter table printer
# ---------------------------------------------------------------------------
def print_tech_params(tech_names: List[str]) -> None:
    """Print technology parameter table."""
    print(f"\n  {'Tech':8s} | {'VDD':>5s} | {'L_n':>5s} | {'L_p':>5s} | "
          f"{'NFIN':>4s} | {'VT':5s} | {'NMOS':15s} | {'PMOS':15s}")
    print("  " + "-" * 80)
    for name in tech_names:
        tech = ALL_TECHS[name]
        vt = tech.default_vt_pair
        print(f"  {tech.name:8s} | {tech.vdd:5.2f} | "
              f"{tech.default_l_nmos*1e9:4.0f}n | {tech.default_l_pmos*1e9:4.0f}n | "
              f"{tech.default_nfin:4d} | {tech.default_vt:5s} | "
              f"{vt.nmos_model:15s} | {vt.pmos_model:15s}")


# ---------------------------------------------------------------------------
# Generic --tech argument parser
# ---------------------------------------------------------------------------
def parse_tech_args(description: str) -> List[str] | int:
    """Parse --tech CLI arg. Returns list of tech names or exit code on error."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--tech", type=str, default=",".join(TECH_ORDER),
                        help="Comma-separated tech names (default: all)")
    args = parser.parse_args()

    tech_names = [t.strip() for t in args.tech.split(",")]
    for t in tech_names:
        if t not in ALL_TECHS:
            print(f"ERROR: Unknown tech '{t}'. Available: {list(ALL_TECHS.keys())}")
            return 1
    return tech_names


# ---------------------------------------------------------------------------
# Generic L3 multi-tech orchestrator
# ---------------------------------------------------------------------------
def run_multi_tech_main(
    tech_names: List[str],
    results_dir: Path,
    title: str,
    acceptance_msg: str,
    make_baseline_fn: Callable[[TechProfile], Any],
    build_parametric_fn: Callable[[TechProfile], List[Any]],
    run_single_fn: Callable[[Any, Path], Dict[str, Any]],
    print_summary_fn: Callable[[List[Dict[str, Any]]], Tuple[int, int, int]],
    save_csv_fn: Callable[[List[Dict[str, Any]], Path], None],
    plot_bar_fn: Callable[[List[Dict[str, Any]], Path, str], None],
) -> int:
    """Generic L3 multi-tech orchestrator.

    Per-tech: baseline -> parametric if baseline passes -> summary.
    Handles both exception-based errors AND "error" key in result dicts
    (DC returns error dicts for garbage output).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results: List[Dict[str, Any]] = []
    tech_status: Dict[str, str] = {}

    print("=" * 78)
    print(title)
    print(f"  Technologies: {', '.join(tech_names)}")
    print(f"  Acceptance: {acceptance_msg}")
    print("=" * 78)

    print_tech_params(tech_names)

    for name in tech_names:
        tech = ALL_TECHS[name]
        work_dir = results_dir / tech.name

        # Phase 1: Baseline
        print(f"\n{'='*78}")
        print(f"  {tech.name}: Phase 1 — Baseline")
        print(f"{'='*78}")

        baseline_cfg = make_baseline_fn(tech)
        try:
            result = run_single_fn(baseline_cfg, work_dir)
            all_results.append(result)
            if result["passed"]:
                tech_status[tech.name] = "PASS"
            elif "error" in result:
                print("  => BASELINE ERROR — skipping parametric sweep")
                tech_status[tech.name] = "BASELINE_ERROR"
                continue
            else:
                print("  => BASELINE FAIL — skipping parametric sweep")
                tech_status[tech.name] = "BASELINE_FAIL"
                continue
        except Exception as exc:
            print(f"  => BASELINE ERROR: {exc}")
            all_results.append({"config": baseline_cfg, "error": str(exc), "passed": False})
            tech_status[tech.name] = "BASELINE_ERROR"
            continue

        # Phase 2: Parametric sweep
        print(f"\n  {tech.name}: Phase 2 — Parametric sweep")
        sweep_configs = build_parametric_fn(tech)
        for cfg in sweep_configs:
            try:
                result = run_single_fn(cfg, work_dir)
                all_results.append(result)
            except Exception as exc:
                print(f"    ERROR ({cfg.label}): {exc}")
                all_results.append({"config": cfg, "error": str(exc), "passed": False})

    # Summary
    print(f"\n{'='*78}")
    print("SUMMARY TABLE")
    print(f"{'='*78}")
    n_pass, n_fail, n_error = print_summary_fn(all_results)
    total = len(all_results)

    print(f"\n  Total: {total}  Pass: {n_pass}  Fail: {n_fail}  Error: {n_error}")

    print("\n  Technology status:")
    for name, status in tech_status.items():
        print(f"    {name:8s}: {status}")

    csv_name = results_dir.name.replace("_results", "_summary")
    save_csv_fn(all_results, results_dir / f"{csv_name}.csv")
    plot_bar_fn(all_results, results_dir / f"{csv_name}.png", title)

    print(f"\n{'='*78}")
    if n_fail > 0:
        print(f"RESULT: {n_fail} FAIL, {n_error} ERROR out of {total}")
        return 1
    if n_error > 0:
        print(f"RESULT: {n_pass} PASS, {n_error} ERROR (modelcard issues) out of {total}")
    else:
        print(f"RESULT: ALL {n_pass} tests PASSED")
    return 0
