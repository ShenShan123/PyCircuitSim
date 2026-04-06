#!/usr/bin/env python3
"""
Multi-technology, multi-variant NN compact model verification:
NN (LEVEL=73) vs PyCMG (LEVEL=72).

Tests NN-based MOSFET models against PyCMG ground truth across 5 technologies
and 2 device variants (SVT/RVT + LVT) per technology.

Verification tests per technology per variant:
  1. NMOS DC sweep (Id-Vgs at Vds=VDD/2) — current accuracy
  2. PMOS DC sweep (Id-Vgs at Vds=-VDD/2) — current accuracy
  3. Inverter VTC (Vout vs Vin) — circuit-level accuracy

Metric: NRMSE (%) normalized to peak-to-peak range.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from bsimar.config import TECH_CONFIGS, TechConfig, CHECKPOINT_DIR, OSDI_PATH
from pycmg import Model, Instance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def nrmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Normalized RMSE as percentage of peak-to-peak range."""
    ptp = true.max() - true.min()
    if ptp < 1e-30:
        return 0.0
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    return rmse / ptp * 100.0


def create_pycmg_instance(
    tech: TechConfig, device_type: str, nfin: float,
    variant: Optional[str] = None,
) -> Instance:
    """Create PyCMG instance for ground-truth evaluation."""
    model_name = tech.get_model_name(device_type, variant)
    modelcard_path = tech.get_modelcard_path(device_type, variant)
    L = tech.get_L(device_type)

    model = Model(
        osdi_path=OSDI_PATH,
        modelcard_path=modelcard_path,
        model_name=model_name,
        model_card_name=model_name,
    )
    return Instance(model=model, params={"L": L, "NFIN": float(nfin)},
                    temperature=tech.temperature)


def create_nn_instance(
    tech: TechConfig, device_type: str, nfin: float,
    variant: Optional[str] = None,
) -> object:
    """Create NN MOSFET instance with process params for the given variant."""
    from pycircuitsim.models.mosfet_nn import NMOS_NN, PMOS_NN

    # Prefer universal checkpoint if it exists
    universal_path = CHECKPOINT_DIR / f"universal_{device_type}_best.pt"
    if universal_path.exists():
        model_path = str(universal_path)
    else:
        tech_name = tech.name.lower()
        if tech_name == "asap7":
            prefix = device_type
        else:
            prefix = f"{tech_name}_{device_type}"
        model_path = str(CHECKPOINT_DIR / f"{prefix}_best.pt")

    L = tech.get_L(device_type)

    # Get process params for the variant
    phig = None
    process_params = None
    vname = variant or tech.default_variant
    if vname and vname in tech.variants:
        pp = tech.variants[vname].get_process_params(device_type)
        phig = pp.phig
        process_params = pp.as_dict()

    nodes = ["drain", "gate", "source", "bulk"]
    if device_type == "nmos":
        return NMOS_NN(name="mn_nn", nodes=nodes, model_path=model_path,
                       L=L, NFIN=nfin, phig=phig, process_params=process_params)
    else:
        return PMOS_NN(name="mp_nn", nodes=nodes, model_path=model_path,
                       L=L, NFIN=nfin, phig=phig, process_params=process_params)


# ---------------------------------------------------------------------------
# Test: Single device DC sweep
# ---------------------------------------------------------------------------
def test_device_dc_sweep(
    tech: TechConfig,
    device_type: str,
    nfin: float = 10.0,
    n_points: int = 71,
    variant: Optional[str] = None,
) -> Tuple[float, int, int]:
    """Compare NN vs PyCMG for a single-device DC sweep.

    NMOS: Sweep Vgs from 0 to VDD at Vds=VDD/2
    PMOS: Sweep Vgs from 0 to -VDD at Vds=-VDD/2 (source-relative)

    Returns: (nrmse_pct, n_converged, n_total)
    """
    vdd = tech.vdd

    # Create instances
    cmg_inst = create_pycmg_instance(tech, device_type, nfin, variant)
    nn_inst = create_nn_instance(tech, device_type, nfin, variant)

    if device_type == "nmos":
        vgs_sweep = np.linspace(0, vdd, n_points)
        vds = vdd / 2
    else:
        # PMOS in source-relative frame (Vs=0)
        vgs_sweep = np.linspace(0, -vdd, n_points)
        vds = -vdd / 2

    id_cmg = []
    id_nn = []
    n_converged = 0

    for vgs in vgs_sweep:
        # PyCMG ground truth
        try:
            result = cmg_inst.eval_dc({"d": vds, "g": vgs, "s": 0.0, "e": 0.0})
            i_cmg = result["id"]
        except Exception:
            continue

        # NN prediction
        voltages = {
            "drain": vds,
            "gate": vgs,
            "source": 0.0,
            "bulk": 0.0,
        }
        nn_inst.clear_cache()
        if device_type == "nmos":
            i_nn = -nn_inst.calculate_current(voltages)  # Back to SPICE convention
        else:
            i_nn = nn_inst.calculate_current(voltages)

        id_cmg.append(i_cmg)
        id_nn.append(i_nn)
        n_converged += 1

    if n_converged == 0:
        return 100.0, 0, n_points

    id_cmg_arr = np.array(id_cmg)
    id_nn_arr = np.array(id_nn)

    return nrmse(id_nn_arr, id_cmg_arr), n_converged, n_points


# ---------------------------------------------------------------------------
# Test: Inverter VTC
# ---------------------------------------------------------------------------
def test_inverter_vtc(
    tech: TechConfig,
    nfin: float = 10.0,
    n_points: int = 71,
    nmos_variant: Optional[str] = None,
    pmos_variant: Optional[str] = None,
) -> Tuple[float, int, int]:
    """Compare NN vs PyCMG inverter VTC using simple Newton-Raphson.

    Sweeps Vin from 0 to VDD and finds Vout at each point.
    Both NN and PyCMG use the same simple NR loop.

    Returns: (nrmse_pct, n_converged, n_total)
    """
    vdd = tech.vdd
    vin_sweep = np.linspace(0, vdd, n_points)

    # Create instances
    nmos_cmg = create_pycmg_instance(tech, "nmos", nfin, nmos_variant)
    pmos_cmg = create_pycmg_instance(tech, "pmos", nfin, pmos_variant)
    nmos_nn = create_nn_instance(tech, "nmos", nfin, nmos_variant)
    pmos_nn = create_nn_instance(tech, "pmos", nfin, pmos_variant)

    vout_cmg_list: List[float] = []
    vout_nn_list: List[float] = []
    n_converged = 0

    for vin in vin_sweep:
        # --- PyCMG inverter ---
        vout_cmg = _solve_inverter_pycmg(nmos_cmg, pmos_cmg, vin, vdd)
        if vout_cmg is None:
            continue

        # --- NN inverter ---
        vout_nn = _solve_inverter_nn(nmos_nn, pmos_nn, vin, vdd)
        if vout_nn is None:
            continue

        vout_cmg_list.append(vout_cmg)
        vout_nn_list.append(vout_nn)
        n_converged += 1

    if n_converged == 0:
        return 100.0, 0, n_points

    return nrmse(np.array(vout_nn_list), np.array(vout_cmg_list)), n_converged, n_points


def _solve_inverter_pycmg(
    nmos: Instance, pmos: Instance, vin: float, vdd: float,
    max_iter: int = 100, tol: float = 1e-9,
) -> Optional[float]:
    """Solve inverter for Vout using PyCMG + Newton-Raphson."""
    vout = vdd / 2  # Initial guess

    for _ in range(max_iter):
        try:
            rn = nmos.eval_dc({"d": vout, "g": vin, "s": 0.0, "e": 0.0})
            rp = pmos.eval_dc({"d": vout, "g": vin, "s": vdd, "e": vdd})
        except Exception:
            return None

        f = rn["id"] + rp["id"]
        J = abs(rn["gds"]) + abs(rp["gds"])
        if J < 1e-15:
            J = 1e-9

        dv = f / J
        if abs(dv) > 0.1:
            dv = 0.1 * np.sign(dv)
        vout += dv

        vout = max(-0.1, min(vdd + 0.1, vout))

        if abs(f) < tol:
            return vout

    return vout


def _solve_inverter_nn(
    nmos_nn: object, pmos_nn: object, vin: float, vdd: float,
    max_iter: int = 100, tol: float = 1e-9,
) -> Optional[float]:
    """Solve inverter for Vout using NN MOSFETs + Newton-Raphson."""
    vout = vdd / 2

    for _ in range(max_iter):
        nmos_v = {"drain": vout, "gate": vin, "source": 0.0, "bulk": 0.0}
        nmos_nn.clear_cache()
        gds_n, gm_n, gmb_n = nmos_nn.get_conductance(nmos_v)
        i_n = nmos_nn.calculate_current(nmos_v)

        pmos_v = {"drain": vout, "gate": vin, "source": vdd, "bulk": vdd}
        pmos_nn.clear_cache()
        gds_p, gm_p, gmb_p = pmos_nn.get_conductance(pmos_v)
        i_p = pmos_nn.calculate_current(pmos_v)

        f = i_n - i_p
        J = gds_n + gds_p
        if J < 1e-15:
            J = 1e-9

        dv = -f / J
        if abs(dv) > 0.1:
            dv = 0.1 * np.sign(dv)
        vout += dv
        vout = max(-0.1, min(vdd + 0.1, vout))

        if abs(f) < tol:
            return vout

    return vout


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    tech_name: str
    variant: str
    test_name: str
    nrmse_pct: float
    n_converged: int
    n_total: int
    elapsed_s: float


def run_all_tests() -> List[TestResult]:
    """Run all NN vs PyCMG tests across all technologies and variants."""
    results: List[TestResult] = []
    nfin = 10.0

    for tech_key, tech in TECH_CONFIGS.items():
        tech_name = tech.name

        # Check if checkpoints exist (universal or per-tech)
        universal_nmos = CHECKPOINT_DIR / "universal_nmos_best.pt"
        universal_pmos = CHECKPOINT_DIR / "universal_pmos_best.pt"
        if universal_nmos.exists() and universal_pmos.exists():
            pass  # Universal checkpoints found
        else:
            if tech_key == "asap7":
                nmos_ckpt = CHECKPOINT_DIR / "nmos_best.pt"
                pmos_ckpt = CHECKPOINT_DIR / "pmos_best.pt"
            else:
                nmos_ckpt = CHECKPOINT_DIR / f"{tech_key}_nmos_best.pt"
                pmos_ckpt = CHECKPOINT_DIR / f"{tech_key}_pmos_best.pt"

            if not nmos_ckpt.exists() or not pmos_ckpt.exists():
                print(f"\n  SKIP {tech_name}: checkpoints not found "
                      f"({nmos_ckpt.name}, {pmos_ckpt.name})")
                continue

        # Test each variant
        for variant_name in tech.variants:
            print(f"\n{'='*60}")
            print(f"  {tech_name} / {variant_name.upper()} (VDD={tech.vdd}V)")
            print(f"{'='*60}")

            # Test 1: NMOS DC sweep
            t0 = time.time()
            nrmse_n, conv_n, total_n = test_device_dc_sweep(
                tech, "nmos", nfin, variant=variant_name)
            dt = time.time() - t0
            status = "PASS" if nrmse_n < 15.0 else "FAIL"
            print(f"  NMOS DC sweep: NRMSE={nrmse_n:6.2f}%  ({conv_n}/{total_n} pts)  "
                  f"[{dt:.1f}s]  {status}")
            results.append(TestResult(tech_name, variant_name, "NMOS DC",
                                      nrmse_n, conv_n, total_n, dt))

            # Test 2: PMOS DC sweep
            t0 = time.time()
            nrmse_p, conv_p, total_p = test_device_dc_sweep(
                tech, "pmos", nfin, variant=variant_name)
            dt = time.time() - t0
            status = "PASS" if nrmse_p < 15.0 else "FAIL"
            print(f"  PMOS DC sweep: NRMSE={nrmse_p:6.2f}%  ({conv_p}/{total_p} pts)  "
                  f"[{dt:.1f}s]  {status}")
            results.append(TestResult(tech_name, variant_name, "PMOS DC",
                                      nrmse_p, conv_p, total_p, dt))

            # Test 3: Inverter VTC (same variant for both NMOS and PMOS)
            t0 = time.time()
            nrmse_inv, conv_inv, total_inv = test_inverter_vtc(
                tech, nfin, nmos_variant=variant_name, pmos_variant=variant_name)
            dt = time.time() - t0
            status = "PASS" if nrmse_inv < 20.0 else "FAIL"
            print(f"  Inverter VTC:  NRMSE={nrmse_inv:6.2f}%  ({conv_inv}/{total_inv} pts)  "
                  f"[{dt:.1f}s]  {status}")
            results.append(TestResult(tech_name, variant_name, "Inv VTC",
                                      nrmse_inv, conv_inv, total_inv, dt))

    return results


def print_summary(results: List[TestResult]) -> None:
    """Print summary table."""
    print(f"\n{'='*85}")
    print(f"  NN vs PyCMG Multi-Technology Multi-Variant Summary")
    print(f"{'='*85}")
    print(f"  {'Tech':<8s} {'Variant':<8s} {'Test':<10s} {'NRMSE(%)':<10s} "
          f"{'Conv':<8s} {'Time':<8s} {'Status'}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*6}")

    for r in results:
        threshold = 20.0 if "Inv" in r.test_name else 15.0
        status = "PASS" if r.nrmse_pct < threshold else "FAIL"
        conv_str = f"{r.n_converged}/{r.n_total}"
        print(f"  {r.tech_name:<8s} {r.variant:<8s} {r.test_name:<10s} "
              f"{r.nrmse_pct:<10.2f} {conv_str:<8s} {r.elapsed_s:<8.1f} {status}")

    # Per-tech-variant summary
    seen = []
    for r in results:
        key = (r.tech_name, r.variant)
        if key not in seen:
            seen.append(key)

    print(f"\n  Per-technology per-variant average NRMSE:")
    for tech_name, variant in seen:
        tech_results = [r for r in results
                        if r.tech_name == tech_name and r.variant == variant]
        avg = np.mean([r.nrmse_pct for r in tech_results])
        worst = max(r.nrmse_pct for r in tech_results)
        worst_name = [r for r in tech_results if r.nrmse_pct == worst][0].test_name
        print(f"    {tech_name:<8s} {variant:<8s}: avg={avg:.2f}%, "
              f"worst={worst:.2f}% ({worst_name})")

    # Overall stats
    n_pass = sum(1 for r in results
                 if r.nrmse_pct < (20.0 if "Inv" in r.test_name else 15.0))
    n_total = len(results)
    print(f"\n  Overall: {n_pass}/{n_total} PASS")


if __name__ == "__main__":
    print("NN Compact Model (LEVEL=73) Multi-Technology Multi-Variant Verification")
    print("Comparing NN predictions against PyCMG ground truth")
    print("Variants: SVT/RVT + LVT per technology")

    results = run_all_tests()

    if results:
        print_summary(results)
    else:
        print("\nNo tests were run. Make sure checkpoints exist.")
        print("Generate data: python -m nn_model.data.generate --device both --tech <tech>")
        print("Train: python -m nn_model.train --device-type <nmos|pmos> --tech <tech>")
