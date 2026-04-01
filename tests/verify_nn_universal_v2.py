#!/usr/bin/env python3
"""
Universal NN compact model verification v2: NN (LEVEL=73) vs PyCMG (LEVEL=72).

Standalone verification — directly evaluates NN and PyCMG without running the
full circuit simulator. Tests universal NMOS/PMOS models (trained on all
techs/variants) against PyCMG ground truth for all 21 device combos:

    ASAP7:  rvt, lvt, slvt, sram          (4 variants)
    TSMC5:  svt, lvt, ulvt, elvt           (4 variants)
    TSMC7:  svt, lvt, ulvt                 (3 variants)
    TSMC12: svt, lvt, ulvt, hvt, lnvt      (5 variants)
    TSMC16: svt, lvt, ulvt, hvt, lnvt      (5 variants)

Per combo: NMOS DC sweep, PMOS DC sweep, Inverter VTC.
Metric: NRMSE (%) normalized to peak-to-peak range.
PASS thresholds: device DC < 10%, inverter VTC < 15%.
"""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — standalone, no simulator dependency beyond mosfet_nn classes
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nn_model.config import TECH_CONFIGS, TechConfig, CHECKPOINT_DIR, OSDI_PATH
from pycmg import Model, Instance

# Thresholds
DEVICE_DC_THRESHOLD = 10.0   # NRMSE % for NMOS/PMOS DC sweeps
INVERTER_VTC_THRESHOLD = 15.0  # NRMSE % for inverter VTC
NFIN = 10.0
N_POINTS = 71

RESULTS_DIR = PROJECT_ROOT / "tests" / "verify_nn_universal_v2_results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def nrmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Normalized RMSE as percentage of peak-to-peak range."""
    ptp = float(true.max() - true.min())
    if ptp < 1e-30:
        return 0.0
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    return rmse / ptp * 100.0


def create_pycmg_instance(
    tech: TechConfig, device_type: str, nfin: float,
    variant: Optional[str] = None,
) -> Instance:
    """Create PyCMG instance for ground-truth evaluation.

    Includes TFIN and DEVTYPE in instance params to match training data
    generation pipeline (nn_model/data/generate.py).
    """
    model_name = tech.get_model_name(device_type, variant)
    modelcard_path = tech.get_modelcard_path(device_type, variant)
    L = tech.get_L(device_type)
    devtype = 1 if device_type == "nmos" else 0

    model = Model(
        osdi_path=OSDI_PATH,
        modelcard_path=modelcard_path,
        model_name=model_name,
        model_card_name=model_name,
    )
    return Instance(
        model=model,
        params={"L": L, "NFIN": float(nfin), "TFIN": tech.tfin, "DEVTYPE": devtype},
        temperature=tech.temperature,
    )


def create_nn_instance(
    tech: TechConfig, device_type: str, nfin: float,
    variant: Optional[str] = None,
) -> object:
    """Create NN MOSFET instance with process params for the given variant.

    Prefers universal checkpoint; falls back to per-tech checkpoint.
    """
    from pycircuitsim.models.mosfet_nn import NMOS_NN, PMOS_NN

    # Prefer universal checkpoint
    universal_path = CHECKPOINT_DIR / f"universal_{device_type}_best.pt"
    if universal_path.exists():
        model_path = str(universal_path)
    else:
        tech_name = tech.name.lower()
        if tech_name == "asap7":
            model_path = str(CHECKPOINT_DIR / f"{device_type}_best.pt")
        else:
            model_path = str(CHECKPOINT_DIR / f"{tech_name}_{device_type}_best.pt")

    L = tech.get_L(device_type)

    # Get process params for the variant
    vname = variant or tech.default_variant
    process_params: Optional[Dict[str, float]] = None
    phig: Optional[float] = None
    if vname and vname in tech.variants:
        pp = tech.variants[vname].get_process_params(device_type)
        process_params = pp.as_dict()
        phig = pp.phig

    nodes = ["drain", "gate", "source", "bulk"]
    if device_type == "nmos":
        return NMOS_NN(
            name="mn_nn", nodes=nodes, model_path=model_path,
            L=L, NFIN=nfin, phig=phig, process_params=process_params,
        )
    else:
        return PMOS_NN(
            name="mp_nn", nodes=nodes, model_path=model_path,
            L=L, NFIN=nfin, phig=phig, process_params=process_params,
        )


# ---------------------------------------------------------------------------
# Test: Single device DC sweep
# ---------------------------------------------------------------------------
def test_device_dc_sweep(
    tech: TechConfig,
    device_type: str,
    nfin: float = NFIN,
    n_points: int = N_POINTS,
    variant: Optional[str] = None,
) -> Tuple[float, int, int]:
    """Compare NN vs PyCMG for a single-device DC sweep.

    NMOS: Sweep Vgs from 0 to VDD at Vds=VDD/2.
    PMOS: Sweep Vgs from 0 to -VDD at Vds=-VDD/2 (source-relative, Vs=0).

    Returns: (nrmse_pct, n_converged, n_total)
    """
    vdd = tech.vdd

    cmg_inst = create_pycmg_instance(tech, device_type, nfin, variant)
    nn_inst = create_nn_instance(tech, device_type, nfin, variant)

    if device_type == "nmos":
        vgs_sweep = np.linspace(0, vdd, n_points)
        vds = vdd / 2
    else:
        # PMOS: source-relative frame (Vs=0), negative Vg/Vd
        vgs_sweep = np.linspace(0, -vdd, n_points)
        vds = -vdd / 2

    id_cmg: List[float] = []
    id_nn: List[float] = []
    n_converged = 0

    for vgs in vgs_sweep:
        # PyCMG ground truth
        try:
            result = cmg_inst.eval_dc({"d": vds, "g": vgs, "s": 0.0, "e": 0.0})
            i_cmg = result["id"]
        except Exception:
            continue

        # NN prediction
        voltages = {"drain": vds, "gate": vgs, "source": 0.0, "bulk": 0.0}
        nn_inst.clear_cache()
        if device_type == "nmos":
            # calculate_current returns positive = leaving drain (negate to SPICE convention)
            i_nn = -nn_inst.calculate_current(voltages)
        else:
            # calculate_current returns positive = entering drain (already SPICE convention)
            i_nn = nn_inst.calculate_current(voltages)

        id_cmg.append(i_cmg)
        id_nn.append(i_nn)
        n_converged += 1

    if n_converged == 0:
        return 100.0, 0, n_points

    return nrmse(np.array(id_nn), np.array(id_cmg)), n_converged, n_points


# ---------------------------------------------------------------------------
# Test: Inverter VTC
# ---------------------------------------------------------------------------
def test_inverter_vtc(
    tech: TechConfig,
    nfin: float = NFIN,
    n_points: int = N_POINTS,
    variant: Optional[str] = None,
) -> Tuple[float, int, int]:
    """Compare NN vs PyCMG inverter VTC.

    For each Vin, solves I_nmos(Vout) + I_pmos(Vout) = 0 via Newton-Raphson
    using both NN and PyCMG device evaluations.

    Returns: (nrmse_pct, n_converged, n_total)
    """
    vdd = tech.vdd
    vin_sweep = np.linspace(0, vdd, n_points)

    nmos_cmg = create_pycmg_instance(tech, "nmos", nfin, variant)
    pmos_cmg = create_pycmg_instance(tech, "pmos", nfin, variant)
    nmos_nn = create_nn_instance(tech, "nmos", nfin, variant)
    pmos_nn = create_nn_instance(tech, "pmos", nfin, variant)

    vout_cmg_list: List[float] = []
    vout_nn_list: List[float] = []
    n_converged = 0

    for vin in vin_sweep:
        vout_cmg = _solve_inverter_pycmg(nmos_cmg, pmos_cmg, vin, vdd)
        if vout_cmg is None:
            continue
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
    """Solve inverter Vout using PyCMG + Newton-Raphson.

    KCL at output: I_nmos(Vout) + I_pmos(Vout) = 0.
    Jacobian: gds_n + gds_p (both positive).
    """
    vout = vdd / 2  # initial guess
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
    """Solve inverter Vout using NN MOSFETs + Newton-Raphson.

    Uses autograd-consistent conductances from mosfet_nn for NR Jacobian.
    """
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

        # KCL: i_nmos_leaving - i_pmos_entering = 0
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
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    tech_name: str
    variant: str
    nmos_dc_nrmse: float
    pmos_dc_nrmse: float
    inv_vtc_nrmse: float
    nmos_conv: int
    pmos_conv: int
    inv_conv: int
    n_total: int
    elapsed_s: float

    @property
    def avg_nrmse(self) -> float:
        return (self.nmos_dc_nrmse + self.pmos_dc_nrmse + self.inv_vtc_nrmse) / 3.0

    @property
    def status(self) -> str:
        if (self.nmos_dc_nrmse < DEVICE_DC_THRESHOLD
                and self.pmos_dc_nrmse < DEVICE_DC_THRESHOLD
                and self.inv_vtc_nrmse < INVERTER_VTC_THRESHOLD):
            return "PASS"
        return "FAIL"


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_all_tests() -> List[TestResult]:
    """Run all NN vs PyCMG tests across all 21 tech+variant combos."""
    results: List[TestResult] = []

    # Check universal checkpoints
    nmos_ckpt = CHECKPOINT_DIR / "universal_nmos_best.pt"
    pmos_ckpt = CHECKPOINT_DIR / "universal_pmos_best.pt"
    has_universal = nmos_ckpt.exists() and pmos_ckpt.exists()

    if has_universal:
        print(f"  Using universal checkpoints:")
        print(f"    NMOS: {nmos_ckpt}")
        print(f"    PMOS: {pmos_ckpt}")
    else:
        print("  Universal checkpoints not found, falling back to per-tech...")

    combo_idx = 0
    total_combos = sum(len(t.variants) for t in TECH_CONFIGS.values())

    for tech_key, tech in TECH_CONFIGS.items():
        tech_name = tech.name

        # Verify per-tech checkpoints exist if universal not available
        if not has_universal:
            if tech_key == "asap7":
                n_ckpt = CHECKPOINT_DIR / "nmos_best.pt"
                p_ckpt = CHECKPOINT_DIR / "pmos_best.pt"
            else:
                n_ckpt = CHECKPOINT_DIR / f"{tech_key}_nmos_best.pt"
                p_ckpt = CHECKPOINT_DIR / f"{tech_key}_pmos_best.pt"
            if not n_ckpt.exists() or not p_ckpt.exists():
                print(f"\n  SKIP {tech_name}: checkpoints not found")
                continue

        for variant_name in tech.variants:
            combo_idx += 1
            print(f"\n{'='*65}")
            print(f"  [{combo_idx}/{total_combos}] {tech_name} / {variant_name.upper()} "
                  f"(VDD={tech.vdd}V)")
            print(f"{'='*65}")

            t0 = time.time()

            # Test 1: NMOS DC sweep
            nrmse_n, conv_n, total_n = test_device_dc_sweep(
                tech, "nmos", NFIN, N_POINTS, variant=variant_name)
            st_n = "PASS" if nrmse_n < DEVICE_DC_THRESHOLD else "FAIL"
            print(f"  NMOS DC:  NRMSE={nrmse_n:6.2f}%  ({conv_n}/{total_n} pts)  {st_n}")

            # Test 2: PMOS DC sweep
            nrmse_p, conv_p, total_p = test_device_dc_sweep(
                tech, "pmos", NFIN, N_POINTS, variant=variant_name)
            st_p = "PASS" if nrmse_p < DEVICE_DC_THRESHOLD else "FAIL"
            print(f"  PMOS DC:  NRMSE={nrmse_p:6.2f}%  ({conv_p}/{total_p} pts)  {st_p}")

            # Test 3: Inverter VTC
            nrmse_inv, conv_inv, total_inv = test_inverter_vtc(
                tech, NFIN, N_POINTS, variant=variant_name)
            st_inv = "PASS" if nrmse_inv < INVERTER_VTC_THRESHOLD else "FAIL"
            print(f"  Inv VTC:  NRMSE={nrmse_inv:6.2f}%  ({conv_inv}/{total_inv} pts)  {st_inv}")

            elapsed = time.time() - t0

            results.append(TestResult(
                tech_name=tech_name,
                variant=variant_name,
                nmos_dc_nrmse=nrmse_n,
                pmos_dc_nrmse=nrmse_p,
                inv_vtc_nrmse=nrmse_inv,
                nmos_conv=conv_n,
                pmos_conv=conv_p,
                inv_conv=conv_inv,
                n_total=total_n,
                elapsed_s=elapsed,
            ))

    return results


def print_summary(results: List[TestResult]) -> None:
    """Print condensed summary table."""
    print(f"\n{'='*90}")
    print(f"  Universal NN (LEVEL=73) vs PyCMG Ground Truth -- Summary")
    print(f"  NFIN={NFIN:.0f}, {N_POINTS} sweep points per test")
    print(f"  Thresholds: Device DC < {DEVICE_DC_THRESHOLD}%, Inverter VTC < {INVERTER_VTC_THRESHOLD}%")
    print(f"{'='*90}")
    header = (f"  {'Tech':<8s} {'Variant':<8s} {'NMOS_DC(%)':<12s} {'PMOS_DC(%)':<12s} "
              f"{'Inv_VTC(%)':<12s} {'Avg(%)':<10s} {'Status'}")
    print(header)
    print(f"  {'-'*8} {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*10} {'-'*6}")

    for r in results:
        print(f"  {r.tech_name:<8s} {r.variant:<8s} "
              f"{r.nmos_dc_nrmse:<12.2f} {r.pmos_dc_nrmse:<12.2f} "
              f"{r.inv_vtc_nrmse:<12.2f} {r.avg_nrmse:<10.2f} {r.status}")

    # Per-technology average
    print(f"\n  Per-technology average:")
    tech_names_seen: List[str] = []
    for r in results:
        if r.tech_name not in tech_names_seen:
            tech_names_seen.append(r.tech_name)

    for tn in tech_names_seen:
        tr = [r for r in results if r.tech_name == tn]
        avg_n = np.mean([r.nmos_dc_nrmse for r in tr])
        avg_p = np.mean([r.pmos_dc_nrmse for r in tr])
        avg_i = np.mean([r.inv_vtc_nrmse for r in tr])
        avg_all = np.mean([r.avg_nrmse for r in tr])
        worst = max(tr, key=lambda x: x.avg_nrmse)
        print(f"    {tn:<8s}: NMOS={avg_n:.2f}%  PMOS={avg_p:.2f}%  "
              f"INV={avg_i:.2f}%  avg={avg_all:.2f}%  "
              f"worst={worst.variant}({worst.avg_nrmse:.2f}%)")

    # Overall
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_total = len(results)
    total_time = sum(r.elapsed_s for r in results)
    print(f"\n  Overall: {n_pass}/{n_total} PASS  ({total_time:.1f}s total)")

    if n_pass == n_total:
        print("  All tests PASSED.")
    else:
        failed = [r for r in results if r.status == "FAIL"]
        print(f"  FAILED combos:")
        for r in failed:
            print(f"    {r.tech_name}/{r.variant}: "
                  f"NMOS={r.nmos_dc_nrmse:.2f}% PMOS={r.pmos_dc_nrmse:.2f}% "
                  f"INV={r.inv_vtc_nrmse:.2f}%")


def export_csv(results: List[TestResult], path: Path) -> None:
    """Export results to CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Tech", "Variant", "NMOS_DC_NRMSE(%)", "PMOS_DC_NRMSE(%)",
            "Inv_VTC_NRMSE(%)", "Avg_NRMSE(%)", "Status",
            "NMOS_Conv", "PMOS_Conv", "Inv_Conv", "N_Total", "Time(s)",
        ])
        for r in results:
            writer.writerow([
                r.tech_name, r.variant,
                f"{r.nmos_dc_nrmse:.4f}", f"{r.pmos_dc_nrmse:.4f}",
                f"{r.inv_vtc_nrmse:.4f}", f"{r.avg_nrmse:.4f}",
                r.status,
                r.nmos_conv, r.pmos_conv, r.inv_conv, r.n_total,
                f"{r.elapsed_s:.2f}",
            ])
    print(f"\n  Results exported to: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Universal NN Compact Model (LEVEL=73) Verification v2")
    print("Comparing universal NN predictions against PyCMG ground truth")
    print(f"21 tech+variant combos x 3 tests = 63 total tests")
    print(f"PyCMG OSDI: {OSDI_PATH}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")

    results = run_all_tests()

    if results:
        print_summary(results)
        csv_path = RESULTS_DIR / "universal_v2_summary.csv"
        export_csv(results, csv_path)
    else:
        print("\nNo tests were run. Make sure checkpoints exist.")
        print("Generate data:  python -m nn_model.data.generate --device both --universal")
        print("Train NMOS:     python -u -m nn_model.train --device-type nmos --universal "
              "--mode direct13 --epochs 800 --hidden 384 --layers 6 --patience 150 "
              "--batch-size 2048")
        print("Train PMOS:     python -u -m nn_model.train --device-type pmos --universal "
              "--mode direct13 --epochs 800 --hidden 384 --layers 6 --patience 150 "
              "--batch-size 2048")
