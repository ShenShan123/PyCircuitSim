#!/usr/bin/env python3
"""
Comprehensive NMOS DC comparison: DirectNet (LEVEL=73) vs BSIMAR (LEVEL=74) vs PyCMG (LEVEL=72).

Runs two sweep types per technology per variant:
  1. Id-Vgs sweep at Vds = VDD/2 (transfer characteristic)
  2. Id-Vds sweep at Vgs = VDD*0.6 (output characteristic)

Plus conductance comparisons (gm, gds) from autograd.

Reports: NRMSE (%), MRE (%), max absolute error, and per-output breakdown.
Generates comparison plots under tests/compare_nn_results/.

NOTE: Only NMOS checkpoints exist. PMOS and full inverter tests are blocked
until PMOS checkpoints are trained.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# PROJECT_ROOT must be first so `tests.common` resolves to our tests, not PyCMG's.
# PyCMG/external_compact_models paths are handled inside tests.common.nn.
sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.nn import (
    TECH_CONFIGS, TechConfig, CHECKPOINT_DIR, OSDI_PATH,
    default_L, get_process_params, directnet_checkpoint,
    nrmse, mre,
)
from pycmg import Model, Instance

# ── Output directory ────────────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "tests" / "compare_nn_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Checkpoint resolution ───────────────────────────────────────────────────

def _resolve_bsimar_checkpoint(device_type: str) -> Path:
    """Resolve BSIMAR Transformer _best.phys.pt checkpoint (physics-space-best)."""
    phys = CHECKPOINT_DIR / f"ar_universal_{device_type}_best.phys.pt"
    if phys.exists():
        return phys
    # fallback to _best.pt
    return CHECKPOINT_DIR / f"ar_universal_{device_type}_best.pt"


# ── Instance creation ───────────────────────────────────────────────────────

def create_pycmg_instance(
    tech: TechConfig, device_type: str, nfin: float,
    variant: Optional[str] = None,
) -> Instance:
    """Create PyCMG instance for ground-truth evaluation."""
    vname = variant or tech.default_variant
    L = default_L(tech.name, device_type)
    model_name = tech.get_model_name(device_type, vname)
    modelcard_path = tech.resolve_modelcard(device_type, vname, L=L, NFIN=nfin)
    model = Model(
        osdi_path=str(OSDI_PATH),
        modelcard_path=str(modelcard_path),
        model_name=model_name,
        model_card_name=model_name,
    )
    return Instance(model=model, params={"L": L, "NFIN": float(nfin)},
                    temperature=tech.temperature)


def create_directnet_instance(
    tech: TechConfig, device_type: str, nfin: float,
    variant: Optional[str] = None,
) -> object:
    """Create DirectNet (LEVEL=73) MOSFET instance."""
    from pycircuitsim.models.mosfet_directnet import NMOS_NN, PMOS_NN

    model_path = str(directnet_checkpoint(device_type, tech.name))
    L = default_L(tech.name, device_type)
    vname = variant or tech.default_variant
    process_params = get_process_params(tech, device_type, vname, L=L, NFIN=nfin)
    phig = process_params.get("phig")

    nodes = ["drain", "gate", "source", "bulk"]
    cls = NMOS_NN if device_type == "nmos" else PMOS_NN
    return cls(name="mn_dn", nodes=nodes, model_path=model_path,
               L=L, NFIN=nfin, phig=phig, process_params=process_params)


def create_bsimar_instance(
    tech: TechConfig, device_type: str, nfin: float,
    variant: Optional[str] = None,
) -> object:
    """Create BSIMAR Transformer (LEVEL=74) MOSFET instance."""
    from pycircuitsim.models.mosfet_bsimar import NMOS_BSIMAR, PMOS_BSIMAR

    model_path = str(_resolve_bsimar_checkpoint(device_type))
    L = default_L(tech.name, device_type)
    vname = variant or tech.default_variant
    process_params = get_process_params(tech, device_type, vname, L=L, NFIN=nfin)
    phig = process_params.get("phig")

    nodes = ["drain", "gate", "source", "bulk"]
    cls = NMOS_BSIMAR if device_type == "nmos" else PMOS_BSIMAR
    return cls(name="mn_ar", nodes=nodes, model_path=model_path,
               L=L, NFIN=nfin, phig=phig, process_params=process_params)


# ── Sweep helpers ───────────────────────────────────────────────────────────

def _eval_pycmg(inst: Instance, vds: float, vgs: float) -> Optional[Dict[str, float]]:
    """Evaluate PyCMG at a single bias point. Returns None on failure."""
    try:
        return inst.eval_dc({"d": vds, "g": vgs, "s": 0.0, "e": 0.0})
    except Exception:
        return None


def _eval_nn(inst: object, vds: float, vgs: float, device_type: str) -> Dict[str, float]:
    """Evaluate an NN MOSFET at a single bias point. Returns id (SPICE sign), gm, gds, gmb."""
    voltages = {"drain": vds, "gate": vgs, "source": 0.0, "bulk": 0.0}
    inst.clear_cache()
    i_calc = inst.calculate_current(voltages)
    gds, gm, gmb = inst.get_conductance(voltages)

    # Convert from simulator sign convention back to device-level:
    # NMOS_NN.calculate_current returns -id (positive=leaving drain)
    # We want positive id when device is ON.
    if device_type == "nmos":
        id_val = -i_calc
    else:
        id_val = i_calc

    return {"id": id_val, "gm": gm, "gds": gds, "gmb": gmb}


def sweep_id_vgs(
    tech: TechConfig, device_type: str, nfin: float,
    variant: Optional[str] = None, n_points: int = 101,
) -> Dict[str, np.ndarray]:
    """Id-Vgs sweep at Vds = VDD/2. Returns arrays for PyCMG, DirectNet, BSIMAR."""
    vdd = tech.vdd
    vgs_arr = np.linspace(0, vdd, n_points)
    vds = vdd / 2

    cmg = create_pycmg_instance(tech, device_type, nfin, variant)
    dn = create_directnet_instance(tech, device_type, nfin, variant)
    ar = create_bsimar_instance(tech, device_type, nfin, variant)

    data: Dict[str, list] = {
        "vgs": [], "id_cmg": [], "id_dn": [], "id_ar": [],
        "gm_cmg": [], "gm_dn": [], "gm_ar": [],
        "gds_cmg": [], "gds_dn": [], "gds_ar": [],
    }

    for vgs in vgs_arr:
        r_cmg = _eval_pycmg(cmg, vds, vgs)
        if r_cmg is None:
            continue
        r_dn = _eval_nn(dn, vds, vgs, device_type)
        r_ar = _eval_nn(ar, vds, vgs, device_type)

        data["vgs"].append(vgs)
        data["id_cmg"].append(r_cmg["id"])
        data["id_dn"].append(r_dn["id"])
        data["id_ar"].append(r_ar["id"])
        data["gm_cmg"].append(r_cmg["gm"])
        data["gm_dn"].append(r_dn["gm"])
        data["gm_ar"].append(r_ar["gm"])
        data["gds_cmg"].append(r_cmg["gds"])
        data["gds_dn"].append(r_dn["gds"])
        data["gds_ar"].append(r_ar["gds"])

    return {k: np.array(v) for k, v in data.items()}


def sweep_id_vds(
    tech: TechConfig, device_type: str, nfin: float,
    variant: Optional[str] = None, n_points: int = 101,
) -> Dict[str, np.ndarray]:
    """Id-Vds sweep at Vgs = 0.6*VDD (saturation region). Returns arrays."""
    vdd = tech.vdd
    vds_arr = np.linspace(0, vdd, n_points)
    vgs = vdd * 0.6  # Strong inversion

    cmg = create_pycmg_instance(tech, device_type, nfin, variant)
    dn = create_directnet_instance(tech, device_type, nfin, variant)
    ar = create_bsimar_instance(tech, device_type, nfin, variant)

    data: Dict[str, list] = {
        "vds": [], "id_cmg": [], "id_dn": [], "id_ar": [],
        "gds_cmg": [], "gds_dn": [], "gds_ar": [],
    }

    for vds in vds_arr:
        r_cmg = _eval_pycmg(cmg, vds, vgs)
        if r_cmg is None:
            continue
        r_dn = _eval_nn(dn, vds, vgs, device_type)
        r_ar = _eval_nn(ar, vds, vgs, device_type)

        data["vds"].append(vds)
        data["id_cmg"].append(r_cmg["id"])
        data["id_dn"].append(r_dn["id"])
        data["id_ar"].append(r_ar["id"])
        data["gds_cmg"].append(r_cmg["gds"])
        data["gds_dn"].append(r_dn["gds"])
        data["gds_ar"].append(r_ar["gds"])

    return {k: np.array(v) for k, v in data.items()}


# ── Metrics ─────────────────────────────────────────────────────────────────

@dataclass
class SweepMetrics:
    tech: str
    variant: str
    sweep_type: str  # "Id-Vgs" or "Id-Vds"
    n_points: int
    # Id metrics
    nrmse_id_dn: float
    nrmse_id_ar: float
    mre_id_dn: float
    mre_id_ar: float
    # gm metrics (Id-Vgs only)
    nrmse_gm_dn: float = 0.0
    nrmse_gm_ar: float = 0.0
    # gds metrics
    nrmse_gds_dn: float = 0.0
    nrmse_gds_ar: float = 0.0


def compute_metrics(data: Dict[str, np.ndarray], sweep_type: str,
                    tech_name: str, variant: str) -> SweepMetrics:
    """Compute NRMSE and MRE for a sweep dataset."""
    n = len(data.get("vgs", data.get("vds", [])))

    id_cmg = data["id_cmg"]
    m = SweepMetrics(
        tech=tech_name, variant=variant, sweep_type=sweep_type, n_points=n,
        nrmse_id_dn=nrmse(data["id_dn"], id_cmg),
        nrmse_id_ar=nrmse(data["id_ar"], id_cmg),
        mre_id_dn=mre(data["id_dn"], id_cmg),
        mre_id_ar=mre(data["id_ar"], id_cmg),
    )

    if "gm_cmg" in data:
        m.nrmse_gm_dn = nrmse(data["gm_dn"], data["gm_cmg"])
        m.nrmse_gm_ar = nrmse(data["gm_ar"], data["gm_cmg"])

    m.nrmse_gds_dn = nrmse(data["gds_dn"], data["gds_cmg"])
    m.nrmse_gds_ar = nrmse(data["gds_ar"], data["gds_cmg"])

    return m


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_comparison(
    data_vgs: Dict[str, np.ndarray],
    data_vds: Dict[str, np.ndarray],
    metrics_vgs: SweepMetrics,
    metrics_vds: SweepMetrics,
    tech_name: str, variant: str, vdd: float,
) -> None:
    """Generate a 2x3 comparison plot: Id-Vgs (lin+log+gm) and Id-Vds (lin+gds+error)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"NMOS DC: {tech_name} / {variant.upper()} (VDD={vdd}V)\n"
                 f"DirectNet vs BSIMAR vs PyCMG", fontsize=14, fontweight="bold")

    # --- Row 1: Id-Vgs ---
    vgs = data_vgs["vgs"]

    # 1a: Id-Vgs linear
    ax = axes[0, 0]
    ax.plot(vgs, data_vgs["id_cmg"] * 1e6, "k-", lw=2, label="PyCMG (truth)")
    ax.plot(vgs, data_vgs["id_dn"] * 1e6, "b--", lw=1.5, label=f"DirectNet ({metrics_vgs.nrmse_id_dn:.2f}%)")
    ax.plot(vgs, data_vgs["id_ar"] * 1e6, "r-.", lw=1.5, label=f"BSIMAR ({metrics_vgs.nrmse_id_ar:.2f}%)")
    ax.set_xlabel("Vgs (V)")
    ax.set_ylabel("Id (uA)")
    ax.set_title(f"Id-Vgs (Vds={vdd/2:.2f}V)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 1b: Id-Vgs log scale
    ax = axes[0, 1]
    id_cmg_abs = np.abs(data_vgs["id_cmg"])
    id_dn_abs = np.abs(data_vgs["id_dn"])
    id_ar_abs = np.abs(data_vgs["id_ar"])
    # Clamp floor for log
    floor = max(id_cmg_abs[id_cmg_abs > 0].min() * 0.1, 1e-15) if np.any(id_cmg_abs > 0) else 1e-15
    ax.semilogy(vgs, np.maximum(id_cmg_abs, floor), "k-", lw=2, label="PyCMG")
    ax.semilogy(vgs, np.maximum(id_dn_abs, floor), "b--", lw=1.5, label="DirectNet")
    ax.semilogy(vgs, np.maximum(id_ar_abs, floor), "r-.", lw=1.5, label="BSIMAR")
    ax.set_xlabel("Vgs (V)")
    ax.set_ylabel("|Id| (A)")
    ax.set_title("Id-Vgs (log scale)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 1c: gm-Vgs
    ax = axes[0, 2]
    ax.plot(vgs, data_vgs["gm_cmg"] * 1e3, "k-", lw=2, label="PyCMG")
    ax.plot(vgs, data_vgs["gm_dn"] * 1e3, "b--", lw=1.5,
            label=f"DirectNet ({metrics_vgs.nrmse_gm_dn:.1f}%)")
    ax.plot(vgs, data_vgs["gm_ar"] * 1e3, "r-.", lw=1.5,
            label=f"BSIMAR ({metrics_vgs.nrmse_gm_ar:.1f}%)")
    ax.set_xlabel("Vgs (V)")
    ax.set_ylabel("gm (mS)")
    ax.set_title("gm-Vgs (autograd)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Row 2: Id-Vds ---
    vds = data_vds["vds"]

    # 2a: Id-Vds linear
    ax = axes[1, 0]
    ax.plot(vds, data_vds["id_cmg"] * 1e6, "k-", lw=2, label="PyCMG (truth)")
    ax.plot(vds, data_vds["id_dn"] * 1e6, "b--", lw=1.5,
            label=f"DirectNet ({metrics_vds.nrmse_id_dn:.2f}%)")
    ax.plot(vds, data_vds["id_ar"] * 1e6, "r-.", lw=1.5,
            label=f"BSIMAR ({metrics_vds.nrmse_id_ar:.2f}%)")
    ax.set_xlabel("Vds (V)")
    ax.set_ylabel("Id (uA)")
    ax.set_title(f"Id-Vds (Vgs={vdd*0.6:.2f}V)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2b: gds-Vds
    ax = axes[1, 1]
    ax.plot(vds, data_vds["gds_cmg"] * 1e3, "k-", lw=2, label="PyCMG")
    ax.plot(vds, data_vds["gds_dn"] * 1e3, "b--", lw=1.5,
            label=f"DirectNet ({metrics_vds.nrmse_gds_dn:.1f}%)")
    ax.plot(vds, data_vds["gds_ar"] * 1e3, "r-.", lw=1.5,
            label=f"BSIMAR ({metrics_vds.nrmse_gds_ar:.1f}%)")
    ax.set_xlabel("Vds (V)")
    ax.set_ylabel("gds (mS)")
    ax.set_title("gds-Vds (autograd)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2c: Id error percentage
    ax = axes[1, 2]
    # Relative error vs PyCMG (only where |id_cmg| > 1% of peak)
    peak = max(np.abs(data_vgs["id_cmg"]).max(), 1e-15)
    mask_vgs = np.abs(data_vgs["id_cmg"]) > 0.01 * peak
    if mask_vgs.sum() > 2:
        rel_err_dn_vgs = np.abs(data_vgs["id_dn"][mask_vgs] - data_vgs["id_cmg"][mask_vgs]) / np.abs(data_vgs["id_cmg"][mask_vgs]) * 100
        rel_err_ar_vgs = np.abs(data_vgs["id_ar"][mask_vgs] - data_vgs["id_cmg"][mask_vgs]) / np.abs(data_vgs["id_cmg"][mask_vgs]) * 100
        ax.plot(vgs[mask_vgs], rel_err_dn_vgs, "b--", lw=1.5, label="DirectNet (Vgs sweep)")
        ax.plot(vgs[mask_vgs], rel_err_ar_vgs, "r-.", lw=1.5, label="BSIMAR (Vgs sweep)")

    peak_vds = max(np.abs(data_vds["id_cmg"]).max(), 1e-15)
    mask_vds = np.abs(data_vds["id_cmg"]) > 0.01 * peak_vds
    if mask_vds.sum() > 2:
        rel_err_dn_vds = np.abs(data_vds["id_dn"][mask_vds] - data_vds["id_cmg"][mask_vds]) / np.abs(data_vds["id_cmg"][mask_vds]) * 100
        rel_err_ar_vds = np.abs(data_vds["id_ar"][mask_vds] - data_vds["id_cmg"][mask_vds]) / np.abs(data_vds["id_cmg"][mask_vds]) * 100
        ax.plot(vds[mask_vds], rel_err_dn_vds, "b:", lw=1.5, alpha=0.6, label="DirectNet (Vds sweep)")
        ax.plot(vds[mask_vds], rel_err_ar_vds, "r:", lw=1.5, alpha=0.6, label="BSIMAR (Vds sweep)")

    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Relative Error (%)")
    ax.set_title("Id Relative Error (|Id|>1% peak)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fname = RESULTS_DIR / f"{tech_name}_{variant}_nmos_dc_comparison.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot saved: {fname.name}")


# ── Main orchestration ──────────────────────────────────────────────────────

def run_all() -> List[SweepMetrics]:
    """Run all NMOS DC comparisons across technologies."""
    all_metrics: List[SweepMetrics] = []
    device_type = "nmos"
    nfin = 10.0

    # Check checkpoint availability
    dn_ckpt = directnet_checkpoint(device_type)
    ar_ckpt = _resolve_bsimar_checkpoint(device_type)
    if not dn_ckpt.exists():
        print(f"ERROR: DirectNet checkpoint not found: {dn_ckpt}")
        return []
    if not ar_ckpt.exists():
        print(f"ERROR: BSIMAR checkpoint not found: {ar_ckpt}")
        return []
    print(f"DirectNet checkpoint: {dn_ckpt.name}")
    print(f"BSIMAR checkpoint:   {ar_ckpt.name}")

    for tech_key, tech in TECH_CONFIGS.items():
        # Test default variant only (for speed; can expand)
        for variant in [tech.default_variant]:
            header = f"{tech.name} / {variant.upper()} (VDD={tech.vdd}V, NFIN={nfin:.0f})"
            print(f"\n{'='*70}")
            print(f"  {header}")
            print(f"{'='*70}")

            # --- Id-Vgs sweep ---
            t0 = time.time()
            try:
                data_vgs = sweep_id_vgs(tech, device_type, nfin, variant)
            except Exception as e:
                print(f"  Id-Vgs sweep FAILED: {e}")
                continue
            dt_vgs = time.time() - t0
            m_vgs = compute_metrics(data_vgs, "Id-Vgs", tech.name, variant)
            all_metrics.append(m_vgs)

            print(f"  Id-Vgs (Vds={tech.vdd/2:.2f}V, {m_vgs.n_points} pts, {dt_vgs:.1f}s):")
            print(f"    Id  NRMSE: DirectNet={m_vgs.nrmse_id_dn:6.3f}%  BSIMAR={m_vgs.nrmse_id_ar:6.3f}%")
            print(f"    Id  MRE:   DirectNet={m_vgs.mre_id_dn:6.3f}%  BSIMAR={m_vgs.mre_id_ar:6.3f}%")
            print(f"    gm  NRMSE: DirectNet={m_vgs.nrmse_gm_dn:6.3f}%  BSIMAR={m_vgs.nrmse_gm_ar:6.3f}%")
            print(f"    gds NRMSE: DirectNet={m_vgs.nrmse_gds_dn:6.3f}%  BSIMAR={m_vgs.nrmse_gds_ar:6.3f}%")

            # --- Id-Vds sweep ---
            t0 = time.time()
            try:
                data_vds = sweep_id_vds(tech, device_type, nfin, variant)
            except Exception as e:
                print(f"  Id-Vds sweep FAILED: {e}")
                continue
            dt_vds = time.time() - t0
            m_vds = compute_metrics(data_vds, "Id-Vds", tech.name, variant)
            all_metrics.append(m_vds)

            print(f"  Id-Vds (Vgs={tech.vdd*0.6:.2f}V, {m_vds.n_points} pts, {dt_vds:.1f}s):")
            print(f"    Id  NRMSE: DirectNet={m_vds.nrmse_id_dn:6.3f}%  BSIMAR={m_vds.nrmse_id_ar:6.3f}%")
            print(f"    Id  MRE:   DirectNet={m_vds.mre_id_dn:6.3f}%  BSIMAR={m_vds.mre_id_ar:6.3f}%")
            print(f"    gds NRMSE: DirectNet={m_vds.nrmse_gds_dn:6.3f}%  BSIMAR={m_vds.nrmse_gds_ar:6.3f}%")

            # --- Plot ---
            try:
                plot_comparison(data_vgs, data_vds, m_vgs, m_vds,
                                tech.name, variant, tech.vdd)
            except Exception as e:
                print(f"    Plot generation failed: {e}")

    return all_metrics


def print_summary(metrics: List[SweepMetrics]) -> None:
    """Print summary tables."""
    print(f"\n{'='*95}")
    print(f"  NMOS DC Comparison Summary: DirectNet vs BSIMAR vs PyCMG (ground truth)")
    print(f"{'='*95}")

    # Table 1: Id NRMSE
    print(f"\n  --- Id NRMSE (%) ---")
    print(f"  {'Tech':<8s} {'Variant':<6s} {'Sweep':<8s} {'DirectNet':>10s} {'BSIMAR':>10s} {'Winner':>8s}")
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for m in metrics:
        winner = "DN" if m.nrmse_id_dn < m.nrmse_id_ar else "AR"
        print(f"  {m.tech:<8s} {m.variant:<6s} {m.sweep_type:<8s} "
              f"{m.nrmse_id_dn:10.3f} {m.nrmse_id_ar:10.3f} {winner:>8s}")

    # Table 2: Id MRE
    print(f"\n  --- Id MRE (%) ---")
    print(f"  {'Tech':<8s} {'Variant':<6s} {'Sweep':<8s} {'DirectNet':>10s} {'BSIMAR':>10s} {'Winner':>8s}")
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for m in metrics:
        winner = "DN" if m.mre_id_dn < m.mre_id_ar else "AR"
        print(f"  {m.tech:<8s} {m.variant:<6s} {m.sweep_type:<8s} "
              f"{m.mre_id_dn:10.3f} {m.mre_id_ar:10.3f} {winner:>8s}")

    # Table 3: Conductance NRMSE
    gm_metrics = [m for m in metrics if m.sweep_type == "Id-Vgs"]
    if gm_metrics:
        print(f"\n  --- gm NRMSE (%) [Id-Vgs sweep] ---")
        print(f"  {'Tech':<8s} {'Variant':<6s} {'DirectNet':>10s} {'BSIMAR':>10s} {'Winner':>8s}")
        print(f"  {'-'*8} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")
        for m in gm_metrics:
            winner = "DN" if m.nrmse_gm_dn < m.nrmse_gm_ar else "AR"
            print(f"  {m.tech:<8s} {m.variant:<6s} "
                  f"{m.nrmse_gm_dn:10.3f} {m.nrmse_gm_ar:10.3f} {winner:>8s}")

    print(f"\n  --- gds NRMSE (%) ---")
    print(f"  {'Tech':<8s} {'Variant':<6s} {'Sweep':<8s} {'DirectNet':>10s} {'BSIMAR':>10s} {'Winner':>8s}")
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for m in metrics:
        winner = "DN" if m.nrmse_gds_dn < m.nrmse_gds_ar else "AR"
        print(f"  {m.tech:<8s} {m.variant:<6s} {m.sweep_type:<8s} "
              f"{m.nrmse_gds_dn:10.3f} {m.nrmse_gds_ar:10.3f} {winner:>8s}")

    # Aggregate
    dn_id_avg = np.mean([m.nrmse_id_dn for m in metrics])
    ar_id_avg = np.mean([m.nrmse_id_ar for m in metrics])
    dn_mre_avg = np.mean([m.mre_id_dn for m in metrics])
    ar_mre_avg = np.mean([m.mre_id_ar for m in metrics])
    dn_gm_avg = np.mean([m.nrmse_gm_dn for m in gm_metrics]) if gm_metrics else 0
    ar_gm_avg = np.mean([m.nrmse_gm_ar for m in gm_metrics]) if gm_metrics else 0
    dn_gds_avg = np.mean([m.nrmse_gds_dn for m in metrics])
    ar_gds_avg = np.mean([m.nrmse_gds_ar for m in metrics])

    print(f"\n  --- Averages Across All Techs ---")
    print(f"  {'Metric':<20s} {'DirectNet':>10s} {'BSIMAR':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10}")
    print(f"  {'Id NRMSE (%)':<20s} {dn_id_avg:10.3f} {ar_id_avg:10.3f}")
    print(f"  {'Id MRE (%)':<20s} {dn_mre_avg:10.3f} {ar_mre_avg:10.3f}")
    print(f"  {'gm NRMSE (%)':<20s} {dn_gm_avg:10.3f} {ar_gm_avg:10.3f}")
    print(f"  {'gds NRMSE (%)':<20s} {dn_gds_avg:10.3f} {ar_gds_avg:10.3f}")

    dn_wins = sum(1 for m in metrics if m.nrmse_id_dn < m.nrmse_id_ar)
    ar_wins = len(metrics) - dn_wins
    print(f"\n  Id NRMSE wins: DirectNet={dn_wins}, BSIMAR={ar_wins} (out of {len(metrics)} sweeps)")


if __name__ == "__main__":
    print("=" * 70)
    print("  NMOS DC Comparison: DirectNet vs BSIMAR vs PyCMG")
    print("  Device: NMOS, NFIN=10, 5 technologies, default variants")
    print("=" * 70)

    metrics = run_all()
    if metrics:
        print_summary(metrics)
    else:
        print("\nNo tests completed. Check checkpoint availability.")
