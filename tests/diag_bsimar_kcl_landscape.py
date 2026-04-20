"""Diagnose BSIMAR transient OP failure: sweep V(out) and plot KCL residual.

The hypothesis: the BSIMAR Transformer has a spurious equilibrium near
V(out)=+4.4V (out of the training Vds range) that the DCSolver
source-stepping locks onto, while the VTC test's manual clamp prevents
the solver from ever reaching that region.

Sweeps V(out) ∈ [-1, +6] V at fixed Vin=0V, Vdd=0.65V (TSMC5 SVT inverter).
For each V(out):
  - compute I_nmos, I_pmos via the same model code path the simulator uses
  - residual f(Vout) = I_nmos - I_pmos (drain-leaving sign convention)
Plot residual + zeros → equilibrium points. Compare against PyCMG ground truth.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("BSIMAR_PREFIX", "v4_probe_signfix")

from bsimar.config import tech_variant_to_code
# Reuse verify script's instance factories so we exactly match the simulator's
# code path (including any latent normalization or correction differences).
from tests.verify_bsimar_v4_inverter import (
    TSMC5_SVT,
    create_pycmg_instance,
    create_bsimar_instance,
    create_directnet_instance,
)

TECH = "tsmc5"
VARIANT = "svt"
TECH_CODE = tech_variant_to_code(TECH, VARIANT)
TECH_CFG = TSMC5_SVT
VDD = TECH_CFG.vdd


def kcl_residual_pycmg(nmos, pmos, vin: float, vout: float, vdd: float) -> tuple:
    """KCL at output node using PyCMG. Returns (residual, i_nmos, i_pmos)."""
    try:
        rn = nmos.eval_dc({"d": vout, "g": vin, "s": 0.0, "e": 0.0})
        rp = pmos.eval_dc({"d": vout, "g": vin, "s": vdd, "e": vdd})
    except Exception:
        return np.nan, np.nan, np.nan
    # PyCMG id is the channel current; use rn["id"] + rp["id"] = 0 at eq.
    return rn["id"] + rp["id"], rn["id"], rp["id"]


def kcl_residual_nn(nmos, pmos, vin: float, vout: float, vdd: float) -> tuple:
    """KCL using NN compact models. Returns (residual, i_nmos, i_pmos)."""
    nv = {"drain": vout, "gate": vin, "source": 0.0, "bulk": 0.0}
    pv = {"drain": vout, "gate": vin, "source": vdd, "bulk": vdd}
    nmos.clear_cache()
    pmos.clear_cache()
    i_n = nmos.calculate_current(nv)  # PMOS frame: positive = into drain
    i_p = pmos.calculate_current(pv)
    # Following the VTC code's convention: f = i_n - i_p (KCL at out node).
    return i_n - i_p, i_n, i_p


def main() -> None:
    print(f"=== BSIMAR KCL landscape diagnostic ===")
    print(f"  Tech: {TECH} {VARIANT}, VDD={VDD}V, "
          f"L_n={TECH_CFG.l_nmos*1e9:.0f}nm, L_p={TECH_CFG.l_pmos*1e9:.0f}nm, "
          f"NFIN={TECH_CFG.nfin}")
    print(f"  Tech code: {TECH_CODE}, BSIMAR_PREFIX={os.environ['BSIMAR_PREFIX']}")
    print()

    print("Loading PyCMG...")
    pycmg_n = create_pycmg_instance(TECH_CFG, "nmos")
    pycmg_p = create_pycmg_instance(TECH_CFG, "pmos")
    print("Loading BSIMAR probe...")
    bs_n = create_bsimar_instance(TECH_CFG, "nmos")
    bs_p = create_bsimar_instance(TECH_CFG, "pmos")
    print("Loading DirectNet probe...")
    dn_n = create_directnet_instance(TECH_CFG, "nmos")
    dn_p = create_directnet_instance(TECH_CFG, "pmos")

    # Sweep V(out) over a wide range to find ALL equilibria.
    vout_sweep = np.linspace(-1.0, 6.0, 281)  # 0.025 V resolution

    for vin_label, vin in [("rail-low (Vin=0)", 0.0),
                            ("rail-high (Vin=VDD)", VDD)]:
        print(f"\n--- Vin = {vin}V [{vin_label}] ---")

        f_pyc = np.full_like(vout_sweep, np.nan)
        i_n_pyc = np.full_like(vout_sweep, np.nan)
        i_p_pyc = np.full_like(vout_sweep, np.nan)
        f_bs = np.full_like(vout_sweep, np.nan)
        i_n_bs = np.full_like(vout_sweep, np.nan)
        i_p_bs = np.full_like(vout_sweep, np.nan)
        f_dn = np.full_like(vout_sweep, np.nan)
        i_n_dn = np.full_like(vout_sweep, np.nan)
        i_p_dn = np.full_like(vout_sweep, np.nan)

        for k, vo in enumerate(vout_sweep):
            f_pyc[k], i_n_pyc[k], i_p_pyc[k] = kcl_residual_pycmg(pycmg_n, pycmg_p, vin, vo, VDD)
            f_bs[k], i_n_bs[k], i_p_bs[k] = kcl_residual_nn(bs_n, bs_p, vin, vo, VDD)
            f_dn[k], i_n_dn[k], i_p_dn[k] = kcl_residual_nn(dn_n, dn_p, vin, vo, VDD)

        # Find zero crossings of each residual
        def zero_crossings(f, vouts):
            zs = []
            for k in range(len(f) - 1):
                if np.isnan(f[k]) or np.isnan(f[k + 1]):
                    continue
                if f[k] * f[k + 1] <= 0 and f[k] != f[k + 1]:
                    # linear interp
                    z = vouts[k] - f[k] * (vouts[k + 1] - vouts[k]) / (f[k + 1] - f[k])
                    zs.append(z)
            return zs

        z_pyc = zero_crossings(f_pyc, vout_sweep)
        z_bs = zero_crossings(f_bs, vout_sweep)
        z_dn = zero_crossings(f_dn, vout_sweep)
        print(f"  PyCMG zeros (eq Vout):    {[f'{z:.3f}V' for z in z_pyc]}")
        print(f"  BSIMAR zeros (eq Vout):   {[f'{z:.3f}V' for z in z_bs]}")
        print(f"  DirectNet zeros (eq Vout):{[f'{z:.3f}V' for z in z_dn]}")

        # Plot residual + per-device currents
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        axes[0].plot(vout_sweep, f_pyc * 1e6, "b-", lw=1.5, label="PyCMG")
        axes[0].plot(vout_sweep, f_bs * 1e6, "r--", lw=1.5, label="BSIMAR probe")
        axes[0].plot(vout_sweep, f_dn * 1e6, "g:", lw=1.5, label="DirectNet probe")
        axes[0].axhline(0, color="k", lw=0.5)
        axes[0].axvspan(0, VDD, alpha=0.10, color="green", label="VDD range")
        axes[0].set_ylabel("KCL residual f(Vout) [µA]")
        axes[0].set_title(f"Inverter KCL landscape | Vin={vin:.2f}V, VDD={VDD}V | TSMC5 SVT")
        axes[0].legend(loc="best", fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[0].set_ylim(-100, 100)

        axes[1].plot(vout_sweep, i_n_pyc * 1e6, "b-", lw=1.5, label="PyCMG NMOS Id")
        axes[1].plot(vout_sweep, i_n_bs * 1e6, "r--", lw=1.5, label="BSIMAR NMOS Id")
        axes[1].plot(vout_sweep, i_n_dn * 1e6, "g:", lw=1.5, label="DirectNet NMOS Id")
        axes[1].axhline(0, color="k", lw=0.5)
        axes[1].axvspan(0, VDD, alpha=0.10, color="green")
        axes[1].set_ylabel("NMOS Id [µA]")
        axes[1].legend(loc="best", fontsize=8)
        axes[1].grid(alpha=0.3)

        axes[2].plot(vout_sweep, i_p_pyc * 1e6, "b-", lw=1.5, label="PyCMG PMOS Id")
        axes[2].plot(vout_sweep, i_p_bs * 1e6, "r--", lw=1.5, label="BSIMAR PMOS Id")
        axes[2].plot(vout_sweep, i_p_dn * 1e6, "g:", lw=1.5, label="DirectNet PMOS Id")
        axes[2].axhline(0, color="k", lw=0.5)
        axes[2].axvspan(0, VDD, alpha=0.10, color="green")
        axes[2].set_xlabel("Vout [V]")
        axes[2].set_ylabel("PMOS Id [µA]")
        axes[2].legend(loc="best", fontsize=8)
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        out_dir = PROJECT_ROOT / "results" / "diag_bsimar_kcl"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"kcl_landscape_vin{int(vin*1000):04d}mV.png"
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
