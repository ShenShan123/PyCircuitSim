"""Generate training data by sweeping PyCMG BSIM-CMG across bias points.

Supports multi-variant data generation: for each technology, sweeps all
device variants (e.g., SVT + LVT) and includes PHIG as an input feature.

Usage:
    conda run -n pycircuitsim python -m nn_model.data.generate [--device nmos|pmos] [--tech asap7]
    conda run -n pycircuitsim python -m nn_model.data.generate --device both --tech all
"""

import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))

from pycmg import Model, Instance
from nn_model.config import (
    OSDI_PATH, TechConfig, VariantConfig, OUTPUT_COLUMNS, DATA_DIR,
    TECH_CONFIGS,
)


def create_pycmg_instance(
    tech: TechConfig,
    device_type: str,
    nfin: float,
    variant: Optional[str] = None,
) -> Instance:
    """Create a PyCMG Instance for the given tech/device/geometry/variant.

    Args:
        tech: Technology configuration.
        device_type: 'nmos' or 'pmos'.
        nfin: Number of fins.
        variant: Device variant name (e.g., 'svt', 'lvt'). If None, uses default.

    Returns:
        PyCMG Instance ready for eval_dc().
    """
    model_name = tech.get_model_name(device_type, variant)
    modelcard_path = tech.get_modelcard_path(device_type, variant)

    model = Model(
        osdi_path=OSDI_PATH,
        modelcard_path=modelcard_path,
        model_name=model_name,
        model_card_name=model_name,
    )

    L = tech.get_L(device_type)
    inst_params = {"L": L, "NFIN": float(nfin)}
    return Instance(model=model, params=inst_params, temperature=tech.temperature)


def generate_voltage_grid(
    vdd: float,
    device_type: str = "nmos",
    n_uniform: int = 71,
    n_dense_vth: int = 20,
    v_margin: float = -1.0,  # -1 means auto = VDD
    vth_approx: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Vgs and Vds sweep points with dense sampling near Vth.

    For NMOS (Vs=0): Vg in [-margin, VDD+margin], Vd in [-margin, VDD+margin]
    For PMOS (Vs=0 in source-relative frame): Vg in [-(VDD+margin), margin],
        Vd in [-(VDD+margin), margin]. PMOS turns ON when Vg < -|Vtp|.

    The NN always operates in a source-relative frame (Vs=0). For PMOS in a
    circuit, mosfet_nn.py shifts voltages by -Vs before feeding to the NN.

    Args:
        vdd: Supply voltage.
        device_type: 'nmos' or 'pmos'.
        n_uniform: Number of uniform grid points per axis.
        n_dense_vth: Extra dense points near Vth (for subthreshold transition).
        v_margin: Extend sweep beyond operating range by this margin.
        vth_approx: Approximate threshold voltage magnitude.

    Returns:
        Tuple of (vg_points, vd_points) as sorted 1D arrays.
    """
    # Auto margin: use VDD to cover NR overshoot
    if v_margin < 0:
        v_margin = vdd

    if device_type == "nmos":
        # NMOS: Vg > Vth to turn on, Vd > 0 typical
        # With margin=VDD: range is [-VDD, 2*VDD] covering NR overshoot
        v_min = -v_margin
        v_max = vdd + v_margin
        vth_center = vth_approx
    else:
        # PMOS in source-relative frame (Vs=0):
        # Vg < 0 to turn on, Vd < 0 typically
        # With margin=VDD: range is [-2*VDD, VDD] covering NR overshoot
        v_min = -(vdd + v_margin)
        v_max = v_margin
        vth_center = -vth_approx

    # Uniform grid for Vg
    vg_uniform = np.linspace(v_min, v_max, n_uniform)

    # Dense points near Vth
    vth_range = 0.1
    vg_dense = np.linspace(vth_center - vth_range, vth_center + vth_range, n_dense_vth)

    vg_all = np.unique(np.concatenate([vg_uniform, vg_dense]))

    # Uniform grid for Vd
    vd_all = np.linspace(v_min, v_max, n_uniform)

    return vg_all, vd_all


def eval_single_point(
    inst: Instance,
    vd: float,
    vg: float,
    vs: float = 0.0,
    vb: float = 0.0,
) -> Optional[Dict[str, float]]:
    """Evaluate PyCMG at a single bias point.

    Returns:
        Dict with 13 output columns, or None if evaluation fails.
    """
    try:
        result = inst.eval_dc({"d": vd, "g": vg, "s": vs, "e": vb})
        return {
            "id": result["id"],
            "gm": result["gm"],
            "gds": result["gds"],
            "gmb": result["gmb"],
            "qg": result["qg"],
            "qd": result["qd"],
            "qs": result["qs"],
            "qb": result["qb"],
            "cgg": result["cgg"],
            "cgd": result["cgd"],
            "cgs": result["cgs"],
            "cdg": result["cdg"],
            "cdd": result["cdd"],
        }
    except Exception as e:
        print(f"  WARNING: eval_dc failed at Vd={vd:.3f} Vg={vg:.3f}: {e}")
        return None


def generate_dataset(
    tech: TechConfig,
    device_type: str,
    variant_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Generate full training dataset for one device type across NFIN values and variants.

    Sweeps Vgs x Vds x NFIN x Variant with special case augmentation:
    - Zero-bias anchors (all V=0)
    - Deep cutoff (Vg far below Vth)
    - Dense subthreshold sampling (near Vth)

    Args:
        tech: Technology configuration.
        device_type: 'nmos' or 'pmos'.
        variant_names: List of variant names to include (default: all variants).
        verbose: Print progress.

    Returns:
        Dict with keys: 'inputs' (N,4), 'geometry' (N,3), 'outputs' (N,13),
        'metadata' containing tech/device info.
    """
    vdd = tech.vdd
    vgs_points, vds_points = generate_voltage_grid(vdd, device_type=device_type)

    # Determine which variants to generate data for
    if variant_names is None:
        variants_to_use = list(tech.variants.items())
    else:
        variants_to_use = [(vn, tech.variants[vn]) for vn in variant_names
                           if vn in tech.variants]

    if not variants_to_use:
        raise ValueError(f"No valid variants found for {tech.name}. "
                         f"Available: {list(tech.variants.keys())}")

    all_inputs: List[np.ndarray] = []    # (Vd, Vg, Vs, Vb)
    all_geometry: List[np.ndarray] = []  # (NFIN, T, PHIG)
    all_outputs: List[np.ndarray] = []   # 13 output columns

    total_points = 0
    failed_points = 0

    for variant_name, variant_cfg in variants_to_use:
        phig = variant_cfg.get_phig(device_type)
        if verbose:
            print(f"\n--- Variant: {variant_name} (PHIG={phig:.4f}) ---")

        for nfin in tech.nfin_values:
            if verbose:
                print(f"\n  NFIN={nfin}: Creating PyCMG instance "
                      f"(model={variant_cfg.get_model_name(device_type)})...")

            inst = create_pycmg_instance(tech, device_type, nfin, variant_name)
            nfin_points = 0
            t0 = time.time()

            # --- Main grid sweep: Vgs x Vds ---
            for vg in vgs_points:
                for vd in vds_points:
                    vs = 0.0
                    vb = 0.0

                    result = eval_single_point(inst, vd, vg, vs, vb)
                    if result is None:
                        failed_points += 1
                        continue

                    all_inputs.append(np.array([vd, vg, vs, vb]))
                    all_geometry.append(np.array([float(nfin), tech.temperature, phig]))
                    all_outputs.append(np.array([result[k] for k in OUTPUT_COLUMNS]))
                    nfin_points += 1

            # --- Special case: zero-bias anchor ---
            result = eval_single_point(inst, 0.0, 0.0, 0.0, 0.0)
            if result is not None:
                for _ in range(3):
                    all_inputs.append(np.array([0.0, 0.0, 0.0, 0.0]))
                    all_geometry.append(np.array([float(nfin), tech.temperature, phig]))
                    all_outputs.append(np.array([result[k] for k in OUTPUT_COLUMNS]))
                    nfin_points += 1

            # --- Special case: deep cutoff ---
            if device_type == "nmos":
                cutoff_vg_values = [-0.1, -0.05, 0.0]
                cutoff_vd_values = [0.0, vdd / 2, vdd]
            else:
                cutoff_vg_values = [0.0, 0.05, 0.1]
                cutoff_vd_values = [0.0, -vdd / 2, -vdd]
            for vg_cutoff in cutoff_vg_values:
                for vd in cutoff_vd_values:
                    result = eval_single_point(inst, vd, vg_cutoff, 0.0, 0.0)
                    if result is not None:
                        all_inputs.append(np.array([vd, vg_cutoff, 0.0, 0.0]))
                        all_geometry.append(np.array([float(nfin), tech.temperature, phig]))
                        all_outputs.append(np.array([result[k] for k in OUTPUT_COLUMNS]))
                        nfin_points += 1

            elapsed = time.time() - t0
            total_points += nfin_points
            if verbose:
                print(f"    Generated {nfin_points} points in {elapsed:.1f}s "
                      f"({nfin_points / elapsed:.0f} pts/s)")

    # Stack into arrays
    inputs = np.array(all_inputs, dtype=np.float64)     # (N, 4)
    geometry = np.array(all_geometry, dtype=np.float64)  # (N, 3) — [NFIN, T, PHIG]
    outputs = np.array(all_outputs, dtype=np.float64)    # (N, 13)

    if verbose:
        print(f"\n  Total: {total_points} points, {failed_points} failures")
        print(f"  Variants: {[v[0] for v in variants_to_use]}")
        print(f"  Input shape:  {inputs.shape}")
        print(f"  Geometry shape: {geometry.shape}")
        print(f"  Output shape: {outputs.shape}")

        # Print data ranges
        print(f"\n  Output ranges (min / max):")
        for i, name in enumerate(OUTPUT_COLUMNS):
            col = outputs[:, i]
            print(f"    {name:>6s}: {col.min():+.4e} / {col.max():+.4e}")

        # Print PHIG range
        phig_vals = np.unique(geometry[:, 2])
        print(f"\n  PHIG values in dataset: {phig_vals}")

    metadata = {
        "tech_name": tech.name,
        "device_type": device_type,
        "vdd": vdd,
        "L": tech.L,
        "nfin_values": np.array(tech.nfin_values),
        "temperature": tech.temperature,
        "output_columns": OUTPUT_COLUMNS,
        "variants": np.array([v[0] for v in variants_to_use]),
    }

    return {
        "inputs": inputs,
        "geometry": geometry,
        "outputs": outputs,
        "metadata": metadata,
    }


def save_dataset(data: Dict[str, np.ndarray], output_path: Path) -> None:
    """Save dataset to .npz file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # np.savez doesn't handle nested dicts well; flatten metadata
    save_dict = {
        "inputs": data["inputs"],
        "geometry": data["geometry"],
        "outputs": data["outputs"],
    }
    # Save metadata as separate keys with prefix
    for k, v in data["metadata"].items():
        if isinstance(v, (list, np.ndarray)):
            save_dict[f"meta_{k}"] = np.array(v)
        elif isinstance(v, str):
            save_dict[f"meta_{k}"] = np.array(v)
        else:
            save_dict[f"meta_{k}"] = np.array(v)

    np.savez(output_path, **save_dict)
    print(f"\n  Saved to {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NN training data from PyCMG")
    parser.add_argument("--device", choices=["nmos", "pmos", "both"], default="nmos",
                        help="Device type to generate data for (default: nmos)")
    parser.add_argument("--tech", choices=list(TECH_CONFIGS.keys()) + ["all"],
                        default="asap7",
                        help="Technology to use (default: asap7)")
    parser.add_argument("--variants", type=str, default="all",
                        help="Comma-separated variant names (default: all)")
    args = parser.parse_args()

    # Select technologies
    if args.tech == "all":
        techs = list(TECH_CONFIGS.values())
    else:
        techs = [TECH_CONFIGS[args.tech]]

    devices = ["nmos", "pmos"] if args.device == "both" else [args.device]

    # Parse variant names
    if args.variants == "all":
        variant_names = None  # generate_dataset will use all
    else:
        variant_names = [v.strip() for v in args.variants.split(",")]

    for tech in techs:
        for device_type in devices:
            L = tech.get_L(device_type)
            avail_variants = list(tech.variants.keys())
            use_variants = variant_names if variant_names else avail_variants
            print(f"\n{'='*60}")
            print(f"Generating {device_type.upper()} data for {tech.name}")
            print(f"  VDD={tech.vdd}V, L={L*1e9:.0f}nm")
            print(f"  NFIN values: {tech.nfin_values}")
            print(f"  Variants: {use_variants}")
            print(f"  Temperature: {tech.temperature}K")
            print(f"{'='*60}")

            data = generate_dataset(tech, device_type,
                                    variant_names=variant_names,
                                    verbose=True)

            output_path = DATA_DIR / f"{tech.name.lower()}_{device_type}.npz"
            save_dataset(data, output_path)


if __name__ == "__main__":
    main()
