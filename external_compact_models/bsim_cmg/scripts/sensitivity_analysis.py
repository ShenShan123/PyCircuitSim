#!/usr/bin/env python3
"""Identify top process parameters by OAT sensitivity analysis.

Perturbs each BSIM-CMG model parameter independently and ranks them by
influence on I-V, Q-V, and C-V characteristics.  The output guides which
parameters to include in process variation modeling for NN compact models.

Usage examples::

    # Quick: ASAP7 NMOS with defaults (top 9, 5% perturbation)
    python scripts/sensitivity_analysis.py \\
        --osdi build/osdi/bsimcmg.osdi \\
        --tech ASAP7 --device nmos_rvt

    # Custom perturbation and top-N
    python scripts/sensitivity_analysis.py \\
        --osdi build/osdi/bsimcmg.osdi \\
        --tech TSMC7 --device nmos_svt \\
        --delta 0.10 --top-n 15

    # Save full results to CSV
    python scripts/sensitivity_analysis.py \\
        --osdi build/osdi/bsimcmg.osdi \\
        --tech ASAP7 --device nmos_rvt \\
        --output sensitivity_results.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List

# Allow importing pycmg from project root when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pycmg.sensitivity import (
    OUTPUT_CATEGORIES,
    SensitivityResult,
    compute_sensitivity,
    format_sensitivity_table,
)
from pycmg.sweep import OUTPUT_KEYS
from pycmg.tech import get_tech_config, list_techs, resolve_modelcard


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="sensitivity_analysis.py",
        description="Identify top process parameters by OAT sensitivity analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # ASAP7 NMOS defaults\n"
            "  %(prog)s --osdi build/osdi/bsimcmg.osdi "
            "--tech ASAP7 --device nmos_rvt\n"
            "\n"
            "  # TSMC7 NMOS with custom delta and top-N\n"
            "  %(prog)s --osdi build/osdi/bsimcmg.osdi "
            "--tech TSMC7 --device nmos_svt --delta 0.10 --top-n 15\n"
        ),
    )

    parser.add_argument(
        "--osdi",
        metavar="PATH",
        required=True,
        help="Path to compiled OSDI binary (e.g., build/osdi/bsimcmg.osdi)",
    )
    parser.add_argument(
        "--tech",
        required=True,
        metavar="TECH",
        help=f"Technology name. Available: {', '.join(list_techs())}",
    )
    parser.add_argument(
        "--device",
        required=True,
        metavar="DEV",
        help="Device name (e.g., nmos_rvt, pmos_svt)",
    )

    # Geometry
    parser.add_argument(
        "--l",
        type=float,
        default=None,
        metavar="M",
        help="Gate length in meters (default: min_l from PDK). "
             "Must match a PDK-defined lmin boundary.",
    )
    parser.add_argument(
        "--nfin",
        type=float,
        default=1.0,
        metavar="F",
        help="Fin count (default: 1.0)",
    )

    # Operating conditions
    parser.add_argument(
        "--temp",
        type=float,
        default=27.0,
        metavar="C",
        help="Temperature in Celsius (default: 27.0). Converted to Kelvin.",
    )

    # Analysis parameters
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        metavar="F",
        help="Perturbation fraction (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=9,
        metavar="N",
        help="Number of top parameters to report per category (default: 9)",
    )

    # Output
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Optional CSV file to write full sensitivity results",
    )

    # Verbosity
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose: per-parameter progress (default)",
    )
    verbosity.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet: summary only",
    )

    return parser


def write_csv(result: SensitivityResult, filepath: str) -> None:
    """Write full sensitivity data to CSV."""
    columns = ["parameter"] + OUTPUT_KEYS
    # Add per-category scores
    for cat_name in sorted(OUTPUT_CATEGORIES.keys()):
        columns.append(f"score_{cat_name}")

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for param_name in sorted(result.sensitivities.keys()):
            sens = result.sensitivities[param_name]
            row: List[str] = [param_name]
            for key in OUTPUT_KEYS:
                row.append(f"{sens.get(key, 0.0):.6e}")
            for cat_name in sorted(OUTPUT_CATEGORIES.keys()):
                score = sum(
                    abs(sens.get(k, 0.0)) for k in OUTPUT_CATEGORIES[cat_name]
                )
                row.append(f"{score:.6e}")
            writer.writerow(row)


def main() -> None:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args()

    # Validate OSDI path
    osdi_path = Path(args.osdi)
    if not osdi_path.exists():
        parser.error(f"OSDI binary not found: {args.osdi}")

    # Resolve technology and device
    try:
        tech = get_tech_config(args.tech)
    except KeyError as e:
        parser.error(str(e))

    if args.device not in tech.devices:
        available = ", ".join(tech.list_devices())
        parser.error(
            f"Device '{args.device}' not found in {args.tech}. "
            f"Available: {available}"
        )

    device = tech.get_device(args.device)
    device_type = "nmos" if "nmos" in args.device else "pmos"

    # Compute gate length
    if args.l is not None:
        L = args.l
    else:
        L = device.get_min_l(tech.pdk_path)

    # Resolve modelcard (NFIN-aware variant selection)
    try:
        modelcard_path = resolve_modelcard(device, tech, L, NFIN=args.nfin)
    except (RuntimeError, ValueError) as e:
        parser.error(f"Cannot resolve modelcard: {e}")

    # Instance parameters
    inst_params = {**device.inst_params, "L": L, "NFIN": args.nfin}

    # Temperature: Celsius -> Kelvin
    temperature = args.temp + 273.15

    # Verbosity
    if args.quiet:
        verbose = 0
    elif args.verbose:
        verbose = 2
    else:
        verbose = 1

    # Run analysis
    print(
        f"Sensitivity analysis: {args.tech} {args.device} "
        f"L={L * 1e9:.1f}nm NFIN={args.nfin:.0f} T={args.temp:.0f}C "
        f"delta={args.delta * 100:.0f}%"
    )

    result = compute_sensitivity(
        osdi_path=str(osdi_path),
        modelcard_path=modelcard_path,
        model_name=device.model_name,
        inst_params=inst_params,
        vdd=tech.vdd,
        device_type=device_type,
        temperature=temperature,
        delta_fraction=args.delta,
        top_n=args.top_n,
        verbose=verbose,
    )

    # Print tables
    print()
    for cat in ["iv", "qv", "cv"]:
        print(format_sensitivity_table(result, cat))
        print()

    # Optional CSV output
    if args.output:
        write_csv(result, args.output)
        print(f"Full results written to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
