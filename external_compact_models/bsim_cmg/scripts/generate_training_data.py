#!/usr/bin/env python3
"""Generate BSIM-CMG training data for NN compact models.

Wraps ``pycmg.sweep.generate_dataset()`` with argparse for command-line use.

Usage examples::

    # Minimal: single tech, single device, small grid
    python scripts/generate_training_data.py \\
        --osdi build/osdi/bsimcmg.osdi \\
        --tech ASAP7 --devices nmos_rvt \\
        --temps 27 --vg-points 5 --vd-points 5

    # Full sweep across all techs with process variation
    python scripts/generate_training_data.py \\
        --osdi build/osdi/bsimcmg.osdi \\
        --process-var eot=0.9e-9,1.0e-9,1.1e-9 \\
        --process-var toxp=1.8e-9,2.1e-9

    # List available devices
    python scripts/generate_training_data.py \\
        --osdi build/osdi/bsimcmg.osdi --tech ASAP7 --list-devices
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow importing pycmg from project root when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pycmg.sweep import generate_dataset, SweepConfig
from pycmg.tech import TECH_REGISTRY, get_tech_config, list_techs


def parse_process_var(s: str) -> Tuple[str, List[float]]:
    """Parse ``'name=v1,v2,v3'`` into ``(name, [v1, v2, v3])``.

    Args:
        s: String in the format ``NAME=V1,V2,V3``.

    Returns:
        Tuple of (parameter name, list of float values).

    Raises:
        argparse.ArgumentTypeError: If the format is invalid.
    """
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"Invalid process-var format: '{s}'. Expected NAME=V1,V2,V3"
        )
    name, _, values_str = s.partition("=")
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(
            f"Empty parameter name in process-var: '{s}'"
        )
    try:
        values = [float(v.strip()) for v in values_str.split(",")]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid float value in process-var '{s}': {e}"
        )
    if not values:
        raise argparse.ArgumentTypeError(
            f"No values provided in process-var: '{s}'"
        )
    return name, values


def handle_list_devices(techs: List[str]) -> None:
    """Print available devices for specified techs and exit.

    Args:
        techs: List of technology names (or ``["all"]`` for all techs).
    """
    resolved: List[str] = list_techs() if "all" in techs else techs

    for tech_name in resolved:
        try:
            tech = get_tech_config(tech_name)
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            continue
        device_names = tech.list_devices()
        print(f"{tech_name}:")
        print(f"  {', '.join(device_names)}")

    sys.exit(0)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="generate_training_data.py",
        description="Generate BSIM-CMG training data for NN compact models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # Quick test with ASAP7 NMOS (single default geometry)\n"
            "  %(prog)s --osdi build/osdi/bsimcmg.osdi --tech ASAP7 "
            "--devices nmos_rvt --no-sweep-geometry --temps 27\n"
            "\n"
            "  # List available devices\n"
            "  %(prog)s --osdi build/osdi/bsimcmg.osdi --list-devices\n"
            "\n"
            "  # Full sweep with process variation\n"
            "  %(prog)s --osdi build/osdi/bsimcmg.osdi "
            "--process-var eot=0.9e-9,1.0e-9,1.1e-9\n"
        ),
    )

    # Required
    parser.add_argument(
        "--osdi",
        metavar="PATH",
        required=True,
        help="Path to compiled OSDI binary (e.g., build/osdi/bsimcmg.osdi)",
    )

    # Technology and device selection
    parser.add_argument(
        "--tech",
        nargs="+",
        default=["all"],
        metavar="TECH",
        help=(
            'Technology names to sweep (default: "all"). '
            f"Available: {', '.join(list_techs())}"
        ),
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        metavar="DEV",
        default=None,
        help=(
            "Device names to sweep, applied to ALL specified techs. "
            "Supports glob patterns (e.g., nmos_*). "
            "Default: all devices in each tech."
        ),
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available devices for specified tech(s) and exit.",
    )

    # Geometry sweep
    parser.add_argument(
        "--no-sweep-geometry",
        action="store_true",
        help="Disable PDK-defined geometry sweep.  Uses single (min_l, 1) "
             "per device instead of all (L, NFIN) bin boundary combinations.",
    )

    # Operating conditions
    parser.add_argument(
        "--temps",
        nargs="+",
        type=float,
        default=[-40.0, 0.0, 27.0, 85.0, 125.0],
        metavar="C",
        help="Temperatures in Celsius (default: -40 0 27 85 125). Converted to Kelvin internally.",
    )
    parser.add_argument(
        "--vg-points",
        type=int,
        default=50,
        metavar="N",
        help="Number of gate voltage grid points (default: 50)",
    )
    parser.add_argument(
        "--vd-points",
        type=int,
        default=50,
        metavar="N",
        help="Number of drain voltage grid points (default: 50)",
    )
    parser.add_argument(
        "--ve-values",
        nargs="+",
        type=float,
        default=[0.0],
        metavar="V",
        help="Body bias voltage magnitudes to sweep (default: 0.0)",
    )

    # Process variation
    parser.add_argument(
        "--process-var",
        action="append",
        metavar="NAME=V1,V2,V3",
        help=(
            "Process variation parameter (repeatable). "
            "Format: NAME=V1,V2,V3. "
            "Example: --process-var eot=0.9e-9,1.0e-9,1.1e-9"
        ),
    )

    # Voltage grid tuning
    parser.add_argument(
        "--dense-ratio",
        type=float,
        default=0.6,
        metavar="F",
        help="Fraction of vg_points in threshold-dense region (default: 0.6)",
    )
    parser.add_argument(
        "--voltage-scale",
        type=float,
        default=1.0,
        metavar="F",
        help="Voltage range multiplier (default: 1.0). Use 2.0 to extend "
             "sweeps to 2*VDD for NN simulator convergence training.",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="./training_data",
        metavar="DIR",
        help="Output directory for CSV files (default: ./training_data)",
    )
    parser.add_argument(
        "--split-by",
        choices=["tech", "device", "none"],
        default="tech",
        help='How to split output into files (default: "tech")',
    )

    # Verbosity (mutually exclusive)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output: per-configuration progress (verbose=2, default)",
    )
    verbosity.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet output: summary only (verbose=1)",
    )
    verbosity.add_argument(
        "--silent",
        action="store_true",
        help="Silent output: no output (verbose=0)",
    )

    return parser


def main() -> None:
    """Entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args()

    # Handle --list-devices early exit
    if args.list_devices:
        handle_list_devices(args.tech)

    # Validate OSDI path exists
    osdi_path = Path(args.osdi)
    if not osdi_path.exists():
        parser.error(f"OSDI binary not found: {args.osdi}")

    # Determine verbosity level
    if args.silent:
        verbose = 0
    elif args.quiet:
        verbose = 1
    else:
        # Default and --verbose both map to 2
        verbose = 2

    # Convert temperatures from Celsius to Kelvin
    temperatures: List[float] = [t + 273.15 for t in args.temps]

    # Parse process variation parameters
    process_vars: Optional[Dict[str, List[float]]] = None
    if args.process_var:
        process_vars = {}
        for pv_str in args.process_var:
            name, values = parse_process_var(pv_str)
            process_vars[name] = values

    # Map --devices flat list to per-tech dict
    devices: Optional[Dict[str, List[str]]] = None
    if args.devices:
        resolved_techs: List[str] = (
            list_techs() if "all" in args.tech else args.tech
        )
        devices = {t: args.devices for t in resolved_techs}

    # Run the sweep
    paths = generate_dataset(
        osdi_path=str(osdi_path),
        techs=args.tech,
        devices=devices,
        output_dir=args.output_dir,
        sweep_geometry=not args.no_sweep_geometry,
        temperatures=temperatures,
        vg_points=args.vg_points,
        vd_points=args.vd_points,
        ve_values=args.ve_values,
        process_vars=process_vars,
        dense_ratio=args.dense_ratio,
        split_by=args.split_by,
        verbose=verbose,
        voltage_scale=args.voltage_scale,
    )

    # Report output
    if verbose >= 1:
        print(f"\nOutput files:")
        for p in paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
