#!/usr/bin/env python3
"""
PyCircuitSim - Simple Python Circuit Simulator

Command-line interface entry point.
"""

import sys
import argparse
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='PyCircuitSim - Simple Python Circuit Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simulation with default output directory
  python main.py circuit.sp

  # Specify output directory
  python main.py circuit.sp -o my_results

  # Enable verbose output
  python main.py circuit.sp -v
        """
    )

    parser.add_argument(
        'netlist',
        help='Path to the HSPICE-format netlist file'
    )

    parser.add_argument(
        '-o', '--output',
        dest='output_dir',
        default='results',
        help='Output directory for plots and results (default: results)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging output'
    )

    args = parser.parse_args()

    # Import here to avoid errors if package not installed
    try:
        from pycircuitsim.main import run_simulation
    except ImportError:
        print("Error: PyCircuitSim package not found.", file=sys.stderr)
        print("Install with: pip install -e .", file=sys.stderr)
        sys.exit(1)

    # Run simulation
    try:
        run_simulation(
            netlist_path=args.netlist,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
