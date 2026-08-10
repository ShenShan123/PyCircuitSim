#!/usr/bin/env python3
"""
Generate naive TSMC modelcards from full TSMC PDK.

This script extracts parameters from the full TSMC PDK (which has .global + variant structure)
and generates naive single-model modelcards that can be used directly with both PyCMG and NGSPICE+OSDI.

Supports all TSMC FinFET technology nodes:
- TSMC5 (5nm)
- TSMC7 (7nm)
- TSMC12 (12nm)
- TSMC16 (16nm)

Usage:
    python scripts/generate_naive_tsmc.py \
        --tech TSMC7 \
        --pdk modelcards/TSMC7/cln7_1d8_sp_v1d2_2p2.l \
        --output modelcards/TSMC7/naive/ \
        --devices nch_svt_mac,nch_lvt_mac,nch_ulvt_mac,pch_svt_mac \
        --lengths 16e-9,20e-9,24e-9
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path to import from pycmg
sys.path.insert(0, str(Path(__file__).parent.parent))

from pycmg.parser import parse_number_with_suffix
from pycmg.tech import generate_naive_tsmc_modelcard


def batch_generate_naive_modelcards(
    pdk_path: str,
    output_dir: str,
    devices: List[str],
    lengths: List[float],
    tech: str,
    NFIN: Optional[float] = None,
) -> None:
    """Batch generate naive modelcards for multiple devices and lengths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_files = 0
    errors = []
    for device in devices:
        parts = device.split("_", 1)
        if len(parts) != 2:
            msg = f"Invalid device name format: {device}"
            print(f"Warning: {msg}, skipping...", file=sys.stderr)
            errors.append(msg)
            continue

        model_type, device_type = parts[0], parts[1]

        for L in lengths:
            L_nm = int(L * 1e9)
            filename = f"{device}_l{L_nm}nm.l"
            file_path = output_path / filename

            try:
                generate_naive_tsmc_modelcard(
                    str(pdk_path),
                    model_type,
                    device_type,
                    L,
                    str(file_path),
                    tech,
                    NFIN=NFIN,
                )
                total_files += 1
                print(f"Generated: {file_path}")
            except Exception as e:
                msg = f"Error generating {filename}: {e}"
                print(msg, file=sys.stderr)
                errors.append(msg)

    print(f"\nTotal files generated: {total_files}")
    print(f"Output directory: {output_dir}")
    if errors:
        print(f"\n{len(errors)} error(s) encountered:")
        for err in errors:
            print(f"  - {err}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate naive TSMC modelcards from full TSMC PDK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate TSMC7 modelcards
  python scripts/generate_naive_tsmc.py \\
      --tech TSMC7 \\
      --pdk modelcards/TSMC7/cln7_1d8_sp_v1d2_2p2.l \\
      --output modelcards/TSMC7/naive/ \\
      --devices nch_svt_mac \\
      --lengths 16e-9

  # Batch generate multiple devices/lengths for TSMC5
  python scripts/generate_naive_tsmc.py \\
      --tech TSMC5 \\
      --pdk modelcards/TSMC5/cln5_1d2_sp_v1d2_2p2.l \\
      --output modelcards/TSMC5/naive/ \\
      --devices nch_svt_mac,nch_lvt_mac,pch_svt_mac,pch_lvt_mac \\
      --lengths 16e-9,20e-9,24e-9
        """
    )

    parser.add_argument(
        "--tech",
        required=True,
        choices=["TSMC5", "TSMC6", "TSMC7", "TSMC12", "TSMC16"],
        help="Technology node (TSMC5, TSMC6, TSMC7, TSMC12, or TSMC16)"
    )
    parser.add_argument(
        "--pdk",
        required=True,
        help="Path to full TSMC PDK file (e.g., cln7_1d8_sp_v1d2_2p2.l)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for naive modelcards"
    )
    parser.add_argument(
        "--devices",
        required=True,
        help="Comma-separated list of device names (e.g., nch_svt_mac,pch_svt_mac)"
    )
    parser.add_argument(
        "--lengths",
        required=True,
        help="Comma-separated list of gate lengths in meters (e.g., 16e-9,20e-9,24e-9)"
    )
    parser.add_argument(
        "--nfin",
        type=float,
        default=None,
        help="Fin count for NFIN-group-specific variant selection."
    )

    args = parser.parse_args()

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]

    lengths: List[float] = []
    for l_str in args.lengths.split(","):
        l_str = l_str.strip()
        try:
            lengths.append(parse_number_with_suffix(l_str))
        except ValueError:
            try:
                lengths.append(float(l_str))
            except ValueError:
                print(f"Warning: Invalid length '{l_str}', skipping...", file=sys.stderr)

    if not devices:
        print("Error: No valid devices specified", file=sys.stderr)
        sys.exit(1)

    if not lengths:
        print("Error: No valid lengths specified", file=sys.stderr)
        sys.exit(1)

    if not Path(args.pdk).exists():
        print(f"Error: PDK file not found: {args.pdk}", file=sys.stderr)
        sys.exit(1)

    print(f"Generating naive {args.tech} modelcards...")
    print(f"  PDK: {args.pdk}")
    print(f"  Output: {args.output}")
    print(f"  Devices: {', '.join(devices)}")
    print(f"  Lengths: {', '.join(f'{L*1e9:.1f}nm' for L in lengths)}")
    print()

    batch_generate_naive_modelcards(
        args.pdk,
        args.output,
        devices,
        lengths,
        args.tech,
        NFIN=args.nfin,
    )


if __name__ == "__main__":
    main()
