"""Dependency-free argument definitions shared by the root CLI and NN backends."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence, Tuple

TRAINING_RECIPES = {
    "clean": {},
    "s7": {"seed": 7}, "s17": {"seed": 17}, "s123": {"seed": 123},
    "sub": {"subthresh": True},
    "ar3": {"full_terminal_ar_targets": 3},
    "ar3roll": {"full_terminal_ar_targets": 3, "autoregressive_training": True},
    "corridor": {"training_overlay_classes": "traj_corridor",
                 "class_weights": "traj_corridor=3.0"},
}

# D1: temperature sweep (paper §3 — full operating range, in Kelvin).
DEFAULT_TEMPERATURES_K: Tuple[float, ...] = (
    248.15,   # -25 °C
    300.15,   #  27 °C
    398.15,   # 125 °C
)
DEFAULT_MAX_L_RATIO: float = 1.35

# D3: voltage box widening factor. paper uses 1.0 (i.e. [0, 1]·VDD).
# v5p (V5'): revert B2 box-factor change. Restore V4 B1 default 2.0.
# Phase A+B evidence (results/v5_v4_vs_phaseA_vs_phaseAB_2026_05_08.md)
# showed B2's 1.5 box + overshoot overlay regressed TSMC7/12/16 to
# NR-runaway 10^12 V. Reverting to V4 B1 box; the ``overshoot`` sample
# class is also disabled below (DEFAULT_OVERSHOOT_PER_AXIS=0).
DEFAULT_VOLTAGE_BOX_FACTOR: float = 2.0

# D3: per-bin LHS sample budget. paper uses ~5K/bin for the [0, 1]·VDD
# range; doubled for [0, 2]·VDD to keep the same density.
DEFAULT_LHS_SAMPLES_PER_BIN: int = 5000

# B1 (v5 plan §4-B1): hybrid uniform-grid sampler defaults.
# Replaces LHS for the bulk of the per-bin samples. The grid is
# strictly more uniform than LHS in the high-current corner that
# dominates the verifier metric (D1 finding: hot region holds 3.07 %
# of LHS samples but 16× the verifier-weighted error mass).
DEFAULT_GRID_PER_AXIS: int = 30        # 30 × 30 = 900 (Vgs, Vds) points
DEFAULT_VBS_LEVELS: int = 5            # {0, ±0.25, ±0.5}·VDD
DEFAULT_HOT_PER_AXIS: int = 12         # 12 × 12 hot-region densification
DEFAULT_JITTER_SIGMA_FRAC: float = 0.05  # σ = 0.05·VDD on each axis
DEFAULT_SAMPLER: str = "grid"          # "grid" | "lhs"

# V5' B2/B3 overlays stay opt-in after their TSMC7/12/16 regressions.
DEFAULT_OVERSHOOT_PER_AXIS: int = 0
DEFAULT_N_VBS_LHS: int = 0


def _nonnegative_integer(arg: str) -> int:
    value = int(arg)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value

def _parse_temperatures(arg: str) -> tuple[float, ...]:
    values = tuple(float(x.strip()) for x in arg.split(","))
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise argparse.ArgumentTypeError("temperatures must be finite positive Kelvin values")
    return values


def generation_parser(techs: Sequence[str] = ("asap7", "tsmc5", "tsmc6", "tsmc7", "tsmc12", "tsmc16")) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate NN training data (.npz) from PyCMG BSIM-CMG"
    )
    parser.add_argument("--device", choices=["nmos", "pmos", "both"], default="nmos")
    parser.add_argument("--tech", choices=list(techs) + ["all"],
                        default="asap7")
    parser.add_argument("--variants", default="all",
                        help="Comma-separated variant names (default: all)")
    parser.add_argument("--universal", action="store_true",
                        help="Generate universal dataset across all techs/variants")

    # D1
    parser.add_argument(
        "--temperatures", type=_parse_temperatures,
        default=DEFAULT_TEMPERATURES_K,
        help="Comma-separated temperatures in Kelvin (default: -25, 27, 125 °C)",
    )

    # D3
    parser.add_argument("--n-lhs-samples", type=int,
                        default=DEFAULT_LHS_SAMPLES_PER_BIN,
                        help="LHS samples per (variant, L, NFIN, T) bin "
                             "(only used when --sampler=lhs)")
    parser.add_argument("--voltage-box-factor", type=float,
                        default=DEFAULT_VOLTAGE_BOX_FACTOR,
                        help="Voltage box width in units of VDD "
                             "(2.0 = [0, 2]·VDD; covers NR overshoot)")

    # B1 (v5 plan §4): hybrid uniform-grid sampler.
    parser.add_argument("--sampler", choices=["grid", "lhs"],
                        default=DEFAULT_SAMPLER,
                        help="Bulk-sample sampler: 'grid' = hybrid "
                             "uniform-grid + jitter + hot densification "
                             "(default, B1); 'lhs' = legacy Latin Hypercube")
    parser.add_argument("--grid-per-axis", type=int,
                        default=DEFAULT_GRID_PER_AXIS,
                        help="[grid] base 2D grid size per axis (Vgs, Vds)")
    parser.add_argument("--vbs-levels", type=int,
                        default=DEFAULT_VBS_LEVELS,
                        help="[grid] number of Vbs levels {0,±0.25,±0.5}·VDD")
    parser.add_argument("--hot-per-axis", type=int,
                        default=DEFAULT_HOT_PER_AXIS,
                        help="[grid] hot-region densification grid size "
                             "(0 to disable)")
    parser.add_argument("--jitter-sigma-frac", type=float,
                        default=DEFAULT_JITTER_SIGMA_FRAC,
                        help="[grid] Gaussian jitter sigma in fractions of VDD")
    parser.add_argument("--overshoot-per-axis", type=_nonnegative_integer,
                        default=DEFAULT_OVERSHOOT_PER_AXIS,
                        help="Experimental overshoot overlay grid size per axis (default: 0, disabled)")
    parser.add_argument("--n-vbs-lhs", type=_nonnegative_integer,
                        default=DEFAULT_N_VBS_LHS,
                        help="Experimental body-bias LHS overlay sample count (default: 0, disabled)")
    parser.add_argument("--quiet", action="store_true",
                        help="Disable per-bin generation progress; retain summaries and errors")

    # D4
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel worker count (1 = serial)")
    parser.add_argument("--seed", type=int, default=42)

    # D8
    parser.add_argument("--finetune-size", type=int, default=0,
                        help="If >0, also write finetune_<base>.npz with a "
                             "stratified random subset of N samples (D8)")

    # v5 plan §4-B5: dataset versioning + tech exclusion.
    parser.add_argument(
        "--version", default="",
        help="Version tag inserted after the scope in output filenames "
             "(e.g. 'v5' -> universal_v5_dnf_nmos.npz). Empty preserves "
             "the unversioned name.",
    )
    parser.add_argument(
        "--exclude-techs", default="",
        help="Comma-separated tech names to exclude from generation "
             "(case-insensitive). Common v5 use: --exclude-techs asap7.",
    )

    # v5p (V5'): inv_trip overlay is now opt-in. Default off matches the
    # V4 B1 base sampler. When set, nn_generate.py additionally gates
    # the overlay to TSMC5 only.
    parser.add_argument(
        "--enable-inv-trip", action="store_true", default=False,
        help="Enable v5 plan §4-B1 inverter-trip overlay. In V5' this "
             "is additionally gated to TSMC5 only inside nn_generate.py.",
    )

    # V6.4.7 S9b: subthreshold/OFF densification + DC-solve floor fix.
    parser.add_argument(
        "--enable-subvt-off", action="store_true", default=False,
        help="Enable the V6.4.7 S9b subthreshold/OFF |id|-space band probe "
             "(sample_class='subvt_off'). Fills the 1e-12..1e-6 A id decades "
             "for the decade-occupancy acceptance gate. Requires the "
             "DC-solve floor fix (see --dc-solve-tol).",
    )
    parser.add_argument(
        "--dc-solve-tol", type=float, default=1e-12,
        help="OSDI internal-node NR tolerance for generated rows, exported "
             "as NN_DC_SOLVE_TOL (default 1e-12). The legacy 1e-9 default "
             "returned EXACT 0 for true |id|<~1e-9 A (the 6-8%% zero-row "
             "artifact); 1e-12 resolves the sub-nA band. 1e-14 is FP-limited.",
    )

    # V7.4.2: intra-bin L sampling. The PDK grid gives one L per length
    # bin (its lower corner), and short-channel bins are wide — TSMC5's
    # shortest spans L in [6, 20] nm. Nothing constrains the model between
    # knots, and higher capacity lets that interpolant drift further, which
    # is what produced the "capacity hurts BSIM-AR" artifact (docs/plans/
    # 2026-08-10-v742-bsimar-capacity.md). Default off = legacy grid.
    parser.add_argument(
        "--max-l-ratio", type=float, default=DEFAULT_MAX_L_RATIO,
        help="Sample inside each PDK length bin so no adjacent pair of L "
             "knots differs by more than this ratio (default: 1.35). Costs roughly "
             "log(bin span)/log(ratio) times the rows.",
    )
    parser.add_argument(
        "--allow-rejected-points", action="store_true",
        help="Write a diagnostic artifact despite rejected points/bins. "
             "Canonical datasets fail instead.",
    )
    parser.add_argument(
        "--allow-safety-rejections", action="store_true",
        help="Keep a canonical dataset after excluding only the declared "
             "NaN/Inf, >1 A terminal-current, or internal-node-solve safety "
             "failures. Dropped bins and other failures remain fatal.",
    )
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Output directory for .npz files")
    return parser


def training_parser(tech_scopes: Sequence[str] = ("universal", "tsmc5", "tsmc6", "tsmc7", "tsmc12", "tsmc16")) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a full-terminal DirectNet or BSIM-AR compact model"
    )
    parser.add_argument(
        "--model", choices=["direct", "transformer"], default="direct",
    )
    parser.add_argument(
        "--size", choices=["small", "medium", "large", "xl"],
        default="medium",
    )
    parser.add_argument(
        "--device-type", choices=["nmos", "pmos"], default="nmos",
    )
    parser.add_argument("--data", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--split-mode", choices=["combo", "random"], default="combo",
    )
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-techs", type=str)
    parser.add_argument("--num-tech-codes", type=int)
    parser.add_argument("--p-unknown", type=float, default=0.1)
    parser.add_argument(
        "--tech-scope", choices=list(tech_scopes), default="universal",
    )
    parser.add_argument("--exp-name", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--swa-mode", choices=["none", "ema", "swa"], default="none",
    )
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--class-weights", type=str)
    parser.add_argument("--training-overlay-classes", type=str)
    parser.add_argument("--init-from", type=str)
    parser.add_argument(
        "--full-terminal-ar-targets", type=int, choices=[3, 6],
    )
    parser.add_argument("--autoregressive-training", action="store_true")
    parser.add_argument("--subthresh", action="store_true")
    parser.add_argument("--lam-subthresh", type=float, default=0.05)
    parser.add_argument("--subthresh-s2", type=float, default=1e-9)
    parser.add_argument("--subthresh-upper", type=float, default=1e-6)
    parser.add_argument("--subthresh-floor", type=float, default=1e-12)
    parser.add_argument("--subthresh-off-floor", type=float, default=1e-10)
    parser.add_argument("--subthresh-ceiling-k", type=float, default=1.0)
    parser.add_argument("--subthresh-ceiling-w", type=float, default=1.0)
    return parser
