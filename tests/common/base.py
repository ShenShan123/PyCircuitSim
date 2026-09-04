"""Shared test infrastructure for BSIM-CMG verification suites.

Provides technology profiles, constants, and generic orchestration helpers
shared across DC and transient 3-level verification suites.

Extracted from bsimcmg_tran_common.py to eliminate duplication between
DC and transient common modules.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# tests/common/base.py → parents[0]=common/, parents[1]=tests/, parents[2]=project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "bsim_cmg"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "bsim_cmg" / "tests"))

OSDI_PATH = (
    PROJECT_ROOT / "external_compact_models" / "bsim_cmg"
    / "build" / "osdi" / "bsimcmg.osdi"
)
MODELCARDS_DIR = (
    PROJECT_ROOT / "PDKs"
)
NGSPICE_BIN = os.environ.get(
    "NGSPICE_BIN", "/usr/local/ngspice-45.2/bin/ngspice")

#: Every circuit a gate simulates is a file under here — gates render these,
#: they never carry a topology of their own. See ``render_template``.
TEMPLATES_DIR = PROJECT_ROOT / "circuit_templates"

#: Circuit templates are ordered by what they demand of a compact model, not
#: by application domain.  The ladder is the point: L0 biases one device from
#: ideal sources, L1 adds passive loading, L2 couples several devices while
#: every bias rail stays ideal, L3 makes the model determine an operating
#: point (internal bias generation or internal state), and L4 closes a
#: negative-feedback loop around it.  A gate that passes L2 and fails L3 has
#: localized its failure to basin selection rather than pointwise error.
CIRCUIT_TIERS: Tuple[str, ...] = (
    "L0_devices",
    "L1_primitives",
    "L2_stages",
    "L3_blocks",
    "L4_systems",
)
TIER_DIRS: Dict[str, Path] = {
    tier: TEMPLATES_DIR / tier for tier in CIRCUIT_TIERS
}
DEVICE_DECKS = TIER_DIRS["L0_devices"]
SUBCIRCUIT_DECKS = TEMPLATES_DIR / "subcircuits"
CONTROLS_DIR = TEMPLATES_DIR / "controls"


def template_deck(name: str, *, tier: str = "") -> Path:
    """Resolve one template filename to its difficulty tier.

    Callers name a topology, not a directory, so a template can be re-tiered
    without editing every gate.  Resolution is strict in both directions: an
    unknown ``tier``, a file that is absent from the tier it declares, a name
    no tier owns, and a name two tiers both own are all errors.  The ambiguity
    check is the one that matters — two copies of a topology is exactly the
    drift that ``circuit_templates/`` exists to prevent.
    """
    if tier:
        if tier not in TIER_DIRS:
            raise ValueError(
                f"unknown circuit tier {tier!r}; available: {list(CIRCUIT_TIERS)}")
        path = TIER_DIRS[tier] / name
        if not path.is_file():
            raise FileNotFoundError(f"{name!r} is not in tier {tier!r}")
        return path
    found = [TIER_DIRS[candidate] / name for candidate in CIRCUIT_TIERS
             if (TIER_DIRS[candidate] / name).is_file()]
    if not found:
        raise FileNotFoundError(
            f"no circuit tier owns {name!r}; searched {list(CIRCUIT_TIERS)}")
    if len(found) > 1:
        raise ValueError(
            f"{name!r} is duplicated across tiers: "
            + ", ".join(str(path.parent.name) for path in found))
    return found[0]


def control_deck(name: str) -> Path:
    """Resolve a solver-control template outside the compact-model ladder."""
    path = CONTROLS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"no solver control owns {name!r}")
    return path


def parse_no_options(description: str, argv: Optional[List[str]] = None) -> None:
    """Give a fixed-matrix gate the repo's argparse surface.

    A gate whose matrix is not selectable still has to answer ``--help`` and
    reject an unknown flag. Without this, ``--tech TSMC5`` on such a gate is
    silently dropped and the operator reads a full-matrix result as if it were
    the requested subset — the same failure mode the accuracy CLIs already
    guard against by rejecting unknown technologies before running.

    Args:
        description: One-line gate description shown by ``--help``.
        argv: Argument vector to check (default: ``sys.argv[1:]``).

    Raises:
        SystemExit: On ``--help`` (code 0) or any unexpected argument (code 2).
    """
    parser = argparse.ArgumentParser(
        description=f"{description} This gate runs a fixed matrix and takes "
                    "no options.")
    parser.parse_args(sys.argv[1:] if argv is None else argv)


def parse_csv_choices(
    parser: argparse.ArgumentParser,
    raw: str,
    *,
    flag: str,
    choices: Sequence[str],
    normalize: Callable[[str], str] = lambda value: value,
) -> List[str]:
    """Parse one comma selection without allowing denominator shrinkage.

    Empty, unknown, and duplicate values are command-line errors. A typo mixed
    with valid values must never run the valid subset and look like a complete
    accuracy result.
    """
    fields = raw.split(",")
    if not fields or any(not value.strip() for value in fields):
        parser.error(f"{flag} contains an empty value: {raw!r}")
    selected = [normalize(value.strip()) for value in fields]
    unknown = [value for value in selected if value not in choices]
    if unknown:
        parser.error(f"unknown {flag} values {unknown}; available: {list(choices)}")
    if len(selected) != len(set(selected)):
        parser.error(f"{flag} contains duplicates: {selected}")
    return selected


from helpers import bake_inst_params as bake_inst_params  # noqa: F401, E402

__all__ = ["bake_inst_params"]


# ---------------------------------------------------------------------------
# VtPair -- matched NMOS/PMOS threshold voltage variant
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VtPair:
    """Matched NMOS/PMOS threshold voltage variant for CMOS inverter."""
    vt_name: str       # e.g. "rvt", "svt", "lvt"
    nmos_model: str    # e.g. "nmos_rvt" or "nch_svt_mac"
    pmos_model: str    # e.g. "pmos_rvt" or "pch_svt_mac"


# ---------------------------------------------------------------------------
# TechProfile -- technology node configuration
# ---------------------------------------------------------------------------
@dataclass
class TechProfile:
    """Technology node specification with available VT/L combinations."""
    name: str
    vdd: float
    tfin: float                  # Fin thickness [m]
    default_l_nmos: float        # Default NMOS channel length [m]
    default_l_pmos: float        # Default PMOS channel length [m]
    l_values: List[float]        # All available L values for sweeping [m]
    default_nfin: int
    vt_pairs: List[VtPair]
    default_vt: str              # Name of default VtPair
    single_file: bool = False    # ASAP7: all models in one file
    modelcard_dir: str = ""      # Subdir under MODELCARDS_DIR
    single_file_name: str = ""   # For ASAP7: the single modelcard filename
    i_per_fin: float = 1e-5      # Estimated Idsat per fin [A]

    def get_vt_pair(self, vt_name: str) -> VtPair:
        for vp in self.vt_pairs:
            if vp.vt_name == vt_name:
                return vp
        raise ValueError(f"Unknown VT '{vt_name}' for {self.name}. "
                         f"Available: {[v.vt_name for v in self.vt_pairs]}")

    @property
    def default_vt_pair(self) -> VtPair:
        return self.get_vt_pair(self.default_vt)

    def _resolve_tsmc_modelcard(self, pdk_device: str, l_m: float) -> Path:
        """Generate a TSMC naive modelcard on-the-fly via pycmg.tech.resolve_modelcard.

        The old committed ``PDKs/TSMC*/naive/*.l`` files were removed; they
        are now regenerated from the raw PDK and cached under PyCMG's
        ``build/modelcards/``. Uses ``NFIN=self.default_nfin`` so the correct
        NFIN-group variant is selected.
        """
        from pycmg.tech import TECH_REGISTRY, resolve_modelcard
        tech_config = TECH_REGISTRY[self.name]
        # Map "nch_svt_mac" -> "nmos_svt", "pch_lvt_mac" -> "pmos_lvt"
        prefix = pdk_device.split("_", 1)[0]
        vt = pdk_device.split("_", 1)[1].replace("_mac", "")
        canonical = ("nmos_" if prefix == "nch" else "pmos_") + vt
        device_config = tech_config.get_device(canonical)
        return Path(resolve_modelcard(
            device_config, tech_config,
            L=l_m, NFIN=float(self.default_nfin),
        ))

    def get_nmos_modelcard(self, vt: VtPair, l_nmos: float) -> Path:
        """Return path to NMOS modelcard file for given VT and L."""
        if self.single_file:
            return MODELCARDS_DIR / self.modelcard_dir / self.single_file_name
        return self._resolve_tsmc_modelcard(vt.nmos_model, l_nmos)

    def get_pmos_modelcard(self, vt: VtPair, l_pmos: float) -> Path:
        """Return path to PMOS modelcard file for given VT and L."""
        if self.single_file:
            return MODELCARDS_DIR / self.modelcard_dir / self.single_file_name
        return self._resolve_tsmc_modelcard(vt.pmos_model, l_pmos)

    def is_combo_available(self, vt: VtPair, l_nmos: float, l_pmos: float) -> bool:
        """Check if modelcard files exist for this VT and L combination."""
        return (self.get_nmos_modelcard(vt, l_nmos).exists()
                and self.get_pmos_modelcard(vt, l_pmos).exists())

    def get_available_l_values(self, vt: VtPair) -> List[float]:
        """Return L values where both NMOS and PMOS modelcards exist."""
        if self.single_file:
            return list(self.l_values)
        return [l for l in self.l_values
                if self.is_combo_available(vt, l, l)]


# ---------------------------------------------------------------------------
# Technology definitions (5 technologies x multiple VT flavors)
# ---------------------------------------------------------------------------
ALL_TECHS: Dict[str, TechProfile] = {
    "ASAP7": TechProfile(
        name="ASAP7", vdd=0.7, tfin=6.5e-9,
        default_l_nmos=30e-9, default_l_pmos=30e-9,
        l_values=[30e-9],
        default_nfin=10, default_vt="rvt",
        vt_pairs=[
            VtPair("rvt",  "nmos_rvt",  "pmos_rvt"),
            VtPair("lvt",  "nmos_lvt",  "pmos_lvt"),
            VtPair("slvt", "nmos_slvt", "pmos_slvt"),
            VtPair("sram", "nmos_sram", "pmos_sram"),
        ],
        single_file=True,
        modelcard_dir="ASAP7",
        single_file_name="7nm_TT_160803.pm",
    ),
    "TSMC5": TechProfile(
        name="TSMC5", vdd=0.65, tfin=6e-9,
        default_l_nmos=16e-9, default_l_pmos=20e-9,
        l_values=[16e-9, 20e-9, 24e-9],
        default_nfin=2, default_vt="lvt",
        vt_pairs=[
            # SVT removed: pch_svt_mac PDIBL2_i<0 at L=20nm NFIN=2
            VtPair("lvt",  "nch_lvt_mac",  "pch_lvt_mac"),
            VtPair("ulvt", "nch_ulvt_mac", "pch_ulvt_mac"),
            VtPair("elvt", "nch_elvt_mac", "pch_elvt_mac"),
        ],
        modelcard_dir="TSMC5/naive",
    ),
    "TSMC6": TechProfile(
        name="TSMC6", vdd=0.75, tfin=6e-9,
        default_l_nmos=16e-9, default_l_pmos=20e-9,
        # V6.9.0: unlike TSMC7 (whose N7-era SVT/LVT bins misbehave), the N6
        # card's updated bins pass ALL 3 VTs and L=16/20/24nm (DC 9/9,
        # tran 14/14 vs NGSPICE) — no empirical pruning needed.
        l_values=[16e-9, 20e-9, 24e-9],
        default_nfin=2, default_vt="ulvt",
        vt_pairs=[
            VtPair("svt",  "nch_svt_mac",  "pch_svt_mac"),
            VtPair("lvt",  "nch_lvt_mac",  "pch_lvt_mac"),
            VtPair("ulvt", "nch_ulvt_mac", "pch_ulvt_mac"),
        ],
        modelcard_dir="TSMC6/naive",
    ),
    "TSMC7": TechProfile(
        name="TSMC7", vdd=0.75, tfin=6e-9,
        default_l_nmos=16e-9, default_l_pmos=20e-9,
        l_values=[20e-9, 24e-9],  # L=16nm removed: ULVT inverter diverges at symmetric L=16nm
        default_nfin=2, default_vt="ulvt",  # SVT/LVT: garbage output or PDIBL2_i<0
        vt_pairs=[
            # SVT removed: inverter garbage output at L=16/20nm
            # LVT removed: pch_lvt_mac PDIBL2_i<0 at L=20nm NFIN=2
            VtPair("ulvt", "nch_ulvt_mac", "pch_ulvt_mac"),
        ],
        modelcard_dir="TSMC7/naive",
    ),
    "TSMC12": TechProfile(
        name="TSMC12", vdd=0.80, tfin=6e-9,
        default_l_nmos=16e-9, default_l_pmos=20e-9,
        l_values=[16e-9, 20e-9, 24e-9],
        default_nfin=2, default_vt="svt",
        vt_pairs=[
            VtPair("svt",  "nch_svt_mac",  "pch_svt_mac"),
            VtPair("lvt",  "nch_lvt_mac",  "pch_lvt_mac"),
            VtPair("hvt",  "nch_hvt_mac",  "pch_hvt_mac"),
            VtPair("ulvt", "nch_ulvt_mac", "pch_ulvt_mac"),
            VtPair("lnvt", "nch_lnvt_mac", "pch_lnvt_mac"),
        ],
        modelcard_dir="TSMC12/naive",
    ),
    "TSMC16": TechProfile(
        name="TSMC16", vdd=0.80, tfin=6e-9,
        default_l_nmos=16e-9, default_l_pmos=20e-9,
        l_values=[16e-9, 20e-9],  # L=24nm removed: nch_svt_mac PDIBL2_i<0
        default_nfin=2, default_vt="svt",
        vt_pairs=[
            VtPair("svt",  "nch_svt_mac",  "pch_svt_mac"),
            VtPair("lvt",  "nch_lvt_mac",  "pch_lvt_mac"),
            VtPair("hvt",  "nch_hvt_mac",  "pch_hvt_mac"),
            VtPair("ulvt", "nch_ulvt_mac", "pch_ulvt_mac"),
            # LNVT removed: nch_lnvt_mac PDIBL2_i<0 at L=16nm NFIN=2
        ],
        modelcard_dir="TSMC16/naive",
    ),
}

TECH_ORDER: List[str] = ["ASAP7", "TSMC5", "TSMC6", "TSMC7", "TSMC12", "TSMC16"]

TECH_COLORS: Dict[str, str] = {
    "ASAP7": "tab:blue",
    "TSMC5": "tab:green",
    "TSMC6": "tab:brown",
    "TSMC7": "tab:orange",
    "TSMC12": "tab:purple",
    "TSMC16": "tab:red",
}


# ---------------------------------------------------------------------------
# circuit_templates/ reference-deck rendering
# ---------------------------------------------------------------------------
_DECK_TOKEN_RE = re.compile(r"<([A-Z][A-Z0-9_]*)>")


def deck_tokens(text: str) -> Tuple[str, ...]:
    """Return the sorted, unique placeholder names used by a deck.

    Keeping token discovery beside rendering lets callers build one broad set
    of technology substitutions and then pass only the values a particular
    deck declares.  ``render_deck_text`` still enforces the important strict
    contract: missing and stale substitutions are errors.
    """
    active = "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith("*")
    )
    return tuple(sorted(set(_DECK_TOKEN_RE.findall(active))))


def render_deck_text(
    template_text: str,
    subs: Dict[str, str],
    *,
    source_name: str = "<memory>",
    body_only: bool = False,
) -> str:
    """Render example-deck text with strict placeholder validation.

    This is the engine-neutral implementation behind
    :func:`render_template`.  Both the LEVEL=72 reference adapter and
    the NN candidate adapter use it, so neither adapter needs a private
    netlist-rewrite convention.
    """
    lines: List[str] = []
    saw_title = False
    for raw in template_text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("*"):
            if body_only or saw_title:
                continue
            saw_title = True
            lines.append(" ".join(stripped.split()))
            continue
        if body_only and stripped.lower() == ".end":
            continue
        lines.append(" ".join(stripped.split()))

    text = "\n".join(lines)
    found = set(_DECK_TOKEN_RE.findall(text))
    missing = found - set(subs)
    if missing:
        raise KeyError(
            f"{source_name}: no substitution for "
            f"{sorted(missing)} (supplied: {sorted(subs)})")
    unused = set(subs) - found
    if unused:
        raise KeyError(
            f"{source_name}: substitutions {sorted(unused)} match no "
            "token in the deck — the template changed under the caller")

    rendered = _DECK_TOKEN_RE.sub(lambda m: subs[m.group(1)], text)
    return rendered if body_only else rendered + "\n"


def render_template(template_path: Path, subs: Dict[str, str], *,
                    body_only: bool = False) -> str:
    """Render an ``circuit_templates/`` template into runnable netlist text.

    The gates own stimulus and tolerances; they do NOT own topology. Every
    circuit a gate simulates is a file under ``circuit_templates/``, and this is how it
    gets read. Keeping one copy is the point: a topology that lives in both a
    documentation deck and an f-string builder drifts, silently, and the
    ``circuit_templates/`` copy is the one nobody re-runs.

    The template is a SPICE deck annotated with ``<TOKEN>`` placeholders for the
    values that vary per technology (``<VDD>``, ``<NMOS>``, the baked-modelcard
    path, ...). Rendering:

    * drops blank lines and ``*`` comments, so the templates can carry the
      commentary that makes them readable as examples;
    * collapses interior whitespace runs, so columns can be aligned in the file
      without changing a single byte of what NGSPICE is handed;
    * substitutes ``<TOKEN>`` **after** normalizing, so a substituted value may
      legitimately contain spaces (a work-dir path, a PULSE argument list).

    ``body_only`` additionally drops the trailing ``.end``, yielding the bare
    circuit body that ``run_ngspice_wrdata`` embeds in a runner deck. Full
    decks keep their title line, because NGSPICE consumes the first line of a
    deck as the title and would otherwise silently eat ``.include``.

    Raises on a token the caller did not supply AND on a substitution the
    template does not use — a stale key means the caller believes it is
    parameterizing something it is not, which is the exact failure this
    function exists to prevent.
    """
    return render_deck_text(
        template_path.read_text(), subs,
        source_name=template_path.name, body_only=body_only,
    )


# ---------------------------------------------------------------------------
# Generic NGSPICE subprocess runner
# ---------------------------------------------------------------------------
def run_ngspice_subprocess(
    runner_path: Path,
    log_path: Path,
    csv_path: Path,
) -> List[str]:
    """Run NGSPICE in batch mode and return raw CSV lines.

    Handles:
      - stale-artifact removal (work dirs are persistent and deterministically
        named, so a previous run's CSV/log MUST NOT survive into this one)
      - subprocess execution + exit-status check
      - OSDI fatal error checking in log
      - CSV existence and emptiness checks

    Returns the raw lines from the CSV file (caller does domain-specific parsing).
    Raises RuntimeError on any failure.
    """
    # Remove the previous run's artifacts BEFORE invoking. Work dirs are
    # persistent and deterministically named, so without this a dead NGSPICE
    # (missing/non-executable binary, bad NGSPICE_BIN) leaves the prior CSV in
    # place and every downstream check passes against stale ground truth — the
    # whole suite reports PASS having compared nothing.
    csv_path.unlink(missing_ok=True)
    log_path.unlink(missing_ok=True)

    raw_timeout = os.environ.get("PYCIRCUITSIM_NGSPICE_TIMEOUT_S", "600")
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise ValueError(
            f"PYCIRCUITSIM_NGSPICE_TIMEOUT_S={raw_timeout!r} is not numeric"
        ) from exc
    if timeout <= 0.0:
        raise ValueError("PYCIRCUITSIM_NGSPICE_TIMEOUT_S must be positive")
    try:
        res = subprocess.run(
            [NGSPICE_BIN, "-b", "-o", str(log_path), str(runner_path)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"NGSPICE timed out after {timeout:g}s for {runner_path}"
        ) from exc

    # Non-zero exit is a hard failure. NGSPICE returns 0 on a healthy batch run;
    # anything else means the simulation did not complete, so nothing downstream
    # may be treated as ground truth.
    if res.returncode != 0:
        log_text = log_path.read_text() if log_path.exists() else "(no log)"
        raise RuntimeError(
            f"NGSPICE failed (rc={res.returncode}) for {runner_path}\n"
            f"bin: {NGSPICE_BIN}\n"
            f"stdout: {res.stdout[-500:]}\n"
            f"stderr: {res.stderr[-500:]}\n"
            f"log (tail): ...{log_text[-500:]}"
        )

    # Check for OSDI fatal errors (NGSPICE may still produce garbage output)
    if log_path.exists():
        log_text = log_path.read_text()
        if "Fatal:" in log_text:
            fatals = re.findall(r"Fatal:.*", log_text)
            raise RuntimeError(
                f"NGSPICE OSDI fatal error(s): {'; '.join(fatals[:3])}"
            )

    if not csv_path.exists():
        log_text = log_path.read_text() if log_path.exists() else "(no log)"
        raise RuntimeError(
            f"NGSPICE produced no output: {csv_path}\n"
            f"RC={res.returncode}, log (tail): ...{log_text[-500:]}"
        )

    with csv_path.open() as f:
        lines = f.readlines()

    if not lines:
        raise RuntimeError(f"Empty NGSPICE output: {csv_path}")

    return lines


# ---------------------------------------------------------------------------
# Generic summary bar chart
# ---------------------------------------------------------------------------
def plot_summary_bar(
    results: List[Dict[str, Any]],
    save_path: Path,
    title: str,
    nrmse_key: str,
    threshold: float,
    y_label: str,
) -> None:
    """Generic bar chart of NRMSE across all configs, colored by tech.

    Parameters
    ----------
    results : list of result dicts (must contain ``nrmse_key`` and ``config``)
    save_path : output PNG path
    title : chart title
    nrmse_key : key in result dict holding the NRMSE fraction (e.g. "nrmse_post", "nrmse")
    threshold : pass/fail threshold (fraction, e.g. 0.05 for 5%)
    y_label : Y-axis label
    """
    valid = [r for r in results if nrmse_key in r and "error" not in r]
    if not valid:
        return

    names = [r["config"].label for r in valid]
    nrmse_pct = [r[nrmse_key] * 100 for r in valid]
    colors = [TECH_COLORS.get(r["config"].tech.name, "tab:gray") for r in valid]

    fig, ax = plt.subplots(figsize=(max(14, len(valid) * 0.7), 6))
    x = np.arange(len(valid))
    ax.bar(x, nrmse_pct, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=threshold * 100, color="red", lw=1.5, ls="--",
               label=f"Threshold ({threshold*100:.0f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, edgecolor="black", label=t)
                  for t, c in TECH_COLORS.items()]
    legend_els.append(plt.Line2D([0], [0], color="red", lw=1.5, ls="--",
                                 label=f"Threshold ({threshold*100:.0f}%)"))
    ax.legend(handles=legend_els, loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Summary saved: {save_path}")


# ---------------------------------------------------------------------------
# Generic L2 test suite orchestrator
# ---------------------------------------------------------------------------
def run_test_suite(
    configs: List[Any],
    results_dir: Path,
    title: str,
    acceptance_msg: str,
    run_single_fn: Callable[[Any, Path], Dict[str, Any]],
    print_summary_fn: Callable[[List[Dict[str, Any]]], Tuple[int, int, int]],
    save_csv_fn: Callable[[List[Dict[str, Any]], Path], None],
    plot_bar_fn: Callable[[List[Dict[str, Any]], Path, str], None],
) -> int:
    """Generic L2 test suite orchestrator.

    Runs a list of configs through ``run_single_fn``, collects results,
    prints a summary table, saves CSV and bar chart, and returns an exit code.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results: List[Dict[str, Any]] = []

    print(f"\n{'='*78}")
    print(title)
    print(f"  Tests: {len(configs)}")
    print(f"  Acceptance: {acceptance_msg}")
    print(f"{'='*78}")

    for i, cfg in enumerate(configs):
        print(f"\n--- [{i+1}/{len(configs)}] ---")
        work_dir = results_dir / cfg.tech.name
        try:
            result = run_single_fn(cfg, work_dir)
            all_results.append(result)
        except Exception as exc:
            print(f"    ERROR: {exc}")
            all_results.append({"config": cfg, "error": str(exc), "passed": False})

    # Summary
    print(f"\n{'='*78}")
    print("SUMMARY TABLE")
    print(f"{'='*78}")
    n_pass, n_fail, n_error = print_summary_fn(all_results)
    total = len(all_results)

    print(f"\n  Total: {total}  Pass: {n_pass}  Fail: {n_fail}  Error: {n_error}")

    save_csv_fn(all_results, results_dir / "summary.csv")
    plot_bar_fn(all_results, results_dir / "summary.png", title)

    print(f"\n{'='*78}")
    if n_fail > 0:
        print(f"RESULT: {n_fail} FAIL, {n_error} ERROR out of {total}")
        return 1
    if n_error > 0:
        print(f"RESULT: {n_pass} PASS, {n_error} ERROR (modelcard issues) out of {total}")
        if n_pass == 0:
            # every test errored (broken ngspice, throwing solver, ...):
            # nothing was verified, so a green exit would be a lie
            print("RESULT: 0 PASS — all tests ERRORED, nothing verified")
            return 1
    else:
        print(f"RESULT: ALL {n_pass} tests PASSED")
    return 0


# ---------------------------------------------------------------------------
# Generic tech parameter table printer
# ---------------------------------------------------------------------------
def print_tech_params(tech_names: List[str]) -> None:
    """Print technology parameter table."""
    print(f"\n  {'Tech':8s} | {'VDD':>5s} | {'L_n':>5s} | {'L_p':>5s} | "
          f"{'NFIN':>4s} | {'VT':5s} | {'NMOS':15s} | {'PMOS':15s}")
    print("  " + "-" * 80)
    for name in tech_names:
        tech = ALL_TECHS[name]
        vt = tech.default_vt_pair
        print(f"  {tech.name:8s} | {tech.vdd:5.2f} | "
              f"{tech.default_l_nmos*1e9:4.0f}n | {tech.default_l_pmos*1e9:4.0f}n | "
              f"{tech.default_nfin:4d} | {tech.default_vt:5s} | "
              f"{vt.nmos_model:15s} | {vt.pmos_model:15s}")


# ---------------------------------------------------------------------------
# Generic L3 multi-tech orchestrator
# ---------------------------------------------------------------------------
def run_multi_tech_main(
    tech_names: List[str],
    results_dir: Path,
    title: str,
    acceptance_msg: str,
    make_baseline_fn: Callable[[TechProfile], Any],
    build_parametric_fn: Callable[[TechProfile], List[Any]],
    run_single_fn: Callable[[Any, Path], Dict[str, Any]],
    print_summary_fn: Callable[[List[Dict[str, Any]]], Tuple[int, int, int]],
    save_csv_fn: Callable[[List[Dict[str, Any]], Path], None],
    plot_bar_fn: Callable[[List[Dict[str, Any]], Path, str], None],
) -> int:
    """Generic L3 multi-tech orchestrator.

    Per-tech: baseline -> parametric if baseline passes -> summary.
    Handles both exception-based errors AND "error" key in result dicts
    (DC returns error dicts for garbage output).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results: List[Dict[str, Any]] = []
    tech_status: Dict[str, str] = {}

    print("=" * 78)
    print(title)
    print(f"  Technologies: {', '.join(tech_names)}")
    print(f"  Acceptance: {acceptance_msg}")
    print("=" * 78)

    print_tech_params(tech_names)

    for name in tech_names:
        tech = ALL_TECHS[name]
        work_dir = results_dir / tech.name

        # Phase 1: Baseline
        print(f"\n{'='*78}")
        print(f"  {tech.name}: Phase 1 — Baseline")
        print(f"{'='*78}")

        baseline_cfg = make_baseline_fn(tech)
        try:
            result = run_single_fn(baseline_cfg, work_dir)
            all_results.append(result)
            if result["passed"]:
                tech_status[tech.name] = "PASS"
            elif "error" in result:
                print("  => BASELINE ERROR — skipping parametric sweep")
                tech_status[tech.name] = "BASELINE_ERROR"
                continue
            else:
                print("  => BASELINE FAIL — skipping parametric sweep")
                tech_status[tech.name] = "BASELINE_FAIL"
                continue
        except Exception as exc:
            print(f"  => BASELINE ERROR: {exc}")
            all_results.append({"config": baseline_cfg, "error": str(exc), "passed": False})
            tech_status[tech.name] = "BASELINE_ERROR"
            continue

        # Phase 2: Parametric sweep
        print(f"\n  {tech.name}: Phase 2 — Parametric sweep")
        sweep_configs = build_parametric_fn(tech)
        for cfg in sweep_configs:
            try:
                result = run_single_fn(cfg, work_dir)
                all_results.append(result)
            except Exception as exc:
                print(f"    ERROR ({cfg.label}): {exc}")
                all_results.append({"config": cfg, "error": str(exc), "passed": False})

    # Summary
    print(f"\n{'='*78}")
    print("SUMMARY TABLE")
    print(f"{'='*78}")
    n_pass, n_fail, n_error = print_summary_fn(all_results)
    total = len(all_results)

    print(f"\n  Total: {total}  Pass: {n_pass}  Fail: {n_fail}  Error: {n_error}")

    print("\n  Technology status:")
    for name, status in tech_status.items():
        print(f"    {name:8s}: {status}")

    csv_name = results_dir.name.replace("_results", "_summary")
    save_csv_fn(all_results, results_dir / f"{csv_name}.csv")
    plot_bar_fn(all_results, results_dir / f"{csv_name}.png", title)

    print(f"\n{'='*78}")
    if n_fail > 0:
        print(f"RESULT: {n_fail} FAIL, {n_error} ERROR out of {total}")
        return 1
    if n_error > 0:
        print(f"RESULT: {n_pass} PASS, {n_error} ERROR (modelcard issues) out of {total}")
        if n_pass == 0:
            # every test errored (broken ngspice, throwing solver, ...):
            # nothing was verified, so a green exit would be a lie
            print("RESULT: 0 PASS — all tests ERRORED, nothing verified")
            return 1
    else:
        print(f"RESULT: ALL {n_pass} tests PASSED")
    return 0
