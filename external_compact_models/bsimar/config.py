"""Configuration for BSIMAR training and inference.

- Re-exports PyCMG's `nn_config` (tech registry, output columns).
- Defines project paths: checkpoints, results, data.
- Defines training hyperparameter dataclasses for both architectures.
- Defines the tech-variant code registry (discrete tech embedding).

Downstream consumers (pycircuitsim parser, mosfet_directnet, mosfet_bsimar,
tests) should import from here.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

# ── Project paths ────────────────────────────────────────────────────────────
# Path hierarchy (after path-depth collapse):
#   parents[0] = bsimar/                          (BSIMAR_ROOT — package lives at the top of its dir)
#   parents[1] = external_compact_models/
#   parents[2] = <project root>                   (PROJECT_ROOT)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BSIMAR_ROOT = Path(__file__).resolve().parents[0]
import os as _os
CHECKPOINT_DIR = Path(_os.environ["BSIMAR_CHECKPOINT_DIR"]) if "BSIMAR_CHECKPOINT_DIR" in _os.environ else BSIMAR_ROOT / "checkpoints"
RESULTS_DIR = BSIMAR_ROOT / "results"
DATA_DIR = BSIMAR_ROOT / "data" / "datasets"

# ── Make pycmg importable ────────────────────────────────────────────────────
PYCMG_DIR = PROJECT_ROOT / "external_compact_models" / "PyCMG"
_PYCMG_PYPATH = str(PYCMG_DIR)
if _PYCMG_PYPATH not in sys.path:
    sys.path.insert(0, _PYCMG_PYPATH)

# ── Re-export PyCMG's NN config (single source of truth) ─────────────────────
from pycmg.nn_config import (  # noqa: E402
    OSDI_PATH,
    DEFAULT_TEMPERATURE,
    NNTechConfig,
    TECH_CONFIGS,
    OUTPUT_COLUMNS,
    DEFAULT_NFIN_VALUES,
)

# Backward-compat alias retained for existing tests/netlist parser
TechConfig = NNTechConfig


# ── Tech-Variant Code Registry ──────────────────────────────────────────────
# Each (tech, variant) pair gets a stable integer ID for the tech embedding.
# TSMC codes occupy 0-16, UNKNOWN is 17, ASAP7 codes are 18-21,
# TSMC6 codes are 22-24 (appended in V6.9.0 — see note below).
# Total vocabulary size: 25.
#
# During TSMC-only pre-training the model sees codes 0-16 + 17 (UNKNOWN).
# ASAP7 codes (18-21) are added when fine-tuning on ASAP7 data.
#
# CODE-STABILITY INVARIANT: codes are baked into every trained checkpoint's
# embedding rows AND every dataset's *_tech_variant_labels.npy sidecar.
# New techs MUST be appended at the tail of the ordering (after ASAP7),
# never inserted into the legacy TSMC block — inserting would renumber
# codes 4+ and silently corrupt all existing checkpoints/sidecars.

def _build_tech_variant_codes() -> Tuple[
    Dict[Tuple[str, str], int],
    Dict[int, Tuple[str, str]],
    List[Tuple[str, str]],
]:
    """Build the canonical (tech, variant) -> code mapping.

    Returns (forward_map, reverse_map, ordered_list).
    """
    # Order: legacy TSMC techs first (sorted by node), then ASAP7, then
    # late-onboarded techs appended at the tail (code-stability invariant).
    # Within each tech, variants follow the order in TECH_CONFIGS.
    ordered: List[Tuple[str, str]] = []
    for tech_name in ("tsmc5", "tsmc7", "tsmc12", "tsmc16"):
        cfg = TECH_CONFIGS[tech_name]
        for variant in cfg.variant_names:
            ordered.append((tech_name, variant))
    # slot 17 = UNKNOWN (reserved, not in the list)
    for variant in TECH_CONFIGS["asap7"].variant_names:
        ordered.append(("asap7", variant))
    # V6.9.0: TSMC6 onboarded after the ASAP7 block -> codes 22-24.
    for variant in TECH_CONFIGS["tsmc6"].variant_names:
        ordered.append(("tsmc6", variant))

    forward: Dict[Tuple[str, str], int] = {}
    reverse: Dict[int, Tuple[str, str]] = {}
    code = 0
    for tv in ordered:
        if code == 17:
            code = 18  # skip the UNKNOWN slot
        forward[tv] = code
        reverse[code] = tv
        code += 1
    return forward, reverse, ordered


TECH_VARIANT_CODES: Dict[Tuple[str, str], int]
CODE_TO_TECH_VARIANT: Dict[int, Tuple[str, str]]
_TECH_VARIANT_ORDER: List[Tuple[str, str]]
TECH_VARIANT_CODES, CODE_TO_TECH_VARIANT, _TECH_VARIANT_ORDER = (
    _build_tech_variant_codes()
)

UNKNOWN_CODE_ID: int = 17
NUM_TSMC_CODES: int = 17           # codes 0-16 (legacy TSMC block only)
NUM_TSMC_CODES_WITH_UNKNOWN: int = 18  # codes 0-17 (pre-train vocab)
NUM_TOTAL_CODES: int = 25          # codes 0-24 (incl. ASAP7 18-21, TSMC6 22-24)

# ── TSMC6: a knowingly-duplicated tech scope ────────────────────────────────
# TSMC6 was onboarded in V6.9.0, retired on 2026-07-24 (audit D1) and restored
# in V7.1.0 by explicit decision. The retirement finding still stands and has
# not been softened: as far as BSIM-CMG is concerned TSMC6 *is* TSMC7.
# tsmc6_{nmos,pmos}.npz were bit-identical to tsmc7_* in inputs, geometry,
# outputs and sample_class (1,816,830 / 2,187,292 rows) with only
# meta_tech_name differing, and two LEVEL=72 Id-Vgs sweeps matched to the last
# printed digit. The raw PDKs do differ, but every differing key
# (tmi_ver_lod, tmi_ver_isocpode, sfxmin, samax_c, wodx5akvth0) is a TSMC
# TMI-proprietary extension with zero occurrences in the BSIM-CMG Verilog-A.
#
# So a TSMC6 checkpoint is a *second training run on the TSMC7 data*, and any
# TSMC6-vs-TSMC7 difference is training-run luck, not tech fidelity. It is
# carried because it is the project's only controlled repeat experiment.
# TSMC6 holds tail codes 22-24, so its presence or absence renumbers nothing.

# Input layout: 7 continuous features (no process params)
INPUT_COLUMNS: List[str] = [
    "Vd", "Vg", "Vs", "Vb",   # 4 terminal voltages
    "NFIN", "L", "T",          # 3 geometry / operating-condition scalars
]
INPUT_DIM: int = 7


def tech_variant_to_code(tech: str, variant: str) -> int:
    """Look up the integer code for a (tech, variant) pair.

    Returns UNKNOWN_CODE_ID if the pair is not in the registry.
    """
    return TECH_VARIANT_CODES.get(
        (tech.lower(), variant.lower()), UNKNOWN_CODE_ID
    )


# ── Per-tech-scope local variant code maps (V6 dedicated per-tech models) ────
# When training a dedicated TSMC5 (or TSMC7) model the embedding vocabulary
# collapses to that tech's variants, with one extra UNKNOWN slot at the tail.
# Codes are 0-indexed local to the scope and saved into the per-tech
# checkpoint. The parser must remap (tech, variant) through
# ``local_variant_code(scope, ...)`` before passing tech_code to the loaded
# model — the universal code would index out-of-range or pick the wrong row.

VALID_TECH_SCOPES: Tuple[str, ...] = (
    "universal", "tsmc5", "tsmc6", "tsmc7", "tsmc12", "tsmc16",
)


def _build_local_codes(tech_name: str) -> Dict[Tuple[str, str], int]:
    cfg = TECH_CONFIGS[tech_name]
    return {(tech_name, v.lower()): i for i, v in enumerate(cfg.variant_names)}


LOCAL_VARIANT_CODES: Dict[str, Dict[Tuple[str, str], int]] = {
    "tsmc5":  _build_local_codes("tsmc5"),
    "tsmc6":  _build_local_codes("tsmc6"),
    "tsmc7":  _build_local_codes("tsmc7"),
    "tsmc12": _build_local_codes("tsmc12"),
    "tsmc16": _build_local_codes("tsmc16"),
}
LOCAL_UNKNOWN_CODE_ID: Dict[str, int] = {
    scope: len(table) for scope, table in LOCAL_VARIANT_CODES.items()
}
LOCAL_VOCAB_SIZE: Dict[str, int] = {
    scope: len(table) + 1 for scope, table in LOCAL_VARIANT_CODES.items()
}


def local_variant_code(scope: str, tech: str, variant: str) -> int:
    """Look up the local-vocab tech code for a per-tech model.

    For ``scope == "universal"`` falls through to ``tech_variant_to_code``.
    """
    if scope == "universal":
        return tech_variant_to_code(tech, variant)
    if scope not in LOCAL_VARIANT_CODES:
        raise ValueError(
            f"Unknown tech_scope: {scope!r} (valid: {VALID_TECH_SCOPES})")
    return LOCAL_VARIANT_CODES[scope].get(
        (tech.lower(), variant.lower()),
        LOCAL_UNKNOWN_CODE_ID[scope],
    )


# Tech pairs this project knowingly carries as duplicates. Only tsmc6/tsmc7 is
# on the list, for the reasons above. Do NOT add a pair here to silence a
# genuine onboarding mistake — the whole point of the guard is that a PDK can
# differ substantially on disk while every differing key is inert.
ACKNOWLEDGED_DUPLICATE_TECHS: FrozenSet[Tuple[str, str]] = frozenset(
    {("tsmc6", "tsmc7"), ("tsmc7", "tsmc6")})


def _bsimcmg_implemented_params() -> frozenset:
    """Parameter names the open BSIM-CMG Verilog-A actually implements.

    A modelcard key outside this set is inert: the OSDI binary never reads it,
    so two techs differing only in such keys produce identical currents.
    """
    import re
    from pathlib import Path as _Path

    va_dir = _Path(__file__).resolve().parents[1] / "PyCMG" / "bsim-cmg-va" / "code"
    if not va_dir.is_dir():
        raise FileNotFoundError(
            f"BSIM-CMG Verilog-A source not found at {va_dir} — cannot verify "
            "tech distinctness. Run `git submodule update --init --recursive`.")
    decl = re.compile(r"^\s*parameter\s+(?:real|integer)\s+(\w+)", re.MULTILINE)
    names = set()
    for src in sorted(va_dir.iterdir()):
        if src.suffix in (".va", ".include"):
            names.update(
                m.lower() for m in decl.findall(src.read_text(errors="replace")))
    if not names:
        raise RuntimeError(f"No parameter declarations parsed from {va_dir}")
    return frozenset(names)


def assert_tech_is_distinct(tech: str, against: Optional[Sequence[str]] = None,
                            ) -> None:
    """Raise if ``tech`` is electrically indistinguishable from another tech.

    Guards the failure that put TSMC6 in the registry for two campaigns: a PDK
    can differ substantially on disk while every differing key is a vendor
    extension the open BSIM-CMG ignores, so the "new" tech trains on data
    bit-identical to an existing one. Call this before onboarding a tech, not
    after gating it.

    Compares resolved modelcards restricted to parameters BSIM-CMG actually
    implements. Techs sharing an identical implemented-parameter fingerprint on
    every device are the same technology as far as this simulator is concerned.

    A pair listed in ``ACKNOWLEDGED_DUPLICATE_TECHS`` is reported loudly and
    allowed through, so that a deliberately-carried duplicate (tsmc6/tsmc7)
    does not require disabling the guard for every other tech.

    Raises:
        ValueError: if ``tech`` collides with a tech that is not an
            acknowledged duplicate.
    """
    from pycmg.parser import parse_modelcard      # noqa: E402
    from pycmg.tech import resolve_modelcard       # noqa: E402

    implemented = _bsimcmg_implemented_params()
    others = list(against) if against is not None else [
        t for t in TECH_CONFIGS if t.lower() != tech.lower()]

    def fingerprint(name: str) -> Dict[str, Tuple]:
        cfg = TECH_CONFIGS[name]
        pycmg_tech = cfg.pycmg_tech
        out: Dict[str, Tuple] = {}
        for device_type in ("nmos", "pmos"):
            for variant in cfg.variant_names:
                combos = cfg.get_geometry_combos(device_type, variant)
                if not combos:
                    continue
                L, NFIN = combos[0]
                dev = pycmg_tech.get_device(f"{device_type}_{variant}")
                card = resolve_modelcard(dev, pycmg_tech, L=L, NFIN=NFIN)
                params = parse_modelcard(card, dev.model_name).params
                out[f"{device_type}_{variant}"] = tuple(sorted(
                    (k, v) for k, v in params.items()
                    if k.lower() in implemented))
        return out

    mine = fingerprint(tech)
    for other in others:
        try:
            theirs = fingerprint(other)
        except Exception:      # a tech whose PDK is absent cannot collide
            continue
        if set(mine) == set(theirs) and all(
                mine[k] == theirs[k] for k in mine):
            msg = (f"Technology {tech!r} is electrically identical to "
                   f"{other!r} under BSIM-CMG: every device's "
                   f"implemented-parameter fingerprint matches, so both train "
                   f"on the same data. The PDKs may differ on disk, but only "
                   f"in keys the open BSIM-CMG does not implement "
                   f"(docs/2026-07-21-systematic-audit.md D1).")
            if (tech.lower(), other.lower()) in ACKNOWLEDGED_DUPLICATE_TECHS:
                print(f"[tech-guard] ACKNOWLEDGED DUPLICATE: {msg} "
                      f"Carried deliberately; its rows are a repeat run, not a "
                      f"sixth technology.")
                continue
            raise ValueError(
                msg + f" Do not onboard {tech!r} as a separate technology "
                      f"(this is exactly how TSMC6 entered the registry).")


def tech_scope_vocab_size(scope: str) -> int:
    """Embedding vocabulary size required for a given tech_scope.

    ``"universal"`` means the TSMC-only pre-train vocabulary — codes 0-16
    plus UNKNOWN 17, i.e. 18 rows — NOT every code in the registry. ASAP7
    owns codes 18-21 (Rule 14 keeps it out of scope, and no ``asap7_*.npz``
    exists), so a dataset carrying one would index past an 18-row
    ``nn.Embedding``; training on ASAP7 needs an explicit
    ``--num-tech-codes 22`` and an ASAP7-aware checkpoint. Widening this
    return value instead would resize the embedding for every universal
    run and break ``load_state_dict`` on the 18-code ``u716_dn_*``
    checkpoints, so the range is guarded at load time
    (``trainer._assert_codes_in_vocab``, audit C6q) rather than here.
    """
    if scope == "universal":
        return NUM_TSMC_CODES_WITH_UNKNOWN
    if scope not in LOCAL_VOCAB_SIZE:
        raise ValueError(
            f"Unknown tech_scope: {scope!r} (valid: {VALID_TECH_SCOPES})")
    return LOCAL_VOCAB_SIZE[scope]


# ── Training hyperparameters ─────────────────────────────────────────────────
@dataclass
class DirectNetConfig:
    """Training hyperparameters for the DirectNet (MLP baseline)."""
    # Data
    batch_size: int = 1024
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    # Architecture
    trunk_hidden: int = 128
    trunk_layers: int = 3
    # Optimization
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 500
    patience: int = 50


@dataclass
class TransformerConfig:
    """Training hyperparameters for the BSIM-AR Transformer model."""
    # Architecture
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.2
    # Optimization
    batch_size: int = 1024
    max_epochs: int = 500
    lr: float = 8e-4
    weight_decay: float = 1e-4
    # Early stopping
    patience: int = 30
    delta: float = 1e-5


@dataclass
class TabPFNConfig:
    """Training hyperparameters for the TabPFN-style compact model (V6.9)."""
    # Architecture (see bsimar/models/tabpfn.py)
    embed_dim: int = 96
    n_inducing: int = 32
    dist_blocks: int = 3
    dist_heads: int = 6
    agg_blocks: int = 3
    agg_heads: int = 6
    n_cls_tokens: int = 2
    icl_num_blocks: int = 4
    icl_heads: int = 6
    ctx_len: int = 2048
    use_rope: bool = True
    ff_factor: int = 2
    feature_group_size: int = 3
    # Optimization
    batch_size: int = 1024
    max_epochs: int = 150
    lr: float = 5e-4
    weight_decay: float = 1e-4
    # Early stopping
    patience: int = 40
    delta: float = 1e-5
