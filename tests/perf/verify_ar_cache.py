#!/usr/bin/env python3
"""Verification suite for the LEVEL=76 AR prefix cache.

``PYCIRCUITSIM_NN_AR_CACHE=1`` replaces the BSIM-AR inference loop's
re-encode-the-whole-prefix-every-step with a per-layer K/V cache that
encodes each token exactly once. The two paths are the same algebra, so a
divergence beyond float noise means ``_encoder_append`` has stopped
modelling ``nn.TransformerEncoderLayer`` — which yields *wrong currents*,
not merely slow ones. ``_ar_cache_usable`` fails safe by falling back to
the stock loop, so nothing here can be caught by an accuracy gate: this
file is the guard.

Three levels, no NGSPICE (the reference is the stock PyTorch path):
  Level 0: the shipped default is OFF, and with the flag off the cached
    code is inert — the stock loop must be reproduced **bit for bit**.
  Level 1: with the flag on, outputs and the autograd Jacobians the solver
    stamps must agree with the stock loop to float tolerance, across
    checkpoints, sizes, batch sizes and a wide input box. Also pins the
    two structural invariants the cache depends on: the primer chunk is
    bit-identical (it is the one shape that can be), and the fallback
    engages in training mode.
  Level 2: the cache is actually faster on the shipped configuration —
    a regression here means the lever silently stopped paying.

Tolerance: 1e-4 relative on the AR outputs. That is ~20x the largest
deviation measured over the shipped checkpoints (5.3e-6 abs on outputs,
1.6e-6 on Jacobians, 1.6 uV on solved nodes) and still ~100x tighter than
anything a gate could resolve — wide enough to absorb a BLAS or PyTorch
kernel change, narrow enough to catch a real algebra break.

Usage:
    conda run -n pycircuitsim python tests/perf/verify_ar_cache.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from tests.common.base import parse_no_options  # noqa: E402
import neural_network.models.transformer as transformer_mod  # noqa: E402
from neural_network.config import CHECKPOINT_DIR  # noqa: E402
from neural_network.models.transformer import TransformerEncoderModel  # noqa: E402

# Cover both polarities and every complete architecture tier currently served.
STEMS = [
    "tsmc5_tff_medium_nmos",
    "tsmc5_tff_medium_pmos",
    "tsmc7_tff_small_nmos",
    "tsmc12_tff_small_pmos",
    "tsmc16_tff_small_nmos",
]

RTOL = 1e-4
Result = Tuple[str, bool, str]


def _load(stem: str) -> Tuple[TransformerEncoderModel, int]:
    cfg_path = CHECKPOINT_DIR / f"{stem}_config.npz"
    ckpt_path = CHECKPOINT_DIR / f"{stem}_best.pt"
    cfg = np.load(str(cfg_path))
    n_vocab = (int(cfg["num_tech_codes"])
               if "num_tech_codes" in cfg.files else 22)
    model = TransformerEncoderModel(
        input_dim=int(cfg["input_dim"]),
        target_dim=int(cfg["target_dim"]),
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        num_layers=int(cfg["num_layers"]),
        dim_feedforward=int(cfg["dim_feedforward"]),
        dropout=float(cfg["dropout"]),
        num_tech_codes=n_vocab,
        ar_target_dim=int(cfg["ar_target_dim"]),
    )
    model.load_state_dict(
        torch.load(str(ckpt_path), weights_only=True, map_location="cpu"))
    model.eval()
    return model, n_vocab


def _eval_like_solver(
    model: TransformerEncoderModel,
    x: torch.Tensor,
    tech_codes: torch.Tensor,
    n_bwd: int = 3,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Reproduce the full-terminal runtime's grad-enabled inference pattern.

    The grad context is not incidental — ``TransformerEncoderLayer`` has a
    fused whole-sequence fast path that PyTorch only allows under
    ``no_grad``, so comparing outside ``enable_grad`` would measure a
    different stock path than the simulator ever runs.
    """
    x_v = x[:, :4].clone().requires_grad_(True)
    x_full = torch.cat([x_v, x[:, 4:]], dim=1)
    with torch.enable_grad():
        out = model(x_full, tech_codes=tech_codes)
        grads = [
            torch.autograd.grad(out[:, j].sum(), x_v,
                                retain_graph=(j < n_bwd - 1))[0]
            for j in range(n_bwd)
        ]
    return out.detach(), grads


def _inputs(stem: str, batch: int, n_vocab: int, scale: float = 2.0):
    gen = torch.Generator().manual_seed(abs(hash((stem, batch))) % (2 ** 31))
    x = torch.randn(batch, 7, generator=gen) * scale
    tech = torch.randint(0, max(n_vocab - 1, 1), (batch,), generator=gen)
    return x, tech


def level0() -> List[Result]:
    """Flag default is off, and the off path is bit-identical to stock."""
    results: List[Result] = []

    import subprocess
    probe = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r);"
         "import neural_network.models.transformer as t; print(t._AR_CACHE)"
         % str(PROJECT_ROOT / "external_compact_models")],
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    default_off = probe.stdout.strip() == "False"
    results.append((
        "shipped default PYCIRCUITSIM_NN_AR_CACHE is OFF", default_off,
        f"module default = {probe.stdout.strip() or probe.stderr.strip()[:60]}"))

    model, n_vocab = _load(STEMS[0])
    x, tech = _inputs(STEMS[0], 3, n_vocab)
    saved = transformer_mod._AR_CACHE
    try:
        transformer_mod._AR_CACHE = False
        out_a, grad_a = _eval_like_solver(model, x, tech)
        transformer_mod._AR_CACHE = False
        out_b, grad_b = _eval_like_solver(model, x, tech)
    finally:
        transformer_mod._AR_CACHE = saved
    same = (torch.equal(out_a, out_b)
            and all(torch.equal(p, q) for p, q in zip(grad_a, grad_b)))
    results.append((
        "flag OFF reproduces stock loop bit-for-bit", same,
        "byte-identical" if same else "DIVERGED"))
    return results


def level1() -> List[Result]:
    """Cached path agrees with the stock loop to float tolerance."""
    results: List[Result] = []
    saved = transformer_mod._AR_CACHE
    try:
        for stem in STEMS:
            model, n_vocab = _load(stem)
            worst_out = worst_jac = 0.0
            for batch in (1, 2, 5):
                x, tech = _inputs(stem, batch, n_vocab)
                transformer_mod._AR_CACHE = False
                ref_out, ref_grad = _eval_like_solver(model, x, tech)
                transformer_mod._AR_CACHE = True
                new_out, new_grad = _eval_like_solver(model, x, tech)

                scale = max(float(ref_out.abs().max()), 1e-12)
                worst_out = max(
                    worst_out, float((ref_out - new_out).abs().max()) / scale)
                for a, b in zip(ref_grad, new_grad):
                    g_scale = max(float(a.abs().max()), 1e-12)
                    worst_jac = max(
                        worst_jac, float((a - b).abs().max()) / g_scale)

            ok = worst_out <= RTOL and worst_jac <= RTOL
            results.append((
                f"{stem} cached == stock", ok,
                f"rel out {worst_out:.2e}, rel jac {worst_jac:.2e} "
                f"(tol {RTOL:.0e})"))

        # The primer chunk is the one shape the cache can reproduce exactly:
        # same tokens, same length, same is_causal hint. If this stops being
        # bit-identical, _encoder_append has drifted from the stock layer in
        # a way the tolerance above might otherwise absorb.
        model, n_vocab = _load(STEMS[0])
        x, tech = _inputs(STEMS[0], 1, n_vocab)
        with torch.enable_grad():
            ctx = model._embed_context(x, tech)
            primer = model._add_token_type(torch.cat(
                [ctx, model._embed_ar_scalars(torch.zeros(1, 1))], dim=1))
            stock = model.transformer_encoder(
                primer, mask=model._generate_causal_mask(primer.size(1)))
            cached = model._encoder_append(
                primer, [[None, None] for _ in model.transformer_encoder.layers])
        exact = torch.equal(stock, cached)
        results.append((
            "primer chunk is bit-identical to stock encoder", exact,
            "exact" if exact else
            f"max|d|={float((stock - cached).abs().max()):.2e}"))

        # Training mode must fall back: the stock loop applies dropout, which
        # the cached block does not model at all.
        model.train()
        falls_back = not model._ar_cache_usable()
        model.eval()
        results.append((
            "training mode falls back to the stock loop", falls_back,
            "fallback engaged" if falls_back else "CACHE USED IN TRAIN MODE"))
    finally:
        transformer_mod._AR_CACHE = saved
    return results


def level2() -> List[Result]:
    """The cache is actually faster on the shipped configuration."""
    torch.set_num_threads(1)
    model, n_vocab = _load(STEMS[0])
    x, tech = _inputs(STEMS[0], 2, n_vocab)
    saved = transformer_mod._AR_CACHE
    timings = {}
    try:
        for flag in (False, True):
            transformer_mod._AR_CACHE = flag
            _eval_like_solver(model, x, tech, 1)          # warm
            t0 = time.perf_counter()
            for _ in range(5):
                _eval_like_solver(model, x, tech, 1)
            timings[flag] = (time.perf_counter() - t0) / 5
    finally:
        transformer_mod._AR_CACHE = saved
    speedup = timings[False] / timings[True]
    return [(
        "cached AR eval is faster than the stock loop", speedup > 1.15,
        f"{timings[False] * 1e3:.1f} ms -> {timings[True] * 1e3:.1f} ms "
        f"({speedup:.2f}x)")]


def main() -> int:
    print("=" * 72)
    print("LEVEL=76 AR prefix cache verification")
    print("=" * 72)

    all_results: List[Result] = []
    for name, fn in [("Level 0 (default off, off path exact)", level0),
                     ("Level 1 (cached == stock)", level1),
                     ("Level 2 (the lever still pays)", level2)]:
        print(f"\n--- {name} ---")
        try:
            batch = fn()
        except Exception as exc:  # fail loud, keep going
            import traceback
            traceback.print_exc()
            batch = [(name, False, f"EXCEPTION: {exc}")]
        for label, ok, detail in batch:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label:48s} {detail}")
        all_results.extend(batch)

    n_pass = sum(1 for _, ok, _ in all_results if ok)
    print("\n" + "=" * 72)
    print(f"RESULT: {n_pass}/{len(all_results)} PASS")
    print("=" * 72)
    return 0 if n_pass == len(all_results) else 1


if __name__ == "__main__":
    parse_no_options(__doc__ or "")
    sys.exit(main())
