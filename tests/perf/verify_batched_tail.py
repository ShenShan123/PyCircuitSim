#!/usr/bin/env python3
"""Verification suite for the V7.2.0 Phase 2a-full batched denorm tail.

``_MOSFETNNBase._unpack_eval_batch`` replaces the per-device
``_unpack_eval`` loop in ``batch_eval`` with one float64 numpy pass over
the whole (N, out_dim) + Jacobian block. The phase ships **default-on
and bit-identical** — so this gate demands *exact bit equality on every
element*, not a tolerance (plan §8.3 row 2a). Any single mismatched bit
is a FAIL: the tail feeds every stamped current, and the repo has
repeatedly seen a last-bit NN perturbation land a bistable circuit in a
different NR basin.

Bit-identity rests on two §8.1 constraints, which are acceptance
criteria for the shipped code, not guidance:

  C1  The Vds-correction exponential is per-element libm ``math.exp``
      over the masked subset that reaches the exp branch. ``np.exp``
      mismatches libm by 1 ULP on ~4.6 % of arguments and the
      ``1 − exp`` cancellation amplifies that ~60× at small |Vds| —
      precisely the SRAM off-device regime.
  C2  Cast to float64 BEFORE any arithmetic. Under NEP-50 a float32
      array times a Python float stays float32 (~2e-7 rel error, worse
      than VNTOL) while passing every smoke test.

Levels, no NGSPICE (the reference is the shipped scalar tail):
  Level 1: batched == scalar, bit for bit, per element — both NN
    families (DirectNet / BSIM-AR), both polarities, caps on/off,
    over an adversarial voltage box (off-state, |Vds|→0, reverse-taper
    window, rail overshoot in both extrapolation branches, fast path).
  Level 2: the §8.1 constraints are present in the shipped code (source
    tripwire) AND survive a dense sweep of the small-|Vds| cancellation
    regime where a SIMD exp would be caught red-handed.
  Level 3: a mixed NMOS/PMOS group sharing ONE checkpoint (the env
    double-pin case) is unpacked per-device correctly — ``_is_pmos``
    must travel as a per-member array, never a group scalar.
  Level 4: end-to-end ``batch_eval`` on group-of-one devices equals the
    per-device ``_eval`` fallback bit-for-bit (wires: raw rows, cache
    tuples, caps flags).

Usage:
    conda run -n pycircuitsim python tests/perf/verify_batched_tail.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from pycircuitsim.models.mosfet_nn import (  # noqa: E402
    _MOSFETNNBase, _stacked_group_inputs)
from pycircuitsim.models.mosfet_directnet import NMOS_NN, PMOS_NN  # noqa: E402
from pycircuitsim.models.mosfet_bsimar import (  # noqa: E402
    NMOS_BSIMAR, PMOS_BSIMAR)

CKPT = PROJECT_ROOT / "external_compact_models" / "neural_network" / "checkpoints"

KEYS8 = ["id", "gm", "gds", "gmb", "qg", "qd", "qs", "qb"]
KEYS13 = KEYS8 + ["cgg", "cgd", "cgs", "cdg", "cdd"]

# (label, class, checkpoint stem, batch) — small AR checkpoints keep the loop
# affordable; DirectNet runs at production size on two technologies.
CONFIGS = [
    ("DN-tsmc5-nmos", NMOS_NN, "tsmc5_dn_large_nmos", 512),
    ("DN-tsmc5-pmos", PMOS_NN, "tsmc5_dn_large_pmos", 512),
    ("DN-tsmc16-nmos", NMOS_NN, "tsmc16_dn_large_nmos", 512),
    ("DN-tsmc16-pmos", PMOS_NN, "tsmc16_dn_large_pmos", 512),
    ("AR-tsmc5-nmos", NMOS_BSIMAR, "tsmc5_tf_small_nmos", 96),
    ("AR-tsmc5-pmos", PMOS_BSIMAR, "tsmc5_tf_small_pmos", 96),
]

Result = Tuple[str, bool, str]


def _make_device(cls, stem: str, name: str = "M1") -> _MOSFETNNBase:
    return cls(name, ["d", "g", "s", "b"], str(CKPT / f"{stem}_best.pt"),
               L=16e-9, NFIN=10.0, tech_code=0)


def _adversarial_rows(
    n: int, vdd: float, seed: int,
) -> List[Tuple[float, float, float, float]]:
    """Voltage rows hitting every branch of the Vds correction."""
    rng = np.random.default_rng(seed)
    q = n // 4
    vt = max(0.06 * vdd, 0.026)
    specials = np.resize(np.array([
        0.0, 1e-9, -1e-9, 1e-3, -1e-3,                # cancellation regime
        0.5 * vt, -0.5 * vt, 20.0 * vt, -20.0 * vt,   # exp-branch edges
        0.20 * vdd, 0.25 * vdd, 0.30 * vdd,           # taper window edges
        -0.20 * vdd, -0.25 * vdd, -0.30 * vdd,
        vdd, -vdd, 1.5 * vdd, -1.5 * vdd,             # overshoot: quad
        4.0 * vdd, -4.0 * vdd, 8.0 * vdd, -8.0 * vdd  # overshoot: linear
    ]), q)
    v_d = np.concatenate([
        rng.choice([0.0, vdd], size=q) + rng.normal(0, 0.02, q),
        rng.uniform(-0.45 * vdd, 0.45 * vdd, q),
        specials,
        rng.uniform(-1.2 * vdd, 1.2 * vdd, n - 3 * q),
    ])
    v_g = rng.choice([0.0, vdd], size=n) + rng.normal(0, 0.05, n)
    return [(float(d), float(g), 0.0, 0.0) for d, g in zip(v_d, v_g)]


def _forward_block(
    dev: _MOSFETNNBase,
    rows: List[Tuple[float, float, float, float]],
    with_caps: bool,
    devs: Optional[List[_MOSFETNNBase]] = None,
):
    """One stacked forward+autograd, exactly as ``batch_eval`` builds it."""
    devs = devs if devs is not None else [dev] * len(rows)
    v_raw = torch.from_numpy(np.asarray(rows, dtype=np.float32))
    v_norm = dev._clamp_norm_voltages(v_raw)
    x_v = v_norm.detach().requires_grad_(True)
    x_g, tech_codes, pmos_arr = _stacked_group_inputs(dev._nn_model, devs)
    x_full = torch.cat([x_v, x_g], dim=1)
    with torch.enable_grad():
        out = dev._nn_model(x_full, tech_codes=tech_codes)
        grad_id = torch.autograd.grad(
            out[:, dev._mcol("id")].sum(), x_v,
            create_graph=False, retain_graph=with_caps)[0]
        grad_qg = grad_qd = None
        if with_caps:
            grad_qg = torch.autograd.grad(
                out[:, dev._mcol("qg")].sum(), x_v,
                create_graph=False, retain_graph=True)[0]
            grad_qd = torch.autograd.grad(
                out[:, dev._mcol("qd")].sum(), x_v,
                create_graph=False, retain_graph=False)[0]
    return out, grad_id, grad_qg, grad_qd, pmos_arr


def _compare(
    ref_res: List[Dict[str, float]],
    bat_res: List[Dict[str, float]],
    keys: List[str],
) -> Tuple[int, str]:
    bad, first = 0, ""
    for k in keys:
        r = np.array([d[k] for d in ref_res])
        b = np.array([d[k] for d in bat_res])
        neq = int((r != b).sum())
        if neq and not first:
            i = int(np.nonzero(r != b)[0][0])
            first = f"first: key={k} row={i} ref={r[i]!r} bat={b[i]!r}"
        bad += neq
    return bad, first


def level1() -> List[Result]:
    results: List[Result] = []
    for label, cls, stem, n in CONFIGS:
        dev = _make_device(cls, stem)
        rows = _adversarial_rows(n, dev._vdd_estimate, seed=hash(stem) % 997)
        for with_caps in (True, False):
            keys = KEYS13 if with_caps else KEYS8
            out, g_id, g_qg, g_qd, pmos_arr = _forward_block(
                dev, rows, with_caps)
            ref_res = [
                dev._unpack_eval(
                    out[i], g_id[i],
                    g_qg[i] if g_qg is not None else None,
                    g_qd[i] if g_qd is not None else None,
                    rows[i][0], rows[i][2])
                for i in range(n)
            ]
            bat_res = _MOSFETNNBase._unpack_eval_batch(
                dev, out, g_id, g_qg, g_qd, rows, pmos_arr)
            bad, first = _compare(ref_res, bat_res, keys)
            types_ok = all(type(v) is float for v in bat_res[0].values())
            ok = bad == 0 and types_ok
            results.append((
                f"{label} caps={'on ' if with_caps else 'off'}", ok,
                f"{len(keys)}x{n} elements, {bad} mismatched"
                + ("" if types_ok else " [non-float values]")
                + (f" [{first}]" if first else "")))
    return results


def level2() -> List[Result]:
    results: List[Result] = []
    src = inspect.getsource(_MOSFETNNBase._unpack_eval_batch)
    results.append((
        "C1 tripwire: masked libm math.exp, no np.exp",
        "math.exp" in src and "np.exp(" not in src
        and "torch.exp" not in src,
        "source inspection of _unpack_eval_batch"))
    results.append((
        "C2 tripwire: float64 cast + dtype asserts present",
        src.count("astype(np.float64)") >= 3
        and "dtype == np.float64" in src,
        "source inspection of _unpack_eval_batch"))

    # Functional discriminator: dense sweep of the 1−exp cancellation
    # regime (|Vds| ≤ 3·VT). A SIMD exp substituted for libm mismatches
    # ~4.6 % of arguments at 1 ULP — invisible to casual tests, loud here.
    dev = _make_device(NMOS_NN, "tsmc5_dn_large_nmos", name="Mx")
    vdd = dev._vdd_estimate
    vt = max(0.06 * vdd, 0.026)
    n = 20000
    v_d = np.linspace(-3.0 * vt, 3.0 * vt, n)
    rows = [(float(d), float(vdd), 0.0, 0.0) for d in v_d]
    out, g_id, g_qg, g_qd, pmos_arr = _forward_block(dev, rows, True)
    ref_res = [
        dev._unpack_eval(out[i], g_id[i], g_qg[i], g_qd[i],
                         rows[i][0], rows[i][2])
        for i in range(n)
    ]
    bat_res = _MOSFETNNBase._unpack_eval_batch(
        dev, out, g_id, g_qg, g_qd, rows, pmos_arr)
    bad, first = _compare(ref_res, bat_res, KEYS13)
    results.append((
        "C1 functional: cancellation-regime sweep bit-exact",
        bad == 0,
        f"13x{n} elements in |Vds|<=3VT, {bad} mismatched"
        + (f" [{first}]" if first else "")))
    return results


def level3() -> List[Result]:
    # Same checkpoint file for both polarities (what an env double-pin
    # produces): one shared module, one group, mixed _is_pmos.
    stem = "tsmc5_dn_large_nmos"
    dn = _make_device(NMOS_NN, stem, name="Mn")
    dp = _make_device(PMOS_NN, stem, name="Mp")
    assert dn._nn_model is dp._nn_model, "shared-module premise broken"
    devs = [dn, dp, dn, dp]
    n = len(devs)
    vdd = dn._vdd_estimate
    # reverse-corridor + overshoot rows, where polarity changes branches
    rows = [(0.25 * vdd, vdd, 0.0, 0.0), (0.25 * vdd, vdd, 0.0, 0.0),
            (-1.5 * vdd, vdd, 0.0, 0.0), (-1.5 * vdd, vdd, 0.0, 0.0)]
    out, g_id, g_qg, g_qd, pmos_arr = _forward_block(
        dn, rows, True, devs=devs)
    ref_res = [
        devs[i]._unpack_eval(out[i], g_id[i], g_qg[i], g_qd[i],
                             rows[i][0], rows[i][2])
        for i in range(n)
    ]
    bat_res = _MOSFETNNBase._unpack_eval_batch(
        dn, out, g_id, g_qg, g_qd, rows, pmos_arr)
    bad, first = _compare(ref_res, bat_res, KEYS13)
    pmos_ok = pmos_arr.tolist() == [False, True, False, True]
    return [(
        "mixed NMOS/PMOS group unpacks per-device", bad == 0 and pmos_ok,
        f"13x{n} elements, {bad} mismatched; pmos_arr={pmos_arr.tolist()}"
        + (f" [{first}]" if first else ""))]


def level4() -> List[Result]:
    # Group-of-one is the case where batch_eval is documented bit-identical
    # to the per-device path — so the whole wiring (rows, tuples, caps
    # flags, batched tail) must reproduce _eval exactly.
    results: List[Result] = []
    dn = _make_device(NMOS_NN, "tsmc5_dn_large_nmos", name="Mn")
    dp = _make_device(PMOS_NN, "tsmc5_dn_large_pmos", name="Mp")
    voltages = {"d": 0.31, "g": 0.7, "s": 0.0, "b": 0.0}
    for with_caps in (True, False):
        for m in (dn, dp):
            m._caps_required = with_caps
            m.clear_cache()
        _MOSFETNNBase.batch_eval([dn, dp], voltages)
        batched = {m.name: dict(m._eval_cache) for m in (dn, dp)}
        caps_flag = all(m._cache_has_caps == with_caps for m in (dn, dp))
        for m in (dn, dp):
            m.clear_cache()
        single = {m.name: dict(m._eval(voltages)) for m in (dn, dp)}
        same = all(
            batched[nm][k] == single[nm][k]
            for nm in batched for k in single[nm])
        keys_same = all(
            set(batched[nm]) == set(single[nm]) for nm in batched)
        results.append((
            f"batch_eval == _eval (group-of-one, caps="
            f"{'on' if with_caps else 'off'})",
            same and keys_same and caps_flag,
            f"keys={sorted(single['Mn'])}"))
    return results


def main() -> int:
    print("=" * 72)
    print("V7.2.0 Phase 2a-full batched denorm tail verification")
    print("=" * 72)

    all_results: List[Result] = []
    for name, fn in [
        ("Level 1 (batched == scalar, bit-exact, 3 families)", level1),
        ("Level 2 (§8.1 constraints C1/C2)", level2),
        ("Level 3 (mixed-polarity group)", level3),
        ("Level 4 (end-to-end batch_eval wiring)", level4),
    ]:
        print(f"\n--- {name} ---")
        try:
            batch = fn()
        except Exception as exc:  # fail loud, keep going
            import traceback
            traceback.print_exc()
            batch = [(name, False, f"EXCEPTION: {exc}")]
        for label, ok, detail in batch:
            print(f"  [{'PASS' if ok else 'FAIL'}] {label:44s} {detail}")
        all_results.extend(batch)

    n_pass = sum(1 for _, ok, _ in all_results if ok)
    print("\n" + "=" * 72)
    print(f"RESULT: {n_pass}/{len(all_results)} PASS")
    print("=" * 72)
    return 0 if n_pass == len(all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
