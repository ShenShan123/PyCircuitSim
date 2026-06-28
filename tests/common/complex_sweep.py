"""Parametric sweep harness for the four DirectNet complex circuits.

Plan ``docs/plans/2026-06-20-directnet-complex-circuit-sweeps.md``. Mirrors the
inverter sweep harness (``tests/common/nn_sweep.py``) but for the opamp /
ring-oscillator / switched-cap / 6T-SRAM benchmarks. Sweeps technology, VT
(symmetric + asymmetric N/P), geometry (L / NFIN / P-N fin ratio), VDD, and
per-circuit input stimuli, baseline-gated, against the NGSPICE BSIM-CMG
(LEVEL=72) ground truth — never a simplified model (CLAUDE.md Validation rule).

The four single-point ``verify_complex_*.py`` ship gates are UNTOUCHED; this is
additive infra. Hard gates: every swept config is a real PASS/FAIL at the
circuit's domain tolerance. Bake / OSDI-Fatal / absent-checkpoint /
out-of-region → ERROR (never a silent FAIL). 3-state exit code:
  0 = every attempted config PASSED and >=1 config ran,
  1 = any FAIL,
  2 = a requested tech was all-ERROR (could-not-characterize).
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Single-thread torch BEFORE any inference — the opamp gain / SRAM trip are
# high-gain fixed points; multi-thread float reduction perturbs the NR basin
# (V6.4.8 gate methodology). Driver scripts also pin OMP/MKL=1.
try:
    import torch

    torch.set_num_threads(1)
except ImportError:  # pragma: no cover
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tests.common.base import ALL_TECHS, TECH_COLORS  # noqa: E402
from tests.common.complex import (  # noqa: E402
    BENCH, BENCH_TECHS, PROJECT_ROOT, RESULTS_BASE,
    BenchTech, OpAmpParams, RingOscParams, SwitchCapParams, SramParams,
    bench_variant, usable_vts,
    get_baked_modelcard, run_ngspice_wrdata,
    run_directnet_transient, run_directnet_dc_sweep,
    full_metrics, fmt_metrics,
    opamp_bias, gain_trip, period_from_wave, at, snm_from_lobes, sc_windows,
    ngspice_opamp, directnet_opamp,
    ngspice_ringosc, directnet_ringosc,
    ngspice_switchcap, directnet_switchcap,
    ngspice_sram_lobe, directnet_sram_lobe, directnet_sram_6t,
)

CKPT_DIR = PROJECT_ROOT / "external_compact_models" / "bsimar" / "checkpoints"

# --- hard-gate thresholds (plan Step 7) -------------------------------------
OPAMP_GAIN_TOL = 0.10        # |dn-ng|/ng gain gate
OPAMP_MIN_GAIN = 5.0         # ng_gain < this V/V → out-of-region ERROR (M4)
RINGOSC_PERIOD_TOL = 0.05    # period gate
RINGOSC_TSTOP_CAP = 8e-9     # bound the worst ring corner (M3)
SC_CHARGE_TOL = 0.05         # charge transfer, fraction of VDD
SC_DROOP_TOL = 0.10          # hold droop, relative part
SC_DROOP_FLOOR = 1e-3        # absolute floor on the droop gate (0.1% VDD)
SRAM_NRMSE_TOL = 0.10        # butterfly-lobe curve NRMSE vs NGSPICE (B5/B7)

# --- sweep dimensions -------------------------------------------------------
SHARED_DIMS = ["vt_sym", "vt_asym", "l", "nfin", "pn_ratio", "vdd"]
STIM_DIMS: Dict[str, List[str]] = {
    "opamp": ["vcm", "cc", "cl", "span"],
    "ringosc": ["n_stages", "cload"],
    "switchcap": ["vin", "csample", "clk_per", "clk_slew"],
    "sram": ["wl_frac"],
}
# SRAM uses its lobe NFIN as the geometry corner (bench_variant nfin), so its
# "nfin" dim is a real corner sweep; pn_ratio is meaningless for a 6T cell.
SRAM_DIMS = ["nfin", "vt_sym", "vt_asym", "l", "wl_frac", "vdd"]

DEFAULT_STIM = {
    "opamp": OpAmpParams(),
    "ringosc": RingOscParams(),
    "switchcap": SwitchCapParams(),
    "sram": SramParams(),
}


def all_dimensions(circuit: str) -> List[str]:
    if circuit == "sram":
        return list(SRAM_DIMS)
    return SHARED_DIMS + STIM_DIMS[circuit]


# ===========================================================================
# Config
# ===========================================================================
@dataclass
class ComplexSweepConfig:
    bt: BenchTech
    tech_key: str
    circuit: str             # opamp | ringosc | switchcap | sram
    stim: Any                # OpAmpParams | RingOscParams | ...
    sweep_type: str          # baseline | vt_sym | vt_asym | l | nfin | ...
    config_name: str
    swept: Dict[str, object] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.tech_key}_{self.circuit}_{self.config_name}"


def make_baseline(circuit: str, tech_key: str) -> ComplexSweepConfig:
    return ComplexSweepConfig(
        bt=BENCH[tech_key], tech_key=tech_key, circuit=circuit,
        stim=DEFAULT_STIM[circuit], sweep_type="baseline",
        config_name="baseline", swept={})


# ---------------------------------------------------------------------------
# Variant builders
# ---------------------------------------------------------------------------
def _vdd_tag(vdd: float) -> str:
    return f"vdd_{vdd:g}".replace(".", "p")


def _shared_variants(tech_key: str,
                     dimension: str) -> List[Tuple[BenchTech, str, Dict]]:
    """Geometry / VT / VDD variants shared by opamp / ringosc / switchcap /
    sram. One dimension at a time so every point stays inside the
    per-(L, NFIN, defaultVT) PDIBL2 pruning base.py already validated."""
    bt0 = BENCH[tech_key]
    prof = ALL_TECHS[tech_key]
    base_vt = bt0.vt
    usable = sorted(usable_vts(tech_key))
    out: List[Tuple[BenchTech, str, Dict]] = []

    if dimension == "vt_sym":
        for v in usable:
            if v == base_vt:
                continue
            out.append((bench_variant(bt0, nmos_vt=v, pmos_vt=v),
                        f"vt_{v}", {"nmos_vt": v, "pmos_vt": v}))
    elif dimension == "vt_asym":
        if len(usable) >= 2:
            for vn in usable:
                for vp in usable:
                    if vn == vp:
                        continue
                    out.append((bench_variant(bt0, nmos_vt=vn, pmos_vt=vp),
                                f"vt_{vn}_{vp}",
                                {"nmos_vt": vn, "pmos_vt": vp}))
    elif dimension == "l":
        # NMOS-only sweep (PMOS pinned at its default). Skip both the baseline
        # (l_val == l_nmos) AND the value that equals the PMOS default — there
        # the (l_val, l_pmos) pair collapses to (l_val, l_val), a duplicate of
        # the symmetric `lsym_{l_val}` point below (bug report B11). Same logic,
        # mirrored, for the PMOS-only sweep.
        for l_val in prof.l_values:
            if (abs(l_val - bt0.l_nmos) > 1e-15
                    and abs(l_val - bt0.l_pmos) > 1e-15):
                out.append((bench_variant(bt0, l_nmos=l_val),
                            f"ln_{l_val*1e9:.0f}nm",
                            {"l_nmos_nm": round(l_val * 1e9)}))
        for l_val in prof.l_values:
            if (abs(l_val - bt0.l_pmos) > 1e-15
                    and abs(l_val - bt0.l_nmos) > 1e-15):
                out.append((bench_variant(bt0, l_pmos=l_val),
                            f"lp_{l_val*1e9:.0f}nm",
                            {"l_pmos_nm": round(l_val * 1e9)}))
        for l_val in prof.l_values:
            out.append((bench_variant(bt0, l_nmos=l_val, l_pmos=l_val),
                        f"lsym_{l_val*1e9:.0f}nm",
                        {"l_sym_nm": round(l_val * 1e9)}))
    elif dimension == "nfin":
        for k in (3, 5, 10):
            out.append((bench_variant(bt0, nfin=k), f"nfin_{k}", {"nfin": k}))
    elif dimension == "pn_ratio":
        # nfin_p=3 only: ≤ nfin+1, the modelcard NFIN-group rule (review C5).
        out.append((bench_variant(bt0, nfin_p=3), "pn_nfinp3", {"nfin_p": 3}))
    elif dimension == "vdd":
        for dv in (-0.1, -0.05, 0.05, 0.1):
            vdd = round(bt0.vdd + dv, 3)
            if vdd <= 0:
                continue
            out.append((bench_variant(bt0, vdd=vdd), _vdd_tag(vdd),
                        {"vdd": vdd}))
    return out


def _stim_variants(circuit: str,
                   dimension: str) -> List[Tuple[Any, str, Dict]]:
    """Per-circuit input-stimulus variants (vary the stim dataclass)."""
    base = DEFAULT_STIM[circuit]
    out: List[Tuple[Any, str, Dict]] = []
    if circuit == "opamp":
        if dimension == "vcm":
            for f in (0.45, 0.50, 0.60, 0.65):
                out.append((replace(base, vcm_frac=f),
                            f"vcm_{f:g}".replace(".", "p"), {"vcm_frac": f}))
        elif dimension == "cc":
            for c in (10e-15, 40e-15):
                out.append((replace(base, cc=c), f"cc_{c*1e15:g}fF",
                            {"cc_fF": round(c * 1e15)}))
        elif dimension == "cl":
            for c in (20e-15, 100e-15):
                out.append((replace(base, cl=c), f"cl_{c*1e15:g}fF",
                            {"cl_fF": round(c * 1e15)}))
        elif dimension == "span":
            for s in (0.10, 0.20):
                out.append((replace(base, span=s),
                            f"span_{s:g}".replace(".", "p"), {"span": s}))
    elif circuit == "ringosc":
        if dimension == "n_stages":
            for n in (3, 7, 9):
                out.append((replace(base, n_stages=n), f"nstg_{n}",
                            {"n_stages": n}))
        elif dimension == "cload":
            for c in (0.25e-15, 1.0e-15, 2.0e-15):
                out.append((replace(base, cload=c), f"cl_{c*1e15:g}fF",
                            {"cload_fF": c * 1e15}))
    elif circuit == "switchcap":
        if dimension == "vin":
            for f in (0.3, 0.4, 0.8):
                out.append((replace(base, vin_frac=f),
                            f"vin_{f:g}".replace(".", "p"), {"vin_frac": f}))
        elif dimension == "csample":
            for c in (50e-15, 200e-15):
                out.append((replace(base, csample=c), f"cs_{c*1e15:g}fF",
                            {"csample_fF": round(c * 1e15)}))
        elif dimension == "clk_per":
            for t in (2e-9, 8e-9):
                out.append((replace(base, clk_per=t), f"per_{t*1e9:g}ns",
                            {"clk_per_ns": t * 1e9}))
        elif dimension == "clk_slew":
            for t in (0.05e-9, 0.2e-9):
                out.append((replace(base, clk_slew=t),
                            f"slew_{t*1e9:g}ns".replace(".", "p"),
                            {"clk_slew_ns": t * 1e9}))
    elif circuit == "sram":
        if dimension == "wl_frac":
            for f in (0.9,):
                out.append((replace(base, wl_frac=f),
                            f"wl_{f:g}".replace(".", "p"), {"wl_frac": f}))
    return out


def build_parametric(circuit: str, tech_key: str,
                     dimension: str) -> List[ComplexSweepConfig]:
    """All parametric configs for one circuit/tech/dimension ('all' → every
    dimension)."""
    dims = all_dimensions(circuit) if dimension in ("all", "") else [dimension]
    base_stim = DEFAULT_STIM[circuit]
    cfgs: List[ComplexSweepConfig] = []
    for d in dims:
        if d in SHARED_DIMS or (circuit == "sram" and d in
                                ("vt_sym", "vt_asym", "l", "nfin", "vdd")):
            for bt, cname, swept in _shared_variants(tech_key, d):
                cfgs.append(ComplexSweepConfig(
                    bt, tech_key, circuit, base_stim, d, cname, swept))
        else:
            bt0 = BENCH[tech_key]
            for stim, cname, swept in _stim_variants(circuit, d):
                cfgs.append(ComplexSweepConfig(
                    bt0, tech_key, circuit, stim, d, cname, swept))
    return cfgs


# ===========================================================================
# Result helpers
# ===========================================================================
_NAN_METRICS = {"mre_pct": float("nan"), "r2": float("nan"),
                "nrmse_pct": float("nan"), "max_err": float("nan")}


def _err(cfg: ComplexSweepConfig, msg: str) -> Dict[str, Any]:
    return {"config": cfg, "status": "error", "error": msg,
            **_NAN_METRICS, "domain": {}}


def _result(cfg: ComplexSweepConfig, passed: bool, metrics: Dict[str, float],
            domain: Dict[str, Any]) -> Dict[str, Any]:
    return {"config": cfg, "status": "pass" if passed else "fail",
            "error": "", **metrics, "domain": domain}


# ===========================================================================
# Single-test orchestrators
# ===========================================================================
def run_single_opamp(cfg: ComplexSweepConfig, work_dir: Path) -> Dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    bt, p = cfg.bt, cfg.stim
    try:
        baked = get_baked_modelcard(bt, bt.nfin, work_dir,
                                    nfin_p=bt.effective_nfin_p)
    except Exception as exc:  # noqa: BLE001
        return _err(cfg, f"bake: {type(exc).__name__}: {exc}")
    try:
        spec = ngspice_opamp(bt, p, baked)
        data = run_ngspice_wrdata(spec["body"], spec["signals"], work_dir,
                                  f"opamp_{cfg.label}", spec["analysis"])
        ng_s, ng_v = data[:, 0], data[:, 1]
        ng_gain, ng_trip, _ = gain_trip(ng_s, ng_v, bt.vdd)
    except Exception as exc:  # noqa: BLE001
        return _err(cfg, f"ngspice: {type(exc).__name__}: {exc}")
    if not np.isfinite(ng_gain) or ng_gain < OPAMP_MIN_GAIN:
        return _err(cfg, f"out-of-region (ng_gain={ng_gain:.2f}<{OPAMP_MIN_GAIN})")
    try:
        path = work_dir / f"opamp_{cfg.label}.sp"
        path.write_text(directnet_opamp(bt, p))
        res = run_directnet_dc_sweep(path, work_dir, f"opamp_{cfg.label}")
        dn_s = np.asarray(res["inp"])
        dn_v = np.asarray(res["vout"])
        if dn_s.size < 3:
            return _err(cfg, "directnet: empty DC sweep")
        dn_gain, dn_trip, _ = gain_trip(dn_s, dn_v, bt.vdd)
    except Exception as exc:  # noqa: BLE001
        return _err(cfg, f"directnet: {type(exc).__name__}: {exc}")

    lo = max(ng_s.min(), dn_s.min())
    hi = min(ng_s.max(), dn_s.max())
    grid = np.linspace(lo, hi, 300)
    m = full_metrics(np.interp(grid, dn_s, dn_v), np.interp(grid, ng_s, ng_v))
    gain_err = abs(dn_gain - ng_gain) / ng_gain * 100.0
    trip_shift = (dn_trip - ng_trip) * 1e3
    passed = gain_err <= OPAMP_GAIN_TOL * 100
    return _result(cfg, passed, m, {
        "metric": "gain_err%", "value": gain_err, "ng_gain": ng_gain,
        "dn_gain": dn_gain, "gain_err_pct": gain_err, "trip_shift_mV": trip_shift})


def run_single_ringosc(cfg: ComplexSweepConfig,
                       work_dir: Path) -> Dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    bt, p = cfg.bt, cfg.stim
    mid = bt.vdd / 2.0
    try:
        baked = get_baked_modelcard(bt, bt.nfin, work_dir,
                                    nfin_p=bt.effective_nfin_p)
    except Exception as exc:  # noqa: BLE001
        return _err(cfg, f"bake: {type(exc).__name__}: {exc}")

    def _ng(tstop: float):
        spec = ngspice_ringosc(bt, p, baked, tstop)
        data = run_ngspice_wrdata(spec["body"], spec["signals"], work_dir,
                                  f"ro_{cfg.label}", spec["analysis"])
        return data[:, 0], data[:, 1]

    try:
        ng_t, ng_v = _ng(p.tstop)
        ng_per = period_from_wave(ng_t, ng_v, mid, p.settle)
        target = p.tstop
        if np.isfinite(ng_per):
            want = p.settle + p.n_measure_cycles * ng_per
            if want > p.tstop:                       # grow window (plan M3)
                target = min(RINGOSC_TSTOP_CAP, want)
                ng_t, ng_v = _ng(target)
                ng_per = period_from_wave(ng_t, ng_v, mid, p.settle)
    except Exception as exc:  # noqa: BLE001
        return _err(cfg, f"ngspice: {type(exc).__name__}: {exc}")
    if not np.isfinite(ng_per):
        return _err(cfg, "ng period NaN (window too short)")

    try:
        path = work_dir / f"ro_{cfg.label}.sp"
        path.write_text(directnet_ringosc(bt, p, target))
        res, partial, _ = run_directnet_transient(path)
        dn_t = np.asarray(res["time"])
        dn_v = np.asarray(res[f"n{p.n_stages}"])
    except Exception as exc:  # noqa: BLE001
        return _err(cfg, f"directnet: {type(exc).__name__}: {exc}")

    dn_per = period_from_wave(dn_t, dn_v, mid, p.settle)
    t_lo, t_hi = p.settle, min(ng_t[-1], dn_t[-1])
    m = dict(_NAN_METRICS)
    if t_hi > t_lo and len(dn_t) > 3:
        grid = np.arange(t_lo, t_hi, p.tstep)
        m = full_metrics(np.interp(grid, dn_t, dn_v),
                         np.interp(grid, ng_t, ng_v))
    if not np.isfinite(dn_per):       # NN failed to oscillate on a good window
        return _result(cfg, False, m, {
            "metric": "period_err%", "value": float("nan"),
            "ng_period_ps": ng_per * 1e12, "dn_period_ps": float("nan"),
            "period_err_pct": float("nan"), "partial": partial})
    per_err = abs(dn_per - ng_per) / ng_per * 100.0
    passed = per_err <= RINGOSC_PERIOD_TOL * 100
    return _result(cfg, passed, m, {
        "metric": "period_err%", "value": per_err,
        "ng_period_ps": ng_per * 1e12, "dn_period_ps": dn_per * 1e12,
        "period_err_pct": per_err, "partial": partial})


def run_single_switchcap(cfg: ComplexSweepConfig,
                         work_dir: Path) -> Dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    bt, p = cfg.bt, cfg.stim
    se, hs, he, tstop, pw, per = sc_windows(p)
    if not (0.0 < se < tstop and hs < he < tstop):
        return _err(cfg, f"degenerate clock windows (se={se*1e9:.2f} "
                         f"hs={hs*1e9:.2f} he={he*1e9:.2f} ns)")
    try:
        baked = get_baked_modelcard(bt, bt.nfin, work_dir,
                                    nfin_p=bt.effective_nfin_p)
    except Exception as exc:  # noqa: BLE001
        return _err(cfg, f"bake: {type(exc).__name__}: {exc}")
    try:
        spec = ngspice_switchcap(bt, p, baked)
        data = run_ngspice_wrdata(spec["body"], spec["signals"], work_dir,
                                  f"sc_{cfg.label}", spec["analysis"])
        ng_t, ng_v = data[:, 0], data[:, 1]
    except Exception as exc:  # noqa: BLE001
        return _err(cfg, f"ngspice: {type(exc).__name__}: {exc}")
    ng_charge = at(ng_t, ng_v, se)
    ng_droop = at(ng_t, ng_v, hs) - at(ng_t, ng_v, he)

    try:
        path = work_dir / f"sc_{cfg.label}.sp"
        path.write_text(directnet_switchcap(bt, p))
        res, partial, _ = run_directnet_transient(path)
        dn_t = np.asarray(res["time"])
        dn_v = np.asarray(res["vsamp"])
        if dn_t.size < 3:
            return _err(cfg, "directnet: transient truncated")
    except Exception as exc:  # noqa: BLE001
        return _err(cfg, f"directnet: {type(exc).__name__}: {exc}")
    dn_charge = at(dn_t, dn_v, se)
    dn_droop = at(dn_t, dn_v, hs) - at(dn_t, dn_v, he)

    t_hi = min(ng_t[-1], dn_t[-1])
    grid = np.arange(0.0, t_hi, p.tstep)
    m = full_metrics(np.interp(grid, dn_t, dn_v), np.interp(grid, ng_t, ng_v))
    charge_err = abs(dn_charge - ng_charge) / bt.vdd * 100.0
    droop_allow = max(SC_DROOP_TOL * abs(ng_droop), SC_DROOP_FLOOR * bt.vdd)
    droop_ok = abs(dn_droop - ng_droop) <= droop_allow
    passed = (charge_err <= SC_CHARGE_TOL * 100) and droop_ok
    return _result(cfg, passed, m, {
        "metric": "charge_err%", "value": charge_err,
        "charge_err_pct": charge_err, "ng_charge": ng_charge,
        "dn_charge": dn_charge, "droop_pct_alw":
        abs(dn_droop - ng_droop) / droop_allow * 100.0, "partial": partial})


def _sram_force_ic(bt: BenchTech, nfin: int, work_dir: Path) -> bool:
    """Full 6T cell, wl=OFF/hold, both storage states must rail (force_ic)."""
    import logging
    from pycircuitsim.solver import DCSolver
    from tests.common.complex import parse_netlist
    ok_all = True
    for tag, (q0, qb0) in (("state1", (bt.vdd, 0.0)),
                           ("state0", (0.0, bt.vdd))):
        path = work_dir / f"sram6t_{bt.name}_nf{nfin}_{tag}.sp"
        path.write_text(directnet_sram_6t(bt, q0, qb0, nfin))
        logging.disable(logging.CRITICAL)
        ok = False
        try:
            parser = parse_netlist(path)
            circuit = parser.circuit
            guess = circuit.initial_conditions or None
            solver = DCSolver(circuit, initial_guess=guess,
                              use_source_stepping=True, force_ic=True)
            sol = solver.solve()
            q_v = sol.get("q", float("nan"))
            qb_v = sol.get("qb", float("nan"))
            resid = getattr(solver, "_last_dc_residual", None)
            thr = getattr(solver, "_last_dc_resid_threshold", None)
            resid_ok = (resid is not None and thr is not None and resid <= thr)
            band = 0.1 * bt.vdd
            rail_ok = (abs(q_v - q0) < band and abs(qb_v - qb0) < band)
            ok = bool(resid_ok and rail_ok)
        except Exception:  # noqa: BLE001
            ok = False
        finally:
            logging.disable(logging.NOTSET)
        ok_all = ok_all and ok
    return ok_all


def run_single_sram(cfg: ComplexSweepConfig,
                    work_dir: Path) -> Dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    bt, p = cfg.bt, cfg.stim
    nfin = bt.nfin
    try:
        baked = get_baked_modelcard(bt, nfin, work_dir)
    except Exception as exc:  # noqa: BLE001
        return _err(cfg, f"bake: {type(exc).__name__}: {exc}")
    try:
        spec = ngspice_sram_lobe(bt, p, nfin, baked)
        data = run_ngspice_wrdata(spec["body"], spec["signals"], work_dir,
                                  f"sram_{cfg.label}", spec["analysis"])
        ng_q, ng_qb = data[:, 0], data[:, 1]
        ng_snm = snm_from_lobes(ng_q, ng_qb)
    except Exception as exc:  # noqa: BLE001
        return _err(cfg, f"ngspice: {type(exc).__name__}: {exc}")
    try:
        path = work_dir / f"sram_{cfg.label}.sp"
        path.write_text(directnet_sram_lobe(bt, p, nfin))
        res = run_directnet_dc_sweep(path, work_dir, f"sram_{cfg.label}")
        dn_q = np.asarray(res["q"])
        dn_qb = np.asarray(res["qb"])
    except Exception as exc:  # noqa: BLE001
        return _err(cfg, f"directnet: {type(exc).__name__}: {exc}")

    grid = np.linspace(0.0, bt.vdd, 300)
    m = full_metrics(np.interp(grid, dn_q, dn_qb), np.interp(grid, ng_q, ng_qb))
    dn_snm = snm_from_lobes(dn_q, dn_qb)
    dn_min = float(np.min(dn_qb))
    positive = dn_min >= -1e-3
    nrmse_ok = m["nrmse_pct"] <= SRAM_NRMSE_TOL * 100
    snm_err = (abs(dn_snm - ng_snm) / ng_snm * 100.0
               if ng_snm > 1e-6 else float("nan"))
    # B7: the SRAM sweep baseline must use the SAME pass definition as the
    # authoritative single-point ship gate (verify_complex_sram_snm) —
    # positivity AND NGSPICE-NRMSE tracking. force_ic is a self-consistency
    # convergence probe (not an NGSPICE comparison): it is reported but NOT
    # gated, exactly as in the ship gate. Previously the sweep AND'd force_ic,
    # so a force_ic miss (it fails on the steep TSMC5/16 techs) sank the whole
    # SRAM sweep via baseline-gating while the ship gate passed.
    fic_gated = abs(p.wl_frac - 1.0) < 1e-9
    fic_ok = _sram_force_ic(bt, nfin, work_dir) if fic_gated else None
    passed = positive and nrmse_ok
    return _result(cfg, passed, m, {
        "metric": "snm_err%", "value": snm_err, "ng_snm_mV": ng_snm * 1e3,
        "dn_snm_mV": dn_snm * 1e3, "snm_err_pct": snm_err,
        "min_qb_mV": dn_min * 1e3, "positive": positive, "nrmse_ok": nrmse_ok,
        "force_ic": ("ok" if fic_ok else "FAIL") if fic_gated else "—"})


_RUNNERS: Dict[str, Callable] = {
    "opamp": run_single_opamp, "ringosc": run_single_ringosc,
    "switchcap": run_single_switchcap, "sram": run_single_sram,
}


# ===========================================================================
# Checkpoint pin (plan m1)
# ===========================================================================
def checkpoint_manifest(tech_keys: List[str]) -> Dict[str, str]:
    man: Dict[str, str] = {}
    for tk in sorted(set(tech_keys)):
        scope = tk.lower()
        for dev in ("nmos", "pmos"):
            f = CKPT_DIR / f"{scope}_dn_medium_{dev}_best.pt"
            if f.exists():
                man[f"{scope}_{dev}"] = hashlib.sha256(
                    f.resolve().read_bytes()).hexdigest()
            else:
                man[f"{scope}_{dev}"] = "ABSENT"
    return man


def verify_checkpoint_pin(tech_keys: List[str], results_dir: Path,
                          strict: bool = False) -> Dict[str, str]:
    """Record (or, if a manifest exists, verify) the resolved checkpoint
    hashes. The opamp gain / SRAM trip amplify weight drift ~20×, so a silent
    checkpoint change must be caught. ``strict`` fails loud on mismatch.

    The manifest is tech-tagged so per-tech runs (e.g. TSMC16 then TSMC7) keep
    independent manifests instead of spuriously cross-flagging each other."""
    man = checkpoint_manifest(tech_keys)
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(sorted(set(tech_keys)))
    path = results_dir / f"checkpoint_manifest_{tag}.json"
    print("  [ckpt-pin] resolved checkpoints:")
    for k, v in man.items():
        print(f"    {k:18s} {v[:16] if v != 'ABSENT' else 'ABSENT'}")
    if path.exists():
        prev = json.loads(path.read_text())
        drift = {k: (prev.get(k), man.get(k))
                 for k in set(prev) | set(man) if prev.get(k) != man.get(k)}
        if drift:
            msg = f"[ckpt-pin] DRIFT vs recorded manifest: {drift}"
            if strict:
                raise RuntimeError(msg)
            print(f"  {msg}")
        else:
            print("  [ckpt-pin] matches recorded manifest")
    else:
        path.write_text(json.dumps(man, indent=2))
        print(f"  [ckpt-pin] recorded manifest → {path}")
    return man


# ===========================================================================
# Baseline-gated multi-tech driver
# ===========================================================================
def run_complex_multi_tech(circuit: str, tech_keys: List[str], dimension: str,
                           results_dir: Path,
                           pin_strict: bool = False) -> List[Dict[str, Any]]:
    runner = _RUNNERS[circuit]
    verify_checkpoint_pin(tech_keys, results_dir, strict=pin_strict)
    results: List[Dict[str, Any]] = []
    for tk in tech_keys:
        print(f"\n{'=' * 72}\n  {tk} — {circuit} ({dimension})\n{'=' * 72}")
        if tk not in BENCH:
            print(f"  SKIP unknown tech {tk}")
            continue
        base_cfg = make_baseline(circuit, tk)
        wd = results_dir / tk / "baseline"
        res = runner(base_cfg, wd)
        results.append(res)
        _print_line(res)
        if res["status"] != "pass":
            print(f"  baseline {res['status'].upper()} — skipping {tk} sweep")
            continue
        for cfg in build_parametric(circuit, tk, dimension):
            wd = results_dir / tk / cfg.config_name
            res = runner(cfg, wd)
            results.append(res)
            _print_line(res)
    return results


def _print_line(res: Dict[str, Any]) -> None:
    cfg = res["config"]
    if res["status"] == "error":
        print(f"  {cfg.label:<40s}  ERROR: {res['error']}")
        return
    d = res.get("domain", {})
    metric = d.get("metric", "")
    val = d.get("value", float("nan"))
    print(f"  {cfg.label:<40s}  {metric}={val:8.2f}  "
          f"NRMSE={res['nrmse_pct']:6.2f}%  R2={res['r2']:7.4f}  "
          f"{res['status'].upper()}")


# ===========================================================================
# Summary table / CSV / bar plot / exit code
# ===========================================================================
def print_summary(results: List[Dict[str, Any]], circuit: str) -> Dict[str, int]:
    print(f"\n{'=' * 96}\n  SUMMARY — {circuit}\n{'=' * 96}")
    print(f"  {'Config':<40s} | {'Sweep':<9s} | {'Domain':>12s} | "
          f"{'NRMSE%':>7s} | {'Status':>6s}")
    print(f"  {'-' * 92}")
    n_pass = n_fail = n_err = 0
    for res in results:
        cfg = res["config"]
        if res["status"] == "error":
            n_err += 1
            print(f"  {cfg.label:<40s} | {cfg.sweep_type:<9s} | "
                  f"{'—':>12s} | {'—':>7s} | {'ERROR':>6s}")
            continue
        n_pass += res["status"] == "pass"
        n_fail += res["status"] == "fail"
        d = res.get("domain", {})
        dom = f"{d.get('metric','')}={d.get('value',float('nan')):.2f}"
        print(f"  {cfg.label:<40s} | {cfg.sweep_type:<9s} | {dom:>12s} | "
              f"{res['nrmse_pct']:7.2f} | {res['status'].upper():>6s}")
    print(f"  {'-' * 92}")
    print(f"  Total: {len(results)}  Pass: {n_pass}  Fail: {n_fail}  "
          f"Error: {n_err}")
    return {"pass": n_pass, "fail": n_fail, "error": n_err}


def save_summary_csv(results: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    swept_keys: List[str] = []
    dom_keys: List[str] = []
    for res in results:
        for k in res["config"].swept:
            if k not in swept_keys:
                swept_keys.append(k)
        for k in res.get("domain", {}):
            if k not in dom_keys and k not in ("metric", "value"):
                dom_keys.append(k)
    base_cols = ["label", "tech", "circuit", "sweep_type", "config_name",
                 "status", "nrmse_pct", "mre_pct", "r2", "max_err", "error"]
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(base_cols + swept_keys + dom_keys)
        for res in results:
            cfg = res["config"]
            row = [cfg.label, cfg.tech_key, cfg.circuit, cfg.sweep_type,
                   cfg.config_name, res["status"].upper(),
                   f"{res['nrmse_pct']:.4f}", f"{res['mre_pct']:.4f}",
                   f"{res['r2']:.6f}", f"{res['max_err']:.6e}", res["error"]]
            row += [cfg.swept.get(k, "") for k in swept_keys]
            row += [res.get("domain", {}).get(k, "") for k in dom_keys]
            w.writerow(row)
    print(f"  [CSV] {csv_path}")


def plot_summary_bar(results: List[Dict[str, Any]], save_path: Path,
                     title: str) -> None:
    ok = [r for r in results if r["status"] != "error"
          and np.isfinite(r["nrmse_pct"])]
    if not ok:
        print("  [Plot] skipped — no non-error results")
        return
    labels = [r["config"].config_name for r in ok]
    vals = [r["nrmse_pct"] for r in ok]
    colors = [TECH_COLORS.get(r["config"].tech_key, "gray") for r in ok]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.32), 6))
    ax.bar(range(len(labels)), vals, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("waveform NRMSE (%)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {save_path}")


def exit_code(results: List[Dict[str, Any]], tech_keys: List[str]) -> int:
    """3-state: 0 = all attempted PASSED and ≥1 ran; 1 = any FAIL;
    2 = a requested tech was all-ERROR (could-not-characterize)."""
    if any(r["status"] == "fail" for r in results):
        return 1
    ran = [r for r in results if r["status"] != "error"]
    # any requested tech that produced only ERROR rows → could-not-characterize
    for tk in tech_keys:
        rows = [r for r in results if r["config"].tech_key == tk]
        if rows and all(r["status"] == "error" for r in rows):
            return 2
    if not ran:
        return 2
    return 0


# ===========================================================================
# Thin-driver entry point (shared by the four verify_complex_*_sweep.py)
# ===========================================================================
def driver_main(circuit: str) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description=f"DirectNet complex-circuit parametric sweep — {circuit}")
    ap.add_argument("--tech", default="TSMC7,TSMC16",
                    help="comma-separated techs (default TSMC7,TSMC16 — the "
                         "nodes with checkpoints present here)")
    ap.add_argument("--dimension", default="all",
                    help=f"one of {all_dimensions(circuit)} or 'all'")
    ap.add_argument("--pin-strict", action="store_true",
                    help="fail loud on checkpoint drift vs recorded manifest")
    args = ap.parse_args()
    techs = [t.strip() for t in args.tech.split(",")]

    results_dir = RESULTS_BASE / circuit / "sweep"
    print("=" * 96)
    print(f"DirectNet complex-circuit parametric sweep — {circuit}")
    print(f"  Techs: {', '.join(techs)}   Dimension: {args.dimension}")
    print(f"  Ground truth: NGSPICE BSIM-CMG (LEVEL=72)")
    print("=" * 96)

    results = run_complex_multi_tech(circuit, techs, args.dimension,
                                     results_dir, pin_strict=args.pin_strict)
    print_summary(results, circuit)
    techtag = "-".join(techs)
    save_summary_csv(results, results_dir / f"{circuit}_sweep_{techtag}.csv")
    plot_summary_bar(results, results_dir / f"{circuit}_sweep_{techtag}.png",
                     f"{circuit} parametric sweep ({techtag}) — DirectNet vs NGSPICE")
    code = exit_code(results, techs)
    # R-VT coverage honesty (plan Risk R-VT): state where VT dims were empty.
    for tk in techs:
        if tk in BENCH and len(usable_vts(tk)) < 2:
            print(f"  [coverage] {tk}: |usable VT|={len(usable_vts(tk))} → "
                  f"asymmetric-VT dim has no witness on this node")
    print(f"\nRESULT exit-code = {code}  "
          f"(0=all-pass, 1=any-fail, 2=could-not-characterize)")
    return code
