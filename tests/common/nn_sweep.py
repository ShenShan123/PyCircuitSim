"""V6.3.2 parametric NN test harness — shared config + orchestration.

Mirrors the BSIM-CMG L3 harness (``tests/common/bsimcmg_{dc,tran}.py`` +
``tests/verify_multi_tech_{dc,tran}.py``) for DirectNet (LEVEL=73) NN models.
Two driver scripts consume this module:

  - ``verify_nn_multi_tech_dc.py``   — single-device Id-Vgs over L / NFIN / VT
  - ``verify_nn_multi_tech_tran.py`` — inverter VTC + transient over P/N ratio,
                                       VDD, Cload, input slew, pulse width

Device geometry / VT / VDD sweeps ride on ``dataclasses.replace()`` of the
existing ``TestTechConfig``; only the inverter-transient circuit knobs and the
P/N-ratio NFIN split needed a behaviour-preserving refactor of
``verify_nn_dc_tran.py`` (see ``InvCircuitParams`` there).

Reproducibility note: run against a *stable* checkpoint set. The inverter VTC
has gain ~-15..-30 at the trip point, so any perturbation of the NN weights —
e.g. the checkpoint files being overwritten by a concurrent retrain — is
amplified ~20x into the VTC NRMSE. Point ``bsimar/checkpoints/`` at a fixed
copy, not a symlink into a directory under active training. The harness also
pins ``torch`` to one thread; driver scripts should be invoked with
``OMP_NUM_THREADS=1 MKL_NUM_THREADS=1``.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

# Single-thread torch BEFORE any inference happens — see module docstring.
try:  # torch is always present in the pycircuitsim env; guard for safety.
    import torch

    torch.set_num_threads(1)
except ImportError:  # pragma: no cover
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tests.common.base import ALL_TECHS  # noqa: E402  (base.py TechProfile registry)
from tests.common.nn import mre, nrmse  # noqa: E402
from tests.verify_nn_dc_tran import (  # noqa: E402
    ALL_TEST_TECHS,
    INV_TRAN_TD,
    INV_TRAN_TF,
    INV_TRAN_TR,
    INV_TRAN_TSTOP,
    TRAN_STARTUP_EXCL,
    TECH_COLORS,
    InvCircuitParams,
    TestTechConfig,
    get_available_checkpoints,
    run_ngspice_inverter_tran,
    run_ngspice_inverter_vtc,
    run_ngspice_nmos_dc,
    run_ngspice_pmos_dc,
    run_pycircuitsim_nn_inverter_tran,
    run_pycircuitsim_nn_inverter_vtc,
    run_pycircuitsim_nn_nmos_dc,
    run_pycircuitsim_nn_pmos_dc,
)

# Techs in scope for V6.3.2: the four TSMC nodes with V6.3.1 DirectNet
# checkpoints. ASAP7 excluded (project Rule 17).
NN_TECHS: List[str] = ["TSMC5", "TSMC7", "TSMC12", "TSMC16"]

# Acceptance thresholds (NRMSE %). Loose, like the legacy NN gate — these
# are stress tests, not the tight inverter gate.
DC_NRMSE_PASS = 10.0
INV_NRMSE_PASS = 15.0

NN_LEVEL = 73          # DirectNet
NN_MODEL_NAME = "directnet_v4"


# ---------------------------------------------------------------------------
# Metrics — one helper for DC / VTC / transient (project Rule 16: always
# report MRE %, R^2, NRMSE %, Max error).
# ---------------------------------------------------------------------------
def curve_metrics(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    x_min: Optional[float] = None,
) -> Dict[str, float]:
    """Interpolate ``test`` onto the ``ref`` grid over the common x-range.

    Returns ``nrmse`` (%), ``mre`` (%), ``r2``, ``max_err`` (raw y units).
    ``x_min`` optionally clips the low end (used to drop transient startup).
    """
    lo = max(ref_x[0], test_x[0])
    hi = min(ref_x[-1], test_x[-1])
    if x_min is not None:
        lo = max(lo, x_min)
    mask = (ref_x >= lo - 1e-12) & (ref_x <= hi + 1e-12)
    rx = ref_x[mask]
    ry = ref_y[mask]
    if len(rx) < 3:
        raise RuntimeError("too few overlapping points for comparison")
    ti = np.interp(rx, test_x, test_y)
    ss_res = float(np.sum((ry - ti) ** 2))
    ss_tot = float(np.sum((ry - ry.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "nrmse": nrmse(ti, ry),
        "mre": mre(ti, ry),
        "r2": r2,
        "max_err": float(np.max(np.abs(ti - ry))),
    }


# ---------------------------------------------------------------------------
# Sweep configs
# ---------------------------------------------------------------------------
@dataclass
class NNDCSweepConfig:
    """One single-device Id-Vgs sweep point. ``tech`` is sim-ready (already
    geometry/VT-replaced); ``tech_key`` is the original node for grouping."""

    tech: TestTechConfig
    tech_key: str
    device: str              # "nmos" | "pmos"
    sweep_type: str          # "baseline" | "l" | "nfin" | "vt"
    config_name: str
    swept: Dict[str, object] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.tech_key}_{self.device}_{self.config_name}"


@dataclass
class NNInvSweepConfig:
    """One inverter sweep point. ``tech`` carries geometry/VDD/P-N overrides;
    ``circuit`` carries transient circuit knobs (None for VTC)."""

    tech: TestTechConfig
    tech_key: str
    analysis: str            # "vtc" | "tran"
    circuit: Optional[InvCircuitParams]
    sweep_type: str          # "baseline"|"pn_ratio"|"vdd"|"cload"|"slew"|"pw"
    config_name: str
    swept: Dict[str, object] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.tech_key}_{self.analysis}_{self.config_name}"


# ---------------------------------------------------------------------------
# Builders — single-device DC
# ---------------------------------------------------------------------------
NFIN_SWEEP_VALUES = [5, 10]   # symmetric NFIN sweep (default_nfin=2 skipped)


def make_dc_baseline(tech_key: str, device: str) -> NNDCSweepConfig:
    """Baseline single-device config — tech defaults, no overrides."""
    return NNDCSweepConfig(
        tech=ALL_TEST_TECHS[tech_key], tech_key=tech_key, device=device,
        sweep_type="baseline", config_name="baseline",
    )


def build_dc_parametric(tech_key: str, device: str) -> List[NNDCSweepConfig]:
    """L / NFIN / VT sweep configs for a single device.

    Off-bin L points (e.g. 24nm vs the per-tech NN training bins
    {16,20,36,72,120}nm) exercise NN extrapolation; elevated NRMSE there is
    expected model behaviour, not a harness fault.
    """
    base = ALL_TEST_TECHS[tech_key]
    profile = ALL_TECHS[tech_key]          # base.py TechProfile (l_values, vt_pairs)
    cfgs: List[NNDCSweepConfig] = []
    base_l = base.l_nmos if device == "nmos" else base.effective_l_pmos

    # L sweep — per-tech modelcard L values, skip the baseline L.
    for l_val in profile.l_values:
        if abs(l_val - base_l) < 1e-12:
            continue
        l_nm = round(l_val * 1e9)
        kw = {"l_nmos": l_val} if device == "nmos" else {"l_pmos": l_val}
        tech = replace(base, name=f"{tech_key}_l{l_nm}", **kw)
        cfgs.append(NNDCSweepConfig(
            tech, tech_key, device, "l", f"l_{l_nm}nm", {"l_nm": l_nm}))

    # NFIN sweep — symmetric, skip the default.
    for nfin in NFIN_SWEEP_VALUES:
        if nfin == base.nfin:
            continue
        tech = replace(base, name=f"{tech_key}_nfin{nfin}", nfin=nfin)
        cfgs.append(NNDCSweepConfig(
            tech, tech_key, device, "nfin", f"nfin_{nfin}", {"nfin": nfin}))

    # VT sweep — per-tech variants, skip the default.
    default_vt = base.nn_vt if device == "nmos" else base.effective_pmos_vt
    for vp in profile.vt_pairs:
        if vp.vt_name == default_vt:
            continue
        if device == "nmos":
            kw = {"nmos_model": vp.nmos_model, "nn_vt": vp.vt_name}
        else:
            kw = {"pmos_model": vp.pmos_model, "nn_pmos_vt": vp.vt_name}
        tech = replace(base, name=f"{tech_key}_vt{vp.vt_name}", **kw)
        cfgs.append(NNDCSweepConfig(
            tech, tech_key, device, "vt", f"vt_{vp.vt_name}",
            {"vt": vp.vt_name}))

    return cfgs


# ---------------------------------------------------------------------------
# Builders — inverter VTC / transient
# ---------------------------------------------------------------------------
def _adaptive_tstop(tr: float, tf: float, pw: float, td: float) -> float:
    """tstop large enough for the full rise+hold+fall to settle."""
    per = tr + pw + tf + max(pw, 1.0e-9)
    return max(INV_TRAN_TSTOP, td + per + 0.5e-9)


def make_inv_baseline(tech_key: str, analysis: str) -> NNInvSweepConfig:
    """Baseline inverter config — tech defaults, default circuit."""
    circuit = None if analysis == "vtc" else InvCircuitParams()
    return NNInvSweepConfig(
        tech=ALL_TEST_TECHS[tech_key], tech_key=tech_key, analysis=analysis,
        circuit=circuit, sweep_type="baseline", config_name="baseline",
    )


def build_inv_parametric(tech_key: str, analysis: str) -> List[NNInvSweepConfig]:
    """P/N-ratio + VDD sweeps (VTC & tran); Cload + slew + PW (tran only).

    The P/N-ratio sweep is bounded by the TSMC naive-modelcard NFIN-group rule
    (``nfin_p > nfin+1`` skipped) exactly as the BSIM-CMG harness
    (``verify_multi_tech_tran.py``) — with default NFIN=2 this admits a single
    point, ``nfin_p=3``. The limiter is the modelcard, not the NN model.
    """
    base = ALL_TEST_TECHS[tech_key]
    cfgs: List[NNInvSweepConfig] = []
    default_circuit = None if analysis == "vtc" else InvCircuitParams()

    # P/N ratio — PMOS fin count varies, NMOS stays at default.
    nfin0 = base.effective_inv_nfin
    for ratio in (0.5, 1.5, 2.0):
        nfin_p = max(2, round(nfin0 * ratio))
        if nfin_p == nfin0 or nfin_p > nfin0 + 1:
            continue
        tech = replace(base, name=f"{tech_key}_pn{nfin_p}", inv_nfin_p=nfin_p)
        cfgs.append(NNInvSweepConfig(
            tech, tech_key, analysis, default_circuit, "pn_ratio",
            f"pn_nfinp{nfin_p}", {"nfin_p": nfin_p}))

    # VDD sweep — +/- 0.1 V.
    for dv in (-0.1, 0.1):
        vdd = round(base.vdd + dv, 3)
        if vdd <= 0:
            continue
        tech = replace(base, name=f"{tech_key}_vdd{vdd}", vdd=vdd)
        tag = f"vdd_{vdd:.2f}".replace(".", "p")
        cfgs.append(NNInvSweepConfig(
            tech, tech_key, analysis, default_circuit, "vdd", tag,
            {"vdd": vdd}))

    if analysis != "tran":
        return cfgs

    # Cload sweep (1 fF == baseline, skipped).
    for c_fF in (5, 50, 100):
        circuit = InvCircuitParams(cload=c_fF * 1e-15)
        cfgs.append(NNInvSweepConfig(
            base, tech_key, "tran", circuit, "cload", f"cload_{c_fF}fF",
            {"cload_fF": c_fF}))

    # Input-slew sweep (tr=tf; 50 ps == baseline, skipped).
    for s_ps in (10, 500):
        s = s_ps * 1e-12
        circuit = InvCircuitParams(
            tr=s, tf=s, tstop=_adaptive_tstop(s, s, InvCircuitParams().pw,
                                              INV_TRAN_TD))
        cfgs.append(NNInvSweepConfig(
            base, tech_key, "tran", circuit, "slew", f"slew_{s_ps}ps",
            {"slew_ps": s_ps}))

    # Pulse-width sweep (1 ns == baseline).
    for pw_ns in (0.2, 0.5, 2.0):
        pw = pw_ns * 1e-9
        circuit = InvCircuitParams(
            pw=pw, tstop=_adaptive_tstop(INV_TRAN_TR, INV_TRAN_TF, pw,
                                         INV_TRAN_TD))
        tag = f"pw_{pw_ns:.1f}ns".replace(".", "p")
        cfgs.append(NNInvSweepConfig(
            base, tech_key, "tran", circuit, "pw", tag, {"pw_ns": pw_ns}))

    return cfgs


# ---------------------------------------------------------------------------
# Single-test orchestrators
# ---------------------------------------------------------------------------
def _fail(cfg, error: str) -> Dict[str, object]:
    return {
        "config": cfg, "passed": False, "error": error,
        "nrmse": float("nan"), "mre": float("nan"),
        "r2": float("nan"), "max_err": float("nan"),
    }


def run_single_nn_dc(
    cfg: NNDCSweepConfig, work_dir: Path, checkpoints: Dict[str, object],
) -> Dict[str, object]:
    """Run one single-device Id-Vgs config: NGSPICE truth vs DirectNet."""
    work_dir.mkdir(parents=True, exist_ok=True)
    tech = cfg.tech
    try:
        if cfg.device == "nmos":
            ref = run_ngspice_nmos_dc(tech, work_dir)
            test = run_pycircuitsim_nn_nmos_dc(
                tech, work_dir, NN_LEVEL, NN_MODEL_NAME, model_path=None)
            rx, tx = ref["sweep"], test["sweep"]
        else:
            ref = run_ngspice_pmos_dc(tech, work_dir)
            test = run_pycircuitsim_nn_pmos_dc(
                tech, work_dir, NN_LEVEL, NN_MODEL_NAME, model_path=None)
            # PMOS NGSPICE sweep is negative-going; align on |Vgs|.
            rx, tx = np.abs(ref["sweep"]), np.abs(test["sweep"])
        r_order = np.argsort(rx)
        t_order = np.argsort(tx)
        m = curve_metrics(
            rx[r_order], ref["id"][r_order], tx[t_order], test["id"][t_order])
    except Exception as exc:  # noqa: BLE001 — report, never crash the sweep
        return _fail(cfg, f"{type(exc).__name__}: {exc}")

    return {
        "config": cfg, "passed": m["nrmse"] < DC_NRMSE_PASS, "error": "",
        "nrmse": m["nrmse"], "mre": m["mre"], "r2": m["r2"],
        "max_err": m["max_err"],   # Amperes
    }


def run_single_nn_inv(
    cfg: NNInvSweepConfig, work_dir: Path, checkpoints: Dict[str, object],
) -> Dict[str, object]:
    """Run one inverter config: NGSPICE truth vs DirectNet (VTC or transient)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    tech = cfg.tech
    dn_n = checkpoints.get("directnet_v4_nmos")
    dn_p = checkpoints.get("directnet_v4_pmos")
    try:
        if cfg.analysis == "vtc":
            ref = run_ngspice_inverter_vtc(tech, work_dir)
            test = run_pycircuitsim_nn_inverter_vtc(
                tech, work_dir, NN_LEVEL, NN_MODEL_NAME, dn_n, dn_p)
            m = curve_metrics(
                ref["sweep"], ref["vout"], test["sweep"], test["vout"])
        else:
            circuit = cfg.circuit or InvCircuitParams()
            ref = run_ngspice_inverter_tran(tech, work_dir, circuit=circuit)
            test = run_pycircuitsim_nn_inverter_tran(
                tech, work_dir, NN_LEVEL, NN_MODEL_NAME, dn_n, dn_p,
                circuit=circuit)
            if len(test["time"]) < 3:
                raise RuntimeError("NN transient truncated — NR diverged")
            m = curve_metrics(
                ref["time"], ref["v(out)"], test["time"], test["v(out)"],
                x_min=TRAN_STARTUP_EXCL)
    except Exception as exc:  # noqa: BLE001
        return _fail(cfg, f"{type(exc).__name__}: {exc}")

    return {
        "config": cfg, "passed": m["nrmse"] < INV_NRMSE_PASS, "error": "",
        "nrmse": m["nrmse"], "mre": m["mre"], "r2": m["r2"],
        "max_err": m["max_err"] * 1e3,   # mV
    }


# ---------------------------------------------------------------------------
# Baseline-gated multi-tech driver loop
# ---------------------------------------------------------------------------
def run_nn_multi_tech(
    tech_keys: List[str],
    dimension: str,
    results_dir: Path,
    make_baseline_fn: Callable,
    build_param_fn: Callable,
    run_single_fn: Callable,
) -> List[Dict[str, object]]:
    """Run baseline per tech, then the parametric sweep only for techs that
    pass baseline (mirrors ``base.run_multi_tech_main``)."""
    checkpoints = get_available_checkpoints()
    results: List[Dict[str, object]] = []
    for tk in tech_keys:
        print(f"\n{'=' * 70}\n  {tk} — {dimension}\n{'=' * 70}")
        base_cfg = make_baseline_fn(tk, dimension)
        wd = results_dir / tk / f"{dimension}_{base_cfg.config_name}"
        res = run_single_fn(base_cfg, wd, checkpoints)
        results.append(res)
        _print_result_line(res)
        if not res["passed"]:
            print(f"  baseline FAILED — skipping {tk} {dimension} sweep")
            continue
        for cfg in build_param_fn(tk, dimension):
            wd = results_dir / tk / f"{dimension}_{cfg.config_name}"
            res = run_single_fn(cfg, wd, checkpoints)
            results.append(res)
            _print_result_line(res)
    return results


def _print_result_line(res: Dict[str, object]) -> None:
    cfg = res["config"]
    if res.get("error"):
        print(f"  {cfg.label:<34s}  ERROR: {res['error']}")
        return
    print(f"  {cfg.label:<34s}  NRMSE={res['nrmse']:6.2f}%  "
          f"MRE={res['mre']:6.2f}%  R2={res['r2']:8.5f}  "
          f"MaxErr={res['max_err']:9.3e}  "
          f"{'PASS' if res['passed'] else 'FAIL'}")


# ---------------------------------------------------------------------------
# Summary table / CSV / bar plot
# ---------------------------------------------------------------------------
def print_nn_summary_table(
    results: List[Dict[str, object]], kind: str,
) -> Dict[str, int]:
    """Print a per-config summary. ``kind`` is 'dc' or 'inv' (sets MaxErr unit)."""
    unit = "uA" if kind == "dc" else "mV"
    err_scale = 1e6 if kind == "dc" else 1.0   # DC max_err is in Amperes
    print(f"\n{'=' * 94}\n  SUMMARY TABLE ({kind})\n{'=' * 94}")
    print(f"  {'Config':<34s} | {'Sweep':<9s} | {'NRMSE%':>7s} | "
          f"{'MRE%':>7s} | {'R2':>8s} | {'MaxErr':>9s}{unit} | {'Status':>7s}")
    print(f"  {'-' * 90}")
    n_pass = n_fail = n_err = 0
    for res in results:
        cfg = res["config"]
        if res.get("error"):
            n_err += 1
            print(f"  {cfg.label:<34s} | {cfg.sweep_type:<9s} | "
                  f"{'—':>7s} | {'—':>7s} | {'—':>8s} | {'—':>11s} | "
                  f"{'ERROR':>7s}")
            continue
        if res["passed"]:
            n_pass += 1
        else:
            n_fail += 1
        print(f"  {cfg.label:<34s} | {cfg.sweep_type:<9s} | "
              f"{res['nrmse']:7.2f} | {res['mre']:7.2f} | "
              f"{res['r2']:8.5f} | {res['max_err'] * err_scale:9.3f}{unit} | "
              f"{'PASS' if res['passed'] else 'FAIL':>7s}")
    print(f"  {'-' * 90}")
    print(f"  Total: {len(results)}  Pass: {n_pass}  Fail: {n_fail}  "
          f"Error: {n_err}")
    return {"pass": n_pass, "fail": n_fail, "error": n_err}


def save_nn_summary_csv(
    results: List[Dict[str, object]], csv_path: Path, kind: str,
) -> None:
    """Write a flat per-config CSV. Swept values land in dedicated columns."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    swept_keys: List[str] = []
    for res in results:
        for k in res["config"].swept:
            if k not in swept_keys:
                swept_keys.append(k)
    base_cols = ["label", "tech", "sweep_type", "config_name",
                 "nrmse_pct", "mre_pct", "r2", "max_err", "status"]
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(base_cols + swept_keys)
        for res in results:
            cfg = res["config"]
            status = ("ERROR" if res.get("error")
                      else ("PASS" if res["passed"] else "FAIL"))
            row = [
                cfg.label, cfg.tech_key, cfg.sweep_type, cfg.config_name,
                f"{res['nrmse']:.4f}", f"{res['mre']:.4f}",
                f"{res['r2']:.6f}", f"{res['max_err']:.6e}", status,
            ]
            row += [cfg.swept.get(k, "") for k in swept_keys]
            writer.writerow(row)
    print(f"  [CSV] Summary saved: {csv_path}")


def plot_nn_summary_bar(
    results: List[Dict[str, object]], save_path: Path, title: str, kind: str,
) -> None:
    """NRMSE-per-config bar chart, coloured by tech, with the pass threshold."""
    ok = [r for r in results if not r.get("error")]
    if not ok:
        print("  [Plot] skipped — no non-error results")
        return
    labels = [r["config"].label for r in ok]
    vals = [r["nrmse"] for r in ok]
    colors = [TECH_COLORS.get(r["config"].tech_key, "gray") for r in ok]
    threshold = DC_NRMSE_PASS if kind == "dc" else INV_NRMSE_PASS

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.4), 6))
    ax.bar(range(len(labels)), vals, color=colors)
    ax.axhline(threshold, color="r", ls="--", lw=1,
               label=f"pass threshold {threshold:.0f}%")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("NRMSE (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Summary bar saved: {save_path}")
