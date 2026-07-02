#!/usr/bin/env python3
"""DirectNet V6.4.7 — S12 (P5): harvest trajectory-corridor bias points.

Run the 4 complex benchmark circuits with PyCircuitSim's NATIVE LEVEL=72
(BSIM-CMG via PyCMG/OSDI) device path — which S6 proved reproduces NGSPICE at
ratio 1.000 (46.64/46.65 ps) — collect the per-device (Vd,Vg,Vs,Vb)
trajectories the transistors actually visit, source-shift them into the NN's
Vs=0 frame, dedup in bias space, evaluate OSDI ground truth at each unique
bias, and save a per-(tech,device) ``traj_corridor`` fragment.

The fragments are appended to the v2 datasets by
``v6_4_7_s12_append_corridors.py`` as sample_class code 12 (traj_corridor),
the same move as ``inv_trip`` — the project's single most successful data
lever (TSMC5 16.90 % -> 0.92 %).

WHY native L72, not NGSPICE node parse: S6 control proved the native L72 path
is the ground-truth trajectory AND it reuses PyCircuitSim's parser
device->terminal mapping + per-timestep converged node solution (no brittle
NGSPICE-node reconstruction). Template: scripts/v6_4_7_s6_l72_ro_control.py.

Geometry: harvest+eval at the TRUE benchmark geometry — NMOS L=16n, PMOS
L=20n, NFIN=2, T=300.15 K (room-temp bin, matching the .temp 27 NGSPICE
decks). NMOS L=16nm is OFF the PDK geometry grid {6,20,36,...}nm (the NN
interpolates to 16nm at inference), so corridor rows cannot be fingerprinted
by the tech-variant labeller — they are labeled at append time via a
pre-seeded label cache (see the append script).

Source-shift (CRITICAL): the dataset + post-P0 inference use the Vs=0 frame,
so each harvested bias is stored as [Vd-Vs, Vg-Vs, 0, Vb-Vs]. OSDI is
difference-only, so eval at the shifted frame == eval at the absolute frame
(validated here). Lifted-source devices (opamp tail pair, SC pass, SRAM
access) are the whole point — they were the P0 blind spot.

Usage:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
      conda run -n pycircuitsim python scripts/v6_4_7_s12_harvest_corridors.py \
      --tech tsmc5,tsmc7,tsmc12,tsmc16 \
      2>&1 | tee results/v6_4_7/s12_harvest_logs/harvest.log
Smoke one tech:
    ... --tech tsmc5
"""
from __future__ import annotations

import argparse
import functools
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

print = functools.partial(print, flush=True)  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
# Force ROOT to the FRONT (ahead of PyCMG/tests) even if it is already present
# elsewhere on sys.path — otherwise PyCMG's own ``tests`` package shadows the
# repo's ``tests.common`` (remove-then-insert; the plain ``if not in`` guard
# leaves a pre-existing ROOT entry stuck behind PyCMG/tests).
for p in (ROOT / "external_compact_models" / "PyCMG" / "tests",
          ROOT / "external_compact_models" / "PyCMG",
          ROOT / "external_compact_models",
          ROOT):
    sp = str(p)
    if sp in sys.path:
        sys.path.remove(sp)
    sys.path.insert(0, sp)

import tests.common.complex as cx  # noqa: E402
from tests.common.complex import (  # noqa: E402
    BENCH, BenchTech, run_directnet_transient, get_baked_modelcard,
)
from tests.common.base import OSDI_PATH, run_ngspice_subprocess  # noqa: E402
from scripts.v6_4_7_s6_l72_ro_control import build_merged_card, make_l72_parse  # noqa: E402
from pycircuitsim.models.mosfet_cmg import NMOS_CMG, PMOS_CMG  # noqa: E402

# PyCMG OSDI eval path (same calls generate_one_bin uses)
from pycmg.nn_generate import (  # noqa: E402
    _create_model_and_instance, eval_single_point,
)
from pycmg.nn_config import TECH_CONFIGS  # noqa: E402
from pycmg.sweep import NN_OUTPUT_COLUMNS  # noqa: E402

OUT_DIR = ROOT / "results" / "v6_4_7" / "s12_corridors"
LOG_DIR = ROOT / "results" / "v6_4_7" / "s12_harvest_logs"

# Bench variant per tech (the checkpoint VT — matches BENCH + S12 build plan).
BENCH_VARIANT = {"tsmc5": "lvt", "tsmc7": "ulvt", "tsmc12": "svt", "tsmc16": "svt"}
ROOM_T_K = 300.15
NFIN = 2
L_NMOS = 16e-9
L_PMOS = 20e-9

BIAS_GRID_V = 0.002      # dedup resolution (2 mV) in (vd,vg,vbs) space
PRE_EVAL_CAP = 200_000   # safety cap on unique biases before OSDI eval

# Tube densification: the benchmark circuits visit a concentrated bias
# manifold (~0.07 % of the dataset as bare trajectory points), far too small
# to carry loss mass at a sane class weight. Around each unique trajectory
# bias we add JITTER_N OSDI-evaluated samples in a +/-JITTER_V box — a thin
# tube the NN must match in a neighborhood of every operating point (exactly
# what NR convergence needs), not just exactly on the trajectory. This is the
# same idea as inv_trip's Vth-centered BAND, applied to the visited corridor.
JITTER_N = 20
JITTER_V = 0.012         # +/-12 mV tube around each (vd,vg,vbs)
JITTER_SEED = 20260614

# Harvest transient windows: SHORTER than the benchmark windows (RO 5n / SC
# 12n) because both circuits are periodic — a few cycles cover every visited
# bias. The native L72 path is ~215 s/ns (S6), so the full windows are
# prohibitive; coarser tstep + fewer cycles sample the SAME trajectory curve
# (the 2 mV dedup grid is far coarser than any per-step spacing). RO period
# ~50 ps -> 1.2 ns = ~24 cycles; SC clock 4 ns -> 5 ns = >1 full sample+hold.
RO_TRAN = ".tran 4p 0.6n"     # ~12 RO periods (period ~50 ps) — full bias cover
SC_TRAN = ".tran 20p 4.5n"    # > 1 clock period (4 ns): sample + hold covered


# ---------------------------------------------------------------------------
# L72 netlist renderers (topology/analysis mirror the verify_complex_* decks)
# ---------------------------------------------------------------------------
def _models_block(bt: BenchTech) -> str:
    return (f".model {bt.nmos_model} NMOS (LEVEL=72)\n"
            f".model {bt.pmos_model} PMOS (LEVEL=72)")


def render_ring_osc(bt: BenchTech, path: Path) -> Path:
    """5-stage RO — verify_complex_ring_osc / S6 control, LEVEL=72."""
    nd = ["n1", "n2", "n3", "n4", "n5"]
    ln, lp, tf = bt.l_nmos * 1e9, bt.l_pmos * 1e9, bt.tfin * 1e9
    lines = [f"* RO L72 corridor harvest ({bt.name})",
             f"Vdd vdd 0 {bt.vdd}",
             f".ic V(n1)=0.0 V(n2)={bt.vdd} V(n3)=0.0 V(n4)={bt.vdd} V(n5)=0.0",
             ""]
    for i in range(5):
        lines += [
            f"Mp{i+1} {nd[i]} {nd[i-1]} vdd vdd {bt.pmos_model} "
            f"L={lp:.0f}n NFIN={bt.nfin} TFIN={tf:.1f}n",
            f"Mn{i+1} {nd[i]} {nd[i-1]} 0   0   {bt.nmos_model} "
            f"L={ln:.0f}n NFIN={bt.nfin} TFIN={tf:.1f}n",
            f"Cl{i+1} {nd[i]} 0 0.5f", ""]
    lines += [_models_block(bt), "", RO_TRAN, "", ".end"]
    path.write_text("\n".join(lines))
    return path


def render_opamp(bt: BenchTech, path: Path) -> Path:
    """Two-stage Miller opamp DC sweep — verify_complex_opamp, LEVEL=72."""
    ln, lp, tf = bt.l_nmos * 1e9, bt.l_pmos * 1e9, bt.tfin * 1e9
    vcm = round(bt.vdd * 0.55, 3)
    vbn = round(bt.vdd * 0.45, 3)
    vbp = round(bt.vdd * 0.55, 3)
    lo, hi = round(vcm - 0.15, 3), round(vcm + 0.15, 3)
    n, p = bt.nmos_model, bt.pmos_model
    lines = [
        f"* Miller opamp L72 corridor harvest ({bt.name})",
        f"Vdd vdd 0 {bt.vdd}", f"Vbn vbn 0 {vbn}", f"Vbp vbp 0 {vbp}",
        f"Vinn inn 0 {vcm}", f"Vinp inp 0 {vcm}",
        f"Mn1 n1   inp vtail 0   {n} L={ln:.0f}n NFIN={bt.nfin} TFIN={tf:.1f}n",
        f"Mn2 vo1i inn vtail 0   {n} L={ln:.0f}n NFIN={bt.nfin} TFIN={tf:.1f}n",
        f"Mp3 n1   n1  vdd   vdd {p} L={lp:.0f}n NFIN={bt.nfin} TFIN={tf:.1f}n",
        f"Mp4 vo1i n1  vdd   vdd {p} L={lp:.0f}n NFIN={bt.nfin} TFIN={tf:.1f}n",
        f"Mn5 vtail vbn 0    0   {n} L={ln:.0f}n NFIN={bt.nfin} TFIN={tf:.1f}n",
        f"Mp6 vout vo1i vdd vdd {p} L={lp:.0f}n NFIN={bt.nfin} TFIN={tf:.1f}n",
        f"Mn7 vout vbn  0   0   {n} L={ln:.0f}n NFIN={bt.nfin} TFIN={tf:.1f}n",
        "Cc vo1i vout 20f", "CL vout 0 50f", "",
        _models_block(bt), "", f".dc Vinp {lo} {hi} 0.002", "", ".end"]
    path.write_text("\n".join(lines))
    return path


def render_switchcap(bt: BenchTech, path: Path) -> Path:
    """Switched-cap unit cell transient — verify_complex_switchcap, LEVEL=72."""
    ln, lp, tf = bt.l_nmos * 1e9, bt.l_pmos * 1e9, bt.tfin * 1e9
    vin = round(bt.vdd * 0.6, 3)
    n, p = bt.nmos_model, bt.pmos_model
    lines = [
        f"* switchcap L72 corridor harvest ({bt.name})",
        f"Vdd vdd 0 {bt.vdd}", f"Vin vin 0 {vin}",
        f"Vphi phi 0 PULSE 0 {bt.vdd} 0.5n 0.1n 0.1n 1.9n 4n",
        f"Mpc phib phi vdd vdd {p} L={lp:.0f}n NFIN={bt.nfin} TFIN={tf:.1f}n",
        f"Mnc phib phi 0   0   {n} L={ln:.0f}n NFIN={bt.nfin} TFIN={tf:.1f}n",
        f"Mnt vin phi  vsamp 0   {n} L={ln:.0f}n NFIN={bt.nfin} TFIN={tf:.1f}n",
        f"Mpt vin phib vsamp vdd {p} L={lp:.0f}n NFIN={bt.nfin} TFIN={tf:.1f}n",
        "Csample vsamp 0 100f", f".ic V(vsamp)=0.0 V(phib)={bt.vdd}", "",
        _models_block(bt), "", SC_TRAN, "", ".end"]
    path.write_text("\n".join(lines))
    return path


def render_sram_halfcell(bt: BenchTech, path: Path) -> Path:
    """SRAM read-SNM butterfly half-cell DC sweep — verify_complex_sram, L72."""
    ln, lp, tf = bt.l_nmos * 1e9, bt.l_pmos * 1e9, bt.tfin * 1e9
    n, p = bt.nmos_model, bt.pmos_model
    lines = [
        f"* SRAM half-cell L72 corridor harvest ({bt.name})",
        f"Vdd vdd 0 {bt.vdd}", f"Vwl wl 0 {bt.vdd}", f"Vbl bl 0 {bt.vdd}",
        "Vq q 0 0.0",
        f"Mpl qb q vdd vdd {p} L={lp:.0f}n NFIN={bt.nfin} TFIN={tf:.1f}n",
        f"Mnl qb q 0   0   {n} L={ln:.0f}n NFIN={bt.nfin} TFIN={tf:.1f}n",
        f"Mna bl wl qb 0   {n} L={ln:.0f}n NFIN={bt.nfin} TFIN={tf:.1f}n", "",
        _models_block(bt), "", f".dc Vq 0 {bt.vdd} 0.005", "", ".end"]
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Device->terminal mapping + trajectory extraction
# ---------------------------------------------------------------------------
def _device_map(netlist: Path, l72_parse) -> List[Tuple[str, bool, List[str]]]:
    """Parse the netlist; return [(name, is_pmos, [d,g,s,b]), ...] MOSFETs."""
    parser = l72_parse(netlist)
    devs = []
    for comp in parser.circuit.components:
        if isinstance(comp, (NMOS_CMG, PMOS_CMG)):
            devs.append((comp.name, isinstance(comp, PMOS_CMG), list(comp.nodes)))
    return devs


def _n_points(results: Dict) -> int:
    return max((len(np.atleast_1d(v)) for v in results.values()
               if hasattr(v, "__len__")), default=0)


def _trace(results: Dict, node: str, n: int) -> np.ndarray:
    if node in ("0", "gnd", "GND"):
        return np.zeros(n)
    arr = results.get(node)
    if arr is None:
        raise KeyError(f"node {node!r} not in results "
                       f"(have {sorted(results)[:12]}...)")
    a = np.asarray(arr, dtype=float)
    if a.shape[0] != n:
        raise ValueError(f"node {node!r} len {a.shape[0]} != {n}")
    return a


def _harvest_traj(results: Dict, devs, nmos_acc: Dict, pmos_acc: Dict) -> int:
    """Accumulate source-shifted, grid-dedup'd biases with residence counts.

    nmos_acc / pmos_acc: dict {(vd,vg,vbs)_rounded: residence_count}.
    Returns the number of raw device-time points processed.
    """
    n = _n_points(results)
    if n == 0:
        return 0
    raw = 0
    for name, is_pmos, nodes in devs:
        d, g, s, b = nodes
        vd = _trace(results, d, n)
        vg = _trace(results, g, n)
        vs = _trace(results, s, n)
        vb = _trace(results, b, n)
        # source-relative frame (Vs == 0): [vd-vs, vg-vs, 0, vb-vs]
        sd = vd - vs
        sg = vg - vs
        sbs = vb - vs
        acc = pmos_acc if is_pmos else nmos_acc
        for k in range(n):
            key = (round(float(sd[k]) / BIAS_GRID_V),
                   round(float(sg[k]) / BIAS_GRID_V),
                   round(float(sbs[k]) / BIAS_GRID_V))
            acc[key] = acc.get(key, 0) + 1
            raw += 1
    return raw


def run_ngspice_nodes(bt: BenchTech, body_lines: List[str], analysis: str,
                      dump_nodes: List[str], work_dir: Path, tag: str
                      ) -> Dict[str, np.ndarray]:
    """Run an NGSPICE BSIM-CMG (L72 ground-truth) DC sweep; return {node: arr}.

    The opamp + SRAM-butterfly DC sweeps diverge under PyCircuitSim's NR for
    the *raw* L72 device (no NN gds-floor / smooth clamp), so their
    ground-truth trajectories are harvested from NGSPICE — the SAME teacher the
    verify_complex_* gates use (S6: native L72 == NGSPICE at ratio 1.000, so
    this is ground-truth-equivalent to the RO/SC L72 transients). Columns are
    mapped by the wr_vecnames header so the (x,y)-per-vector layout cannot
    confound. Returns ``{"__axis__": sweep, node: values, ...}``.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    cir = work_dir / f"ng_{tag}.cir"
    runner = work_dir / f"ng_{tag}_runner.cir"
    csv = work_dir / f"ng_{tag}.csv"
    log = work_dir / f"ng_{tag}.log"
    sig = " ".join(f"v({n})" for n in dump_nodes)
    cir.write_text(f"* L72 ground-truth node dump ({tag})\n"
                   + "\n".join(body_lines) + "\n.end\n")
    runner.write_text(
        f"* runner ({tag})\n.control\nosdi {OSDI_PATH}\nsource {cir}\n"
        f"set filetype=ascii\nset wr_vecnames\n{analysis}\n"
        f"wrdata {csv} {sig}\n.endc\n.end\n")
    lines = run_ngspice_subprocess(runner, log, csv)
    header = lines[0].split()
    rows = [[float(x) for x in ln.split()] for ln in lines[1:] if ln.strip()]
    data = np.array(rows)
    if data.size == 0 or not np.all(np.isfinite(data)):
        raise RuntimeError(f"NGSPICE produced no/NaN data ({tag})")
    low = [h.lower() for h in header]
    out: Dict[str, np.ndarray] = {"__axis__": data[:, 0]}
    for n in dump_nodes:
        key = f"v({n})".lower()
        if key not in low:
            raise RuntimeError(f"node {n!r} not in ngspice header {header}")
        out[n] = data[:, low.index(key)]
    return out


# ---------------------------------------------------------------------------
# Per-tech harvest
# ---------------------------------------------------------------------------
def harvest_tech(tech: str, work_dir: Path, circuits=None) -> Dict[str, Dict]:
    bt = BENCH[tech.upper()]
    variant = BENCH_VARIANT[tech]
    merged, _, _ = build_merged_card(bt, work_dir)
    l72_parse = make_l72_parse(merged, bt)

    nmos_acc: Dict[Tuple[int, int, int], int] = {}
    pmos_acc: Dict[Tuple[int, int, int], int] = {}
    raw_total = 0
    # V6.6.2: circuit selection — a ring-only ("ring_osc,switchcap") corridor
    # opens the value-owned rings WITHOUT tightly supervising the opamp
    # trajectory (whose value-surface tightening collapses the delicate high-gain
    # OP — the S10 derivative-fidelity⟂opamp-gain tension). None = all 4 circuits.
    if circuits is None:
        circuits = {"ring_osc", "switchcap", "opamp", "sram_half"}

    # --- transient circuits (RO, switchcap) ---
    for tag, render in (("ring_osc", render_ring_osc),
                        ("switchcap", render_switchcap)):
        if tag not in circuits:
            continue
        netlist = render(bt, work_dir / f"{tag}_{tech}_l72.sp")
        devs = _device_map(netlist, l72_parse)
        orig = cx.parse_netlist
        cx.parse_netlist = l72_parse
        try:
            results, partial, err = run_directnet_transient(netlist)
        finally:
            cx.parse_netlist = orig
        r = _harvest_traj(results, devs, nmos_acc, pmos_acc)
        raw_total += r
        print(f"    [{tech}/{tag}] devs={len(devs)} pts={_n_points(results)} "
              f"raw={r} partial={partial} err={err!r}")

    # --- DC-sweep circuits (opamp, SRAM butterfly) via NGSPICE L72 ground
    #     truth (raw L72 DC diverges under PyCircuitSim NR for these). Device
    #     map comes from parsing the matching PyCircuitSim L72 netlist; the
    #     node names are identical, so trajectories map 1:1. ---
    n, p = bt.nmos_model, bt.pmos_model
    baked = get_baked_modelcard(bt, bt.nfin, work_dir)
    vcm = round(bt.vdd * 0.55, 3)
    vbn = round(bt.vdd * 0.45, 3)
    vbp = round(bt.vdd * 0.55, 3)
    lo, hi = round(vcm - 0.15, 3), round(vcm + 0.15, 3)
    dc_specs = [
        ("opamp", render_opamp,
         [f'.include "{baked}"', ".temp 27", f"Vdd vdd 0 {bt.vdd}",
          f"Vbn vbn 0 {vbn}", f"Vbp vbp 0 {vbp}", f"Vinn inn 0 {vcm}",
          f"Vinp inp 0 {vcm}",
          f"Nn1 n1 inp vtail 0 {n}", f"Nn2 vo1i inn vtail 0 {n}",
          f"Np3 n1 n1 vdd vdd {p}", f"Np4 vo1i n1 vdd vdd {p}",
          f"Nn5 vtail vbn 0 0 {n}",
          f"Np6 vout vo1i vdd vdd {p}", f"Nn7 vout vbn 0 0 {n}",
          "Cc vo1i vout 20f", "CL vout 0 50f"],
         f"dc Vinp {lo} {hi} 0.002", ["n1", "vtail", "vo1i", "vout"],
         "inp", {"vdd": bt.vdd, "vbn": vbn, "vbp": vbp, "inn": vcm}),
        ("sram_half", render_sram_halfcell,
         [f'.include "{baked}"', ".temp 27", f"Vdd vdd 0 {bt.vdd}",
          f"Vwl wl 0 {bt.vdd}", f"Vbl bl 0 {bt.vdd}", "Vq q 0 0.0",
          f"Npl qb q vdd vdd {p}", f"Nnl qb q 0 0 {n}",
          f"Nna bl wl qb 0 {n}"],
         f"dc Vq 0 {bt.vdd} 0.005", ["qb"],
         "q", {"vdd": bt.vdd, "wl": bt.vdd, "bl": bt.vdd}),
    ]
    for tag, render, body, analysis, dump_nodes, swept, const in dc_specs:
        if tag not in circuits:
            continue
        devs = _device_map(render(bt, work_dir / f"{tag}_{tech}_l72.sp"),
                           l72_parse)
        ng = run_ngspice_nodes(bt, body, analysis, dump_nodes, work_dir,
                               f"{tag}_{tech}")
        N = len(ng["__axis__"])
        results = {swept: ng["__axis__"]}
        for nd in dump_nodes:
            results[nd] = ng[nd]
        for node, val in const.items():
            results[node] = np.full(N, float(val))
        r = _harvest_traj(results, devs, nmos_acc, pmos_acc)
        raw_total += r
        print(f"    [{tech}/{tag}] devs={len(devs)} pts={N} raw={r} (NGSPICE)")

    print(f"  [{tech}] raw points={raw_total}  unique NMOS={len(nmos_acc)}  "
          f"unique PMOS={len(pmos_acc)}")
    return {"nmos": nmos_acc, "pmos": pmos_acc, "bt": bt, "variant": variant}


def eval_and_save(tech: str, dev: str, acc: Dict, bt: BenchTech,
                  variant: str, frag_tag: str = "") -> Dict:
    """OSDI-eval each unique bias at the bench geometry; save a fragment npz."""
    is_pmos = dev == "pmos"
    L = L_PMOS if is_pmos else L_NMOS
    cfg = TECH_CONFIGS[tech]
    built = _create_model_and_instance(cfg, dev, variant, L, float(NFIN),
                                       ROOM_T_K)
    if built is None:
        raise RuntimeError(f"OSDI instance build failed {tech}/{dev}/{variant}")
    _model, inst, proc = built
    geo = np.array([float(NFIN), L, ROOM_T_K] + proc.as_array(),
                   dtype=np.float64)

    # pre-eval cap by residence (keep the most-visited biases)
    items = sorted(acc.items(), key=lambda kv: -kv[1])
    if len(items) > PRE_EVAL_CAP:
        print(f"    [{tech}/{dev}] {len(items)} unique > cap {PRE_EVAL_CAP}; "
              f"keeping top-residence")
        items = items[:PRE_EVAL_CAP]

    inputs, outputs, residence = [], [], []
    n_fail = 0
    rng = np.random.default_rng(JITTER_SEED + (1 if is_pmos else 0))
    for (kd, kg, kbs), res in items:
        cvd = kd * BIAS_GRID_V
        cvg = kg * BIAS_GRID_V
        cvbs = kbs * BIAS_GRID_V
        # center (exactly on the trajectory) + a +/-JITTER_V tube
        samples = [(cvd, cvg, cvbs)]
        if JITTER_N > 0:
            j = rng.uniform(-JITTER_V, JITTER_V, size=(JITTER_N, 3))
            for dvd, dvg, dvbs in j:
                samples.append((cvd + dvd, cvg + dvg, cvbs + dvbs))
        for vd, vg, vbs in samples:
            out = eval_single_point(inst, vd=vd, vg=vg, vs=0.0, vb=vbs)
            if out is None:
                n_fail += 1
                continue
            inputs.append([vd, vg, 0.0, vbs])
            outputs.append([out[k] for k in NN_OUTPUT_COLUMNS])
            residence.append(res)

    inputs = np.asarray(inputs, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64)
    residence = np.asarray(residence, dtype=np.int64)
    geometry = np.tile(geo, (len(inputs), 1))

    # |id| is column 0 of NN_OUTPUT_COLUMNS? -> find index robustly
    id_idx = list(NN_OUTPUT_COLUMNS).index("id")
    idmag = np.abs(outputs[:, id_idx]) if len(outputs) else np.array([])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frag = OUT_DIR / f"{tech}_{dev}_corridor{frag_tag}.npz"
    np.savez(frag, inputs=inputs, geometry=geometry, outputs=outputs,
             residence=residence, idmag=idmag,
             meta_tech=tech, meta_device=dev, meta_variant=variant,
             meta_L=L, meta_NFIN=NFIN, meta_T=ROOM_T_K,
             meta_output_columns=np.array(list(NN_OUTPUT_COLUMNS)))
    # id decade coverage
    nz = idmag[idmag > 0]
    decs = (np.floor(np.log10(nz)).astype(int) if len(nz) else np.array([]))
    dec_hist = {int(d): int((decs == d).sum()) for d in np.unique(decs)} if len(decs) else {}
    print(f"    [{tech}/{dev}] saved {len(inputs)} rows (fail={n_fail}) -> "
          f"{frag.name}  |id| decades={dec_hist}")
    return {"tech": tech, "dev": dev, "rows": int(len(inputs)),
            "fail": n_fail, "frag": str(frag),
            "id_decades": dec_hist}


def validate_shift_equivalence(tech: str) -> None:
    """OSDI is difference-only: eval at a lifted frame == eval at Vs=0 frame."""
    bt = BENCH[tech.upper()]
    variant = BENCH_VARIANT[tech]
    cfg = TECH_CONFIGS[tech]
    built = _create_model_and_instance(cfg, "nmos", variant, L_NMOS,
                                       float(NFIN), ROOM_T_K)
    _m, inst, _p = built
    # absolute frame: Vd=0.5, Vg=0.4, Vs=0.2, Vb=0.0  ->  shifted Vs=0:
    #   vd'=0.3, vg'=0.2, vbs'=-0.2
    a = inst.eval_dc({"d": 0.5, "g": 0.4, "s": 0.2, "e": 0.0})
    b = inst.eval_dc({"d": 0.3, "g": 0.2, "s": 0.0, "e": -0.2})
    da = abs(a["id"] - b["id"])
    print(f"  [{tech}] Vs-shift equivalence: id_abs={a['id']:.6e} "
          f"id_shift={b['id']:.6e}  |Δ|={da:.2e} "
          f"({'OK' if da < 1e-12 + 1e-6 * abs(a['id']) else 'MISMATCH'})")


def main() -> int:
    ap = argparse.ArgumentParser(description="S12/P5 trajectory-corridor harvest")
    ap.add_argument("--tech", default="tsmc5,tsmc7,tsmc12,tsmc16")
    ap.add_argument("--circuits", default="ring_osc,switchcap,opamp,sram_half",
                    help="Comma list of circuits to harvest (V6.6.2: a "
                         "'ring_osc,switchcap' ring-only corridor opens the "
                         "rings without the opamp-trajectory supervision that "
                         "collapses the high-gain OP).")
    ap.add_argument("--frag-tag", default="",
                    help="Suffix for fragment filenames "
                         "({tech}_{dev}_corridor{tag}.npz) so a ring-only "
                         "harvest does not clobber the full-corridor fragments.")
    args = ap.parse_args()
    techs = [t.strip().lower() for t in args.tech.split(",")]
    circuits = {c.strip() for c in args.circuits.split(",") if c.strip()}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"harvest circuits={sorted(circuits)}  frag_tag={args.frag_tag!r}")

    summary = []
    for tech in techs:
        print(f"\n=== harvest {tech} (variant={BENCH_VARIANT[tech]}) ===")
        t0 = time.time()
        validate_shift_equivalence(tech)
        work_dir = Path(tempfile.mkdtemp(prefix=f"s12_{tech}_"))
        accs = harvest_tech(tech, work_dir, circuits=circuits)
        for dev in ("nmos", "pmos"):
            row = eval_and_save(tech, dev, accs[dev], accs["bt"],
                                accs["variant"], frag_tag=args.frag_tag)
            summary.append(row)
        print(f"  [{tech}] done in {time.time() - t0:.0f}s")

    print("\n=== SUMMARY ===")
    for r in summary:
        print(f"  {r['tech']:6s} {r['dev']:4s}  rows={r['rows']:7d}  "
              f"fail={r['fail']:5d}  id_decades={r['id_decades']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
