"""V6.4.8 S3 — cheap pre-training smoke test for the EKV analytic backbone.

Builds an UNTRAINED DirectNet(ekv_core=True), seeds its core norm/sign buffers
from a real norm.npz, and checks the structural physics the EKV CORE is supposed
to guarantee BEFORE spending any GPU time. Properties (1)-(4) probe the core's
physical current ``core_current`` directly (the additive trunk residual is
untrained noise and is separately bounded + zeroed-at-Vds=0 by the inference
correction); property (5) round-trips a saved core.* checkpoint through the
simulator inference path.

  1. core_current finite over a voltage sweep;
  2. Id_core(Vds -> 0) -> 0   (triode self-limiting — the switchcap fix);
  3. Id_core monotone in the conducting gate drive (NMOS more negative ↑Vgs);
  4. autograd gds = d(id_core)/d(Vds) finite, correct sign, rolls off in sat;
  5. saved core.* checkpoint loads via the simulator (_build_from_state core.*
     detection) and yields finite id/gm/gds with positive stamped gds.

Run: conda run -n pycircuitsim python tests/diag_ekv_core_smoke.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from bsimar.data.normalize import NormStats  # noqa: E402
from bsimar.models.direct_net import DirectNet  # noqa: E402

CKPT = PROJECT_ROOT / "external_compact_models" / "bsimar" / "checkpoints"


def _build(norm_path: Path, device_type: str) -> tuple[DirectNet, NormStats]:
    st = NormStats.load(str(norm_path))
    model = DirectNet(
        input_dim=7, hidden_dim=256, n_layers=6, output_dim=13,
        num_tech_codes=5, tech_embed_dim=32, ekv_core=True,
    )
    model.core.set_norm(
        v_mean=st.input_mean[:4], v_std=st.input_std[:4],
        id_s=float(st.asinh_scale[0]),
        id_mean=float(st.output_mean[0]), id_std=float(st.output_std[0]),
        device_type=device_type,
    )
    model.eval()
    return model, st


def _xnorm(st: NormStats, vgs, vds, vbs, nfin=10.0, L=16e-9, T=300.15):
    nfin_log = float(np.log2(max(nfin, 1.0)))
    raw = np.array([vds, vgs, 0.0, vbs, nfin_log, L, T], dtype=np.float64)
    in_std = st.input_std.copy(); in_std[in_std < 1e-12] = 1.0
    return torch.tensor((raw - st.input_mean) / in_std,
                        dtype=torch.float32).unsqueeze(0)


def _core_id_gds(model, st, vgs, vds, vbs, tc=0):
    """Physical Id_core and physical gds=d(Id_core)/d(Vds) via autograd."""
    xn = _xnorm(st, vgs, vds, vbs).clone().requires_grad_(True)
    emb = model.tech_embedding(torch.tensor([tc]))
    idc = model.core.core_current(xn, emb)            # (1,) Amps
    g = torch.autograd.grad(idc.sum(), xn, retain_graph=True)[0][0, 0]
    # chain rule: d(Id_phys)/d(Vds_phys) = d(Id_phys)/d(x0) / in_std[0]
    in_std0 = float(st.input_std[0]) if st.input_std[0] > 1e-12 else 1.0
    return float(idc.item()), float(g.item()) / in_std0


def main() -> int:
    norm_path = CKPT / "tsmc5_dn_c17_nmos_norm.npz"
    if not norm_path.exists():
        print(f"FAIL: norm file missing {norm_path}")
        return 1
    model, st = _build(norm_path, "nmos")
    vdd = float(st.input_max[0]) / 2.0
    print(f"VDD(est)={vdd:.3f}  id_s={float(st.asinh_scale[0]):.3e}")

    ok = True

    # (1) finite over a sweep
    for vgs in np.linspace(0.0, vdd, 6):
        for vds in np.linspace(0.0, vdd, 6):
            idp, gds = _core_id_gds(model, st, float(vgs), float(vds), 0.0)
            if not (np.isfinite(idp) and np.isfinite(gds)):
                print(f"FAIL finite: vgs={vgs:.2f} vds={vds:.2f} id={idp} gds={gds}")
                ok = False

    # (2) Id_core(Vds->0) -> 0  (triode self-limiting)
    id0, _ = _core_id_gds(model, st, vdd, 0.0, 0.0)
    idsat, _ = _core_id_gds(model, st, vdd, vdd, 0.0)
    print(f"(2) Id_core(Vds=0)   = {id0:.3e} A  (want ~0)")
    print(f"    Id_core(Vds=VDD) = {idsat:.3e} A")
    if abs(id0) > 1e-3 * max(abs(idsat), 1e-12):
        print("FAIL: Id_core(Vds=0) not self-limiting")
        ok = False

    # (3) monotone in gate drive (NMOS conducting id < 0, more -ve ↑Vgs)
    idlo, _ = _core_id_gds(model, st, 0.2 * vdd, vdd, 0.0)
    idhi, _ = _core_id_gds(model, st, vdd, vdd, 0.0)
    print(f"(3) Id_core(0.2VDD)={idlo:.3e}  Id_core(VDD)={idhi:.3e} (NMOS hi more -ve)")
    if not (idhi <= idlo):
        print("FAIL: Id_core not monotone in gate drive")
        ok = False

    # (4) gds: physical output conductance = -d(id_phys)/dVds (NMOS) > 0,
    #     rolling off from triode to saturation.
    _, g_lin = _core_id_gds(model, st, vdd, 0.1 * vdd, 0.0)
    _, g_sat = _core_id_gds(model, st, vdd, vdd, 0.0)
    gds_lin, gds_sat = -g_lin, -g_sat   # NMOS conventional gds = -d(id)/dVds
    print(f"(4) gds(lin)={gds_lin:.3e}  gds(sat)={gds_sat:.3e} (want >0, lin>sat)")
    if not (gds_lin > 0 and gds_sat > 0 and gds_lin > gds_sat):
        print("FAIL: gds not positive/rolling-off")
        ok = False

    # (5) round-trip through the simulator inference path
    with tempfile.TemporaryDirectory() as td:
        stem = Path(td) / "ekv_smoke_nmos"
        torch.save(model.state_dict(), str(stem) + "_best.pt")
        st.save(str(stem) + "_norm.npz")
        from pycircuitsim.models.mosfet_directnet import NMOS_NN
        dev = NMOS_NN(
            name="M1", nodes=["d", "g", "s", "b"],
            model_path=str(stem) + "_best.pt", L=16e-9, NFIN=10, tech_code=0)
        r = dev._eval({"d": vdd, "g": vdd, "s": 0.0, "b": 0.0})
        r0 = dev._eval({"d": 0.0, "g": vdd, "s": 0.0, "b": 0.0})
        print(f"(5) sim path: id={r['id']:.3e} gm={r['gm']:.3e} "
              f"gds={r['gds']:.3e} qg={r['qg']:.3e}; id(Vds=0)={r0['id']:.3e}")
        if not all(np.isfinite(r[k]) for k in ("id", "gm", "gds", "qg")):
            print("FAIL: sim-path eval not finite"); ok = False
        if r["gds"] <= 0:
            print("FAIL: stamped gds not positive"); ok = False
        if abs(r0["id"]) > 1e-9:
            print("FAIL: inference id(Vds=0) not zeroed by correction"); ok = False

    print("\n" + ("PASS — EKV core structural smoke OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
