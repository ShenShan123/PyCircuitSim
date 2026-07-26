"""Micro-benchmark: DirectNet-large eval cost vs batch size, CPU(1thr) vs CUDA.

Mirrors the shipped eval shape in _MOSFETNNBase.batch_eval:
  DC path   = 1 forward + 1 backward (id Jacobian only, V7.0.1)
  TRAN path = 1 forward + 3 backwards (id, qg, qd)
"""
import os
import sys
import time

import torch

sys.path.insert(0, "external_compact_models")

CKPT = ("external_compact_models/bsimar/checkpoints/"
        "tsmc5_dn_large_nmos_best.pt")


def build(state):
    from bsimar.models.direct_net import DirectNet
    net_keys = [k for k in state if k.startswith("net.") and k.endswith(".weight")]
    output_dim = state[net_keys[-1]].shape[0]
    hidden_dim = state[net_keys[-1]].shape[1]
    n_layers = len(net_keys) - 1
    num_tech_codes = state["tech_embedding.weight"].shape[0]
    tech_embed_dim = state["tech_embedding.weight"].shape[1]
    input_dim = state[net_keys[0]].shape[1] - tech_embed_dim
    m = DirectNet(input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers,
                  output_dim=output_dim, num_tech_codes=num_tech_codes,
                  tech_embed_dim=tech_embed_dim)
    m.load_state_dict(state)
    return m.eval()


def timeit(fn, reps, warm=3):
    for _ in range(warm):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def run(dev_str, batches):
    dev = torch.device(dev_str)
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model = build(state).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"\n== {dev_str}  params={nparam:,} ==")
    print(f"{'N':>8} {'DC ms':>10} {'DC us/dev':>11} "
          f"{'TRAN ms':>10} {'TRAN us/dev':>12}")
    out = {}
    for n in batches:
        x = torch.randn(n, 7, device=dev)
        xv = x[:, :4].detach().requires_grad_(True)
        xg = x[:, 4:].detach()
        tc = torch.zeros(n, dtype=torch.long, device=dev)

        def dc():
            xf = torch.cat([xv, xg], dim=1)
            o = model(xf, tech_codes=tc)
            torch.autograd.grad(o[:, 0].sum(), xv, retain_graph=False)

        def tran():
            xf = torch.cat([xv, xg], dim=1)
            o = model(xf, tech_codes=tc)
            torch.autograd.grad(o[:, 0].sum(), xv, retain_graph=True)
            torch.autograd.grad(o[:, 5].sum(), xv, retain_graph=True)
            torch.autograd.grad(o[:, 6].sum(), xv, retain_graph=False)

        reps = max(3, min(20, int(2e5 / max(n, 1))))
        t_dc = timeit(dc, reps)
        t_tr = timeit(tran, reps)
        out[n] = (t_dc, t_tr)
        print(f"{n:>8} {t_dc*1e3:>10.3f} {t_dc/n*1e6:>11.2f} "
              f"{t_tr*1e3:>10.3f} {t_tr/n*1e6:>12.2f}")
    return out


if __name__ == "__main__":
    BATCHES = [1, 96, 384, 1536, 6144]
    mode = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    if mode == "cpu":
        torch.set_num_threads(1)
        run("cpu", BATCHES)
    else:
        run("cuda", BATCHES)
