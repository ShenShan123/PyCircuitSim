"""Is the GPU 'tensor' bucket GEMM, or per-device input marshalling?

batch_eval builds its input with:
    torch.tensor(list_of_N_tuples, device=dev)     # H2D from Python list
    torch.stack([m._geo_norm_t for m in devs])     # N separate device tensors
    torch.cat([m._tech_code_tensor for m in devs]) # N separate device tensors
"""
import time, torch, sys

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda")
if dev.type == "cpu":
    torch.set_num_threads(1)

for N in (384, 1536, 6144):
    geo = [torch.randn(3, device=dev) for _ in range(N)]
    tc = [torch.zeros(1, dtype=torch.long, device=dev) for _ in range(N)]
    rows = [(0.1, 0.2, 0.0, 0.0)] * N

    def marshal():
        v = torch.tensor(rows, dtype=torch.float32, device=dev)
        g = torch.stack(geo, dim=0)
        t = torch.cat(tc, dim=0)
        return v, g, t

    # pre-stacked equivalent (what a batched rewrite would do)
    geo_pre = torch.stack(geo, dim=0)
    tc_pre = torch.cat(tc, dim=0)
    rows_np = torch.tensor(rows, dtype=torch.float32)

    def marshal_batched():
        v = rows_np.to(dev, non_blocking=True)
        return v, geo_pre, tc_pre

    for fn, name in ((marshal, "per-device"), (marshal_batched, "batched")):
        for _ in range(3):
            fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 20
        print(f"{dev.type:>5} N={N:>6} {name:>11}: {dt*1e3:8.3f} ms/call", flush=True)
