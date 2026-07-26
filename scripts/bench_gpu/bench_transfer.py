"""CPU<->GPU data movement cost for the NN eval, at SRAM batch sizes.

The NR loop is CPU-resident (MNA assembly, spsolve, convergence tests), so
every NR iteration is a SYNCHRONOUS round trip:
    H2D  voltages (N,4)                       = 16N bytes
    D2H  out (N,13) + grad_id (N,4)           = 68N bytes   [DC]
         + grad_qg (N,4) + grad_qd (N,4)      = 100N bytes  [tran]
The solver cannot proceed until the results land, so none of it can be
hidden behind async compute.

Compares:
  pageable   - what the code does today (torch.tensor(python_list, device=))
  contiguous - one pre-built CPU tensor -> .to(dev)
  pinned     - pinned staging buffer, non_blocking
"""
import time
import torch

dev = torch.device("cuda")


def timeit(fn, reps=50, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


print("=== round-trip latency floor (H2D 1 float + sync + D2H 1 float) ===")
h = torch.zeros(1)


def rt_floor():
    x = h.to(dev, non_blocking=False)
    return x.cpu()


print(f"  empty round trip: {timeit(rt_floor)*1e6:8.1f} us\n")

print(f"{'N':>8} {'H2D KB':>8} {'D2H KB':>8} | "
      f"{'pageable':>10} {'contig':>10} {'pinned':>10} | "
      f"{'D2H .cpu()':>11} {'D2H tolist':>12} | {'total pinned':>13}")

for N in (384, 1536, 6144, 24576, 98304):
    rows = [(0.1, 0.2, 0.0, 0.0)] * N          # what batch_eval builds from
    cpu_contig = torch.tensor(rows, dtype=torch.float32)
    pinned = torch.empty(N, 4, dtype=torch.float32).pin_memory()
    pinned.copy_(cpu_contig)

    out_gpu = torch.randn(N, 13, device=dev)
    gid_gpu = torch.randn(N, 4, device=dev)
    out_pin = torch.empty(N, 13, dtype=torch.float32).pin_memory()
    gid_pin = torch.empty(N, 4, dtype=torch.float32).pin_memory()

    def h2d_pageable():
        return torch.tensor(rows, dtype=torch.float32, device=dev)

    def h2d_contig():
        return cpu_contig.to(dev)

    def h2d_pinned():
        return pinned.to(dev, non_blocking=True)

    def d2h_cpu():
        return out_gpu.cpu(), gid_gpu.cpu()

    def d2h_tolist():
        return out_gpu.tolist(), gid_gpu.tolist()

    def d2h_pinned():
        out_pin.copy_(out_gpu, non_blocking=True)
        gid_pin.copy_(gid_gpu, non_blocking=True)
        torch.cuda.synchronize()
        return out_pin, gid_pin

    t_pg = timeit(h2d_pageable, 20)
    t_ct = timeit(h2d_contig, 20)
    t_pn = timeit(h2d_pinned, 20)
    t_dc = timeit(d2h_cpu, 20)
    t_dl = timeit(d2h_tolist, 5)
    t_dp = timeit(d2h_pinned, 20)

    print(f"{N:>8} {16*N/1024:>8.1f} {68*N/1024:>8.1f} | "
          f"{t_pg*1e3:>9.3f}m {t_ct*1e3:>9.3f}m {t_pn*1e3:>9.3f}m | "
          f"{t_dc*1e3:>10.3f}m {t_dl*1e3:>11.3f}m | "
          f"{(t_pn+t_dp)*1e3:>12.3f}m")
