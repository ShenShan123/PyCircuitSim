# Definitive routing table at the user's real target range.
# solver: measured synthetic spsolve, /12 for the documented synthetic pessimism
# NN GPU: measured movement-inclusive (NMOS) x1.5 for the PMOS group
# NN CPU: measured us/dev curve (50.5@384, 55.4@1536, 85.4@6144, 201@24576), extrapolated
ITERS=41
rows=[  # array, devices, N, spsolve_ms_synthetic, gpu_nmos_ms
 ("32x16",   3072,  1154,    14.9,  1.65),
 ("64x32",  12288,  4354,   410.0,  2.07),
 ("96x48",  27648,  9602,  2153.9,  4.0),
 ("128x64", 49152, 16898, 27747.3,  6.17),
 ("160x80", 76800, 26242, 67325.4,  9.5),
 ("192x96",110592, 37634,131513.2, 13.04),
 ("256x128",196608,66562,401000.0, 22.24),  # spsolve extrapolated ~N^1.95
]
def nncpu_us(n):  # us/device, extrapolated beyond 24576 at the observed slope
    import numpy as np
    xs=[384,1536,6144,24576]; ys=[50.5,55.4,85.4,201.3]
    if n<=24576: return float(np.interp(n,xs,ys))
    return 201.3*(n/24576)**0.72
print(f"{'array':>9} {'devices':>8} {'N':>7} | {'NN CPU':>9} {'NN GPU':>8} | "
      f"{'solver':>10} | {'solver/NN_gpu':>13} | verdict")
for a,dev,N,sp_ms,gpu_ms in rows:
    nn_cpu = nncpu_us(dev)*dev/1e6*ITERS
    nn_gpu = gpu_ms*1.5/1e3*ITERS
    solver = sp_ms/12/1e3*ITERS
    ratio  = solver/nn_gpu
    v = "NN-bound" if ratio<1 else ("SOLVER-bound" if ratio>10 else "mixed")
    print(f"{a:>9} {dev:8d} {N:7d} | {nn_cpu:8.1f}s {nn_gpu:7.2f}s | {solver:9.1f}s | "
          f"{ratio:12.0f}x | {v}")
