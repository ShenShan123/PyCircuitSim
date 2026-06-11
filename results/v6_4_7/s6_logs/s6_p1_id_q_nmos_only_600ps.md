# S6=P1 swap matrix — mode `id_q_nmos_only` (TSMC7 RO, window 0.600 ns, settle 0.300 ns)

Start-state: S5b constrained-.ic (uic-equivalent), commit 7454034.

| mode | period (ps) | NG (ps) | err% | Δ vs pre-S5b 50.83 | NRMSE% | R2 | partial | reached (ns) | wall (s) | osdi calls/hits/fail |
|------|------------:|--------:|-----:|-------------------:|-------:|---:|:-------:|-------------:|---------:|---------------------:|
| id_q_nmos_only | 92.91 | 46.65 | 99.17 | +42.08 | 60.92 | -1.0736 | False | 0.6000 | 3104 | 357923/1085552/0 |

NN-vs-OSDI at injected biases: injected_evals=360095 fallbacks=0  max|Δqg|=2.977 aC  max|Δqd|=1.316 aC  max|Δid|=57.917 uA  max rel id (|id|>1uA)=116.1% at bias=(np.float64(0.782), np.float64(0.233), 0.0, 0.0)

**VERDICT: 92.91 ps vs NG 46.65 ps (99.2% off; P0-I id-only gave 92.30 ps) — INDICTS solver/harness**
