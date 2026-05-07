# V5 Phase C — C0 FD-vs-autograd Jacobian diagnostic

Tolerance: BAD when |FD - autograd| > rel_tol * max(|FD|, 1e-6).

| ckpt | polarity | n_total | BAD% (ID) | BAD% (OOD) | mean rel.err (ID) | mean rel.err (OOD) |
|---|---|---|---|---|---|---|
| `v4_dn_universal_nmos` | nmos | 9600 | 7.8 | 4.9 | 0.1321 | 0.0601 |
| `v4_dn_universal_pmos` | pmos | 9600 | 9.0 | 4.9 | 0.0962 | 0.0677 |
| `v5_dn_s_nmos_mae_nmos` | nmos | 9600 | 2.9 | 3.0 | 0.0373 | 0.0217 |
| `v5_dn_s_pmos_mae_pmos` | pmos | 9600 | 5.1 | 4.6 | 0.0367 | 0.0350 |
| `v5_dn_s_nmos_jac_nmos` | nmos | 9600 | 7.0 | 3.5 | 0.0902 | 0.0409 |
| `v5_dn_s_pmos_jac_pmos` | pmos | 9600 | 8.7 | 6.8 | 0.0557 | 0.0568 |
