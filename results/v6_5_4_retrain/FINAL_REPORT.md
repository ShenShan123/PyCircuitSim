# V6.5.4 FINAL — fresh retrain, best config per tech (size + seed)

## tsmc5 — best: **tsmc5_dn_large** (3/4)

| candidate | npass | opamp | ring | swcap | sram |
|---|---|---|---|---|---|
| `tsmc5_dn_large` ⭐ | 3/4 | **P** 2.10 | F 12.66 | **P** 3.48 | **P** 4.40 |
| `tsmc5_dn_lgs7` | 3/4 | **P** 0.15 | F 12.64 | **P** 3.56 | **P** 3.40 |
| `tsmc5_dn_medium` | 2/4 | F 100.00 | F 5.89 | **P** 1.56 | **P** 4.80 |
| `tsmc5_dn_small` | 2/4 | F 100.00 | F 8.06 | **P** 1.72 | **P** 7.80 |
| `tsmc5_dn_xl` | 2/4 | F 100.00 | F 13.50 | **P** 3.18 | **P** 3.90 |
| `tsmc5_dn_lgs17` | 2/4 | F 100.00 | F 9.64 | **P** 2.53 | **P** 4.30 |
| `tsmc5_dn_lgs31` | 2/4 | F 100.00 | F 12.53 | **P** 3.35 | **P** 3.30 |

## tsmc7 — best: **tsmc7_dn_large** (3/4)

| candidate | npass | opamp | ring | swcap | sram |
|---|---|---|---|---|---|
| `tsmc7_dn_large` ⭐ | 3/4 | F 99.99 | **P** 4.82 | **P** 2.45 | **P** 68.20 |
| `tsmc7_dn_medium` | 2/4 | F 99.99 | F 10.86 | **P** 2.81 | **P** 43.20 |
| `tsmc7_dn_xl` | 2/4 | F 99.99 | F 14.31 | **P** 2.67 | **P** 4.30 |
| `tsmc7_dn_lgs7` | 2/4 | F 99.99 | F 7.15 | **P** 2.42 | **P** 1.80 |
| `tsmc7_dn_lgs31` | 2/4 | F 99.99 | F 12.62 | **P** 2.62 | **P** 46.00 |
| `tsmc7_dn_lgs17` | 2/4 | F 121.20 | F 8.69 | **P** 2.50 | **P** 3.20 |
| `tsmc7_dn_small` | 1/4 | F 10.33 | F 5.94 | F 2.34 | **P** 72.40 |

## tsmc12 — best: **tsmc12_dn_large** (4/4)

| candidate | npass | opamp | ring | swcap | sram |
|---|---|---|---|---|---|
| `tsmc12_dn_large` ⭐ | 4/4 | **P** 6.25 | **P** 4.04 | **P** 4.14 | **P** 5.10 |
| `tsmc12_dn_medium` | 3/4 | F 100.00 | **P** 2.26 | **P** 4.19 | **P** 5.40 |
| `tsmc12_dn_xl` | 3/4 | F 100.00 | **P** 3.40 | **P** 4.19 | **P** 5.20 |
| `tsmc12_dn_small` | 2/4 | F 100.00 | **P** 1.95 | F 4.09 | **P** 5.10 |

## tsmc16 — best: **tsmc16_dn_lgs17** (4/4)

| candidate | npass | opamp | ring | swcap | sram |
|---|---|---|---|---|---|
| `tsmc16_dn_lgs17` ⭐ | 4/4 | **P** 6.16 | **P** 3.40 | **P** 3.31 | **P** 5.40 |
| `tsmc16_dn_medium` | 3/4 | F 100.00 | **P** 2.22 | **P** 3.22 | **P** 5.70 |
| `tsmc16_dn_large` | 3/4 | F 100.00 | **P** 2.59 | **P** 3.32 | **P** 5.30 |
| `tsmc16_dn_xl` | 3/4 | F 100.00 | **P** 3.05 | **P** 3.42 | **P** 5.30 |
| `tsmc16_dn_lgs7` | 3/4 | F 100.00 | **P** 2.43 | **P** 3.30 | **P** 5.30 |
| `tsmc16_dn_lgs31` | 3/4 | F 99.99 | **P** 4.16 | **P** 3.31 | **P** 5.30 |
| `tsmc16_dn_small` | 2/4 | F 10.23 | **P** 1.47 | F 2.76 | **P** 6.80 |

## FINAL best-config-per-tech mix: **14/16**

| tech | config | gates |
|---|---|---|
| tsmc5 | `tsmc5_dn_large` | opamp✓ ring✗ swcap✓ sram✓ |
| tsmc7 | `tsmc7_dn_large` | opamp✗ ring✓ swcap✓ sram✓ |
| tsmc12 | `tsmc12_dn_large` | opamp✓ ring✓ swcap✓ sram✓ |
| tsmc16 | `tsmc16_dn_lgs17` | opamp✓ ring✓ swcap✓ sram✓ |