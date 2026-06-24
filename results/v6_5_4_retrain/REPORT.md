# V6.5.4 fresh-retrain per-size evaluation

## tsmc5 — best size: **large** (3/4)

| size | npass | opamp | ring | swcap | sram |
|---|---|---|---|---|---|
| large ⭐ | 3/4 | **P** 2.10 | F 12.66 | **P** 3.48 | **P** 4.40 |
| small | 2/4 | F 100.00 | F 8.06 | **P** 1.72 | **P** 7.80 |
| medium | 2/4 | F 100.00 | F 5.89 | **P** 1.56 | **P** 4.80 |
| xl | 2/4 | F 100.00 | F 13.50 | **P** 3.18 | **P** 3.90 |

## tsmc7 — best size: **large** (3/4)

| size | npass | opamp | ring | swcap | sram |
|---|---|---|---|---|---|
| large ⭐ | 3/4 | F 99.99 | **P** 4.82 | **P** 2.45 | **P** 68.20 |
| medium | 2/4 | F 99.99 | F 10.86 | **P** 2.81 | **P** 43.20 |
| xl | 2/4 | F 99.99 | F 14.31 | **P** 2.67 | **P** 4.30 |
| small | 1/4 | F 10.33 | F 5.94 | F 2.34 | **P** 72.40 |

## tsmc12 — best size: **large** (4/4)

| size | npass | opamp | ring | swcap | sram |
|---|---|---|---|---|---|
| large ⭐ | 4/4 | **P** 6.25 | **P** 4.04 | **P** 4.14 | **P** 5.10 |
| medium | 3/4 | F 100.00 | **P** 2.26 | **P** 4.19 | **P** 5.40 |
| xl | 3/4 | F 100.00 | **P** 3.40 | **P** 4.19 | **P** 5.20 |
| small | 2/4 | F 100.00 | **P** 1.95 | F 4.09 | **P** 5.10 |

## tsmc16 — best size: **medium** (3/4)

| size | npass | opamp | ring | swcap | sram |
|---|---|---|---|---|---|
| medium ⭐ | 3/4 | F 100.00 | **P** 2.22 | **P** 3.22 | **P** 5.70 |
| large | 3/4 | F 100.00 | **P** 2.59 | **P** 3.32 | **P** 5.30 |
| xl | 3/4 | F 100.00 | **P** 3.05 | **P** 3.42 | **P** 5.30 |
| small | 2/4 | F 10.23 | **P** 1.47 | F 2.76 | **P** 6.80 |

## Best-size-per-tech mix: **13/16**

| tech | best size | gates |
|---|---|---|
| tsmc5 | large | opamp✓ ring✗ swcap✓ sram✓ |
| tsmc7 | large | opamp✗ ring✓ swcap✓ sram✓ |
| tsmc12 | large | opamp✓ ring✓ swcap✓ sram✓ |
| tsmc16 | medium | opamp✗ ring✓ swcap✓ sram✓ |