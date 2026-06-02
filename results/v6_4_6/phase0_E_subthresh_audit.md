# V6.4.6 Phase-0 P0-E — Subthreshold normalisation / sign-noise audit

- **Dataset:** `external_compact_models/bsimar/data/datasets/tsmc7_nmos.npz` (N=2,078,136)
- **Normaliser:** `external_compact_models/bsimar/checkpoints/tsmc7_dn_medium_nmos_norm.npz` (mode=`asinh`)
- **id column:** output col 0; gm col 1; gds col 2 (per `meta_output_columns`)
- **subthresh class:** code 2 (`meta_sample_class_names[2]`)
- **`asinh_scale[id]`** = `5.730942e-05`, `output_mean[id]`=`-1.0605`, `output_std[id]`=`2.0861`, `meta_vdd`=0.75

Input columns confirmed: `inputs (N,4)` = (Vgs, Vds, Vbs[=0 for NMOS], col3[Vbs sweep var]). Body col2 is identically 0 in this NMOS set; col3 carries the Vbs LHS sweep (range -0.541..0.750).

## 1. Asinh crush of the leakage band

- Full normalised id range over all finite rows: `[-3.1439, 4.1606]`, span **7.3046** norm-units.
- Normalised id at band edges (positive id, full mean/std forward): `z(1nA)=+0.50836`, `z(10nA)=+0.50844`, `z(100nA)=+0.50919`, `z(0)=+0.50836`.
- **The entire 1nA-100nA band spans only 0.00083 normalised-id units = 0.0113% of the full normalised id range.**
- 1 nA is **4.76 decades** below `asinh_scale[id]` (5.731e-05); `asinh(1nA/scale)=1.74e-05` (deep in the asinh linear regime) -> the whole sub-100nA band is crushed to within 0.0115% of z@0.
- On-state reference (99th-pct |id|=1.225e-02 A) maps to `z=+3.4124`; the leakage band sits 2.903 norm-units away.

**Headline:** asinh confirmed to crush the leakage band — the full 1nA-100nA leakage band is **0.0113% of the normalised id range**, so a pointwise MAE loss in normalised space carries effectively no signal there (training-loss gradient on the leakage band is negligible relative to the on-state band).

## 2. Floor-noise sign/zero fractions in the leakage band

Threshold: `|id| < 1e-07 A`. 263,924 rows (12.70% of 2,078,136).

| quantity | count | % of leakage band | % of ALL N rows |
|---|---:|---:|---:|
| negative (id<0) | 118,851 | **45.03%** | 5.72% |
| literal-zero (id==0) | 124,786 | **47.28%** | **6.00%** |
| positive (id>0) | 20,287 | 7.69% | 0.98% |

> The plan's preliminary estimate was *~45% negative, ~6% literal-0*. Confirmed: **45.0% negative of the band**, and literal-0 is **6.00% of ALL rows** (== the plan's ~6%) which is 47.3% *of the leakage band*. Most literal zeros are structural (Vds=0 forces id=0).

**Non-zero leakage tail** (`0<|id|<1e-07`, the part a reweight would actually touch): n=139,138, **85.42% negative** — sign-randomness is even stronger once structural zeros are excluded.

Decade histogram of `|id|` in the leakage band:

```
|id| == 0 (literal)           : 124786
0 < |id| < 1e-12             : 0
1e-12 <= |id| < 1e-11        : 1
1e-11 <= |id| < 1e-10        : 0
1e-10 <= |id| < 1e-9        : 27
1e-9 <= |id| < 1e-8        : 68552
1e-8 <= |id| < 1e-7        : 70558
```

Sign split per decade (what fraction is negative inside each decade):

| decade | n | neg% |
|---|---:|---:|
| 1e-12..1e-11 | 1 | 100.00% |
| 1e-11..1e-10 | 0 | - |
| 1e-10..1e-9 | 27 | 29.63% |
| 1e-9..1e-8 | 68,552 | 84.15% |
| 1e-8..1e-7 | 70,558 | 86.68% |
| <1e-12 (incl 0) | 124,786 | 0.00% |

## 3. Subthreshold OSDI gm/gds cleanliness (class 2)

- subthresh class rows: **97,200**
- `|id|` range in class: `[0.000e+00, 6.585e-03] A` (id sign: neg 71.57%, zero 21.11%, pos 7.32%).

| metric | gm (col 1) | gds (col 2) |
|---|---:|---:|
| NaN/Inf | 0 (0.0000%) | 0 (0.0000%) |
| < 0 (wrong sign) | 1597 (1.6430%) | 0 (0.0000%) |
| == 0 | 3946 (4.0597%) | 0 (0.0000%) |
| range (finite) | [-8.252e-19, 3.632e-03] | [5.587e-15, 6.568e-03] |

Deep-off subset of the class (`|id|<1nA`, n=20,525): gm neg 7.77%, gm zero 19.23%, gm range `[-8.252e-19,1.241e-05]`.

## 4. DECISION

Decision criteria (per plan §4 P0-E):

- id-band sign-random (neg frac > 10%)? **True** (observed 45.0% negative).
- OSDI gm clean (finite & <5% wrong-sign)? **True**.
- OSDI gds clean (finite & <5% wrong-sign)? **True**.

**DECISION (subthresh band clean? no):** UNSAFE for a raw log-reweight / asinh-floor drop on the id VALUE.

- *Rationale:* the |id|<1e-7 band is sign-random PyCMG floor noise (45.0% negative of band, 47.28% literal-zero of band = 6.00% of all rows; the non-zero tail alone is 85.4% negative). Dropping/lowering the asinh floor (e.g. to 1e-9) would amplify this junk into the surface. 
- *Allowed lever:* ONLY a clipped/winsorised signed-floor target is allowed: restrict support to |id|>1e-12 A and apply Huber on ln(|id|) (signed), NOT a plain log-reweight of the raw id value. HOWEVER the analytic OSDI gm/gds in the subthresh class ARE clean (NaN/Inf 0.0000%/0.0000%, wrong-sign 1.6430%/0.0000%), so a Phase-2 Jacobian/log-DERIVATIVE distillation against OSDI gm/gds on the clipped support (|id|>1e-12) is the safe lever — the derivative target, unlike the raw id value, is not floor-noise.

### Plan §4 decision-tree line

`P0-E subthresh band clean? -> no -> clipped/winsorised signed-floor target only (|id|>1e-12, Huber on ln-current); raw asinh-floor drop / log-reweight on the id value is UNSAFE`

## Exact commands

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run -n pycircuitsim python scripts/v6_4_6_p0e_subthresh.py
# log : results/v6_4_6/phase0_logs/p0e_subthresh.log
# report (this file): results/v6_4_6/phase0_E_subthresh_audit.md
```

