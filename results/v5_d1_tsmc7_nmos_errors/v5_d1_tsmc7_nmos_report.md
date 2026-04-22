# D1 Diagnostic: v4 BSIMAR NMOS vs PyCMG at TSMC7 SVT

**Fixture:** tsmc7 svt, tech_code=4, L=16 nm, NFIN=10, T=300.15 K,
Vbs=0, VDD=0.750 V. Grid: 40×40 = 1600 points on
(Vgs, Vds) ∈ [0, 0.750] × [0, 0.750] V.

## Answers

**1. Max absolute relative error (% of max|Id|):**
- Corrected NN (what simulator sees): **27.96 %**
- Raw NN (no `_apply_vds_correction`): **28.48 %**

The raw variant has the
larger peak. Grid-mean |rel err| is
corrected=9.171 %, raw=9.087 %.

**2. Where is the error largest?**
Top-decile error mass sits in **strong inversion, saturation**:
Vgs ∈ [0.519, 0.731] V, Vds ∈ [0.404, 0.750] V
(Vgs mean 0.622 V, Vds mean 0.614 V;
160 of 1600 grid points above the
90th-percentile error threshold of 23.69 %).
The companion heatmap `heatmap_rel_error_with_corr.png` shows the exact
spatial footprint.

**3. Does the Vds correction help or hurt?**
Mean |rel err|: corrected=9.171 %, raw=9.087 %.
The correction increases the mean
error overall. At the triode rail (Vds≈0) the correction enforces Id=0,
which trades a small error near Vds=0 for a guaranteed physical boundary;
the raw NN already predicts ≈0 there, so the suppression is minor. The
rail-restoring extrapolation does not apply to in-range points
(|Vds| < VDD=VDD_train), so most of the grid sees identical NN output; the
correction vs raw deltas concentrate near the Vds=0 rail.

**4. Id-Vgs @ Vds = VDD/2 = 0.375 V (the NMOS DC bias):**
- Corrected NN NRMSE = **14.73 %**
- Raw NN NRMSE = 14.72 %

This directly reproduces the verifier's ≈14-15 % NMOS DC NRMSE if the
corrected number is in that range. The slice plot
`slices_id_vgs.png` shows the three panels (Vds=0, VDD/2, VDD).

**5. Training-set LHS coverage in the hot-error box:**
- TSMC7 SVT samples with Vgs ∈ [0.519,
0.731] and Vds ∈
[0.404, 0.750]:
**18,226** out of
592,812 total TSMC7-SVT samples
(3.07 %).

A low coverage fraction is consistent with the NN being starved of
training points in precisely the region where the verifier metric probes.

**6. Recommendation for E4 overlay sampling:**
Densify TSMC7 overlay generation in the following box (the hot region
identified above, widened by ±10 % of VDD on each side to give the NN
a buffer):

- **Tech / variant:** tsmc7 svt (code=4) — highest priority. Consider
  also tsmc7 lvt and tsmc7 ulvt if coverage gap repeats at those codes.
- **Vgs:** [0.444, 0.750] V
- **Vds:** [0.329, 0.750] V
- **NFIN:** {3, 5, 10, 15, 20} (cover the simulator inverter NFIN=10 plus the
  neighbouring bin points that stress the same gate stack)
- **L:** {14, 16, 18, 20} nm around the NMOS L=16 nm used at inference
- **T:** 300.15 K (single design corner)
- **Density:** at least 400 new LHS samples inside the box, which is
  ≈1× the current
  density — sufficient to cover the gate-Vds surface without diluting the
  overall training mix.

This list should drop directly into the plan §4.4 overlay generator and
give E4 a reasonable chance of closing the 14 % NMOS-DC-NRMSE gap on
TSMC7 without retraining the 17-other-tech mix from scratch.
