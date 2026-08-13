* BSIM-AR NMOS DC Sweep (Id-Vgs) -- LEVEL=74
* The autoregressive Transformer counterpart of directnet_nmos_dc.sp: same
* circuit, same bias, different compact-model family. AR inference is
* ~30-100x slower on CPU than DirectNet, so this deck is deliberately small.
* TECH/VT are REQUIRED -- see the note in directnet_nmos_dc.sp.
* Resolves tsmc5_tf_large_nmos.

* Power supply
Vds drain 0 0.5
Vgs gate 0 0.0

* NMOS using BSIM-AR (LEVEL=74)
Mn1 drain gate 0 0 nmos_ar L=30n NFIN=10

* Model definition
.model nmos_ar NMOS (LEVEL=74 TECH=tsmc5 VT=svt)

* DC sweep: Vgs from 0 to 0.65V (the tsmc5 rail)
.dc Vgs 0 0.65 0.01

.end
