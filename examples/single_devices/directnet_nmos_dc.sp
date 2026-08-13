* DirectNet NMOS DC Sweep (Id-Vgs) -- LEVEL=73
* TECH/VT are REQUIRED: without them the parser defaults to TECH=asap7, which
* maps to the UNKNOWN embedding row and resolves the universal-scope
* checkpoint that is not built on this hardware (fixed V7.5.7).
* tsmc5 svt at its own 0.65 V rail; resolves tsmc5_dn_large_nmos.

* Power supply
Vds drain 0 0.5
Vgs gate 0 0.0

* NMOS using DirectNet (LEVEL=73)
Mn1 drain gate 0 0 nmos_nn L=30n NFIN=10

* Model definition
.model nmos_nn NMOS (LEVEL=73 TECH=tsmc5 VT=svt)

* DC sweep: Vgs from 0 to 0.65V (the tsmc5 rail)
.dc Vgs 0 0.65 0.01

.end
