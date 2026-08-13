* DirectNet NMOS Operating Point -- LEVEL=73
* Single bias point; the counterpart of directnet_nmos_dc.sp.
* TECH/VT are REQUIRED -- see the note in directnet_nmos_dc.sp.

* Power supply
Vds drain 0 0.5
Vgs gate 0 0.5

* NMOS using DirectNet (LEVEL=73)
Mn1 drain gate 0 0 nmos_nn L=30n NFIN=10

* Model definition
.model nmos_nn NMOS (LEVEL=73 TECH=tsmc5 VT=svt)

* Operating point analysis
.op

.end
