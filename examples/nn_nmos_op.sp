* NN-based NMOS Operating Point Test
* Tests LEVEL=73 NN model integration

* Power supply
Vds drain 0 0.5
Vgs gate 0 0.5

* NMOS using NN model (LEVEL=73)
Mn1 drain gate 0 0 nmos_nn L=30n NFIN=10

* Model definition
.model nmos_nn NMOS (LEVEL=73)

* Operating point analysis
.op

.end
