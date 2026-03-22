* NN-based NMOS DC Sweep (Id-Vgs)
* Tests LEVEL=73 NN model with DC sweep

* Power supply
Vds drain 0 0.5
Vgs gate 0 0.5

* NMOS using NN model (LEVEL=73)
Mn1 drain gate 0 0 nmos_nn L=30n NFIN=10

* Model definition
.model nmos_nn NMOS (LEVEL=73)

* DC sweep: Vgs from 0 to 0.7V
.dc Vgs 0 0.7 0.01

.end
