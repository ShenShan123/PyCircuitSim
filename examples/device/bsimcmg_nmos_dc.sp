* BSIM-CMG NMOS DC Characterization
* Tests LEVEL=72 model integration with pycircuitsim

* Power supply
Vds 2 0 0.05
Vgs 1 0 0.8

* NMOS using BSIM-CMG (LEVEL=72)
* Terminal order: drain gate source bulk
Mn1 2 1 0 0 nmos1 L=30n NFIN=10

* Model definition (references generic BSIM-CMG modelcard)
.model nmos1 NMOS (LEVEL=72)

* DC sweep: Vgs from 0 to 1V
.dc Vgs 0 1.0 0.02

.end
