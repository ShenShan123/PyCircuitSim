* BSIM-CMG PMOS DC Characterization
* Tests LEVEL=72 PMOS model integration

* Power supply
Vdd 1 0 1.0
Vgs 2 1 -0.8
Vds 3 1 -0.05

* PMOS using BSIM-CMG (LEVEL=72)
* Terminal order: drain gate source bulk
Mp1 3 2 1 1 pmos1 L=30n NFIN=10

* Model definition (references generic BSIM-CMG modelcard)
.model pmos1 PMOS (LEVEL=72)

* DC sweep: Vgs from 1V to 0V (relative to Vdd)
.dc Vgs 1.0 0.0 -0.02

.end
