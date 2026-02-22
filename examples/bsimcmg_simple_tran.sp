* BSIM-CMG Simple Transient Test
* Just a capacitor to test transient analysis

* Power supply
Vdd 1 0 1.0

* Resistor
R1 1 2 1k

* Capacitor
C1 2 0 10e-15

* Transient analysis: 0 to 1ns with 10ps steps
.tran 10p 1n

.end
