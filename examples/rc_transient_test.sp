* Simple RC Circuit Transient Test

* Voltage source with step input
Vin 1 0 PULSE 0 1.0 1n 0.1n 0.1n 4n 10n

* Resistor
R1 1 2 1k

* Capacitor
C1 2 0 1p

* Transient analysis
.tran 10p 10n

.end
