* BSIM-CMG CMOS Inverter - NO pseudo-capacitors for testing
.include /home/shenshan/NN_SPICE/PyCMG/tech_model_cards/asap7_pdk_r1p7/models/hspice/7nm_TT_160803.pm

* Higher VDD (2V)
Vdd 1 0 2.0

* Input pulse: 0 -> 2V
Vin 2 0 PULSE 0 2.0 0.5n 0.1n 0.1n 0.8n 2n

* PMOS LVT
Mp1 3 2 1 1 pmos_lvt L=30n NFIN=10

* NMOS LVT
Mn1 3 2 0 0 nmos_lvt L=30n NFIN=10

* Load capacitance (10fF)
Cload 3 0 10e-15

* Transient analysis with small dt (1ps)
.tran 1p 5n

.end
