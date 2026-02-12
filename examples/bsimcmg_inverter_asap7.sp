* BSIM-CMG CMOS Inverter DC Test using ASAP7 PDK model
* Include the ASAP7 modelcard file
.include /home/shenshan/NN_SPICE/PyCMG/tech_model_cards/asap7_pdk_r1p7/models/hspice/7nm_TT_160803.pm

* Power supply
Vdd 1 0 1.0

* Input voltage (DC)
Vin 2 0 0.0

* PMOS (source=Vdd, drain=out, gate=in, bulk=Vdd)
Mp1 3 2 1 1 pmos_lvt L=30n NFIN=10

* NMOS (drain=out, gate=in, source=GND, bulk=GND)
Mn1 3 2 0 0 nmos_lvt L=30n NFIN=10

* Note: Model definitions for LEVEL=72 are in the ASAP7 modelcard
* The .model names are: nmos_lvt, nmos_rvt, nmos_slvt, nmos_sram
*                   pmos_lvt, pmos_slvt, pmos_sram
* Reference the model by name directly (nmos1 references nmos_lvt, pmos1 references pmos_lvt)

.end
