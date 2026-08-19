# Results -- AnalogGym on TSMC12 (BSIM-CMG via PyCMG)

Every number below comes from running the decks in this tree under ngspice-45.2 with the
`bsimcmg.osdi` binary from `PyCircuitSim/external_compact_models/bsim_cmg`. Nothing is copied from
the AnalogGym paper or from the sky130 port.


## Coverage

| category | designs | fully passing | partial |
|---|--:|--:|--:|
| amplifier | 5 | 5 | 0 |
| ldo | 2 | 2 | 0 |
| sensing_front_end | 3 | 3 | 0 |
| voltage_reference | 1 | 1 | 0 |
| charge_pump | 1 | 1 | 0 |
| **total** | **12** | **12** | **0** |


## Amplifier

| design | A_v (dB) | GBW | PM (deg) | P | CMRR (dB) | PSRR+ (dB) | PSRR- (dB) | Vos | SR+ (V/us) | SR- (V/us) | pass |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| Fan_SMC_Pin_3 | 112.7 | 506 kHz | 55.7 | 8.93 µW | -35.5 | -35.0 | -61.3 | 1.08 mV | 0.038 | 0.056 | 10/10 |
| Leung_NMCNR_Pin_3 | 127.0 | 421 kHz | 163.4 | 11.8 µW | -79.2 | -81.4 | -92.2 | 118 µV | 0.015 | 0.014 | 10/10 |
| Peng_IAC_Pin_3 | 106.0 | 856 kHz | 55.0 | 141 µW | -36.0 | -35.0 | -54.4 | 2.07 mV | 0.077 | 0.119 | 10/10 |
| Qu2017_AZC_Pin_3 | 127.2 | 5.15 MHz | 99.3 | 99.7 µW | -38.0 | -36.0 | -50.2 | 838 µV | 0.244 | 1.655 | 10/10 |
| Song_DACFC_Pin_3 | 111.8 | 633 kHz | 55.8 | 21.8 µW | -74.7 | -54.6 | -55.5 | -1.85 mV | 0.061 | 0.320 | 10/10 |


## Low Dropout Regulator

| design | Vout max load | Vout min load | line reg | load reg | P max load | A_v (dB) | GBW | PM (deg) | PSRR (dB) | undershoot | pass |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| ldo_1 | 621 mV | 621 mV | 0.0022 | 0.0159 | 44 mW | 63.3 | 174 kHz | 94.3 | -62.8 | 167 mV | 9/9 |
| ldo_2 | 558 mV | 558 mV | 0.0140 | 0.0005 | 44.2 mW | 60.8 | 21.4 MHz | 165.5 | -43.9 | 27.6 mV | 9/9 |


## Sensing Front End

| design | Vout @25 C | sensitivity 25-75 C | spread -20..120 C | Vout @0 C | Vout @100 C | pass |
|---|--:|--:|--:|--:|--:|:--:|
| front_end_25_6T_schematic | 118 mV | 355 µV/C | 50.5 mV | 110 mV | 146 mV | 4/4 |
| ptat_1 | 115 mV | 336 µV/C | 48.2 mV | 106 mV | 140 mV | 4/4 |
| ptat_4 | 116 mV | 393 µV/C | 47 mV | 108 mV | 146 mV | 4/4 |


## Voltage Reference

| design | output | Vref @25 C | spread | TC (ppm/C) | pass |
|---|---|--:|--:|--:|:--:|
| dual_output_subthreshold_vref | vref1 | 124 mV | 3.35 mV | 163 | 3/3 |
|  | vref2 | 343 mV | 6.32 mV | 111 |  |


## Charge Pump

| measurement | value |
|---|--:|
| source (up) current, average | 9.11 µA |
| sink (down) current, average | 9.11 µA |
| up/down mismatch | 0.04 % |
| up current range | 4.6 µA ... 11.7 µA |
| down current range | 8.09 µA ... 11.2 µA |



## Audit criteria

`!` means that one or more requested analyses did not complete, so the design
is partial even if all available numeric gates pass. Amplifiers include gain,
GBW, true (unwrapped) phase margin, power, offset, CMRR, both PSRR polarities,
temperature drift, and both slew directions. LDOs include regulation, loop
gain/GBW/true phase margin, power, line and load regulation, both-load PSRR,
and load-step excursion. Sensors require in-rail output that rises
monotonically over every solved sweep point, 0.3-6 mV/C sensitivity, and a
staircase-free characteristic (local slope and single-step share gates in
tools/sfe.py); references require every output in range with <=500 ppm/C; the
charge pump requires both current directions, 2-200 uA magnitude, and <=5%
mismatch. Simulations use the 0.8 V core rail for TSMC12.
