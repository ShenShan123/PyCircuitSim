# Results -- AnalogGym on TSMC12 (BSIM-CMG via PyCMG)

Every number below comes from running the decks in this tree under ngspice-45.2 with the
`bsimcmg.osdi` binary from `PyCircuitSim/external_compact_models/PyCMG`. Nothing is copied from
the AnalogGym paper or from the sky130 port.


## Coverage

| category | designs | fully passing | partial |
|---|--:|--:|--:|
| amplifier | 7 | 7 | 0 |
| ldo | 3 | 3 | 0 |
| sensing_front_end | 6 | 6 | 0 |
| voltage_reference | 1 | 1 | 0 |
| charge_pump | 1 | 1 | 0 |
| **total** | **18** | **18** | **0** |


## Amplifier

| design | A_v (dB) | GBW | PM (deg) | P | CMRR (dB) | PSRR+ (dB) | PSRR- (dB) | Vos | SR+ (V/us) | SR- (V/us) | pass |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| Alfio_RAFFC_Pin_3 | 110.4 | 3.64 MHz | 82.9 | 281 µW | -85.5 | -79.7 | -86.5 | -237 µV | 0.688 | 1.303 | 10/10 |
| Fan_SMC_Pin_3 | 112.7 | 506 kHz | 55.7 | 8.93 µW | -35.5 | -35.0 | -61.3 | 1.08 mV | 0.038 | 0.056 | 10/10 |
| Leung_NMCNR_Pin_3 | 127.0 | 421 kHz | 163.4 | 11.8 µW | -79.2 | -81.4 | -92.2 | 118 µV | 0.015 | 0.014 | 10/10 |
| Peng_IAC_Pin_3 | 106.0 | 856 kHz | 55.0 | 141 µW | -36.0 | -35.0 | -54.4 | 2.07 mV | 0.077 | 0.119 | 10/10 |
| Qu2017_AZC_Pin_3 | 127.2 | 5.15 MHz | 99.3 | 99.7 µW | -38.0 | -36.0 | -50.2 | 838 µV | 0.244 | 1.655 | 10/10 |
| Qu_LEC_Pin_3 | 125.6 | 2.63 MHz | 88.4 | 13.1 µW | -93.8 | -108.8 | -95.6 | 7.09 µV | 0.177 | 0.714 | 10/10 |
| Song_DACFC_Pin_3 | 111.8 | 633 kHz | 55.8 | 21.8 µW | -74.7 | -54.6 | -55.5 | -1.85 mV | 0.061 | 0.320 | 10/10 |


## Low Dropout Regulator

| design | Vout max load | Vout min load | line reg | load reg | P max load | A_v (dB) | GBW | PM (deg) | PSRR (dB) | undershoot | pass |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| Basic_LDO | 599 mV | 602 mV | 0.0161 | 0.0815 | 44.6 mW | 66.4 | 5.71 MHz | 109.1 | -47.5 | 5.19 mV | 9/9 |
| ldo_1 | 621 mV | 621 mV | 0.0022 | 0.0159 | 44 mW | 63.3 | 174 kHz | 94.3 | -62.8 | 167 mV | 9/9 |
| ldo_2 | 558 mV | 558 mV | 0.0140 | 0.0005 | 44.2 mW | 60.8 | 21.4 MHz | 165.5 | -43.9 | 27.6 mV | 9/9 |


## Sensing Front End

| design | Vout @25 C | sensitivity 25-75 C | spread -20..120 C | Vout @0 C | Vout @100 C | pass |
|---|--:|--:|--:|--:|--:|:--:|
| front_end_11_6T_schematic | 58.6 mV | 2.55 mV/C | 148 mV | 55.3 mV | 195 mV | 4/4 |
| front_end_25_6T_schematic | 118 mV | 355 µV/C | 50.5 mV | 110 mV | 146 mV | 4/4 |
| ptat_1 | 115 mV | 336 µV/C | 48.2 mV | 106 mV | 140 mV | 4/4 |
| ptat_2 | 139 mV | 3.87 mV/C | 223 mV | 128 mV | 338 mV | 4/4 |
| ptat_4 | 116 mV | 393 µV/C | 47 mV | 108 mV | 146 mV | 4/4 |

The amplifier shipped in this category runs its own AC bench (gain, GBW, true phase margin, power):

| design | A_v (dB) | GBW | PM (deg) | P | pass |
|---|--:|--:|--:|--:|:--:|
| SMCNR_SE_2st_AMP | 72.5 | 654 kHz | 121.2 | 2.09 µW | 4/4 |


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
