# Results -- AnalogGym on TSMC12 (BSIM-CMG via PyCMG)

Every number below comes from running the decks in this tree under ngspice-45.2 with the
`bsimcmg.osdi` binary from `PyCircuitSim/external_compact_models/PyCMG`. Nothing is copied from
the AnalogGym paper or from the sky130 port.


## Coverage

| category | designs | fully passing | partial |
|---|--:|--:|--:|
| amplifier | 17 | 17 | 0 |
| ldo | 5 | 5 | 0 |
| sensing_front_end | 13 | 13 | 0 |
| voltage_reference | 2 | 2 | 0 |
| charge_pump | 1 | 1 | 0 |
| **total** | **38** | **38** | **0** |


## Amplifier

| design | A_v (dB) | GBW | PM (deg) | P | CMRR (dB) | PSRR+ (dB) | PSRR- (dB) | Vos | SR+ (V/us) | SR- (V/us) | pass |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| Alfio_RAFFC_Pin_3 | 110.4 | 3.64 MHz | 82.9 | 281 µW | -85.5 | -79.7 | -86.5 | -237 µV | 0.688 | 1.303 | 10/10 |
| Fan_SMC_Pin_3 | 112.7 | 506 kHz | 55.7 | 8.93 µW | -35.5 | -35.0 | -61.3 | 1.08 mV | 0.038 | 0.056 | 10/10 |
| HoiLee_AFFC_Pin_3 | 97.3 | 598 kHz | 54.4 | 6.32 µW | -40.1 | -40.3 | -75.1 | 1.57 mV | 0.893 | 0.033 | 10/10 |
| Leung_DFCFC1_Pin_3 | 98.8 | 540 kHz | 64.6 | 52 µW | -72.9 | -64.5 | -68.4 | 783 µV | 0.113 | 0.108 | 10/10 |
| Leung_DFCFC2_Pin_3 | 116.2 | 3.52 MHz | 55.4 | 68.2 µW | -36.3 | -35.0 | -52.2 | 1.73 mV | 0.264 | 1.018 | 10/10 |
| Leung_NMCF_Pin_3 | 113.6 | 543 kHz | 62.8 | 28.9 µW | -49.0 | -48.9 | -94.2 | 317 µV | 0.062 | 0.071 | 10/10 |
| Leung_NMCNR_Pin_3 | 127.0 | 421 kHz | 163.4 | 11.8 µW | -79.2 | -81.4 | -92.2 | 118 µV | 0.015 | 0.014 | 10/10 |
| Peng_ACBC_Pin_3 | 104.7 | 513 kHz | 67.7 | 2.53 µW | -38.6 | -38.4 | -65.9 | 1.26 mV | 0.048 | 0.017 | 10/10 |
| Peng_IAC_Pin_3 | 106.0 | 856 kHz | 55.0 | 141 µW | -36.0 | -35.0 | -54.4 | 2.07 mV | 0.077 | 0.119 | 10/10 |
| Peng_TCFC_Pin_3 | 121.9 | 500 kHz | 55.2 | 6.01 µW | -58.9 | -58.2 | -80.4 | -720 µV | 0.038 | 0.136 | 10/10 |
| Qu2017_AZC_Pin_3 | 127.2 | 5.15 MHz | 99.3 | 99.7 µW | -38.0 | -36.0 | -50.2 | 838 µV | 0.244 | 1.655 | 10/10 |
| Qu_LEC_Pin_3 | 125.6 | 2.63 MHz | 88.4 | 13.1 µW | -93.8 | -108.8 | -95.6 | 7.09 µV | 0.177 | 0.714 | 10/10 |
| Ramos_PFC_Pin_3 | 132.6 | 517 kHz | 178.6 | 34.3 µW | -51.0 | -50.8 | -83.2 | 73 µV | 0.061 | 0.146 | 10/10 |
| Sau_CFCC_Pin_3 | 90.8 | 907 kHz | 55.1 | 2.82 µW | -72.1 | -71.0 | -84.4 | 812 µV | 0.244 | 0.492 | 10/10 |
| Song_DACFC_Pin_3 | 111.8 | 633 kHz | 55.8 | 21.8 µW | -74.7 | -54.6 | -55.5 | -1.85 mV | 0.061 | 0.320 | 10/10 |
| Tan_CLIA_Pin_3 | 119.7 | 644 kHz | 55.9 | 3.72 µW | -36.9 | -36.1 | -57.5 | 2.61 mV | 0.168 | 0.042 | 10/10 |
| Yan_AZ_Pin_3 | 128.1 | 601 kHz | 57.5 | 6.41 µW | -48.8 | -44.1 | -51.8 | 137 µV | 0.013 | 0.016 | 10/10 |


## Low Dropout Regulator

| design | Vout max load | Vout min load | line reg | load reg | P max load | A_v (dB) | GBW | PM (deg) | PSRR (dB) | undershoot | pass |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| Basic_LDO | 599 mV | 602 mV | 0.0161 | 0.0815 | 44.6 mW | 66.4 | 5.71 MHz | 109.1 | -47.5 | 5.19 mV | 9/9 |
| ldo_1 | 621 mV | 621 mV | 0.0022 | 0.0159 | 44 mW | 63.3 | 174 kHz | 94.3 | -62.8 | 167 mV | 9/9 |
| ldo_2 | 558 mV | 558 mV | 0.0140 | 0.0005 | 44.2 mW | 60.8 | 21.4 MHz | 165.5 | -43.9 | 27.6 mV | 9/9 |
| ldo_folded_cascode | 607 mV | 621 mV | 0.0201 | 0.4524 | 45.1 mW | 47.7 | 19.9 MHz | 54.1 | -43.4 | 14 mV | 9/9 |
| ldo_simple | 643 mV | 645 mV | 0.1485 | 0.0875 | 44 mW | 47.7 | 1.44 MHz | 63.5 | -28.5 | 61.7 mV | 9/9 |


## Sensing Front End

| design | Vout @25 C | sensitivity 25-75 C | spread -20..120 C | Vout @0 C | Vout @100 C | pass |
|---|--:|--:|--:|--:|--:|:--:|
| PTAT_65_classic1 | 537 mV | 375 µV/C | 52.2 mV | 528 mV | 565 mV | 4/4 |
| PTAT_CLASSIC | 472 mV | 664 µV/C | 72.8 mV | 455 mV | 516 mV | 4/4 |
| PTAT_SENSOR | 207 mV | 527 µV/C | 77.3 mV | 195 mV | 249 mV | 4/4 |
| front_end_11_6T_schematic | 58.6 mV | 2.55 mV/C | 148 mV | 55.3 mV | 195 mV | 4/4 |
| front_end_25_6T_schematic | 118 mV | 355 µV/C | 50.5 mV | 110 mV | 146 mV | 4/4 |
| front_end_31_3T_schematic | 226 mV | 669 µV/C | 94.3 mV | 213 mV | 279 mV | 4/4 |
| front_end_42_2_2015_REF_schematic | 124 mV | 405 µV/C | 57.5 mV | 115 mV | 155 mV | 4/4 |
| ptat_1 | 115 mV | 336 µV/C | 48.2 mV | 106 mV | 140 mV | 4/4 |
| ptat_2 | 139 mV | 3.87 mV/C | 223 mV | 128 mV | 338 mV | 4/4 |
| ptat_3 | 144 mV | 520 µV/C | 76.6 mV | 132 mV | 185 mV | 4/4 |
| ptat_4 | 116 mV | 393 µV/C | 47 mV | 108 mV | 146 mV | 4/4 |
| ptat_6 | 139 mV | 3.87 mV/C | 223 mV | 128 mV | 338 mV | 4/4 |

The amplifier shipped in this category runs its own AC bench (gain, GBW, true phase margin, power):

| design | A_v (dB) | GBW | PM (deg) | P | pass |
|---|--:|--:|--:|--:|:--:|
| SMCNR_SE_2st_AMP | 72.5 | 654 kHz | 121.2 | 2.09 µW | 4/4 |


## Voltage Reference

| design | output | Vref @25 C | spread | TC (ppm/C) | pass |
|---|---|--:|--:|--:|:--:|
| dual_output_subthreshold_vref | vref1 | 124 mV | 3.35 mV | 163 | 3/3 |
|  | vref2 | 343 mV | 6.32 mV | 111 |  |
| three_output_vref | vref1 | 124 mV | 3.35 mV | 163 | 3/3 |
|  | vref2 | 343 mV | 6.32 mV | 111 |  |
|  | vref3 | 171 mV | 3.16 mV | 111 |  |


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
