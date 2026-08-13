# Results -- AnalogGym on TSMC5 (BSIM-CMG via PyCMG)

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
| Alfio_RAFFC_Pin_3 | 75.9 | 541 kHz | 55.3 | 24 µW | -35.5 | -36.6 | -53.1 | -1.62 mV | 0.080 | 0.046 | 10/10 |
| Fan_SMC_Pin_3 | 108.9 | 3.87 MHz | 55.0 | 33.9 µW | -36.9 | -35.0 | -54.0 | 470 µV | 0.253 | 1.011 | 10/10 |
| Leung_NMCNR_Pin_3 | 89.2 | 413 kHz | 56.9 | 11.2 µW | -36.0 | -34.5 | -50.6 | 302 µV | 0.016 | 0.047 | 10/10 |
| Peng_IAC_Pin_3 | 106.6 | 548 kHz | 78.1 | 14.6 µW | -51.0 | -50.2 | -71.1 | 122 µV | 0.068 | 0.156 | 10/10 |
| Qu2017_AZC_Pin_3 | 105.6 | 1.21 MHz | 110.2 | 15.3 µW | -44.0 | -50.1 | -49.9 | -339 µV | 0.018 | 0.108 | 10/10 |
| Qu_LEC_Pin_3 | 119.6 | 2.11 MHz | 89.8 | 13.5 µW | -63.3 | -61.8 | -77.7 | 173 µV | 0.148 | 0.567 | 10/10 |
| Song_DACFC_Pin_3 | 98.6 | 561 kHz | 69.8 | 16.9 µW | -52.7 | -34.4 | -35.5 | -2.46 mV | 0.040 | 0.227 | 10/10 |


## Low Dropout Regulator

| design | Vout max load | Vout min load | line reg | load reg | P max load | A_v (dB) | GBW | PM (deg) | PSRR (dB) | undershoot | pass |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| Basic_LDO | 481 mV | 486 mV | 0.2019 | 0.2026 | 35.9 mW | 62.0 | 2.63 MHz | 98.1 | -21.3 | 43.8 mV | 9/9 |
| ldo_1 | 477 mV | 477 mV | 0.0677 | 0.0138 | 35.8 mW | 74.1 | 10.5 MHz | 64.6 | -25.3 | 51.1 mV | 9/9 |
| ldo_2 | 454 mV | 454 mV | 0.0584 | 0.0048 | 35.8 mW | 59.2 | 16.1 MHz | 169.5 | -38.1 | 101 mV | 9/9 |


## Sensing Front End

| design | Vout @25 C | sensitivity 25-75 C | spread -20..120 C | Vout @0 C | Vout @100 C | pass |
|---|--:|--:|--:|--:|--:|:--:|
| front_end_11_6T_schematic | 98.9 mV | 2.17 mV/C | 268 mV | 90.5 mV | 346 mV | 4/4 |
| front_end_25_6T_schematic | 108 mV | 439 µV/C | 61.8 mV | 97.2 mV | 141 mV | 4/4 |
| ptat_1 | 61.9 mV | 1.29 mV/C | 396 mV | 55.3 mV | 260 mV | 4/4 |
| ptat_2 | 107 mV | 2.58 mV/C | 155 mV | 96.8 mV | 241 mV | 4/4 |
| ptat_4 | 155 mV | 1.68 mV/C | 446 mV | 140 mV | 426 mV | 4/4 |

The amplifier shipped in this category runs its own AC bench (gain, GBW, true phase margin, power):

| design | A_v (dB) | GBW | PM (deg) | P | pass |
|---|--:|--:|--:|--:|:--:|
| SMCNR_SE_2st_AMP | 69.9 | 638 kHz | 120.8 | 1.68 µW | 4/4 |


## Voltage Reference

| design | output | Vref @25 C | spread | TC (ppm/C) | pass |
|---|---|--:|--:|--:|:--:|
| dual_output_subthreshold_vref | vref1 | 251 mV | 13.7 mV | 335 | 3/3 |
|  | vref2 | 494 mV | 25.7 mV | 323 |  |


## Charge Pump

| measurement | value |
|---|--:|
| source (up) current, average | 8.62 µA |
| sink (down) current, average | 8.62 µA |
| up/down mismatch | 0.00 % |
| up current range | -4.03 µA ... 9.02 µA |
| down current range | 8.24 µA ... 9.48 µA |



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
mismatch. Simulations use the 0.65 V core rail for TSMC5.
