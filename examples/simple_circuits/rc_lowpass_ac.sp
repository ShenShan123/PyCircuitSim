* RC Low-Pass Filter - AC (small-signal frequency-domain) Analysis
* Single-pole low-pass.  H(jw) = 1 / (1 + jw*R*C)
* -3dB corner:  fc = 1 / (2*pi*R*C)
*   R = 1k, C = 159.155nF  ->  fc = 1000 Hz (exact analytic anchor)

* AC voltage source: DC bias 0, AC magnitude 1V, phase 0 deg
V1 in 0 DC=0 AC=1 0

R1 in out 1k
C1 out 0 159.155n

* AC sweep: 20 points/decade from 10 Hz to 1 MHz (3 decades below/above fc)
.ac dec 20 10 1e6

.end
