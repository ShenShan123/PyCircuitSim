* Voltage Divider Circuit
* This circuit demonstrates a simple resistive voltage divider
* Input: 10V DC source
* Output: Should be 5V at the middle node (equal resistors)

V1 1 0 10
R1 1 2 1k
R2 2 0 1k

* DC Sweep Analysis
* Sweep V1 from 0V to 10V in 1V steps
.dc V1 0 10 1

.end
