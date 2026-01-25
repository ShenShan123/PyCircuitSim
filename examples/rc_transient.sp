* RC Low-Pass Filter with Step Response
* Two-stage RC filter for more interesting transient response

V1 1 0 5
R1 1 2 1k
C1 2 0 1n
R2 2 3 1k
C2 3 0 1n

* Transient Analysis
* Simulate 20 microseconds to see full charging
.tran 200n 20u

.end
