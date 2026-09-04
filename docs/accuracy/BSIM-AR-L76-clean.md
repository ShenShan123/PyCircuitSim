# BSIM-AR-Full (LEVEL=76) — clean-matrix status

V7.7.0 keeps LEVEL=76 as the supported autoregressive full-terminal
alternative, but no complete five-technology clean matrix is available. This
file deliberately contains no synthesized scoreboard.

The retained TSMC5 development evidence is in
[`BSIM-AR-L76-simple-circuits.md`](BSIM-AR-L76-simple-circuits.md). It records
complete small/medium/large/xl experiments, including explicit Miller opamp
errors and non-monotonic capacity results. Those counts cannot be promoted to
or compared with a five-technology `/20` clean matrix.

A replacement report may be generated only after `dnf`/`tff` campaign tooling
confirms every declared LEVEL=76 checkpoint bundle and gate cell in one
provenance-bound pass. Until then, missing rows remain missing rather than
being backfilled from LEVEL=74 or partial LEVEL=76 campaigns.
