"""Tiny probe: print BSIMAR/DirectNet Id and gds at various Vds."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["BSIMAR_PREFIX"] = "v4_probe_signfix"

from tests.verify_bsimar_v4_inverter import (
    TSMC5_SVT, create_bsimar_instance, create_directnet_instance,
)

print("=== NMOS @ Vgs=0, varying Vds ===")
bs = create_bsimar_instance(TSMC5_SVT, "nmos")
dn = create_directnet_instance(TSMC5_SVT, "nmos")
print(f"BSIMAR _vdd_estimate: {bs._vdd_estimate:.4f}V")
print(f"DirectNet _vdd_estimate: {dn._vdd_estimate:.4f}V")
print(f"{'Vds':>6s} | {'BS id [µA]':>12s} {'BS gds [µS]':>12s} | {'DN id [µA]':>12s} {'DN gds [µS]':>12s}")
print("-" * 75)
for vd in [0.0, 0.3, 0.65, 0.7, 1.0, 1.3, 2.0, 4.4]:
    bs.clear_cache()
    dn.clear_cache()
    rb = bs._eval({"drain": vd, "gate": 0.0, "source": 0.0, "bulk": 0.0})
    rd = dn._eval({"drain": vd, "gate": 0.0, "source": 0.0, "bulk": 0.0})
    print(f"{vd:>6.2f} | {rb['id']*1e6:>12.4f} {rb['gds']*1e6:>12.4f} "
          f"| {rd['id']*1e6:>12.4f} {rd['gds']*1e6:>12.4f}")
