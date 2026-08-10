"""Tests for the sweep module."""
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
OSDI_PATH = ROOT / "build" / "osdi" / "bsimcmg.osdi"
ASAP7_MODELCARD = ROOT / "modelcards" / "ASAP7" / "7nm_TT_160803.pm"


def test_build_nodes_nmos():
    from pycmg.sweep import build_nodes
    nodes = build_nodes(0.4, 0.5, 0.0, 0.9, "nmos")
    assert nodes == {"g": 0.4, "d": 0.5, "s": 0.0, "e": 0.0}


def test_build_nodes_pmos():
    from pycmg.sweep import build_nodes
    nodes = build_nodes(0.4, 0.5, 0.0, 0.9, "pmos")
    assert nodes["s"] == 0.9
    assert abs(nodes["g"] - 0.5) < 1e-12
    assert abs(nodes["d"] - 0.4) < 1e-12
    assert abs(nodes["e"] - 0.9) < 1e-12


def test_build_voltage_grid_bounds():
    from pycmg.sweep import build_voltage_grid
    vg, vd = build_voltage_grid(0.9, 0.35, vg_points=50, vd_points=50)
    assert vg[0] >= 0.0
    assert vg[-1] <= 0.9
    assert vd[0] >= 0.0
    assert vd[-1] <= 0.9


def test_build_voltage_grid_density():
    from pycmg.sweep import build_voltage_grid
    vg, _ = build_voltage_grid(0.9, 0.35, vg_points=50, vd_points=10, dense_ratio=0.6)
    dense_count = np.sum((vg >= 0.215) & (vg <= 0.485))
    sparse_count = len(vg) - dense_count
    assert dense_count > sparse_count


def test_build_voltage_grid_no_duplicates():
    from pycmg.sweep import build_voltage_grid
    vg, vd = build_voltage_grid(0.75, 0.3, vg_points=50, vd_points=50)
    assert len(vg) == len(np.unique(vg))
    assert len(vd) == len(np.unique(vd))


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="OSDI not built")
def test_find_threshold_nmos():
    from pycmg import Model, Instance
    from pycmg.sweep import find_threshold
    model = Model(str(OSDI_PATH), str(ASAP7_MODELCARD), "nmos_rvt")
    inst = Instance(model, params={"L": 21e-9, "TFIN": 6.5e-9, "NFIN": 1.0})
    vth = find_threshold(inst, vdd=0.9, device_type="nmos")
    assert 0.1 < vth < 0.7


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="OSDI not built")
def test_find_threshold_pmos():
    from pycmg import Model, Instance
    from pycmg.sweep import find_threshold
    model = Model(str(OSDI_PATH), str(ASAP7_MODELCARD), "pmos_rvt")
    inst = Instance(model, params={"L": 21e-9, "TFIN": 6.5e-9, "NFIN": 1.0})
    vth = find_threshold(inst, vdd=0.9, device_type="pmos")
    assert 0.1 < vth < 0.7


def test_resolve_devices_all():
    from pycmg.tech import TECH_REGISTRY
    from pycmg.sweep import resolve_devices
    devices = resolve_devices(None, "ASAP7", TECH_REGISTRY["ASAP7"])
    assert "nmos_rvt" in devices
    assert len(devices) >= 8


def test_resolve_devices_filter():
    from pycmg.tech import TECH_REGISTRY
    from pycmg.sweep import resolve_devices
    filt = {"ASAP7": ["nmos_rvt", "pmos_rvt"]}
    devices = resolve_devices(filt, "ASAP7", TECH_REGISTRY["ASAP7"])
    assert devices == ["nmos_rvt", "pmos_rvt"]


def test_resolve_devices_glob():
    from pycmg.tech import TECH_REGISTRY
    from pycmg.sweep import resolve_devices
    filt = {"ASAP7": ["nmos_*"]}
    devices = resolve_devices(filt, "ASAP7", TECH_REGISTRY["ASAP7"])
    assert all(d.startswith("nmos_") for d in devices)
    assert len(devices) >= 4


def test_resolve_devices_missing_skipped():
    from pycmg.tech import TECH_REGISTRY
    from pycmg.sweep import resolve_devices
    filt = {"TSMC7": ["nmos_rvt"]}
    devices = resolve_devices(filt, "TSMC7", TECH_REGISTRY["TSMC7"])
    assert devices == []


def test_sweep_config_defaults():
    from pycmg.sweep import SweepConfig
    config = SweepConfig(techs=["ASAP7"])
    assert config.sweep_geometry is True
    assert len(config.temperatures) == 5
    assert config.process_vars is None


def test_build_all_columns_no_process():
    from pycmg.sweep import build_all_columns
    cols = build_all_columns([])
    assert len(cols) == 28
    assert cols[0] == "tech"
    assert cols[-1] == "cdd"


def test_build_all_columns_with_process():
    from pycmg.sweep import build_all_columns
    cols = build_all_columns(["eot", "toxp"])
    assert len(cols) == 30
    assert "eot" in cols
    assert "toxp" in cols
    # Process vars should be between geometry and voltage
    eot_idx = cols.index("eot")
    vg_idx = cols.index("Vg")
    temp_idx = cols.index("temp_K")
    assert temp_idx < eot_idx < vg_idx


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="OSDI not built")
def test_sweep_dc_smoke():
    """Minimal sweep: 1 tech, 1 device, no geometry sweep, 1 temp, 5x5 grid."""
    from pycmg.sweep import SweepConfig, sweep_dc, build_all_columns
    config = SweepConfig(
        techs=["ASAP7"],
        devices={"ASAP7": ["nmos_rvt"]},
        sweep_geometry=False,
        temperatures=[300.15],
        vg_points=5,
        vd_points=5,
    )
    result = sweep_dc(str(OSDI_PATH), config, verbose=0)
    expected_cols = build_all_columns([])
    assert result.columns == expected_cols
    assert len(result.data) == 25  # 5 Vg x 5 Vd
    assert result.data[0][0] == "ASAP7"
    assert result.data[0][1] == "nmos_rvt"


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="OSDI not built")
def test_sweep_dc_pmos_voltages():
    """PMOS should have Vs=Vdd."""
    from pycmg.sweep import SweepConfig, sweep_dc
    config = SweepConfig(
        techs=["ASAP7"],
        devices={"ASAP7": ["pmos_rvt"]},
        sweep_geometry=False,
        temperatures=[300.15],
        vg_points=3,
        vd_points=3,
    )
    result = sweep_dc(str(OSDI_PATH), config, verbose=0)
    vs_idx = result.columns.index("Vs")
    for row in result.data:
        assert abs(row[vs_idx] - 0.9) < 1e-12


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="OSDI not built")
def test_sweep_dc_with_process_vars():
    """Process variation should multiply config count."""
    from pycmg.sweep import SweepConfig, sweep_dc
    config = SweepConfig(
        techs=["ASAP7"],
        devices={"ASAP7": ["nmos_rvt"]},
        sweep_geometry=False,
        temperatures=[300.15],
        vg_points=3,
        vd_points=3,
        process_vars={"eot": [0.9e-9, 1.1e-9]},
    )
    result = sweep_dc(str(OSDI_PATH), config, verbose=0)
    assert "eot" in result.columns
    assert len(result.data) == 3 * 3 * 2  # 3Vg x 3Vd x 2 process combos


def test_to_csv_split_by_tech(tmp_path):
    from pycmg.sweep import SweepResult, to_csv, build_all_columns
    cols = build_all_columns([])
    result = SweepResult(
        columns=cols,
        data=[
            ["ASAP7", "nmos_rvt"] + [0.0] * (len(cols) - 2),
            ["TSMC7", "nmos_svt"] + [0.0] * (len(cols) - 2),
        ],
        metadata={},
    )
    paths = to_csv(result, str(tmp_path), split_by="tech")
    assert len(paths) == 2
    assert any("ASAP7_dc.csv" in p for p in paths)
    assert any("TSMC7_dc.csv" in p for p in paths)
    # Verify CSV content
    import csv
    with open(paths[0]) as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header[0] == "tech"
        assert len(header) == len(cols)


def test_to_csv_split_by_none(tmp_path):
    from pycmg.sweep import SweepResult, to_csv, build_all_columns
    cols = build_all_columns([])
    result = SweepResult(
        columns=cols,
        data=[["ASAP7", "nmos_rvt"] + [0.0] * (len(cols) - 2)],
        metadata={},
    )
    paths = to_csv(result, str(tmp_path), split_by="none")
    assert len(paths) == 1
    assert "training_data_dc.csv" in paths[0]


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="OSDI not built")
def test_generate_dataset_smoke(tmp_path):
    from pycmg.sweep import generate_dataset
    paths = generate_dataset(
        osdi_path=str(OSDI_PATH),
        techs=["ASAP7"],
        devices={"ASAP7": ["nmos_rvt"]},
        output_dir=str(tmp_path),
        sweep_geometry=False,
        temperatures=[300.15],
        vg_points=3,
        vd_points=3,
        verbose=0,
    )
    assert len(paths) == 1
    assert Path(paths[0]).exists()
    assert Path(paths[0]).stat().st_size > 0


# ---------------------------------------------------------------------------
# Extended voltage range (voltage_scale) tests
# ---------------------------------------------------------------------------


def test_build_voltage_grid_extended_bounds():
    from pycmg.sweep import build_voltage_grid
    vdd = 0.9
    vg, vd = build_voltage_grid(vdd, 0.35, vg_points=50, vd_points=50,
                                 voltage_scale=2.0)
    assert vg[0] >= 0.0
    assert vg[-1] == pytest.approx(2.0 * vdd, abs=1e-12)
    assert vd[0] >= 0.0
    assert vd[-1] == pytest.approx(2.0 * vdd, abs=1e-12)


def test_build_voltage_grid_extended_dense_region():
    """Dense region width should stay at ~0.3*vdd regardless of voltage_scale."""
    from pycmg.sweep import build_voltage_grid
    vdd = 0.9
    vth = 0.35

    vg_1x, _ = build_voltage_grid(vdd, vth, vg_points=50, vd_points=10,
                                    voltage_scale=1.0, dense_ratio=0.6)
    vg_2x, _ = build_voltage_grid(vdd, vth, vg_points=50, vd_points=10,
                                    voltage_scale=2.0, dense_ratio=0.6)

    # Dense region: Vth +/- 0.15*vdd
    dense_lo = max(0.0, vth - 0.15 * vdd)
    dense_hi = vth + 0.15 * vdd

    dense_1x = int(np.sum((vg_1x >= dense_lo) & (vg_1x <= dense_hi)))
    dense_2x = int(np.sum((vg_2x >= dense_lo) & (vg_2x <= dense_hi)))

    # Dense point count should be similar (small difference from deduplication
    # is acceptable since sparse grid has different step size with 2x range)
    assert abs(dense_1x - dense_2x) <= 5


def test_build_voltage_grid_scale_default_unchanged():
    """voltage_scale=1.0 should produce identical results to no argument."""
    from pycmg.sweep import build_voltage_grid
    vg_a, vd_a = build_voltage_grid(0.75, 0.3, vg_points=40, vd_points=40)
    vg_b, vd_b = build_voltage_grid(0.75, 0.3, vg_points=40, vd_points=40,
                                     voltage_scale=1.0)
    np.testing.assert_array_equal(vg_a, vg_b)
    np.testing.assert_array_equal(vd_a, vd_b)


def test_sweep_config_voltage_scale_default():
    from pycmg.sweep import SweepConfig
    config = SweepConfig(techs=["ASAP7"])
    assert config.voltage_scale == 1.0


@pytest.mark.skipif(not OSDI_PATH.exists(), reason="OSDI not built")
def test_sweep_dc_extended_range():
    """With voltage_scale=2.0, Vg and Vd should exceed nominal vdd."""
    from pycmg.sweep import SweepConfig, sweep_dc
    config = SweepConfig(
        techs=["ASAP7"],
        devices={"ASAP7": ["nmos_rvt"]},
        sweep_geometry=False,
        temperatures=[300.15],
        vg_points=5,
        vd_points=5,
        voltage_scale=2.0,
    )
    result = sweep_dc(str(OSDI_PATH), config, verbose=0)
    vg_idx = result.columns.index("Vg")
    vd_idx = result.columns.index("Vd")
    max_vg = max(row[vg_idx] for row in result.data)
    max_vd = max(row[vd_idx] for row in result.data)
    # ASAP7 vdd=0.9, so max should be near 1.8
    assert max_vg > 0.9
    assert max_vd > 0.9
    assert max_vg == pytest.approx(1.8, abs=0.01)
    assert max_vd == pytest.approx(1.8, abs=0.01)


# ---------------------------------------------------------------------------
# PDK geometry combo tests
# ---------------------------------------------------------------------------


def test_scan_pdk_geometry_combos_tsmc7():
    """TSMC7 pch_lvt_mac should return combos for all (L, NFIN) bin boundaries."""
    from pycmg.parser import scan_pdk_geometry_combos
    pdk = str(ROOT / "modelcards" / "TSMC7" / "cln7_1d8_sp_v1d2_2p2.l")
    combos = scan_pdk_geometry_combos(pdk, "pch_lvt_mac")
    # 6 L bins x 7 unique NFIN boundaries = 42
    assert len(combos) == 42
    # All combos should be sorted
    assert combos == sorted(combos)
    # Smallest L should be 8nm, smallest NFIN should be 1
    assert combos[0][0] == pytest.approx(8e-9)
    assert combos[0][1] == pytest.approx(1.0)


def test_find_length_variant_nfin_aware():
    """NFIN-aware variant selection should pick correct NFIN group."""
    from pycmg.parser import _find_length_variant
    pdk = str(ROOT / "modelcards" / "TSMC7" / "cln7_1d8_sp_v1d2_2p2.l")
    # NFIN=1 should select variant 35 (nfinmin=1, nfinmax=2)
    v1 = _find_length_variant(pdk, "pch_lvt_mac", 16e-9, NFIN=1)
    # NFIN=20 should select variant 5 (nfinmin=20, nfinmax=24.888)
    v20 = _find_length_variant(pdk, "pch_lvt_mac", 16e-9, NFIN=20)
    assert v1 != v20
    assert v1 == 35  # nfinmin=1
    assert v20 == 5   # nfinmin=20
