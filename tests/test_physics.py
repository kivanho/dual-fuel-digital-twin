import pytest

from app.calculate_massflows import LHV_CH4, LHV_DIESEL, calculate_fuel_mass_flows
from app.engine_map import ENGINE_MAP_COLUMNS
from app.knnr_model import train_knnr_model
from app.regulator import FlowLoop
from app.simulation_core import co2_in_fuel_gas_kg_h, line_current_a, shaft_torque_nm
from app.synthetic_data import generate_engine_map


def test_synthetic_engine_map_is_plausible():
    df = generate_engine_map()
    assert list(df.columns) == ENGINE_MAP_COLUMNS
    assert df.notna().all().all()
    assert df["efficiency_electric"].between(5, 40).all()
    # Efficiency and exhaust temperature both rise with load
    assert df["efficiency_electric"].corr(df["power_output"]) > 0.7
    assert df["exhaust_temp"].corr(df["power_output"]) > 0.9


def test_synthetic_engine_map_is_reproducible():
    assert generate_engine_map().equals(generate_engine_map())


def test_synthetic_engine_map_covers_the_design_corners():
    df = generate_engine_map()
    assert df["power_output"].min() == 2.0 and df["power_output"].max() == 14.4
    assert df["calculated_ch4_share_percent"].min() == 55.0 and df["calculated_ch4_share_percent"].max() == 100.0
    assert df["des_percent"].min() == 10.0 and df["des_percent"].max() == 60.0
    assert len(df) == 400


def test_mass_flows_close_energy_balance():
    flows = calculate_fuel_mass_flows(10.0, 0.30, 20.0)
    fuel_power_kw = (flows["diesel_mass_flow_kg_h"] * LHV_DIESEL
                     + flows["ch4_mass_flow_kg_h"] * LHV_CH4) / 3.6
    assert fuel_power_kw * 0.30 == pytest.approx(10.0, rel=1e-2)


def test_torque_matches_datasheet_bmep():
    # Standby: 649 kPa BMEP × 2.2 L / 4π ≈ 113.6 Nm at 14.4 kWe
    assert shaft_torque_nm(14.4) == pytest.approx(113.6, rel=0.02)


def test_line_current_at_rated_power():
    # 18 kVA at 400 V three-phase: 18000 / (√3 · 400) ≈ 26.0 A
    assert line_current_a(14.4) == pytest.approx(26.0, abs=0.1)


def test_synthetic_efficiency_matches_diesel_datasheet():
    # Continuous rating 13.2 kWe at 4.4 l/h diesel (0.85 kg/l) -> 29.8 %
    df = generate_engine_map(n_points=4000)
    near_anchor = df[df["power_output"].between(12.7, 13.7)
                     & df["calculated_ch4_share_percent"].between(95, 100)
                     & df["des_percent"].between(25, 35)]
    assert near_anchor["efficiency_electric"].mean() == pytest.approx(29.8, abs=1.0)


def test_knnr_model_fits_engine_map():
    _, metrics = train_knnr_model(generate_engine_map())
    assert metrics["r2"] > 0.8


def test_flow_loop_settles_without_overshoot():
    loop = FlowLoop()
    flows = [loop.step(3.0, 0.1) for _ in range(100)]  # 10 s at 0.1 s steps
    assert max(flows) <= 3.0 + 1e-9
    settled_after = next(i for i, f in enumerate(flows) if abs(f - 3.0) <= 0.02 * 3.0) * 0.1
    assert settled_after < 5.0


def test_flow_loop_anti_windup():
    # A target above the valve limit must not wind up the integrator
    loop = FlowLoop(max_flow=2.0)
    for _ in range(300):
        loop.step(5.0, 0.1)
    assert loop.flow == pytest.approx(2.0, abs=1e-3)
    flows = [loop.step(1.0, 0.1) for _ in range(100)]
    assert flows[-1] == pytest.approx(1.0, abs=0.02)


def test_co2_from_ch4_volume_share():
    # 80 vol-% CH4: n_CO2 / n_CH4 = 0.25, so m_CO2 = 0.25 · 44.01 / 16.04 · m_CH4
    assert co2_in_fuel_gas_kg_h(2.0, 80.0) == pytest.approx(2.0 * 0.25 * 44.01 / 16.04)
