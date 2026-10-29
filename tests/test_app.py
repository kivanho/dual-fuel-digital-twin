import json
import math

import pytest
from fastapi.testclient import TestClient

from app.calculate_massflows import LHV_CH4, LHV_DIESEL, calculate_fuel_mass_flows
from app.main import app, sim
from app.regulator import apply_regulation
from app.simulation_core import line_current_a, shaft_torque_nm
from app.synthetic_data import ENGINE_MAP_COLUMNS, generate_engine_map


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


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
    assert sim.knnr_metrics["r2"] > 0.8


def test_pid_converges_to_target():
    controller_type = {"x": "pid"}
    state, vel, integral, prev = {"x": 0.0}, {"x": 0.0}, {"x": 0.0}, {"x": 0.0}
    gains = dict(T={"x": 2.0}, zeta={"x": 0.0}, omega_0={"x": 0.05},
                 kp={"x": 1.8}, ki={"x": 0.1}, kd={"x": 0.08})
    for _ in range(200):
        state, vel, integral, prev = apply_regulation(
            state, vel, integral, prev, {"x": 3.0}, 1, controller_type,
            gains["T"], gains["zeta"], gains["omega_0"], gains["kp"], gains["ki"], gains["kd"])
    assert state["x"] == pytest.approx(3.0, abs=0.05)


def test_root(client):
    assert client.get("/").status_code == 200


def test_update_inputs_and_live_snapshot(client):
    payload = {"power_output": 12.0, "diesel_energy_share": 20.0, "methane_percentage": 80.0}
    response = client.post("/update_inputs", json=payload)
    assert response.status_code == 200

    for _ in range(100):
        sim.step()
    state = client.get("/live").json()
    assert state["input"] == {"power": 12.0, "ch4_percent": 80.0, "des_percent": 20.0}
    assert state["actual"]["power"] == pytest.approx(12.0, rel=0.02)
    assert 0 < state["target"]["efficiency"] < 0.4


def test_update_inputs_rejects_out_of_range(client):
    response = client.post("/update_inputs", json={"power_output": 5, "diesel_energy_share": 10,
                                                   "methane_percentage": 0})
    assert response.status_code == 422


def test_live_state_streams_json(client):
    sim.step()
    with client.stream("GET", "/live_state", params={"max_events": 2}) as response:
        assert response.status_code == 200
        events = [line[len("data: "):] for line in response.iter_lines() if line.startswith("data: ")]
    assert len(events) == 2
    state = json.loads(events[0])
    assert {"input", "target", "actual"} <= set(state)


def test_reset(client):
    sim.step()
    assert client.post("/reset").status_code == 200
    assert client.get("/live").json()["actual"]["power"] is None


def test_full_diesel_share_does_not_produce_nan():
    sim.update_inputs({"power_output": 8.0, "diesel_energy_share": 100.0, "methane_percentage": 85.0})
    sim.reset()
    for _ in range(5):
        sim.step()
    assert all(v is None or math.isfinite(v) for v in sim.get_current_state()["actual"].values())
