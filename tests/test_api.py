import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.engine_specs import RATED_POWER_KW
from app.main import app, create_app
from app.simulation_service import INITIAL_SETPOINT

POINT_12 = {"power_output": 12.0, "diesel_energy_share": 20.0, "methane_percentage": 80.0}


@pytest.fixture
def client():
    """The module-level app: bundle from MODEL_BUNDLE_DIR (set in conftest), background loop on."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def quiet_client(trained_bundle_dir):
    """An app without the background loop, so tests control every tick."""
    with TestClient(create_app(bundle_dir=trained_bundle_dir, background_loop=False)) as c:
        yield c


def sim_of(client):
    return client.app.state.sim


def test_root(client):
    assert client.get("/").status_code == 200


def test_health_and_ready(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["ready"] is True
    assert h["model_version"] == "test-bundle"
    assert h["simulation"] in ("running", "stopped")
    assert h["supported_workers"] == 1
    r = client.get("/ready")
    assert r.status_code == 200 and r.json() == {"ready": True, "model_version": "test-bundle"}


def test_not_ready_before_startup(trained_bundle_dir):
    # Without entering the lifespan (no `with`), the bundle is not loaded
    c = TestClient(create_app(bundle_dir=trained_bundle_dir, background_loop=False))
    assert c.get("/ready").status_code == 503
    assert c.get("/live").status_code == 503
    assert c.get("/health").json()["ready"] is False


def test_startup_fails_without_bundle(tmp_path):
    with pytest.raises(RuntimeError, match="python -m app.train"):
        with TestClient(create_app(bundle_dir=str(tmp_path / "missing"), background_loop=False)):
            pass


def test_model_endpoint_exposes_domain_and_features(client):
    m = client.get("/model").json()
    assert m["model_version"] == "test-bundle"
    assert m["models"]["efficiency"]["features"] == ["power_output", "calculated_ch4_share_percent"]
    assert m["domain"]["training_range"]["power_output"] == [2.0, 14.4]
    assert m["domain"]["units"]["power_output"] == "kW"


def test_update_inputs_and_live_snapshot(quiet_client):
    response = quiet_client.post("/update_inputs", json=POINT_12)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "Inputs updated." and body["warnings"] == [] and body["input_support"]["supported"]

    for _ in range(12):
        sim_of(quiet_client).step()
    state = quiet_client.get("/live").json()
    assert state["input"] == {"power": 12.0, "ch4_percent": 80.0, "des_percent": 20.0}
    assert state["status"] == "running" and state["model_version"] == "test-bundle"
    assert state["actual"]["power"] == pytest.approx(12.0, rel=0.005)
    assert state["actual"]["diesel_energy_share"] == pytest.approx(20.0, abs=0.2)
    assert state["actual"]["within_training_domain"] is True
    assert 0 < state["target"]["efficiency"] < 0.4


@pytest.mark.parametrize("payload, where", [
    ({"power_output": 1.0, "diesel_energy_share": 20, "methane_percentage": 80}, "domain"),     # below the map
    ({"power_output": 15.0, "diesel_energy_share": 20, "methane_percentage": 80}, "equipment"), # above the rating
    ({"power_output": 8.0, "diesel_energy_share": 100, "methane_percentage": 80}, "domain"),    # outside the DES range
    ({"power_output": 8.0, "diesel_energy_share": 20, "methane_percentage": 10}, "domain"),     # outside the CH4 range
    ({"power_output": 0.0, "diesel_energy_share": 20, "methane_percentage": 80}, "equipment"),
])
def test_update_inputs_rejects_out_of_range(quiet_client, payload, where):
    before = quiet_client.get("/live").json()["input"]
    r = quiet_client.post("/update_inputs", json=payload)
    assert r.status_code == 422
    if where == "domain":
        detail = r.json()["detail"]
        assert detail["message"] == "operating point outside the supported domain"
        assert any("training range" in v for v in detail["violations"])
        assert detail["support"]["within_equipment_limits"] is True
    assert quiet_client.get("/live").json()["input"] == before, "a rejected point leaves the setpoint unchanged"


def test_sparse_point_inside_box_rejected_with_explanation(quiet_client):
    r = quiet_client.post("/update_inputs", json={"power_output": 14.4, "diesel_energy_share": 60, "methane_percentage": 100})
    assert r.status_code == 422
    assert any("sparse region" in v for v in r.json()["detail"]["violations"])


def test_warn_policy_via_env(trained_bundle_dir, monkeypatch):
    monkeypatch.setenv("UNSUPPORTED_POINT_POLICY", "warn")
    with TestClient(create_app(bundle_dir=trained_bundle_dir, background_loop=False)) as c:
        r = c.post("/update_inputs", json={"power_output": 1.0, "diesel_energy_share": 20, "methane_percentage": 80})
        assert r.status_code == 200
        assert r.json()["warnings"] and "training range" in r.json()["warnings"][0]
        assert c.get("/live").json()["warnings"] == r.json()["warnings"]
        assert c.get("/health").json()["unsupported_point_policy"] == "warn"


def test_live_state_streams_json(quiet_client):
    sim_of(quiet_client).step()
    with quiet_client.stream("GET", "/live_state", params={"max_events": 2}) as response:
        assert response.status_code == 200
        events = [line[len("data: "):] for line in response.iter_lines() if line.startswith("data: ")]
    assert len(events) == 2
    state = json.loads(events[0])
    assert {"input", "target", "actual", "status", "model_version"} <= set(state)


def test_reset_endpoint_restarts_at_initial_condition(quiet_client):
    sim = sim_of(quiet_client)
    quiet_client.post("/update_inputs", json=POINT_12)
    for _ in range(10):
        sim.step()
    r = quiet_client.post("/reset")
    assert r.status_code == 200
    assert r.json()["state"]["input"] == INITIAL_SETPOINT
    assert quiet_client.get("/live").json()["actual"]["power"] is None
    powers = []
    for _ in range(12):
        sim.step()
        powers.append(quiet_client.get("/live").json()["actual"]["power"])
    assert powers[0] < 5.0 and powers[-1] == pytest.approx(5.0, rel=0.03)


def test_stop_and_start_endpoints(quiet_client):
    sim = sim_of(quiet_client)
    quiet_client.post("/update_inputs", json=POINT_12)
    for _ in range(12):
        sim.step()
    assert quiet_client.post("/stop").json() == {"status": "Simulation stopped.", "simulation": "stopped"}
    for _ in range(25):
        sim.step()
    s = quiet_client.get("/live").json()
    assert s["status"] == "stopped" and s["actual"]["power"] < 0.05 and s["input"]["power"] == 12.0
    assert quiet_client.get("/health").json()["simulation"] == "stopped"
    assert quiet_client.post("/start").json()["simulation"] == "running"
    for _ in range(15):
        sim.step()
    assert quiet_client.get("/live").json()["actual"]["power"] == pytest.approx(12.0, rel=0.01)


def test_background_loop_steps_the_simulation(client):
    t0 = client.get("/live").json()["time_s"]
    deadline = time.time() + 5
    while time.time() < deadline and client.get("/live").json()["time_s"] <= t0:
        time.sleep(0.1)
    assert client.get("/live").json()["time_s"] > t0


def test_background_loop_is_cancelled_on_shutdown(trained_bundle_dir):
    with TestClient(create_app(bundle_dir=trained_bundle_dir)) as c:
        sim = c.app.state.sim
        c.get("/live")
    t_end = sim.time
    time.sleep(1.5)
    assert sim.time == t_end, "no stepping after shutdown"
    assert c.app.state.ready is False


def test_concurrent_api_clients_see_consistent_snapshots(client):
    """Several clients post set points and read snapshots while the background loop steps."""
    errors = []
    points = [POINT_12, {"power_output": 8.0, "diesel_energy_share": 30.0, "methane_percentage": 90.0}]

    def worker(i):
        try:
            for n in range(25):
                r = client.post("/update_inputs", json=points[(n + i) % 2])
                assert r.status_code == 200
                s = client.get("/live").json()
                assert s["input"]["power"] in (12.0, 8.0)
                expected = s["input"]["power"] / s["target"]["efficiency"] * 3.6 * s["input"]["des_percent"] / 100 / 42.7
                assert s["target"]["diesel_mass_flow_kg_h"] == pytest.approx(expected, rel=0.01)
                assert s["actual"]["power"] is None or 0 <= s["actual"]["power"] <= RATED_POWER_KW
        except AssertionError as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors[:3]
