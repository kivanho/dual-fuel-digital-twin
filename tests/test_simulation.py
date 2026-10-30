import math
import threading

import pytest

from app.engine_specs import RATED_POWER_KW
from app.simulation_service import INITIAL_SETPOINT, SimulationService, UnsupportedOperatingPoint

POINT_12 = {"power_output": 12.0, "diesel_energy_share": 20.0, "methane_percentage": 80.0}
POINT_8 = {"power_output": 8.0, "diesel_energy_share": 30.0, "methane_percentage": 90.0}


@pytest.fixture
def sim(bundle):
    return SimulationService(bundle)


def run(sim, n):
    out = []
    for _ in range(n):
        sim.step()
        out.append(sim.get_current_state())
    return out


def finite_numbers(d):
    for k, v in d.items():
        if isinstance(v, dict):
            finite_numbers(v)
        elif isinstance(v, float):
            assert math.isfinite(v), k


# ------------------------------------------------------------------ initial condition and reset

def test_initial_condition(sim):
    s = sim.get_current_state()
    assert s["time_s"] == 0.0 and s["status"] == "running"
    assert s["input"] == INITIAL_SETPOINT
    assert s["actual"]["power"] is None and s["actual"]["within_training_domain"] is None
    assert s["model_version"] == sim.model_version
    assert s["warnings"] == [] and s["input_support"]["supported"]
    # targets are available before the first step and consistent with the input
    assert 0 < s["target"]["efficiency"] < 0.4 and s["target"]["diesel_mass_flow_kg_h"] > 0


def test_reset_returns_to_initial_setpoint_over_subsequent_ticks(sim):
    sim.update_inputs(POINT_12)
    run(sim, 15)
    assert sim.get_current_state()["actual"]["power"] == pytest.approx(12.0, rel=0.01)

    sim.reset()
    s = sim.get_current_state()
    assert s["time_s"] == 0.0 and s["actual"]["power"] is None
    assert s["input"] == INITIAL_SETPOINT, "reset restores the documented initial operating point"

    powers = [st["actual"]["power"] for st in run(sim, 15)]
    assert powers[0] < 5.0, "the first tick after reset starts from standstill, not from the old 12 kW request"
    assert all(p <= 5.0 + 1e-6 for p in powers)
    assert powers[-1] == pytest.approx(5.0, rel=0.02), "the engine ramps to the initial 5 kW setpoint"
    assert sim.get_current_state()["time_s"] == 15.0


def test_reset_also_clears_a_stopped_state(sim):
    sim.stop()
    sim.reset()
    assert sim.get_current_state()["status"] == "running"


# ------------------------------------------------------------------ stop / start

def test_stop_winds_down_and_stays_stopped(sim):
    sim.update_inputs(POINT_12)
    run(sim, 15)
    sim.stop()
    states = run(sim, 20)
    powers = [s["actual"]["power"] for s in states]
    assert all(s["status"] == "stopped" for s in states)
    assert powers == sorted(powers, reverse=True), "power decays monotonically after stop"
    assert powers[-1] < 0.05 and states[-1]["actual"]["diesel_mass_flow_kg_h"] < 0.01
    assert states[-1]["target"]["diesel_mass_flow_kg_h"] == 0.0 and states[-1]["target"]["ch4_mass_flow_kg_h"] == 0.0
    assert states[-1]["input"]["power"] == 12.0, "the requested operating point is kept while stopped"
    assert states[-1]["time_s"] == 35.0, "simulated time keeps running while stopped"
    # it does not restart by itself, even over many more ticks
    assert max(s["actual"]["power"] for s in run(sim, 30)) < 0.01


def test_update_inputs_while_stopped_does_not_restart(sim):
    sim.stop()
    run(sim, 5)
    sim.update_inputs(POINT_12)
    states = run(sim, 10)
    assert all(s["status"] == "stopped" and s["actual"]["power"] < 1e-6 for s in states)
    assert states[-1]["input"]["power"] == 12.0


def test_start_resumes_towards_the_kept_setpoint(sim):
    sim.update_inputs(POINT_12)
    sim.stop()
    run(sim, 10)
    sim.start()
    powers = [s["actual"]["power"] for s in run(sim, 15)]
    assert powers == sorted(powers)
    assert powers[-1] == pytest.approx(12.0, rel=0.01)


def test_power_never_exceeds_rating_after_step(sim):
    sim.update_inputs({"power_output": RATED_POWER_KW, "diesel_energy_share": 30.0, "methane_percentage": 90.0})
    powers = [s["actual"]["power"] for s in run(sim, 10)]
    assert max(powers) <= RATED_POWER_KW
    assert powers == sorted(powers)  # rises monotonically, no overshoot


# ------------------------------------------------------------------ operating-point policy

def test_reject_policy_leaves_setpoint_unchanged(sim):
    sim.update_inputs(POINT_12)
    with pytest.raises(UnsupportedOperatingPoint, match="outside the training range"):
        sim.update_inputs({"power_output": 1.0, "diesel_energy_share": 20.0, "methane_percentage": 80.0})
    assert sim.get_current_state()["input"]["power"] == 12.0


def test_warn_policy_accepts_and_flags(bundle):
    sim = SimulationService(bundle, unsupported_policy="warn")
    check = sim.update_inputs({"power_output": 1.0, "diesel_energy_share": 20.0, "methane_percentage": 80.0})
    assert not check.supported
    s = run(sim, 3)[-1]
    assert s["input"]["power"] == 1.0
    assert s["warnings"] and "training range" in s["warnings"][0]
    assert s["input_support"]["supported"] is False
    sim.update_inputs(POINT_12)
    assert sim.get_current_state()["warnings"] == []


def test_invalid_policy_rejected(bundle):
    with pytest.raises(ValueError):
        SimulationService(bundle, unsupported_policy="ignore")


def test_actual_state_flagged_outside_domain_during_transients(sim):
    sim.update_inputs(POINT_12)
    states = run(sim, 15)
    # At steady state the actual point coincides with the supported setpoint
    assert states[-1]["actual"]["within_training_domain"] is True
    # Winding down after stop: the plant state leaves the steady-state training domain (power -> 0 kW)
    sim.stop()
    states = run(sim, 20)
    assert states[-1]["actual"]["power"] < 1.0
    assert states[-1]["actual"]["within_training_domain"] is False
    assert states[-1]["actual"]["exhaust_temp"] is not None, "the model still extrapolates; the flag says so"


def test_targets_follow_inputs_immediately_and_consistently(sim):
    sim.update_inputs(POINT_12)
    s = sim.get_current_state()
    assert s["input"]["power"] == 12.0
    assert s["target"]["diesel_mass_flow_kg_h"] == pytest.approx(
        12.0 / s["target"]["efficiency"] * 3.6 * 0.2 / 42.7, rel=0.01)


# ------------------------------------------------------------------ snapshots and concurrency

def test_snapshot_is_a_copy(sim):
    run(sim, 2)
    s = sim.get_current_state()
    s["input"]["power"] = -1
    s["actual"]["power"] = -1
    assert sim.get_current_state()["input"]["power"] == INITIAL_SETPOINT["power"]
    assert sim.get_current_state()["actual"]["power"] > 0


def test_concurrent_updates_steps_and_snapshots_are_consistent(sim):
    """Threads hammer update_inputs, step and get_current_state; every snapshot must be self-consistent."""
    stop = threading.Event()
    errors = []
    points = [POINT_12, POINT_8]

    def updater(i):
        n = 0
        while not stop.is_set():
            sim.update_inputs(points[(n + i) % 2])
            n += 1

    def stepper():
        while not stop.is_set():
            sim.step()

    def reader():
        last_t = -1.0
        while not stop.is_set():
            s = sim.get_current_state()
            try:
                finite_numbers(s)
                assert s["time_s"] >= last_t
                last_t = s["time_s"]
                assert s["input"] in ({"power": 12.0, "des_percent": 20.0, "ch4_percent": 80.0},
                                      {"power": 8.0, "des_percent": 30.0, "ch4_percent": 90.0})
                # target block was computed from the same input as reported
                expected = s["input"]["power"] / s["target"]["efficiency"] * 3.6 * s["input"]["des_percent"] / 100 / 42.7
                assert s["target"]["diesel_mass_flow_kg_h"] == pytest.approx(expected, rel=0.01)
                assert s["actual"]["power"] is None or s["actual"]["power"] <= RATED_POWER_KW
            except AssertionError as e:
                errors.append(e)
                stop.set()

    threads = [threading.Thread(target=updater, args=(i,)) for i in range(2)]
    threads += [threading.Thread(target=stepper) for _ in range(2)]
    threads += [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    stop.wait(1.0)
    stop.set()
    for t in threads:
        t.join(timeout=5)
    assert not errors, errors[:3]
    assert sim.get_current_state()["time_s"] > 0


def test_concurrent_resets_do_not_corrupt_state(sim):
    stop = threading.Event()

    def resetter():
        while not stop.is_set():
            sim.reset()

    def stepper():
        while not stop.is_set():
            sim.step()

    threads = [threading.Thread(target=resetter), threading.Thread(target=stepper), threading.Thread(target=stepper)]
    for t in threads:
        t.start()
    stop.wait(0.5)
    stop.set()
    for t in threads:
        t.join(timeout=5)
    s = sim.get_current_state()
    finite_numbers(s)
    assert s["input"] == INITIAL_SETPOINT
    assert set(sim.loops) == {"diesel", "ch4"}
