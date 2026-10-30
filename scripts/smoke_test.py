"""
End-to-end smoke test against a trained bundle, without a network port.

    python -m app.train --out models/synthetic
    python -m scripts.smoke_test [--bundle models/synthetic] [--steps 20]

Starts the app in-process (TestClient, so the lifespan loads the bundle and the
background loop runs), checks /ready, sets an operating point, steps the
simulation, and verifies that every number in the snapshot is finite and the
plant settled near the request. Floating-point errors are raised, not hidden.
Exit status 0 means pass.
"""
import argparse
import math
import sys
import warnings

import numpy as np
from fastapi.testclient import TestClient

from app.main import create_app

POINT = {"power_output": 12.0, "diesel_energy_share": 20.0, "methane_percentage": 80.0}


def assert_finite(obj, path="state"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert_finite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            assert_finite(v, f"{path}[{i}]")
    elif isinstance(obj, float) and not math.isfinite(obj):
        raise AssertionError(f"non-finite value at {path}: {obj}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", default=None, help="bundle directory (default: MODEL_BUNDLE_DIR or models/synthetic)")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args(argv)

    np.seterr(all="raise")
    warnings.simplefilter("error", RuntimeWarning)

    with TestClient(create_app(bundle_dir=args.bundle, background_loop=False)) as client:
        ready = client.get("/ready")
        assert ready.status_code == 200, ready.text
        health = client.get("/health").json()
        print(f"ready: model {health['model_version']} from {health['bundle_path']}")

        r = client.post("/update_inputs", json=POINT)
        assert r.status_code == 200, r.text
        sim = client.app.state.sim
        for _ in range(args.steps):
            sim.step()
            state = client.get("/live").json()
            assert_finite(state)
        power = state["actual"]["power"]
        assert abs(power - POINT["power_output"]) < 0.05 * POINT["power_output"], f"power {power} did not settle"
        assert state["actual"]["within_training_domain"] is True
        assert 0 < state["target"]["efficiency"] < 0.5
        print(f"after {args.steps} steps: power {power} kW, exhaust {state['actual']['exhaust_temp']} degC, "
              f"diesel {state['actual']['diesel_mass_flow_kg_h']} kg/h, CH4 {state['actual']['ch4_mass_flow_kg_h']} kg/h")

        assert client.post("/stop").status_code == 200
        for _ in range(args.steps):
            sim.step()
        stopped = client.get("/live").json()
        assert_finite(stopped)
        assert stopped["status"] == "stopped" and stopped["actual"]["power"] < 0.1, stopped["actual"]
        assert client.post("/reset").status_code == 200
        reset_state = client.get("/live").json()
        assert reset_state["time_s"] == 0.0 and reset_state["status"] == "running"
        print("stop winds down to", stopped["actual"]["power"], "kW; reset returns to t = 0. Smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
