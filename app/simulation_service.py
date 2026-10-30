"""
The simulated engine and its state.

Ownership model
---------------
There is exactly ONE simulated engine per service process, created at startup
from a trained model bundle and shared by every client. Clients are not
isolated from each other on purpose: the twin stands for one physical genset,
and any client may change its operating point, stop or reset it, like several
operators at one control panel. Run the API with a single worker process; with
several workers each would hold its own independent engine and clients would
see different states depending on which worker answered.

Every public method takes the service lock, so concurrent requests and the
background stepping task cannot interleave half-applied updates, and
`get_current_state` returns a snapshot built under the lock whose `input`,
`target` and `actual` blocks are from one consistent state.

Simulation controls (not controls for a physical engine)
---------------------------------------------------------
- start:  running; the flow loops move the fuel flows towards the targets for
          the requested operating point.
- stop:   both fuel valves are commanded closed; flows and power wind down to
          zero and stay there until `start`. The requested operating point is
          kept, and `update_inputs` while stopped only changes that setpoint.
- reset:  the simulation is restarted at the documented initial condition:
          t = 0, valves closed and controllers cleared, the operating point
          back to INITIAL_SETPOINT, running. The next ticks therefore ramp
          towards 5 kW, not towards whatever was requested before.

Operating-point policy
----------------------
`update_inputs` checks the point against the bundle's supported domain
(equipment limits, training range and support distance; see app/domain.py).
With policy "reject" an unsupported point raises UnsupportedOperatingPoint and
the setpoint is unchanged. With policy "warn" the point is accepted and every
snapshot carries the violations in `warnings` until a supported point is set.
"""
import copy
import threading

from app.bundle import ModelBundle, load_bundle
from app.calculate_massflows import LHV_CH4, LHV_DIESEL
from app.engine_specs import RATED_POWER_KW
from app.regulator import FlowLoop
from app.simulation_core import advance_plant, compute_targets

# Valve range: enough fuel for the rated power at 25 % electrical efficiency
MAX_FUEL_MJ_H = RATED_POWER_KW / 0.25 * 3.6
MAX_FLOW_KG_H = {"diesel": MAX_FUEL_MJ_H / LHV_DIESEL, "ch4": MAX_FUEL_MJ_H / LHV_CH4}

INITIAL_SETPOINT = {"power": 5.0, "des_percent": 10.0, "ch4_percent": 85.0}
POLICIES = ("reject", "warn")

# API payload key -> internal key
INPUT_KEYS = {"power_output": "power", "diesel_energy_share": "des_percent", "methane_percentage": "ch4_percent"}
DOMAIN_KEYS = {"power": "power_output", "ch4_percent": "calculated_ch4_share_percent", "des_percent": "des_percent"}


class UnsupportedOperatingPoint(ValueError):
    def __init__(self, check):
        self.check = check
        super().__init__("; ".join(check.violations) or "operating point outside the supported domain")


def _round(value, digits):
    return None if value is None else round(float(value), digits)


class SimulationService:
    def __init__(self, bundle: ModelBundle, dt: float = 1.0, unsupported_policy: str = "reject"):
        if unsupported_policy not in POLICIES:
            raise ValueError(f"unsupported_policy must be one of {POLICIES}")
        self.bundle = bundle
        self.model = bundle.efficiency_model
        self.exhaust_model = bundle.exhaust_model
        self.domain = bundle.domain
        self.model_version = bundle.version
        self.dt = dt  # seconds per step
        self.unsupported_policy = unsupported_policy
        self._lock = threading.Lock()
        self._initialise()

    @classmethod
    def from_bundle_dir(cls, bundle_dir=None, **kwargs):
        return cls(load_bundle(bundle_dir), **kwargs)

    # ------------------------------------------------------------------ state
    def _initialise(self):
        """Documented initial condition: t = 0, valves closed, default setpoint, running."""
        self.time = 0.0
        self.running = True
        self.loops = {name: FlowLoop(max_flow=limit) for name, limit in MAX_FLOW_KG_H.items()}
        self.last_results = {}
        self._set_input(dict(INITIAL_SETPOINT))

    def _set_input(self, input_data):
        self.input_data = input_data
        self.input_check = self.domain.check({DOMAIN_KEYS[k]: v for k, v in input_data.items()})
        self.targets = compute_targets(input_data, self.model, self.exhaust_model)

    @property
    def status(self):
        return "running" if self.running else "stopped"

    # ------------------------------------------------------------------ operations (all under the lock)
    def step(self):
        """Advance the simulated engine by dt seconds."""
        with self._lock:
            self.time += self.dt
            self.last_results = advance_plant(self.input_data, self.targets, self.exhaust_model, self.loops,
                                              self.dt, running=self.running)

    def update_inputs(self, inputs_dict):
        """
        Set a new operating point. Returns the SupportCheck. Raises
        UnsupportedOperatingPoint under the "reject" policy. Does not change
        the running/stopped state.
        """
        input_data = {INPUT_KEYS[k]: float(inputs_dict[k]) for k in INPUT_KEYS}
        check = self.domain.check({DOMAIN_KEYS[k]: v for k, v in input_data.items()})
        if not check.supported and self.unsupported_policy == "reject":
            raise UnsupportedOperatingPoint(check)
        with self._lock:
            self._set_input(input_data)
        return check

    def stop(self):
        with self._lock:
            self.running = False

    def start(self):
        with self._lock:
            self.running = True

    def reset(self):
        with self._lock:
            self._initialise()

    # ------------------------------------------------------------------ snapshot
    def get_current_state(self):
        """A consistent copy of the state (built under the lock; safe to hand to any client)."""
        with self._lock:
            r = self.last_results
            t = self.targets
            running = self.running
            warnings = [] if self.input_check.supported else list(self.input_check.violations)
            actual_in_domain = None
            if r:
                actual_in_domain = bool(self.domain.contains([r["power"], self.input_data["ch4_percent"], r["des"]]))
            state = {
                "time_s": self.time,
                "status": "running" if running else "stopped",
                "model_version": self.model_version,
                "input": dict(self.input_data),
                "input_support": self.input_check.to_dict(),
                "warnings": warnings,
                "target": {
                    "efficiency": _round(t["efficiency_target"], 3),
                    "diesel_mass_flow_kg_h": _round(t["diesel_target"] if running else 0.0, 3),
                    "ch4_mass_flow_kg_h": _round(t["ch4_target"] if running else 0.0, 3),
                    "exhaust_temp": _round(t["temp_target"], 1),
                },
                "actual": {
                    "power": _round(r.get("power"), 2),
                    "diesel_mass_flow_kg_h": _round(r.get("diesel"), 3),
                    "ch4_mass_flow_kg_h": _round(r.get("ch4"), 3),
                    "diesel_energy_share": _round(r.get("des"), 1),
                    "co2_in_fuel_gas_kg_h": _round(r.get("co2_in_fuel_gas"), 3),
                    "exhaust_temp": _round(r.get("temp"), 1),
                    "current": _round(r.get("current"), 1),
                    "torque": _round(r.get("torque"), 1),
                    "within_training_domain": actual_in_domain,
                },
            }
        return copy.deepcopy(state)
