from app.calculate_massflows import LHV_CH4, LHV_DIESEL
from app.engine_specs import RATED_POWER_KW
from app.exhaust_temp_model import train_exhaust_temp_model
from app.knnr_model import train_knnr_model
from app.regulator import FlowLoop
from app.simulation_core import run_simulation_step
from app.synthetic_data import load_engine_map

# Valve range: enough fuel for the rated power at 25 % electrical efficiency
MAX_FUEL_MJ_H = RATED_POWER_KW / 0.25 * 3.6
MAX_FLOW_KG_H = {"diesel": MAX_FUEL_MJ_H / LHV_DIESEL, "ch4": MAX_FUEL_MJ_H / LHV_CH4}


def _round(value, digits):
    return None if value is None else round(float(value), digits)


class SimulationService:
    def __init__(self, df=None):
        # Synthetic engine map by default; see app/synthetic_data.py
        self.df = load_engine_map() if df is None else df
        self.model, self.knnr_metrics = train_knnr_model(self.df)
        self.exhaust_model = train_exhaust_temp_model(self.df)

        self.input_data = {"power": 5.0, "des_percent": 10.0, "ch4_percent": 85.0}
        self.dt = 1.0  # seconds per step
        self.time = 0.0
        self.loops = {name: FlowLoop(max_flow=limit) for name, limit in MAX_FLOW_KG_H.items()}
        self.last_results = {}

    def step(self):
        self.time += self.dt
        self.last_results = run_simulation_step(
            self.input_data, self.model, self.exhaust_model, self.loops, self.dt
        )

    def get_current_state(self):
        r = self.last_results
        return {
            "time_s": self.time,
            "input": dict(self.input_data),
            "target": {
                "efficiency": _round(r.get("efficiency_target"), 3),
                "diesel_mass_flow_kg_h": _round(r.get("diesel_target"), 3),
                "ch4_mass_flow_kg_h": _round(r.get("ch4_target"), 3),
                "exhaust_temp": _round(r.get("temp_target"), 1),
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
            },
        }

    def update_inputs(self, inputs_dict):
        self.input_data = {
            "power": inputs_dict["power_output"],
            "des_percent": inputs_dict["diesel_energy_share"],
            "ch4_percent": inputs_dict["methane_percentage"],
        }

    def reset(self):
        self.time = 0.0
        for loop in self.loops.values():
            loop.reset()
        self.last_results = {}
