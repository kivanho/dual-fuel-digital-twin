from app.simulation_core import run_simulation_step
from app.synthetic_data import load_engine_map
from app.knnr_model import train_knnr_model
from app.exhaust_temp_model import train_exhaust_temp_model

class SimulationService:
    def __init__(self, df=None):
        # Synthetic engine map by default; see app/synthetic_data.py
        self.df = load_engine_map() if df is None else df
        self.model, self.knnr_metrics = train_knnr_model(self.df)
        self.exhaust_model = train_exhaust_temp_model(self.df)

        self.input_data = {
            "power": 5.0,
            "des_percent": 10.0,
            "ch4_percent": 85.0
        }

        self.dt = 1  # seconds
        self.time = 0.0

        self.controller_type = {"diesel": "pid", "ch4": "pid"}
        self.T = {k: 2.0 for k in self.controller_type}
        self.zeta = {k: 0 for k in self.controller_type}
        self.omega_0 = {k: 0.05 for k in self.controller_type}
        self.kp = {k: 1.8 for k in self.controller_type}
        self.ki = {k: 0.1 for k in self.controller_type}
        self.kd = {k: 0.08 for k in self.controller_type}

        self.sim_state = {k: 0.0 for k in self.controller_type}
        self.sim_vel = {k: 0.0 for k in self.controller_type}
        self.integral = {k: 0.0 for k in self.controller_type}
        self.previous_error = {k: 0.0 for k in self.controller_type}

        self.last_results = {}

    def step(self):
        self.time += self.dt
        self.last_results, self.sim_state, self.sim_vel, self.integral, self.previous_error = run_simulation_step(
            self.time, self.input_data, self.df, self.model, self.exhaust_model,
            self.sim_state, self.sim_vel, self.integral, self.previous_error, self.dt,
            self.controller_type, self.T, self.zeta, self.omega_0, self.kp, self.ki, self.kd
        )

    def get_current_state(self):
        return {
            "input": {
                "power": self.input_data["power"],
                "ch4_percent": self.input_data["ch4_percent"],
                "des_percent": self.input_data["des_percent"]
            },
            "target": {
                "efficiency": self.last_results.get("eff_soll"),
                "diesel_mass_flow_kg_h": self.last_results.get("diesel_soll"),
                "ch4_mass_flow_kg_h": self.last_results.get("ch4_soll"),
                "exhaust_temp": self.last_results.get("temp_soll")
            },
            "actual": {
                "efficiency": self.last_results.get("eff_actual"),
                "power": self.last_results.get("power_actual"),
                "diesel_mass_flow_kg_h": self.last_results.get("diesel"),
                "ch4_mass_flow_kg_h": self.last_results.get("ch4"),
                "current": self.last_results.get("current"),
                "torque": self.last_results.get("torque"),
                "exhaust_temp": self.last_results.get("temp"),
                "q_co2_kg_h": self.last_results.get("q_co2_kg_h"),
                "ch4_share": self.last_results.get("ch4_new")
            }
        }

    def update_inputs(self, inputs_dict):
        self.input_data = {
            "power": inputs_dict["power_output"],
            "des_percent": inputs_dict["diesel_energy_share"],
            "ch4_percent": inputs_dict["methane_percentage"]
        }

    def reset(self):
        self.time = 0.0
        self.sim_state = {k: 0.0 for k in self.controller_type}
        self.sim_vel = {k: 0.0 for k in self.controller_type}
        self.integral = {k: 0.0 for k in self.controller_type}
        self.previous_error = {k: 0.0 for k in self.controller_type}
        self.last_results = {}