import numpy as np
from app.calculate_massflows import calculate_fuel_mass_flows, LHV_DIESEL, LHV_CH4
from app.knnr_model import predict_efficiency
from app.engine_specs import ENGINE_SPEED_RPM, GENSET_EFFICIENCY, POWER_FACTOR, RATED_POWER_KW, VOLTAGE_LL

M_CH4 = 16.04  # molar mass [g/mol]
M_CO2 = 44.01  # molar mass [g/mol]


def shaft_torque_nm(power_kw):
    """Engine shaft torque for an electrical output: T = P_el / (η_genset · ω)."""
    omega = 2 * np.pi * ENGINE_SPEED_RPM / 60  # [rad/s]
    return power_kw * 1000 / (GENSET_EFFICIENCY * omega)


def line_current_a(power_kw):
    """Three-phase line current: I = P / (√3 · U_LL · cos φ)."""
    return power_kw * 1000 / (np.sqrt(3) * VOLTAGE_LL * POWER_FACTOR)


def co2_in_fuel_gas_kg_h(ch4_mass_flow_kg_h, ch4_vol_percent):
    """CO2 carried in with the gas, from the CH4 mass flow and the CH4 volume share of the CH4/CO2 mix."""
    x = ch4_vol_percent / 100
    return ch4_mass_flow_kg_h * (1 - x) / x * M_CO2 / M_CH4


def compute_targets(input_data, efficiency_model, exhaust_model):
    """
    Steady-state targets for an operating point: efficiency and exhaust temperature
    from the models, fuel mass flows from the energy balance.
    """
    power = input_data["power"]
    ch4 = input_data["ch4_percent"]
    des = input_data["des_percent"]
    efficiency_target = predict_efficiency(efficiency_model, power, ch4) / 100
    flows = calculate_fuel_mass_flows(power, efficiency_target, des)
    return {
        "efficiency_target": efficiency_target,
        "diesel_target": flows["diesel_mass_flow_kg_h"],
        "ch4_target": flows["ch4_mass_flow_kg_h"],
        "temp_target": float(exhaust_model.predict(np.array([[power, ch4, des]]))[0]),
    }


def advance_plant(input_data, targets, exhaust_model, loops, dt, substeps=10, running=True):
    """
    Advance the fuel-flow loops by dt seconds and derive the plant outputs from the actual flows.

    While running, the loops move towards the target flows. When stopped, both
    valves are commanded closed (target flow 0) and the flows decay with the
    valve dynamics; the engine winds down to zero power. During transients the
    electrical efficiency is held at the target operating point's value.
    """
    ch4 = input_data["ch4_percent"]
    des = input_data["des_percent"]
    efficiency_target = targets["efficiency_target"]
    flow_targets = ({"diesel": targets["diesel_target"], "ch4": targets["ch4_target"]} if running
                    else {"diesel": 0.0, "ch4": 0.0})

    for _ in range(substeps):
        for name, loop in loops.items():
            loop.step(flow_targets[name], dt / substeps)

    diesel = loops["diesel"].flow
    ch4_flow = loops["ch4"].flow
    q_diesel = diesel * LHV_DIESEL  # [MJ/h]
    q_ch4 = ch4_flow * LHV_CH4      # [MJ/h]
    q_fuel = q_diesel + q_ch4
    power_actual = min(efficiency_target * q_fuel / 3.6, RATED_POWER_KW)
    des_actual = 100 * q_diesel / q_fuel if q_fuel > 0 else des
    temp_actual = float(exhaust_model.predict(np.array([[power_actual, ch4, des_actual]]))[0])

    return {
        **targets,
        "diesel_target": flow_targets["diesel"],
        "ch4_target": flow_targets["ch4"],
        "power": power_actual,
        "diesel": diesel,
        "ch4": ch4_flow,
        "des": des_actual,
        "co2_in_fuel_gas": co2_in_fuel_gas_kg_h(ch4_flow, ch4),
        "temp": temp_actual,
        "current": line_current_a(power_actual),
        "torque": shaft_torque_nm(power_actual),
    }


def run_simulation_step(input_data, efficiency_model, exhaust_model, loops, dt, substeps=10, running=True):
    """
    Advance the twin by dt seconds: compute the targets for the requested
    operating point, then move the plant towards them (see advance_plant).
    """
    targets = compute_targets(input_data, efficiency_model, exhaust_model)
    return advance_plant(input_data, targets, exhaust_model, loops, dt, substeps, running)
