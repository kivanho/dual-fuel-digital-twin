import numpy as np
from app.calculate_massflows import calculate_fuel_mass_flows, LHV_DIESEL, LHV_CH4
from app.regulator import apply_regulation
from app.knnr_model import predict_efficiency_knnr
from app.engine_specs import ENGINE_SPEED_RPM, GENSET_EFFICIENCY, POWER_FACTOR, VOLTAGE_LL


def shaft_torque_nm(power_kw):
    """Engine shaft torque for an electrical output: T = P_el / (η_genset · ω)."""
    omega = 2 * np.pi * ENGINE_SPEED_RPM / 60  # [rad/s]
    return power_kw * 1000 / (GENSET_EFFICIENCY * omega)


def line_current_a(power_kw):
    """Three-phase line current: I = P / (√3 · U_LL · cos φ)."""
    return power_kw * 1000 / (np.sqrt(3) * VOLTAGE_LL * POWER_FACTOR)


def run_simulation_step(
    frame_time, input_data, df, model, exhaust_model,
    sim_state, sim_vel, integral, previous_error, dt,
    controller_type, T, zeta, omega_0, kp, ki, kd
):
    power = input_data["power"]
    ch4 = input_data["ch4_percent"]
    des = input_data["des_percent"]

    result = predict_efficiency_knnr(model, df, power, ch4)
    eff_soll = result["predicted_efficiency"] / 100
    eff_real = result["measured_efficiency"] / 100
    flows = calculate_fuel_mass_flows(power, eff_soll, des)

    diesel_soll = flows["diesel_mass_flow_kg_h"]
    ch4_soll = flows["ch4_mass_flow_kg_h"]
    temp_soll = exhaust_model.predict(np.array([[power, ch4, des]]))[0]

    # Apply regulation
    targets = {"diesel": diesel_soll, "ch4": ch4_soll}
    sim_state, sim_vel, integral, previous_error = apply_regulation(
        sim_state, sim_vel, integral, previous_error,
        targets, dt, controller_type, T, zeta, omega_0, kp, ki, kd
    )

    # Actual values
    actual_diesel = sim_state["diesel"]
    actual_ch4 = sim_state["ch4"]
    fuel_power = actual_diesel * LHV_DIESEL + actual_ch4 * LHV_CH4  # [MJ/h]
    q_co2_kg_h = (100 / ch4 - 1) * actual_ch4
    power_actual = eff_soll * fuel_power / 3.6
    actual_eff = (power_actual * 3.6) / fuel_power if fuel_power else 0.0
    gas_flow = q_co2_kg_h + actual_ch4
    ch4_new = actual_ch4 / gas_flow * 100 if gas_flow else ch4
    temp_actual = exhaust_model.predict(np.array([[power_actual, ch4_new, des]]))[0]

    # Match closest engine-map point for validation
    match = df[
        np.isclose(df["power_output"], result["closest_measured_power"], atol=0.1) &
        np.isclose(df["calculated_ch4_share_percent"], result["closest_measured_ch4"], atol=0.5)
    ]
    closest = match.iloc[0] if not match.empty else None

    diesel_real = closest["diesel_mass_flow"] if closest is not None else np.nan
    ch4_real = closest["ch4_mass_flow_calc"] if closest is not None else np.nan
    temp_real = closest["exhaust_temp"] if closest is not None else np.nan
    des_closest = f"{closest['des_percent']:.2f}" if closest is not None else "N/A"

    results = {
        "eff_soll": eff_soll,
        "eff_real": eff_real,
        "eff_actual": actual_eff,
        "power_actual": power_actual,
        "diesel_soll": diesel_soll,
        "ch4_soll": ch4_soll,
        "diesel": actual_diesel,
        "ch4": actual_ch4,
        "q_co2_kg_h": q_co2_kg_h,
        "temp_soll": temp_soll,
        "temp": temp_actual,
        "current": line_current_a(power_actual),
        "torque": shaft_torque_nm(power_actual),
        "frame_time": frame_time,

        # Closest engine-map point, for comparison
        "closest_measured_power": result["closest_measured_power"],
        "closest_measured_ch4": result["closest_measured_ch4"],
        "diesel_real": diesel_real,
        "ch4_real": ch4_real,
        "temp_real": temp_real,
        "des_closest": des_closest,
        "ch4_new": ch4_new,
    }

    return results, sim_state, sim_vel, integral, previous_error
