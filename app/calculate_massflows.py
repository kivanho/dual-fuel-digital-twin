# Lower heating values [MJ/kg]
LHV_DIESEL = 42.7
LHV_CH4 = 50.03


def calculate_fuel_mass_flows(power_output_kW, efficiency, des):
    """
    Calculates fuel mass flows from power output, efficiency, and diesel energy share.

    Parameters:
    - power_output_kW: float, electrical power output [kW]
    - efficiency: float, electrical efficiency (e.g. 0.30)
    - des: float, Diesel Energy Share [%] (0–100)

    Returns:
    - dict with total energy input and mass flows for Diesel & CH4
    """
    des = des / 100  # Convert percentage to fraction

    # 1. Total energy input (MJ/h)
    Q_total = power_output_kW / efficiency * 3.6

    # 2. Energy split
    Q_diesel = des * Q_total
    Q_ch4 = (1 - des) * Q_total

    # 3. Mass flows
    m_diesel = Q_diesel / LHV_DIESEL
    m_ch4 = Q_ch4 / LHV_CH4

    return {
        "Q_total_MJ_h": round(Q_total, 2),
        "Q_diesel_MJ_h": round(Q_diesel, 2),
        "Q_ch4_MJ_h": round(Q_ch4, 2),
        "diesel_mass_flow_kg_h": round(m_diesel, 2),
        "ch4_mass_flow_kg_h": round(m_ch4, 2)
    }
