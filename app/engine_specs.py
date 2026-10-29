"""
Generator set specifications, taken from the manufacturer datasheet of the
diesel genset the dual-fuel test bench is based on (50 Hz values).
"""

# Electrical side: three-phase alternator
VOLTAGE_LL = 400.0          # line-to-line voltage [V]
FREQUENCY = 50.0            # [Hz]
POWER_FACTOR = 0.8          # cos φ the ratings are given at
RATED_POWER_KW = 14.4       # standby electrical rating (18 kVA × 0.8) [kWe]
CONTINUOUS_POWER_KW = 13.2  # continuous electrical rating (16.5 kVA × 0.8) [kWe]

# Mechanical side
ENGINE_SPEED_RPM = 1500.0   # 4-pole alternator at 50 Hz
ENGINE_GROSS_POWER_KW = 18.0  # standby gross engine power [kW]

# Shaft-to-terminal efficiency at the standby rating (14.4 kWe / 18.0 kW).
# Includes alternator and auxiliary losses; used as a constant.
GENSET_EFFICIENCY = RATED_POWER_KW / ENGINE_GROSS_POWER_KW
