"""
Synthetic engine map for an 18 kVA (14.4 kWe) diesel/methane dual-fuel generator.

The original project trained on lab measurements that are not public. This
module generates plausible operating points with the same columns so the
models, the energy balance and the flow loops can run end to end. The shapes
(efficiency rising with load and saturating, CO2 dilution lowering
efficiency and exhaust temperature) follow textbook dual-fuel behaviour.
At 100 % CH4 and 30 % diesel energy share the curves are anchored to the
diesel-only genset datasheet (electrical efficiency from the fuel consumption
at 50/75/100 % load, exhaust temperature at the prime rating). Treating the
dual-fuel engine like the diesel original at that point is an assumption;
everything else is illustrative, not measured.
"""
import os

import numpy as np
import pandas as pd

from app.calculate_massflows import LHV_CH4, LHV_DIESEL
from app.engine_map import ENGINE_MAP_COLUMNS, load_engine_map_csv  # noqa: F401 (re-exported)
from app.engine_specs import RATED_POWER_KW

# Operating range the synthetic generator samples. The API does NOT use these
# constants: the accepted domain comes from the trained bundle (app/domain.py).
POWER_RANGE_KW = (2.0, RATED_POWER_KW)
CH4_RANGE_PERCENT = (55.0, 100.0)
DES_RANGE_PERCENT = (10.0, 60.0)

DEFAULT_N_POINTS = 400
DEFAULT_SEED = 42


def generate_engine_map(n_points=DEFAULT_N_POINTS, seed=DEFAULT_SEED):
    """Return a DataFrame of synthetic steady-state operating points."""
    rng = np.random.default_rng(seed)

    power = rng.uniform(*POWER_RANGE_KW, n_points)
    ch4 = rng.uniform(*CH4_RANGE_PERCENT, n_points)
    des = rng.uniform(*DES_RANGE_PERCENT, n_points)

    load = power / RATED_POWER_KW
    efficiency = (
        30.5 * (1 - np.exp(-power / 3.5))       # part-load losses
        - 0.06 * (100 - ch4) * (1 - 0.5 * load) # CO2 dilution hurts most at low load
        + 0.03 * (des - 30) * (1 - load)        # more pilot diesel stabilises low-load combustion
        + rng.normal(0, 0.6, n_points)
    )
    efficiency = np.clip(efficiency, 5.0, 40.0)

    exhaust_temp = (
        140.0
        + 17.0 * power
        - 0.9 * (100 - ch4)
        + 0.5 * (des - 30)
        + rng.normal(0, 6.0, n_points)
    )

    # Energy balance: fuel power [MJ/h] split by diesel energy share
    q_total = power / (efficiency / 100) * 3.6
    diesel_mass_flow = des / 100 * q_total / LHV_DIESEL
    ch4_mass_flow = (1 - des / 100) * q_total / LHV_CH4

    return pd.DataFrame({
        "power_output": power,
        "calculated_ch4_share_percent": ch4,
        "des_percent": des,
        "efficiency_electric": efficiency,
        "exhaust_temp": exhaust_temp,
        "diesel_mass_flow": diesel_mass_flow,
        "ch4_mass_flow_calc": ch4_mass_flow,
    })[ENGINE_MAP_COLUMNS]


def load_engine_map(path=None, **validation_kwargs):
    """
    Return the engine map the twin trains on.

    Synthetic by default. Pass a CSV path (or set ENGINE_MAP_CSV) with the
    columns and units documented in app/engine_map.py to train on your own
    measurements; the file is validated and rejected with an explicit message
    if it is unusable.
    """
    path = path or os.environ.get("ENGINE_MAP_CSV")
    if not path:
        return generate_engine_map()
    df, _report = load_engine_map_csv(path, **validation_kwargs)
    return df
