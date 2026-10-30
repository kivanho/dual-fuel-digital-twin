"""
Engine-map schema, units and validation.

An engine map is a table of steady-state operating points. The models are
trained on it, so a bad map silently becomes a bad twin. `validate_engine_map`
therefore checks types, finiteness, physical ranges, missing values, duplicates
and sample counts, and raises `EngineMapError` with an actionable message
instead of dropping rows quietly.

Columns and units (one row per steady-state operating point):

| column                         | unit  | meaning                                          | accepted range |
|--------------------------------|-------|--------------------------------------------------|----------------|
| power_output                   | kW    | electrical power at the alternator terminals     | 0 < P <= 1.05 x 14.4 |
| calculated_ch4_share_percent   | vol-% | CH4 share of the CH4/CO2 fuel-gas mix            | 0 < x <= 100  |
| des_percent                    | %     | diesel share of the fuel energy                  | 0 <= x <= 100 |
| efficiency_electric            | %     | electrical efficiency (fuel energy -> electrical) | 0 < eta < 60  |
| exhaust_temp                   | degC  | exhaust gas temperature                          | 0 < T < 1000  |
| diesel_mass_flow               | kg/h  | diesel mass flow                                 | >= 0          |
| ch4_mass_flow_calc             | kg/h  | CH4 mass flow                                    | >= 0          |

The power limit is tied to the genset this twin models (14.4 kWe standby
rating, `app/engine_specs.py`); the service clamps power to that rating, so a
map for a larger engine would not be represented correctly anyway.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.calculate_massflows import LHV_CH4, LHV_DIESEL
from app.engine_specs import RATED_POWER_KW

ENGINE_MAP_COLUMNS = [
    "power_output",
    "calculated_ch4_share_percent",
    "des_percent",
    "efficiency_electric",
    "exhaust_temp",
    "diesel_mass_flow",
    "ch4_mass_flow_calc",
]

UNITS = {
    "power_output": "kW",
    "calculated_ch4_share_percent": "vol-%",
    "des_percent": "%",
    "efficiency_electric": "%",
    "exhaust_temp": "degC",
    "diesel_mass_flow": "kg/h",
    "ch4_mass_flow_calc": "kg/h",
}

OPERATING_POINT_COLUMNS = ["power_output", "calculated_ch4_share_percent", "des_percent"]

# (lower, upper, lower_inclusive, upper_inclusive)
PHYSICAL_RANGES = {
    "power_output": (0.0, 1.05 * RATED_POWER_KW, False, True),
    "calculated_ch4_share_percent": (0.0, 100.0, False, True),
    "des_percent": (0.0, 100.0, True, True),
    "efficiency_electric": (0.0, 60.0, False, False),
    "exhaust_temp": (0.0, 1000.0, False, False),
    "diesel_mass_flow": (0.0, float("inf"), True, False),
    "ch4_mass_flow_calc": (0.0, float("inf"), True, False),
}

MIN_ROWS = 50               # below this a hold-out split and a k=10 kNN are not meaningful
MIN_TEST_ROWS = 10
MAX_DROPPED_FRACTION = 0.05  # rows that may be dropped under missing="drop" / duplicates="drop"
ENERGY_BALANCE_TOLERANCE = 0.10  # relative mismatch between mass flows and power/efficiency/des


class EngineMapError(ValueError):
    """The engine map cannot be used for training; the message says why and what to fix."""


@dataclass
class ValidationReport:
    n_rows_in: int
    n_rows_out: int
    dropped_missing: int = 0
    dropped_duplicates: int = 0
    repeated_operating_points: int = 0
    warnings: List[str] = field(default_factory=list)
    ranges: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self):
        return {
            "n_rows_in": self.n_rows_in,
            "n_rows_out": self.n_rows_out,
            "dropped_missing": self.dropped_missing,
            "dropped_duplicates": self.dropped_duplicates,
            "repeated_operating_points": self.repeated_operating_points,
            "warnings": list(self.warnings),
            "ranges": {k: list(v) for k, v in self.ranges.items()},
        }


def _preview(values, n=5):
    return ", ".join(str(v) for v in list(values)[:n]) + (" ..." if len(values) > n else "")


def validate_engine_map(df: pd.DataFrame, *, missing: str = "error", duplicates: str = "error",
                        n_neighbors: int = 10, test_size: float = 0.2,
                        source: Optional[str] = None):
    """
    Validate an engine map and return (clean_df, ValidationReport).

    missing:    "error" (default) rejects any NaN in the required columns;
                "drop" removes those rows if they are at most MAX_DROPPED_FRACTION of the data.
    duplicates: "error" (default) rejects exact duplicate rows (they would leak the hold-out);
                "drop" removes them under the same fraction limit.
    Repeated operating points with different measurements are kept and only counted.
    """
    if missing not in ("error", "drop"):
        raise ValueError("missing must be 'error' or 'drop'")
    if duplicates not in ("error", "drop"):
        raise ValueError("duplicates must be 'error' or 'drop'")
    where = f"{source}: " if source else ""
    report = ValidationReport(n_rows_in=len(df), n_rows_out=len(df))

    missing_cols = [c for c in ENGINE_MAP_COLUMNS if c not in df.columns]
    if missing_cols:
        raise EngineMapError(f"{where}missing columns {missing_cols}. Required columns and units: "
                             + ", ".join(f"{c} [{UNITS[c]}]" for c in ENGINE_MAP_COLUMNS))
    if len(df) == 0:
        raise EngineMapError(f"{where}the engine map has no rows")

    df = df[ENGINE_MAP_COLUMNS].copy()

    # Numeric dtypes: coerce, and report values that were not numbers
    for col in ENGINE_MAP_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            coerced = pd.to_numeric(df[col], errors="coerce")
            bad = df[col][coerced.isna() & df[col].notna()]
            if len(bad):
                raise EngineMapError(f"{where}column '{col}' has {len(bad)} non-numeric value(s), e.g. "
                                     f"{_preview(bad.astype(str).unique())}. Expected numbers in {UNITS[col]}.")
            df[col] = coerced
        df[col] = df[col].astype(float)

    # Infinite values are never acceptable
    inf_counts = {c: int(np.isinf(df[c]).sum()) for c in ENGINE_MAP_COLUMNS}
    inf_counts = {c: n for c, n in inf_counts.items() if n}
    if inf_counts:
        raise EngineMapError(f"{where}non-finite values: {inf_counts}")

    # Missing-value policy
    na_mask = df.isna().any(axis=1)
    n_na = int(na_mask.sum())
    if n_na:
        per_col = {c: int(df[c].isna().sum()) for c in ENGINE_MAP_COLUMNS if df[c].isna().any()}
        if missing == "error":
            raise EngineMapError(f"{where}{n_na} row(s) have missing values ({per_col}). Fill or remove them, "
                                 f"or load with missing='drop' (allowed up to {MAX_DROPPED_FRACTION:.0%} of rows).")
        if n_na > MAX_DROPPED_FRACTION * len(df):
            raise EngineMapError(f"{where}{n_na} of {len(df)} rows ({n_na / len(df):.1%}) have missing values "
                                 f"({per_col}); dropping more than {MAX_DROPPED_FRACTION:.0%} is refused.")
        df = df[~na_mask]
        report.dropped_missing = n_na
        report.warnings.append(f"dropped {n_na} row(s) with missing values: {per_col}")

    # Physical ranges and units
    for col, (lo, hi, lo_inc, hi_inc) in PHYSICAL_RANGES.items():
        v = df[col]
        below = v < lo if lo_inc else v <= lo
        above = v > hi if hi_inc else v >= hi
        bad = v[below | above]
        if len(bad):
            lo_s = "[" if lo_inc else "("
            hi_s = "]" if hi_inc else ")"
            raise EngineMapError(f"{where}{len(bad)} value(s) of '{col}' outside the physical range "
                                 f"{lo_s}{lo:g}, {hi:g}{hi_s} {UNITS[col]}, e.g. {_preview(bad.round(3).unique())}. "
                                 f"Check the units ({UNITS[col]} expected).")

    # Duplicate policy
    dup_mask = df.duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup:
        if duplicates == "error":
            raise EngineMapError(f"{where}{n_dup} exact duplicate row(s). Duplicates leak hold-out points into "
                                 f"training; remove them or load with duplicates='drop'.")
        if n_dup > MAX_DROPPED_FRACTION * len(df):
            raise EngineMapError(f"{where}{n_dup} of {len(df)} rows ({n_dup / len(df):.1%}) are exact duplicates; "
                                 f"dropping more than {MAX_DROPPED_FRACTION:.0%} is refused.")
        df = df[~dup_mask]
        report.dropped_duplicates = n_dup
        report.warnings.append(f"dropped {n_dup} exact duplicate row(s)")
    report.repeated_operating_points = int(df.duplicated(subset=OPERATING_POINT_COLUMNS).sum())
    if report.repeated_operating_points:
        report.warnings.append(f"{report.repeated_operating_points} repeated operating point(s) with different "
                               f"measurements kept (repeat measurements)")

    # Enough data for the split and the estimator
    n = len(df)
    n_test = int(round(n * test_size))
    n_train = n - n_test
    if n < MIN_ROWS:
        raise EngineMapError(f"{where}only {n} usable row(s); at least {MIN_ROWS} are needed for a "
                             f"{test_size:.0%} hold-out and a kNN with k={n_neighbors}.")
    if n_train < 3 * n_neighbors or n_test < MIN_TEST_ROWS:
        raise EngineMapError(f"{where}{n} usable rows give {n_train} training / {n_test} test points; need at least "
                             f"{3 * n_neighbors} training points for k={n_neighbors} and {MIN_TEST_ROWS} test points.")
    for col in OPERATING_POINT_COLUMNS:
        if df[col].nunique() < 2:
            raise EngineMapError(f"{where}'{col}' is constant ({df[col].iloc[0]:g} {UNITS[col]}); the models need "
                                 f"variation in every operating-point feature.")

    # Energy-balance consistency (warning only: the twin recomputes the flows itself)
    q_total = df["power_output"] / (df["efficiency_electric"] / 100) * 3.6
    q_from_flows = df["diesel_mass_flow"] * LHV_DIESEL + df["ch4_mass_flow_calc"] * LHV_CH4
    mismatch = ((q_from_flows - q_total).abs() / q_total)
    n_mismatch = int((mismatch > ENERGY_BALANCE_TOLERANCE).sum())
    if n_mismatch:
        report.warnings.append(f"{n_mismatch} row(s) where the fuel mass flows disagree with power/efficiency/des "
                               f"by more than {ENERGY_BALANCE_TOLERANCE:.0%} (max {mismatch.max():.1%})")

    report.n_rows_out = n
    report.ranges = {c: [float(df[c].min()), float(df[c].max())] for c in ENGINE_MAP_COLUMNS}
    return df.reset_index(drop=True), report


def load_engine_map_csv(path: str, **kwargs):
    """Read a CSV engine map and validate it. Returns (df, ValidationReport)."""
    try:
        raw = pd.read_csv(path)
    except FileNotFoundError:
        raise EngineMapError(f"engine map file not found: {path}")
    except pd.errors.EmptyDataError:
        raise EngineMapError(f"{path}: file is empty")
    return validate_engine_map(raw, source=path, **kwargs)
