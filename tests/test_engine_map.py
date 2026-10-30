import numpy as np
import pandas as pd
import pytest

from app.engine_map import (ENGINE_MAP_COLUMNS, EngineMapError, MIN_ROWS, UNITS, load_engine_map_csv,
                            validate_engine_map)
from app.synthetic_data import generate_engine_map, load_engine_map


def test_units_documented_for_every_column():
    assert set(UNITS) == set(ENGINE_MAP_COLUMNS)


def test_synthetic_map_passes_validation():
    df, report = validate_engine_map(generate_engine_map())
    assert len(df) == 400
    assert report.n_rows_out == 400
    assert report.dropped_missing == 0 and report.dropped_duplicates == 0
    assert report.ranges["power_output"][0] >= 2.0
    # The synthetic flows close the energy balance, so no mismatch warning
    assert not any("disagree" in w for w in report.warnings)


def test_missing_column_error_lists_units():
    df = generate_engine_map().drop(columns=["exhaust_temp"])
    with pytest.raises(EngineMapError, match=r"missing columns \['exhaust_temp'\].*exhaust_temp \[degC\]"):
        validate_engine_map(df)


def test_non_numeric_values_are_reported():
    df = generate_engine_map()
    df["power_output"] = df["power_output"].astype(object)
    df.loc[3, "power_output"] = "n/a"
    with pytest.raises(EngineMapError, match="non-numeric.*n/a.*kW"):
        validate_engine_map(df)


def test_infinite_values_rejected():
    df = generate_engine_map()
    df.loc[0, "exhaust_temp"] = np.inf
    with pytest.raises(EngineMapError, match="non-finite"):
        validate_engine_map(df)


def test_missing_values_error_by_default_and_drop_when_few():
    df = generate_engine_map()
    df.loc[[1, 2], "efficiency_electric"] = np.nan
    with pytest.raises(EngineMapError, match="2 row\\(s\\) have missing values.*missing='drop'"):
        validate_engine_map(df)
    clean, report = validate_engine_map(df, missing="drop")
    assert len(clean) == 398 and report.dropped_missing == 2
    assert any("dropped 2" in w for w in report.warnings)


def test_dropping_many_missing_rows_is_refused():
    df = generate_engine_map()
    df.loc[df.index[:40], "exhaust_temp"] = np.nan  # 10 % of the rows
    with pytest.raises(EngineMapError, match="10.0%.*refused"):
        validate_engine_map(df, missing="drop")


@pytest.mark.parametrize("column, value, unit", [
    ("power_output", 20.0, "kW"),        # above the genset rating: wrong engine or wrong unit
    ("power_output", 0.0, "kW"),
    ("calculated_ch4_share_percent", 0.0, "vol-%"),
    ("des_percent", 101.0, "%"),
    ("efficiency_electric", 65.0, "%"),
    ("exhaust_temp", 1200.0, "degC"),
    ("diesel_mass_flow", -0.1, "kg/h"),
])
def test_physical_range_violations(column, value, unit):
    df = generate_engine_map()
    df.loc[0, column] = value
    with pytest.raises(EngineMapError, match=f"'{column}' outside the physical range.*{unit}"):
        validate_engine_map(df)


def test_efficiency_as_fraction_triggers_energy_balance_warning():
    # 0.05..0.40 instead of 5..40 % lies inside the physical range, so the range check cannot
    # catch it; the mass flows no longer close the energy balance, which is reported.
    df = generate_engine_map()
    df["efficiency_electric"] = df["efficiency_electric"] / 100
    _, report = validate_engine_map(df)
    assert any("disagree" in w for w in report.warnings)


def test_exact_duplicates_error_or_drop():
    df = generate_engine_map()
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)
    with pytest.raises(EngineMapError, match="3 exact duplicate"):
        validate_engine_map(df)
    clean, report = validate_engine_map(df, duplicates="drop")
    assert len(clean) == 400 and report.dropped_duplicates == 3


def test_repeated_operating_points_are_kept_and_counted():
    df = generate_engine_map()
    repeat = df.iloc[:5].copy()
    repeat["efficiency_electric"] += 0.1  # repeat measurement, different reading
    clean, report = validate_engine_map(pd.concat([df, repeat], ignore_index=True))
    assert len(clean) == 405
    assert report.repeated_operating_points == 5


def test_too_few_rows():
    df = generate_engine_map().iloc[:MIN_ROWS - 1]
    with pytest.raises(EngineMapError, match=f"at least {MIN_ROWS}"):
        validate_engine_map(df)


def test_split_must_leave_enough_training_points_for_knn():
    df = generate_engine_map().iloc[:60]
    with pytest.raises(EngineMapError, match="training points for k=30"):
        validate_engine_map(df, n_neighbors=30)


def test_constant_feature_rejected():
    df = generate_engine_map()
    df["des_percent"] = 30.0
    with pytest.raises(EngineMapError, match="'des_percent' is constant"):
        validate_engine_map(df)


def test_csv_loader_roundtrip_and_errors(tmp_path):
    path = tmp_path / "map.csv"
    generate_engine_map().to_csv(path, index=False)
    df, report = load_engine_map_csv(str(path))
    assert len(df) == 400 and report.n_rows_in == 400
    assert load_engine_map(str(path)).equals(df)
    with pytest.raises(EngineMapError, match="not found"):
        load_engine_map_csv(str(tmp_path / "nope.csv"))
    (tmp_path / "empty.csv").write_text("")
    with pytest.raises(EngineMapError, match="empty"):
        load_engine_map_csv(str(tmp_path / "empty.csv"))


def test_env_var_selects_csv(tmp_path, monkeypatch):
    path = tmp_path / "map.csv"
    generate_engine_map(n_points=120, seed=1).to_csv(path, index=False)
    monkeypatch.setenv("ENGINE_MAP_CSV", str(path))
    assert len(load_engine_map()) == 120
