import numpy as np
import pytest

from app.domain import DOMAIN_FEATURES, OperatingDomain
from app.synthetic_data import CH4_RANGE_PERCENT, DES_RANGE_PERCENT, POWER_RANGE_KW, generate_engine_map


@pytest.fixture(scope="module")
def domain():
    return OperatingDomain.fit(generate_engine_map(), DOMAIN_FEATURES, k=10)


def point(power, ch4, des):
    return {"power_output": power, "calculated_ch4_share_percent": ch4, "des_percent": des}


def test_training_range_covers_the_design_range(domain):
    assert domain.lower.tolist() == [POWER_RANGE_KW[0], CH4_RANGE_PERCENT[0], DES_RANGE_PERCENT[0]]
    assert domain.upper.tolist() == [POWER_RANGE_KW[1], CH4_RANGE_PERCENT[1], DES_RANGE_PERCENT[1]]


def test_at_least_95_percent_of_training_points_are_supported(domain):
    supported = sum(domain.contains(row) for row in domain.training_points)
    assert supported >= 0.95 * len(domain.training_points)


def test_interior_points_supported_including_rated_power(domain):
    assert domain.check(point(12.0, 80.0, 20.0)).supported
    assert domain.check(point(14.4, 90.0, 30.0)).supported  # rated power, mid CH4/DES
    assert domain.check(point(2.0, 80.0, 30.0)).supported


def test_isolated_corner_is_flagged_as_sparse_not_out_of_box(domain):
    check = domain.check(point(14.4, 100.0, 60.0))  # a training point, but the kNN neighbourhood is one-sided
    assert check.within_training_box and check.within_equipment_limits
    assert not check.supported and any("sparse region" in v for v in check.violations)


def test_outside_box_lists_the_violated_features(domain):
    check = domain.check(point(1.0, 50.0, 70.0))
    assert not check.supported and check.within_equipment_limits and not check.within_training_box
    assert len(check.violations) == 3
    assert any("power_output=1" in v and "training range" in v for v in check.violations)


def test_equipment_limits_are_separate_from_training_range(domain):
    check = domain.check(point(15.0, 80.0, 20.0))
    assert not check.within_equipment_limits
    assert any("equipment limits" in v for v in check.violations)


def test_sparse_region_inside_box_is_unsupported():
    # Two dense clusters at low and high power: the middle of the box is empty
    df = generate_engine_map(n_points=600, include_corners=False)
    df = df[~df["power_output"].between(5.0, 11.5)]
    domain = OperatingDomain.fit(df, DOMAIN_FEATURES, k=10)
    middle = point(8.0, 80.0, 30.0)
    check = domain.check(middle)
    assert check.within_training_box and not check.supported
    assert any("sparse region" in v for v in check.violations)
    assert domain.check(point(3.0, 80.0, 30.0)).supported


def test_save_load_roundtrip(tmp_path, domain):
    domain.save(tmp_path / "d.npz")
    loaded = OperatingDomain.load(tmp_path / "d.npz")
    assert loaded.features == domain.features
    assert loaded.support_threshold == domain.support_threshold
    np.testing.assert_array_equal(loaded.training_points, domain.training_points)
    assert loaded.check(point(12.0, 80.0, 20.0)).support_distance == domain.check(point(12.0, 80.0, 20.0)).support_distance


def test_describe_is_json_friendly(domain):
    d = domain.describe()
    assert d["k"] == 10 and d["n_training_points"] == 400
    assert set(d["training_range"]) == set(DOMAIN_FEATURES)
