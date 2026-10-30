import math

from app.evaluation import EFFICIENCY_MODELS, EXHAUST_MODELS, PRODUCTION, REGIONS, evaluate, to_markdown
from app.synthetic_data import generate_engine_map


def test_evaluate_covers_all_splits_models_and_residuals():
    df = generate_engine_map()
    r = evaluate(df, seed=42)
    assert set(r["splits"]) == {"random_holdout", *REGIONS}
    rh = r["splits"]["random_holdout"]
    assert rh["n_train"] == 320 and rh["n_test"] == 80
    assert set(rh["efficiency"]) == set(EFFICIENCY_MODELS)
    assert set(rh["exhaust_temp"]) == set(EXHAUST_MODELS)
    for split in r["splits"].values():
        assert "skipped" not in split
        for key in ("efficiency", "exhaust_temp"):
            for scores in split[key].values():
                assert math.isfinite(scores["rmse"]) and scores["rmse"] >= 0
    # the production models beat the mean baseline on the interpolation hold-out
    assert rh["efficiency"][PRODUCTION["efficiency"]]["rmse"] < rh["efficiency"]["mean"]["rmse"]
    assert rh["exhaust_temp"][PRODUCTION["exhaust_temp"]]["rmse"] < rh["exhaust_temp"]["mean"]["rmse"]
    res = r["residuals"]
    assert set(res) == {"efficiency_knn_2f", "exhaust_temp_linear_3f"}
    for table in res.values():
        assert set(table["by_feature"]) == {"power_output", "calculated_ch4_share_percent", "des_percent"}
        assert sum(row["n"] for row in table["by_feature"]["power_output"]) == 80


def test_evaluate_is_deterministic():
    df = generate_engine_map()
    assert evaluate(df, seed=42) == evaluate(df, seed=42)


def test_small_region_is_skipped_not_crashed():
    df = generate_engine_map(n_points=400)
    df = df[df["power_output"] < 10.5]  # no points in the high-load band
    r = evaluate(df)
    assert "skipped" in r["splits"]["heldout_high_load"]
    md = to_markdown(r, "t", "cmd")
    assert "skipped" in md
