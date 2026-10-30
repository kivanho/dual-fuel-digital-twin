"""
Model evaluation on an engine map.

Everything here is computed on whatever map is passed in. On the synthetic map
the numbers are implementation checks (the generator is a known smooth formula
plus noise); they say nothing about a real engine.

Splits
- random_holdout: 20 % random hold-out. Interpolation inside the sampled box.
- heldout_*: a whole operating REGION (load band, CH4 band, DES band) is held
  out and the models are trained on the rest. Edge bands test extrapolation,
  the mid-load band tests interpolation across a gap.

Models (names used in the result tables)
- mean:       predicts the training mean (floor baseline)
- ridge_2f:   ridge regression on power + CH4 share
- ridge_3f:   ridge regression on power + CH4 share + DES
- knn_2f:     kNN (k=10, Manhattan, standardised) on power + CH4 share  <- production efficiency model
- knn_3f:     same kNN with DES as a third feature
- linear_3f:  ordinary least squares on all three features             <- production exhaust-temperature model

The with/without-DES comparison is knn_2f vs knn_3f (and ridge_2f vs ridge_3f).
R² on a narrow held-out band is computed against that band's own variance and
can be low or negative even when the RMSE is small; read both columns.
"""
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.exhaust_temp_model import FEATURES as TEMP_FEATURES, TARGET as TEMP_TARGET, build_exhaust_temp_model
from app.knnr_model import FEATURES as EFF_FEATURES, TARGET as EFF_TARGET, MINKOWSKI_P, N_NEIGHBORS, build_knnr_model

F2 = list(EFF_FEATURES)                      # power, CH4 share
F3 = list(TEMP_FEATURES)                     # power, CH4 share, DES
RIDGE_ALPHA = 1.0

PRODUCTION = {"efficiency": "knn_2f", "exhaust_temp": "linear_3f"}

EFFICIENCY_MODELS = {
    "mean": (lambda: DummyRegressor(strategy="mean"), F2),
    "ridge_2f": (lambda: Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=RIDGE_ALPHA))]), F2),
    "ridge_3f": (lambda: Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=RIDGE_ALPHA))]), F3),
    "knn_2f": (build_knnr_model, F2),
    "knn_3f": (build_knnr_model, F3),
}
EXHAUST_MODELS = {
    "mean": (lambda: DummyRegressor(strategy="mean"), F3),
    "linear_3f": (build_exhaust_temp_model, F3),
    "knn_3f": (build_knnr_model, F3),
}
TARGETS = {"efficiency": (EFF_TARGET, EFFICIENCY_MODELS, "%-pt"), "exhaust_temp": (TEMP_TARGET, EXHAUST_MODELS, "degC")}

# Held-out regions (name -> (column, lower, upper, kind)); the test set is lower <= x < upper
REGIONS = {
    "heldout_high_load": ("power_output", 11.0, np.inf, "extrapolation: train below 11 kW, test at >= 11 kW"),
    "heldout_low_load": ("power_output", -np.inf, 4.5, "extrapolation: train above 4.5 kW, test at < 4.5 kW"),
    "heldout_mid_load": ("power_output", 7.0, 9.0, "interpolation across a gap: test at 7-9 kW"),
    "heldout_low_ch4": ("calculated_ch4_share_percent", -np.inf, 65.0, "extrapolation: test at CH4 < 65 vol-%"),
    "heldout_high_des": ("des_percent", 50.0, np.inf, "extrapolation: test at DES >= 50 %"),
}


def _scores(model_factory, features, target, train, test):
    model = model_factory()
    model.fit(train[features].values, train[target].values)
    pred = model.predict(test[features].values)
    y = test[target].values
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if len(y) > 1 and np.var(y) > 0 else float("nan"),
        "mae": float(np.mean(np.abs(y - pred))),
    }, pred


def _evaluate_split(train, test):
    out = {"n_train": int(len(train)), "n_test": int(len(test))}
    for key, (target, models, _unit) in TARGETS.items():
        out[key] = {name: _scores(factory, feats, target, train, test)[0] for name, (factory, feats) in models.items()}
    return out


def _residual_table(df_test, residuals, column, edges):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        last = hi == edges[-1]
        mask = (df_test[column] >= lo) & ((df_test[column] <= hi) if last else (df_test[column] < hi))
        r = residuals[mask.values]
        rows.append({
            "bin": f"[{lo:.3g}, {hi:.3g}{']' if last else ')'}",
            "n": int(mask.sum()),
            "mean_residual": float(r.mean()) if len(r) else float("nan"),
            "rmse": float(np.sqrt(np.mean(r ** 2))) if len(r) else float("nan"),
        })
    return rows


def evaluate(df: pd.DataFrame, seed: int = 42, test_size: float = 0.2, n_bins: int = 3) -> Dict:
    """Run all evaluations on `df` and return a JSON-serialisable dict."""
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    splits = {"random_holdout": {"description": f"{test_size:.0%} random hold-out (interpolation)",
                                 **_evaluate_split(train, test)}}
    for name, (col, lo, hi, kind) in REGIONS.items():
        mask = (df[col] >= lo) & (df[col] < hi)
        if mask.sum() < 5 or (~mask).sum() < 3 * N_NEIGHBORS:
            splits[name] = {"description": kind, "skipped": "not enough points in this region of the map"}
            continue
        splits[name] = {"description": kind, **_evaluate_split(df[~mask], df[mask])}

    # Residuals of the production models on the random hold-out, binned over each feature
    residuals = {}
    for key, (target, models, unit) in TARGETS.items():
        factory, feats = models[PRODUCTION[key]]
        _, pred = _scores(factory, feats, target, train, test)
        res = test[target].values - pred
        tables = {}
        for col in F3:
            edges = np.linspace(df[col].min(), df[col].max(), n_bins + 1)
            tables[col] = _residual_table(test, res, col, edges)
        worst = max((abs(r["mean_residual"]) for t in tables.values() for r in t if r["n"]), default=float("nan"))
        residuals[f"{key}_{PRODUCTION[key]}"] = {"unit": unit, "overall_rmse": float(np.sqrt(np.mean(res ** 2))),
                                               "max_abs_bin_mean_residual": float(worst), "by_feature": tables}

    return {
        "config": {"seed": seed, "test_size": test_size, "knn": {"n_neighbors": N_NEIGHBORS, "p": MINKOWSKI_P},
                   "ridge_alpha": RIDGE_ALPHA, "residual_bins": n_bins},
        "production_models": dict(PRODUCTION),
        "features": {"2f": F2, "3f": F3},
        "splits": splits,
        "residuals": residuals,
    }


def _fmt(x, digits):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def to_markdown(results: Dict, title: str, command: str) -> str:
    """Render the evaluation as Markdown tables."""
    lines = [f"# {title}", "", f"Generated by `{command}`.", "",
             "All numbers are on the data named in the title. Synthetic-map scores are implementation checks "
             "of the pipeline, not evidence about a real engine. R² on a held-out band is relative to that "
             "band's own variance.", ""]
    for key, (_t, models, unit) in TARGETS.items():
        lines += [f"## {key} (RMSE in {unit}, R²)", ""]
        names = list(models)
        header = "| split | n train / test | " + " | ".join(f"{n}{' (prod)' if n == PRODUCTION[key] else ''}" for n in names) + " |"
        lines += [header, "|" + "---|" * (len(names) + 2)]
        for split, res in results["splits"].items():
            if "skipped" in res:
                lines.append(f"| {split} | - | " + " | ".join("skipped" for _ in names) + " |")
                continue
            cells = [f"{_fmt(res[key][n]['rmse'], 2)} / {_fmt(res[key][n]['r2'], 3)}" for n in names]
            lines.append(f"| {split} | {res['n_train']} / {res['n_test']} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Residuals of the production models (random hold-out)", ""]
    for name, r in results["residuals"].items():
        lines += [f"### {name} (unit {r['unit']}, overall RMSE {r['overall_rmse']:.2f}, "
                  f"largest |bin mean residual| {r['max_abs_bin_mean_residual']:.2f})", "",
                  "| feature | bin | n | mean residual | RMSE |", "|---|---|---|---|---|"]
        for col, rows in r["by_feature"].items():
            for row in rows:
                lines.append(f"| {col} | {row['bin']} | {row['n']} | {_fmt(row['mean_residual'], 2)} | {_fmt(row['rmse'], 2)} |")
        lines.append("")
    return "\n".join(lines)
