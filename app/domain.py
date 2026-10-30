"""
Supported operating domain of a trained model bundle.

Two checks decide whether an operating point (power, CH4 share, diesel energy
share) is accepted:

1. Equipment limits: the physical range of the genset (0 < P <= 14.4 kWe,
   0 < CH4 <= 100 vol-%, 0 <= DES <= 100 %). Independent of the training data.
2. Training support: the point must lie inside the per-feature min/max box of
   the training data AND its Manhattan distance (in standardised feature space,
   the same metric the kNN uses) to its k-th nearest training point must not
   exceed a threshold derived from the training set itself: the 95th percentile
   (SUPPORT_QUANTILE) of the k-th neighbour distances the training points have
   to each other. The box alone does not establish support in sparse regions;
   the distance check does. Points as isolated as the sparsest 5 % of the
   training data are flagged, which includes the extreme corners of a
   uniformly sampled map: the kNN averages one-sidedly there, so the flag is
   deliberate. (The maximum instead of a percentile would be dominated by those
   corners and let a sizeable hole in the middle of the map pass.)

Both models are evaluated at the same three operating-point features, so one
domain covers both (the efficiency model simply ignores DES).

Steady-state only: the training points are steady-state operating points. The
simulator passes *transient* plant states (e.g. power ramping from 0 kW) to the
exhaust-temperature model, which then extrapolates linearly. Snapshots report
`actual.within_training_domain` so a client can see when that happens; no
accuracy is claimed for such states.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from sklearn.neighbors import NearestNeighbors

from app.engine_map import UNITS
from app.engine_specs import RATED_POWER_KW

DOMAIN_FEATURES = ["power_output", "calculated_ch4_share_percent", "des_percent"]
SUPPORT_QUANTILE = 0.95

# Equipment limits (lower, upper, lower_inclusive, upper_inclusive)
EQUIPMENT_LIMITS = {
    "power_output": (0.0, RATED_POWER_KW, False, True),
    "calculated_ch4_share_percent": (0.0, 100.0, False, True),
    "des_percent": (0.0, 100.0, True, True),
}

# Friendly names used in the API payload
API_NAMES = {
    "power_output": "power_output",
    "calculated_ch4_share_percent": "methane_percentage",
    "des_percent": "diesel_energy_share",
}


@dataclass
class SupportCheck:
    supported: bool
    within_equipment_limits: bool
    within_training_box: bool
    support_distance: float
    support_threshold: float
    violations: List[str]

    def to_dict(self):
        return {
            "supported": self.supported,
            "within_equipment_limits": self.within_equipment_limits,
            "within_training_box": self.within_training_box,
            "support_distance": round(self.support_distance, 4),
            "support_threshold": round(self.support_threshold, 4),
            "violations": list(self.violations),
        }


class OperatingDomain:
    def __init__(self, features: Sequence[str], lower, upper, scale_mean, scale_std,
                 training_points, k: int, support_threshold: float):
        self.features = list(features)
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.scale_mean = np.asarray(scale_mean, dtype=float)
        self.scale_std = np.asarray(scale_std, dtype=float)
        self.training_points = np.asarray(training_points, dtype=float)
        self.k = int(k)
        self.support_threshold = float(support_threshold)
        self._nn = NearestNeighbors(n_neighbors=self.k, p=1).fit(self._scale(self.training_points))

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(cls, df, features=DOMAIN_FEATURES, k=10, quantile=SUPPORT_QUANTILE):
        X = df[list(features)].values.astype(float)
        mean, std = X.mean(axis=0), X.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        Xs = (X - mean) / std
        # k-th neighbour distance of each training point to the *other* training points
        nn = NearestNeighbors(n_neighbors=k + 1, p=1).fit(Xs)
        dist, _ = nn.kneighbors(Xs)
        threshold = float(np.quantile(dist[:, k], quantile))
        return cls(features, X.min(axis=0), X.max(axis=0), mean, std, X, k, threshold)

    # ------------------------------------------------------------------ checks
    def _scale(self, X):
        return (np.asarray(X, dtype=float) - self.scale_mean) / self.scale_std

    def support_distance(self, point) -> float:
        """Manhattan distance in standardised space to the k-th nearest training point."""
        dist, _ = self._nn.kneighbors(self._scale([point]))
        return float(dist[0, -1])

    def check(self, point: Dict[str, float]) -> SupportCheck:
        """Check one operating point given as {feature: value}."""
        values = [float(point[f]) for f in self.features]
        violations = []
        in_limits = True
        for name, v in zip(self.features, values):
            lo, hi, lo_inc, hi_inc = EQUIPMENT_LIMITS[name]
            below = v < lo if lo_inc else v <= lo
            above = v > hi if hi_inc else v >= hi
            if below or above:
                in_limits = False
                violations.append(f"{API_NAMES[name]}={v:g} outside equipment limits "
                                  f"{'[' if lo_inc else '('}{lo:g}, {hi:g}{']' if hi_inc else ')'} {UNITS[name]}")
        in_box = True
        for name, v, lo, hi in zip(self.features, values, self.lower, self.upper):
            if v < lo or v > hi:
                in_box = False
                violations.append(f"{API_NAMES[name]}={v:g} outside the training range "
                                  f"[{lo:.3g}, {hi:.3g}] {UNITS[name]}")
        distance = self.support_distance(values)
        if in_box and distance > self.support_threshold:
            violations.append(f"operating point is in a sparse region of the training data "
                              f"(support distance {distance:.3f} > {self.support_threshold:.3f})")
        supported = in_limits and in_box and distance <= self.support_threshold
        return SupportCheck(supported, in_limits, in_box, distance, self.support_threshold, violations)

    def contains(self, values: Sequence[float]) -> bool:
        """Fast box + distance check on a feature vector in `self.features` order (no equipment limits)."""
        v = np.asarray(values, dtype=float)
        if np.any(v < self.lower) or np.any(v > self.upper):
            return False
        return self.support_distance(v) <= self.support_threshold

    # ------------------------------------------------------------------ (de)serialisation
    def describe(self) -> dict:
        """JSON-serialisable summary without the training points."""
        return {
            "features": self.features,
            "units": {f: UNITS[f] for f in self.features},
            "training_range": {f: [float(lo), float(hi)] for f, lo, hi in zip(self.features, self.lower, self.upper)},
            "equipment_limits": {f: [EQUIPMENT_LIMITS[f][0], EQUIPMENT_LIMITS[f][1]] for f in self.features},
            "support_metric": "manhattan distance in standardised feature space to the k-th nearest training point",
            "k": self.k,
            "support_threshold": self.support_threshold,
            "support_threshold_rule": f"{SUPPORT_QUANTILE:.0%} quantile of the training points' own k-th neighbour distances",
            "n_training_points": int(len(self.training_points)),
        }

    def save(self, path):
        np.savez(path, features=np.array(self.features), lower=self.lower, upper=self.upper,
                 scale_mean=self.scale_mean, scale_std=self.scale_std,
                 training_points=self.training_points, k=self.k, support_threshold=self.support_threshold)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls([str(f) for f in z["features"]], z["lower"], z["upper"], z["scale_mean"], z["scale_std"],
                       z["training_points"], int(z["k"]), float(z["support_threshold"]))
