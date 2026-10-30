"""
Versioned model bundle: what the training command writes and the service loads.

Layout of a bundle directory

    manifest.json           schema/version, features, units, domain, config, data fingerprint,
                            evaluation results, dependency versions, provenance, file checksums
    efficiency_model.joblib sklearn Pipeline (StandardScaler + KNeighborsRegressor)
    exhaust_model.joblib    sklearn Pipeline (StandardScaler + LinearRegression)
    domain.npz              per-feature range, scaler statistics, training points, support threshold

Trust requirement: the .joblib files are pickles. Loading a pickle executes
code chosen by whoever wrote it, so only load bundles you produced yourself or
obtained from a trusted source. The manifest carries SHA-256 checksums of the
model files and `load_bundle` refuses a bundle whose files do not match them,
which detects corruption and accidental mix-ups, not a malicious author.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn

from app.domain import DOMAIN_FEATURES, OperatingDomain
from app.engine_map import UNITS
from app.exhaust_temp_model import FEATURES as TEMP_FEATURES, TARGET as TEMP_TARGET
from app.knnr_model import FEATURES as EFF_FEATURES, TARGET as EFF_TARGET

BUNDLE_FORMAT = 1
MANIFEST = "manifest.json"
EFFICIENCY_FILE = "efficiency_model.joblib"
EXHAUST_FILE = "exhaust_model.joblib"
DOMAIN_FILE = "domain.npz"
DEFAULT_BUNDLE_DIR = "models/synthetic"
ENV_BUNDLE_DIR = "MODEL_BUNDLE_DIR"
ENV_ALLOW_VERSION_MISMATCH = "MODEL_BUNDLE_ALLOW_VERSION_MISMATCH"

EXPECTED_FEATURES = {"efficiency": list(EFF_FEATURES), "exhaust_temp": list(TEMP_FEATURES), "domain": list(DOMAIN_FEATURES)}


class BundleError(RuntimeError):
    """The bundle is missing, incomplete, of another format, or inconsistent with this code."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dataframe(df: pd.DataFrame) -> str:
    """Fingerprint of the exact training table (column order and float text are deterministic)."""
    return hashlib.sha256(df.to_csv(index=False, float_format="%.12g").encode()).hexdigest()


def git_commit() -> Optional[str]:
    try:
        root = Path(__file__).resolve().parent.parent
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def dependency_versions() -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


@dataclass
class ModelBundle:
    efficiency_model: object
    exhaust_model: object
    domain: OperatingDomain
    manifest: dict
    path: str

    @property
    def version(self) -> str:
        return self.manifest["model_version"]


def write_bundle(out_dir, efficiency_model, exhaust_model, domain: OperatingDomain, *, model_version: str,
                 data: dict, config: dict, evaluation: Optional[dict], data_validation: Optional[dict] = None,
                 command: Optional[str] = None) -> str:
    """Write a bundle directory and return the manifest path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(efficiency_model, out / EFFICIENCY_FILE)
    joblib.dump(exhaust_model, out / EXHAUST_FILE)
    domain.save(out / DOMAIN_FILE)
    manifest = {
        "bundle_format": BUNDLE_FORMAT,
        "model_version": model_version,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": {"git_commit": git_commit(), "command": command or " ".join(sys.argv),
                       "platform": platform.platform()},
        "dependencies": dependency_versions(),
        "models": {
            "efficiency": {"file": EFFICIENCY_FILE, "type": type(efficiency_model.steps[-1][1]).__name__,
                           "features": list(EFF_FEATURES), "target": EFF_TARGET,
                           "feature_units": [UNITS[f] for f in EFF_FEATURES], "target_unit": UNITS[EFF_TARGET],
                           "preprocessing": "StandardScaler (inside the pipeline)"},
            "exhaust_temp": {"file": EXHAUST_FILE, "type": type(exhaust_model.steps[-1][1]).__name__,
                             "features": list(TEMP_FEATURES), "target": TEMP_TARGET,
                             "feature_units": [UNITS[f] for f in TEMP_FEATURES], "target_unit": UNITS[TEMP_TARGET],
                             "preprocessing": "StandardScaler (inside the pipeline)"},
        },
        "domain": {"file": DOMAIN_FILE, **domain.describe()},
        "config": config,
        "data": data,
        "data_validation": data_validation,
        "evaluation": evaluation,
        "checksums": {name: sha256_file(out / name) for name in (EFFICIENCY_FILE, EXHAUST_FILE, DOMAIN_FILE)},
    }
    manifest_path = out / MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=True) + "\n")
    return str(manifest_path)


def _check(cond, message):
    if not cond:
        raise BundleError(message)


def load_bundle(bundle_dir=None) -> ModelBundle:
    """Load and validate a bundle; raises BundleError with the reason if it cannot be used."""
    bundle_dir = bundle_dir or os.environ.get(ENV_BUNDLE_DIR) or DEFAULT_BUNDLE_DIR
    path = Path(bundle_dir)
    _check(path.is_dir(), f"model bundle directory not found: {path}. Train one with "
                          f"`python -m app.train --out {path}` or set {ENV_BUNDLE_DIR}.")
    manifest_path = path / MANIFEST
    _check(manifest_path.is_file(), f"{path} has no {MANIFEST}; is it a model bundle?")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        raise BundleError(f"{manifest_path} is not valid JSON: {e}")

    _check(manifest.get("bundle_format") == BUNDLE_FORMAT,
           f"bundle format {manifest.get('bundle_format')!r} not supported by this code (expects {BUNDLE_FORMAT}); retrain.")
    for key in ("model_version", "models", "domain", "checksums", "dependencies"):
        _check(key in manifest, f"manifest is missing '{key}'")

    for name, expected in (("efficiency", EXPECTED_FEATURES["efficiency"]), ("exhaust_temp", EXPECTED_FEATURES["exhaust_temp"])):
        got = manifest["models"].get(name, {}).get("features")
        _check(got == expected, f"{name} model was trained with features {got}, this code expects {expected}")
    _check(manifest["domain"].get("features") == EXPECTED_FEATURES["domain"],
           f"domain features {manifest['domain'].get('features')} differ from {EXPECTED_FEATURES['domain']}")

    for name, digest in manifest["checksums"].items():
        file = path / name
        _check(file.is_file(), f"bundle file missing: {file}")
        actual = sha256_file(file)
        _check(actual == digest, f"checksum mismatch for {file}: manifest {digest[:12]}..., file {actual[:12]}...")

    trained_with = manifest["dependencies"].get("scikit-learn")
    if trained_with != sklearn.__version__ and not os.environ.get(ENV_ALLOW_VERSION_MISMATCH):
        raise BundleError(f"bundle was trained with scikit-learn {trained_with}, running {sklearn.__version__}. "
                          f"Pickled estimators are not portable across versions: retrain, or set "
                          f"{ENV_ALLOW_VERSION_MISMATCH}=1 to load anyway.")

    efficiency_model = joblib.load(path / EFFICIENCY_FILE)
    exhaust_model = joblib.load(path / EXHAUST_FILE)
    domain = OperatingDomain.load(path / DOMAIN_FILE)
    _check(hasattr(efficiency_model, "predict") and hasattr(exhaust_model, "predict"), "model files are not estimators")

    # Smoke prediction at the centre of the training box must be finite
    centre = (domain.lower + domain.upper) / 2
    eff_idx = [domain.features.index(f) for f in EFF_FEATURES]
    temp_idx = [domain.features.index(f) for f in TEMP_FEATURES]
    eff = efficiency_model.predict(centre[eff_idx].reshape(1, -1))[0]
    temp = exhaust_model.predict(centre[temp_idx].reshape(1, -1))[0]
    _check(np.isfinite(eff) and np.isfinite(temp), "models return non-finite predictions at the domain centre")

    return ModelBundle(efficiency_model, exhaust_model, domain, manifest, str(path))
