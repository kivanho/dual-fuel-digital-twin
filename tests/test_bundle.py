import json
import os

import numpy as np
import pytest

from app import train as train_cli
from app.bundle import (BUNDLE_FORMAT, EFFICIENCY_FILE, MANIFEST, BundleError, load_bundle, sha256_dataframe)
from app.evaluation import PRODUCTION
from app.synthetic_data import generate_engine_map


def test_training_cli_writes_a_complete_bundle(tmp_path):
    out = tmp_path / "bundle"
    report = tmp_path / "report"
    assert train_cli.main(["--out", str(out), "--report-dir", str(report), "--n-points", "200"]) == 0
    manifest = json.loads((out / MANIFEST).read_text())
    assert manifest["bundle_format"] == BUNDLE_FORMAT
    assert manifest["model_version"].startswith("synthetic-")
    assert manifest["data"]["n_points"] == 200 and manifest["data"]["params"]["seed"] == 42
    assert manifest["data"]["sha256"] == sha256_dataframe(generate_engine_map(200, 42))
    assert manifest["dependencies"]["scikit-learn"]
    assert manifest["provenance"]["command"].startswith("python -m app.train")
    assert manifest["models"]["efficiency"]["features"] == ["power_output", "calculated_ch4_share_percent"]
    assert manifest["models"]["efficiency"]["feature_units"] == ["kW", "vol-%"]
    assert manifest["domain"]["n_training_points"] == 200
    assert manifest["evaluation"]["splits"]["random_holdout"]["efficiency"][PRODUCTION["efficiency"]]["r2"] > 0.8
    assert manifest["data_validation"]["n_rows_out"] == 200
    assert set(manifest["checksums"]) == {"efficiency_model.joblib", "exhaust_model.joblib", "domain.npz"}
    # report files
    rep = json.loads((report / "evaluation.json").read_text())
    assert rep["model_version"] == manifest["model_version"]
    assert "heldout_high_load" in rep["splits"]
    assert "| random_holdout |" in (report / "evaluation.md").read_text()


def test_load_bundle_validates_and_predicts(trained_bundle_dir):
    bundle = load_bundle(trained_bundle_dir)
    assert bundle.version == bundle.manifest["model_version"]
    eff = bundle.efficiency_model.predict(np.array([[12.0, 80.0]]))[0]
    temp = bundle.exhaust_model.predict(np.array([[12.0, 80.0, 20.0]]))[0]
    assert 20 < eff < 35 and 250 < temp < 400


def test_load_bundle_uses_env_var(trained_bundle_dir, monkeypatch):
    monkeypatch.setenv("MODEL_BUNDLE_DIR", trained_bundle_dir)
    assert load_bundle().path == trained_bundle_dir


def test_missing_bundle_gives_actionable_error(tmp_path):
    with pytest.raises(BundleError, match="python -m app.train"):
        load_bundle(tmp_path / "nothing")


def _copy_bundle(src, dst):
    import shutil
    shutil.copytree(src, dst)
    return dst


def test_wrong_format_version_rejected(trained_bundle_dir, tmp_path):
    b = _copy_bundle(trained_bundle_dir, tmp_path / "b")
    m = json.loads((b / MANIFEST).read_text())
    m["bundle_format"] = 99
    (b / MANIFEST).write_text(json.dumps(m))
    with pytest.raises(BundleError, match="format 99"):
        load_bundle(b)


def test_feature_order_mismatch_rejected(trained_bundle_dir, tmp_path):
    b = _copy_bundle(trained_bundle_dir, tmp_path / "b")
    m = json.loads((b / MANIFEST).read_text())
    m["models"]["efficiency"]["features"] = ["calculated_ch4_share_percent", "power_output"]
    (b / MANIFEST).write_text(json.dumps(m))
    with pytest.raises(BundleError, match="trained with features"):
        load_bundle(b)


def test_checksum_mismatch_rejected(trained_bundle_dir, tmp_path):
    b = _copy_bundle(trained_bundle_dir, tmp_path / "b")
    with open(b / EFFICIENCY_FILE, "ab") as f:
        f.write(b"tampered")
    with pytest.raises(BundleError, match="checksum mismatch"):
        load_bundle(b)


def test_sklearn_version_mismatch_rejected_unless_allowed(trained_bundle_dir, tmp_path, monkeypatch):
    b = _copy_bundle(trained_bundle_dir, tmp_path / "b")
    m = json.loads((b / MANIFEST).read_text())
    m["dependencies"]["scikit-learn"] = "0.0.1"
    (b / MANIFEST).write_text(json.dumps(m))
    with pytest.raises(BundleError, match="scikit-learn 0.0.1"):
        load_bundle(b)
    monkeypatch.setenv("MODEL_BUNDLE_ALLOW_VERSION_MISMATCH", "1")
    assert load_bundle(b).version


def test_csv_training_records_file_fingerprint(tmp_path):
    csv = tmp_path / "map.csv"
    generate_engine_map(150, seed=3).to_csv(csv, index=False)
    out = tmp_path / "b"
    assert train_cli.main(["--csv", str(csv), "--out", str(out), "--no-eval"]) == 0
    m = json.loads((out / MANIFEST).read_text())
    assert m["data"]["source"] == "csv" and m["data"]["path"] == str(csv)
    assert len(m["data"]["file_sha256"]) == 64
    assert m["evaluation"] is None
    assert m["model_version"].startswith("csv-")


def test_csv_training_fails_loudly_on_bad_data(tmp_path, capsys):
    csv = tmp_path / "bad.csv"
    df = generate_engine_map(150, seed=3)
    df.loc[0, "exhaust_temp"] = np.nan
    df.to_csv(csv, index=False)
    from app.engine_map import EngineMapError
    with pytest.raises(EngineMapError, match="missing values"):
        train_cli.main(["--csv", str(csv), "--out", str(tmp_path / "b"), "--no-eval"])
    assert not (tmp_path / "b").exists()
