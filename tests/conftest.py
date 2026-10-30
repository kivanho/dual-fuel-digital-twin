import os

import pytest

from app import train as train_cli


@pytest.fixture(scope="session")
def trained_bundle_dir(tmp_path_factory):
    """One synthetic bundle per test session (training is deterministic, evaluation skipped for speed)."""
    out = tmp_path_factory.mktemp("bundle") / "synthetic"
    assert train_cli.main(["--out", str(out), "--no-eval", "--version", "test-bundle"]) == 0
    return str(out)


@pytest.fixture(scope="session", autouse=True)
def _bundle_env(trained_bundle_dir):
    """The API loads the bundle named by MODEL_BUNDLE_DIR at startup (see app/main.py)."""
    previous = os.environ.get("MODEL_BUNDLE_DIR")
    os.environ["MODEL_BUNDLE_DIR"] = trained_bundle_dir
    yield
    if previous is None:
        os.environ.pop("MODEL_BUNDLE_DIR", None)
    else:
        os.environ["MODEL_BUNDLE_DIR"] = previous


@pytest.fixture(scope="session")
def bundle(trained_bundle_dir):
    from app.bundle import load_bundle
    return load_bundle(trained_bundle_dir)
