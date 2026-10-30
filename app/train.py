"""
Train the twin's models and write a versioned model bundle.

    python -m app.train                                  # synthetic map -> models/synthetic
    python -m app.train --out models/synthetic --report-dir docs
    python -m app.train --csv my_map.csv --out models/my_engine
    python -m app.train --no-eval                        # faster, for tests

The bundle (see app/bundle.py) holds both fitted models, the supported operating
domain, the configuration and seed, a fingerprint of the training data, the
evaluation results and provenance. The service never trains; it loads a bundle.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.bundle import DEFAULT_BUNDLE_DIR, sha256_dataframe, sha256_file, write_bundle
from app.domain import DOMAIN_FEATURES, OperatingDomain
from app.engine_map import load_engine_map_csv, validate_engine_map
from app.evaluation import evaluate, to_markdown
from app.exhaust_temp_model import train_exhaust_temp_model
from app.knnr_model import MINKOWSKI_P, N_NEIGHBORS, fit_knnr_model
from app.synthetic_data import DEFAULT_N_POINTS, DEFAULT_SEED, generate_engine_map

TEST_SIZE = 0.2


def train(df, *, data: dict, seed: int = DEFAULT_SEED, evaluate_models: bool = True, data_validation=None,
          out_dir=DEFAULT_BUNDLE_DIR, model_version=None, command=None):
    """Fit both models and the domain on `df`, evaluate, and write the bundle. Returns (manifest_path, results)."""
    efficiency_model = fit_knnr_model(df)
    exhaust_model = train_exhaust_temp_model(df)
    domain = OperatingDomain.fit(df, DOMAIN_FEATURES, k=N_NEIGHBORS)
    results = evaluate(df, seed=seed, test_size=TEST_SIZE) if evaluate_models else None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_version = model_version or f"{data['source']}-{stamp}-{data['sha256'][:8]}"
    config = {"seed": seed, "test_size": TEST_SIZE, "knn": {"n_neighbors": N_NEIGHBORS, "p": MINKOWSKI_P},
              "exhaust_model": "LinearRegression on standardised features", "domain_k": N_NEIGHBORS}
    manifest_path = write_bundle(out_dir, efficiency_model, exhaust_model, domain, model_version=model_version,
                                 data=data, config=config, evaluation=results, data_validation=data_validation,
                                 command=command)
    return manifest_path, results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", help="train on this engine-map CSV instead of the synthetic map")
    parser.add_argument("--out", default=DEFAULT_BUNDLE_DIR, help=f"bundle directory (default {DEFAULT_BUNDLE_DIR})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed for the synthetic map and the hold-out split")
    parser.add_argument("--n-points", type=int, default=DEFAULT_N_POINTS, help="size of the synthetic map")
    parser.add_argument("--missing", choices=["error", "drop"], default="error", help="CSV missing-value policy")
    parser.add_argument("--duplicates", choices=["error", "drop"], default="error", help="CSV duplicate-row policy")
    parser.add_argument("--no-eval", action="store_true", help="skip the evaluation (bundle has no scores)")
    parser.add_argument("--report-dir", help="also write evaluation.json and evaluation.md here")
    parser.add_argument("--version", help="model version string (default: <source>-<timestamp>-<data hash>)")
    args = parser.parse_args(argv)
    command = "python -m app.train " + " ".join(argv if argv is not None else sys.argv[1:])

    if args.csv:
        df, report = load_engine_map_csv(args.csv, missing=args.missing, duplicates=args.duplicates,
                                         n_neighbors=N_NEIGHBORS, test_size=TEST_SIZE)
        data = {"source": "csv", "path": str(args.csv), "file_sha256": sha256_file(args.csv),
                "sha256": sha256_dataframe(df), "n_points": int(len(df))}
    else:
        params = {"generator": "app.synthetic_data.generate_engine_map", "n_points": args.n_points, "seed": args.seed}
        df, report = validate_engine_map(generate_engine_map(args.n_points, args.seed), n_neighbors=N_NEIGHBORS,
                                         test_size=TEST_SIZE)
        data = {"source": "synthetic", "params": params,
                "params_sha256": hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest(),
                "sha256": sha256_dataframe(df), "n_points": int(len(df))}
    for w in report.warnings:
        print(f"warning: {w}")
    print(f"Engine map: {len(df)} points ({data['source']}), fingerprint {data['sha256'][:12]}")

    manifest_path, results = train(df, data=data, seed=args.seed, evaluate_models=not args.no_eval,
                                   data_validation=report.to_dict(), out_dir=args.out, model_version=args.version,
                                   command=command)
    manifest = json.loads(Path(manifest_path).read_text())
    print(f"Wrote bundle {manifest['model_version']} to {args.out}")

    if results:
        rh = results["splits"]["random_holdout"]
        print(f"Random hold-out: efficiency kNN RMSE {rh['efficiency']['knn_2f']['rmse']:.2f} %-pt, "
              f"R² {rh['efficiency']['knn_2f']['r2']:.3f}; exhaust temperature linear RMSE "
              f"{rh['exhaust_temp']['linear_3f']['rmse']:.1f} °C, R² {rh['exhaust_temp']['linear_3f']['r2']:.3f}")
        if args.report_dir:
            rep = Path(args.report_dir)
            rep.mkdir(parents=True, exist_ok=True)
            payload = {"model_version": manifest["model_version"], "data": data, "command": command, **results}
            (rep / "evaluation.json").write_text(json.dumps(payload, indent=2) + "\n")
            title = f"Model evaluation on the {data['source']} engine map ({len(df)} points, seed {args.seed})"
            (rep / "evaluation.md").write_text(to_markdown(results, title, command) + "\n")
            print(f"Wrote {rep / 'evaluation.json'} and {rep / 'evaluation.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
