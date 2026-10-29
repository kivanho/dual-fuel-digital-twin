"""
Train both models on the engine map and print their test-set metrics.

    python -m scripts.train_models                      # synthetic engine map
    python -m scripts.train_models --export map.csv     # also write the map to CSV
    ENGINE_MAP_CSV=my_map.csv python -m scripts.train_models
"""
import argparse

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from app.exhaust_temp_model import FEATURES as TEMP_FEATURES, TARGET as TEMP_TARGET, train_exhaust_temp_model
from app.knnr_model import train_knnr_model
from app.synthetic_data import load_engine_map


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", metavar="CSV", help="write the engine map to this CSV file")
    args = parser.parse_args()

    df = load_engine_map()
    print(f"Engine map: {len(df)} points")

    _, knnr_metrics = train_knnr_model(df)
    print(f"Efficiency (kNN, k=10, Manhattan): RMSE {knnr_metrics['rmse']:.2f} %-pt, R² {knnr_metrics['r2']:.3f}")

    train, test = train_test_split(df, test_size=0.2, random_state=42)
    temp_model = train_exhaust_temp_model(train)
    y_pred = temp_model.predict(test[TEMP_FEATURES].values)
    rmse = np.sqrt(mean_squared_error(test[TEMP_TARGET], y_pred))
    print(f"Exhaust temperature (linear): RMSE {rmse:.1f} °C, R² {r2_score(test[TEMP_TARGET], y_pred):.3f}")

    if args.export:
        df.to_csv(args.export, index=False)
        print(f"Wrote {args.export}")


if __name__ == "__main__":
    main()
