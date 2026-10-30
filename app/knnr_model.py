"""
Efficiency model: k-nearest-neighbour regression on standardised features.

Production features are power and CH4 share. Diesel energy share is deliberately
not a feature here; `app/evaluation.py` quantifies that choice (kNN with and
without DES) instead of assuming it.
"""
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

FEATURES = ['power_output', 'calculated_ch4_share_percent']
TARGET = 'efficiency_electric'
N_NEIGHBORS = 10
MINKOWSKI_P = 1  # Manhattan distance


def build_knnr_model(n_neighbors=N_NEIGHBORS, p=MINKOWSKI_P):
    """kNN regression with k=10 and Manhattan distance on standardised features."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor(n_neighbors=n_neighbors, p=p))
    ])


def fit_knnr_model(df, features=FEATURES, target=TARGET, **kwargs):
    """Fit the efficiency model on all rows of `df` (no evaluation)."""
    model = build_knnr_model(**kwargs)
    model.fit(df[list(features)].values, df[target].values)
    return model


def train_knnr_model(df, test_size=0.2, random_state=42):
    """
    Score the kNN model on a random hold-out, then refit on the whole map.

    Kept for scripts that want a quick interpolation score; the training CLI
    (`python -m app.train`) uses app/evaluation.py for the full comparison.

    Returns:
        model, metrics dict with RMSE and R² on the hold-out
    """
    df_clean = df[FEATURES + [TARGET]].dropna()
    X = df_clean[FEATURES].values
    y = df_clean[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = build_knnr_model()
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "r2": float(r2_score(y_test, y_pred_test)),
    }

    # Refit on all points so the twin uses the full engine map
    model.fit(X, y)
    return model, metrics


def predict_efficiency(model, power_output, ch4_percent):
    """Predicted electrical efficiency [%] at one operating point."""
    return float(model.predict(np.array([[power_output, ch4_percent]]))[0])
