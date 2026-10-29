import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

FEATURES = ['power_output', 'calculated_ch4_share_percent']
TARGET = 'efficiency_electric'


def build_knnr_model():
    """kNN regression with k=10 and Manhattan distance on standardised features."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor(n_neighbors=10, p=1))
    ])


def train_knnr_model(df):
    """
    Trains the KNNR model using power output and CH₄ %, evaluates on a held-out test set.

    Returns:
        model, metrics dict with RMSE and R² on the test set
    """
    df_clean = df[FEATURES + [TARGET]].dropna()
    X = df_clean[FEATURES].values
    y = df_clean[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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


def predict_efficiency_knnr(model, df, target_power_output, target_ch4_percent):
    """
    Predicts efficiency and compares with the closest engine-map point (in scaled feature space).

    Returns:
        Dictionary with prediction, closest engine-map values, and difference.
    """
    df_clean = df[FEATURES + [TARGET]].dropna()

    # Prepare input
    X_input = np.array([[target_power_output, target_ch4_percent]])
    predicted_eff = model.predict(X_input)[0]

    # Standardize for correct distance comparison
    scaler = model.named_steps['scaler']
    X_full_scaled = scaler.transform(df_clean[FEATURES].values)
    X_input_scaled = scaler.transform(X_input)

    # Find closest point (Euclidean in scaled space)
    distances = np.linalg.norm(X_full_scaled - X_input_scaled, axis=1)
    closest_row = df_clean.iloc[distances.argmin()]

    real_eff = closest_row[TARGET]

    return {
        "target_power_output": round(target_power_output, 2),
        "target_ch4_percent": round(target_ch4_percent, 2),
        "predicted_efficiency": round(predicted_eff, 2),
        "closest_measured_power": round(closest_row['power_output'], 2),
        "closest_measured_ch4": round(closest_row['calculated_ch4_share_percent'], 2),
        "measured_efficiency": round(real_eff, 2),
        "difference": round(predicted_eff - real_eff, 2),
    }
