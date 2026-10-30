"""
Exhaust-temperature model: linear regression on standardised features
(power, CH4 share, diesel energy share).
"""
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FEATURES = ['power_output', 'calculated_ch4_share_percent', 'des_percent']
TARGET = 'exhaust_temp'


def build_exhaust_temp_model():
    return Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('regressor', LinearRegression())
    ])


def train_exhaust_temp_model(df, features=FEATURES, target=TARGET):
    """
    Fit the linear exhaust-gas-temperature model on all rows of `df`.
    """
    df_clean = df[list(features) + [target]].dropna()
    model = build_exhaust_temp_model()
    model.fit(df_clean[list(features)].values, df_clean[target].values)
    return model
