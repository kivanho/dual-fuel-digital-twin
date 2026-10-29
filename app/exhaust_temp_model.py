from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FEATURES = ['power_output', 'calculated_ch4_share_percent', 'des_percent']
TARGET = 'exhaust_temp'


def train_exhaust_temp_model(df):
    """
    Trains a simple linear regression model to predict exhaust gas temperature
    based on power output, CH₄ share, and diesel energy share.
    """
    df_clean = df[FEATURES + [TARGET]].dropna()
    X = df_clean[FEATURES].values
    y = df_clean[TARGET].values

    # Create a pipeline with scaling and linear regression
    model = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('regressor', LinearRegression())
    ])

    model.fit(X, y)

    return model
