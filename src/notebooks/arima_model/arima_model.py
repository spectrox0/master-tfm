import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from src.utils.load_dataframe import load_time_series_60min


# Test stationarity using Augmented Dickey-Fuller test
def test_stationarity(timeseries):
    result = adfuller(timeseries)
    return result[1] <= 0.05  # Stationary if p-value <= 0.05

# Wrapper class for ARIMA to use with GridSearchCV
class ARIMAWrapped(BaseEstimator):
    def __init__(self, p=1, d=1, q=1):
        self.p = p
        self.d = d
        self.q = q
        self.model = None

    def fit(self, X, y=None):
        # Catch convergence warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.model = ARIMA(X, order=(self.p, self.d, self.q)).fit()
        return self

    def predict(self, X):
        return self.model.forecast(steps=len(X))

    def get_params(self, deep=True):
        return {"p": self.p, "d": self.d, "q": self.q}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

# Scorer function for GridSearchCV
def arima_scorer(estimator, X_train, y_true):
    y_pred = estimator.predict(y_true)
    return mean_squared_error(y_true, y_pred)

# Main function
def main():
    warnings.filterwarnings("ignore")

    df = load_time_series_60min()
    df.head()

    # Convert 'utc_timestamp' to datetime and set as index
    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'], utc=True)
    df.set_index('utc_timestamp', inplace=True)

    # Fill missing values
    df['DE_load_actual_entsoe_transparency'] = df['DE_load_actual_entsoe_transparency'].ffill()

    # Target variable
    target_variable = 'DE_load_actual_entsoe_transparency'
    series = df[target_variable].dropna()

    # Check stationarity
    if not test_stationarity(series):
        series = series.diff().dropna()

    # Scaling data using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

    # Train/test split
    train_data = scaled_series[:int(0.8 * len(scaled_series))]
    test_data = scaled_series[int(0.8 * len(scaled_series)):]

    # Ensure test data is not empty
    if len(test_data) == 0:
        raise ValueError("The test data is empty. Check the subset ranges.")

    # Perform Grid Search to optimize ARIMA hyperparameters
    param_grid = {
        'p': [0, 1, 2, 3],
        'd': [0, 1],
        'q': [0, 1, 2, 3]
    }

    arima_model = ARIMAWrapped()
    grid_search = GridSearchCV(
        estimator=arima_model,
        param_grid=param_grid,
        scoring=make_scorer(arima_scorer, greater_is_better=False),
        cv=TimeSeriesSplit(n_splits=3),
        n_jobs=4
    )

    # Fit the GridSearchCV
    grid_search.fit(train_data, train_data)

    # Output best parameters
    print(f"Best Parameters: {grid_search.best_params_}")

    # Fit ARIMA model with best parameters
    best_model = ARIMAWrapped(**grid_search.best_params_)
    best_model.fit(train_data)

    # Make predictions
    predicted_values = best_model.predict(test_data)

    # Inverse transform to original scale
    predicted_values = scaler.inverse_transform(predicted_values.reshape(-1, 1)).flatten()
    true_values = scaler.inverse_transform(test_data.reshape(-1, 1)).flatten()

    # Calculate error metrics
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    print(f'Optimized Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')

    # Plot predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label='True')
    plt.plot(predicted_values, label='Predicted')
    plt.legend()
    plt.title('True vs Predicted')
    plt.show()


if __name__ == "__main__":
    main()