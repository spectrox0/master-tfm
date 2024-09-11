import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX


def main():
    # Load the dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, '../Data_processing/processed_data_final.csv')
    df = pd.read_csv(file_path)

    # Convert 'utc_timestamp' to datetime and set as index
    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'], utc=True)
    df.set_index('utc_timestamp', inplace=True)

    # Fill missing values forward
    df['DE_load_actual_entsoe_transparency'] = df['DE_load_actual_entsoe_transparency'].ffill()

    # Target variable and scaling
    target_variable = 'DE_load_actual_entsoe_transparency'
    series = df[target_variable].dropna()

    # Standard scaling of the target variable
    scaler = StandardScaler()
    scaled_series = scaler.fit_transform(series.values.reshape(-1, 1))
    scaled_series = pd.Series(scaled_series.flatten(), index=series.index)

    # Subset for training
    subset_start_time = pd.to_datetime('2015-01-01 00:00:00').tz_localize('UTC')
    subset_end_time = pd.to_datetime('2016-01-01 00:00:00').tz_localize('UTC')
    scaled_series = scaled_series[subset_start_time:subset_end_time]

    # Define prediction time range
    start_time = pd.to_datetime('2015-06-01 00:00:00').tz_localize('UTC')
    end_time = pd.to_datetime('2016-01-01 00:00:00').tz_localize('UTC')

    # Ensure monotonicity and frequency setting
    train_data = scaled_series[:start_time].asfreq('h', method='ffill')
    test_data = scaled_series[start_time:].asfreq('h', method='ffill')

    # Check if test data is empty
    if test_data.empty:
        raise ValueError("The test data is empty. Check the subset ranges.")

    # SARIMAX Model Wrapper for GridSearchCV
    class SARIMAXModelCV:
        def __init__(self, order=(1, 1, 1), seasonal_order=(1, 0, 1, 24)):
            self.order = order
            self.seasonal_order = seasonal_order
            self.model = None

        def fit(self, X, y=None):
            self.model = SARIMAX(X, order=self.order, seasonal_order=self.seasonal_order,
                                 enforce_stationarity=False, enforce_invertibility=False)
            self.results = self.model.fit(disp=False)
            return self

        def predict(self, X):
            if X.empty:
                raise ValueError("The data for prediction is empty.")
            start = X.index[0]
            end = X.index[-1]
            return self.results.get_prediction(start=start, end=end).predicted_mean

        def get_params(self, deep=True):
            return {"order": self.order, "seasonal_order": self.seasonal_order}

        def set_params(self, **params):
            for key, value in params.items():
                setattr(self, key, value)
            return self

    # Custom scoring function for SARIMAX
    def sarimax_scorer(estimator, X_test, y_true):
        y_pred = estimator.predict(X_test)
        y_true_transformed = scaler.inverse_transform(y_true.values.reshape(-1, 1))
        y_pred_transformed = scaler.inverse_transform(y_pred.values.reshape(-1, 1))
        return mean_squared_error(y_true_transformed, y_pred_transformed)

    # Create a custom scorer for GridSearchCV
    sarimax_scorer_mse = make_scorer(sarimax_scorer, greater_is_better=False)

    # Parameter grid for GridSearchCV with expanded search space
    param_grid = {
        'order': [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2), (3, 1, 1)],  # More combinations
        'seasonal_order': [(1, 0, 1, 24), (1, 1, 1, 24), (1, 0, 2, 24), (2, 1, 1, 24)]
    }

    # Initialize GridSearchCV
    sarimax_grid = GridSearchCV(SARIMAXModelCV(), param_grid, cv=3, scoring=sarimax_scorer_mse, verbose=1)

    # Fit GridSearchCV
    sarimax_grid.fit(train_data, train_data)

    # Best model parameters from GridSearchCV
    best_params = sarimax_grid.best_params_
    print(f"Mejores parámetros SARIMAX: {best_params}")

    # Fit the best model using the best parameters
    best_model = SARIMAXModelCV(order=best_params['order'], seasonal_order=best_params['seasonal_order'])
    best_model.fit(train_data)

    # Make predictions using the best model
    predictions = best_model.predict(test_data)

    # Ensure predicted and actual values have the same length
    predicted_values = predictions[:len(test_data)]
    true_values = test_data

    # Inverse transform to original scale
    predicted_values = scaler.inverse_transform(predicted_values.values.reshape(-1, 1))
    true_values = scaler.inverse_transform(true_values.values.reshape(-1, 1))

    # Calculate error metrics
    mse = mean_squared_error(true_values, predicted_values)
    print('Optimized Mean Squared Error:', mse)

    mae = mean_absolute_error(true_values, predicted_values)
    print('Mean Absolute Error:', mae)

    # Plot predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.plot(test_data.index, predicted_values, label='Predicciones')
    plt.plot(test_data.index, true_values, label='Valores Reales')
    plt.title('Predicciones vs Valores Reales (Optimizado)')
    plt.xlabel('Tiempo')
    plt.ylabel('Demanda de energía')
    plt.legend()
    plt.show()

    # Decompose the series to see the trend, seasonality, and residuals
    result = seasonal_decompose(series, model='multiplicative', period=24*7)  # Assuming weekly seasonality
    result.plot()
    plt.show()

    # Plot residuals
    residuals = true_values.flatten() - predicted_values.flatten()
    plt.figure(figsize=(10, 6))
    plt.plot(test_data.index, residuals)
    plt.title('Residuals (Optimized)')
    plt.show()

    # Plot histogram of residuals
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30)
    plt.title('Distribution of Residuals (Optimized)')
    plt.show()


if __name__ == "__main__":
    main()