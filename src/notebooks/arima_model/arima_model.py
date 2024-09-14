# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import boxcox
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from src.utils.load_dataframe import load_time_series_60min


# Function to reverse Box-Cox transformation
def inv_boxcox(y, lam):
    return np.exp(np.log(lam * y + 1) / lam) if lam != 0 else np.exp(y)

def main():
    # Load the CSV
    df = load_time_series_60min()
    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
    df.set_index('utc_timestamp', inplace=True)

    exogenous_variables = [
        'DE_solar_generation_actual',    # Solar generation for Germany
        'DE_wind_onshore_generation_actual',  # Wind generation for Germany
        'FR_load_actual_entsoe_transparency',  # Load from France
        'NL_load_actual_entsoe_transparency',  # Load from Netherlands
        'AT_price_day_ahead'  # Price day ahead for Austria (as a proxy for price)
    ]

    # Select the target variable and the exogenous variables, dropping rows with missing values
    df = df[['DE_load_actual_entsoe_transparency'] + exogenous_variables].dropna()

    # Define the target variable and exogenous variables
    y = df['DE_load_actual_entsoe_transparency']
    X = df[exogenous_variables]

    # 2. Check for stationarity with ADF test and apply Box-Cox if necessary
    adf_result = adfuller(y)
    print(f"ADF Statistic: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")
    if adf_result[1] > 0.05:
        print("The series is non-stationary, applying Box-Cox transformation...")
        y_transformed, lam = boxcox(y)  # Apply Box-Cox transformation
    else:
        y_transformed = y  # No transformation needed

    # Visualize ACF and PACF for the target variable
    plt.figure(figsize=(10, 6))
    plot_acf(y_transformed, lags=40)
    plt.show()

    plt.figure(figsize=(10, 6))
    plot_pacf(y_transformed, lags=40)
    plt.show()

    # 3. Scaling (optional): Scale the exogenous variables if necessary
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Only scaling the exogenous variables

    # Split the data into training and testing sets (80% training, 20% testing)
    train_size = int(len(df) * 0.8)
    train_y, test_y = y_transformed[:train_size], y_transformed[train_size:]
    train_X, test_X = X_scaled[:train_size], X_scaled[train_size:]

    # 4. Define the range of hyperparameters to optimize
    p = q = range(0, 3)  # AR and MA parameters
    d = range(0, 2)  # Differentiation

   # Seasonal parameters (for yearly seasonality)
    P = Q = range(0, 3)
    D = range(0, 2)
    S = [24]  # Daily seasonality (24 hours)

    # Generate combinations of parameters
    param_grid = [
        ((p1, d1, q1), (P1, D1, Q1, S1))
        for p1 in p for d1 in d for q1 in q
        for P1 in P for D1 in D for Q1 in Q for S1 in S
    ]

    # 5. Train the model with simple grid search and TimeSeriesSplit
    best_aic = np.inf
    best_params = None
    best_model = None

    for param in param_grid:
        try:
            model = SARIMAX(train_y, exog=train_X, order=param[0], seasonal_order=param[1])
            result = model.fit(disp=False)
            if result.aic < best_aic:
                best_aic = result.aic
                best_params = param
                best_model = result
        except:
            continue

    # Show the best parameters
    print(f"Best parameters: {best_params}")
    print(f"Best model AIC: {best_aic}")
    
    # 6. Evaluate the model on the test set
    predictions = best_model.predict(start=test_y.index[0], end=test_y.index[-1], exog=test_X)

    # Reverse Box-Cox transformation if applied
    if 'lam' in locals():
        predictions = np.exp(np.log(predictions * lam + 1) / lam)  # Reverse transformation
        test_y = np.exp(np.log(test_y * lam + 1) / lam)

    # 7. Calculate metrics: RMSE, MAE, R2
    rmse = np.sqrt(mean_squared_error(test_y, predictions))
    mae = mean_absolute_error(test_y, predictions)
    r2 = r2_score(test_y, predictions)
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")

    # 8. Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(test_y, label='Actual')
    plt.plot(predictions, label='Prediction', linestyle='--')
    plt.title('Prediction vs Actual - SARIMAX Model')
    plt.xlabel('Date')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()