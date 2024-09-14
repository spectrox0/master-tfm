"""
This module contains functions to calculate various evaluation metrics
for given true and predicted values.
"""
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from pandas import DataFrame
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MatrixLike = Union[np.ndarray, DataFrame]


# Custom Accuracy: Calculate how many predictions fall within a defined tolerance
def custom_accuracy(y_true:MatrixLike | ArrayLike, y_pred:MatrixLike | ArrayLike, tolerance:float = 0.05) -> float:
    """
    Calculate the accuracy of the model within a defined tolerance.

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.
    tolerance (float): The tolerance within which a prediction is considered accurate.

    Returns:
    float: The accuracy score as a percentage.
    """

    accurate_predictions = np.abs((y_pred - y_true) / y_true) < tolerance
    return np.mean(accurate_predictions) * 100

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE).

    MAPE is a measure of prediction accuracy of a forecasting method in statistics.
    It expresses accuracy as a percentage, and is calculated as the average of the
    absolute percentage errors of the predictions.

    Parameters:
    y_true (array-like): Array of actual values.
    y_pred (array-like): Array of predicted values.

    Returns:
    float: The MAPE value as a percentage.
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_all_metrics(y_true:MatrixLike | ArrayLike, y_pred:MatrixLike | ArrayLike):
    """
    Calculate various evaluation metrics for given true and predicted values.

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.

    Returns:
    tuple: A tuple containing the following metrics:
        - rmse (float): Root Mean Squared Error.
        - mae (float): Mean Absolute Error.
        - r2 (float): R-squared score.
        - accuracy (float): Accuracy score.
        - precision (float): Precision score.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    accuracy = custom_accuracy(y_true, y_pred)


    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")
    print(f"MAPE: {mape}%")
    print(f"Accuracy: {accuracy}% with a tolerance of 5%")
    return mse, rmse, mae, r2, mape , accuracy
