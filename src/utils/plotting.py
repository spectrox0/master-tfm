import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import ndarray
from pandas import DataFrame, DatetimeIndex


def plot_loss_evolution(history) -> None:
    """
    Plot the loss evolution during the training of the model.
    Parameters:
    history (History): The history object from model training,
    which contains the loss and validation loss over epochs.
    """

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Evolution During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_predictions(y_test: DataFrame, y_pred: DataFrame, test_dates: DatetimeIndex, title ='Energy Demand Prediction' ) -> None:
    """
    Plot the actual vs predicted values.
    Parameters:
    y_test (pd.DataFrame): The real values of the target (actual demand).
    y_pred (pd.DataFrame): The predicted values from the model.
    test_dates (pd.DatetimeIndex): The corresponding dates for the test set.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test, color='blue', label='Actual Demand')
    plt.plot(test_dates, y_pred, color='red', label='Predicted Demand')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Energy Demand (MW)')
    plt.legend()

    # Format the X-axis to properly show the dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_monthly_predictions(monthly_predictions: DataFrame) -> None:
    """
    Plot the monthly predictions for energy demand.
    Parameters:
    monthly_predictions (pd.DataFrame): DataFrame with predicted monthly demand.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_predictions.index, monthly_predictions['predictions'], color='blue', marker='o', label='Monthly Demand Predictions')
    plt.title('Monthly Energy Demand Predictions for 2020')
    plt.xlabel('Time')
    plt.ylabel('Energy Demand (MW)')
    plt.legend()

    # Format the X-axis to show the months
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_monthly_comparison(monthly_predictions: DataFrame, real_monthly_demand: DataFrame) -> None:
    """
    Plot the monthly predictions versus the real demand.
    Parameters:
    monthly_predictions (pd.DataFrame): DataFrame with predicted monthly demand.
    real_monthly_demand (pd.DataFrame): DataFrame with actual monthly demand.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_predictions.index, monthly_predictions['predictions'],
            color='red', marker='o', label='Monthly Predictions')

    plt.plot(real_monthly_demand.index, real_monthly_demand['real_demand'],
            color='blue', marker='o', label='Real Monthly Demand')
    plt.title('Monthly Predictions vs Real Demand (2020)')
    plt.xlabel('Time')
    plt.ylabel('Energy Demand (MW)')
    plt.legend()

    # Format x-axis to show months
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# src/utils/plotting.py

def plot_real_vs_predicted(test_dates: DatetimeIndex, y_test: ndarray, y_pred: ndarray) -> None:
    """
    Plot the real vs predicted energy demand with proper dates on the X-axis.
    Parameters:
    test_dates (pd.DatetimeIndex): The dates corresponding to the test data.
    y_test (np.ndarray): Real values of the energy demand.
    y_pred (np.ndarray): Predicted values of the energy demand.
    """
    plt.subplot(2, 1, 1)
    plt.plot(test_dates, y_test, color='blue', label='Real Energy Demand')
    plt.plot(test_dates, y_pred, color='red', label='Predicted Energy Demand')
    plt.title('Energy Demand Prediction')
    plt.xlabel('Date')
    plt.ylabel('Energy Demand (MW)')
    plt.legend()

    # Format the X-axis to properly show the dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

    plt.xticks(rotation=45)



def plot_residuals(test_dates: DatetimeIndex, residuals: ndarray) -> None:
    """
    Plot the residuals between real and predicted energy demand with proper dates on the X-axis.
    Parameters:
    test_dates (pd.DatetimeIndex): The dates corresponding to the test data.
    residuals (np.ndarray): The residuals (differences between real and predicted values).
    """
    plt.subplot(2, 1, 2)
    plt.plot(test_dates, residuals, color='green', label='Residuals')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title('Residuals')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.legend()

    # Format the X-axis to properly show the dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

    plt.xticks(rotation=45)
    plt.tight_layout()


def plot_correlation_matrix(correlation_matrix: DataFrame) -> None:
    """
    Plot the correlation matrix of the given DataFrame.
    Parameters:
    df (pd.DataFrame): The DataFrame for which to plot the correlation matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True)
    plt.title('Correlation Matrix')
    plt.show()