"""
This module contains utility functions for loading datasets.
"""
import os

import pandas as pd
from pandas import DataFrame

DEFAULT_FILE_NAME = "time_series_60min_singleindex.csv"
def load_time_series_60min(file_name=DEFAULT_FILE_NAME) -> DataFrame:
    """
    Loads a time series dataset with 60-minute intervals from a CSV file.

    The function constructs the file path relative to the script's directory
    and reads the CSV file located at '../datasets/time_series_60min_singleindex.csv'.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the time series data.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, f'../datasets/{file_name}')
    return pd.read_csv(file_path)
