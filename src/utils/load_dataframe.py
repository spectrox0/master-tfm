from enum import Enum
from typing import List, Type

import pandas as pd
from pandas import DataFrame

file_path_15 = '../../datasets/opsd-time_series-2020-10-06/time_series_15min_singleindex.csv'
file_path_30 = '../../datasets/opsd-time_series-2020-10-06/time_series_30min_singleindex.csv'
file_path_60 = '../../datasets/opsd-time_series-2020-10-06/time_series_60min_singleindex.csv'

# Define Enum ['15min', '30min', '60min']

class TimeGranularity(str, Enum):
    """
    Enumeration class representing different time granularities.

    Attributes:
        MIN_15 (str): Represents a time granularity of 15 minutes.
        MIN_30 (str): Represents a time granularity of 30 minutes.
        MIN_60 (str): Represents a time granularity of 60 minutes.
    """
    MIN_15 = '15min'
    MIN_30 = '30min'
    MIN_60 = '60min'

dict_file_path = {
    TimeGranularity.MIN_15: file_path_15,
    TimeGranularity.MIN_30: file_path_30,
    TimeGranularity.MIN_60: file_path_60
}



def load_dataframe(sheet_name: TimeGranularity = TimeGranularity.MIN_15) -> Type[DataFrame]:
    """
    Load a pandas DataFrame from an Excel file.

    Parameters:
        sheet_name (Type[TimeGranularity]): The name of the sheet to load from the Excel file.
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    df:Type[pd.DataFrame] = pd.read_csv(dict_file_path[sheet_name])
    return df