import sys
sys.path.append("/home/ec2-user/SageMaker/")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib.ticker import MultipleLocator
from datetime import datetime, timedelta, date
import datetime
from dateutil.relativedelta import relativedelta, MO

from utils import Athena_Query, s3, LabelStore
from utils.sql_query import SqlQuery
from IPython.display import clear_output, HTML
from scipy.interpolate import CubicSpline
from utils.waveform_viewer2 import Waveform_Extract, Waveform_Helper


wh = Waveform_Helper()
athena = Athena_Query()


def fetch_ecg_data_first_24_hours(df, row_index):
    """
    Fetch ECG data for a specific patient based on their row index in the DataFrame, limited to the first 24 hours from 'fromtime'.

    Parameters:
    - df: pandas DataFrame containing patient details.
    - row_index: Integer index of the patient's row in the DataFrame.

    Returns:
    - ecg_data: DataFrame containing the ECG data for the specified patient within the first 24 hours from 'fromtime'.
    """
    
    df['fromtime'] = pd.to_datetime(df['fromtime'])
    patient_row = df.iloc[row_index]
    totime = patient_row['fromtime'] + pd.Timedelta(hours=24)
    from_year, from_month, from_day = patient_row['fromtime'].year, patient_row['fromtime'].month, patient_row['fromtime'].day
    to_year, to_month, to_day = totime.year, totime.month, totime.day
    
    query = f"""
        SELECT timestamp, "values" FROM "waveform"."waveform-new"
        WHERE hospital = 'GCUH'
        AND unit = 'ICU'
        AND room = '-'
        AND bed = '{patient_row['bedname']}'
        AND year BETWEEN {from_year} AND {to_year}
        AND month BETWEEN {from_month} AND {to_month}
        AND day BETWEEN {from_day} AND {to_day}
        AND observation_type = 'ECG'
        AND observation_subtype = 'II'
        AND timestamp BETWEEN TIMESTAMP '{patient_row['fromtime']}' AND TIMESTAMP '{totime}'
        ORDER BY timestamp;
    """
    athena = Athena_Query()
    print(query)  # Just printing the query for demonstration
    ecg_data = athena.query_as_pandas(query)  # Replace with actual data fetching
    
    return ecg_data
