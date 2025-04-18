# ========================================================
# Stormwater system data acqusition.
# ========================================================

# Library imports.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
import math

# UrbanSurge imports.

def align_measurements(df_ref, df, time_col, corr_col, pulse_length):
    """
    Aligns a data frame time column to a reference data frame based on the lag
    where the mean absolute difference (MAD) is minimized.

    :param df_ref: Reference data frame containing a time column and column to compare.
    :param df: Data frame to shift to the reference data frame.
    :param time_col: Name of the time column in both data frames.
    :param corr_col: Name of the column to compare.
    :param pulse_length: Length of pulse in number of time steps.

    :return t_align: Same as df, but with a shifted time column to match the reference data frame.
    """
    t = df_ref[time_col].to_numpy()
    Nt = len(t)
    dt = t[1] - t[0]

    # Extract relevant data for comparison.
    ref_data = df_ref[corr_col].to_numpy()[:pulse_length]
    working_data = df[corr_col].to_numpy()[:pulse_length]

    # Buffer zone for comparisons.
    buffer_size = 20

    m1s = []
    m2s = []
    for overlap in range(buffer_size, len(ref_data) + 1):
        # 1st.
        ref_1 = ref_data[:overlap]
        work_1 = working_data[-overlap:]
        # 2nd.
        ref_2 = ref_data[-overlap:]
        work_2 = working_data[:overlap]

        # Compute MAD within bounds.
        m1 = np.mean(np.abs(ref_1 - work_1))
        m2 = np.mean(np.abs(ref_2 - work_2))
        m1s.append(m1)
        m2s.append(m2)

    ms = np.concatenate((m1s, np.flip(m2s)))
    min_m = np.argmin(ms)
    shift = buffer_size - len(ref_data) + min_m

    # Compute the shift.
    t_align = np.arange(t[0] + shift * dt, t[-1] + shift * dt, dt)

    # Handle edge cases where t_align ends up one dt off from t.
    if len(t_align) < Nt:
        t_align = np.append(t_align, t_align[-1] + dt)
    elif len(t_align) > Nt:
        t_align = t_align[:Nt]

    return t_align


def V_to_htank(V):
    """
    Convert tank sensor voltage to height (cm).
    """
    return 2.3017*V + 4.3401


def voltage_to_value(V, b0, b1):
    """
    Convert sensor voltage to desired measurement units based on calibration regression.

    :param V: Voltage from sensor.
    :param b0: Intercept.
    :param b1: Slope.

    :return value: Value in desired units.
    """
    return b0 + b1 * V


def approx_dVdt(t,V):
    """
    Approximate dV/dt for tank.

    :param t: Time array.
    :param V: Voltage array.
    """
    dt = t[1]-t[0]
    dVdt = np.zeros(len(V))
    for k in range(1,len(t)-1):
        dVdt[k] = (V[k+1] - V[k-1])/(2*dt)

    return dVdt

def tank_area(): 
    """
    Return value for tank cross-sectional area. 

    :return: Area of the tank for the experimental setup. 
    """
    # Tank measurements
    thickness = 0.125 # inches
    tank_OD = 57.75/np.pi # inches
    tank_ID = (tank_OD - 2*thickness)*2.54 # cm

    return 0.25*np.pi*(tank_ID**2)


def flow_rate(V, t):
    """
    Calculate flow rate from smoothed voltage and time series.

    :param V: Numpy array of voltage readings from tank sensor.
    :param t: Numpy array of corresponding times for voltage readings. 
    """

    dVdt = approx_dVdt(t, V)
    A = tank_area()

    return 2.3017*dVdt*A

def discrete_flow_series(Q: np.ndarray, t: np.ndarray, h = 1): #### chop off after 20 sec
    """
    Convert a continuous flow measurement to a discrete flow series.

    :param Q: Numpy array containing original flow measurements.
    :param t: Numpy array containing corresponding times for Q.
    :param h: Desired timestep for discrete flow series.

    :return: Tuple containing numpy array of discrete flow series, and list of datetime objects.
    """
    # Compute number of points per timestep.
    pts_per_h = math.ceil(h/(t[1] - t[0]))

    # Initialize lists.
    flow_series = []
    k = 0
    dt = []

    # Compute discrete values by averaging within each timestep.
    for i in range(0, len(Q)-pts_per_h, pts_per_h):
        sum = 0
        for j in range(i, i+pts_per_h):
            sum += Q[j]
        if sum < 0:
            sum = 0
        flow_series.append(sum/pts_per_h)
        dt.append(datetime(2024, 1, 1, 0, 0, k))
        k += 1

    return (np.array(flow_series[:30]), dt[:30])


def mean_reading(df_list: List, time_align_col: str, data_col: str) -> Tuple[np.array, np.array]:
    """
    Computes the ensemble mean sensor reading across the data frames in df_list.

    :param df_list: List of data frames.
    :param time_align_col: Name of time alignment column.
    :param data_col: Name of column containing data to ensemble.

    :return: Ensemble of data.
    :return: Corresponding time array.
    """
    time_align_min = max([min(df[time_align_col]) for df in df_list])
    time_align_max = min([max(df[time_align_col]) for df in df_list])

    for i, df in enumerate(df_list):
        min_idx = np.argmin(np.abs(df[time_align_col].to_numpy() - time_align_min))
        max_idx = np.argmin(np.abs(df[time_align_col].to_numpy() - time_align_max))
        y = df[data_col].iloc[min_idx:max_idx].to_numpy()
        if i == 0:
            data_mean = y
        else:
            data_mean += y

    data_mean /= len(df_list)

    return data_mean, df[time_align_col].iloc[min_idx:max_idx].to_numpy()


def flow_to_swmm_readable(Q: np.ndarray, t: List[datetime], file_dir: Path, file_name: str) -> None:
    """
    Convert flow data and save as SWMM-readable .dat files.

    :param Q: Flow time series.
    :param t: List of datetime objects.
    :param file_dir: File directory as Path object.
    :param file_name: Name of file with .dat.

    :return: None
    """
    # Date strings.
    dates = [datetime.strftime(d, '%m-%d-%Y') for d in t]
    times = [datetime.strftime(d, '%H:%M:%S') for d in t]

    # Format data into a data frame.
    df = pd.DataFrame({'date': dates, 'time': times, 'Q': Q})

    # Save the dataframe as a dat file
    df.to_csv(file_dir / file_name, index=False, header=False, sep='\t')


def select_experiments(db_filepath: Path, experiment_dir: Path, date: str, fault_level: int, drained: int):
    """
    Select experiment file names corresponding to a specified date, fault level, and drainage scenario.

    :param db_filepath: Path to database csv file.
    :param date: Date of the data in yyyy-mm-dd.
    :param fault_level: Fault level value.
    :param drained: Drainage scenario.

    :return: List of corresponding file names from database.
    """
    # Load database as data frame.
    db = pd.read_csv(db_filepath)

    # Extract date from filename (assumes format: yyyy-mm-dd_HH-MM-SS_sensor_data.csv)
    db['date'] = db['filename'].str.extract(r'(^\d{4}-\d{2}-\d{2})')

    # Filter based on date, fault_level, and drained
    filtered = db[
        (db['date'] == date) &
        (db['fault_level'] == fault_level) &
        (db['drained'] == drained)
    ]

    # List of file names.
    fnames = filtered['filename'].tolist()

    # Load data frames.
    dfs = [pd.read_csv(experiment_dir / f) for f in fnames]

    return dfs





