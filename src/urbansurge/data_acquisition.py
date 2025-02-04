# ========================================================
# Stormwater system data acqusition.
# ========================================================

# Library imports.
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

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
    ref_data = df_ref.loc[:pulse_length, corr_col].to_numpy()
    working_data = df.loc[:pulse_length, corr_col].to_numpy()

    # Buffer zone for comparisons.
    buffer_size = 20

    m1s = []
    m2s = []
    for overlap in range(buffer_size, Nt):
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


def flow_rate(dVdt, A):
    """
    Calculate flow rate from voltage derivative and area of tank.

    :param dVdt: Approximated dVdt from output of approx_dVdt().
    :param A: Cross-sectional area of the tank.
    """
    return 2.3017*dVdt*A


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
    dates = [datetime.strftime(d, '%d-%m-%Y') for d in t]
    times = [datetime.strftime(d, '%H:%M:%S') for d in t]

    # Format data into a data frame.
    df = pd.DataFrame({'date': dates, 'time': times, 'Q': Q})

    # Save the dataframe as a dat file
    df.to_csv(file_dir / file_name, index=False, header=False, sep='\t')