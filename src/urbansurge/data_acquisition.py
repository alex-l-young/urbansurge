# ========================================================
# Stormwater system data acqusition.
# ========================================================

# Library imports.
import numpy as np
from scipy.signal import correlate

# UrbanSurge imports.

def align_measurement_dfs(df_ref, df, time_col, corr_col):
    """
    Aligns a data frame to a reference data frame based on the lag of the maximum correlation.  

    :param df_ref: Reference data frame containing a time column and column to correlate.
    :param df: Data frame to shift to reference data frame.
    :param time_col: Name of the time column in both data frames.
    :param corr_col: Name of the column to correlate on.

    :return df_shift: Same as df, but with a shifted time column to match reference data frame.
    """
    t = df_ref[time_col].to_numpy()
    Nt = len(t)
    dt = t[1] - t[0]

    # Cross correlation.
    corr_ar = correlate(df_ref[corr_col], df[corr_col])

    # Index of maximum cross correlation.
    max_corr_idx = np.argmax(corr_ar)

    # Shift. How far df is shifted from df_ref (negative is to left, positive is to right).
    shift = Nt - max_corr_idx
    t_shifted = np.arange(t[0] - shift * dt, t[-1] - shift * dt, dt)

    # Handle edge cases where t_shifted ends up one dt off from t.
    if len(t_shifted) < Nt:
        t_shifted = np.append(t_shifted, t_shifted[-1] + dt)
    elif len(t_shifted) > Nt:
        t_shifted = t_shifted[:Nt]

    # Update time column in df_test.
    df_shift = df.copy()
    df_shift['time'] = t_shifted

    return df_shift