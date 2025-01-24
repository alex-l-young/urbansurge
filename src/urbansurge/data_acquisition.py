# ========================================================
# Stormwater system data acqusition.
# ========================================================

# Library imports.
import numpy as np
from scipy.signal import correlate

# UrbanSurge imports.

def align_measurements(df_ref, df, time_col, corr_col):
    """
    Aligns a data frame time column to a reference data frame based on the lag of the maximum correlation.  

    :param df_ref: Reference data frame containing a time column and column to correlate.
    :param df: Data frame to shift to reference data frame.
    :param time_col: Name of the time column in both data frames.
    :param corr_col: Name of the column to correlate on.

    :return t_align: Same as df, but with a shifted time column to match reference data frame.
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
    t_align = np.arange(t[0] - shift * dt, t[-1] - shift * dt, dt)

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


