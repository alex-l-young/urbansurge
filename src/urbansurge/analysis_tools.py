##########################################################################
# Fault detection analysis tools.
# Alex Young
##########################################################################

# Library imports.
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def flatten_df(df):
    '''
    Flattens Pandas data frame columns to rows.
    :param df:
    :return:
    '''

    df_T = df.T

    single_row_dict = {}
    for i, name in enumerate(df_T.index):
        col_dict = {f'{name}_{j}': df_T.iloc[i, j] for j in range(df_T.shape[1])}

        # Append to single_row_dict.
        single_row_dict.update(col_dict)

    single_row_df = pd.DataFrame(single_row_dict, index=[0])

    return single_row_df


def impulse(a, L, dt):
    """
    Generates a cos^2 impulse of defined magnitude, and duration.

    :param a: Impulse peak magnitude.
    :param L: Length of impulse in time steps.
    :param dt: Time step length.

    :return I: Impulse values.
    :return t: Time array.  
    """
    b = np.pi / 2
    t = np.arange(0, L, dt)
    I = a * np.cos(np.pi / L * t + b)**2
    
    return I, t


def split_storms(R: List[float], t: List[float]) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
    """
    Splits rainfall data into individual storms.

    :param R: List of rainfall intensity values.
    :type R: List[float]
    :param t: List of corresponding time stamps.
    :type t: List[float]
    :raises ValueError: If `R` and `t` have different lengths.
    :return: A tuple containing two dictionaries:
        - `S`: Maps storm indices to lists of rainfall intensity values.
        - `St`: Maps storm indices to lists of corresponding time stamps.
    :rtype: Tuple[Dict[int, List[float]], Dict[int, List[float]]]
    """
    if len(R) != len(t):
        raise ValueError("R and t must have the same length.")

    S: Dict[int, List[float]] = {}  # Storm index to rainfall slices
    St: Dict[int, List[float]] = {}  # Storm index to time slices

    storm_index = -1
    in_storm = False
    start_idx = 0

    for i in range(len(R)):
        if R[i] > 0 and not in_storm:
            # Start of a new storm
            storm_index += 1
            start_idx = i
            in_storm = True

        # End of a storm detected when R falls back to 0 and stays at 0
        if in_storm and (i == len(R) - 1 or R[i + 1] == 0):
            # Capture the storm slice
            S[storm_index] = R[start_idx:i + 1]
            St[storm_index] = t[start_idx:i + 1]
            in_storm = False

    return S, St


def perturb_storm_arrival(R: List[float], t: List[float], sig_t: float) -> List[float]:
    """
    Perturbs arrival time of storms.

    :param R: List of rainfall intensity values.
    :type R: List[float]
    :param t: List of corresponding time stamps.
    :type t: List[float]
    :param sig_t: Standard deviation of arrival time normal distribution.
    :type sig_t: float
    
    """

    # Split the storms.
    S, St = split_storms(R, t)

    # Time step.
    dt = t[1] - t[0]
    
    # Perturbed runoff array.
    Rp = np.zeros_like(R)

    for Si, Ri in S.items():
        # Index of first time step.
        ti = np.argwhere(t == St[Si][0])[0][0]

        # Shift in number of array indices. Ensure Ri not shifted completely off of Rp.
        storm_start_idx = len(Rp) + 1
        storm_end_idx = -1
        while storm_start_idx > len(Rp) or storm_end_idx < 0:
            shift = round(np.random.normal(0, sig_t) / dt)

            storm_start_idx = ti + shift
            storm_end_idx = ti + shift + len(Ri)

        if storm_start_idx < 0:
            idx_diff = -storm_start_idx
            storm_start_idx = 0
            # print('SS', 'Rp', len(Rp[storm_start_idx:storm_end_idx]), 'Ri', len(Ri[idx_diff:]), len(Ri), 'SH', shift)
            Rp[storm_start_idx:storm_end_idx] += Ri[idx_diff:]
        elif storm_end_idx > len(Rp):
            idx_diff = storm_end_idx - len(Rp)
            storm_end_idx = len(Rp)
            # print('SE', 'Rp', len(Rp[storm_start_idx:storm_end_idx]), 'Ri', len(Ri[:-idx_diff]), len(Ri), 'SH', shift)
            Rp[storm_start_idx:storm_end_idx] += Ri[:-idx_diff]
        else:
            Rp[storm_start_idx:storm_end_idx] += Ri

    return Rp





if __name__ == '__main__':
    data = {'dt': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
          'node1': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
          'node2': [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215],
          'node3': [301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315]}
    df = pd.DataFrame(data)
    flatten_df(df)