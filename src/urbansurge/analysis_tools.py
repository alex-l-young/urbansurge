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


def combine_storms(
    S: Dict[int, List[float]], 
    St: Dict[int, List[float]], 
    t: List[float]
) -> List[float]:
    # Initialize the output rainfall array with zeros
    Rp = np.zeros_like(t, dtype=float)

    # Iterate over each storm
    for Si, Ri in S.items():
        ti = np.array(St[Si])  # Convert storm timestamps to array

        # Find indices in t corresponding to ti
        indices = np.searchsorted(t, ti)

        # Add rainfall values at corresponding indices
        np.add.at(Rp, indices, Ri)

    return Rp.tolist()
    

def perturb_storm_arrival(
    S: Dict[int, List[float]], 
    St: Dict[int, List[float]], 
    t: List[float], 
    sig_t: float
) -> Tuple[Dict[int, List[float]], Dict[int, List[float]]]:
    """
    Perturbs the arrival time of storms.

    :param S: Dictionary mapping storm indices to rainfall intensity slices.
    :type S: Dict[int, List[float]]
    :param St: Dictionary mapping storm indices to corresponding time stamp slices.
    :type St: Dict[int, List[float]]
    :param t: List of time stamps for the full time series.
    :type t: List[float]
    :param sig_t: Standard deviation for the normal distribution used to shift storms.
    :type sig_t: float
    :return: Two dictionaries containing the perturbed rainfall intensities and corresponding time stamps.
    :rtype: Tuple[Dict[int, List[float]], Dict[int, List[float]]]
    """
    # Time step.
    dt = t[1] - t[0]
    Nt = len(t)
    
    # Perturbed runoff array.
    Sp: Dict[int, List[float]] = {}  # Storm index to rainfall slices
    Stp: Dict[int, List[float]] = {}  # Storm index to time slices

    for Si, Ri in S.items():
        # Index of first time step.
        ti_start = np.argwhere(t == St[Si][0])[0][0]

        # Time stamps corresponding to Ri.
        ti = St[Si]

        # Shift in number of array indices. Ensure Ri not shifted completely off of Rp.
        storm_start_idx = Nt + 1
        storm_end_idx = -1
        while storm_start_idx > Nt or storm_end_idx < 0:
            shift = round(np.random.normal(0, sig_t) / dt)

            storm_start_idx = ti_start + shift
            storm_end_idx = ti_start + shift + len(Ri)

        if storm_start_idx < 0:
            idx_diff = -storm_start_idx
            Sp[Si] = Ri[idx_diff:]
            Stp[Si] = ti[idx_diff:] + shift
        elif storm_end_idx > Nt:
            idx_diff = storm_end_idx - Nt
            Sp[Si] = Ri[:-idx_diff]
            Stp[Si] = ti[:-idx_diff] + shift
        else:
            Sp[Si] = Ri
            Stp[Si] = ti + shift

    return Sp, Stp


# def perturb_storm_magnitude(R: List[float], t: List[float], sig_t: float) -> List[float]:


if __name__ == '__main__':
    data = {'dt': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
          'node1': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
          'node2': [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215],
          'node3': [301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315]}
    df = pd.DataFrame(data)
    flatten_df(df)