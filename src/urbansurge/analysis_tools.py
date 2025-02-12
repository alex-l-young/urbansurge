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
    :param t: List of corresponding time stamps.
    :raises ValueError: If `R` and `t` have different lengths.
    :return: A tuple containing two dictionaries:
        - `S`: Maps storm indices to lists of rainfall intensity values.
        - `St`: Maps storm indices to lists of corresponding time stamps.
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
    S: Dict[int, np.ndarray], 
    St: Dict[int, np.ndarray], 
    t: np.ndarray
) -> np.ndarray:
    """
    Combines storm dictionaries into a single rainfall time series.

    :param S: Dictionary mapping storm indices to rainfall intensity slices.
    :param St: Dictionary mapping storm indices to corresponding time stamp slices.
    :param t: List of time stamps for the full time series.
    """
    # Initialize the output rainfall array with zeros
    Rp = np.zeros_like(t, dtype=float)

    # Iterate over each storm
    for Si, Ri in S.items():
        ti = np.array(St[Si])  # Convert storm timestamps to array

        # Find indices in t corresponding to ti
        indices = np.searchsorted(t, ti)

        # Add rainfall values at corresponding indices
        np.add.at(Rp, indices, Ri.flatten())

    return Rp
    

def perturb_storm_arrival(
    S: Dict[int, np.ndarray], 
    St: Dict[int, np.ndarray], 
    t: np.ndarray, 
    sig_t: float
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Perturbs the arrival time of storms.

    :param S: Dictionary mapping storm indices to rainfall intensity slices.
    :param St: Dictionary mapping storm indices to corresponding time stamp slices.
    :param t: List of time stamps for the full time series.
    :param sig_t: Standard deviation for the normal distribution used to shift storms.
    :return: Two dictionaries containing the perturbed rainfall intensities and corresponding time stamps.
    """
    # Time step.
    dt = t[1] - t[0]
    Nt = len(t)
    
    # # Perturbed arrival time dictionaries.
    Sp: Dict[int, np.ndarray] = {}
    Stp: Dict[int, np.ndarray] = {}

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


def perturb_storm_magnitude(
    S: Dict[int, np.ndarray], 
    St: Dict[int, np.ndarray], 
    t: np.ndarray, 
    sig_m: float
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Perturbs the magnitude of storms.

    :param S: Dictionary mapping storm indices to rainfall intensity slices.
    :param St: Dictionary mapping storm indices to corresponding time stamp slices.
    :param t: List of time stamps for the full time series.
    :param sig_m: Standard deviation for the normal distribution used to perturb storm magnitudes.
    :return: Two dictionaries containing the perturbed rainfall intensities and corresponding time stamps.
    """
    # Perturbed magnitude dictionaries.
    Sp: Dict[int, np.ndarray] = {} 

    for Si, Ri in S.items():
        # Perturb the storm magnitude.
        perturbation = np.random.normal(0, sig_m)
        Rpi = Ri * (1 + perturbation)
        Rpi[Rpi < 0] = 0
        Sp[Si] = Rpi
    
    return Sp, St


def generate_runoff(
        dt: float,
        T: float,
        n_pulse: int,
        a_bounds: Tuple[float, float],
        L_bounds: Tuple[float, float],
        sig_a: float,
        sig_L: float,
        sig_imp: float,
        n_runoff: int,
        seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate N runoff time series.

    :param dt: Time step.
    :param T: Simulation length in days.
    :param n_pulse: Number of pulses.
    :param a_bounds: Magnitude bounds.
    :param L_bounds: Impulse length bounds in seconds.
    :param sig_a: Standard deviation for magnitude.
    :param sig_L: Standard deviation for impulse length in seconds.
    :param sig_imp: Standard deviation for impulse start in seconds.
    :param n_runoff: Number of runoff time series to generate.
    :return: Runoff array and corresponding array of time stamps.
    """
    # Generate a long time series of pulses.
    T_sec = T * 86400 # T in seconds.
    t = np.arange(0, T_sec, dt)

    # Set random seed.
    if seed is not None:
        np.random.seed(seed) # Using 10 and 30.

    # Runoff array.
    R = np.zeros((len(t), n_runoff))

    for i in range(n_pulse):
        # Sample a from uniform.
        a = np.random.uniform(a_bounds[0], a_bounds[1])

        # Sample start time from uniform.
        impulse_start = np.random.choice(t, 1)[0]

        # Impulse length.
        L = np.random.uniform(L_bounds[0], L_bounds[1])

        for j in range(n_runoff):
            # Sample individual runoff time series from parameters.

            # Magnitude. Ensure it is between bounds.
            a_j = np.min([np.max([np.random.normal(a, sig_a), a_bounds[0]]), a_bounds[1]])

            # Impulse start. Ensure it lies between 0 and max(t).
            impulse_start_j = np.min([np.max([int(np.random.normal(impulse_start, sig_imp)), 0]), np.max(t)])

            # Impulse length. Ensure it is between bounds.
            L_j = np.min([np.max([np.random.normal(L, sig_L), L_bounds[0]]), L_bounds[1]])

            # Generate impulse.
            R_impulse, _ = impulse(a_j, L_j, dt)

            # Add to runoff array at correct start point.
            idx = np.arange(impulse_start_j, np.min([impulse_start_j + len(R_impulse), len(t)])).astype(int)

            R[idx,j] = R[idx,j] + R_impulse[:len(idx)]

    return R, t


if __name__ == '__main__':
    data = {'dt': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
          'node1': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
          'node2': [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215],
          'node3': [301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315]}
    df = pd.DataFrame(data)
    flatten_df(df)