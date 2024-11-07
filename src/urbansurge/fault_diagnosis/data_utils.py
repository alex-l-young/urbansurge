
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class ANNDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]

        return data, label


def normalize_residuals(swmm, residual_df, sensor_links):
    # Loop through link IDs and normalize depth and velocity residuals.
    for link_id in sensor_links:
        # Get pipe diameter.
        max_depth = swmm.get_link_geometry(link_id)[0]

        # Maximum velocity occurs at 78% of diameter.
        d_vmax = 0.78 * max_depth

        # Link roughness.
        n = swmm.get_link_roughness(link_id)

        # Link slope.
        S = swmm.get_link_slope(link_id)
        S = np.abs(S)

        # Link hydraulic radius at maximum velocity depth.
        Rh = swmm.get_link_circular_Rh(d_vmax, max_depth)

        # Compute maximum velocity.
        max_velocity = (1.49 / n) * Rh ** (2 / 3) * S ** (1 / 2)

        # Normalize depth.
        if f'Depth_link_{link_id}' in residual_df.columns:
            residual_df[f'Depth_link_{link_id}'] = residual_df[f'Depth_link_{link_id}'] / max_depth

        # Normalize velocity.
        if f'Velocity_link_{link_id}' in residual_df.columns:
            residual_df[f'Velocity_link_{link_id}'] = residual_df[f'Velocity_link_{link_id}'] / max_velocity

    return residual_df

def prepare_residuals(residual_df, dep_vel_cols):
    # Fault scenarios.
    scenarios = np.unique(residual_df.scenario)

    # Scenario length from first scenario.
    scenario_rows = residual_df.loc[residual_df.scenario == scenarios[0], dep_vel_cols].shape[0]

    # Create X and y arrays.
    nrows = scenario_rows
    X = np.zeros((nrows, len(dep_vel_cols), len(scenarios)))
    y_series = []

    for i, scenario in enumerate(scenarios):
        X_scenario = residual_df.loc[residual_df.scenario == scenario, dep_vel_cols].to_numpy()
        X[:, :, i] = X_scenario[:nrows, :]
        y_scenario = residual_df.loc[
                         residual_df.scenario == scenario, ['fault_component', 'fault_value', 'fault_type']].iloc[0, :]
        y_series.append(y_scenario)

    y = pd.DataFrame(y_series)
    y.reset_index(inplace=True, drop=True)

    return X, y

def scale_to_unity(x):
    return (x - x.min()) / (x.max() - x.min())

def unscale_from_unity(x, xmin, xmax):
    return x * (xmax - xmin) + xmin


def clip_timeseries(df, clip_hours, by_scenario=True):
    # Convert datetime column to datetime.
    df.datetime = pd.to_datetime(df.datetime)

    # Clip datetime.
    clip_datetime = df['datetime'].iloc[0] + timedelta(hours=clip_hours)

    if by_scenario == True:
        # Group by scenario.
        df_groups = df.groupby('scenario')

        # Loop through each group
        filtered_groups = []
        for name, group in df_groups:
            # Filter the group as needed
            group = group.loc[group.datetime <= clip_datetime, :]

            # Append the filtered group to the list
            filtered_groups.append(group)

        # Recombine all filtered groups into a single DataFrame
        df_filtered = pd.concat(filtered_groups)

        # Optionally, reset the index if needed
        df_filtered.reset_index(drop=True, inplace=True)
    else:
        df_filtered = df.loc[df.datetime <= clip_datetime, :]

    return df_filtered


def compute_residuals(fault_df, baseline_df, state_columns):
    # Depth and velocity from fault data.
    fault_ar = fault_df[state_columns].to_numpy()

    # Depth and velocity from baseline data.
    baseline_ar = baseline_df[state_columns].to_numpy()

    # Number of times to repeat the baseline data.
    repeats = int(fault_df.shape[0] / baseline_df.shape[0])
    baseline_repeats = np.tile(baseline_ar.T, repeats).T

    # Create residual data.
    residual_ar = fault_ar - baseline_repeats

    # Residual data frame with fault data added back in.
    extra_fault_cols = [col for col in fault_df.columns if col not in state_columns]
    residual_df = pd.DataFrame(residual_ar, columns=state_columns)
    residual_df = pd.merge(residual_df, fault_df[extra_fault_cols], left_index=True, right_index=True)

    return residual_df


def normalize_states(swmm, state_df, sensor_links, norm_type='geometry'):
    # Copy data frame.
    df = state_df.copy()

    # Loop through link IDs and normalize depth and velocity residuals.
    for link_id in sensor_links:
        if norm_type == 'geometry':
            # Get pipe diameter.
            max_depth = swmm.get_link_geometry(link_id)[0]

            # Maximum velocity occurs at 78% of diameter.
            d_vmax = 0.78 * max_depth

            # Link roughness.
            n = swmm.get_link_roughness(link_id)

            # Link slope.
            S = swmm.get_link_slope(link_id)
            S = np.abs(S)

            # Link hydraulic radius at maximum velocity depth.
            Rh = swmm.get_link_circular_Rh(d_vmax, max_depth)

            # Compute maximum velocity.
            max_velocity = (1.49 / n) * Rh ** (2 / 3) * S ** (1 / 2)

            # Normalize depth.
            if f'Depth_link_{link_id}' in df.columns:
                df[f'Depth_link_{link_id}'] = df[f'Depth_link_{link_id}'] / max_depth

            # Normalize velocity.
            if f'Velocity_link_{link_id}' in df.columns:
                df[f'Velocity_link_{link_id}'] = df[f'Velocity_link_{link_id}'] / max_velocity

        elif norm_type == 'mean_var':
            # Fault scenarios.
            scenarios = np.unique(state_df.scenario)

            # Loop through scenarios and normalize time series to mean of 0 and variance of 1.
            for i, scenario in enumerate(scenarios):
                for link_id in sensor_links:
                    # Normalize depth.
                    if f'Depth_link_{link_id}' in df.columns:
                        depth_ar = df[f'Depth_link_{link_id}']
                        df[f'Depth_link_{link_id}'] = (depth_ar - depth_ar.mean(axis=0)) / depth_ar.std(axis=0)

                    # Normalize velocity.
                    if f'Velocity_link_{link_id}' in df.columns:
                        vel_ar = df[f'Velocity_link_{link_id}']
                        df[f'Velocity_link_{link_id}'] = (vel_ar - vel_ar.mean(axis=0)) / vel_ar.std(axis=0)

        elif norm_type is None:
            pass

    return df


def prepare_states(state_df, dep_vel_cols):
    # Fault scenarios.
    scenarios = np.unique(state_df.scenario)

    # Scenario length from first scenario.
    scenario_rows = state_df.loc[state_df.scenario == scenarios[0], dep_vel_cols].shape[0]

    # Create X and y arrays.
    nrows = scenario_rows
    X = np.zeros((nrows, len(dep_vel_cols), len(scenarios)))
    y_series = []

    for i, scenario in enumerate(scenarios):
        X_scenario = state_df.loc[state_df.scenario == scenario, dep_vel_cols].to_numpy()
        X[:, :, i] = X_scenario[:nrows, :]
        y_scenario = state_df.loc[state_df.scenario == scenario, ['fault_component', 'fault_value', 'fault_type', 'scenario']].iloc[
                     0, :]
        y_series.append(y_scenario)

    y = pd.DataFrame(y_series)
    y.reset_index(inplace=True, drop=True)

    return X, y


def generate_baseline_observations(baseline_df, dep_vel_cols, n_samples, noise_std, relative_noise=False):
    """
    Generate baseline observations by simulating noise during a sampling run.
    :param baseline_df: Data frame containing measurements from baseline system run.
    :param dep_vel_cols: Depth or velocity columns to use as measurements.
    :param n_samples: Number of times to sample the baseline data.
    :param noise_std: Standard deviation of the noise. Assumes normal distribution.
    :param relative_noise: If set to True, noise_std will be multiplied by the measurement so errors will be heteroskedastic.
    
    :return d_obs_baseline: Numpy array of baseline samples. (t, ncol, n_samples)
    :return baseline_ens_df: Data frame containing the ensembled d_obs_baseline.
    """
    d_obs_baseline = np.zeros((baseline_df.shape[0], len(dep_vel_cols), n_samples))
    for i in range(n_samples):
        d_obs_noiseless = baseline_df[dep_vel_cols].to_numpy()
        if relative_noise is False:
            d_obs = d_obs_noiseless + np.random.normal(0, noise_std, size=d_obs_noiseless.shape)
        else:
            d_obs = d_obs_noiseless + np.random.normal(0, noise_std * d_obs_noiseless)
            
        d_obs_baseline[:, :, i] = d_obs

    # Ensemble of baseline observations.
    baseline_ens = np.mean(d_obs_baseline, axis=2)

    # Replace the depth-velocity columns with the ensembled data.
    baseline_ens_df = baseline_df.copy()
    baseline_ens_df[dep_vel_cols] = baseline_ens

    return d_obs_baseline, baseline_ens_df