import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, spatial

def moment_diagnose(dt, y_base, y_obs, significance=0.05):
    """
    :param dt: Time stamps as datetime objects.
    :param y_base: Baseline samples at one sensor location. Dimension: (samples, timesteps, sensors)
    :param y_obs: Observed sensor reading. Dimension: (sensors, timesteps)
    :param alpha: Significance level.
    :return: Whether or not a fault was detected at the sensor.
    """
    # Number of samples and time steps.
    n_samples = y_base.shape[0]
    Nt = len(dt)
    n_sensor = y_base.shape[2]

    # Add new axis if there is only one sensor to maintain dimensional consistency.
    if n_sensor == 1:
        y_obs = y_obs[np.newaxis,:]

    # Time steps to hours since first time step.
    t = np.array([(dt[i] - dt[0]).total_seconds() / 3600 for i in range(Nt)])

    # Significance level for individual tests from family-wise error rate.
    alpha = 1 - (1 - significance)**(1 / n_sensor)

    detections = np.zeros(n_sensor)
    for si in range(n_sensor):
        # Extract data from one sensor.
        y_base_sensor = np.squeeze(y_base[:,:,si])

        # Compute moments for baseline scenarios.
        m0_base = np.zeros(n_samples)
        m1_base = np.zeros_like(m0_base)
        m2_base = np.zeros_like(m0_base)
        m3_base = np.zeros_like(m0_base)
        for i in range(n_samples):
            y_samp = y_base_sensor[i,:]
            m0_base[i] = np.sum(y_samp)
            m1_base[i] = np.sum(t * y_samp) / np.sum(y_samp)
            m2_base[i] = np.sum((t - m1_base[i])**2 * y_samp) / np.sum(y_samp)
            m3_base[i] = np.sum((t - m1_base[i])**3 * y_samp) / np.sum(y_samp)

        # Fit an MVN on the moments.
        # Stack the vectors to create a data matrix of shape (n_samples, n_features)
        base_moments = np.stack((m0_base, m1_base, m2_base, m3_base), axis=1)

        # Calculate mean and covariance matrix
        base_mean = np.mean(base_moments, axis=0)
        base_cov = np.cov(base_moments, rowvar=False)

        # Compute moments for observation.
        y_obs_sensor = y_obs[si,:]
        m0_obs = np.sum(y_obs_sensor)
        m1_obs = np.sum(t * y_obs_sensor) / np.sum(y_obs_sensor)
        m2_obs = np.sum((t - m1_obs)**2 * y_obs_sensor) / np.sum(y_obs_sensor)
        m3_obs = np.sum((t - m1_obs)**3 * y_obs_sensor) / np.sum(y_obs_sensor)

        # Concatenate into a single vector.
        obs_moments = np.array([m0_obs, m1_obs, m2_obs, m3_obs])

        # fig, axes = plt.subplots(2, 2)
        # axes[0,0].scatter(base_moments[:,0], base_moments[:,1], c='k', marker='x')
        # axes[0,0].scatter(obs_moments[0], obs_moments[1])

        # axes[0,1].scatter(base_moments[:,1], base_moments[:,2], c='k', marker='x')
        # axes[0,1].scatter(obs_moments[1], obs_moments[2])

        # axes[1,0].scatter(base_moments[:,2], base_moments[:,3], c='k', marker='x')
        # axes[1,0].scatter(obs_moments[2], obs_moments[3])

        # If covariance matrix is singular, there is no noise, so return detection if observations are different from baseline at all.
        SMALL = 10e-5
        if np.linalg.matrix_rank(base_cov) < base_cov.shape[0]:
            print('COVARIANCE MATRIX NON-INVERTIBLE. Assuming measurement error is 0.0 and using L2-norm detection.')
            if np.linalg.norm(obs_moments - base_moments[0,:]) > SMALL:
                detections[si] = 1
                continue

        # Calculate the Mahalanobis distance of the test vector
        mahalanobis_distance = spatial.distance.mahalanobis(obs_moments, base_mean, np.linalg.inv(base_cov))

        # Determine the confidence threshold (critical value from chi-squared distribution with 4 DOF)
        threshold = np.sqrt(stats.chi2.ppf(1 - alpha, df=4))

        # Check if the observed vector falls outside the 95% confidence hyperellipse. If yes, fault detected.
        detections[si] = mahalanobis_distance > threshold

    # If there are detections at any of the sensors, return detection.
    if np.all(detections == 1):
        detect = True
    else:
        detect = False

    return detect