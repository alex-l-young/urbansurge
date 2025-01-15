import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, spatial

def moment_fault_detect(dt, y_base, y_obs, significance=0.05, detection_method='diverge', pad=0.0, return_moments=False):
    """

    :param dt: Time stamps as datetime objects.
    :param y_base: Baseline samples at one sensor location. Dimension: (samples, timesteps, sensors)
    :param y_obs: Observed sensor reading. Dimension: (sensors, timesteps)
    :param significance: Significance level.
    :param pad: If detection_method is 'diverge, the padding is added to the range of the baseline moments, decreasing false positives at the expense of true sensitivity.
    :param return_moments: Return baseline and observation moment arrays.

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
        # Stack the vectors to create a data matrix of shape (n_observations, n_variables)
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

        # If m0_obs is 0.0, then other moments will be nan. This is a fault.
        if m0_obs == 0:
            print('NO FLOW OBSERVED >>> Declaring fault')
            detections[si] = 1
            continue
        
        if detection_method == 'mahal':
            # If covariance matrix is singular, there is no noise, so return detection if observations are different from baseline at all.
            SMALL = 1e-10
            try:
                if np.linalg.matrix_rank(base_cov) < base_cov.shape[0]:
                    print('COVARIANCE MATRIX NON-INVERTIBLE. Assuming measurement error is 0.0 and using L2-norm detection.')
                    if np.linalg.norm(obs_moments - base_moments[0,:]) > SMALL:
                        detections[si] = 1
                        continue
                    else:
                        detections[si] = 0
                        continue
            except Exception as e:
                print(e)
                fig, ax = plt.subplots()
                for si in range(n_sensor):
                    ax.plot(y_obs[si,:])
            
            # Calculate the Mahalanobis distance of the test vector
            mahalanobis_distance = spatial.distance.mahalanobis(obs_moments, base_mean, np.linalg.inv(base_cov))

            # Determine the confidence threshold (critical value from chi-squared distribution with 4 DOF)
            threshold = np.sqrt(stats.chi2.ppf(1 - alpha, df=4))

            # Check if the observed vector falls outside the 95% confidence hyperellipse. If yes, fault detected.
            detections[si] = mahalanobis_distance > threshold

        elif detection_method == 'diverge':
            # Detection when observed moment falls outside range of baseline moment samples.
            # Find column-wise min and max of baseline moments.
            b_min = base_moments.min(axis=0)
            b_max = base_moments.max(axis=0)

            # print('BASE M MAX:', b_max)
            # print('BASE M MIN:', b_min)
            # print('OBS MOMENTS:', obs_moments)

            # Check if elements of observed moments are outside the range of the respective baseline moments.
            baseline_range = np.abs(b_max - b_min)
            pad_absolute = baseline_range * pad
            detection = np.all((obs_moments < b_min - pad_absolute) | (obs_moments > b_max + pad_absolute))
            detections[si] = detection

            # print('DETECTION:', detection)

    # If there are detections at any of the sensors, return detection.
    if np.any(detections):
        detect = True
    else:
        detect = False

    # Optional return moments.
    if return_moments is True:
        moments = {'baseline': base_moments, 'observation': obs_moments}
        return detect, moments
    else:
        return detect