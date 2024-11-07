
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def residual_fault_detect(Y_baseline, y_obs, significance=0.05, test_type='binomial'):
    """
    Fault detection on residual distribution.
    :param Y_baseline: Baseline samples from all sensors. Dimension: (samples, timesteps, sensors)
    :param y_obs: Observed sensor readings during fault detection operation. Dimension: (sensors, timesteps)
    """
    # Dimension of Y_baseline.
    baseline_dim = Y_baseline.ndim

    # Expand dimension if there is a single sensor.
    if baseline_dim == 2:
        Y_baseline = Y_baseline[:,:,np.newaxis]

    # Number of sensors.
    Nsensor = Y_baseline.shape[2]

    # Peel off first baseline sample for all sensors.
    y_1 = Y_baseline[0,:,:].T

    # Compute ensemble of baseline across rows. Dimension is now (sensors, timesteps)
    y_ens = np.mean(Y_baseline[1:,:,:], axis=0).T

    # Estimate of baseline residuals.
    res_base = y_1 - y_ens

    # Estimate of observed residuals.
    res_obs = y_obs - y_ens
    
    # Significance level for individual tests from family-wise error rate.
    alpha = 1 - (1 - significance)**(1 / Nsensor)

    p_values = np.ones(Nsensor)
    for i in range(Nsensor):
        # K-S Test
        if test_type == 'KS':
            ks_result = stats.ks_2samp(np.squeeze(res_base[i,:]), np.squeeze(res_obs[i,:]))
            p_values[i] = ks_result.pvalue
        elif test_type == 'CvM':
            cvm_result = stats.cramervonmises_2samp(np.squeeze(res_base[i,:]), np.squeeze(res_obs[i,:]))
            p_values[i] = cvm_result.pvalue
        elif test_type == 'binomial':
            # Find X% exceedance threshold of res_base.
            exceed_prc = 95
            threshold = np.percentile(np.abs(res_base[i,:]), exceed_prc)

            # Number of exceedences from observed residuals.
            n_exceed = np.sum(np.abs(res_obs[i,:]) > threshold)
            binom_result = stats.binomtest(n_exceed, np.size(res_obs[i,:]), p=(100 - exceed_prc) / 100, alternative='greater')
            p_values[i] = binom_result.pvalue
        elif test_type == 'noise_floor':
            SNR = 1.5
            # Noise floor.
            noise_floor = np.mean(np.square(res_base[i,:]))

            # Maximum observed residual squared.
            max_obs = np.max(np.square(res_obs[i,:]))

            if max_obs / noise_floor > SNR:
                p_values[i] = 0.1 * alpha # Fault detected.

        elif test_type == 'gpd':
            # Use a generalized pareto distribution to detect extreme anomalies.
            ts = np.abs(res_base[i,:])

            # Step 1: MLE to find the optimal threshold value.
            # Compute threshold as percentile.
            threshold = np.percentile(np.abs(res_base[i,:]), 95)

            # Extract peaks over the threshold
            peaks = ts[ts > threshold] - threshold

            # Fit the generalized Pareto distribution to the peaks
            params = stats.genpareto.fit(peaks)
            shape, loc, scale = params

            # Step 2: Peaks over threshold for observed data.
            robs = res_obs[i,:]
            peaks_obs = robs[robs > threshold] - threshold
            print(peaks_obs)

            # Step 4: Compute probability of peaks over threshold for observed residuals.
            p_peaks = np.array(stats.genpareto.cdf(peaks_obs, shape, loc, scale))
            print('P(peaks)', p_peaks)

            if np.any(p_peaks > 1 - significance):
                return True

        elif test_type == 'moment':
            # Time.
            t = np.array([(dt[i] - dt[0]).total_seconds() / 3600 for i in range(len(dt))])


            # Compute moments for baseline scenarios.
            m0_base = np.zeros(Y_baseline.shape[0])
            m1_base = np.zeros_like(m0_base)
            m2_base = np.zeros_like(m0_base)
            m3_base = np.zeros_like(m0_base)
            for i in range(d_obs_baseline.shape[0]):
                y_base = np.squeeze(d_obs_baseline[i,:,:])
                m0_base[i] = np.sum(y_base)
                m1_base[i] = np.sum(t * y_base) / np.sum(y_base)
                m2_base[i] = np.sum((t - m1_base[i])**2 * y_base) / np.sum(y_base)
                m3_base[i] = np.sum((t - m1_base[i])**3 * y_base) / np.sum(y_base)

            
            # Fit a 2D multivariate normal distribution
            xe = np.mean(m0_base)
            ye = np.mean(m1_base)
            mean = [xe, ye]
            covariance = np.cov(m0_base, m1_base)

    # print(p_values)

    # Fault is detected if any of the p-values are below the significance level.
    if np.any(p_values < alpha):
        detect = True
    else:
        detect = False

    print(p_values, alpha, detect)

    return detect


if __name__ == '__main__':
    Nsample = 5
    Nsensor = 1
    Nt = 100
    Y_baseline = np.random.normal(0, 5, size=(Nsample, Nt, Nsensor))
    y_obs = np.random.normal(0, 1, size=(Nsensor, Nt))

    detect = ks_fault_detect(Y_baseline, y_obs)
    print(detect)

