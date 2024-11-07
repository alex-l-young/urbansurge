
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def ks_fault_detect(Y_baseline, y_obs, significance=0.05):
    """
    Kolmogorof-Smirnov fault detection on residual distribution.
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

    # print('res_base', res_base.shape)
    # print('res_obs', res_obs.shape)

    # fig, ax = plt.subplots()
    # ax.plot(np.squeeze(res_base))
    # ax.plot(np.squeeze(res_obs))

    # fig, ax = plt.subplots()
    # baseline_ecdf = stats.ecdf(np.squeeze(res_base))
    # baseline_ecdf.cdf.plot(ax)
    # obs_ecdf = stats.ecdf(np.squeeze(res_obs))
    # obs_ecdf.cdf.plot(ax)
    
    ### Two-sample K-S test.
    # Significance level for individual tests from family-wise error rate.
    alpha = 1 - (1 - significance)**(1 / Nsensor)

    # 2-sample K-S test.
    p_values = np.zeros(Nsensor)
    for i in range(Nsensor):
        # ks_result = stats.ks_2samp(np.squeeze(res_base[i,:]), np.squeeze(res_obs[i,:]))
        # p_values[i] = ks_result.pvalue
        cvm_result = stats.cramervonmises_2samp(np.squeeze(res_base[i,:]), np.squeeze(res_obs[i,:]))
        p_values[i] = cvm_result.pvalue

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

