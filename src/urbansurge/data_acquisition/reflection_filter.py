##############################################################
# Banner s18uuaq reflection filtering script.
##############################################################

# Library imports.
import numpy as np
from sklearn.cluster import dbscan

def filter_reflections(t, y, k=4):
    """
    Filter Banner s18uuaq sensor data for reflections which show up as regular voltage jumps in the data.

    Parameter
        t : Timestamps for the data.
        y : Sensor data read through a DAQ at the same sampling rate as the sensor.
    Return
        y_labels : Labels for y indexed from 0 where each label corresponds to a reflection level.
    """
    # Time step.
    dt = t[1] - t[0]
    fs = 1 / dt

    # Resampled frequency.
    fs_rs = 400

    # Sampling interval.
    skip = int(fs / fs_rs)  

    # Resampled signal.
    t_rs = t[::skip]
    y_rs = y[::skip]

    # 4-pt Moving average centered on each value of y.
    kernel = np.ones(k) / k

    # Apply convolution with 'same' mode to keep the output the same length as the input.
    y_filt = np.convolve(y_rs, kernel, mode='same')

    return t_rs, y_filt