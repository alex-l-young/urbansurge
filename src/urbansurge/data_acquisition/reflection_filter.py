##############################################################
# Banner s18uuaq reflection filtering script.
##############################################################

# Library imports.
import numpy as np
from sklearn.cluster import dbscan

def filter_reflections(y):
    """
    Filter Banner s18uuaq sensor data for reflections which show up as regular voltage jumps in the data.

    Parameter
        y : Sensor data read through a DAQ at the same sampling rate as the sensor.
    Return
        y_labels : Labels for y indexed from 0 where each label corresponds to a reflection level.
    """
    # Take the differences of y.
    y_diff = np.diff(y)

    # Sort the differences.
    sorted_diff = np.sort(abs(y_diff))

    # Difference the sorted differences.
    sorted_diff_diff = np.diff(sorted_diff)

    # Sort again.
    sorted_sorted_diff_diff = np.sort(sorted_diff_diff)

    # Find the maximum difference. This is the estimate of the shift.
    shift = np.max(sorted_diff_diff)
    shift = sorted_sorted_diff_diff[-1]

    # DBScan data.
    _, labels = dbscan(y_diff.reshape(-1, 1), eps=shift/3)

    # Compute means of differences with each label.
    unique_labels = np.unique(labels)
    label_means = np.zeros(len(unique_labels))
    for i, ul in enumerate(unique_labels):
        label_means[i] = np.mean(y_diff[labels == ul])

    # Sort unique labels by label means.
    sort_label_means = np.sort(label_means)
    label_mean_sort_idx = np.argsort(label_means)
    sort_unique_labels = unique_labels[label_mean_sort_idx]

    # Index of label mean closest to 0.
    zero_idx = np.argmin(np.abs(sort_label_means - 0))

    # Shift indices.
    shift_lookup = dict(zip(sort_unique_labels, np.array(range(-zero_idx, zero_idx + 1))))
    shift_idx = np.array([shift_lookup[l] for l in labels])

    # Labels for y.
    y_labels = np.cumsum(np.insert(shift_idx, 0, 0))
    y_labels += np.abs(np.min(y_labels)) # Index from 0

    return y_labels