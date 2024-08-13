##########################################################
# Accuracy metrics for model testing.
#
# Alex Young, 2024
##########################################################

# Library imports.
import numpy as np

###### ACCURACY METRICS ######

def mean_absolute_error(label, prediction):
    me = np.sum(np.abs(prediction - label)) / len(label)

    return me

def true_positive_rate(label, prediction):
    N = len(label)
    match = prediction == label
    TP = np.sum(match)
    TPR = TP / N

    return TPR