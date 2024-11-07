# Description : Diagnose faults based on distance between observation and simulation.
# Author      : Alex Young
# Email       : ay434@cornell.edu

# Library imports.
import numpy as np
import scipy


class DistanceDiagnoser:
    def __init__(self):
        pass

    def fit_model(self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train


    def predict(self, X_test, n_l2=1):
        """
        Compute L2-norm between X_test and X_train. Return rank-ordered list of y_train elements that are the closest
        to X_test.
        :param X_test: Testing X data.
        :param n_l2: If n_l2 is greater than 1, l2_min includes L2-norm values for the next n_l2 closest samples as extra columns.
        :return: Rank ordered label predictions.
        """
        # # Compute L2 norm between X_test and X_train.
        # l2_norms = np.linalg.norm(self.X_train - X_test, axis=1)

        # diff = self.X_train[:, np.newaxis, :] - X_test[np.newaxis, :, :]

        # Step 2: Compute the L2 norm (Euclidean distance) along the last axis
        l2_norm_matrix = scipy.spatial.distance.cdist(self.X_train, X_test)
        # print(X_test.shape)
        print(l2_norm_matrix.shape)

        # # Argmin across rows.
        # if n_l2 == 1:
        #     l2_min = np.min(l2_norm_matrix, axis=0)
        # else:
        l2_sort = np.sort(l2_norm_matrix, axis=0)[:n_l2, :]
        l2_argsort = np.argsort(l2_norm_matrix, axis=0)[:n_l2, :]
        # l2_argmin = np.argmin(l2_norm_matrix, axis=0)

        # # L2-norm indices.
        # l2_sort = np.sort(l2_norms)
        # l2_sort_idx = np.argsort(l2_norms)

        # # Corresponding labels sorted by L2-norm.
        # y_pred = self.y_train.iloc[l2_argmin, :]

        # Create y-predict based on the sorted L2-norm indices.
        y_pred = self.y_train.iloc[l2_argsort.flatten(order='F'), :].copy()

        # Add the L2-norm values.
        y_pred['l2_norm'] = l2_sort.flatten(order='F')

        # Add an index corresponding to the Xtest rows.
        y_pred['xtest_idx'] = np.repeat(np.arange(X_test.shape[0]), n_l2)
        y_pred.reset_index(inplace=True, drop=True)

        return y_pred