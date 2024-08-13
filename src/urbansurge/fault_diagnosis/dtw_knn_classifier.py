#########################################################
# Description : Dynamic time warping k-nearest neighbors classifier.
# Author      : Alex Young
# Email       : ay434@cornell.edu
#########################################################


# Library imports.
from fastdtw import fastdtw
import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean


class DTWKNNClassifier():
    def __init__(self, n_neighbors):
        """
        Dynamic time warping + k-NN classifier.
        :param n_neighbors [Int]: Number of N neighbors to consider.
        """
        # Model.
        self.clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric=self.DTW)

    def fit_model(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def test_model(self, X_test, y_test):
        y_pred = self.clf.predict(X_test)

        return y_pred, y_test

    def predict(self, X_test):
        y_pred = self.clf.predict(X_test)

        return y_pred

    @staticmethod
    def DTW(a, b):
        # Flatten input vectors.
        a = a[None]
        b = b[None]

        # Compute fast dtw.
        distance, path = fastdtw(a, b, dist=euclidean)

        return distance