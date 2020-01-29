# Harold Ainsworth
# January 2020
# K Nearest Neighbors Classifier Implementation

import numpy as np
from sklearn.neighbors import BallTree


# TODO: implement own BallTree class and add distance weighting functionality


class KNN:
    """
    Class to store training data, act as a general KNN classifier for any applicable data set
    """
    def __init__(self, X_train, y_train, K, distance_weighted=False):
        """
        initialize the KNN classifier object
        :param X_train: ndarray of points
        :param y_train: array of corresponding classifications
        :param K: the number of nearest points to compare to while classifying
        :param distance_weighted: indicates the use of distance weighting
        """
        # store training data
        self.balltree = BallTree(X_train)
        self.y_train = y_train
        self.K = K
        self.distance_weighted = distance_weighted

    def majority(self, neighbor_indices, neighbor_distances=None):
        """
        given nearest neighbor information, return the majority classification label
        ties broken by removing furthest neighbor and considering the remaining neighbors
        :param neighbor_indices: the indices of the k closest neighbors
        :param neighbor_distances: the distances from the point in question to the neighbors
        :return: the majority classification label
        """
        # base case for recursive function call
        if len(neighbor_indices) == 1:
            # if only 1 index, return classification
            return self.y_train[neighbor_indices[0]]
        else:
            # get count of each classification in list of neighbors
            classification_count = {}
            for n in neighbor_indices:
                if self.y_train[n] not in classification_count:
                    classification_count[self.y_train[n]] = 1
                else:
                    classification_count[self.y_train[n]] += 1

            # find the most frequent digit in the neighbors
            majority_digit = None
            maj_count = 0
            tie = False
            for k in classification_count.keys():
                if classification_count[k] > maj_count:
                    maj_count = classification_count[k]
                    majority_digit = k

            # find if tie for most freq digit
            if list(classification_count.values()).count(maj_count) > 1:
                # if there is more than one digit that has the highest freq_count (tie for most freq digit)
                # remove furthest neighbor from neighbor_indices/neighbor_distances and call function recursively
                furthest_dist = 0
                furthest_index = None
                for i in range(len(neighbor_distances)):
                    if neighbor_distances[i] >= furthest_dist:
                        furthest_dist = neighbor_distances[i]
                        furthest_index = i
                # remove furthest neighbor index and distance from lists, recursively call majority
                neighbor_indices = np.delete(neighbor_indices, furthest_index)
                neighbor_distances = np.delete(neighbor_distances, furthest_index)
                return self.majority(neighbor_indices, neighbor_distances)

            else:
                # if no tie
                return majority_digit



    def classify(self, x):
        """
        given a single point x, predict the correct classification label
        :param x: point to classify
        :return: the predicted classification label for x
        """
        # get the distances and indices of k nearest neighbors
        distances, indices = self.balltree.query(x.reshape(1,-1), k=self.K)
        # find and return the predicted classification label based on the k nearest neighbors
        return self.majority(indices[0], distances[0])

    def predict(self, X):
        """
        given an ndarray of points, classify each point and return an array of all predicted classification labels
        :param X: ndarray of points to be classified
        :return: array of corresponding prediction labels
        """
        yhat = []
        for m in X:
            yhat.append(self.classify(m))
        return yhat

    def test_accuracy(self, X_test, y_test):
        """
        given testing data, calculate the accuracy (0.0-1.0) of the classifier
        :param X_test: ndarray of points to test
        :param y_test: array of actual point classification labels to compare predictions to
        :return: the accuracy (0.0-1.0) of the classifier
        """
        # make predictions for X_test
        yhat = self.predict(X_test)
        # calculate number of correct predictions
        correct_preds = 0
        for i in range(len(yhat)):
            # compare each prediction to actual classification value
            if yhat[i] == data.val_y[i]:
                correct_preds += 1
        # return accuracy
        return correct_preds/len(yhat)