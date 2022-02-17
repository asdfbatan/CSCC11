"""
CSCC11 - Introduction to Machine Learning, Winter 2021, Assignment 4
B. Chan, S. Wei, D. Fleet
"""

import numpy as np

class KMeans:
    def __init__(self, init_centers):
        """ This class represents the K-means model.

        TODO: You will need to implement the methods of this class:
        - train: ndarray, int -> ndarray

        Implementation description will be provided under each method.

        For the following:
        - N: Number of samples.
        - D: Dimension of input features.
        - K: Number of centers.
             NOTE: K > 1

        Args:
        - init_centers (ndarray (shape: (K, D))): A KxD matrix consisting K D-dimensional centers.
        """

        assert len(init_centers.shape) == 2, f"init_centers should be a KxD matrix. Got: {init_centers.shape}"
        (self.K, self.D) = init_centers.shape
        assert self.K > 1, f"There must be at least 2 clusters. Got: {self.K}"

        # Shape: K x D
        self.centers = np.copy(init_centers)

    def train(self, train_X, max_iterations=1000):
        """ This method trains the K-means model.

        NOTE: This method updates self.centers

        The algorithm is the following:
        - Assigns data points to the closest cluster center.
        - Re-computes cluster centers based on the data points assigned to them.
        - Update the labels array to contain the index of the cluster center each point is assigned to.
        - Loop ends when the labels do not change from one iteration to the next. 

        Args:
        - train_X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional input data.
        - max_iterations (int): Maximum number of iterations.

        Output:
        - labels (ndarray (shape: (N, 1))): A N-column vector consisting N labels of input data.
        """
        assert len(train_X.shape) == 2 and train_X.shape[1] == self.D, f"train_X should be a NxD matrix. Got: {train_X.shape}"
        assert max_iterations > 0, f"max_iterations must be positive. Got: {max_iterations}"
        N = train_X.shape[0]

        labels = np.empty(shape=(N, 1), dtype=np.long)
        distances = np.empty(shape=(N, self.K))

        for _ in range(max_iterations):
            old_labels = labels

            # ====================================================
            # TODO: Implement your solution within the box

            #print(self.centers.shape)
            centers_append = self.centers[:, None]
            difference = (train_X - centers_append)**2
            distances = np.sqrt(difference.sum(axis = 2))
            
            #print(distance.shape)
            #print(self.centers.shape)
            labels = np.argmin(distances, axis = 0)
            #print(labels.shape)
            #print(train_X.shape) 4000*2     
            
            new_centers = np.zeros((self.K, self.D))
            for i in range(self.K):
                new_centers[i, :] = train_X[labels==i].mean(axis = 0)
            
            self.centers = new_centers
            labels = labels.T
            labels = labels.reshape(N,1)
            
            # ====================================================

            # Check convergence
            if np.allclose(old_labels, labels):
                break

        return labels
