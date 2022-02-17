"""
CSCC11 - Introduction to Machine Learning, Winter 2021, Exam
B. Chan, D. Fleet
"""

import numpy as np

from gcc import GCC

class NB(GCC):
    def __init__(self, num_features, num_classes):
        """ This class represents a Naive Bayes model where the generative model is modeled as a Gaussian.
        NOTE: We assume lables are 0 to K - 1, where K is number of classes.

        TODO: Implement the methods of this class:
        - train: ndarray, ndarray -> None

        Implementation description will be provided below under each method.
        
        For the following:
        - N: Number of samples.
        - D: Dimension of input features vectors.
        - K: Number of classes.

        We have three parameters to keep track of:
        - self.means (ndarray (shape: (K, D))): Mean for each of K Gaussian likelihoods.
        - self.covariances (ndarray (shape: (K, D, D))): Covariance for each of K Gaussian likelihoods.
        - self.priors (shape: (K, 1))): Prior probabilty of drawing samples from each of K class.

        Args:
        - num_features (int): The number of features in the input vector
        - num_classes (int): The number of classes in the task.
        """
        super().__init__(num_features, num_classes)

    def train(self, train_X, train_y):
        """ This trains the parameters of the NB model, given training data.

        Args:
        - train_X (ndarray (shape: (N, D))): NxD matrix storing N D-dimensional training inputs.
        - train_y (ndarray (shape: (N, 1))): Column vector with N scalar training outputs (labels).
                                             NOTE: train_y is a vector of scalar values. You might represent train_y with one-hot encoding.
        """
        assert len(train_X.shape) == len(train_y.shape) == 2, f"Input/output pairs must be 2D-arrays. train_X: {train_X.shape}, train_y: {train_y.shape}"
        (N, D) = train_X.shape
        assert N == train_y.shape[0], f"Number of samples must match for input/output pairs. train_X: {N}, train_y: {train_y.shape[0]}"
        assert D == self.D, f"Expected {self.D} features. Got: {D}"
        assert train_y.shape[1] == 1, f"train_Y must be a column vector. Got: {train_y.shape}"

        # ====================================================
        # TODO: Implement your solution within the box
        classes = np.unique(train_y)

        for i in classes:
            checking = np.empty([N], dtype = bool)  #N*1
            for j in range(N):
                if train_y[j] == i:
                    checking[j] = True
                else:
                    checking[j] = False
            self.priors[i] = np.sum(checking)/N #2*1 D*1
            
            self.means[i] = np.mean(train_X[checking], axis=0) #K*D
            
            self.covariances[i] = np.var(train_X[checking], axis=0) #K*D*D    
            I = np.identity(D)
            self.covariances[i] = I*self.covariances[i]
        # ====================================================

        assert self.means.shape == (self.K, self.D), f"means shape mismatch. Expected: {(self.K, self.D)}. Got: {self.means.shape}"
        assert self.covariances.shape == (self.K, self.D, self.D), f"covariances shape mismatch. Expected: {(self.K, self.D, self.D)}. Got: {self.covariances.shape}"
        assert self.priors.shape == (self.K, 1), f"priors shape mismatch. Expected: {(self.K, 1)}. Got: {self.priors.shape}"
