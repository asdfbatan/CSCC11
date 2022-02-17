"""
CSCC11 - Introduction to Machine Learning, Winter 2021, Exam
B. Chan, D. Fleet
"""

import numpy as np

class GCC:
    def __init__(self, num_features, num_classes):
        """ This class represents a Gaussian Class Conditional model.
        NOTE: We assume lables are 0 to K - 1, where K is number of classes.

        TODO: Implement the methods of this class:
        - train: ndarray, ndarray -> None
        - predict: ndarray -> ndarray

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

        self.D = num_features
        self.K = num_classes

        # Shape: K x D
        self.means = np.zeros((self.K, self.D))

        # Shape: K x D x D
        self.covariances = np.tile(np.eye(self.D), reps=(self.K, 1, 1))

        # Shape: K x 1
        self.priors = np.ones(shape=(self.K, 1)) / self.K

    def train(self, train_X, train_y):
        """ This trains the parameters of the GCC model, given training data.

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
            self.covariances[i] = np.cov(train_X[checking].T) #K*D*D

        # ====================================================

        assert self.means.shape == (self.K, self.D), f"means shape mismatch. Expected: {(self.K, self.D)}. Got: {self.means.shape}"
        assert self.covariances.shape == (self.K, self.D, self.D), f"covariances shape mismatch. Expected: {(self.K, self.D, self.D)}. Got: {self.covariances.shape}"
        assert self.priors.shape == (self.K, 1), f"priors shape mismatch. Expected: {(self.K, 1)}. Got: {self.priors.shape}"

    def predict(self, X):
        """ This computes the probability of each class given X, a matrix of input vectors.

        Args:
        - X (ndarray (shape: (N, D))): NxD matrix with N D-dimensional inputs.

        Output:
        - probs (ndarray (shape: (N, K))): NxK matrix storing N K-vectors (i.e. the K class probabilities)
        """
        assert len(X.shape) == 2, f"Input/output pairs must be 2D-arrays. X: {X.shape}"
        (N, D) = X.shape
        assert D == self.D, f"Expected {self.D} features. Got: {D}"

        unnormalized_probs = np.zeros((N, self.K))
        # ====================================================
        # TODO: Implement your solution within the box
        
        #This is calculating with likelihood but get wrong result don't know why
        probs = np.zeros((N, self.K))
        for i in range(N):
            for j in range(self.K):
                data_x = X[i]
                mean = self.means[j]
                prior = self.priors[j]
                covariance = self.covariances[j]
                inv_covariance = np.linalg.inv(covariance)
                det_covariance = np.linalg.det(covariance)
                
                inside_exp = -0.5*(data_x-mean)@inv_covariance@(data_x-mean).T
                
                coefficient = np.power(2*np.pi, self.D)*det_covariance  
                log = -0.5*np.log(coefficient)+inside_exp
                posterior = np.exp(log)*prior

                unnormalized_probs[i][j] = posterior
        
        for i in range(N):
            for j in range(self.K):
                row_sum = np.sum(unnormalized_probs, axis=1)[i]
                probs[i][j] = unnormalized_probs[i][j]/row_sum
        '''
        #This is calculating without likelihood but gt the correct result don't know why
        probs = np.zeros((N, self.K))
        for i in range(N):
            for j in range(self.K):
                data_x = X[i]
                mean = self.means[j]
                prior = self.priors[j]
                covariance = self.covariances[j]
                inv_covariance = np.linalg.inv(covariance)
                det_covariance = np.linalg.det(covariance)
                
                inside_exp = -0.5*(data_x-mean)@inv_covariance@(data_x-mean).T
                
                coefficient = 1/np.sqrt(np.power(2*np.pi, self.D)*det_covariance)
                unnormalized_probs[i][j] = coefficient*np.exp(inside_exp)
        
        for i in range(N):
            for j in range(self.K):
                row_sum = np.sum(unnormalized_probs, axis=1)[i]
                probs[i][j] = unnormalized_probs[i][j]/row_sum
        '''
        # ====================================================

        return probs
