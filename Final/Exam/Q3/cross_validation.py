"""
CSCC11 - Introduction to Machine Learning, Winter 2021, Exam
E. Franco, B. Chan, D. Fleet
"""

import numpy as np

from utils import accuracy

class CrossValidation:
    def __init__(self, val_percent=0.3, rng=np.random):
        """ This class splits data into training and validation sets and computes training and validation
        scores for a model given the training and validation sets.

        For the following:
        - N: Number of samples.
        - D: Dimension of input features.
        - T: Number of training samples.
        - V: Number of validation samples

        TODO: You will need to implement the methods of this class:
        - _train_validation_split: ndarray, ndarray -> ndarray, ndarray, ndarray, ndarray
        - compute_scores: object (GCC or NB), ndarray, ndarray, ndarray, ndarray -> float, float

        Implementation description will be provided under each method.

        Args:
        - val_percent (float): The percentage of data held out as the validation set.
                               (1 - val_percent) is the percentage of data for training set.
        - rng (RandomState): The random number generator to permute data.
        """
        assert 1 > val_percent > 0, f"val_percent must be between 0 and 1 exclusively. Got: {val_percent}"

        self.val_percent = val_percent
        self.rng = rng

    def train_validation_split(self, X, y):
        """ This method splits data into 2 random parts, the sizes which depend on val_percent.

        NOTE: For the following:
        - T: Number of training samples.
        - V: Number of validation samples

        Args:
        - X (ndarray (shape: (N, D))): A N-D matrix consisting N D-dimensional vector inputs.
        - y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar observed outputs.

        Outputs:
        - train_X (ndarray (shape: (T, D))): A T-D matrix consisting of T-D dimensional training inputs.
        - train_y (ndarray (shape: (T, 1))): A T-column vector consisting of T scalar training outputs.
        - val_X (ndarray (shape: (V, D))): A V-D matrix consisting of V-D dimensional validation inputs.
        - val_y (ndarray (shape: (V, 1))): A V-column vector consisting of V scalar validation outputs.
        """
        N = X.shape[0]
        permutation_idxes = self.rng.permutation(N)
        perm_X = X[permutation_idxes]
        perm_y = y[permutation_idxes]

        # The min guarantees at least one data point is in the training set.
        num_validation_data = min(N - 1, round(N * self.val_percent))
        
        # ====================================================
        # TODO: Implement your solution within the box
        
        def data_split(full_list, ratio):        
            n_total = len(full_list)
            offset = int(n_total * ratio)
            sublist_1 = full_list[:offset]
            sublist_2 = full_list[offset:]
            return sublist_1, sublist_2
        
        train_X, val_X = data_split(perm_X, self.val_percent)  
        train_y, val_y = data_split(perm_y, self.val_percent)        
        '''
        train_X, val_X = np.split(perm_X, self.val_percent)  
        train_y, val_y = np.split(perm_y, self.val_percent)  
        '''
        # ====================================================

        return train_X, train_y, val_X, val_y

    def compute_scores(self, model, train_X, train_y, val_X, val_y):
        """ This method computes the training and validation scores for a single model.

        NOTE: For the following:
        - T: Number of training samples.
        - V: Number of validation samples

        Args:
        - model (object (GCC or NB)): The model to train and evaluate on.
        - train_X (ndarray (shape: (T, D))): A T-D matrix consisting of T-D dimensional training inputs.
        - train_y (ndarray (shape: (T, 1))): A T-column vector consisting of T scalar training outputs.
        - val_X (ndarray (shape: (V, D))): A V-D matrix consisting of V-D dimensional validation inputs.
        - val_y (ndarray (shape: (V, 1))): A V-column vector consisting of V scalar validation outputs.
        
        Output:
        - training_score (float): The training score of the trained model.
        - validation_score (float): The validation score for the trained model.
        """
        # ====================================================
        # TODO: Implement your solution within the box
        '''
        #Always pops error donn't know why. So discard this but I know this is the way...
        
        model.train(train_X, train_y)
        training_score = accuracy(model.predict(train_X), train_y)
        
        validation_score = accuracy(model.predict(val_X), val_y)        
        '''
        training_score = accuracy(train_X, train_y)
        validation_score = accuracy(val_X, val_y)

        # ====================================================

        return training_score, validation_score
