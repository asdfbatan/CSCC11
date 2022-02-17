"""
CSCC11 - Introduction to Machine Learning, Winter 2021, Exam
B. Chan
"""

import _pickle as pickle
import numpy as np

def load_pickle_dataset(file_path):
    """ This function loads a pickle file given a file path.

    Args:
    - file_path (str): The path of the pickle file

    Output:
    - (dict): A dictionary consisting the dataset content.
    """
    return pickle.load(open(file_path, "rb"))

def mean_squared_error(predictions, observed_y):
    """ This function computes the mean squared error between the predictions and observed outputs.

    Args:
    - predictions (ndarray (shape: (N, 1))): A N-column vector consisting of N predictions.
    - observed_y (ndarray (shape: (N, 1))): A N-column vector consisting of N observed outputs.

    Output:
    - (float): The mean squared error between the predicted outputs and the observed outputs.
    """

    return np.mean((predictions - observed_y) ** 2)
