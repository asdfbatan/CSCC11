"""
CSCC11 - Introduction to Machine Learning, Winter 2021, Exam
B. Chan, D. Fleet
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

def accuracy(probs, target):
    """ This function computes the accuracy given the prediction probability and 
    ground truth label.

    Args:
    - probs (ndarray (shape: (N, K))): A NxK matrix consisting N K-probabilities for each input.
    - target (ndarray (shape: (N, 1))): A N-column vector consisting N labels.

    Output:
    - accuracy (float): The prediction accuracy.
    """
    preds = np.argmax(probs, axis=1)
    accuracy = 100 * np.mean(preds == target.flatten())
    return accuracy
