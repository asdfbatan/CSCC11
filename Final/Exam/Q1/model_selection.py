"""
CSCC11 - Introduction to Machine Learning, Winter 2021, Exam
B. Chan
"""

import numpy as np
import matplotlib.pyplot as plt

from polynomial_regression import PolynomialRegression
from utils import load_pickle_dataset, mean_squared_error

def model_selection(dataset, seed=0):
    """ This function helps select the best polynomial regression model for the dataset.

	This code generates a plot training/validation error curves, where the x-axis corresponds
    to the degree of polynomial used, and the y-axis corresponds to the error.

	NOTE: You will need to finish polynomial_regression.py before running this script.

    Args:
    - dataset (str): The path to the dataset that is in form of:
                     {"train_X": np.ndarray, "train_y": np.ndarray, "val_X": np.ndarray, "val_y": np.ndarray}
    """
    data = load_pickle_dataset(file_path=dataset)
    train_X = data["train_X"]
    train_y = data["train_y"]
    val_X = data["val_X"]
    val_y = data["val_y"]
    training_errors = []
    validation_errors = []
    K = 2

    for k in range(1, K + 1):
        model = PolynomialRegression(k)
        model.fit(train_X, train_y)
        train_mse = mean_squared_error(model.predict(train_X), train_y)
        val_mse = mean_squared_error(model.predict(val_X), val_y)
        training_errors.append(train_mse)
        validation_errors.append(val_mse)

    # Plot error curves
    range_x = range(1, K + 1)
    plt.plot(range_x, training_errors, label="Training", marker="o")
    plt.plot(range_x, validation_errors, label="Validation", marker="o")
    plt.title("Multivariate Polynomial Regression on Oil Dataset")
    plt.xlabel("Degree of Polynomial")
    plt.ylabel("Mean Squared Errors")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dataset = "./data/oil_500.pkl"
    seed = 0
    model_selection(dataset=dataset, seed=seed)
