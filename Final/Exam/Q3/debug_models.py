"""
CSCC11 - Introduction to Machine Learning, Winter 2021, Exam
B. Chan
"""

import _pickle as pickle
import numpy as np
import os

from gcc import GCC
from nb import NB
from utils import load_pickle_dataset

BASE_PATH = "./debug_data"

def debug_d_dim_k_class(d, k):
    print("=" * 75)
    print(f"Testing GCC with {d} dimensional inputs and {k} classes")
    print("=" * 75)

    filename = os.path.join(BASE_PATH, f"{d}_dim-{k}_class.pkl")
    gt_data = load_pickle_dataset(filename)

    for constructor in (GCC, NB):
        print(f"Model: {constructor.__name__}")
        data = gt_data[constructor.__name__.lower()]
        model = constructor(d, k)

        model.train(gt_data["X"], gt_data["y"])
        correct_means = np.allclose(model.means, data["means"])
        correct_covariances = np.allclose(model.covariances, data["covariances"])
        correct_priors = np.allclose(model.priors, data["priors"])

        model_correct_params = GCC(d, k)
        model_correct_params.means = data["means"]
        model_correct_params.covariances = data["covariances"]
        model_correct_params.priors = data["priors"]
        model_probs = model_correct_params.predict(gt_data["X"])
        correct_predictions = np.allclose(model_probs, data["predictions"])

        print(f"Correct Means: {correct_means}")
        print(f"Correct Covariances: {correct_covariances}")
        print(f"Correct Priors: {correct_priors}")
        print(f"Correct Predictions: {correct_predictions}")

        print("Details:")
        if not correct_means:
            print("Expected Mean:")
            print(data["means"])
            print("Got:")
            print(model.means)

        if not correct_covariances:
            print("Expected Covariances:")
            print(data["covariances"])
            print("Got:")
            print(model.covariances)

        if not correct_priors:
            print("Expected Priors:")
            print(data["priors"])
            print("Got:")
            print(model.priors)

        if not correct_predictions:
            print("Expected Predictions:")
            print(data["predictions"])
            print("Got:")
            print(model_probs)

        print("=" * 75)
    print("\n")

if __name__ == "__main__":
    debug_d_dim_k_class(2, 2)
    debug_d_dim_k_class(4, 2)
    debug_d_dim_k_class(4, 4)
