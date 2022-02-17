"""
CSCC11 - Introduction to Machine Learning, Winter 2021, Assignment 3
B. Chan, E. Franco, D. Fleet

This script runs an experiment on the Lending Club dataset.
It fetches hyperparameters LENDING_CLUB_HYPERPARAMETERS from hyperparameters.py 
and check model's train, validation, and test accuracies over 10 different seeds.
NOTE: As a rule of thumb, each seed should take no longer than 5 minutes.
"""

import _pickle as pickle
import numpy as np

from experiments import run_experiment
from hyperparameters import LENDING_CLUB_HYPERPARAMETERS

def main(final_hyperparameters):
    with open("./datasets/lending_club.pkl", "rb") as f:
        lc_data =  pickle.load(f)

    # Train data
    train_X = lc_data['train_X']
    train_y = lc_data['train_y']
    
    # Validation data
    validation_X = lc_data['validation_X']
    validation_y = lc_data['validation_y']

    # Test data
    test_X, test_y = None, None
    if final_hyperparameters:
        test_X = lc_data['test_X']
        test_y = lc_data['test_y']

    # You can try different seeds and check the model's performance!
    seeds = np.random.RandomState(0).randint(low=0, high=65536, size=(10))

    train_accuracies = []
    validation_accuracies = []
    test_accuracies = []

    LENDING_CLUB_HYPERPARAMETERS["debug"] = False
    LENDING_CLUB_HYPERPARAMETERS["num_classes"] = 50
    for seed in seeds:
        LENDING_CLUB_HYPERPARAMETERS["rng"] = np.random.RandomState(seed)

        train_accuracy, validation_accuracy, test_accuracy = run_experiment(LENDING_CLUB_HYPERPARAMETERS,
                                                                            train_X,
                                                                            train_y,
                                                                            validation_X,
                                                                            validation_y,
                                                                            test_X,
                                                                            test_y)

        print(f"Seed: {seed} - Train Accuracy: {train_accuracy} - Validation Accuracy: {validation_accuracy} - Test Accuracy: {test_accuracy}")
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)
        test_accuracies.append(test_accuracy)

    print(f"Train Accuracies - Mean: {np.mean(train_accuracies)} - Standard Deviation: {np.std(train_accuracies, ddof=0)}")
    print(f"Validation Accuracies - Mean: {np.mean(validation_accuracies)} - Standard Deviation: {np.std(validation_accuracies, ddof=0)}")
    print(f"Test Accuracies - Mean: {np.mean(test_accuracies)} - Standard Deviation: {np.std(test_accuracies, ddof=0)}")


if __name__ == "__main__":
    final_hyperparameters = True
    main(final_hyperparameters=final_hyperparameters)
