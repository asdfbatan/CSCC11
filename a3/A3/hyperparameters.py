"""
CSCC11 - Introduction to Machine Learning, Winter 2021, Assignment 3
B. Chan, E. Franco, D. Fleet

This file specifies the hyperparameters for the two real life datasets.
Note that different hyperparameters will affect the runtime of the 
algorithm.
"""

# ====================================================
# TODO: Use Validation Set to Tune hyperparameters for the Lending Club dataset
# Use Optimal Parameters to get good accuracy on Test Set
LENDING_CLUB_HYPERPARAMETERS = {
    "num_trees": 50,
    "features_percent": 0.75,
    "data_percent": 0.75,
    "max_depth": 8,
    "min_leaf_data": 10,
    "min_entropy": 0.3,
    "num_split_retries": 10
}
# ====================================================

# ====================================================
# TODO: Use Validation Set to Tune hyperparameters for the Occupancy dataset
# Use Optimal Parameters to get good accuracy on Test Set
OCCUPANCY_HYPERPARAMETERS = {
    "num_trees": 10,
    "features_percent": 0.75,
    "data_percent": 0.75,
    "max_depth": 8,
    "min_leaf_data": 10,
    "min_entropy": 0.4,
    "num_split_retries": 10
}
# ====================================================
