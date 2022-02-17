"""
CSCC11 - Introduction to Machine Learning, Winter 2021, Exam
E. Franco, B. Chan, D. Fleet
"""

import numpy as np
import matplotlib.pyplot as plt

from cross_validation import CrossValidation
from gcc import GCC
from nb import NB
from pca import PCA
from utils import load_pickle_dataset

def model_selection(model_constructor, train_X, train_y, val_X, val_y, title):
    """ This function helps select a classification model for the input data.  For a given classification model (GCC or NB),
        we need to select the corresponding dimension (K) of the linear subspace (found with PCA) that provides 
 	    the low-dimensional classification input.

	It generates the training/validation score curves, where the x-axis corresponds to the number of 
    principal components used, and the y-axis corresponds to the score.

    It also displays the best cross validation score and the corresponding number of principal components for that model.

	TODO: In this function, you will need to implement the following:
	- Receive the training and validation scores over all number of principal components using the CrossValidation object for the GCC and NB models. 
	- Compute the best cross validation score and the corresponding number of principal components used.

	NOTE: You will need to finish gcc.py, nb.py, and cross_validation.py before completing this function.

    Args:
    - model_constructor (constructor (GCC or Naive Bayes)): The constructor of the model to train and evaluate.
    - train_X (ndarray (shape: (N, D))): A N-D matrix consisting of N-D dimensional inputs.
    - train_y (ndarray (shape: (N, 1))): A N-column vector consisting of N scalar outputs.
    - val_X (ndarray (shape: (M, D))): A M-D matrix consisting of M-D dimensional inputs.
    - val_y (ndarray (shape: (M, 1))): A M-column vector consisting of M scalar outputs.
    - title (str): The title of the plot.
    """
    training_scores = []
    validation_scores = []
    
    range_x = range(1, 125)
    num_classes = len(np.unique(train_y))
    for reduce_dim in range_x:
        # ====================================================
        # TODO: Implement your solution within the box
        # Receive training and validation scores
        '''
        cv = CrossValidation(val_percent=0.3, rng=np.random)
        model = model_constructor(reduce_dim, num_classes)
        pca = PCA(train_X)
        new_train_X = pca.reduce_dimensionality(train_X, reduce_dim)
        new_val_X = pca.reduce_dimensionality(val_X, reduce_dim)
        
        training_score, validation_score = cv.compute_scores(model, new_train_X, train_y, new_val_X, val_y)
        '''
        cv = CrossValidation(val_percent=0.3, rng=np.random)
        training_score, validation_score = cv.compute_scores(model_constructor, train_X[:,[0, reduce_dim]], train_y, val_X[:,[0, reduce_dim]], val_y)

        # ====================================================

        training_scores.append(training_score)
        validation_scores.append(validation_score)

    # ====================================================
    # TODO: Implement your solution within the box
    # Assign cv_scores and compute index of the best cross validation score.
    cv_scores = 0.5*(np.array(training_scores) + np.array(validation_scores))
    best_cv_idx = int(validation_scores.index(max(validation_scores)))    
    # ====================================================

    best_dim = list(range_x)[best_cv_idx]

    print(f"Model: {title}")
    print("Best Cross Validation Score: {}".format(cv_scores[best_cv_idx]))
    print("Number of Principal Components with Best Cross Validation Score: {}".format(best_dim))

    # Plot score curves
    plt.plot(range_x, training_scores, label="Training", marker="o")
    plt.plot(range_x, validation_scores, label="Validation", marker="o")
    plt.title(title)
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cross Validation Scores")
    plt.legend()
    plt.show()

    return best_dim

def visualize(eyes, non_eyes, rec_eyes, rec_non_eyes, title):
    img_shape = (20, 25)
    fig, axes = plt.subplots(4, 10, figsize=(10, 10))

    axes[0, 0].set_ylabel("Original Eye Images", size='medium')
    axes[1, 0].set_ylabel("Reconstructed Eye Images", size='medium')
    axes[2, 0].set_ylabel("Original Non Eye Images", size='medium')
    axes[3, 0].set_ylabel("Reconstructed Non Eye Images", size='medium')
    for idx, (eye, non_eye, rec_eye, rec_non_eye) in enumerate(zip(eyes, non_eyes, rec_eyes, rec_non_eyes)):
        for row in range(4):
            axes[row, idx].set_yticklabels('')
            axes[row, idx].set_xticklabels('')
            axes[row, idx].tick_params(axis='both',
                                       which='both',
                                       bottom=False,
                                       top=False,
                                       labelbottom=False)

        axes[0, idx].imshow(eye.reshape(img_shape).T)
        axes[1, idx].imshow(rec_eye.reshape(img_shape).T)
        axes[2, idx].imshow(non_eye.reshape(img_shape).T)
        axes[3, idx].imshow(rec_non_eye.reshape(img_shape).T)
    fig.suptitle(title)
    plt.show()

if __name__ == '__main__':
    dataset = "./data/eye_image_data.pkl"
    
    data = load_pickle_dataset(dataset)

    X = data["X"]
    y = data["y"]

    seed = 0
    val_percent = 0.3
    D = X.shape[1]
    
    # Cross Validation
    cv = CrossValidation(val_percent, np.random.RandomState(seed))
    train_X, train_y, val_X, val_y = cv.train_validation_split(X, y)
    non_eyes = np.where(train_y == 0)[0][:10]
    eyes = np.where(train_y == 1)[0][:10]

    # Apply PCA to training and validation sets
    pca = PCA(train_X)
    project_train_X = pca.reduce_dimensionality(train_X, D)
    project_val_X = pca.reduce_dimensionality(val_X, D)

    # GCC and NB train/validation
    for constructor in (GCC, NB):
        model_name = constructor.__name__
        best_dim = model_selection(constructor,
                                   project_train_X,
                                   train_y,
                                   project_val_X,
                                   val_y,
                                   model_name)
        
        # Visualization of images using best GCC dimensionality
        visualize(train_X[eyes],
                  train_X[non_eyes],
                  pca.reconstruct(project_train_X[eyes, :best_dim]),
                  pca.reconstruct(project_train_X[non_eyes, :best_dim]),
                  title=f"{model_name} - Best Dimensionality = {best_dim}")

    # Visualization using more dimensionality
    predefined_dim = 100
    visualize(train_X[eyes],
              train_X[non_eyes],
              pca.reconstruct(project_train_X[eyes, :predefined_dim]),
              pca.reconstruct(project_train_X[non_eyes, :predefined_dim]),
              title="Dimensionality = 100")
