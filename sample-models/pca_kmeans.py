import scipy
from scipy import stats
from scipy import special
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV

def generate_rand():
    """Generate 10k x 5 random data"""
    n_cols = 5
    n_rows = 100
    n_class = 10
    trial_x = np.random.rand(n_rows, n_cols)
    trial_y = np.random.random_integers(1, n_class, size = (n_rows, 1))

    # Append response to data
    trial_data = np.append(trial_x, trial_y, 1)
    return trial_data, n_class, n_cols

def scale_data(x_data):
    """Function to scale the data"""

    # Scale based on maximum
    x_max = np.amax(x_data)
    scaled_data = x_data / x_max
    return scaled_data

def do_pca(x_data, n_class, n_feats):
    """Function to do PCA"""

    run_pca = PCA()
    pca_fit = run_pca.fit(x_data)
    #pca_cov = run_pca.get_covariance(x_data)
    #pca_score = run_pca.score(x_data)
    pca_noise = pca_fit.noise_variance_
    pca_var_explained = pca_fit.explained_variance_ratio_

    print("\n---")
    print("pca_noise (estimated noise covariance): ", pca_noise)
    print("pca_var_explained: ", pca_var_explained)
    print("---\n")

    return run_pca

def do_knn(x_data):
    """Function to do k-nearest neighbor clustering"""
    return True

def cv_lsvm(x_data, y_data, n_class, n_feats):
    """Function to calculate cross-validation for linear SVM

    Note: code form Scikit-learn

    Input:
    x_data = x data (features / predictors)
    y_data = y data (response classes / labels)
    n_class = number of classes (binary versus multiclass)
    n_feats = number of features (number of predictors / columns)
    """

    # Do standard scaling to all data (should do train and project unto test)
    #sample_scaler = StandardScaler()
    #x_scaled = sample_scaler.fit_transform(x_data)

    # Grid search CV to search for optimal model parameter C
    #C_range = np.logspace(-2, 10, 13)
    C_range = np.logspace(-2, 10, 5)
    param_grid = dict(C = C_range)
    cv = StratifiedShuffleSplit(y_data, n_iter = 3, test_size = 0.2, random_state = 0)
    grid = GridSearchCV(SVC(kernel = "linear"), param_grid = param_grid, cv = cv)
    grid.fit(x_data, y_data)

    return grid

def do_lsvm(x_data, y_data, bst_model):
    """Function to run linear SVM

    Input:
    x_data = x data (features / predictors)
    y_data = y data (response classes / labels)
    bst_model = best linear svm model from cross validation function
    """
    bst_lsvm = SVC(C = bst_model.best_estimator_.C, kernel = "linear")
    bst_lsvm_fit = bst_lsvm.fit(x_data, y_data)
    return True

def cv_rbfsvm(x_data, y_data, n_class, n_feats):
    """Function to calculate cross-validation for radial basis function kernel SVM

    Note: code form Scikit-learn

    Input:
    x_data = x data (features / predictors)
    y_data = y data (response classes / labels)
    n_class = number of classes (binary versus multiclass)
    n_feats = number of features (number of predictors / columns)
    """

    # Do standard scaling to all data (should do train and project unto test)
    sample_scaler = StandardScaler()
    x_scaled = sample_scaler.fit_transform(x_data)

    # Grid search CV to search for optimal model parameters C and gamma
    #C_range = np.logspace(-2, 10, 13)
    #gamma_range = np.logspace(-9, 3, 13)
    C_range = np.logspace(-2, 10, 5)
    gamma_range = np.logspace(-9, 3, 5)
    param_grid = dict(gamma = gamma_range, C = C_range)
    # do 80 / 20 - train / test split
    cv = StratifiedShuffleSplit(y_data, n_iter = 3, test_size = 0.2, random_state = 1)
    grid = GridSearchCV(SVC(), param_grid = param_grid, cv = cv)
    grid.fit(x_scaled, y_data)

    return grid

def do_rbfsvm(x_data, y_data, bst_model):
    """Function to perform Radial Basis Function kernel SVM

    Input:
    x_data = x data (features / predictors)
    y_data = y data (response classes / labels)
    bst_model = best rbfsvm model from cross validation function
    """

    bst_rbfsvm = SVC(C = bst_model.best_estimator_.C, gamma = bst_model.best_estimator_.gamma)
    bst_rbfsvm_fit = bst_lsvm.fit(x_data, y_data)
    return True
