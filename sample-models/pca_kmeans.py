import scipy
from scipy import stats
from scipy import special
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

def generate_rand():
    """Generate 10k x 5 random data"""
    n_cols = 5
    n_rows = 10000
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

def do_lsvm(x_data):
    """Function to do linear SVM"""
    return True

def do_rbfsvm(x_data):
    """Function to do radial basis function kernel SVM"""
    
    return True
