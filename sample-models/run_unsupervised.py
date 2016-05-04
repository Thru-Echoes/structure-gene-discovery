import scipy
from scipy import stats
from scipy import special
import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import palettable as pal
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import cross_decomposition
from sklearn import linear_model
from sklearn.decomposition import TruncatedSVD
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop, SGD
from keras import backend as K

from sklearn.manifold import TSNE
from sklearn.preprocessing import Imputer

from run_random_data import *
from pca_kmeans import *

def unsuper_linreg(X, properties, concentration):
    lm = linear_model.LinearRegression()
    lm.fit(X, preprocessing.scale(np.array(properties)))
    see_lm_score = lm.score(X, preprocessing.scale(np.array(properties)))

    print
    print("Lin Reg score: ", see_lm_score)
    ### R Squared

    R2 = 1 - ((lm.predict(X) - preprocessing.scale(np.array(properties))) ** 2).sum(axis = 0) / ((preprocessing.scale(np.array(properties))) ** 2).sum(axis = 0)

    print
    print("And R2 score: ", R2)
    return True

def run_tsne(properties, concentration):

    return True

def run_kpca(properties, concentration):

    run_pca = decomposition.PCA(n_components = n_class)
    pca_fit = run_pca.fit(x_data)
    #pca_fit
    x_pca = run_pca.transform(x_data);
    #pca_cov = run_pca.get_covariance(x_pca)
    #pca_score = run_pca.score(x_data)
    pca_noise = pca_fit.noise_variance_
    pca_var_explained = pca_fit.explained_variance_ratio_

    return x_pca, pca_noise, pca_var_explained

def run_autoencode(properties, concentration):

    ae = Sequential()
    ae.add(Dense(10, input_dim = concentration.shape[1]))
    ae.add(Activation('relu'))
    ae.add(Dense(concentration.shape[1]))
    ae.add(Activation('softmax'))
    ae.compile(optimizer = SGD(lr = .01, momentum = .9, decay = 0.001, nesterov = True), loss = 'categorical_crossentropy')

    x = preprocessing.scale(np.array(concentration))
    y = np.array(concentration)
    ae.fit(x, y, batch_size = 100, nb_epoch = 10, verbose = 1)

    np.mean(x)

    get_code = K.function([ae.layers[0].input], [ae.layers[1].output])
    code = get_code([x])[0]

    lm3 = linear_model.LinearRegression()
    lm3.fit(code, preprocessing.scale(np.array(properties)))
    see_lm3_score = lm3.score(code, preprocessing.scale(np.array(properties)))

    print
    print ("Lin Ref score with autoencoder: ", see_lm3_score)
    print

    return True

def main_execute():
    print
    print("Here we are in main_execute for example run_unsuper")
    print

    #properties = pd.DataFrame.from_csv("create-sim-data/default_parameters_sim_species_prop.csv")

    #concentration = pd.DataFrame.from_csv("../create-sim-data/simulated_counts.csv")
    #concentration = pd.DataFrame.from_csv("create-sim-data/default_parameters_sim_gene_counts.csv")
    x_arr = pd.DataFrame.from_csv("row_col_normalized_concentration.csv")
    x_arr = Imputer().fit_transform(x_arr)

    #conc_row_sum = concentration.copy()
    #conc_col_sum = concentration.copy()

    #for i in range(conc_row_sum.shape[0]):
    #        conc_row_sum.iloc[i, :] = conc_row_sum.iloc[i, :] / conc_row_sum.iloc[i, :].sum()
    #        concentration.iloc[i, :] = concentration.iloc[i, :] / concentration.iloc[i, :].sum()

    #for i in range(conc_col_sum.shape[1]):
    #        conc_col_sum.iloc[:, i] = conc_col_sum.iloc[:, i] / conc_col_sum.iloc[:, i].sum()
    #        concentration.iloc[:, i] = concentration.iloc[:, i] / concentration.iloc[:, i].max()


    #print
    #print("Conc_row_sum: ", conc_row_sum)
    #print
    #print("conc_col_sum: ", conc_col_sum)
    #print
    #print("concentration: ", concentration)
    #print

    #x_arr = np.array(concentration)
    #np.savetxt("row_col_normalized_concentration.csv", x_arr, delimiter = ",")

    x_reduced = TruncatedSVD(n_components = 50, random_state = 0).fit_transform(x_arr)

    print
    print("Finished TruncatedSVD")
    print
    print("Trying TSNE...")

    tsne_2pc = TSNE(n_components = 2, random_state = 0, verbose = 3).fit_transform(x_reduced)

    print
    print("Finished TSNE")
    print
    print("Trying clustering on 2 PC of t-SNE...")




    #x_pca, pca_noise, pca_var_explained = do_pca(concentration, 10)

    #print
    #print("---")
    #print("x_pca: ", x_pca)
    #print("pca_noise (estimated noise covariance): ", pca_noise)
    #print("pca_var_explained: ", pca_var_explained)
    #print("---")
    #print

    #unsuper_linreg(x_pca, properties, concentration)

    # run_kpca(properties, concentration)

    # run_autoencode(properties, concentration)

    # run_tsne(properties, concentration)

    return True
