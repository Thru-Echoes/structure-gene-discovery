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

import matplotlib.cm as cm

from sklearn.manifold import TSNE
from sklearn.preprocessing import Imputer

from run_random_data import *
from pca_kmeans import *

def unsuper_linreg(X, properties):
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

    #x_arr = pd.DataFrame.from_csv("row_col_normalized_concentration.csv")
    #x_arr = Imputer().fit_transform(x_arr)
    #raw = pd.DataFrame.from_csv('data/pseudomonas.tsv', sep='\t').transpose()
    raw = pd.DataFrame.from_csv('data/pseudomonas.tsv', sep='\t')
    y = np.array(raw)
    for i in range(raw.shape[0]):
        y[i,:] = y[i,:] / np.sum(y[i,:])
    x = preprocessing.scale(y)
    x_arr = x.copy()

    print
    print("x_arr.shape: ", x_arr.shape)

    ### Sample Pseudomonas data has 50 PCs to try to recover!
    x_pca, pca_noise, pca_var_explained = do_pca(x_arr, 10)

    print
    print("---")
    #print("x_pca: ", x_pca)
    print("pca_noise (estimated noise covariance): ", pca_noise)
    print("pca_var_explained: ", pca_var_explained)
    print("---")
    print

    #print("Now try PCA to 100 components for t-SNE...")

    #x_100pca, pca_100noise, pca_100var_explained = do_pca(x_arr, 100)
    #print
    #print("---")
    #print("x_150pca: ", x_150pca)
    #print("pca_100noise (estimated noise covariance): ", pca_100noise)
    #print("pca_100var_explained: ", pca_100var_explained)
    #print("---")
    #print

    #unsuper_linreg(x_pca, properties)

    #x_reduced = TruncatedSVD(n_components = 50, random_state = 0).fit_transform(x_arr)

    print
    print("Trying TSNE with 2 PCs from 10 PCA-PCs...")

    #tsne_default = TSNE(n_components = 2)
    #tsne_default_t = tsne_default.fit_transform(x_arr)

    tsne_10pca = TSNE(n_components = 2)
    tsne_10pca_t = tsne_10pca.fit_transform(x_pca)

    print
    print("tsne_10pca_t: ", tsne_10pca_t)
    print
    #print("tnse_default_t.embedding_: ", tsne_default_t.embedding_)
    #print("tsne_10pca_t[:, 0]: ", tsne_10pca_t[:, 0])
    #print
    #print("tsne_10pca_t[:, 1]: ", tsne_10pca_t[:, 1])
    print

    tsne_10pca_save_x = pd.DataFrame(tsne_10pca_t[:, 0])
    tsne_10pca_save_y = pd.DataFrame(tsne_10pca_t[:, 1])

    tsne_10pca_save_x.to_csv("sample-models/tsne_10pca_x.csv")
    tsne_10pca_save_y.to_csv("sample-models/tsne_10pca_y.csv")

    ### Save t-SNE plots in 2D
    #plt.scatter(tsne_default_t[:, 0], tsne_default_t[:, 1])
    #plt.savefig("sample-models/tsne_default_2d.png")

    plot_colors = cm.rainbow(np.linspace(0, 1, 50))

    plt.scatter(tsne_10pca_t[:, 0], tsne_10pca_t[:, 1], color = plot_colors)
    plt.savefig("sample-models/tsne_10pca_2d.png")

    #tsne_10pc = TSNE(n_components = 2, random_state = 0, verbose = 3, perplexity = 4).fit_transform(x_100pca)

    #print
    #print("Finished tnse-10pc with 100 PCs from PCA.")
    #print
    #print("Trying TSNE with 25 PCs from 100 PCA-PCs...")

    #tsne_25pc = TSNE(n_components = 25, random_state = 0, verbose = 3, perplexity = 4).fit_transform(x_100pca)

    #print
    #print("Finished tnse-25pc with 100 PCs from PCA.")
    #print

    #tsne_pca_2pc = TSNE(n_components = 2, random_state = 0, verbose = 3).fit_transform(x_pca)
    #tsne_2pc = TSNE(n_components = 2, random_state = 0, verbose = 3).fit_transform(x_arr)
    #tsne_2pc = TSNE(n_components = 2, random_state = 0, verbose = 3).fit_transform(x_reduced)

    print
    print("Finished TSNE")

    # run_kpca(properties, concentration)

    # run_autoencode(properties, concentration)

    # run_tsne(properties, concentration)

    return True
