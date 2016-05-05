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
    raw = pd.DataFrame.from_csv('/mnt/pseudomonas.tsv', sep='\t')
    y = np.array(raw)
    for i in range(raw.shape[0]):
        y[i,:] = y[i,:] / np.sum(y[i,:])
    x = preprocessing.scale(y)
    x_arr = x.copy()

    """
    ae_node_assign = pd.DataFrame.from_csv('sample-models/only_node_assignment_per_gene_ae.csv', sep = ',')
    ae_nodes = np.array(ae_node_assign)

    ae_raw = pd.DataFrame.from_csv('pseudomonas/ae_codes.csv', sep=',').transpose()
    x_ae_codes = np.array(ae_raw)

    ae_weights_raw = pd.DataFrame.from_csv('pseudomonas/ae_weights.csv', sep = ',')
    ae_weights_raw_t = pd.DataFrame.from_csv('pseudomonas/ae_weights.csv', sep = ',').transpose()
    x_ae_weights = np.array(ae_weights_raw)
    x_ae_weights_t = np.array(ae_weights_raw_t)
    """

    ae_raw = pd.DataFrame.from_csv('pseudomonas/ae_codes.csv', sep=',')
    x_ae_codes = np.array(ae_raw)

    sample_labels = pd.read_csv('data/pseudo_sample_names.csv', sep = ',', header = None)
    np_labels = np.array(sample_labels)

    sample_colors = cm.rainbow(np.linspace(0, 1, 950))

    print
    print("Running t-SNE with the AE weight data (950 species x 50 nodes)...")

    tsne_ae_species = TSNE(n_components = 2).fit_transform(x_ae_codes)

    fig, axes = plt.subplots(1)
    fig.set_size_inches(30, 30)
    #plt.scatter(tsne_ae_species[:, 0], tsne_ae_species[:, 1], edgecolors='none')
    plt.scatter(tsne_ae_species[:, 0], tsne_ae_species[:, 1], c = sample_colors, edgecolors='none')
    #plt.colorbar(ticks = range(950))
    plt.show()
    plt.savefig("cluster-and-viz/may5_tsne_samples_by_nodes.png")
    plt.clf()

    #ae_y = np.array(ae_raw)
    #for i in range(ae_raw.shape[0]):
    #        ae_y[i,:] = ae_y[i,:] / np.sum(ae_y[i,:])
    #ae_x = preprocessing.scale(ae_y)
    #x_ae = ae_x.copy()


    ########################################################################
    ########################################################################
    """
    ### Sample Pseudomonas data has 50 PCs to try to recover!
    x_pca, pca_noise, pca_var_explained = do_pca(x_arr, 50)

    print
    print("---")
    #print("x_pca: ", x_pca)
    print("pca_noise (estimated noise covariance): ", pca_noise)
    print("pca_var_explained: ", pca_var_explained)
    print("---")
    print


    tsne_default = TSNE(n_components = 2).fit_transform(x_arr)

    plt.scatter(tsne_default[:, 0], tsne_default[:, 1], c = ae_nodes)
    plt.colorbar(ticks = range(50))
    plt.savefig("sample-models/may4_tsne_default_2d.png")
    #plt.show()

    plt.clf()

    tsne_pca = TSNE(n_components = 2).fit_transform(x_pca)
    plt.scatter(tsne_pca[:, 0], tsne_pca[:, 1], c = ae_nodes)
    plt.colorbar(ticks = range(50))
    plt.savefig("sample-models/may4_tsne_50pca_2d.png")
    #plt.show()

    plt.clf()
    """
    ########################################################################
    # Now break up t-SNE to plot 1k genes at a time (5 plots)
    ########################################################################
    """
    plt.scatter(tsne_default[0:999, 0], tsne_default[0:999, 1], c = ae_nodes[0:999])
    plt.colorbar(ticks = range(50))
    plt.savefig("sample-models/may4_tsne_fisrt1k_2d.png")
    plt.clf()

    print
    print("Finished first 1k...")

<<<<<<< HEAD
    tsne_default = TSNE(n_components = 2)
    tsne_default_t = tsne_default.fit_transform(x_arr)
    
    plt.scatter(tsne_default_t[:, 0], tsne_default_t[:, 1])
    plt.savefig("tsne_default_2d.png")

    #tsne_default_df = pd.DataFrame(tsne_default)
    #tsne_default_t_df = pd.DataFrame(tsne_default_t)

    
    #tsne_default_df.to_csv("tsne_default_model.csv")
    #tsne_default_t_df.to_csv("tsne_default_trans_model.csv")
=======
    plt.scatter(tsne_default[1000:1999, 0], tsne_default[1000:1999, 1], c = ae_nodes[1000:1999])
    plt.colorbar(ticks = range(50))
    plt.savefig("sample-models/may4_tsne_second1k_2d.png")
    plt.clf()

    print
    print("Finished second 1k...")
>>>>>>> 7ce053820ce87bbc45b6a1c029e47da7eb12c605

    plt.scatter(tsne_default[2000:2999, 0], tsne_default[2000:2999, 1], c = ae_nodes[2000:2999])
    plt.colorbar(ticks = range(50))
    plt.savefig("sample-models/may4_tsne_third1k_2d.png")
    plt.clf()

    print
    print("Finished third 1k...")

    plt.scatter(tsne_default[3000:3999, 0], tsne_default[3000:3999, 1], c = ae_nodes[3000:3999])
    plt.colorbar(ticks = range(50))
    plt.savefig("sample-models/may4_tsne_fourth1k_2d.png")
    plt.clf()


    print
    print("Finished fourth 1k...")

    plt.scatter(tsne_default[4000:5549, 0], tsne_default[4000:5549, 1], c = ae_nodes[4000:5549])
    plt.colorbar(ticks = range(50))
    plt.savefig("sample-models/may4_tsne_last1549_2d.png")
    plt.clf()

    """
    ########################################################################
    # Now use AE codings to do t-SNE visualization
    ########################################################################
    np_50colors = np.linspace(0, 50, 50)
    """
    print
    print("Now we are actually running t-SNE with the AE weight data (5549 genes x 50 nodes)...")

    tsne_ae_weights_t = TSNE(n_components = 2).fit_transform(x_ae_weights_t)
    plt.scatter(tsne_ae_weights_t[:, 0], tsne_ae_weights_t[:, 1], c = ae_nodes)
    plt.colorbar(ticks = range(50))
    plt.savefig("cluster-and-viz/may4_tsne_ae_weights_t_2d.png")
    plt.clf()


    print
    print("Now we are actually running t-SNE with the AE weight data (50 nodes x 5549 genes)...")

    #plot_colors = cm.rainbow(np.linspace(0, 1, 50))

    tsne_ae_weights = TSNE(n_components = 2).fit_transform(x_ae_weights)
    plt.scatter(tsne_ae_weights[:, 0], tsne_ae_weights[:, 1], c = np_50colors)
    plt.colorbar(ticks = range(50))
    plt.savefig("sample-models/may4_tsne_ae_weights_not_t_2d.png")
    plt.clf()

    print
    print("Now we are actually running t-SNE with the AE codings data (50 nodes x 950 samples)...")

    tsne_ae_codes = TSNE(n_components = 2).fit_transform(x_ae_codes)
    plt.scatter(tsne_ae_codes[:, 0], tsne_ae_codes[:, 1], c = np_50colors)
    plt.colorbar(ticks = range(50))
    plt.savefig("sample-models/may4_tsne_ae_codes_2d.png")
    plt.clf()
    """

    ########################################################################
    ########################################################################

    #unsuper_linreg(x_pca, properties)

    #x_reduced = TruncatedSVD(n_components = 50, random_state = 0).fit_transform(x_arr)

    #tsne_default = TSNE(n_components = 2)
    #tsne_default_t = tsne_default.fit_transform(x_arr)

    #tsne_10pca = TSNE(n_components = 2)
    #tsne_10pca_t = tsne_10pca.fit_transform(x_pca)

    #tsne_10pca_save_x = pd.DataFrame(tsne_10pca_t[:, 0])
    #tsne_10pca_save_y = pd.DataFrame(tsne_10pca_t[:, 1])

    #tsne_10pca_save_x.to_csv("sample-models/tsne_10pca_x.csv")
    #tsne_10pca_save_y.to_csv("sample-models/tsne_10pca_y.csv")

    ### Save t-SNE plots in 2D
    #plt.scatter(tsne_default_t[:, 0], tsne_default_t[:, 1])
    #plt.savefig("sample-models/tsne_default_2d.png")

    #plot_colors = cm.rainbow(np.linspace(0, 1, 50))

    #plt.scatter(tsne_10pca_t[:, 0], tsne_10pca_t[:, 1], color = plot_colors)
    #plt.savefig("sample-models/tsne_10pca_2d.png")

    print
    print("Finished TSNE")

    ########################################################################
    ########################################################################

    # run_kpca(properties, concentration)

    # run_autoencode(properties, concentration)

    # run_tsne(properties, concentration)
    return True
