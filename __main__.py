import sys
import scipy
from scipy import stats
from scipy import special
import numpy as np
import pandas as pd

from Simulate import simulate_species_properties
# point to sample-models/ dir for import
sys.path.insert(0, 'sample-models')
from pca_kmeans import *

def main(args = None):
    """Main - serves project"""
    if args is None:
        args = sys.argv[1:]


    ## Uncomment when ready to call simulate_species_properties

    #print("Calling simulate_species_properties()...")
    #species_prop, gp_sign, gp_break = simulate_species_properties()
    #species_prop.to_csv('Species_properites_likelihood.csv')
    #print("Finished.\n")
    #print("species_prop: ", species_prop)
    #print("gp_sign: ", gp_sign)
    #print("gp_break: ", gp_break, "\n")

    ## Comment when ready to call simulate_species_properties

    print("Calling sample file for generating data...")
    trial_data, n_class, n_feats = generate_rand()
    print("Finished.\n")
    #print("trial_data.shape: ", trial_data.shape)
    #print("trial_data: ", trial_data)

    ## Call PCA on x (features) of data to get variance coverage

    print("Calling do_pca...")
    sample_pca = do_pca(trial_data[:, 0:4], n_class, n_feats)
    print("Finished.\n")

    ## Call linear SVM

    print("Calling do_lsvm...")
    grid, best_lsvm = do_lsvm(trial_data[:, 0:4], trial_data[:, 5], n_class, n_feats)
    print("Finished.\n")
    print("The best parameters for lin-SVM are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_))

    ## Call Radial Basis Function (RBF) kernel SVM

    print("Calling do_rbfsvm...")
    grid = do_rbfsvm(trial_data[:, 0:4], trial_data[:, 5], n_class, n_feats)
    print("Finished.\n")
    print("The best parameters for kernel-SVM are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_))

if __name__ == "__main__":
    main()
