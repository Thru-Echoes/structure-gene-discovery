import sys
import scipy
from scipy import stats
from scipy import special
import numpy as np
import pandas as pd

sys.path.insert(0, 'create-sim-data')
from simulate_counts import *
# point to sample-models/ dir for import
sys.path.insert(0, 'sample-models')
from run_random_data import *
from pca_kmeans import *
from run_unsupervised import *

def main(args = None):
    """Main - serves project"""
    if args is None:
        args = sys.argv[1:]

    ## Run with simulated data (reflecting biological data)
    #simulated_counts()

    ## Run with random generated data
    #do_random_data()

    ## Run with Pseudomon. data
    main_execute()

if __name__ == "__main__":
    main()
