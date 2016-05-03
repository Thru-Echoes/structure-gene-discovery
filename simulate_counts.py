"""
Takes in a matrix of organism properties and simulates gene expression distribution

"""

import scipy
from scipy import stats
from scipy import special
from scipy.stats import beta
import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import palettable as pal

def read_input(input):
	# Creates a pandas dataframe with species as row, properties as cols

	df = pd.DataFrame.from_csv(input)
	return df

def property_beta_distribution(n_properties):
	# Samples from beta probability distribution
	# Output: matrix of probabilities (size n_properties)

	property_probs = beta.rvs(1,5,size = n_properties)
	probabilities = np.array(property_probs)
	probs_shape = probabilities.reshape(1,n_properties)
	return probs_shape


def gene_beta_distribution(n_genes):
	# Samples from beta probability distribution
    	# Output: matrix of probabilities (size n_genes)

	gene_probs = beta.rvs(5, 5, size = n_genes)
	genes = np.array(gene_probs)
	properties_genes = genes.reshape(n_genes,1)
	return properties_genes
	

def coin_toss(probability_distribution):
	# Samples from binomial distribution
	
	coin_toss_data = scipy.stats.binom.rvs(1,probability_distribution)
	return coin_toss_data

def norm_distribution(coin_toss_data,n_genes,n_properties):
	# Samples from a normal (Gaussian) distribution, for all coin_toss data

	effect = random.normal(size=np.size(coin_toss_data))
	effect_shape = effect.reshape(n_genes,n_properties)
	mult = pd.DataFrame(effect_shape * coin_toss_data)
	return mult

def get_counts(species_properties,mult):
	# Calculates slope from product of species properties data and distribution sampling

	mult_t = mult.transpose()
	property_effects = np.dot(species_properties, mult_t)
	# Use exp to ensure that all the counts will be positive
	expected_counts = np.exp(property_effects*6 + 2)
	# Define negative binomial parameters, n and p
	n = 2
	p = n / (n + expected_counts)
	# Randomly fill in the observed counts with negative binomial based on n and p
	counts = expected_counts * 0 # start empty
	for i in range(0, counts.shape[0]):
  		counts[i] = scipy.stats.nbinom.rvs(n=n, p=p[i], size = expected_counts.shape[1])
	
	count_df = pd.DataFrame(counts, dtype="int")	
	return count_df

def execute(input_file,n_genes):
	# Main code:

	species_properties= read_input(input_file)
	n_properties=len(species_properties.axes[1])	
	properties=property_beta_distribution(n_properties)
	genes=gene_beta_distribution(n_genes)
	probability_distribution = np.dot(genes,properties)
	coin_toss_data = coin_toss(probability_distribution)	
	mult = norm_distribution(coin_toss_data,n_genes,n_properties)
	counts = get_counts(species_properties,mult)
	print counts
	return counts


input_file = "Species_properites_likelihood.csv"  
n_genes = 10000
counts = execute(input_file,n_genes) 
counts.to_csv("simulated_counts.csv")
print "Hell yeah! Counts simulated."
print "File: simulated_counts.csv"
