"""
Takes in a matrix of organism properties and simulates gene expression distribution

"""

import sys
import scipy
from scipy import stats
from scipy import special
from scipy.stats import beta
import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import palettable as pal
import argparse

# input parameters

def simulate_species_properties(n_props, n_species, n_group):
	"""
	Simulate organism properites in a matrix, where the columns are the transcripts
	and the rows are the species.

	inputs:
	n_transcripts: number of transcripts/genes, ~50k
	n_species: number of species, >100
	prop_core: proportions of genes that are present in all species, ~10% of species

	n_props: ~10
	n_species: ~100
	n_group: ~3, the number of similar species in groups for each properties,
	 like phylum

	outputs:
	species_prop: a matrix of values [0-1] measuring the intensity of properties
	 each species have. Species stands for row, and properties are in the head of
	 each column.
	"""
    # assign group index randomly
	gp_sign = pd.DataFrame(np.random.randint(n_group, size=(n_species,n_props)))

    # assign break in the property range randomly
	gp_break = np.random.random([n_props, n_group, 2])

    # generate matrix according to group and threshold
	species_prop = gp_sign.copy() # actual intensity of the properties
	for i in range(n_props):
	    for j in range(n_species):
	        group_num = gp_sign.iloc[j,i]
	        species_prop.iloc[j,i] = np.random.uniform(gp_break[i, group_num].min(),
	                                                       gp_break[i,group_num].max())

	return species_prop, gp_sign, gp_break


def read_input(input):
	# Creates a pandas dataframe with species as row, properties as cols

	df = pd.DataFrame.from_csv(input)
	return df

def property_beta_distribution(n_props, beta_prop_1, beta_prop_2):
	# Samples from beta probability distribution
	# Output: matrix of probabilities (size n_props)

	property_probs = beta.rvs(beta_prop_1, beta_prop_2, size = n_props)
	probabilities = np.array(property_probs)
	probs_shape = probabilities.reshape(1, n_props)
	return probs_shape


def gene_beta_distribution(n_genes, beta_gene_1, beta_gene_2):
	# Samples from beta probability distribution
	# Output: matrix of probabilities (size n_genes)

	gene_probs = beta.rvs(beta_gene_1, beta_gene_2, size = n_genes)
	genes = np.array(gene_probs)
	properties_genes = genes.reshape(n_genes, 1)
	return properties_genes


def coin_toss(probability_distribution):
	# Samples from binomial distribution

	coin_toss_data = scipy.stats.binom.rvs(1,probability_distribution)
	return coin_toss_data

def norm_distribution(coin_toss_data, n_genes, n_props):
	# Samples from a normal (Gaussian) distribution, for all coin_toss data

	effect = random.normal(size=np.size(coin_toss_data))
	effect_shape = effect.reshape(n_genes,n_props)
	prop_genes = pd.DataFrame(effect_shape * coin_toss_data)
	return prop_genes

def get_counts(species_properties, prop_genes, multi_fact, add_fact, num_flips):
	# Calculates slope from product of species properties data and distribution sampling

	prop_genes_t = prop_genes.transpose()
	property_effects = np.dot(species_properties, prop_genes_t)
	# Use exp to ensure that all the counts will be positive
	expected_counts = np.exp(property_effects * multi_fact + add_fact)
	# Define negative binomial parameters, n and p
	p = num_flips / (num_flips + expected_counts)
	# Randomly fill in the observed counts with negative binomial based on n and p
	counts = expected_counts * 0 # start empty
	for i in range(0, counts.shape[0]):
		counts[i] = scipy.stats.nbinom.rvs(n = num_flips, p = p[i], size = expected_counts.shape[1])
	count_df = pd.DataFrame(counts, dtype="int")
	return count_df

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# >>> python simulate_counts.py -o FILE_NAME -ng NUM_GENES -ns NUM_SPECIES -np NUM_PROPS
	# ... -mf MULTIPLICATIVE_FACTOR -af ADDITIVE_FACTOR -nf NUMBER_BINOMIAL_FLIPS
	parser.add_argument("-o", "--output", help = "Set output file name")
	parser.add_argument("-ng", "--n_genes", help = "Set number of genes", type = int)
	parser.add_argument("-ns", "--n_species", help = "Set number of species", type = int)
	parser.add_argument("-np", "--n_props", help = "Set the number of hidden genetic properties in the species to be discovered", type = int)
	parser.add_argument("-mf", "--multi_fact", help = "Set multiplicative factor for negative binomial distribution", type = int)
	parser.add_argument("-af", "--add_fact", help = "Set additive factor for negative binomial distribution")
	parser.add_argument("-nf", "--num_flips", help = "Set number of flips for negative binomial distribution")
	parser.add_argument("-b1", "--beta_gene_1", help = "Set beta distribution a value for genes")
	parser.add_argument("-b2", "--beta_gene_2", help = "Set beta distribution b value for genes")
	parser.add_argument("-p1", "--beta_prop_1", help = "Set beta distribution a value for props")
	parser.add_argument("-p2", "--beta_prop_2", help = "Set beta distribution a value for props")
	parser.add_argument("-ngp", "--n_group", help = "number of phylogenetic groups")

	args = parser.parse_args()

	if args.output is None:
		output_filename = "counts"
	else:
		output_filename = args.output

	if args.n_genes is None:
		n_genes = 1000
	else:
		n_genes = int(args.n_genes)

	if args.beta_gene_1 is None:
		beta_gene_1 = 5
	else:
		beta_gene_1 = int(args.beta_gene_1)

	if args.beta_gene_2 is None:
		beta_gene_2 = 5
	else:
		beta_gene_2 = int(args.beta_gene_2)

	if args.beta_prop_1 is None:
		beta_prop_1 = 1
	else:
		beta_prop_1 = int(args.beta_prop_1)

	if args.beta_prop_2 is None:
		beta_prop_2 = 5
	else:
		beta_prop_2 = int(args.beta_prop_2)

	if args.n_group is None:
		n_group = 3
	else:
		n_group = int(args.n_group)

	if args.n_species is None:
		n_species = 100
	else:
		n_species = int(args.n_species)

	if args.n_props is None:
		n_props = 10
	else:
		n_props = int(args.n_props)

	if args.multi_fact is None:
		multi_fact = 6
	else:
		multi_fact = int(args.multi_fact)

	if args.add_fact is None:
		add_fact = 2
	else:
		add_fact = int(args.add_fact)

	if args.num_flips is None:
		num_flips = 2
	else:
		num_flips = int(args.num_flips)

	try:
		#species_properties = read_input(input_file)
		species_prop, gp_sign, gp_break = simulate_species_properties(n_props, n_species, n_group)
		n_props = len(species_prop.axes[1])
		properties = property_beta_distribution(n_props, beta_prop_1, beta_prop_2)
		genes = gene_beta_distribution(n_genes, beta_gene_1, beta_gene_2)
		probability_distribution = np.dot(genes, properties)
		coin_toss_data = coin_toss(probability_distribution)
		prop_genes = norm_distribution(coin_toss_data, n_genes, n_props)
		counts = get_counts(species_prop, prop_genes, multi_fact, add_fact, num_flips)

		### Write to CSV files
		# output_filename
		counts.to_csv(output_filename + "_sim_gene_counts.csv")
		prop_genes.to_csv(output_filename + "_sim_prop_genes.csv")
		species_prop.to_csv(output_filename + "_sim_species_prop.csv")

	except IOError as e:
		print "I/O error({0}): {1}".format(e.errno, e.strerror)
	except ValueError:
		print "Some value error."
	except:
		print "Unexpected error:", sys.exc_info()[0]
		raise
