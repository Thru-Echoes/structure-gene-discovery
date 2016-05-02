import os
import os.path
from os.path import basename
import subprocess
from subprocess import Popen, PIPE
import scipy
from scipy import stats
from scipy import special
import numpy as np
import pandas as pd
import seaborn as sns

## script takes in properties of species, e.g. requires lipids, requires sunlight, heterotrophic
# creates gene expression values
# variance of expression values depend on properties

def read_input():
# reads in data, parses
	return species_properties

def prob_function(array,n_properties):
	# rows
	genes=25000
	# columns
	properties = 20
	gene_probs = beta.rvs(5, 5, size=genes)
	a = np.array(gene_probs)
	a_shape = a.reshape(genes,1)
	print a_shape
	a_shape.shape
        rows = array[n_properties]
        x = scipy.stats.beta(array)
	property_probs = beta.rvs(1,5,size=properties)
	b = np.array(property_probs)
	b_shape= b.reshape(1,properties)
	print b_shape
	b_shape.shape
	probs = np.dot(a_shape,b_shape)
	probs.shape
	mask = scipy.stats.binom.rvs(1,probs)
	type(mask)
	sns.pairplot(mask,size=2.5)
	effect = numpy.random.normal(length(),sd = x_binary + 0.1))

def execute(input):
	#array=read_input()
		
	# effect sizes for that property on the gene
	
	return output


# input = file 






execute() 

