"""
Simulate organism properites in a matrix, where the columns are the transcripts
and the rows are the species.

inputs:
n_transcripts: number of transcripts/genes, ~50k
n_species: number of species, >100
prop_core: proportions of genes that are present in all species, ~10% of species

n_properties: ~10
n_species: ~100
n_group: ~3, the number of similar species in groups for each properties,
 like phylum

outputs:
species_prop: a matrix of values [0-1] measuring the intensity of properties
 each species have. Species stands for row, and properties are in the head of
 each column.
"""

import pandas as pd
import numpy as np

# input parameters

def simulate_species_properties(n_properties=10,n_species=100,n_group=3):
    # assign group index randomly
    gp_sign = pd.DataFrame(np.random.randint(n_group, size=(n_species,n_properties)))

    # assign break in the property range randomly
    gp_break = np.random.random([n_properties, n_group, 2])

    # generate matrix according to group and threshold
    species_prop = gp_sign.copy() # actual intensity of the properties
    for i in range(n_properties):
        for j in range(n_species):
            group_num = gp_sign.iloc[j,i]
            species_prop.iloc[j,i] = np.random.uniform(gp_break[i, group_num].min(),
                                                           gp_break[i,group_num].max())

    return species_prop, gp_sign, gp_break

if __name__ == "__main__":
    species_prop, gp_sign, gp_break = simulate_species_properties()
    species_prop.to_csv('Species_properites_likelihood.csv')
