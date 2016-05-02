<strong>MDI Biological Labs. Barnraising workshop - (Maine) May 2016</strong>

This project is a collaborative effort between Lisa (Titus group - UC Davis), Harriet (Titus group - UC Davis), Dave Harris (U. Florida), Young (Princeton), and Oliver Muellerklein (me - Wayne Getz's group at UC Berkeley). We are working on a way to model structure in pathway emergence from a dataset of

# Structure / Pathway Discovery

Have a set of organisms. Have rows of genes. Columns of conditions / specifications / etc per gene. Background: perhaps there are pathways of gene expression we could discover that relate to a species / organism liking burgers and another pathway of genes that relate to liking mushrooms. There can be overlap in genes in pathways for liking burgers and liking mushrooms. So each set of pathways can be an

## 1. Summary / Overview

<strong>To-Do:</strong>

- [ ] Create Github repo and Slack channel
- [ ] Simulate organism properties = create simulated data (1000 rows x 500k cols)
- [ ] Create f(...) map projection of properties to expression levels
- [ ] Apply f() to properties for each species (i.e. each row in the simulated data)
- [ ] Run models: PCA, kernel-PCA / Spectral Clustering, multi-class clustering boosted + bagged, Autoencoder
- [ ] Model validity, feature importance
- [ ] Test model(s) on real data = does it work?
- [ ] Use of ensemble?

We will create a set of response classes in simulated to train on:

**Note: there are no classes in the test (real) data** - but creating the model will involve us training and testing validity on only simulated data. I.e. can we recover the simulated response classes from the simulated data?

These response classes;
E.g. *"likes burgers", "likes mushrooms"*

## 2. Data

We are going to use simulated data that represents rows of genes with columns of condition expression. Simulated data comes from a number of parameters of gene-properties-transcript relationships / complex. 

## 3. Model Approaches

Want to examine dimensional reduction methods. Want to examine the use of PCA, clustering or other dimensional reduction methods (e.g. kernel-PCA, spectral clustering), autoencoding, multi-class classification wiwth soft predictions (i.e. get probabilities per class instead of discrete predictions), auto-encoder.

<hr>

# Background

## Titus - structure gene discovery pitch

#### Overview

- matrix with rows of genes and cols of some conditions
- there may exist some hidden layer of interactions of these conditions
- these hidden layers would correspond to some pathways of conditional relationships (relationships of the features)

<hr>

<strong>Goals:</strong>

- find deep / hidden layer with either deep-NN and / or XGBoost
- visualize clustering of features via [t-SNE](https://lvdmaaten.github.io/tsne/)
