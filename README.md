<strong>MDI Biological Labs. Barnraising workshop - (Maine) May 2016</strong>

This project is a collaborative effort between Lisa (Titus group - UC Davis), Harriet (Titus group - UC Davis), Dave Harris (U. Florida), Yuan (Princeton), and Oliver Muellerklein (me - Wayne Getz's group at UC Berkeley). We are working on a way to model structure in pathway emergence from a dataset of

# Structure / Pathway Discovery

Have a set of organisms. Have rows of genes. Columns of conditions / specifications / etc per gene. Background: perhaps there are pathways of gene expression we could discover that relate to a species / organism liking burgers and another pathway of genes that relate to liking mushrooms. There can be overlap in genes in pathways for liking burgers and liking mushrooms. So each set of genes / pathway can exist independently.

## 1. Summary / Overview

### 1.1 Group To Do

- [X] Create Github repo and Slack channel
- [X] Simulate organism properties = create simulated data (1000 rows x 500k cols)
- [X] Create f(...) map projection of properties to expression levels
- [X] Apply f() to properties for each species (i.e. each row in the simulated data)
- [ ] Finalize simulated data w.r.t. biological expectations 
- [ ] Run models: PCA, kernel-PCA / Spectral Clustering, multi-class clustering boosted + bagged, Autoencoder through AWC
- [ ] Test models w/ Pseudomonas data from Greene Lab 
- [ ] Add denoisey to Autoencorder
- [ ] Tune kernel PCA
- [ ] Try clustering (e.g. Spectral)
- [ ] Run t-SNE for data viz and comparison
- [ ] Model validity, feature importance
- [ ] Test model(s) on real data = does it work?
- [ ] Use of ensemble (this means averaging a set of models - may not make sense / be necessary)?

<hr>

We will create a set of response classes in simulated to train on:

**Note: there are no classes in the test (real) data** - but creating the model will involve us training and testing validity on only simulated data. I.e. can we recover the simulated response classes from the simulated data?

These response classes;
E.g. *"likes burgers", "likes mushrooms"*

## 2. Data

We are going to use simulated data that represents rows of genes with columns of condition expression. Simulated data comes from a number of parameters of gene-properties-transcript relationships / complex.

Setting 10 target labels (~10 multi-class predictor probabilities).

<hr>

Steps for creating simulation data:

* create 2 _Beta_ probability distributions
* multiplication of the 2 _Betas_
* take *Binomial* distribution (*Bernoulli*) of point above

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
