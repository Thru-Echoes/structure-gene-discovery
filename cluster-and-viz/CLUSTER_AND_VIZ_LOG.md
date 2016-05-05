# Description of post-model analysis, clustering, and data visualization

Oliver: here I explain my use of t-SNE, clustering, and weighting methods on the output of *single-layer Neural Network Autoencoder* and other dimensional reduction techniques.

## 1. Generate gene and pathway counts

The file **calc-gene-pathway-counts.R** is an R script that can be used to calculate the number of genes that are within each pathway and the number of pathways that each gene appears in - output as CSV for each.

## 2. Post-autoencoder visualizations

The script for visualizing the clustering of the nodes (hidden layers) that the Autoencoder method found can be found in the **run_unsupervised.py** script that is within the *sample-models/* directory and can be called from main.

#### 2.1 Background: call from main

Note: calling from *main* looks like this (from the top directory):

```
    $ python __main__.py
```

<strong>EZPZ!</strong> - but of course you have to call the *main_execute()* function from within the main file to call *run_unsupervised.py*.


#### 2.2 What are these autoencoder clusters?

I went through and found the weights - from *ae_weights.csv* - for the 50 hidden layers (i.e. 50 nodes) that the Autoencoder method found. Each gene - all 5549 of them - have a weight per node as calculated from Autoencoder. I did some quick summary statistics and exploration of the relationship between the weights, nodes, and genes and saw that many genes had a max weight value (i.e. the maximum weight value for one of the 50 nodes) that was an order of magnitude or so more than the mean and median weights for that gene. Thus, I created a "label" per gene by assigning it the index of its maximum node - so each gene is assigned a value from 1-50 (or 0-49).

Then I run t-SNE to dimensionally reduce the 5549 rows of genes and 50 columns of nodes into a 2-dimensional space where a scatter plot of genes is visualized. This scatter plot shows potential clusterings of the genes and color-coded based on the nodes I assigned them to. This is the same method used for **PCA** and only **t-SNE** runs to see if the clusters they find match the clusters that the Autoencoder found.

**NOTE:** the method of *hard* setting the node assignment per gene could / should be changed. This was just a rough first approach but may be sufficient for visualization.

#### 2.3 Autoencoder weights and codings

I also generated t-SNE visualization plots in 2-dimensions that plotted the following:

**Autoencoder weights** - visualizing 50 nodes (rows) x 5549 genes (columns) to explore the clustering of nodes within a subspace of the gene space.

**Autoencoder codings** - visualize 50 nodes (rows) x 950 samples (columns) to explore the clustering of nodes within a subspace of the sample space.

## 3. What about clustering methods on the original data?

Since I assigned each gene a node label from 1-50 - we can now retroactively do some clustering or regression method(s) to see if we can use the 5549 genes (rows) x 950 samples (columns) to predict the node assignments (which were from their relative Autoencoder weights!).

I used *XGBoost* - random forest with boosting and bagging. Can do 5-Fold Cross-Validation and (potentially) random column sampling per iteration to generate an accuracy for predicting these clusters / labeles. Also - this can be used to find a feature importance mapping. This may be useful (or perhaps a different way of approach feature importance in this project) down the road.

## 4. Results and Discussions

For t-SNE visualizations by itself and with PCA we did not find great clustering. This was expected because PCA does not perform very well on clustering this data!

There are now visualizations of the following:

* <strong>PCA into 50 dimensions and then scatter plot of all genes in 2-dimensions with t-SNE</strong> to try to visualize the clustering of those 50 PCs as matching the 50 hidden nodes from Autoencoding (bad results as expected since PCA does not do well on this data)

* <strong>t-SNE directly on raw data to try to find clustering - scatter plot of all genes in 2-dimensions</strong> - this is also not good as expected (since PCA did not work well)

* <strong>2-dimensional plot of all nodes from hidden layer (Autoencoder) clustered with t-SNE</strong> - this is actually two plots: one for the Autoencoder weights and another for the Autoencoder codings. In these plots you can start to see some nodes (some hidden layers) that really stand out from the rest - **such as node 18** - which probably has a major influence on genetic expression

And we also have visualizations of the following *post-structure-discovery* methods:

* <strong>Feature importance of nodes (from hidden layers of Autoencoder) as predictors</strong> for classifying genes to their respective nodes of highest weight - through random forest multi-classification - this means I assigned each gene a label that corresponded to the node that had the highest weight value for it

* <strong>Feature importance of samples (from original dataset of 950 samples / predictors)</strong> for classifying genes to their respective nodes of highest weight - through random forest multi-classification and then boosted classification methods (*GBM*)

<strong>NOTE:</strong> feature importance is measured as the gain in reducing misclassifications per feature / predictor / column. So this means feature importance is a measure of (generally) how much better any specific feature is at making the entire model predict the nodes / classes of the genes. 
