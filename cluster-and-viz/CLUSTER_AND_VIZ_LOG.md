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

<hr>

## 5. Unsupervised Random Forest

This is an exciting way to do **unsupervised random forests** on the entire data set - 950 samples (rows) by 5549 genes (columns). This method uses measures of feature (i.e. genes) dissimilarities to construct a simulated matrix of features with some dynamic number of labels (like classes / clustered groups). This simulated data is used to run a *semi-supervised Random Forest* on the original data to find the number of labels and classifying each sample to one of those labels.

**NOTE:** it may (or probably?) makes more sense to run this as soft-probability outputs so that we can analyze / examine the relative probabilities of each sample to be a part of each label / cluster that the method found.

### 5.1 Relating this to single-layer NN Autoencoder

This is a great way to compare the weights / codings of the Autoencoder for mapping the data unto the 50 hidden nodes (which are hidden pathways biologically). Also - this method gives us a set of feature importance measurements that are global, local, and more - see below.

### 5.2 Feature Importance - 950 samples x 5549 genes

This is an alternative method to using single-layer NN Autoencoder. As another approach we can analyze the Autoencoder codings back unto the 950 samples (**see below**).

#### 5.2.1 Global Variable Importance:

Important in terms of information gain when trying to use these features to predict labels:

```
        variables score class class.frequency percent percent.importance
    1      V1185    12     1            0.50  100.00                  1
    2      V5338     7     3            1.00   56.64                  1
    3      V1108     6     1            1.00   53.52                  1
    4      V3725     5     2            1.00   44.66                  1
    5      V3810     5     3            1.00   38.57                  0
    6      V3689     4     2            0.67   37.42                  0
    7      V3823     4     2            0.50   32.81                  0
    8      V4222     4     1            0.33   30.94                  0
    9      V2819     3     3            0.67   27.14                  0
    10     V1769     3     1            0.86   24.52                  0
    11     V4226     3     2            0.50   23.52                  0
    12     V1739     3     1            1.00   23.41                  0
    13     V3418     3     1            0.67   23.07                  0
    14     V3253     3     1            0.50   22.59                  0
    15     V3166     3     1            0.50   21.92                  0
    16     V4588     3     1            0.33   21.43                  0
    17     V4254     3     1            0.33   21.28                  0
    18      V555     2     1            1.00   19.62                  0
    19     V4083     2     1            1.00   19.12                  0
    20     V4225     2     2            0.50   18.66                  0
    21     V1915     2     1            0.75   18.52                  0
    22     V2981     2     2            0.60   17.99                  0
    23     V3579     2     2            0.50   17.79                  0
    24      V484     2     3            0.60   16.88                  0
    25     V1410     2     1            0.50   16.73                  0
    26     V4903     2     3            0.75   16.54                  0
    27     V4576     2     1            0.60   16.50                  0
    28     V5363     2     1            1.00   16.14                  0
    29     V1342     2     2            0.67   15.68                  0
    30     V1242     2     1            0.50   15.64                  0
```

Ordering by score = best information gain predictors (i.e. best genes for predicting pathways)

Ordering by class + class.freq = most discriminant predictors (i.e. best genes for discriminating)

#### 5.2.2 Local Variable Importance:

The 10 most important variables (i.e. genes) and their relative amount of interactions with other variables (i.e. genes):

```
    V2603  V1930  V2055  V3631  V1770   V555  V1316  V3725  V2058  V1590  V2900   V490  V1265   V322
    V2058        0.2855 0.2401 0.1648 0.1286 0.1081 0.1052 0.1045 0.1031 0.1024 0.1017 0.1017 0.1017 0.1010 0.1003
    V1316        0.2744 0.2290 0.1538 0.1176 0.0970 0.0942 0.0935 0.0920 0.0913 0.0906 0.0906 0.0906 0.0899 0.0892
    V3631        0.2481 0.2027 0.1275 0.0913 0.0707 0.0679 0.0671 0.0657 0.0650 0.0643 0.0643 0.0643 0.0636 0.0629
    V555         0.2176 0.1722 0.0969 0.0608 0.0402 0.0373 0.0366 0.0352 0.0345 0.0338 0.0338 0.0338 0.0331 0.0324
    V72          0.2134 0.1680 0.0927 0.0565 0.0360 0.0331 0.0324 0.0310 0.0303 0.0296 0.0296 0.0296 0.0289 0.0282
    V3725        0.2102 0.1648 0.0896 0.0534 0.0328 0.0300 0.0293 0.0278 0.0271 0.0264 0.0264 0.0264 0.0257 0.0250
    V1590        0.2097 0.1643 0.0891 0.0529 0.0323 0.0294 0.0287 0.0273 0.0266 0.0259 0.0259 0.0259 0.0252 0.0245
    V1265        0.2029 0.1574 0.0822 0.0460 0.0254 0.0226 0.0219 0.0205 0.0198 0.0190 0.0190 0.0190 0.0183 0.0176
    V190         0.2023 0.1569 0.0817 0.0455 0.0249 0.0221 0.0214 0.0199 0.0192 0.0185 0.0185 0.0185 0.0178 0.0171
    V2055        0.1997 0.1543 0.0791 0.0429 0.0223 0.0194 0.0187 0.0173 0.0166 0.0159 0.0159 0.0159 0.0152 0.0145
    avg1rstOrder 0.1976 0.1521 0.0769 0.0407 0.0201 0.0173 0.0166 0.0152 0.0145 0.0137 0.0137 0.0137 0.0130 0.0123
```

#### 5.2.3 Variable Importance on gene-to-gene interactions:

```
Gene:           V2603  V2058  V1930  V1316  V3631  V2055   V555  V3725    V72  V1590
Importance:    0.1172 0.0968 0.0920 0.0839 0.0673 0.0542 0.0324 0.0209 0.0201 0.0191
```

#### 5.2.4 Variable Importance for labels:

<strong>NOTE:</strong> this method found that there are *three core labels / classes*.

```
            Class 1 Class 2 Class 3
    V2603    0.71    0.00    0.00
    V1930    0.00    0.71    0.00
    V2055    0.00    0.00    0.36
    V3631    0.00    0.00    0.18
    V1770    0.00    0.00    0.08
    V555     0.04    0.00    0.00
    V1316    0.04    0.00    0.00
    V2058    0.00    0.04    0.00
    V3725    0.00    0.04    0.00
    V490     0.00    0.00    0.04
```

<hr>

### 5.3 Feature Importance - 53 species (rows) x 81k RNAs (cols)

This is the real data collected from a ton of data cleaning and creation time (big shout out to Lisa and Harriet!)

<strong>NOTE:</strong> this was a quick trial run of unsupervised RF on 53 species and **only** 20k columns (the first 20k columns).

#### 5.3.1 Global Variable Importance:

Important in terms of information gain when trying to use these features to predict labels:

```
        variables score class class.frequency percent percent.importance
    1  OG_336808     4     3            0.62  100.00                  1
    2  OG_154639     3     3            0.71   88.32                  1
    3  OG_336361     3     1            0.50   70.44                  1
    4  OG_051751     3     1            0.50   68.88                  1
    5  OG_295085     2     3            0.67   66.03                  1
    6  OG_217123     2     3            0.50   58.80                  1
    7  OG_103079     2  <NA>              NA   54.94                  1
    8  OG_138279     2     3            0.40   54.55                  1
    9  OG_226526     2     1            0.50   54.54                  1
    10 OG_330880     2     3            0.50   48.02                  1
    11 OG_036823     2     2            0.50   46.24                  1
    12 OG_102182     2     3            0.75   43.25                  0
    13 OG_291398     2     2            0.50   42.25                  0
    14 OG_113519     2     1            0.50   41.03                  0
    15 OG_301231     1     3            1.00   40.75                  0
    16 OG_000275     1     4            0.67   39.67                  0
    17 OG_093574     1     4            0.67   38.75                  0
    18 OG_110301     1     2            0.67   37.87                  0
    19 OG_200230     1     2            0.50   37.73                  0
    20 OG_105836     1     3            0.57   36.44                  0
    21 OG_228850     1     1            0.33   36.07                  0
    22 OG_062277     1     1            0.50   36.04                  0
    23 OG_340681     1     3            0.50   35.07                  0
    24 OG_223015     1     2            1.00   34.62                  0
    25 OG_301356     1  <NA>              NA   34.19                  0
    26 OG_081208     1     2            0.50   34.02                  0
    27 OG_104248     1  <NA>              NA   32.77                  0
    28 OG_338206     1     3            0.50   32.54                  0
    29 OG_226712     1     1            0.50   30.76                  0
    30 OG_262173     1     1            0.33   30.53                  0
```

#### 5.3.2 Local Variable Importance:

The 10 most important variables (i.e. genes) and their relative amount of interactions with other variables (i.e. genes):

```
    OG_105836 OG_301231 OG_336808 OG_081208 OG_180181 OG_217123 OG_142060 OG_038261 OG_051751 OG_102182
    OG_105836       0.2570    0.2127    0.1685    0.1574    0.1574    0.1574    0.1353    0.1243    0.1243    0.1243
    OG_340647       0.2098    0.1656    0.1213    0.1103    0.1103    0.1103    0.0882    0.0771    0.0771    0.0771
    OG_089133       0.1721    0.1278    0.0836    0.0725    0.0725    0.0725    0.0504    0.0394    0.0394    0.0394
    OG_180181       0.1721    0.1278    0.0836    0.0725    0.0725    0.0725    0.0504    0.0394    0.0394    0.0394
    OG_336808       0.1721    0.1278    0.0836    0.0725    0.0725    0.0725    0.0504    0.0394    0.0394    0.0394
    OG_340681       0.1721    0.1278    0.0836    0.0725    0.0725    0.0725    0.0504    0.0394    0.0394    0.0394
    OG_154639       0.1626    0.1184    0.0742    0.0631    0.0631    0.0631    0.0410    0.0299    0.0299    0.0299
    OG_285581       0.1626    0.1184    0.0742    0.0631    0.0631    0.0631    0.0410    0.0299    0.0299    0.0299
    OG_301231       0.1626    0.1184    0.0742    0.0631    0.0631    0.0631    0.0410    0.0299    0.0299    0.0299
    OG_336361       0.1626    0.1184    0.0742    0.0631    0.0631    0.0631    0.0410    0.0299    0.0299    0.0299
    avg1rstOrder    0.1681    0.1239    0.0796    0.0686    0.0686    0.0686    0.0464    0.0354    0.0354    0.0354

```

#### 5.3.3 Variable Importance on gene-to-gene interactions:

```
    OG_105836 OG_301231 OG_340647 OG_336808 OG_180181 OG_340681 OG_089133 OG_142060 OG_154639 OG_285581
   0.2251    0.1005    0.0867    0.0803    0.0743    0.0511    0.0434    0.0404    0.0399    0.0369
```

#### 5.3.4 Variable Importance for labels:

RF found 4 labels to classify genes into (i.e. 4 main pathways).

```
            Class 1 Class 2 Class 3 Class 4
OG_081208    0.00    0.50    0.00    0.11
OG_301231    0.00    0.00    0.36    0.05
OG_105836    0.12    0.25    0.27    0.26
OG_217123    0.25    0.00    0.05    0.05
OG_138279    0.00    0.25    0.00    0.00
OG_180181    0.00    0.00    0.00    0.21
OG_336808    0.12    0.00    0.18    0.00
OG_336361    0.12    0.00    0.00    0.00
OG_051751    0.12    0.00    0.00    0.00
OG_340686    0.12    0.00    0.00    0.00
```
