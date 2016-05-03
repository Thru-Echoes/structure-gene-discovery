# Concepts of model runs / data exploration


### Model Ideas / Concepts:

* use K-means for preliminary examination of the data
* use PCA for first step dimensional reduction
* use ICA (indpendent)
* use autoencode
* use XGBoost with 10 non-softprob / max for classification
* use XGBoost for feature importance

### Data Exploration Ideas / Concepts:

The following should be <strong>COMPARED</strong> to auto-encoder or PCA or some other dimensional reduction method by itself.

<strong>Instead of just dimensional reduction - can use 2-layer clustering method:<strong>

* use PCA and / or K-means and / or t-SNE to find number of clusters
* use the optimal number of clusters into *XGBoost* without normalized output probabilities so it is not fixed
* get feature importance and predictive power of XGBoost model

*t-SNE* can be used to do visualization and dimensional reduction with a specific cost function. May be a useful method for comparison, in addition to other methods, or as a prep-step before running some other method. 

<strong>Example code for t-SNE</strong>

```
    from sklearn.manifold import TSNE

    from matplotlib import pyplot

    tsne = TSNE(verbose = 1)
    X_2d = tsne.fit_transform(output_activations)
    pyplot.scatter(X_2d[:, 0], X_2d[:, 1])
    pyplot.showfig()
    # output_activations = SAMPLE_DATA
```
