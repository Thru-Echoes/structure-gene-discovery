# Code from here: https://github.com/FlorianMuellerklein/OttoGroup_Kaggle

import pandas as pd
import scipy as sp
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn import feature_extraction
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from nolearn.dbn import DBN

# multiclass loss
def MultiLogLoss(y_true, y_pred, eps = 1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    idea from this post:
    http://www.kaggle.com/c/emc-data-science/forums/t/2149/is-anyone-noticing-difference-betwen-validation-and-leaderboard-error/12209#post12209

    Parameters
    ----------
    y_true : array, shape = [n_samples]
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    rows = actual.shape[0]
    actual[np.arange(rows), y_true.astype(int)] = 1
    vsota = np.sum(actual * np.log(predictions))
    return -1.0 / rows * vsota

# import data
train = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")
sample = pd.read_csv("Data/sampleSubmission.csv")

# drop ids and get labels
labels = train.target.values
train = train.drop("id", axis = 1)
train = train.drop("target", axis = 1)
test = test.drop("id", axis = 1)

# transform counts to TFIDF features
tfidf = feature_extraction.text.TfidfTransformer()
train = tfidf.fit_transform(train).toarray()
test = tfidf.transform(test).toarray()

# generate polynomial features
poly = PolynomialFeatures()
train = poly.fit_transform(train)
test = poly.transform(test)
#train = np.hstack((train, poly_train))
#test = np.hstack((test, poly_test))

# encode labels
lbl_enc = LabelEncoder()
labels = lbl_enc.fit_transform(labels)

# set up datasets for cross eval
x_train, x_test, y_train, y_test = train_test_split(train, labels)

# train a DBN classifier
clf = DBN([train.shape[1], 8000, 9], learn_rates = 0.3,
            learn_rate_decays = 0.9, epochs = 50, verbose = 1) # l2_costs = 0.0001,

clf.fit(x_train, y_train)

# predict on test set
preds = clf.predict_proba(x_test)

# ----------------------  cross eval  -----------------------------------------

#y_test = label_binary.inverse_transform(y_test)
#y_test = LabelEncoder().fit_transform(y_test)

print("Multiclass Log Loss: ", MultiLogLoss(y_test, preds))
