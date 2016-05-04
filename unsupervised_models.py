import scipy
from scipy import stats
from scipy import special
import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import palettable as pal
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import cross_decomposition
from sklearn import linear_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop, SGD
from keras import backend as K
#%matplotlib inline

properties = pd.DataFrame.from_csv("data/Species_properites_likelihood.csv")

concentration = pd.DataFrame.from_csv("data/simulated_counts.csv")
for i in range(concentration.shape[0]):
        concentration.iloc[i, :] = concentration.iloc[i, :] / concentration.iloc[i, :].sum()


### Do PCA:

pca = decomposition.PCA(n_components = 10)
pca.fit(concentration)
X = pca.transform(concentration)

dictlearn = decomposition.DictionaryLearning(n_components = 10)
dictlearn.fit(concentration)
X2 = dictlearn.transform(concentration)

### Do Linear Regression

lm = linear_model.LinearRegression()
lm.fit(X, preprocessing.scale(np.array(properties)))

lm2 = linear_model.LinearRegression()
lm2.fit(X2, preprocessing.scale(np.array(properties)))

see_lm_score = lm.score(X, preprocessing.scale(np.array(properties)))
see_lm2_score = lm2.score(X2, preprocessing.scale(np.array(properties)))

print
print("Lin Reg score: ", see_lm_score)
print
print("Lin Reg score 2: ", see_lm2_score)

### R Squared

R2 = 1 - ((lm.predict(X) - preprocessing.scale(np.array(properties))) ** 2).sum(axis = 0) / ((preprocessing.scale(np.array(properties))) ** 2).sum(axis = 0)

R2

### Do Autoencoder

ae = Sequential()
ae.add(Dense(10, input_dim = concentration.shape[1]))
ae.add(Activation('relu'))
ae.add(Dense(concentration.shape[1]))
ae.add(Activation('softmax'))
ae.compile(optimizer = SGD(lr = .01, momentum = .9, decay = 0.001, nesterov = True), loss = 'categorical_crossentropy')

x = preprocessing.scale(np.array(concentration))
y = np.array(concentration)
ae.fit(x, y, batch_size = 100, nb_epoch = 10, verbose = 1)

np.mean(x)

get_code = K.function([ae.layers[0].input], [ae.layers[1].output])
code = get_code([x])[0]

lm3 = linear_model.LinearRegression()
lm3.fit(code, preprocessing.scale(np.array(properties)))
see_lm3_score = lm3.score(code, preprocessing.scale(np.array(properties)))

print
print ("Lin Ref score with autoencoder: ", see_lm3_score)
print

##### Ideas about visualization / interpretation
#
# calculate correlation over the weights of a single gene
#
# after building network from above
#
# see difference between "field" samples and "lab" samples
