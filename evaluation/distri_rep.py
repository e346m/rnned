#!/usr/bin/env python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import six.moves.cPickle as pickle
from ipdb import set_trace

#with open("./word_analysis_w", "r") as f:
#  word_rep = pickle.load(f)
#
#X = [word_rep[word].data[0] for word in word_rep]
#set_trace()
#pca = PCA(n_components = 3)
#pca.fit(X)
#X_pca= pca.transform(X)
#set_trace()
#colors = ['b.']
#fig, ax = plt.subplots()
#ax.plot(X_pca[:,0],  X_pca[:,1],  'b.', label='Setosa')
#
#ax.set_title("PCA for distributed representaion")
#ax.legend(numpoints=1)


iris = datasets.load_iris()
X = iris.data
Y = iris.target

set_trace()

model = TSNE(n_components=2, perplexity=50, n_iter=500, verbose=3, random_state=1)
X = model.fit_transform(X)

plt.figure(2, figsize=(8, 6))
plt.clf()
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.xlabel('tsne1')
plt.ylabel('tsne2')

plt.show()
