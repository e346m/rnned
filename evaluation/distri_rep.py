#!/usr/bin/env python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import six.moves.cPickle as pickle
from ipdb import set_trace

with open("./source_analysis_w", "r") as f:
  s_rep = pickle.load(f)
#with open("./target_analysis_w", "r") as f:
#  t_rep = pickle.load(f)
#
sX = [s_rep[word].data[0] for word in s_rep]
#tX = [t_rep[word].data[0] for word in t_rep]
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
model = TSNE(n_components=2, perplexity=50, n_iter=500, verbose=3, random_state=1)
sX = model.fit_transform(sX)
#tX = model.fit_transform(tX)

plt.figure(2, figsize=(8, 6))
plt.clf()
plt.scatter(sX[:, 0], sX[:, 1], c="red", cmap=plt.cm.Paired)
#plt.scatter(tX[:, 0], tX[:, 1], c="blue", cmap=plt.cm.Paired)
plt.xlabel('tsne1')
plt.ylabel('tsne2')

plt.show()
