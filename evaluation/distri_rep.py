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
sX = [s_rep[word].data[0] for word in s_rep]
model = TSNE(n_components=2,
             perplexity=50, n_iter=500, verbose=3, random_state=1)
sX = model.fit_transform(sX)

plt.figure(2, figsize=(8, 6))
plt.clf()
plt.scatter(sX[:, 0], sX[:, 1], c="red", cmap=plt.cm.Paired)
plt.xlabel('tsne1')
plt.ylabel('tsne2')

plt.show()
