from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import six.moves.cPickle as pickle
from ipdb import set_trace

with open("./word_analysis_w", "r") as f:
  word_rep = pickle.load(f)

X = [word_rep[word].data[0] for word in word_rep]
set_trace()
pca = PCA(n_components = 3)
pca.fit(X)
X_pca= pca.transform(X)
set_trace()
colors = ['b.']
fig, ax = plt.subplots()
ax.plot(X_pca[:,0],  X_pca[:,1],  'b.', label='Setosa')
#ax.annotate(["test", "1"], xy=(X_pca[:,0],  X_pca[:,1]), size=10)

ax.set_title("PCA for distributed representaion")
ax.legend(numpoints=1)

plt.show()
