from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import datasets
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import six.moves.cPickle as pickle
from ipdb import set_trace
mpl.rcParams["font.family"] = 'Source Han Code JP'

parser = argparse.ArgumentParser()
parser.add_argument('--word_representaion', '-wr',
                    help="Datasets of word representaions")
parser.add_argument('--tsne', '-tsne', action='store_true',
                    help="choose tsne visualization")
parser.add_argument('--pca', '-pca', action='store_true',
                    help="choose pca visualization")
args = parser.parse_args()

with open(args.word_representaion, "rb") as f:
    rep = pickle.load(f)
sX = [embeddings.data for embeddings in rep.values()]
keys = rep.keys()

# TEST
#with open(args.word_vocab, "rb") as f:
#    vocab = pickle.load(f)
#with open(args.word_representaion, "rb") as f:
#    rep = pickle.load(f)
#sX = rep
##vocab = [vocab[i] for i in vocab]
#vocab = vacab.values


class t_sne:
    def __init__(self):
        self.model = TSNE(n_components=2,
                          perplexity=50, n_iter=500, verbose=3, random_state=1)

    def transform(self, sX):
        return self.model.fit_transform(sX)


class pca:
    def __init__(self):
        self.model = PCA(n_components=2)

    def transform(self, sX):
        return self.model.fit_transform(sX)

if args.pca:
    reduction = pca()
else:
    reduction = t_sne()

sX = reduction.transform(sX)
plt.figure(2, figsize=(12, 8))
plt.clf()
plt.scatter(sX[:, 0], sX[:, 1], c="red", cmap=plt.cm.Paired)

for v, s in zip(keys, sX):
    plt.annotate(v, s, fontsize=10)

plt.show()
