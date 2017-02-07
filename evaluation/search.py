import numpy as np
import six
import six.moves.cPickle as pickle
import argparse
from chainer import cuda
from pathlib import Path
from ipdb import set_trace

n_result = 10  # number of search result to show

parser = argparse.ArgumentParser()
parser.add_argument('--word_representaion', '-wr', help='Word Representaion')
args = parser.parse_args()

parent = Path(args.word_representaion).parent.parent.parent
source_path = parent / "source.vocab"
with source_path.open("rb") as f:
    source_vocab = pickle.load(f, encoding='bytes')
index2word = {v:k for k, v in source_vocab.items()}

with open(args.word_representaion, "rb") as f:
    word2vec = pickle.load(f)
vec = [embeddings.data for embeddings in word2vec.values()]
word = word2vec.keys()

#with open('word2vec.model', 'r') as f:
#    ss = f.readline().split()
#    n_vocab, n_units = int(ss[0]), int(ss[1])
#    word2index = {}
#    index2word = {}
#    w = numpy.empty((n_vocab, n_units), dtype=numpy.float32)
#    for i, line in enumerate(f):
#        ss = line.split()
#        assert len(ss) == n_units + 1
#        word = ss[0]
#        word2index[word] = i
#        index2word[i] = word
#        w[i] = numpy.array([float(s) for s in ss[1:]], dtype=numpy.float32)
#
#
#s = numpy.sqrt((w * w).sum(1))
#w /= s.reshape((s.shape[0], 1))  # normalize

try:
    while True:
        q = six.moves.input('>> ')
        if q not in word2vec:
            print('"{0}" is not found'.format(q))
            continue
        v = word2vec[q].data
        similarity = np.asarray(vec).dot(v)
        print('query: {}'.format(q))
        count = 0
        for i in (-similarity).argsort():
            if np.isnan(similarity[i]):
                continue
            if index2word[i] == q:
                continue
            print('{0}: {1}'.format(index2word[i], similarity[i]))
            count += 1
            if count == n_result:
                break

except EOFError:
    pass
