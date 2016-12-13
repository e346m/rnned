import numpy as np
import six.moves.cPickle as pickle
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
from chainer import cuda

parser = argparse.ArgumentParser()
parser.add_argument('--dir', '-d', help='Target file path')
args = parser.parse_args()

with open(args.dir, "r") as f:
  loss = cuda.to_cpu(pickle.load(f))

t1 = np.arange(0, len(loss), 1)

plt.plot(t1, loss, "k")
plt.show()
