import numpy as np
import six.moves.cPickle as pickle
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
from chainer import cuda
from ipdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument('--dir', '-d', help='Target file path')
args = parser.parse_args()

with open(args.dir, "rb") as f:
    data = pickle.load(f)
    loss = [cuda.to_cpu(_f) for _f in data["train"]]
    v_loss = [cuda.to_cpu(_f) for _f in data["validation"]]

t1 = np.arange(0, len(loss), 1)
t_step = np.arange(0, len(v_loss), 1)

plt.plot(t1, loss, "k")
plt.plot(t_step, v_loss, "r")
plt.show()
