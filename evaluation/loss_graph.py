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

arr = []
size = len(v_loss) - 1
for i, l in enumerate(v_loss):
    arr.append(l)
    if i == size:
        break
    linear_element = (v_loss[i] - v_loss[i + 1]) / 24
    diff = l
    for i in range(24):
        diff = diff - linear_element
        arr.append(diff)

t1 = np.arange(0, len(loss), 1)
t_step = np.arange(0, len(arr), 1)

plt.plot(t1, loss, "k")
plt.plot(t_step, arr, "r")
plt.show()
