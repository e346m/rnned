import numpy as np
import six.moves.cPickle as pickle
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--dir', '-d', help='Target file path')
args = parser.parse_args()

with open(args.dir, "r") as f:
  loss = pickle.load(f)

plt.plot(loss, loss, "bs")
plt.show()
