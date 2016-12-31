import numpy as np

import sys
sys.path.append("./mylink")
import lstm as ll

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda
from ipdb import set_trace

class RNNEncoder(chainer.Chain):
  def __init__(self, source_vocab, emb_units, n_units, gpu, train=True):
    super(RNNEncoder, self).__init__(
      embed = L.EmbedID(source_vocab, emb_units, ignore_label=-1),
      l1 = ll.LSTMEnc(emb_units, n_units),
      )
    for param in self.params():
      param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
      self.train = train
    self.n_units = n_units
    self.gpu = gpu

  def reset_state(self):
    self.l1.reset_state()

  def set_condition(self, x):
    enable = (x != -1)
    _enable = enable.reshape(enable.shape[0], 1)
    cond = np.repeat(_enable, self.n_units, axis=1)
    if self.gpu >= 0:
        cond = cuda.to_gpu(cond, device=self.gpu)
    return cond

  def __call__(self, x):
    x_cswr = self.embed(x)
    cond = self.set_condition(x)
    self.l1(F.dropout(x_cswr, train=self.train), cond)
    return None

  def eval_call(self, x):
    x_cswr = self.embed(x)
    cond = self.set_condition(x)
    self.l1(F.dropout(x_cswr, train=self.train), cond)
    return x_cswr
