import numpy as np
import psutil

import chainer
import chainer.links as L
import chainer.functions as F
class RNNEncoder(chainer.Chain):
  @profile
  def __init__(self, source_vocab, emb_units, n_units, train=True):
    super(RNNEncoder, self).__init__(
      embed = L.EmbedID(source_vocab, emb_units),
      l1 = L.LSTM(emb_units, n_units),
      )
    for param in self.params():
      param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
      self.train = train

  @profile
  def reset_state(self):
    self.l1.reset_state()

  def print_memory(self, phase):
    print(phase )
    print(psutil.virtual_memory())
    print(psutil.swap_memory())

  @profile
  def __call__(self, x):
    self.print_memory("before_cswr")
    x_cswr = self.embed(x)
    self.print_memory("after_cswr")
    self.l1(F.dropout(x_cswr, train=self.train))
    self.print_memory("after_lstm")
    return None
