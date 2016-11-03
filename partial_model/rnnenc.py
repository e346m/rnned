import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
class RNNEncoder(chainer.Chain):
  @profile
  def __init__(self, source_vocab, n_units, train=True):
    super(RNNEncoder, self).__init__(
      embed = L.EmbedID(source_vocab, n_units),
      l1 = L.LSTM(n_units, n_units),
      )
    for param in self.params():
      param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
      self.train = train

  @profile
  def reset_state(self):
    self.l1.reset_state()

  @profile
  def __call__(self, x):
    x_cswr = self.embed(x)
    self.l1(F.dropout(x_cswr, train=self.train))
    return None
