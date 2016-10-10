import numpy as np
import sys
sys.path.append("./origlink")

import chainer
import chainer.functions as F
import chainer.links as L
import l_maxout as lm
import lstm as ll

class RNNDecoder(chainer.Chain):
  def __init__(self, target_vocab, n_units, train=True):
    super(RNNDecoder, self).__init__(
      embed = L.EmbedID(target_vocab, n_units),
      l1 = ll.LSTMDec(n_units, n_units),
      l2 = lm.Maxout(n_units, target_vocab, 100)
      )
    for param in self.params():
      param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
      self.train = train

  def reset_state(self, target_vocab):
    self.l1.reset_state()
    self.prev_output = np.zeros((20, target_vocab), dtype=np.float32)

  def __call__(self, prev_y_id, cfe, num, dec_h0):
    y_cswr = self.embed(prev_y_id)
    h1= self.l1(F.dropout(y_cswr, train=self.train), cfe, num, dec_h0)
    y = self.l2(h1, self.prev_output, cfe)
    self.prev_output = y
    return y
