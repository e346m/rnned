import numpy as np
import sys
sys.path.append("./origlink")

import chainer
import chainer.functions as F
import chainer.links as L
import l_maxout as lm
import lstm as ll
from chainer.functions.activation import softmax
from ipdb import set_trace

class RNNDecoder(chainer.Chain):
  @profile
  def __init__(self, target_vocab, n_units, batchsize, train=True):
    super(RNNDecoder, self).__init__(
      embed = L.EmbedID(target_vocab, n_units),
      l1 = ll.LSTMDec(n_units, n_units),
      l2 = lm.Maxout(n_units, target_vocab, 100),
      )
    for param in self.params():
      param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
      self.train = train
    self.prev_output = np.zeros((batchsize, target_vocab), dtype=np.float32)

  @profile
  def reset_state(self):
    self.l1.reset_state()

  @profile
  def __call__(self, prev_y_id, middle, num):
    y_cswr = self.embed(prev_y_id)
    h1= self.l1(F.dropout(y_cswr, train=self.train), middle.mid_c, num, middle.dec_h0)
    y = self.l2(h1, self.prev_output, middle.mid_c)
    self.prev_output = y
    return y
