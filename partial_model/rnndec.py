# -*- coding: utf-8 -*-
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
from chainer import cuda

class RNNDecoder(chainer.Chain):
  def __init__(self, target_vocab, n_units, batchsize, train=True):
    super(RNNDecoder, self).__init__(
      embed = L.EmbedID(target_vocab, n_units),
      l1 = ll.LSTMDec(n_units, n_units),
      l2 = lm.Maxout(n_units, target_vocab, 500),
      )
    for param in self.params():
      param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
      self.train = train
    self.n_units = n_units
    self.batchsize = batchsize

  def reset_state(self):
    self.l1.reset_state()
    y_cpu = np.zeros((self.batchsize, self.n_units), dtype=np.float32)
    self.prev_y_cswr =  cuda.to_gpu(y_cpu, device=0)

  def set_l1(self, middle):
    self.l1.set_state(middle.dec_h0, middle.mid_c)

  def set_next_params(self, prev_y_id):
    self.prev_y_cswr = self.embed(prev_y_id)

  #management hidden state h and c in l1 not in this object
  #TODO I don't need concatinate? reason why for GPU because different length is not accceptable in GPU unit
  def __call__(self, prev_y_ids, middle):
    batch = prev_y_ids.shape[0]
    h = self.l1(F.dropout(self.prev_y_cswr[:batch], train=self.train))
    y = self.l2(h[:batch], self.prev_y_cswr[:batch], middle.mid_c[:batch])
    self.set_next_params(prev_y_ids)
    return y
