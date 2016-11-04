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

class RNNDecoder(chainer.Chain):
  @profile
  def __init__(self, target_vocab, emb_units, n_units, batchsize, train=True):
    super(RNNDecoder, self).__init__(
      embed = L.EmbedID(target_vocab, emb_units),
      l1 = ll.LSTMDec(emb_units, n_units),
      l2 = lm.Maxout(n_units, emb_units, target_vocab, 500),
      )
    for param in self.params():
      param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
      self.train = train
    self.emb_units = emb_units
    self.batchsize = batchsize

  @profile
  def reset_state(self):
    self.l1.reset_state()
    self.prev_y_cswr = np.zeros((self.batchsize, self.emb_units), dtype=np.float32)

  @profile
  def set_l1(self, middle):
    self.l1.set_state(middle.dec_h0, middle.mid_c)

  @profile
  def set_next_params(self, prev_y_id):
    self.prev_y_cswr = self.embed(prev_y_id)

  #management hidden state h and c in l1 not in this object
  #TODO I don't need concatinate? reason why for GPU because different length is not accceptable in GPU unit
  @profile
  def __call__(self, prev_y_ids, middle):
    batch = prev_y_ids.shape[0]
    h = self.l1(F.dropout(self.prev_y_cswr[:batch], train=self.train))
    y = self.l2(F.dropout(h[:batch]), F.dropout(self.prev_y_cswr[:batch]), F.dropout(middle.mid_c[:batch]))
    self.set_next_params(prev_y_ids)
    return y
