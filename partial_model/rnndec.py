# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.path.append("./mylink")


import chainer
import chainer.functions as F
import chainer.links as L
import l_maxout as lm
import lstm as ll
from chainer import cuda
from chainer.functions.activation import softmax
from ipdb import set_trace

class RNNDecoder(chainer.Chain):
  def __init__(self, target_vocab, emb_units, n_units, batchsize, gpu, train=True):
    super(RNNDecoder, self).__init__(
      embed = L.EmbedID(target_vocab, emb_units, ignore_label=-1),
      l1 = ll.LSTMDec(emb_units, n_units),
      l2 = lm.Maxout(n_units, emb_units, target_vocab, 2),
      )
    for param in self.params():
      param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
      self.train = train
    self.emb_units = emb_units
    self.batchsize = batchsize
    self.gpu = gpu

  def reset_state(self):
    self.l1.reset_state()

  def set_l1(self, middle):
    self.l1.set_state(middle.dec_h0, middle.mid_c)

  #management hidden state h and c in l1 not in this object
  def __call__(self, prev_y_ids, middle, batch):
    prev_y_cswr = self.embed(prev_y_ids)
    h = self.l1(F.dropout(prev_y_cswr, train=self.train))
    y = self.l2(F.dropout(h[:batch], train=self.train),
        F.dropout(prev_y_cswr[:batch], train=self.train),
        F.dropout(middle.mid_c[:batch], train=self.train))
    return y
