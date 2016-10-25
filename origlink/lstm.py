import numpy
import six

import chainer
from chainer.functions.activation import lstm
from chainer.functions.array import concat
from chainer.functions.array import split_axis
from chainer import initializers
from chainer import link
from chainer.links.connection import linear
from chainer import variable

import func_lstm
import ipdb

class LSTMDec(chainer.Chain):

  def __init__(self, in_size, out_size,
       lateral_init = None, upward_init = None, diagonal_init = None,
       bias_init = 0, forget_bias_init = 0):
    super(LSTMDec, self).__init__(
      upward = linear.Linear(in_size, 4 * out_size, initialW = 0),
      diagonal = linear.Linear(in_size, 4 * out_size, initialW = 0),
      lateral = linear.Linear(out_size, 4 * out_size, initialW = 0, nobias = True),
    )
    self.state_size = out_size

    for i in six.moves.range(0, 4 * out_size, out_size):
      initializers.init_weight(
        self.lateral.W.data[i:i + out_size, :], lateral_init)
      initializers.init_weight(
        self.upward.W.data[i:i + out_size, :], upward_init)
      initializers.init_weight(
        self.diagonal.W.data[i:i + out_size, :], diagonal_init)

    a, i, f, o = func_lstm._extract_gates(
      self.upward.b.data.reshape(1, 4 * out_size, 1))
    initializers.init_weight(a, bias_init)
    initializers.init_weight(i, bias_init)
    initializers.init_weight(f, forget_bias_init)
    initializers.init_weight(o, bias_init)

    self.reset_state()

  def to_cpu(self):
    super(LSTM, self).to_cpu()
    if self.c is not None:
      self.c.to_cpu()
    if self.h is not None:
      self.h.to_cpu()

  def to_gpu(self, device=None):
    super(LSTM, self).to_gpu(device)
    if self.c is not None:
      self.c.to_gpu(device)
    if self.h is not None:
      self.h.to_gpu(device)

  def set_state(self, c, h):
    assert isinstance(c, chainer.Variable)
    assert isinstance(h, chainer.Variable)
    assert isinstance(cfe, chainer.Variable)
    c_ = c
    h_ = h
    cfe_ = cfe
    if self.xp == numpy:
      c_.to_cpu()
      h_.to_cpu()
      cfe_.to_cpu()
    else:
      c_.to_gpu()
      h_.to_gpu()
      cfe_.to_gpu()
    self.c = c_
    self.h = h_
    self.cfe = cfe_

  def reset_state(self):
    self.c = self.h = None


  def __call__(self, input_y, cfe, num, dec_h0):

      def to_fix(batch, val):
        return split_axis.split_axis(val, [batch], axis=0)

      batch = input_y.shape[0]

      if cfe.shape[0] > batch:
        cfe_update, cfe_rest = to_fix(batch, cfe)
        lstm_in = self.diagonal(cfe_update)
      else:
        lstm_in = self.diagonal(cfe)

      if num != 0:
        if self.prev_y.shape[0] > batch:
          py_update, py_rest = to_fix(batch, self.prev_y)
          lstm_in += self.diagonal(py_update)
        else:
          lstm_in += self.upward(self.prev_y)
          self.h = dec_h0

      self.prev_y = input_y
      h_rest = None
      if self.h is not None:
          h_size = self.h.shape[0]
          if batch == 0:
              h_rest = self.h
          elif h_size < batch:
              msg = ('The batch size of prev_y be equal to or less than the '
                   'size of the previous state h.')
              raise TypeError(msg)
          elif h_size > batch:
              h_update, h_rest = to_fix(batch, self.h)
              lstm_in += self.lateral(h_update)
          else:
              lstm_in += self.lateral(self.h)
      if self.c is None:
          xp = self.xp
          self.c = variable.Variable(
              xp.zeros((batch, self.state_size), dtype=input_y.dtype),
              volatile='auto')
      self.c, y = func_lstm.lstm(self.c, lstm_in)

      if h_rest is None:
          self.h = y
      elif len(y.data) == 0:
          self.h = h_rest
      else:
          self.h = concat.concat([y, h_rest], axis=0)

      return y
