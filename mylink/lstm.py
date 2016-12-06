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
import chainer.functions as F
import chainer.links as L
from ipdb import set_trace

import func_lstm
import ipdb

class LSTMDec(chainer.Chain):

  def __init__(self, in_size, out_size,
       lateral_init = None, upward_init = None, diagonal_init = None,
       bias_init = 0, forget_bias_init = 0):
    super(LSTMDec, self).__init__(
      upward = linear.Linear(in_size, 4 * out_size, initialW = 0),
      diagonal = linear.Linear(out_size, 4 * out_size, initialW = 0),
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
    super(LSTMDec, self).to_cpu()
    if self.c is not None:
      self.c.to_cpu()
    if self.h is not None:
      self.h.to_cpu()
    if self.cfe is not None:
      self.cfe.to_cpu()

  def to_gpu(self, device=None):
    super(LSTMDec, self).to_gpu(device)
    if self.c is not None:
      self.c.to_gpu(device)
    if self.h is not None:
      self.h.to_gpu(device)
    if self.cfe is not None:
      self.cfe.to_gpu(device)

  # if anther method call set_state restore c state
  #def set_state(self, c, h, cfe):
  #  assert isinstance(c, chainer.Variable)
  #  c_ = c
  #  if self.xp == numpy:
  #    c_.to_cpu()
  #  else:
  #    c_.to_gpu()
  #  self.c = c_

  #TODO when will you use? it's for generlize gpu cpu
  def set_initial_state(self, h0, cfe):
    assert isinstance(h, chainer.Variable)
    assert isinstance(cfe, chainer.Variable)
    h0_ = h0
    cfe_ = cfe
    if self.xp == numpy:
      h0_.to_cpu()
      cfe_.to_cpu()
    else:
      h0_.to_gpu()
      cfe_.to_gpu()
    self.h = h0_
    self.cfe = cfe_

  def reset_state(self):
    self.cfe = self.c = self.h = None

  def __call__(self, prev_y):
    def to_fix(batch, val):
      return split_axis.split_axis(val, [batch], axis=0)

    def batch_process(batch, state, link, lstm_in):
      rest = None
      if state is not None:
        size = state.shape[0]
        if batch == 0:
          rest = state
        elif size < batch:
          msg = ('The batch size of prev_y be equal to or less than the '
            'size of the previous state h.')
          raise TypeError(msg)
        elif size > batch:
          update, rest = to_fix(batch, state)
          lstm_in += link(update)
        else:
          lstm_in += link(state)
      return lstm_in, rest

    def restore_status(y, state, rest):
      if rest is None:
        state = y
      elif len(y.data) == 0:
        state = rest
      else:
        state = concat.concat([y, rest], axis=0)

    batch = prev_y.shape[0]
    lstm_in = self.upward(prev_y)
    lstm_in, cfe_rest = batch_process(batch, self.cfe, self.diagonal, lstm_in)
    lstm_in, h_rest = batch_process(batch, self.h, self.lateral, lstm_in)

    if self.c is None:
      xp = self.xp
      self.c = variable.Variable(
        xp.zeros((batch, self.state_size), dtype=prev_y.dtype),
        volatile='auto')

    self.c, y = func_lstm.lstm(self.c, lstm_in)

    self.h = restore_status(y, self.h, h_rest)
    self.cfe = restore_status(y, self.cfe, cfe_rest)
    return y

class LSTMEnc(L.LSTM):
  def __call__(self, x, cond):
    lstm_in = self.upward(x)

    if self.h is not None:
        lstm_in += self.lateral(self.h)
    if self.c is None:
        xp = self.xp
        self.c = variable.Variable(
            xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype),
            volatile='auto')
    _c, _h = lstm.lstm(self.c, lstm_in)

    if self.h is not None:
      self.c = F.where(cond, _c, self.c)
      self.h = F.where(cond, _h, self.h)
    else:
      self.c = _c
      self.h = _h

    return self.h
