import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

class MiddleC(chainer.Chain):
  @profile
  def __init__(self, n_units, train=True):
    super(MiddleC, self).__init__(
      to_c = L.Linear(n_units, n_units),
      from_c = L.Linear(n_units, n_units),
      )
    for param in self.params():
      param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
      self.train = train

  @profile
  def __call__(self, target):
    Vh = self.to_c(target.predictor.l1.h)
    self.mid_c = F.tanh(Vh)
    self.dec_h0 = F.tanh(self.from_c(self.mid_c))
    return None
