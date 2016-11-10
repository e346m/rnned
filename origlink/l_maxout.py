import numpy as np

from chainer import cuda
from chainer.functions.activation import maxout
from chainer import link
from chainer.links.connection import linear
from chainer.functions.connection import linear as fl
from chainer.utils import type_check
from chainer.functions.array import split_axis
#from pudb import set_trace
from ipdb import set_trace
#set_trace()

class FLinear(fl.LinearFunction):
    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim >= 2,
            w_type.ndim == 2,
            type_check.prod(x_type.shape[1:]) <= w_type.shape[1],
        )
        if n_in.eval() == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] <= w_type.shape[0],
            )

class LLinear(linear.Linear):
    def __call__(self, x):
        if self.has_uninitialized_params:
            with cuda.get_device(self._device_id):
                self._initialize_params(x.size // len(x.data))
        return FLinear()(x, self.W, self.b)

class Maxout(link.Chain):
  #TODO ask k input out pool to pfn with my option
  def __init__(self, in_size, emb_size ,out_size, pool_size,
               wscale=1, initialW=None, initial_bias=0):
      linear_out_size = out_size * pool_size
      if initialW is not None:
          initialW = initialW.reshape(linear_out_size, in_size)

      if initial_bias is not None:
          if np.isscalar(initial_bias):
              initial_bias = np.full(
                  (linear_out_size,), initial_bias, dtype=np.float32)
          elif isinstance(initial_bias, (np.ndarray, cuda.ndarray)):
              initial_bias = initial_bias.reshape(linear_out_size)
          else:
              raise ValueError(
                  'initial bias must be float, ndarray, or None')

      super(Maxout, self).__init__(
          upward=LLinear(
              in_size, linear_out_size, wscale,
              nobias=initial_bias is None, initialW=initialW,
              initial_bias=initial_bias),
          lateral=LLinear(
              emb_size, linear_out_size, wscale,
              nobias=initial_bias is None, initialW=initialW,
              initial_bias=initial_bias),
          diagonal=LLinear(
              in_size, linear_out_size, wscale,
              nobias=initial_bias is None, initialW=initialW,
              initial_bias=initial_bias))
      self.out_size = out_size
      self.pool_size = pool_size

  def __call__(self, hidd, prev_y, cfe):
    s_prime = self.upward(hidd) + self.lateral(prev_y) + self.diagonal(cfe)
    return maxout.maxout(s_prime, self.pool_size)
