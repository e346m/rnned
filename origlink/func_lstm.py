import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _extract_gates(x):
    r = x.reshape((len(x), x.shape[1] // 4, 4) + x.shape[2:])
    return [r[:, :, i] for i in six.moves.range(4)]


def _sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def _grad_sigmoid(x):
    return x * (1 - x)


def _grad_tanh(x):
    return 1 - x * x


_preamble = '''
template <typename T> __device__ T sigmoid(T x) { return 1 / (1 + exp(-x)); }
template <typename T> __device__ T grad_sigmoid(T y) { return y * (1 - y); }
template <typename T> __device__ T grad_tanh(T y) { return 1 - y * y; }

#define COMMON_ROUTINE \
    T aa = tanh(a); \
    T ai = sigmoid(i_); \
    T af = sigmoid(f); \
    T ao = sigmoid(o);
'''


class LSTM(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        c_type, x_type = in_types

        type_check.expect(
            c_type.dtype.kind == 'f',
            x_type.dtype == c_type.dtype,

            c_type.ndim >= 2,
            x_type.ndim >= 2,
            c_type.ndim == x_type.ndim,

            x_type.shape[0] <= c_type.shape[0],
            x_type.shape[1] == 4 * c_type.shape[1],
        )
        for i in six.moves.range(2, c_type.ndim.eval()):
            type_check.expect(x_type.shape[i] == c_type.shape[i])

    @profile
    def forward(self, inputs):
        c_prev, y_n_cfe = inputs
        a, i, f, o = _extract_gates(y_n_cfe)
        batch = len(y_n_cfe)

        if isinstance(y_n_cfe, numpy.ndarray):
            self.a = numpy.tanh(a)
            self.i = _sigmoid(i)
            self.f = _sigmoid(f)
            self.o = _sigmoid(o)

            c_next = numpy.empty_like(c_prev)
            c_next[:batch] = self.a * self.i + self.f * c_prev[:batch]
            h = self.o * numpy.tanh(c_next[:batch])
        else:
            c_next = cuda.cupy.empty_like(c_prev)
            h = cuda.cupy.empty_like(c_next[:batch])
            cuda.elementwise(
                'T c_prev, T a, T i_, T f, T o', 'T c, T h',
                '''
                    COMMON_ROUTINE;
                    c = aa * ai + af * c_prev;
                    h = ao * tanh(c);
                ''',
                'lstm_fwd', preamble=_preamble)(
                    c_prev[:batch], a, i, f, o, c_next[:batch], h)

        c_next[batch:] = c_prev[batch:]
        self.c = c_next[:batch]
        return c_next, h

    @profile
    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        c_prev, y_n_cfe = inputs
        batch = len(y_n_cfe)
        gc, gh = grad_outputs

        gy = xp.empty_like(y_n_cfe)
        ga, gi, gf, go = _extract_gates(gy)

        # Consider the case that either gradient is not given
        if gc is None:
            gc_update = 0
            gc_rest = 0
        else:
            gc_update = gc[:batch]
            gc_rest = gc[batch:]
        if gh is None:
            gh = 0

        if xp is numpy:
            co = numpy.tanh(self.c)
            gc_prev = numpy.empty_like(c_prev)
            # multiply f later
            gc_prev[:batch] = gh * self.o * _grad_tanh(co) + gc_update
            gc = gc_prev[:batch]
            ga[:] = gc * self.i * _grad_tanh(self.a)
            gi[:] = gc * self.a * _grad_sigmoid(self.i)
            gf[:] = gc * c_prev[:batch] * _grad_sigmoid(self.f)
            go[:] = gh * co * _grad_sigmoid(self.o)
            gc_prev[:batch] *= self.f  # multiply f here
            gc_prev[batch:] = gc_rest
        else:
            a, i, f, o = _extract_gates(y_n_cfe)
            gc_prev = xp.empty_like(c_prev)
            cuda.elementwise(
                'T c_prev, T c, T gc, T gh, T a, T i_, T f, T o',
                'T gc_prev, T ga, T gi, T gf, T go',
                '''
                    COMMON_ROUTINE;
                    T co = tanh(c);
                    T temp = gh * ao * grad_tanh(co) + gc;
                    ga = temp * ai * grad_tanh(aa);
                    gi = temp * aa * grad_sigmoid(ai);
                    gf = temp * c_prev * grad_sigmoid(af);
                    go = gh * co * grad_sigmoid(ao);
                    gc_prev = temp * af;
                ''',
                'lstm_bwd', preamble=_preamble)(
                    c_prev[:batch], self.c, gc_update, gh, a, i, f, o,
                    gc_prev[:batch], ga, gi, gf, go)
            gc_prev[batch:] = gc_rest

        return gc_prev, gy


def lstm(c_prev, y_n_cfe):
    return LSTM()(c_prev, y_n_cfe)
