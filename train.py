#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import argparse
import read_data
import ext_classifier
import lstm as ll
import l_maxout as lm
import transpose

#from pudb import set_trace
from ipdb import set_trace

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import variable
from chainer.training import extensions

class RNNEncoder(chainer.Chain):
  def __init__(self, source_vocab, n_units, train=True):
    super(RNNEncoder, self).__init__(
      embed = L.EmbedID(source_vocab, n_units),
      l1 = L.LSTM(n_units, n_units),
      )
    for param in self.params():
      param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
      self.train = train

  def reset_state(self):
    self.l1.reset_state()

  def __call__(self, x):
    x_cswr = self.embed(x)
    self.l1(F.dropout(x_cswr, train=self.train))
    return None

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

class MiddleC(chainer.Chain):
  def __init__(self, n_units, train=True):
    super(MiddleC, self).__init__(
      to_c = L.Linear(n_units, n_units),
      from_c = L.Linear(n_units, n_units),
      )
    for param in self.params():
      param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
      self.train = train

  def __call__(self, opt_enc):
    Vh = self.to_c(opt_enc.target.predictor.l1.h)
    self.mid_c = F.tanh(Vh)
    self.dec_h0 = F.tanh(self.from_c(self.mid_c))
    return None

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batchsize', '-b', type=int, default=20,
      help='Number of examples in each mini batch')
  #parser.add_argument('--bproplen', '-l', type=int, default=35,
  #    help='Number of words in each mini batch '
  #    '(= length of truncated BPTT)')
  parser.add_argument('--epoch', '-e', type=int, default=39,
      help='Number of sweeps over the dataset to train')
  parser.add_argument('--gpu', '-g', type=int, default=-1,
      help='GPU ID (negative value indicates CPU)')
  parser.add_argument('--gradclip', '-c', type=float, default=5,
      help='Gradient norm threshold to clip')
  parser.add_argument('--out', '-o', default='result',
      help='Directory to output the result')
  parser.add_argument('--resume', '-r', default='',
      help='Resume the training from snapshot')
  parser.add_argument('--test', action='store_true',
      help='Use tiny datasets for quick tests')
  parser.set_defaults(test=False)
  parser.add_argument('--unit', '-u', type=int, default=650,
      help='Number of LSTM units in each layer')
  args = parser.parse_args()

  def desc_order_seq(dataset):
    dataset.sort(key=lambda x: len(x))
    dataset.reverse()

  def get_lines(dataset, itre):
    offsets = [i * len(dataset) // args.batchsize for i in range(args.batchsize)]
    return [dataset[((itre + offset) % len(dataset))] for offset in offsets]

  ja, source_vocab = read_data.ja_load_data('./ja.utf')
  en, target_vocab = read_data.en_load_data('./en.utf')

  rnnenc = RNNEncoder(source_vocab, args.unit)
  rnndec = RNNDecoder(target_vocab, args.unit)
  middle = MiddleC(args.unit)
  enc_model = ext_classifier.EncClassifier(rnnenc)
  dec_model = ext_classifier.DecClassifier(rnndec)
  transposer = transpose.Transpose()

  dec_model.compute_accuracy = False  # we only want the perplexity
  enc_model.compute_accuracy = False  # we only want the perplexity
  if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()  # make the GPU current
    dec_model.to_gpu()
    enc_model.to_gpu()

  opt_enc = chainer.optimizers.SGD(lr=0.5)
  opt_enc.setup(enc_model) #RNNENCの中でoptimizerを呼び出せるようにするかclassiierを拡張する必要がある
  opt_enc.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

  opt_dec = chainer.optimizers.SGD(lr=0.5)
  opt_dec.setup(dec_model)
  opt_dec.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

  opt_middle = chainer.optimizers.SGD(lr=0.5)
  opt_middle.setup(middle)
  opt_middle.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

  for i in range(args.epoch):
    _ja = get_lines(ja, i)
    _en = get_lines(en, i)

    desc_order_seq(_ja)
    desc_order_seq(_en)
    rnnenc.reset_state()
    rnndec.reset_state(target_vocab)

    minibatching_ja = transposer.transpose_sequnce(_ja)
    for seq in minibatching_ja:
      opt_enc.target(seq)

    middle(opt_enc)

    loss = 0
    minibatching_en = transposer.transpose_sequnce(_en)

    for i, seq in enumerate(minibatching_en):
      loss += opt_dec.target(seq, middle.mid_c, i, middle.dec_h0)
    print(loss.data)

    opt_dec.target.cleargrads()  # Clear the parameter gradients
    opt_enc.target.cleargrads()  # Clear the parameter gradients
    opt_middle.target.cleargrads()  # Clear the parameter gradients
    loss.backward()  # Backprop
    opt_dec.update()  # Update the parameters
    opt_enc.update()  # Update the parameters
    opt_middle.update()

if __name__ == '__main__':
  main()
