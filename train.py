#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("./origlink")
sys.path.append("./partial_model")
import argparse
import read_data as rd
import ext_classifier as ec
import transpose
import rnndec
import rnnenc
import middle
#from pudb import set_trace
from ipdb import set_trace

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import variable
from chainer import serializers
from chainer.training import extensions

#class RNNED(chainer.Chain):
#  def __init__(self, source_vocab, target_vocab, n_units):
#    super(RNNED, self).__init__(
#    rnnenc = rnnenc.RNNEncoder(source_vocab, n_units),
#    rnndec = rnndec.RNNDecoder(target_vocab, n_units),
#    middle = middle.MiddleC(n_units)
#    )

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batchsize', '-b', type=int, default=20,
      help='Number of examples in each mini batch')
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

  ja, source_vocab = rd.ja_load_data('./ja.utf')
  en, target_vocab = rd.en_load_data('./en.utf')

  #rnned = RNNED(source_vocab, target_vocab, args.unit)

  enc = rnnenc.RNNEncoder(source_vocab, args.unit)
  dec = rnndec.RNNDecoder(target_vocab, args.unit)
  middle_c = middle.MiddleC(args.unit)

  enc_model = ec.EncClassifier(enc)
  dec_model = ec.DecClassifier(dec)
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
  opt_middle.setup(middle_c)
  opt_middle.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

  for i in range(args.epoch):
    _ja = get_lines(ja, i)
    _en = get_lines(en, i)

    desc_order_seq(_ja)
    desc_order_seq(_en)
    enc.reset_state()
    dec.reset_state(target_vocab)

    minibatching_ja = transposer.transpose_sequnce(_ja)
    for seq in minibatching_ja:
      opt_enc.target(seq)

    set_trace()
    middle_c(opt_enc)

    loss = 0
    minibatching_en = transposer.transpose_sequnce(_en)

    for i, seq in enumerate(minibatching_en):
      loss += opt_dec.target(seq, middle_c, i)
    print(loss.data)

    opt_dec.target.cleargrads()  # Clear the parameter gradients
    opt_enc.target.cleargrads()  # Clear the parameter gradients
    opt_middle.target.cleargrads()  # Clear the parameter gradients
    loss.backward()  # Backprop
    opt_dec.update()  # Update the parameters
    opt_enc.update()  # Update the parameters
    opt_middle.update()

  print("save the model")
  serializers.save_npz("dec.model", dec_model)
  serializers.save_npz("enc.model", enc_model)
  serializers.save_npz("midlle.model", middle_c)
  print("save the optimizer")
  serializers.save_npz("dec.state", opt_dec)
  serializers.save_npz("enc.state", opt_enc)

if __name__ == '__main__':
  main()
