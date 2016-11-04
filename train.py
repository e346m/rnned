#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append("./origlink")
sys.path.append("./partial_model")
import six.moves.cPickle as pickle
import argparse
import time

import read_data as rd
import ext_classifier as ec
import transpose
import rnndec
import rnnenc
import middle
#from pudb import set_trace
from ipdb import set_trace

import datetime
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import variable
from chainer import serializers
from chainer.training import extensions
import chainer.computational_graph as c

#class RNNED(chainer.Chain):
#  def __init__(self, source_vocab, target_vocab, n_units):
#    super(RNNED, self).__init__(
#    rnnenc = rnnenc.RNNEncoder(source_vocab, n_units),
#    rnndec = rnndec.RNNDecoder(target_vocab, n_units),
#    middle = middle.MiddleC(n_units)
#    )

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batchsize', '-b', type=int, default=64,
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
  parser.add_argument('--unit', '-u', type=int, default=1000,
      help='Number of LSTM units in each layer')
  parser.add_argument('--emb_unit', '-eu', type=int, default=100,
      help='Number of LSTM units in each layer')
  parser.add_argument('--source_s', '-ss', default="./input/source.sentence",
      help='Source file path')
  parser.add_argument('--target_s', '-ts', default="./input/target.sentence",
      help='Target file path')
  parser.add_argument('--source_v', '-sv', default="./input/source.vocab",
      help='Source file path')
  parser.add_argument('--target_v', '-tv', default="./input/target.vocab",
      help='Target file path')
  args = parser.parse_args()

  def desc_order_seq(dataset):
    dataset.sort(key=lambda x: len(x))
    dataset.reverse()

  def get_lines(dataset, itre):
    offsets = [i * len(dataset) // args.batchsize for i in range(args.batchsize)]
    return [dataset[((itre + offset) % len(dataset))] for offset in offsets]

  with open(args.source_s, "r") as f:
    s = pickle.load(f)
  with open(args.target_s, "r") as f:
    t = pickle.load(f)
  with open(args.source_v, "r") as f:
    source_vocab = pickle.load(f)
  with open(args.target_v, "r") as f:
    target_vocab = pickle.load(f)

  #rnned = RNNED(source_vocab, target_vocab, args.unit, args.batchsize)
  enc = rnnenc.RNNEncoder(len(source_vocab), args.emb_unit, args.unit)
  dec = rnndec.RNNDecoder(len(target_vocab), args.emb_unit, args.unit, args.batchsize)
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
  opt_enc.setup(enc_model)
  opt_enc.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

  opt_dec = chainer.optimizers.SGD(lr=0.5)
  opt_dec.setup(dec_model)
  opt_dec.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

  opt_middle = chainer.optimizers.SGD(lr=0.5)
  opt_middle.setup(middle_c)
  opt_middle.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

  for i in range(args.epoch):
    _s = get_lines(s, i)
    _t = get_lines(t, i)

    desc_order_seq(_s)
    desc_order_seq(_t)
    enc.reset_state()
    dec.reset_state()
    middle_c.reset_state()

    minibatching_s = transposer.transpose_sequnce(_s)
    for seq in minibatching_s:
      opt_enc.target(seq)

    middle_c(opt_enc.target)
    opt_dec.target.predictor.set_l1(middle_c)

    loss = 0
    minibatching_t = transposer.transpose_sequnce(_t)

    for seq in minibatching_t:
      loss += opt_dec.target(seq[::-1], middle_c)
      print(loss.data)

    opt_dec.target.cleargrads()  # Clear the parameter gradients
    opt_enc.target.cleargrads()  # Clear the parameter gradients
    opt_middle.target.cleargrads()  # Clear the parameter gradients

    start = time.clock()
    print ("backward\n")
    loss.backward()  # Backprop
    print ("done: ", time.clock() - start, "\n")
    opt_dec.update()  # Update the parameters
    opt_enc.update()  # Update the parameters
    opt_middle.update()

    #if i == 1:
    #  with open("graph.dot", "w") as o:
    #      variable_style = {"shape": "octagon", "fillcolor": "#E0E0E0",
    #                        "style": "filled"}
    #      function_style = {"shape": "record", "fillcolor": "#6495ED",
    #                        "style": "filled"}
    #      g = c.build_computational_graph(
    #          (loss, ),
    #          variable_style=variable_style,
    #          function_style=function_style)
    #      o.write(g.dump())
    #  print("graph generated")

  path = "./%s"  %datetime.datetime.now().strftime("%s")
  print("save the model")
  os.mkdir(path, 0755)
  serializers.save_npz("./%s/dec.model" %path, dec_model)
  serializers.save_npz("./%s/enc.model" %path, enc_model)
  serializers.save_npz("./%s/middle.model" %path, middle_c)

  print("save the optimizer")
  serializers.save_npz("./%s/dec.state" %path, opt_dec)
  serializers.save_npz("./%s/enc.state" %path, opt_enc)
  serializers.save_npz("./%s/middle.state" %path, opt_middle)

if __name__ == '__main__':
  main()
