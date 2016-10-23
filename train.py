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
  parser.add_argument('--source', '-s', default="./ja.utf",
      help='Source file path')
  parser.add_argument('--target', '-t', default="./en.utf",
      help='Target file path')
  args = parser.parse_args()

  def desc_order_seq(dataset):
    dataset.sort(key=lambda x: len(x))
    dataset.reverse()

  def get_lines(dataset, itre):
    offsets = [i * len(dataset) // args.batchsize for i in range(args.batchsize)]
    return [dataset[((itre + offset) % len(dataset))] for offset in offsets]

  def save_vocab(vocab, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(vocab, f)

  def save_sentence(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

  start = time.clock()
  print ("start reading ja file\n")
  ja, source_vocab = rd.ja_load_data(args.source)
  print ("read ja file: ", time.clock() - start, "\n")

  start = time.clock()
  print ("start reading en file\n")
  en, target_vocab = rd.en_load_data(args.target)
  print ("read en file: ", time.clock() - start, "\n")

  set_trace()

  #rnned = RNNED(source_vocab, target_vocab, args.unit)
  enc = rnnenc.RNNEncoder(len(source_vocab), args.unit)
  dec = rnndec.RNNDecoder(len(target_vocab), args.unit, args.batchsize)
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
    dec.reset_state()

    minibatching_ja = transposer.transpose_sequnce(_ja)
    for seq in minibatching_ja:
      opt_enc.target(seq)

    middle_c(opt_enc.target)

    loss = 0
    minibatching_en = transposer.transpose_sequnce(_en)

    for i, seq in enumerate(minibatching_en):
      loss += opt_dec.target(seq, middle_c, i)
      print(loss.data)

    opt_dec.target.cleargrads()  # Clear the parameter gradients
    opt_enc.target.cleargrads()  # Clear the parameter gradients
    opt_middle.target.cleargrads()  # Clear the parameter gradients
    start = time.clock()
    print ("backward\n")
    loss.backward()  # Backprop
    print ("done: ", time.clock() - start, "\n")
    start = time.clock()
    print ("dec update\n")
    opt_dec.update()  # Update the parameters
    print ("done: ", time.clock() - start, "\n")
    start = time.clock()
    print ("enc update\n")
    opt_enc.update()  # Update the parameters
    print ("done: ", time.clock() - start, "\n")
    start = time.clock()
    print ("middle update\n")
    opt_middle.update()
    print ("done: ", time.clock() - start, "\n")

    if i == 1:
      with open("graph.dot", "w") as o:
          variable_style = {"shape": "octagon", "fillcolor": "#E0E0E0",
                            "style": "filled"}
          function_style = {"shape": "record", "fillcolor": "#6495ED",
                            "style": "filled"}
          g = c.build_computational_graph(
              (loss, ),
              variable_style=variable_style,
              function_style=function_style)
          o.write(g.dump())
      print("graph generated")

  path = "./%s"  %datetime.datetime.now().strftime("%s")
  print("save the model")
  os.mkdir(path, 0755)
  serializers.save_npz("./%s/dec.model" %path, dec_model)
  serializers.save_npz("./%s/enc.model" %path, enc_model)
  serializers.save_npz("./%s/middle.model" %path, middle_c)
  print("save the optimizer")
  serializers.save_npz("./%s/dec.state" %path, opt_dec)
  serializers.save_npz("./%s/enc.state" %path, opt_enc)

  print("save vocab")
  save_vocab(source_vocab, "./%s/ja.bin" %path)
  save_vocab(target_vocab, "./%s/en.bin" %path)

  print("save sentence")
  save_vocab(ja, "./%s/ja.sentence" %path)
  save_vocab(en, "./%s/en.sentence" %path)

if __name__ == '__main__':
  main()
