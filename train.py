#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append("./origlink")
sys.path.append("./partial_model")
import six.moves.cPickle as pickle
import numpy as np
import argparse
import time

import read_data as rd
import ext_classifier as ec
import transpose
import rnndec
import rnnenc
import middle
from ipdb import set_trace

import datetime
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import variable
from chainer import serializers
from chainer import cuda
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
  parser.add_argument('--dir', '-d', default="./input/",
    help='Using dirctory')
  parser.add_argument('--output_label', '-ol', default="",
    help='output label')
  parser.add_argument('--eval', '-eval',
    help='Eval')
  args = parser.parse_args()

  def concatinate_sort(_s, _t):
    dataset = [[__s, __t] for __s, __t in zip(_s, _t)]
    dataset.sort(key=lambda x: len(x[1]))
    dataset.reverse()
    return dataset

  def separate_dataset(dataset):
    t = []
    s = []
    for data in dataset:
      tmp_t = data.pop()
      #TODO for gpu use xp system in initialze
      t.append(np.concatenate((tmp_t[:-1][::-1], tmp_t[-1:])))
      s.append(data[0])
    return s, t

  def formating(s, ls):
    ret = []
    for _s in s:
      diff = ls - len(_s)
      if diff is not 0:
        balance = np.empty(diff, np.int32)
        balance.fill(-1)
        _s = np.hstack((_s, balance))
      ret.append(_s)
    return ret

  def get_lines(dataset, _indeces):
    return [dataset[_index] for _index in _indeces]

  with open(args.dir + "source.sentence", "r") as f:
    ss = pickle.load(f)
  with open(args.dir + "target.sentence", "r") as f:
    ts = pickle.load(f)
  with open(args.dir + "source.vocab", "r") as f:
    source_vocab = pickle.load(f)
  with open(args.dir + "target.vocab", "r") as f:
    target_vocab = pickle.load(f)

  #rnned = RNNED(source_vocab, target_vocab, args.unit, args.batchsize)
  enc = rnnenc.RNNEncoder(len(source_vocab), args.emb_unit, args.unit, args.gpu)
  dec = rnndec.RNNDecoder(len(target_vocab), args.emb_unit, args.unit, args.batchsize, args.gpu)
  middle_c = middle.MiddleC(args.unit)

  enc_model = ec.EncClassifier(enc)
  dec_model = ec.DecClassifier(dec)
  transposer = transpose.Transpose()

  if args.eval:
    print('Load model from %s/dec.model' %args.eval )
    serializers.load_npz("%s/dec.model" %args.eval, dec_model)
    print('Load model from %s/enc.model' %args.eval )
    serializers.load_npz("%s/enc.model" %args.eval, enc_model)
    print('Load model from %s/middle.model' %args.eval )
    serializers.load_npz("%s/middle.model" %args.eval, middle_c)

  dec_model.compute_accuracy = False
  enc_model.compute_accuracy = False
  if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()  # make the GPU current
    dec_model.to_gpu()
    enc_model.to_gpu()
    middle_c.to_gpu()

  opt_enc = chainer.optimizers.AdaDelta()
  opt_enc.setup(enc_model)
  opt_enc.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

  opt_dec = chainer.optimizers.AdaDelta()
  opt_dec.setup(dec_model)
  opt_dec.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

  opt_middle = chainer.optimizers.AdaDelta()
  opt_middle.setup(middle_c)
  opt_middle.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

  indeces = np.random.permutation(len(ss))
  limit = len(ss) - args.batchsize
  report = []

  #if args.eval:
  #  i = np.random.permutation(1)
  #  _indeces = indeces[i[0] % limit : i[0] % limit + args.batchsize]
  #  _s = get_lines(ss, _indeces)
  #  _t = get_lines(ts, _indeces)

  for i in range(args.epoch):
    start = time.time()
    print ("start epoch:", i, "times\n")
    _indeces = indeces[i % limit : i % limit + args.batchsize]
    _s = get_lines(ss, _indeces)
    _t = get_lines(ts, _indeces)

    dataset = concatinate_sort(_s, _t)
    _s, _t = separate_dataset(dataset)

    largest_size = len(max(_s, key=lambda x: len(x)))
    _s = formating(_s, largest_size)

    enc.reset_state()
    dec.reset_state()

    minibatching_s = transposer.transpose_sequnce(_s)
    if args.gpu >= 0:
      minibatching_s = [cuda.to_gpu(seq, device=args.gpu) for seq in minibatching_s]

    for seq in minibatching_s:
      opt_enc.target(seq)

    middle_c(opt_enc.target.predictor.l1.h)
    opt_dec.target.predictor.set_l1(middle_c)

    loss = 0
    minibatching_t = transposer.transpose_sequnce(_t)
    first_ids = np.zeros(args.batchsize, dtype=np.int32)
    first_ids.fill(-1)
    minibatching_t.insert(0, first_ids)

    if args.gpu >= 0:
      minibatching_t = [cuda.to_gpu(seq, device=args.gpu) for seq in minibatching_t]

    for num in range(len(minibatching_t) - 1):
      seq = minibatching_t[num]
      next_seq = minibatching_t[num + 1]
      loss += opt_dec.target(seq, middle_c, next_seq)

    report.append(loss.data)
    opt_dec.target.cleargrads()
    opt_enc.target.cleargrads()
    opt_middle.target.cleargrads()

    loss.backward()  # Backprop
    opt_dec.update()  # Update the parameters
    opt_middle.update() # Update the parameters
    opt_enc.update()  # Update the parameters
    print ("done: ", time.time() - start, "s\n")

    if i == 0:
      path = "./%s_%s" %(args.output_label, datetime.datetime.now().strftime("%s"))
      os.mkdir(path, 0755)
      continue
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

    if i % 500 == 0:
      print ("epoch ", i, "\n")
      print("loss: ", loss.data, "\n")
      os.mkdir("./%s/%s" %(path, i), 0755)
      print("save the model")
      serializers.save_npz("./%s/%s/dec.model" %(path, i), dec_model)
      serializers.save_npz("./%s/%s/enc.model" %(path, i), enc_model)
      serializers.save_npz("./%s/%s/middle.model" %(path, i), middle_c)

      print("save the optimizer")
      serializers.save_npz("./%s/%s/dec.state" %(path, i), opt_dec)
      serializers.save_npz("./%s/%s/enc.state" %(path, i), opt_enc)
      serializers.save_npz("./%s/%s/middle.state" %(path, i), opt_middle)

      print("save the loss")
      with open("./%s/%s/report.dump" %(path, i), "wb") as f:
        pickle.dump(report, f)

  print ("epoch ", i, "\n")
  print("loss: ", loss.data, "\n")
  os.mkdir("./%s/%s" %(path, i), 0755)
  print("save the model")
  serializers.save_npz("./%s/%s/dec.model" %(path, i), dec_model)
  serializers.save_npz("./%s/%s/enc.model" %(path, i), enc_model)
  serializers.save_npz("./%s/%s/middle.model" %(path, i), middle_c)

  print("save the optimizer")
  serializers.save_npz("./%s/%s/dec.state" %(path, i), opt_dec)
  serializers.save_npz("./%s/%s/enc.state" %(path, i), opt_enc)
  serializers.save_npz("./%s/%s/middle.state" %(path, i), opt_middle)

  print("save the loss")
  with open("./%s/%s/report.dump" %(path, i), "wb") as f:
    pickle.dump(report, f)

if __name__ == '__main__':
  main()
