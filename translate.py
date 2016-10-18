#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("./origlink")
sys.path.append("./partial_model")

import argparse

import numpy as np
import six
import six.moves.cPickle as pickle

import rnndec
import rnnenc
import middle
import ext_classifier as ec
import MeCab

import chainer
from chainer import serializers
from chainer import variable
from ipdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument('--unit', '-u', type=int, default=650,
                    help='Number of LSTM units in each layer')
parser.add_argument('--dir', '-d', default="",
                    help='Which result data')

parser.set_defaults(test=False)
args = parser.parse_args()

with open("%s/ja_vocab.bin" %args.dir, "r") as f:
  ja_vocab = pickle.load(f)

with open("%s/en_vocab.bin" %args.dir, "r") as f:
  en_vocab = pickle.load(f)

enc = rnnenc.RNNEncoder(len(ja_vocab), args.unit)
dec = rnndec.RNNDecoder(len(en_vocab), args.unit)
middle_c = middle.MiddleC(args.unit)

enc_model = ec.EncClassifier(enc)
dec_model = ec.DecClassifier(dec)

enc_model.train = False
dec_model.train = False
dec_model.predictor.reset_state(len(en_vocab))


if args.dir:
    print('Load model from %s/dec.model' %args.dir )
    serializers.load_npz("%s/dec.model" %args.dir, dec_model)
    print('Load model from %s/enc.model' %args.dir )
    serializers.load_npz("%s/enc.model" %args.dir, enc_model)
    print('Load model from %s/middle.model' %args.dir )
    serializers.load_npz("%s/middle.model" %args.dir, middle_c)

mt = MeCab.Tagger("-Owakati")
unk_id = ja_vocab["unk"]

while True:
  print("日本語を入力してください 終了する場合はexitを入力してください")
  line = raw_input(">>> ")
  if line == "exit":
    break

  inputs = mt.parse(line).strip().split()
  inputs.append("<eos>")
  ids = [ja_vocab.get(word, unk_id) for word in inputs]

  for _id in ids:
    enc_model(np.array([_id], dtype=np.int32))

  middle_c(enc_model)

  first_y = np.array([0], dtype=np.int32)
  i = 0
  rev_en_vocab = {v:k for k, v in en_vocab.items()}

  word = []
  while True:
    y = dec_model.predictor(first_y, middle_c, i)
    i += 1
    wid = y.data.argmax(1)[0]
    word.append(rev_en_vocab[wid])
    if wid == en_vocab["<eos>"]:
      break
    elif i == 30:
      break
  print(word)
