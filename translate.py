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
#from ipdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument('--dec_model', '-dec', default='',
                    help='Initialize the model from given file')
parser.add_argument('--enc_model', '-enc', default='',
                    help='Initialize the model from given file')
parser.add_argument('--middle_model', '-mid', default='',
                    help='Initialize the model from given file')
parser.add_argument('--unit', '-u', type=int, default=650,
                    help='Number of LSTM units in each layer')
parser.set_defaults(test=False)
args = parser.parse_args()

with open("ja_vocab.bin", "r") as f:
  ja_vocab = pickle.load(f)

with open("en_vocab.bin", "r") as f:
  en_vocab = pickle.load(f)

enc = rnnenc.RNNEncoder(len(ja_vocab), args.unit)
dec = rnndec.RNNDecoder(len(en_vocab), args.unit)
middle_c = middle.MiddleC(args.unit)

enc_model = ec.EncClassifier(enc)
dec_model = ec.DecClassifier(dec)

enc_model.train = False
dec_model.train = False
dec_model.predictor.reset_state(len(en_vocab))

if args.dec_model:
    print('Load model from', args.dec_model)
    serializers.load_npz(args.dec_model, dec_model)
if args.enc_model:
    print('Load model from', args.enc_model)
    serializers.load_npz(args.enc_model, enc_model)
if args.middle_model:
    print('Load model from', args.middle_model)
    serializers.load_npz(args.middle_model, middle_c)

mt = MeCab.Tagger("-Owakati")

while True:
  print("日本語を入力してください 終了する場合はexitを入力してください")
  line = raw_input(">>> ")
  if line == "exit":
    break

  inputs = mt.parse(line).strip().split()
  inputs.append("<eos>")
  ids = [ja_vocab.get(word, "unk") for word in inputs]

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
