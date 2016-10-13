# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("./origlink")
sys.path.append("./partial_model")

import argparse

import numpy as np
import six

import rnndec
import rnnenc
import middle
import ext_classifier as ec

import chainer
from chainer import serializers
from ipdb import set_trace

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

#復元処理

enc = rnnenc.RNNEncoder(len(ja_vocab), args.unit)
dec = rnndec.RNNDecoder(len(en_vocab), args.unit)
middle_c = middle.MiddleC(args.unit)

enc_model = ec.EncClassifier(enc)
dec_model = ec.DecClassifier(dec)

if args.dec_model:
    print('Load model from', args.dec_model)
    serializers.load_npz(args.dec_model, dec_model)
if args.enc_model:
    print('Load model from', args.enc_model)
    serializers.load_npz(args.enc_model, enc_model)
if args.middle_model:
    print('Load model from', args.middle_model)
    serializers.load_npz(args.middle_model, middle_c)

inputs = "人 は 皆 、 労働 を やめる べき で ある 。"
ids = [ja_vocab[word] for words in inputs]

enc_model(ids)
middle_c(enc_model) #中で呼び出している構造があるからそれに合わせる必要があるかも

#try:
#    while True:
#        q = six.moves.input('>> ')
#
