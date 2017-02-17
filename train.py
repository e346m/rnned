# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import sys
import six.moves.cPickle as pickle
import numpy as np
import argparse
import time
import datetime

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import variable
from chainer import serializers
from chainer import cuda
from chainer.training import extensions

from ipdb import set_trace
sys.path.append("./mylink")
sys.path.append("./_models")

import read_data as rd
import ext_classifier as ec
import transpose
import data_wrangler
import rnndec
import rnnenc
import middle


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
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--emb_unit', '-eu', type=int, default=100,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--input', '-i', default="./input/",
                        help='Using dirctory')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--itr', '-itr', type=int, default=1000,
                        help='point write timing')
    parser.add_argument('--validation', '-v',
                        help='use validation sentences')
    args = parser.parse_args()

    def get_lines(dataset, _indeces):
        return [dataset[_index] for _index in _indeces]

    def set_clipping(model, gradclip):
        opt_model = chainer.optimizers.AdaDelta()
        opt_model.setup(model)
        opt_model.add_hook(chainer.optimizer.GradientClipping(gradclip))
        return opt_model

    def save_models(i):
       print("epoch ", i, "\n")
       print("loss: ", loss.data, "\n")
       os.mkdir("./%s/itre_%s" % (path, i), 0o755)
       print("save the model")
       serializers.save_npz("./%s/itre_%s/dec.model" % (path, i), dec_model)
       serializers.save_npz("./%s/itre_%s/enc.model" % (path, i), enc_model)
       serializers.save_npz("./%s/itre_%s/middle.model" % (path, i), middle_c)

       print("save the optimizer")
       serializers.save_npz("./%s/itre_%s/dec.state" % (path, i), opt_dec)
       serializers.save_npz("./%s/itre_%s/enc.state" % (path, i), opt_enc)
       serializers.save_npz("./%s/itre_%s/middle.state" % (path, i),
                            opt_middle)

       print("save the loss")
       with open("./%s/itre_%s/report.dump" % (path, i), "wb") as f:
           pickle.dump(report, f)

    # need to use trainer to get it abstract
    def forward_computaion(source, target, rearmost, flag):
        indeces = np.random.permutation(len(source))
        _indeces = indeces[i % rearmost : i % rearmost + args.batchsize]
        dwran._s = get_lines(source, _indeces)
        dwran._t = get_lines(target, _indeces)
        dwran.reverse_source_seq_without_last_word()
        dwran.sort_alignment_key_target()
        dwran.filling_ingnore_label()

        enc.reset_state()  # don't remove ()
        dec.reset_state()  # don't remove ()

        # transposer will be into dwran
        minibatching_s = transposer.transpose_sequnce(dwran._s)
        if args.gpu >= 0:
            minibatching_s = [cuda.to_gpu(seq, device=args.gpu)
                              for seq in minibatching_s]

        for seq in minibatching_s:
            seq = chainer.Variable(seq, volatile=flag)
            opt_enc.target(seq)

        middle_c(opt_enc.target.predictor.l1.h)
        opt_dec.target.predictor.set_initial_l1(middle_c)

        loss = 0
        # transposer will be into dwran
        minibatching_t = transposer.transpose_sequnce(dwran._t)
        first_ids = np.zeros(args.batchsize, dtype=np.int32)
        first_ids.fill(-1)
        minibatching_t.insert(0, first_ids)

        if args.gpu >= 0:
            minibatching_t = [cuda.to_gpu(seq, device=args.gpu)
                              for seq in minibatching_t]

        for num in range(len(minibatching_t) - 1):
            seq = chainer.Variable(minibatching_t[num], volatile=flag)
            next_seq = minibatching_t[num + 1]
            loss += opt_dec.target(seq, middle_c, next_seq)
        return loss

    with open(args.input + "source.sentence", "rb") as f:
        ss = pickle.load(f)
    with open(args.input + "target.sentence", "rb") as f:
        ts = pickle.load(f)
    with open(args.input + "source.vocab", "rb") as f:
        source_vocab = pickle.load(f)
    with open(args.input + "target.vocab", "rb") as f:
        target_vocab = pickle.load(f)
    with open(args.validation + "source.sentence", "rb") as f:
        vss = pickle.load(f)
    with open(args.validation + "target.sentence", "rb") as f:
        vts = pickle.load(f)

    enc = rnnenc.RNNEncoder(len(source_vocab),
                            args.emb_unit, args.unit, args.gpu)
    dec = rnndec.RNNDecoder(len(target_vocab),
                            args.emb_unit, args.unit, args.batchsize, args.gpu)
    middle_c = middle.MiddleC(args.unit)
    enc_model = ec.EncClassifier(enc)
    dec_model = ec.DecClassifier(dec)

    if args.resume:
        print('Load model from %s/dec.model' % args.resume)
        serializers.load_npz("%s/dec.model" % args.resume, dec_model)
        print('Load model from %s/enc.model' % args.resume)
        serializers.load_npz("%s/enc.model" % args.resume, enc_model)
        print('Load model from %s/middle.model' % args.resume)
        serializers.load_npz("%s/middle.model" % args.resume, middle_c)
        print('Load model from %s/report.dump' % args.resume)
        with open("%s/report.dump" % args.resume, "rb") as f:
            report = pickle.load(f)
    else:
        report = {}
        report["train"] = []
        report["validation"] = []
        report["itre_step"] = args.itr

    transposer = transpose.Transpose()
    dwran = data_wrangler.DataWrangler()

    dec_model.compute_accuracy = False
    enc_model.compute_accuracy = False
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # make the GPU current
        dec_model.to_gpu()
        enc_model.to_gpu()
        middle_c.to_gpu()

    opt_enc = set_clipping(enc_model, args.gradclip)
    opt_dec = set_clipping(dec_model, args.gradclip)
    opt_middle = set_clipping(middle_c, args.gradclip)

    limit = len(ss) - args.batchsize
    v_limit = len(vss) - args.batchsize

    for i in range(args.epoch):
        start = time.time()
        print("start epoch:", i, "times\n")
        loss = forward_computaion(ss, ts, limit, "OFF")
        print("forwarding done:", time.time() - start, "s\n")

        report["train"].append(loss.data) #need data coz won't hold history
        opt_dec.target.cleargrads()
        opt_enc.target.cleargrads()
        opt_middle.target.cleargrads()

        loss.backward()  # Backprop
        opt_dec.update()  # Update the parameters
        opt_middle.update()  # Update the parameters
        opt_enc.update()  # Update the parameters
        print("backward done: ", time.time() - start, "s\n")

        if i == 0:
            path = "%s/%s_%s" % (args.input, args.out,
                                datetime.datetime.now().strftime("%s"))
            os.mkdir(path, 0o755)
            vfs = time.time()
            validation_loss = 0
            validation_loss = forward_computaion(vss, vts, v_limit, "ON")
            report["validation"].append(validation_loss.data)
            print("validation done: ", time.time() - vfs, "s\n")
            continue

        if (i + 1) % 25 == 0:
            vfs = time.time()
            validation_loss = 0
            validation_loss = forward_computaion(vss, vts, v_limit, "ON")
            report["validation"].append(validation_loss.data)
            print("validation done: ", time.time() - vfs, "s\n")

        if (i + 1) % args.itr == 0:
            save_models(i + 1)

if __name__ == '__main__':
    main()
