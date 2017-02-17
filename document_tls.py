from __future__ import division
from __future__ import print_function
import argparse
import chainer
import chainer.functions as F
from chainer import serializers
from chainer import variable
from ipdb import set_trace

import numpy as np
import six
import six.moves.cPickle as pickle
from pathlib import Path

import sys
sys.path.append("./mylink")
sys.path.append("./_models")

import rnndec
import rnnenc
import middle
import ext_classifier as ec


parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', '-b', type=int, default=64,
                    help='Number of examples in each mini batch')
parser.add_argument('--unit', '-u', type=int, default=1000,
                    help='Number of LSTM units in each layer')
parser.add_argument('--emb_unit', '-eu', type=int, default=100,
                    help='Number of LSTM units in each layer')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--length', type=int, default=20,
                    help='length of the generated text')
parser.add_argument('--models', '-m', default="",
                    help='result models')
parser.add_argument('--source_sentence', '-s', default="",
                    help='input sentence')
parser.add_argument('--output', '-o', default="",
                    help='translated sentence')

parser.set_defaults(test=False)
args = parser.parse_args()

parent = Path(args.models).parent.parent
source_path = parent / "source.vocab"
target_path = parent / "target.vocab"
with source_path.open("rb") as f:
    source_vocab = pickle.load(f, encoding='bytes')

with target_path.open("rb") as f:
    target_vocab = pickle.load(f, encoding='bytes')


def softmax(x):
    e = numpy.exp(x - numpy.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    else:
        return e / numpy.array([numpy.sum(e, axis=1)]).T

enc = rnnenc.RNNEncoder(len(source_vocab),
                        args.emb_unit, args.unit, args.gpu, train=False)
dec = rnndec.RNNDecoder(len(target_vocab),
                        args.emb_unit, args.unit, args.batchsize,
                        args.gpu, train=False)
middle_c = middle.MiddleC(args.unit, train=False)

enc_model = ec.EncClassifier(enc)
dec_model = ec.DecClassifier(dec)


if args.models:
    print('Load model from %s/dec.model' % args.models)
    serializers.load_npz("%s/dec.model" % args.models, dec_model)
    print('Load model from %s/enc.model' % args.models)
    serializers.load_npz("%s/enc.model" % args.models, enc_model)
    print('Load model from %s/middle.model' % args.models)
    serializers.load_npz("%s/middle.model" % args.models, middle_c)

dec_model.to_cpu()
enc_model.to_cpu()
middle_c.to_cpu()
unk_id = source_vocab["<unk>"]

source_embeddings = {}
target_embeddings = {}
for word, ID in source_vocab.items():
    source_embeddings[word] = enc.emdeddings(ID)
for word, ID in target_vocab.items():
    target_embeddings[word] = dec.emdeddings(ID)

out = []
with open("%s" % args.source_sentence, "r") as f:
    for line in f:
        enc.reset_state()
        dec.reset_state()

        inputs = line.strip().split()
        inputs = inputs[::-1]
        inputs.append("<eos>")
        ids = [source_vocab.get(word, unk_id) for word in inputs]
        rev_source_vocab = {v:k for k, v in source_vocab.items()}
        for _id in ids:
            #print(rev_source_vocab[_id])
            seq = chainer.Variable(np.array([_id], dtype=np.int32), volatile="OFF")
            enc_model.predictor(seq)

        middle_c(enc_model.predictor.l1.h)
        dec_model.predictor.set_initial_l1(middle_c)

        prev_y = chainer.Variable(np.array([-1], dtype=np.int32), volatile="OFF")
        rev_target_vocab = {v:k for k, v in target_vocab.items()}

        tmp_out = []
        for i in six.moves.range(args.length):
            _prob = dec_model.predictor(prev_y, middle_c, 1)
            prob = F.softmax(_prob)
            wid = prob.data.argmax(1)[0]

            if rev_target_vocab[wid] == '<eos>':
                break
            else:
                tmp_out.append(rev_target_vocab[wid])

            prev_y = chainer.Variable(np.array([wid], dtype=np.int32), volatile="OFF")
        tmp_out.append('\n')
        out.append(tmp_out)

with open("%s/translated" % args.output, "w") as f:
    for line in out:
        f.write(" ".join(line))

with open("%s/source_embeddings" % args.output, "wb") as f:
    pickle.dump(source_embeddings, f)
with open("%s/target_embeddings" % args.output, "wb") as f:
    pickle.dump(target_embeddings, f)
