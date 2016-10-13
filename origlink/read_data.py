#!/usr/bin/env python
import six.moves.cPickle as pickle
import numpy as np
import MeCab

def en_load_data(filename):
  lines = []
  for line in open(filename, "r"):
    lines.append(line.lower().replace('\n', '<eos>').strip().split())

  return r_info(lines, "en_vocab.bin")

def ja_load_data(filename):
  mt = MeCab.Tagger("-Owakati")

  lines = []
  for line in open(filename, "r"):
    lines.append(mt.parse(line).replace('\n', '<eos>').strip().split())

  return r_info(lines, "ja_vocab.bin")

def r_info(lines, file_name):
  vocab = {}
  dataset = []
  for line in lines:
    tmp_line = []
    for word in line:
      if word not in vocab:
        vocab[word] = len(vocab)
      tmp_line.append(vocab[word])
    dataset.append(np.array(tmp_line, dtype=np.int32))

  with open(file_name, 'wb') as f:
      pickle.dump(vocab, f)
  return dataset, len(vocab)
