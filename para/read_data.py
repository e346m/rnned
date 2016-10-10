#!/usr/bin/env python

import numpy as np
import MeCab

ja_vocab = {}
vocab = {}

def en_load_data(filename):
  global vocab
  words = open(filename).read().replace('\n', ' <eos>').strip().split()
  dataset = np.ndarray((len(words),), dtype=np.int32)
  for i, word in enumerate(words):
    if word not in vocab:
        vocab[word] = len(vocab)
    dataset[i] = vocab[word]
  return dataset


def ja_load_data(filename):
  global ja_vocab
  mt = MeCab.Tagger("-Owakati")

  words = []
  for line in open(filename, "r"):
    words += mt.parse(line).replace('\n', '<eos>').strip().split()

  dataset = np.ndarray((len(words),), dtype=np.int32)
  for i, word in enumerate(words):
    if word not in ja_vocab:
        ja_vocab[word] = len(ja_vocab)
    dataset[i] = ja_vocab[word]
  return dataset
