#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import MeCab
from collections import Counter
from itertools import chain
from ipdb import set_trace
UNK = "unk"

def en_load_data(filename):
  lines = []
  for line in open(filename, "r"):
    lines.append(line.lower().replace('\n', ' <eos>').strip().split())

  return r_info(lines)

def ja_load_data(filename):
  mt = MeCab.Tagger("-Owakati")

  lines = []
  for line in open(filename, "r"):
    lines.append(mt.parse(line).replace('\n', ' <eos>').strip().split())

  return r_info(lines)

def r_info(lines):
  vocab = {}
  dataset = []
  flat_lines = chain.from_iterable(lines)
  count = Counter(flat_lines)
  #TODO UKNの扱いは1回しか出てきてない単語かそれとも上位~単語以外の単語をuknにするか..
  UNKS = [k for k, v in count.items() if v == 1]
  for line in lines:
    tmp_line = []
    for word in line:
      if word in UNKS:
        word = UNK
      if word not in vocab:
        vocab[word] = len(vocab)
      tmp_line.append(vocab[word])
    dataset.append(np.array(tmp_line, dtype=np.int32))

  return dataset, vocab
