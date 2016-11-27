#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import MeCab
from collections import Counter
from itertools import chain
from ipdb import set_trace
import time
import pandas as pd
UNK = "unk"

def en_load_data(filename):
  start = time.clock()
  print ("import file\n")
  fs = np.array(pd.read_table(filename))
  print ("done: ", time.clock() - start, "\n")
  return r_info([line[0].lower().strip().split() for line in fs])


def ja_load_data(filename):

  start = time.clock()
  print ("import file\n")
  fs = np.array(pd.read_table(filename))
  print ("done: ", time.clock() - start, "\n")

  return r_info([line[0].strip().split() for line in fs])

def r_info(lines):
  vocab = {}
  dataset = []
  flat_lines = chain.from_iterable(lines)
  count = Counter(flat_lines)
  #UNKS = [k for k, v in count.items() if v == 1]
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
