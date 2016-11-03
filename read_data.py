#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
import six.moves.cPickle as pickle
from itertools import chain
from collections import Counter

from ipdb import set_trace
#TODO UKNの扱いは1回しか出てきてない単語かそれとも上位~単語以外の単語をuknにするか..
UNK = "unk"

class Load(object):
  def __init__(self, dump_label):
    self.dump_label = dump_label

  def normalize_load_data(self, filename):
    fs = np.array(pd.read_table(filename))
    self.r_info([line[0].lower().strip().split() for line in fs])

  def load_data(self, filename):
    fs = np.array(pd.read_table(filename))
    self.r_info([line[0].strip().split() for line in fs])

  def r_info(self, lines):
    vocab = {}
    dataset = []
    flat_lines = chain.from_iterable(lines)
    count = Counter(flat_lines)
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
    with open("./input/%s.sentence" %self.dump_label, "wb") as f:
      pickle.dump(dataset, f)
    with open("./input/%s.vocab" %self.dump_label, "wb") as f:
      pickle.dump(vocab, f)

class SourceLoader(Load):
  def __init__(self, dump_label="source"):
    super(SourceLoader, self).__init__(dump_label)


class TargetLoader(Load):
  def __init__(self, dump_label="target"):
    super(TargetLoader, self).__init__(dump_label)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--source', '-s', help='Source file path')
  parser.add_argument('--target', '-t', help='Target file path')
  args = parser.parse_args()

  SourceLoader().normalize_load_data(args.source)
  TargetLoader().normalize_load_data(args.target)

if __name__ == '__main__':
  main()
