# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd
import six.moves.cPickle as pickle
from itertools import chain
from collections import Counter

from ipdb import set_trace
UNK = "<unk>"


class Load(object):
    def __init__(self, dump_label):
        self.dump_label = dump_label

    def normalize_load_data(self, output, filename):
        #fs = np.array(pd.read_table(filename, header=None)) # pandas fails to read some lines about vn
        with open(filename, "r") as fs:
            self.r_info([line.lower().strip().split() for line in fs], output)

    def load_data(self, filename):
        fs = np.array(pd.read_table(filename, header=None))
        self.r_info([line[0].strip().split() for line in fs])

    def r_info(self, lines, output):
        vocab = {"<eos>": 0, "<unk>": 1}
        dataset = []
        flat_lines = chain.from_iterable(lines)
        all_word = Counter(flat_lines)
        print(self.dump_label)
        print("all_word: %s count" %  sum(all_word.values()))
        print("vocab: %s count" %  len(all_word))
        V_SIZE = round(len(all_word) * 0.93)
        count = [k for k, v in all_word.most_common(V_SIZE)]
        for line in lines:
            tmp_line = []
            for word in line:
                if word not in count:
                    word = UNK
                if word not in vocab:
                    vocab[word] = len(vocab)
                tmp_line.append(vocab[word])
            dataset.append(np.array(tmp_line, dtype=np.int32))
        print("%s:%s" % (self.dump_label, len(vocab)))

        with open("./%s/%s.sentence" % (output, self.dump_label), "wb") as f:
            pickle.dump(dataset, f)
        with open("./%s/%s.vocab" % (output, self.dump_label), "wb") as f:
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
    parser.add_argument('--output', '-o', required=True,
                        help='Target file path')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output, 0o755)
    SourceLoader().normalize_load_data(args.output, args.source)
    TargetLoader().normalize_load_data(args.output, args.target)

if __name__ == '__main__':
    main()
