#!/usr/local/var/pyenv/shims/python
# -*- coding: utf-8 -*-
import argparse
from nltk.translate import bleu_score
from ipdb import set_trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypothesis', '-hy', help='hypothesis document')
    parser.add_argument('--references', '-ref', help='reference document')
    args = parser.parse_args()
    each_score = []
    with open(args.hypothesis, "r") as hyps:
        with open(args.references, "r") as refs:
            each_score = [bleu_score.sentence_bleu([ref], hyp)
                          for hyp, ref in zip(hyps, refs)]
    set_trace()
    print(sum(each_score)/len(args.hypothesis))

if __name__ == '__main__':
    main()
