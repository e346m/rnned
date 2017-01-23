# -*- coding: utf-8 -*-
import argparse
from nltk.translate import bleu_score
from ipdb import set_trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypothesis', '-hyp', help='hypothesis document')
    parser.add_argument('--references', '-ref', help='reference document')
    args = parser.parse_args()
    with open(args.hypothesis, "r") as hyps:
        list_of_hyps = [hyp for hyp in hyps]
    with open(args.references, "r") as refs:
        list_of_refs = [ref for ref in refs]
    cc = bleu_score.SmoothingFunction()
    score = bleu_score.corpus_bleu(list_of_refs, list_of_hyps,
                                   smoothing_function=cc.method4)
    print(score)

if __name__ == '__main__':
    main()
