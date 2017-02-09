from nltk import tokenize
import nltk
import six.moves.cPickle as pickle
import argparse
import pandas as pd
import re
import itertools
import glob
from ipdb import set_trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i',
                        help='Input Dir')
    parser.add_argument('--output', '-o',
                        help='Output Dir')
    args = parser.parse_args()

    articles = glob.glob("%s/*.csv" % args.input)

    for article in articles:
        filename = article.rsplit("/", 1)[-1]
        ex0 = []
        df = pd.read_csv(article, header=None)
        for row in df[0]:
            ex0.extend(tokenize.sent_tokenize(row))
        vn = [re.sub(r'\'+', '"', re.sub(r'`+', '"', " ".join(tokenize.word_tokenize(line)))) for line in ex0]

        ex1 = []
        for row in df[1]:
            results = re.split(re.compile(r"([。！][」）？]?[….―]*[」）]?)"), row)
            if results[-1] != "":
                results.append("")
            ex1.extend([results[i] + results[i + 1] for i in range(0, len(results) - 1, 2)])

        pD = pd.DataFrame([[v, j] for v, j in itertools.zip_longest(vn, ex1)])
        pD.to_csv("%s/%s" % (args.output, filename), header=None, index=False)

if __name__ == '__main__':
    main()
