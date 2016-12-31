from nltk.translate import bleu_score
from ipdb import set_trace


bleu_score.corpus_bleu()

print bleu_score.sentence_bleu([reference1, reference2, reference3], candidate1, weights)
print bleu_score.sentence_bleu([reference1, reference2, reference3], candidate2, weights)
