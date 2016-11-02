import numpy as np
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter
from ipdb import set_trace

class Classifier(link.Chain):

    compute_accuracy = True

    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy):
        super(Classifier, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None


class DecClassifier(Classifier):
    def __call__(self, *args):

        assert len(args) >= 2
        t = args[0]
        middle = args[1]

        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(t, middle)
        self.loss = self.lossfun(self.y, t) # compare y' and y
        reporter.report({'loss': self.loss}, t)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss

class EncClassifier(Classifier):
    def __call__(self, args):

        x = args

        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(x)
