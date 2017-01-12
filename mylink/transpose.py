import numpy as np


class Transpose(object):
    def transpose_sequnce(self, dataset):
        return [self._transpose_sequnce(dataset, i)
                for i in range(len(dataset[0]))]

    def _transpose_sequnce(self, dataset, col):
        point_rc = lambda x: lambda y: dataset[y][x:x+1]
        point_r = point_rc(col)

        for i in range(len(dataset)):
            if i == 0:
                _seq = point_r(i)
            else:
                _seq = np.hstack([_seq, point_r(i)])
        return _seq
