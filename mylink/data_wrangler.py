import numpy as np


class DataWrangler(object):
    def reverse_source_seq_without_last_word(self):
        self._s = [np.concatenate((__s[:-1][::-1], __s[-1:]))
                   for __s in self._s]

    def sort_alignment_key_target(self):
        self.concatinate_sort()
        self.separate()

    def concatinate_sort(self):
        dataset = [[__s, __t] for __s, __t in zip(self._s, self._t)]
        dataset.sort(key=lambda x: len(x[1]))
        dataset.reverse()
        self.dataset = dataset

    def separate(self):
        t = []
        s = []
        for data in self.dataset:
            t.append(data.pop())
            s.append(data[0])
        self._s = s
        self._t = t

    def largest_size(self):
        return len(max(self._s, key=lambda x: len(x)))

    def filling_ingnore_label(self):
        ls = self.largest_size()
        ret = []
        for _s in self._s:
            diff = ls - len(_s)
            if diff is not 0:
                balance = np.empty(diff, np.int32)
                balance.fill(-1)
                _s = np.hstack((_s, balance))
            ret.append(_s)
        self._s = ret

    @property
    def _s(self):
        return self.__s

    @property
    def _t(self):
        return self.__t

    @_s.setter
    def _s(self, source_sentence):
        self.__s = source_sentence

    @_t.setter
    def _t(self, target_sentence):
        self.__t = target_sentence
