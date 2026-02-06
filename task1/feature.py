import random

import numpy

from constant import DESC_LOC, REAL_TYPE_LOC, TYPE_NUM


def data_split(data_list, test_rate):
    train_list = list()
    test_list = list()
    i = 0
    for line in data_list:
        i += 1
        if random.random() > test_rate:
            train_list.append(line)
        else:
            test_list.append(line)
    return train_list, test_list


class Bag():
    def __init__(self, data, test_rate, max_item, gram):
        data = data[:max_item]
        train, test = data_split(data, test_rate)
        train_y = [int(line[REAL_TYPE_LOC]) for line in train]
        test_y = [int(line[REAL_TYPE_LOC]) for line in test]
        self.gram = gram
        self.word_map = dict()
        self.get_dict(data)
        # shape(x)=[feature,data]
        self.train_x = self.get_vec(train)
        self.test_x = self.get_vec(test)
        # shape(y)=[type,data]
        self.train_y = self.get_one_hot(train_y, TYPE_NUM)
        self.test_y = self.get_one_hot(test_y, TYPE_NUM)
        self.train_type = train_y
        self.test_type = test_y

    def get_dict(self, data):
        for line in data:
            words = line[DESC_LOC].upper().split()
            for i in range(len(words)):
                for j in range(self.gram):
                    if i + j + 1 > len(words):
                        break
                    word = words[i:i + j + 1]
                    word = "_".join(word)
                    if word not in self.word_map:
                        cnt = len(self.word_map)
                        self.word_map[word] = cnt

    def get_vec(self, data):
        n = len(data)
        d = len(self.word_map)
        x = numpy.zeros((d, n))
        for loc, line in enumerate(data):
            words = line[DESC_LOC].upper().split()
            for i in range(len(words)):
                for j in range(self.gram):
                    if i + j + 1 > len(words):
                        break
                    word = words[i:i + j + 1]
                    word = "_".join(word)
                    x[self.word_map[word], loc] = 1
        return x

    def get_one_hot(self, labels, num_classes):
        n = len(labels)
        y = numpy.zeros((num_classes, n))
        y[labels, numpy.arange(n)] = 1
        return y
