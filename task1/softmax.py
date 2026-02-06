import numpy.random
from numpy import argmax

from constant import  max_item, TYPE_NUM,  SHUFFLE


class SoftMax:
    def __init__(self, feature):
        self.feature = feature
        self.W = numpy.random.randn(TYPE_NUM, self.feature)
        self.seq_idx = 0  # 顺序遍历的位置指针

    def softmax_bat(self, o):
        # shape(o)=[type,batch]
        o = o - numpy.max(o, axis=0, keepdims=True)
        o = numpy.exp(o)
        return o / numpy.sum(o, axis=0, keepdims=True)

    def regression(self, train_x, train_y, strategy, learning_rate, batch_size):
        n_samples = train_x.shape[1]

        # batch_size <= 0 表示全量
        if batch_size <= 0:
            batch_size = n_samples

        if strategy == SHUFFLE:
            # 随机采样 batch_size 个样本
            idx = numpy.random.randint(0, n_samples, size=batch_size)
            x = train_x[:, idx]
            y = train_y[:, idx]
        else:
            # 顺序遍历，每次只取一个 batch
            l = self.seq_idx
            r = min(l + batch_size, n_samples)
            x = train_x[:, l:r]
            y = train_y[:, l:r]
            # 更新位置指针，到末尾则重置
            self.seq_idx = r if r < n_samples else 0

        o = self.W.dot(x)
        y_hat = self.softmax_bat(o)
        actual_batch = x.shape[1]
        grad = (y_hat - y).dot(x.T) / actual_batch * learning_rate
        self.W -= grad

    def predict(self, x):
        y = self.W.dot(x)
        return argmax(y, axis=0)

    def correct_rate(self, x, labels):
        ans = self.predict(x)
        correct = sum([labels[i] == ans[i] for i in range(len(labels))]) / len(labels)
        return correct
