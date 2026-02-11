import csv
import random
import re
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from constant import test_rate, len_feature, max_item


class GlobalEmbedding:
    def __init__(self, data, lines):
        self.data, self.train_list, self.test_list = self.data_split(data)
        self.pattern = '[A-Za-z|\']+'
        # 词序号对应字典
        self.wordDict = dict()
        # 初始化词嵌入权值矩阵 padding对应0
        self.embedding = torch.zeros(1, len_feature)
        self.get_dict(lines)
        self.type_dict = {'-': 0, 'contradiction': 1, 'entailment': 2, 'neutral': 3}
        self.train_matrix_s1 = list()
        self.train_matrix_s2 = list()
        self.train_y = list()
        self.test_matrix_s1 = list()
        self.test_matrix_s2 = list()
        self.test_y = list()

    def data_split(self, data):
        # 划分训练集，测试集
        train_list = list()
        test_list = list()
        i = 0
        _data = [item.split('\t') for item in data]
        data = [[item[5], item[6], item[0]] for item in _data]
        for line in data:
            i += 1
            if random.random() > test_rate:
                train_list.append(line)
            else:
                test_list.append(line)
            if (i > max_item):
                break
        return data, train_list, test_list

    def get_dict(self, lines):
        # 词嵌入
        trained_dict = dict()
        n = len(lines)
        for i in range(n):
            line = lines[i].split()
            trained_dict[line[0].decode("utf-8").upper()] = torch.tensor(
                [float(line[j]) for j in range(1, len_feature + 1)]).reshape(1, len_feature)

        for i in range(len(self.data)):
            for j in range(2):
                str = self.data[i][j].upper()
                words = re.findall(self.pattern, str)
                for word in words:
                    if (word not in self.wordDict):
                        self.wordDict[word] = len(self.wordDict) + 1
                        if (word not in trained_dict):
                            self.embedding = torch.cat((self.embedding, torch.zeros(1, len_feature)), dim=0)
                        else:
                            self.embedding = torch.cat((self.embedding, trained_dict[word]), dim=0)

    def get_id(self):
        self.train_y = [self.type_dict[term[2]] for term in self.train_list]
        self.test_y = [self.type_dict[term[2]] for term in self.test_list]

        for s in self.train_list:
            words = re.findall(self.pattern, s[0].upper())
            item = [self.wordDict[word] for word in words]
            self.train_matrix_s1.append(item)

            words = re.findall(self.pattern, s[1].upper())
            item = [self.wordDict[word] for word in words]
            self.train_matrix_s2.append(item)

        for s in self.test_list:
            words = re.findall(self.pattern, s[0].upper())
            item = [self.wordDict[word] for word in words]
            self.test_matrix_s1.append(item)

            words = re.findall(self.pattern, s[1].upper())
            item = [self.wordDict[word] for word in words]
            self.test_matrix_s2.append(item)


class ClsDataset(Dataset):  # 定义 ClsDataset 的类，继承了 Dataset 类
    def __init__(self, sentence1, sentence2, relation):
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.relation = relation  # 标签

    # 用于获取数据集中的一个样本，item 表示样本的索引，函数返回索引为 item 的句子和关系
    def __getitem__(self, item):
        return self.sentence1[item], self.sentence2[item], self.relation[item]

    def __len__(self):
        # 返回数据集中样本的数量
        return len(self.relation)


def collate_fn(batch_data):
    sents1, sents2, labels = zip(*batch_data)
    # 转换为张量
    sentences1 = [torch.LongTensor(sent) for sent in sents1]
    padded_sents1 = pad_sequence(sentences1, batch_first=True, padding_value=0)
    sentences2 = [torch.LongTensor(sent) for sent in sents2]
    padded_sents2 = pad_sequence(sentences2, batch_first=True, padding_value=0)
    return torch.LongTensor(padded_sents1), torch.LongTensor(padded_sents2), torch.LongTensor(labels)


def get_batch(x1, x2, y, _batch_size):
    dataset = ClsDataset(x1, x2, y)
    dataloader = DataLoader(dataset, batch_size=_batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)
    return dataloader
