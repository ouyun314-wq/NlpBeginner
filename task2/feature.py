import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from constant import DESC_LOC, REAL_TYPE_LOC, TYPE_NUM, feature_len


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


class RandomEmbedding:
    def __init__(self, data, test_rate, max_item):
        data = data[:max_item]
        train, test = data_split(data, test_rate)
        self.train_y = [int(line[REAL_TYPE_LOC]) for line in train]
        self.test_y = [int(line[REAL_TYPE_LOC]) for line in test]
        self.wordDict = dict()
        self.train_matrix = list()
        self.test_matrix = list()
        self.get_dict(train)
        self.get_id(train, test)

    def get_dict(self, data):
        for i in range(len(data)):
            words = data[i][DESC_LOC].upper().split()
            for word in words:
                if word not in self.wordDict:
                    self.wordDict[word] = len(self.wordDict)+1

    def get_id(self, train_list, test_list):
        for line in train_list:
            words = line[DESC_LOC].upper().split()
            item = [self.wordDict.get(word, 0) for word in words]
            self.train_matrix.append(item)
        for line in test_list:
            words = line[DESC_LOC].upper().split()
            item = [self.wordDict.get(word, 0) for word in words]
            self.test_matrix.append(item)

class GlobalEmbedding:
    def __init__(self, data, test_rate, max_item,lines):
        data = data[:max_item]
        train, test = data_split(data, test_rate)
        self.train_y = [int(line[REAL_TYPE_LOC]) for line in train]
        self.test_y = [int(line[REAL_TYPE_LOC]) for line in test]
        self.trained_dict = dict()
        self.get_trained(lines)
        self.wordDict = dict()
        self.embedding=torch.zeros(1, feature_len)
        self.get_dict(train)
        self.train_matrix = list()
        self.test_matrix = list()
        self.get_id(train, test)

    def get_trained(self,lines):
        n = len(lines)
        for i in range(n):
            line = lines[i].split()
            self.trained_dict[line[0].decode("utf-8").upper()] = torch.tensor(
                [float(line[j]) for j in range(1, feature_len+1)]).reshape(1, feature_len)

    def get_dict(self, data):
        for i in range(len(data)):
            words = data[i][DESC_LOC].upper().split()
            for word in words:
                if word not in self.wordDict:
                    self.wordDict[word] = len(self.wordDict) + 1
                    if word not in self.trained_dict:
                        self.embedding = torch.cat([self.embedding, torch.zeros(1, feature_len)], dim=0)
                    else:
                        self.embedding = torch.cat([self.embedding, self.trained_dict[word]], dim=0)

    def get_id(self, train_list, test_list):
        for line in train_list:
            words = line[DESC_LOC].upper().split()
            item = [self.wordDict.get(word, 0) for word in words]
            self.train_matrix.append(item)
        for line in test_list:
            words = line[DESC_LOC].upper().split()
            item = [self.wordDict.get(word, 0) for word in words]
            self.test_matrix.append(item)


class ClsDataset(Dataset):
    def __init__(self, sentence, emotion):
        self.sentence = sentence  # 句子
        self.emotion= emotion  # 情感类别

    def __getitem__(self, item):
        return self.sentence[item], self.emotion[item]

    def __len__(self):
        return len(self.emotion)

def collate_fn(batch_data):
    sentence, emotion = zip(*batch_data)
    sentences = [torch.LongTensor(sent) for sent in sentence]  # 把句子变成Longtensor类型
    padded_sents = pad_sequence(sentences, batch_first=True, padding_value=0)  # 自动padding操作！！！
    return torch.LongTensor(padded_sents), torch.LongTensor(emotion)



def get_batch(x,y,batch_size):
    dataset = ClsDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,drop_last=True,collate_fn=collate_fn)
    return dataloader
