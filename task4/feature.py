import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from constant import len_feature, batch_size


def pre_process(data):
    sentences = list()
    tags = list()
    sentence = list()
    tag = list()
    for line in data:
        if line == '\n':
            if sentence:
                sentences.append(sentence)
                tags.append(tag)
                sentence = list()
                tag = list()
        else:
            elements = line.split()
            if elements[0] == '-DOCSTART-':
                continue
            sentence.append(elements[0].upper())
            tag.append(elements[-1])
    if sentence:
        sentences.append(sentence)
        tags.append(tag)

    return list(zip(sentences, tags))


class GlobalEmbedding:
    def __init__(self, data, lines, train_zip, test_zip):
        self.trained_dict = dict()
        self.get_dict(lines)
        self.train_x, self.train_y = zip(*train_zip)
        self.test_x, self.test_y = zip(*test_zip)
        self.train_x_matrix = list()
        self.test_x_matrix = list()
        self.train_y_matrix = list()
        self.test_y_matrix = list()
        self.tag_dict = {'<pad>': 0, '<begin>': 1, '<end>': 2}
        self.embedding = torch.zeros(1, len_feature)
        self.dict_words = dict()
        self.get_id()
        self.train = get_batch(self.train_x_matrix, self.train_y_matrix, batch_size)
        self.test = get_batch(self.test_x_matrix, self.test_y_matrix, batch_size)

    def get_dict(self, lines):
        n = len(lines)
        for i in range(n):
            line = lines[i].split()
            self.trained_dict[line[0].decode("utf-8").upper()] = torch.tensor(
                [float(line[j]) for j in range(1, len_feature + 1)]).reshape(1, len_feature)

    def get_id(self):
        for term in self.train_x:
            for word in term:  # Process every word
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words) + 1
                    if word in self.trained_dict:
                        self.embedding = torch.cat((self.embedding, self.trained_dict[word]), dim=0)
                    else:
                        # print(word)
                        # raise Exception("words not found!")
                        self.embedding = torch.cat((self.embedding, torch.zeros(1, len_feature)), dim=0)
        for term in self.test_x:
            for word in term:  # Process every word
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words) + 1
                    if word in self.trained_dict:
                        self.embedding = torch.cat((self.embedding, torch.zeros(1, len_feature)), dim=0)
                    else:
                        # print(word)
                        # raise Exception("words not found!")
                        self.embedding = torch.cat((self.embedding, torch.zeros(1, len_feature)), dim=0)
        for tags in self.train_y:
            for tag in tags:
                if tag not in self.tag_dict:
                    self.tag_dict[tag] = len(self.tag_dict)
            for tags in self.test_y:
                for tag in tags:
                    if tag not in self.tag_dict:
                        self.tag_dict[tag] = len(self.tag_dict)
        for term in self.train_x:
            item = [self.dict_words[word] for word in term]
            self.train_x_matrix.append(item)
        for term in self.test_x:
            item = [self.dict_words[word] for word in term]
            self.test_x_matrix.append(item)
        for tags in self.train_y:
            item = [self.tag_dict[tag] for tag in tags]
            self.train_y_matrix.append(item)
        for tags in self.test_y:
            item = [self.tag_dict[tag] for tag in tags]
            self.test_y_matrix.append(item)


class ClsDataset(Dataset):
    def __init__(self, sentence, tag):
        self.sentence = sentence
        self.tag = tag

    def __getitem__(self, item):
        return self.sentence[item], self.tag[item]

    def __len__(self):
        return len(self.tag)


def collate_fn(batch_data):
    sentence, tag = zip(*batch_data)
    sentences = [torch.LongTensor(sent) for sent in sentence]
    padded_sents = pad_sequence(sentences, batch_first=True, padding_value=0)
    tags = [torch.LongTensor(t) for t in tag]
    padded_tags = pad_sequence(tags, batch_first=True, padding_value=0)
    return torch.LongTensor(padded_sents), torch.LongTensor(padded_tags)


def get_batch(x, y, batch_size):
    dataset = ClsDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)
    return dataloader
