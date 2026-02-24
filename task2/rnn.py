import torch
from torch import nn
from torch.cuda import device
import torch.nn.functional as F

from constant import TYPE_NUM, feature_len
class MY_RNN(nn.Module):
    def __init__(self, len_feature, len_hidden, len_words, weight, typenum=5, layer=1, nonlinearity='tanh',
                 batch_first=True, drop_out=0.5):
        super(MY_RNN, self).__init__()
        self.len_feature = len_feature  # d的大小
        self.len_hidden = len_hidden  # l_h的大小
        self.len_words = len_words  # 单词的个数（包括padding）
        self.layer = layer  # 隐藏层层数
        self.dropout = nn.Dropout(drop_out)  # dropout层

        #         x = nn.init.xavier_normal_(torch.Tensor(len_words, len_feature))
        #         self.randEmbedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=x).cuda()

        self.globEmbedding = nn.Embedding.from_pretrained(weight, freeze=False).cuda()
        # 用nn.Module的内置函数定义隐藏层
        self.rnn = nn.RNN(input_size=len_feature, hidden_size=len_hidden, num_layers=layer, nonlinearity=nonlinearity,
                          batch_first=batch_first, dropout=drop_out).cuda()
        # 全连接层
        self.fc = nn.Linear(len_hidden, typenum).cuda()
        # 冗余的softmax层，可以不加
        # self.act = nn.Softmax(dim=1)

    def forward(self, x):
        """x:数据，维度为[batch_size， 句子长度]"""
        x = torch.LongTensor(x).cuda()
        batch_size = x.size(0)
        """经过词嵌入后，维度为[batch_size，句子长度，d]"""
        out_put = self.globEmbedding(x)  # 词嵌入
        out_put = self.dropout(out_put)  # dropout层
        # 另一种初始化h_0的方式
        # h0 = torch.randn(self.layer, batch_size, self.len_hidden).cuda()
        # 初始化h_0为0向量
        h0 = torch.autograd.Variable(torch.zeros(self.layer, batch_size, self.len_hidden)).cuda()
        """dropout后不变，经过隐藏层后，维度为[1，batch_size, l_h]"""
        _, hn = self.rnn(out_put, h0)  # 隐藏层计算
        """经过全连接层后，维度为[1，batch_size, 5]"""
        out_put = self.fc(hn).squeeze(0)  # 全连接层
        """挤掉第0维度，返回[batch_size, 5]的数据"""
        # out_put = self.act(out_put)  # 冗余的softmax层，可以不加
        return out_put
