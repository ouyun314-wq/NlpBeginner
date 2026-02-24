import csv
import random
import re
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
""" 定义PyTorch模型类Input_Encoding，将输入的文本序列进行编码 """


class Input_Encoding(nn.Module):
    def __init__(self, len_feature, len_hidden, len_words, weight=None, layer=1, batch_first=True, drop_out=0.5):
        super(Input_Encoding, self).__init__()
        # 输入的特征维度
        self.len_feature = len_feature
        # LSTM 的隐藏层大小
        self.len_hidden = len_hidden
        # 词嵌入的单词数
        self.len_words = len_words
        #  LSTM 的层数
        self.layer = layer
        # dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(drop_out)
        # 如果 weight 是 None，则使用 xavier_normal_ 函数初始化一个形状为 (len_words, len_feature) 的张量 x，并使用这个张量作为词嵌入层的权重
        if weight is None:
            x = nn.init.xavier_normal_(torch.Tensor(len_words, len_feature))
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=x).cuda()
        # 否则，使用给定的权重 weight 初始化词嵌入层
        else:
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=weight).cuda()
        # 初始化双向 LSTM
        self.lstm = nn.LSTM(input_size=len_feature, hidden_size=len_hidden, num_layers=layer, batch_first=batch_first,
                            bidirectional=True).cuda()

    # 定义前向传播函数，用于对输入的句子进行编码
    def forward(self, x):
        # 将输入的数据转换为 torch.LongTensor 类型并放到 GPU 上
        x = torch.LongTensor(x).cuda()
        # 使用词嵌入层对输入进行词嵌入
        x = self.embedding(x)
        # 使用 dropout 对词嵌入结果进行正则化
        x = self.dropout(x)
        # 使用 flatten_parameters() 函数将 LSTM 的参数展开
        self.lstm.flatten_parameters()
        # 将结果输入到 LSTM 中进行编码
        x, _ = self.lstm(x)
        return x


""" 局部推理模块 """
"""基于注意力机制的编码器-解码器框架：注意力机制通常被用来计算输入序列中每个位置对于输出序列的重要程度，进而加强对于相关信息的关注和利用。
本代码中，a_bar和b_bar分别表示输入序列a和b的编码表示，通过计算矩阵e，可以得到a_bar和b_bar之间的交互关系。
接着，通过softmax函数将e中的数值归一化，得到a_bar和b_bar在交互关系下的注意力分布a_tilde和b_tilde，分别用来加权计算b_bar和a_bar的信息。
最后，将a_bar、a_tilde、b_bar、b_tilde之间的差异和乘积信息进行拼接，形成新的表示m_a和m_b。
进一步用于后续的分类或回归任务。"""


class Local_Inference_Modeling(nn.Module):
    def __init__(self):
        super(Local_Inference_Modeling, self).__init__()
        # 将输入值转换为概率分布
        self.softmax_1 = nn.Softmax(dim=1).cuda()
        self.softmax_2 = nn.Softmax(dim=2).cuda()

    # 这个模型类的前向函数实现了两个句子之间的局部推理模型
    def forward(self, a_bar, b_bar):
        # e 是注意力矩阵。matmul()函数计算两个输入张量a_bar和b_bar的矩阵乘积
        e = torch.matmul(a_bar, b_bar.transpose(1, 2)).cuda()
        # 将 e 矩阵的第二个维度进行 softmax 操作，得到一个概率分布矩阵
        a_tilde = self.softmax_2(e)
        # 将 a_tilde 与 b_bar 做矩阵乘法，得到一个新的矩阵
        a_tilde = a_tilde.bmm(b_bar)
        b_tilde = self.softmax_1(e)
        b_tilde = b_tilde.transpose(1, 2).bmm(a_bar)
        # 这四个矩阵在最后一个维度上进行拼接，得到一个新的矩阵
        m_a = torch.cat([a_bar, a_tilde, a_bar - a_tilde, a_bar * a_tilde], dim=-1)
        m_b = torch.cat([b_bar, b_tilde, b_bar - b_tilde, b_bar * b_tilde], dim=-1)
        return m_a, m_b


""" 组合推理模块 （对蕴含假设和前提进行编码和组合） """


class Inference_Composition(nn.Module):
    # len_feature 特征向量的维度；len_hidden_m 在Local_Inference_Modeling中得到的组合向量的维度
    def __init__(self, len_feature, len_hidden_m, len_hidden, layer=1, batch_first=True, drop_out=0.5):
        # 调用父类构造函数以初始化该模块
        super(Inference_Composition, self).__init__()
        # 定义线性变换层，将Local_Inference_Modeling得到的组合向量降维为len_feature维
        self.linear = nn.Linear(len_hidden_m, len_feature).cuda()
        # 定义LSTM层，其中包含了两个方向的隐状态，并将降维后的组合向量作为输入；bidirectional=True表示为双向LSTM
        self.lstm = nn.LSTM(input_size=len_feature, hidden_size=len_hidden, num_layers=layer, batch_first=batch_first,
                            bidirectional=True).cuda()
        self.dropout = nn.Dropout(drop_out).cuda()

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        # 将LSTM层的权重参数展开以提高效率
        self.lstm.flatten_parameters()
        # 将降维后的组合向量作为LSTM层的输入；输出为x
        x, _ = self.lstm(x)
        return x


""" 预测类 """


class Prediction(nn.Module):
    # len_v 输入的向量的长度；len_mid 表示中间层的大小
    def __init__(self, len_v, len_mid, type_num=4, drop_out=0.5):
        super(Prediction, self).__init__()
        # 定义 mlp 多层感知机
        # nn.Dropout(drop_out)：dropout 层，防止过拟合。
        # nn.Linear(len_v, len_mid)：全连接层，将输入的向量映射到中间层。
        # nn.Tanh()：激活函数，使用双曲正切函数进行非线性变换。
        # nn.Linear(len_mid, type_num)：全连接层，将中间层的特征映射到预测的类别数。
        self.mlp = nn.Sequential(nn.Dropout(drop_out), nn.Linear(len_v, len_mid), nn.Tanh(),
                                 nn.Linear(len_mid, type_num)).cuda()

    # 定义前向传播方法
    def forward(self, a, b):
        # 计算 m_a 在第二个维度上的平均值
        v_a_avg = a.sum(1) / a.shape[1]
        # 计算 m_a 在第二个维度上的最大值
        v_a_max = a.max(1)[0]
        # 计算 m_b 在第二个维度上的平均值
        v_b_avg = b.sum(1) / b.shape[1]
        # # 计算 m_b 在第二个维度上的最大值
        v_b_max = b.max(1)[0]
        # 将四个向量连接在一起形成一个新的向量
        out_put = torch.cat((v_a_avg, v_a_max, v_b_avg, v_b_max), dim=-1)
        # 将新的向量输入到多层感知机中进行预测，返回预测结果
        return self.mlp(out_put)


""" ESIM模型 """
"""在 ESIM 模型中，首先将输入的句子 a 和 b 通过 Input_Encoding 模块编码成相应的表示，
然后将编码后的 a 和 b 作为输入传给 Local_Inference_Modeling 模块进行局部推理，
接着将得到的结果传递给 Inference_Composition 模块进行推理融合，
最后通过 Prediction 模块预测两个句子是否具有某种关系"""


class ESIM(nn.Module):
    def __init__(self, len_feature, len_hidden, len_words, type_num=4, weight=None, layer=1, batch_first=True,
                 drop_out=0.5):
        super(ESIM, self).__init__()
        self.len_words = len_words
        # 创建input_encoding对象，是Input_Encoding类的一个实例，Input_Encoding类是用来对输入进行编码的
        # layer表示ESIM模型的层数
        self.input_encoding = Input_Encoding(len_feature, len_hidden, len_words, weight=weight, layer=layer,
                                             batch_first=batch_first, drop_out=drop_out)
        # 创建对象，用来对两个句子进行本地推理的，也就是计算两个句子中每个词语之间的关联度
        self.local_inference_modeling = Local_Inference_Modeling()
        # Inference Composition层中的输入是由四个部分拼接而成的，分别是a_bar, a_tilde, a_bar - a_tilde和 a_bar * a_tilde。
        # 每个部分的特征向量维度都是len_hidden，所以总共是4 * len_hidden，且在拼接之前需要将a_bar和b_bar的维度都扩展为4 * len_hidden。
        # 因此，输入到Inference Composition层中的向量维度是8 * len_hidden。
        self.inference_composition = Inference_Composition(len_feature, 8 * len_hidden, len_hidden, layer=layer,
                                                           batch_first=batch_first, drop_out=drop_out)
        self.prediction = Prediction(8 * len_hidden, len_hidden, type_num=type_num, drop_out=drop_out)

    def forward(self, a, b):
        a_bar = self.input_encoding(a)
        b_bar = self.input_encoding(b)

        m_a, m_b = self.local_inference_modeling(a_bar, b_bar)

        v_a = self.inference_composition(m_a)
        v_b = self.inference_composition(m_b)

        out_put = self.prediction(v_a, v_b)

        return out_put