import torch
from torch import nn
from torch.cuda import device
import torch.nn.functional as F

from constant import TYPE_NUM, feature_len
class CNN2(nn.Module):
    def __init__(self, len_feature, len_words, weight, typenum=5, drop_out=0.5):
        super(CNN2, self).__init__()
        self.len_feature = len_feature  # d的大小
        self.len_words = len_words  # 单词数目
        self.dropout = nn.Dropout(drop_out)  # Dropout层
        self.randEmbedding = nn.Embedding(
            num_embeddings=len_words + 1,  # +1 是为了pad_idx
            embedding_dim=len_feature,
            padding_idx=0  # 通常pad token的索引是0
        )
        nn.init.uniform_(self.randEmbedding.weight[1:], -0.1, 0.1)  # 只初始化非pad部分
        self.randEmbedding.weight.data[0].fill_(0)  # pad token显式设为0
        # 移到GPU
        self.randEmbedding = self.randEmbedding.cuda()

        self.globEmbedding = nn.Embedding.from_pretrained(weight, freeze=True).cuda()
        # Conv2d参数详解：（输入通道数：1，输出通道数：l_l，卷积核大小：（行数，列数））
        # padding是指往句子两侧加 0，因为有的句子只有一个单词
        # 那么 X 就是 1*50 对 W=2*50 的卷积核根本无法进行卷积操作
        # 因此要在X两侧行加0（两侧列不加），（padding=（1，0））变成 3*50
        # 又比如 padding=（2，0）变成 5*50

        self.conv1 = nn.Sequential(nn.Conv2d(2, 16, (2, len_feature), padding=(1, 0)),
                                   nn.ReLU()).cuda()  # 第1个卷积核+激活层
        self.conv2 = nn.Sequential(nn.Conv2d(2, 16, (3, len_feature), padding=(1, 0)),
                                   nn.ReLU()).cuda()  # 第2个卷积核+激活层
        self.conv3 = nn.Sequential(nn.Conv2d(2, 16, (4, len_feature), padding=(2, 0)),
                                   nn.ReLU()).cuda()  # 第3个卷积核+激活层
        self.conv4 = nn.Sequential(nn.Conv2d(2, 16, (5, len_feature), padding=(2, 0)),
                                   nn.ReLU()).cuda()  # 第4个卷积核+激活层
        # 全连接层
        self.fc = nn.Linear(4 * 16, typenum).cuda()
        # 冗余的softmax层，可以不加
        # self.act = nn.Softmax(dim=1)

    def forward(self, x):
        # batchsize*sentencelen*d
        x = torch.LongTensor(x).cuda()
        randE = self.randEmbedding(x)  # 词嵌入
        globE = self.globEmbedding(x)
        tmp = torch.stack([randE, globE], 1)
        out_put = self.dropout(tmp)  # dropout层

        # batchsize*16*senlen-2
        conv1 = self.conv1(out_put).squeeze(3)  # 第1个卷积

        conv2 = self.conv2(out_put).squeeze(3)  # 第2个卷积

        conv3 = self.conv3(out_put).squeeze(3)  # 第3个卷积

        conv4 = self.conv4(out_put).squeeze(3)  # 第4个卷积
        pool1 = F.max_pool1d(conv1, conv1.shape[2])
        pool2 = F.max_pool1d(conv2, conv2.shape[2])
        pool3 = F.max_pool1d(conv3, conv3.shape[2])
        pool4 = F.max_pool1d(conv4, conv4.shape[2])

        pool = torch.cat([pool1, pool2, pool3, pool4], 1).squeeze(2)  # 拼接起来

        out_put = self.fc(pool)  # 全连接层

        return out_put

