import torch
from torch import nn
from torch.cuda import device
import torch.nn.functional as F

from constant import TYPE_NUM, feature_len


class CNN1(nn.Module):
    def __init__(self, len_feature, len_words, len_kernel=50, typenum=TYPE_NUM, embedding=None, drop_out=0.5):
        super(CNN1, self).__init__()
        self.len_feature = len_feature
        self.len_words = len_words
        self.len_kernel = len_kernel
        self.dropout = nn.Dropout(drop_out)
        if embedding is None:
            # 创建embedding层
            self.embedding = nn.Embedding(
                num_embeddings=len_words + 1,  # +1 是为了pad_idx
                embedding_dim=len_feature,
                padding_idx=0  # 通常pad token的索引是0
            )
            # 初始化权重
            # pad token (索引0) 保持全零
            nn.init.uniform_(self.embedding.weight[1:], -0.1, 0.1)  # 只初始化非pad部分
            self.embedding.weight.data[0].fill_(0)  # pad token显式设为0
            # 移到GPU
            self.embedding = self.embedding.cuda()
        else:
            self.embedding = nn.Embedding(num_embeddings=len_words+1, embedding_dim=len_feature, _weight=embedding).cuda()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, (2, len_feature), padding=(1, 0)), nn.ReLU()).cuda()
        self.conv2 = nn.Sequential(nn.Conv2d(1, 16, (3, len_feature), padding=(1, 0)), nn.ReLU()).cuda()
        self.conv3 = nn.Sequential(nn.Conv2d(1, 16, (4, len_feature), padding=(2, 0)), nn.ReLU()).cuda()
        self.conv4 = nn.Sequential(nn.Conv2d(1, 16, (5, len_feature), padding=(2, 0)), nn.ReLU()).cuda()
        # Fully connected layer
        self.fc = nn.Linear(4 * 16, typenum).cuda()
        # An extra softmax layer may be redundant
        # self.act = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.LongTensor(x).cuda()
        out_put = self.embedding(x).view(x.shape[0], 1, x.shape[1], self.len_feature)
        #shape(output)=[batch_size,1,len_sentence,len_feature]
        out_put=self.dropout(out_put)
        #shape(conv1)=[batch_size,16,len_sentence+2*pad-k+1]
        conv1 = self.conv1(out_put).squeeze(3)
        #shape(pool1)=[batch_size,16,1]
        pool1 = F.max_pool1d(conv1, conv1.shape[2])

        conv2 = self.conv2(out_put).squeeze(3)
        pool2 = F.max_pool1d(conv2, conv2.shape[2])

        conv3 = self.conv3(out_put).squeeze(3)
        pool3 = F.max_pool1d(conv3, conv3.shape[2])

        conv4 = self.conv4(out_put).squeeze(3)
        pool4 = F.max_pool1d(conv4, conv4.shape[2])

        pool = torch.cat([pool1, pool2, pool3, pool4], 1).squeeze(2)
        out_put = self.fc(pool)
        # out_put = self.act(out_put)
        return out_put
