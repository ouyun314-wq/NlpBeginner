import torch
from torch import nn


class CRF(nn.Module):

    def __init__(self, type_num, pad_id, start_id, end_id):
        super(CRF, self).__init__()
        self.type_num = type_num
        self.pad_id = pad_id
        self.start_id = start_id
        self.end_id = end_id

        transition = torch.zeros(type_num, type_num)
        transition[:, start_id] = -10000.0
        transition[end_id, :] = -10000.0
        transition[:, pad_id] = -10000.0
        transition[pad_id, :] = -10000.0
        transition[pad_id, pad_id] = 0.0
        transition[pad_id, :end_id] = 0.0

        self.transition = nn.Parameter(transition).cuda()

    def forward(self, scores, tags, mask):
        true_prob = self.true_prob(scores, tags, mask)
        total_prob = self.total_prob(scores, mask)
        return -torch.sum(true_prob - total_prob)

    def true_prob(self, scores, tags, mask):
        batch_size, sequence_len = tags.shape
        true_prob = torch.zeros(batch_size).cuda()

        first_tag = tags[:, 0]
        last_tag_index = mask.sum(1) - 1
        last_tag = torch.gather(tags, 1, last_tag_index.unsqueeze(1)).squeeze(1)

        tran_score = self.transition[self.start_id, first_tag]
        tag_score = torch.gather(scores[:, 0], 1, first_tag.unsqueeze(1)).squeeze(1)

        true_prob += tran_score + tag_score

        for i in range(1, sequence_len):
            non_pad = mask[:, i]
            pre_tag = tags[:, i - 1]
            curr_tag = tags[:, i]

            tran_score = self.transition[pre_tag, curr_tag]
            tag_score = torch.gather(scores[:, i], 1, curr_tag.unsqueeze(1)).squeeze(1)

            true_prob += tran_score * non_pad + tag_score * non_pad

        true_prob += self.transition[last_tag, self.end_id]

        return true_prob

    def total_prob(self, scores, mask):
        batch_size, sequence_len, num_tags = scores.shape
        log_sum_exp_prob = self.transition[self.start_id, :].unsqueeze(0) + scores[:, 0]
        for i in range(1, sequence_len):
            every_log_sum_exp_prob = list()
            for j in range(num_tags):
                # 1 k
                tran_score = self.transition[:, j].unsqueeze(0)
                # b 1
                tag_score = scores[:, i, j].unsqueeze(1)
                # b k
                prob = tran_score + tag_score + log_sum_exp_prob
                # k b
                every_log_sum_exp_prob.append(torch.logsumexp(prob, dim=1))
            # b k
            new_prob = torch.stack(every_log_sum_exp_prob).t()
            # b l   b 1
            non_pad = mask[:, i].unsqueeze(-1)
            log_sum_exp_prob = non_pad * new_prob + (1 - non_pad) * log_sum_exp_prob

        tran_score = self.transition[:, self.end_id].unsqueeze(0)
        return torch.logsumexp(log_sum_exp_prob + tran_score, dim=1)

    def predict(self, scores, mask):
        batch_size, sequence_len, num_tags = scores.shape

        # 1. 初始化：START -> 第一个时间步
        total_prob = self.transition[self.start_id, :].unsqueeze(0) + scores[:, 0]
        # total_prob: (batch_size, num_tags)

        # 2. 存储每一步的回溯指针
        backpointers = []

        # 3. 前向递推
        for i in range(1, sequence_len):
            # total_prob[:, :, None] -> (batch, num_tags, 1)
            # 对每个目标标签 j，计算从所有前驱标签转移过来的得分
            # prob: (batch, num_tags_prev, num_tags_next)
            prob = total_prob.unsqueeze(2) + self.transition.unsqueeze(0) + scores[:, i].unsqueeze(1)

            # 沿前驱标签维度取 max
            max_prob, max_tag = prob.max(dim=1)  # (batch, num_tags)

            # mask 处理：padding 位置保持不变
            non_pad = mask[:, i].unsqueeze(-1)  # (batch, 1)
            total_prob = non_pad * max_prob + (1 - non_pad) * total_prob

            backpointers.append(max_tag)  # (batch, num_tags)

        # 4. 加上到 END 的转移分数，找全局最优结束标签
        total_prob = total_prob + self.transition[:, self.end_id].unsqueeze(0)
        _, best_last_tag = total_prob.max(dim=1)  # (batch,)

        # 5. 反向回溯，还原完整路径
        best_path = [best_last_tag]
        for bp in reversed(backpointers):
            # bp: (batch, num_tags)
            # best_last_tag: (batch,)
            # 从回溯指针中取出每个样本对应的前驱标签
            best_last_tag = bp.gather(1, best_last_tag.unsqueeze(1)).squeeze(1)
            best_path.append(best_last_tag)

        # 反转得到正序路径
        best_path.reverse()
        best_path = torch.stack(best_path, dim=1)  # (batch, sequence_len)

        # 6. 把 padding 位置替换为 pad_id
        best_path = best_path * mask.long() + (1 - mask.long()) * self.pad_id

        return best_path


class NamedEntityRecognition(nn.Module):

    def __init__(self, len_feature, len_words, len_hidden, type_num, pad_id, start_id, end_id, weight=None,
                 drop_out=0.5):
        super(NamedEntityRecognition, self).__init__()
        self.len_feature = len_feature
        self.len_words = len_words
        self.len_hidden = len_hidden
        self.dropout = nn.Dropout(drop_out)
        if weight is None:
            x = nn.init.xavier_normal_(torch.Tensor(len_words, len_feature))
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=x).cuda()
        else:
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=weight).cuda()
        self.lstm = nn.LSTM(input_size=len_feature, hidden_size=len_hidden, batch_first=True, bidirectional=True).cuda()
        self.fc = nn.Linear(2 * len_hidden, type_num).cuda()
        self.crf = CRF(type_num, pad_id, start_id, end_id).cuda()

    def forward(self, x, tags, mask):
        mask = mask.int()
        x = self.embedding(x)
        x = self.dropout(x)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        scores = self.fc(x)
        loss = self.crf(scores, tags, mask)
        return loss

    def predict(self, x, mask):
        mask = mask.int()
        x = self.embedding(x)
        x = self.dropout(x)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        scores = self.fc(x)

        return self.crf.predict(scores, mask)


