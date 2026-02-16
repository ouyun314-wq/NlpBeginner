import torch.nn as nn
import torch
import torch.nn.functional as F


class Language(nn.Module):
    def __init__(self, len_feature, len_words, len_hidden, num_to_word, word_to_num, strategy='lstm', pad_id=0,
                 start_id=1, end_id=2, drop_out=0.5):
        super(Language, self).__init__()
        self.pad_id = pad_id
        self.start_id = start_id
        self.end_id = end_id
        self.num_to_word = num_to_word
        self.word_to_num = word_to_num
        self.len_feature = len_feature
        self.len_words = len_words
        self.len_hidden = len_hidden
        self.dropout = nn.Dropout(drop_out)
        _x = nn.init.xavier_normal_(torch.Tensor(len_words, len_feature))
        self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=_x)
        if strategy == 'lstm':
            self.gate = nn.LSTM(input_size=len_feature, hidden_size=len_hidden, batch_first=True)
        elif strategy == 'gru':
            self.gate = nn.GRU(input_size=len_feature, hidden_size=len_hidden, batch_first=True)
        else:
            raise Exception("Unknown Strategy!")
        self.fc = nn.Linear(len_hidden, len_words)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        self.gate.flatten_parameters()
        x, _ = self.gate(x)
        logits = self.fc(x)

        return logits

    def _sample_token(self, logits, temperature):
        """根据 logits 采样一个 token，temperature 控制随机性"""
        if temperature <= 0:
            return logits.topk(1)[1][0].item()
        probs = F.softmax(logits.squeeze() / temperature, dim=-1)
        return torch.multinomial(probs, 1).item()

    def generate_random_poem(self, max_len, num_sentence, random=False, temperature=0.8):
        device = next(self.parameters()).device
        if random:
            initialize = torch.randn
        else:
            initialize = torch.zeros
        hn = initialize((1, 1, self.len_hidden)).to(device)
        cn = initialize((1, 1, self.len_hidden)).to(device)
        x = torch.LongTensor([self.start_id]).to(device)
        poem = list()

        while (len(poem) != num_sentence):
            word = x
            sentence = list()
            for j in range(max_len):
                word = torch.LongTensor([word]).to(device)
                word = self.embedding(word).view(1, 1, -1)
                output, (hn, cn) = self.gate(word, (hn, cn))
                output = self.fc(output)
                word = self._sample_token(output, temperature)

                if word == self.end_id:
                    x = torch.LongTensor([self.start_id]).to(device)
                    break
                sentence.append(self.num_to_word[word])
                if self.word_to_num['。'] == word:
                    break
            else:
                x = self.word_to_num['。']
            if sentence:
                poem.append(sentence)

        return poem

    def continue_poem(self, first_line, num_sentence=4, temperature=0.8):
        """输入第一句诗，续写剩余诗句"""
        device = next(self.parameters()).device
        initialize = torch.zeros
        hn = initialize((1, 1, self.len_hidden)).to(device)
        cn = initialize((1, 1, self.len_hidden)).to(device)

        # 用 <begin> 开头，喂入第一句的每个字符以积累隐状态
        word = torch.LongTensor([self.start_id]).to(device)
        word = self.embedding(word).view(1, 1, -1)
        _, (hn, cn) = self.gate(word, (hn, cn))

        for ch in first_line:
            if ch not in self.word_to_num:
                continue
            word_id = self.word_to_num[ch]
            word = torch.LongTensor([word_id]).to(device)
            word = self.embedding(word).view(1, 1, -1)
            output, (hn, cn) = self.gate(word, (hn, cn))

        # 判断第一句包含几个句号，计算还需生成几句
        sentence_count = first_line.count('。')
        remaining = num_sentence - max(sentence_count, 1)

        # 从第一句最后一个字符的输出开始续写
        output = self.fc(output)
        word = self._sample_token(output, temperature)

        poem = [list(first_line)]
        current_sentence = []

        generated = 0
        max_chars = 50
        char_count = 0

        while generated < remaining and char_count < max_chars * remaining:
            char_count += 1
            if word == self.end_id:
                break
            current_sentence.append(self.num_to_word[word])

            if self.word_to_num['。'] == word:
                poem.append(current_sentence)
                current_sentence = []
                generated += 1
                if generated >= remaining:
                    break

            word_tensor = torch.LongTensor([word]).to(device)
            word_tensor = self.embedding(word_tensor).view(1, 1, -1)
            output, (hn, cn) = self.gate(word_tensor, (hn, cn))
            output = self.fc(output)
            word = self._sample_token(output, temperature)

        if current_sentence and generated < remaining:
            poem.append(current_sentence)

        return poem
