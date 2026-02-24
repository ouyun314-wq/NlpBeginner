import torch
import torch.nn as nn
import time
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from feature import Word_Embedding, get_batch
from model import Language

# ============ 超参数 ============
BATCH_SIZE = 64
EPOCHS = 100
LEN_FEATURE = 128
LEN_HIDDEN = 256
LR = 1e-3
STRATEGY = 'lstm'
DROP_OUT = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============ 数据处理 ============
with open('..\\data\\poetryFromTang.txt', 'rb') as f:
    temp = f.readlines()

we = Word_Embedding(temp)
we.data_process()

dataloader = get_batch(we.matrix, BATCH_SIZE)
len_words = len(we.word_dict)

print(f"词表大小: {len_words}")
print(f"诗歌数量: {len(we.matrix)}")
print(f"设备: {DEVICE}")

# ============ 模型初始化 ============
model = Language(
    len_feature=LEN_FEATURE,
    len_words=len_words,
    len_hidden=LEN_HIDDEN,
    num_to_word=we.tag_dict,
    word_to_num=we.word_dict,
    strategy=STRATEGY,
    drop_out=DROP_OUT
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# ============ 实时画图设置 ============
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Task5 - Poetry Language Model Training')

train_loss_history = []
train_ppl_history = []
test_loss_history = []
test_ppl_history = []

def update_plot():
    ax1.clear()
    ax2.clear()

    epochs_x = list(range(1, len(train_loss_history) + 1))

    ax1.plot(epochs_x, train_loss_history, 'b-o', markersize=3, label='Train Loss')
    ax1.plot(epochs_x, test_loss_history, 'g-s', markersize=3, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Test Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_x, train_ppl_history, 'r-o', markersize=3, label='Train PPL')
    ax2.plot(epochs_x, test_ppl_history, 'm-s', markersize=3, label='Test PPL')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Perplexity')
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.01)

# ============ 测试集数据处理 ============
with open('..\\data\\poetryTest.txt', 'rb') as f:
    test_temp = f.readlines()

test_we = Word_Embedding(test_temp)
test_we.form_poem()
test_we.data.sort(key=lambda x: len(x))

# 用训练集词表编码测试集，未登录词映射为 <unk>
unk_id = we.word_dict['<unk>']
oov_count = 0
total_chars = 0
test_matrix = []
for poem in test_we.data:
    ids = []
    for word in poem:
        total_chars += 1
        if word in we.word_dict:
            ids.append(we.word_dict[word])
        else:
            ids.append(unk_id)
            oov_count += 1
    test_matrix.append(ids)

print(f"测试集: {len(test_matrix)} 首诗, {total_chars} 字符, "
      f"OOV {oov_count} 个 ({oov_count/total_chars*100:.2f}%)")

test_dataloader = get_batch(test_matrix, min(BATCH_SIZE, len(test_matrix)))


def evaluate(data_loader):
    """在 eval 模式下计算数据集的 loss 和困惑度"""
    model.eval()
    eval_loss = 0.0
    eval_tokens = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(DEVICE)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, len_words), targets.reshape(-1))
            mask = (targets != 0)
            num_tokens = mask.sum().item()
            eval_loss += loss.item() * num_tokens
            eval_tokens += num_tokens
    avg_loss = eval_loss / eval_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ============ 用于收集报告数据 ============
generated_poems = []  # [(epoch, poem_str), ...]

# ============ 训练 ============
print("=" * 50)
print("开始训练")
print("=" * 50)

total_start = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        batch = batch.to(DEVICE)
        # 输入: 去掉最后一个token; 目标: 去掉第一个token
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs)
        # logits: (batch, seq_len, vocab_size), targets: (batch, seq_len)
        loss = criterion(logits.reshape(-1, len_words), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # 统计非pad token数量用于困惑度计算
        mask = (targets != 0)
        num_tokens = mask.sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    epoch_time = time.time() - epoch_start

    # 训练集困惑度（eval模式）
    train_avg_loss, train_ppl = evaluate(dataloader)
    train_loss_history.append(train_avg_loss)
    train_ppl_history.append(train_ppl)

    # 测试集困惑度
    test_avg_loss, test_ppl = evaluate(test_dataloader)
    test_loss_history.append(test_avg_loss)
    test_ppl_history.append(test_ppl)

    print(f"Epoch [{epoch:3d}/{EPOCHS}] | "
          f"Train Loss: {train_avg_loss:.4f} PPL: {train_ppl:.2f} | "
          f"Test Loss: {test_avg_loss:.4f} PPL: {test_ppl:.2f} | "
          f"Time: {epoch_time:.2f}s")

    update_plot()

    # 每10个epoch生成一首诗展示效果
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            poem = model.generate_random_poem(max_len=50, num_sentence=4, random=False)
            poem_str = '\n'.join([''.join(s) for s in poem])
            generated_poems.append((epoch, poem_str))
            print(f"--- 生成诗歌 ---\n{poem_str}\n----------------")

total_time = time.time() - total_start
print("=" * 50)
print(f"训练完成 | 总耗时: {total_time:.2f}s | 平均每epoch: {total_time / EPOCHS:.2f}s")
print(f"训练集 Loss: {train_loss_history[-1]:.4f} | PPL: {train_ppl_history[-1]:.2f}")
print(f"测试集 Loss: {test_loss_history[-1]:.4f} | PPL: {test_ppl_history[-1]:.2f}")
print("=" * 50)

# ============ 保存模型 ============
checkpoint = {
    'model_state_dict': model.state_dict(),
    'word_dict': we.word_dict,
    'tag_dict': we.tag_dict,
    'len_feature': LEN_FEATURE,
    'len_hidden': LEN_HIDDEN,
    'strategy': STRATEGY,
    'drop_out': DROP_OUT,
}
torch.save(checkpoint, 'poetry_checkpoint.pth')
print("模型已保存至 poetry_checkpoint.pth")

# ============ 最终生成诗歌 ============
model.eval()
with torch.no_grad():
    # 随机生成
    poem = model.generate_random_poem(max_len=50, num_sentence=4, random=False)
    final_random_poem = '\n'.join([''.join(s) for s in poem])
    print(f"\n最终随机生成诗歌:\n{final_random_poem}")

    # 续写生成
    test_first_lines = ["床前明月光，", "白日依山尽，", "春眠不觉晓，"]
    continue_results = []
    for first_line in test_first_lines:
        poem = model.continue_poem(first_line, num_sentence=4)
        poem_str = '\n'.join([''.join(s) for s in poem])
        continue_results.append((first_line, poem_str))
        print(f"\n续写 [{first_line}]:\n{poem_str}")

# ============ 保存图表 ============
plt.ioff()
plt.savefig('training_curve.png', dpi=150)
print("训练曲线已保存至 training_curve.png")

# ============ 自动生成实验报告 ============
# 训练过程表格（每10个epoch采样一行）
epoch_table_rows = ""
for i in range(0, EPOCHS, 10):
    epoch_table_rows += (
        f"| {i+10} | {train_loss_history[i+9]:.4f} | {train_ppl_history[i+9]:.2f} "
        f"| {test_loss_history[i+9]:.4f} | {test_ppl_history[i+9]:.2f} |\n"
    )

# 训练过程中生成的诗歌
training_poems_section = ""
for ep, ps in generated_poems:
    training_poems_section += f"**Epoch {ep}：**\n\n```\n{ps}\n```\n\n"

# 续写结果
continue_section = ""
for fl, ps in continue_results:
    continue_section += f"**输入：** `{fl}`\n\n```\n{ps}\n```\n\n"

plt.show()
