import random
import time
import torch
from torch import optim
import matplotlib

import matplotlib.pyplot as plt

from constant import seed, len_feature, learning_rate, epoch, batch_size
from feature import pre_process, GlobalEmbedding
from ner import NamedEntityRecognition

random.seed(seed)
torch.manual_seed(seed)

matplotlib.use('TkAgg')

# ============ 数据加载 ============
print("Loading GloVe embeddings...")
with open('..\\data\\glove.6B\\glove.6B.50d.txt', 'rb') as f:
    lines = f.readlines()

print("Loading training data...")
with open('..\\data\\ner\\train.txt', 'r') as f:
    temp = f.readlines()
data = temp[2:]
train_zip = pre_process(data)

print("Loading test data...")
with open('..\\data\\ner\\test.txt', 'r') as f:
    temp = f.readlines()
data = temp[2:]
test_zip = pre_process(data)

# ============ 构建词表与嵌入 ============
print("Building vocabulary and embeddings...")
ge = GlobalEmbedding(data, lines, train_zip, test_zip)

len_words = ge.embedding.shape[0]
type_num = len(ge.tag_dict)
pad_id = ge.tag_dict['<pad>']
start_id = ge.tag_dict['<begin>']
end_id = ge.tag_dict['<end>']
weight = ge.embedding

print(f"Vocab size: {len_words}, Tag types: {type_num}")
print(f"Tag dict: {ge.tag_dict}")

# ============ 模型初始化 ============
model = NamedEntityRecognition(
    len_feature=len_feature,
    len_words=len_words,
    len_hidden=128,
    type_num=type_num,
    pad_id=pad_id,
    start_id=start_id,
    end_id=end_id,
    weight=weight,
    drop_out=0.5
).cuda()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ============ 评估函数 ============
def evaluate(model, dataloader, type_num):
    """计算 dataloader 上的 micro F1（跳过前3个特殊标签）"""
    model.eval()
    tp, fp, fn = [0] * type_num, [0] * type_num, [0] * type_num
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            mask = (x != 0).float()
            pred = model.predict(x, mask)
            seq_len = mask.int().sum(1)
            for i in range(pred.shape[0]):
                for j in range(seq_len[i]):
                    a = pred[i][j].int().item()
                    b = y[i][j].int().item()
                    if a == b:
                        tp[a] += 1
                    else:
                        fp[a] += 1
                        fn[b] += 1
    tps = sum(tp[3:])
    fps = sum(fp[3:])
    fns = sum(fn[3:])
    p = tps / (tps + fps) if (tps + fps) > 0 else 0
    r = tps / (tps + fns) if (tps + fns) > 0 else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

def update_plot(ax1, ax2, steps, losses, train_f1s, test_f1s):
    """实时更新图表"""
    ax1.clear()
    ax1.plot(steps, losses, 'b-o', markersize=3, label='Train Loss')
    ax1.set_xlabel('Step (epoch)')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.clear()
    ax2.plot(steps, train_f1s, 'g-o', markersize=3, label='Train F1')
    ax2.plot(steps, test_f1s, 'r-s', markersize=3, label='Test F1')
    ax2.set_xlabel('Step (epoch)')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score')
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.01)

# ============ 实时画图初始化 ============
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
plt.show(block=False)
fig.canvas.flush_events()

steps = []        # x 轴：epoch 进度（如 0, 0.2, 0.4, ... 1.0, 1.2, ...）
train_losses = []
train_f1s = []
test_f1s = []

# 每个 epoch 内评估的次数（即把 1 个 epoch 分成几段）
eval_per_epoch = 5

# ============ 未训练时的评估（epoch 0） ============
print("Evaluating untrained model...")
train_f1 = evaluate(model, ge.train, type_num)
test_f1 = evaluate(model, ge.test, type_num)
steps.append(0.0)
train_losses.append(0.0)
train_f1s.append(train_f1)
test_f1s.append(test_f1)
print(f"Epoch 0 (untrained) | Train F1: {train_f1:.4f} | Test F1: {test_f1:.4f}")
update_plot(ax1, ax2, steps, train_losses, train_f1s, test_f1s)

# ============ 训练循环 ============
total_start_time = time.time()
epoch_times = []

# 预先统计每个 epoch 的 batch 数
total_batches = sum(1 for _ in ge.train)
eval_interval = max(1, total_batches // eval_per_epoch)  # 每隔多少 batch 评估一次

for ep in range(1, epoch + 1):
    epoch_start_time = time.time()

    model.train()
    running_loss = 0
    batch_in_loss = 0

    for batch_idx, (x, y) in enumerate(ge.train, 1):
        x, y = x.cuda(), y.cuda()
        mask = (x != 0).float()
        loss = model(x, y, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        batch_in_loss += 1

        # 在 epoch 内部按间隔评估
        if batch_idx % eval_interval == 0 or batch_idx == total_batches:
            avg_loss = running_loss / batch_in_loss
            frac = (ep - 1) + batch_idx / total_batches
            train_f1 = evaluate(model, ge.train, type_num)
            test_f1 = evaluate(model, ge.test, type_num)

            steps.append(round(frac, 2))
            train_losses.append(avg_loss)
            train_f1s.append(train_f1)
            test_f1s.append(test_f1)

            elapsed = time.time() - total_start_time
            print(f"Epoch {frac:.1f}/{epoch} | Loss: {avg_loss:.4f} | "
                  f"Train F1: {train_f1:.4f} | Test F1: {test_f1:.4f} | "
                  f"Elapsed: {elapsed:.1f}s")

            update_plot(ax1, ax2, steps, train_losses, train_f1s, test_f1s)

            # 重置 running loss
            running_loss = 0
            batch_in_loss = 0
            model.train()

    epoch_time = time.time() - epoch_start_time
    epoch_times.append(epoch_time)

# ============ 训练结束统计 ============
total_time = time.time() - total_start_time
avg_epoch_time = sum(epoch_times) / len(epoch_times)

print("\n" + "=" * 50)
print("Training Complete!")
print(f"Total training time: {total_time:.1f}s ({total_time / 60:.1f} min)")
print(f"Average epoch time:  {avg_epoch_time:.1f}s")
print(f"Best Train F1: {max(train_f1s):.4f} (Step {steps[train_f1s.index(max(train_f1s))]})")
print(f"Best Test F1:  {max(test_f1s):.4f} (Step {steps[test_f1s.index(max(test_f1s))]})")
print(f"Final Loss:    {train_losses[-1]:.4f}")
print("=" * 50)

plt.ioff()
plt.savefig('training_result.png', dpi=150)
plt.show()
