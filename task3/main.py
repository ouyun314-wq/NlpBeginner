"""
Task3: 基于ESIM模型的句子蕴含关系判断实验
使用SNLI数据集 + GloVe预训练词向量
实时绘图 + 时间统计
"""

import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from constant import seed, len_feature, len_hidden, test_rate, learning_rate, epoch, batch_size
from feature import GlobalEmbedding, get_batch
from esim import ESIM

# 设置随机种子
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


class ExperimentTracker:
    """实验跟踪器，记录训练过程中的各种指标"""
    def __init__(self, model_name):
        self.model_name = model_name
        self.train_losses = []
        self.train_accs = []
        self.test_accs = []
        self.epochs = []
        self.epoch_times = []
        self.total_time = 0

    def add_epoch(self, ep, train_loss, train_acc, test_acc, epoch_time):
        self.epochs.append(ep)
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.test_accs.append(test_acc)
        self.epoch_times.append(epoch_time)


class RealtimePlotter:
    """实时绘图类，动态显示训练过程"""
    def __init__(self):
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        plt.ion()
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
        self.fig.suptitle('ESIM 句子蕴含关系判断实验 - 实时监控', fontsize=16, fontweight='bold')
        self.tracker = None
        self.setup_axes()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

    def setup_axes(self):
        self.axes[0, 0].set_title('训练损失', fontweight='bold')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].grid(True, alpha=0.3)

        self.axes[0, 1].set_title('训练准确率', fontweight='bold')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('Accuracy (%)')
        self.axes[0, 1].grid(True, alpha=0.3)

        self.axes[0, 2].set_title('测试准确率', fontweight='bold')
        self.axes[0, 2].set_xlabel('Epoch')
        self.axes[0, 2].set_ylabel('Accuracy (%)')
        self.axes[0, 2].grid(True, alpha=0.3)

        self.axes[1, 0].set_title('每Epoch训练时间', fontweight='bold')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('时间 (秒)')
        self.axes[1, 0].grid(True, alpha=0.3)

        self.axes[1, 1].set_title('累计训练时间', fontweight='bold')
        self.axes[1, 1].set_xlabel('Epoch')
        self.axes[1, 1].set_ylabel('累计时间 (秒)')
        self.axes[1, 1].grid(True, alpha=0.3)

        self.axes[1, 2].set_title('训练总览', fontweight='bold')
        self.axes[1, 2].grid(True, alpha=0.3, axis='y')

    def set_tracker(self, tracker):
        self.tracker = tracker

    def update_plot(self):
        if self.tracker is None or not self.tracker.epochs:
            return

        t = self.tracker
        for ax in self.axes.flat:
            ax.clear()
        self.setup_axes()

        # 训练损失
        self.axes[0, 0].plot(t.epochs, t.train_losses, 'b-o', linewidth=2, markersize=4, label='Train Loss')
        self.axes[0, 0].legend()

        # 训练准确率
        self.axes[0, 1].plot(t.epochs, t.train_accs, 'g-s', linewidth=2, markersize=4, label='Train Acc')
        self.axes[0, 1].legend()

        # 测试准确率
        self.axes[0, 2].plot(t.epochs, t.test_accs, 'r-^', linewidth=2, markersize=4, label='Test Acc')
        self.axes[0, 2].legend()

        # 每epoch时间（柱状图）
        self.axes[1, 0].bar(t.epochs, t.epoch_times, alpha=0.7, color='tab:blue')

        # 累计时间（折线图）
        cumulative = np.cumsum(t.epoch_times)
        self.axes[1, 1].plot(t.epochs, cumulative, 'r-*', linewidth=2, markersize=6)

        # 训练总览（文字信息）
        ax_info = self.axes[1, 2]
        ax_info.axis('off')
        total = sum(t.epoch_times)
        avg = total / len(t.epoch_times)
        best_test = max(t.test_accs)
        best_ep = t.test_accs.index(best_test) + 1
        info = (
            f"已完成: {len(t.epochs)} / {epoch} Epochs\n\n"
            f"当前训练损失: {t.train_losses[-1]:.4f}\n"
            f"当前训练准确率: {t.train_accs[-1]:.2f}%\n"
            f"当前测试准确率: {t.test_accs[-1]:.2f}%\n\n"
            f"最佳测试准确率: {best_test:.2f}% (Epoch {best_ep})\n\n"
            f"累计时间: {total:.1f}s\n"
            f"平均每Epoch: {avg:.2f}s"
        )
        ax_info.text(0.1, 0.5, info, transform=ax_info.transAxes,
                     fontsize=13, verticalalignment='center',
                     fontfamily='SimHei',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save_final_plot(self, filename):
        self.update_plot()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n实验图表已保存至: {filename}")
        plt.ioff()


def train_and_evaluate(model, train_loader, test_loader, epochs, lr, tracker, plotter, device):
    """训练并评估模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n{'='*60}")
    print(f"开始训练 ESIM 模型")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr} | Hidden: {len_hidden}")
    print(f"设备: {device}")
    print(f"{'='*60}")

    start_time = time.time()

    for ep in range(1, epochs + 1):
        epoch_start = time.time()

        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for s1, s2, labels in train_loader:
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(s1, s2)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # 测试阶段
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for s1, s2, labels in test_loader:
                labels = labels.to(device)
                output = model(s1, s2)
                _, predicted = torch.max(output.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_acc = 100.0 * test_correct / test_total
        epoch_time = time.time() - epoch_start

        # 记录指标
        tracker.add_epoch(ep, avg_loss, train_acc, test_acc, epoch_time)

        print(f"Epoch [{ep:3d}/{epochs}] | "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s")

        # 实时更新图表
        plotter.update_plot()

    tracker.total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"总训练时间: {tracker.total_time:.2f}秒")
    print(f"平均每Epoch时间: {tracker.total_time / epochs:.2f}秒")
    print(f"最终训练准确率: {tracker.train_accs[-1]:.2f}%")
    print(f"最终测试准确率: {tracker.test_accs[-1]:.2f}%")
    print(f"最佳测试准确率: {max(tracker.test_accs):.2f}% (Epoch {tracker.test_accs.index(max(tracker.test_accs)) + 1})")
    print(f"{'='*60}")


def save_results_to_csv(tracker, filename):
    """保存实验数据到CSV"""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Train Acc (%)', 'Test Acc (%)', 'Epoch Time (s)', 'Cumulative Time (s)'])

        cumulative = 0
        for i in range(len(tracker.epochs)):
            cumulative += tracker.epoch_times[i]
            writer.writerow([
                tracker.epochs[i],
                f'{tracker.train_losses[i]:.4f}',
                f'{tracker.train_accs[i]:.2f}',
                f'{tracker.test_accs[i]:.2f}',
                f'{tracker.epoch_times[i]:.2f}',
                f'{cumulative:.2f}'
            ])

        writer.writerow([])
        writer.writerow(['总结'])
        writer.writerow(['总训练时间(秒)', f'{tracker.total_time:.2f}'])
        writer.writerow(['平均每Epoch时间(秒)', f'{tracker.total_time / len(tracker.epochs):.2f}'])
        writer.writerow(['最终测试准确率(%)', f'{tracker.test_accs[-1]:.2f}'])
        writer.writerow(['最佳测试准确率(%)', f'{max(tracker.test_accs):.2f}'])

    print(f"实验数据已保存至: {filename}")


if __name__ == '__main__':
    # ==================== 数据加载 ====================
    print("正在加载GloVe词向量...")
    with open('..\\data\\glove.6B\\glove.6B.50d.txt', 'rb') as f:
        lines = f.readlines()
    print(f"GloVe加载完成，共 {len(lines)} 个词向量")

    print("正在加载SNLI数据集...")
    with open('..\\data\\snli_1.0\\snli_1.0_train.txt', 'r') as f:
        temp = f.readlines()

    data = temp[1:]
    print(f"SNLI数据加载完成，共 {len(data)} 条样本")

    # ==================== 特征构建 ====================
    print("\n正在构建词典和Embedding矩阵...")
    ge = GlobalEmbedding(data, lines)
    ge.get_id()
    print(f"词典大小: {len(ge.wordDict)}")
    print(f"训练集: {len(ge.train_matrix_s1)} 条 | 测试集: {len(ge.test_matrix_s1)} 条")
    print(f"Embedding矩阵: {ge.embedding.shape}")

    # ==================== 构建DataLoader ====================
    train_loader = get_batch(ge.train_matrix_s1, ge.train_matrix_s2, ge.train_y, batch_size)
    test_loader = get_batch(ge.test_matrix_s1, ge.test_matrix_s2, ge.test_y, batch_size)

    # ==================== 创建模型 ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    model = ESIM(
        len_feature=len_feature,
        len_hidden=len_hidden,
        len_words=len(ge.wordDict) + 1,
        type_num=4,
        weight=ge.embedding
    )
    print(f"ESIM模型创建完成，参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ==================== 训练与评估 ====================
    tracker = ExperimentTracker("ESIM")
    plotter = RealtimePlotter()
    plotter.set_tracker(tracker)

    train_and_evaluate(model, train_loader, test_loader, epoch, learning_rate, tracker, plotter, device)

    # ==================== 保存结果 ====================
    os.makedirs('result', exist_ok=True)
    plotter.save_final_plot('result/esim_result.png')
    save_results_to_csv(tracker, 'result/esim_results.csv')

    # 保持图表显示
    plt.show()
