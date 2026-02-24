"""
CNN情感分类对比实验
比较两种词嵌入方式：
1. Random Embedding (随机初始化)
2. Pre-trained Embedding (预训练词向量，如GloVe)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用交互式后端
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
from threading import Thread
import queue

from cnn1 import CNN1
from feature import RandomEmbedding, GlobalEmbedding, get_batch
from constant import seed, test_rate, max_item, DESC_LOC, feature_len, TYPE_NUM

# 设置随机种子
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
        self.train_time = 0
        self.total_time = 0

    def add_epoch(self, epoch, train_loss, train_acc, test_acc):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.test_accs.append(test_acc)


class RealtimePlotter:
    """实时绘图类，用于动态显示训练过程"""
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('CNN情感分类对比实验 - 实时监控', fontsize=16, fontweight='bold')

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        self.trackers = {}
        self.colors = {'Random Embedding': 'blue', 'Pre-trained Embedding': 'red'}
        self.update_queue = queue.Queue()

    def setup_axes(self):
        """设置图表"""
        # 训练损失
        self.axes[0, 0].set_title('训练损失对比', fontweight='bold')
        self.axes[0, 0].set_xlabel('Epoch')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].grid(True, alpha=0.3)

        # 训练准确率
        self.axes[0, 1].set_title('训练准确率对比', fontweight='bold')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('Accuracy (%)')
        self.axes[0, 1].grid(True, alpha=0.3)

        # 测试准确率
        self.axes[1, 0].set_title('测试准确率对比', fontweight='bold')
        self.axes[1, 0].set_xlabel('Epoch')
        self.axes[1, 0].set_ylabel('Accuracy (%)')
        self.axes[1, 0].grid(True, alpha=0.3)

        # 训练时间对比
        self.axes[1, 1].set_title('训练时间对比', fontweight='bold')
        self.axes[1, 1].set_ylabel('时间 (秒)')
        self.axes[1, 1].grid(True, alpha=0.3, axis='y')

    def add_tracker(self, tracker):
        """添加实验跟踪器"""
        self.trackers[tracker.model_name] = tracker

    def update_plot(self):
        """手动更新图表"""
        # 清空所有子图
        for ax in self.axes.flat:
            ax.clear()

        self.setup_axes()

        if not self.trackers:
            return

        # 绘制训练损失
        for name, tracker in self.trackers.items():
            if tracker.epochs:
                self.axes[0, 0].plot(tracker.epochs, tracker.train_losses,
                                    label=name, color=self.colors.get(name, 'black'),
                                    marker='o', linewidth=2, markersize=4)
        if any(tracker.epochs for tracker in self.trackers.values()):
            self.axes[0, 0].legend()

        # 绘制训练准确率
        for name, tracker in self.trackers.items():
            if tracker.epochs:
                self.axes[0, 1].plot(tracker.epochs, tracker.train_accs,
                                    label=name, color=self.colors.get(name, 'black'),
                                    marker='s', linewidth=2, markersize=4)
        if any(tracker.epochs for tracker in self.trackers.values()):
            self.axes[0, 1].legend()

        # 绘制测试准确率
        for name, tracker in self.trackers.items():
            if tracker.epochs:
                self.axes[1, 0].plot(tracker.epochs, tracker.test_accs,
                                    label=name, color=self.colors.get(name, 'black'),
                                    marker='^', linewidth=2, markersize=4)
        if any(tracker.epochs for tracker in self.trackers.values()):
            self.axes[1, 0].legend()

        # 绘制训练时间对比（柱状图）
        names = list(self.trackers.keys())
        times = [self.trackers[name].total_time for name in names]
        if any(t > 0 for t in times):
            bars = self.axes[1, 1].bar(names, times,
                                       color=[self.colors.get(name, 'gray') for name in names],
                                       alpha=0.7, edgecolor='black', linewidth=1.5)
            # 在柱状图上显示数值
            for bar, t in zip(bars, times):
                height = bar.get_height()
                if height > 0:
                    self.axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                                        f'{t:.2f}s',
                                        ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show(self):
        """显示图表窗口"""
        plt.ion()  # 开启交互模式
        self.setup_axes()
        plt.show(block=False)
        plt.pause(0.1)  # 确保窗口显示

    def save_final_plot(self, filename='comparison_result.png'):
        """保存最终图表"""
        self.update_plot()  # 最后更新一次
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n最终对比图已保存至: {filename}")
        plt.ioff()  # 关闭交互模式


def train_model(model, train_loader, test_loader, epochs, tracker, device, plotter=None):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\n{'='*60}")
    print(f"开始训练: {tracker.model_name}")
    print(f"{'='*60}")

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            target = target.to(device)  # 只转换target

            optimizer.zero_grad()
            output = model(data)  # data在forward中会自动转换
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # 测试阶段
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for test_data, test_target in test_loader:
                test_target = test_target.to(device)

                outputs = model(test_data)
                _, predicted = torch.max(outputs.data, 1)

                test_total += test_target.size(0)
                test_correct += (predicted == test_target).sum().item()

        test_accuracy = 100 * test_correct / test_total

        epoch_time = time.time() - epoch_start
        tracker.train_time += epoch_time

        # 记录结果
        tracker.add_epoch(epoch + 1, avg_train_loss, train_accuracy, test_accuracy)

        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {train_accuracy:.2f}% | "
              f"Test Acc: {test_accuracy:.2f}% | "
              f"Time: {epoch_time:.2f}s")

        # 实时更新图表
        if plotter is not None:
            plotter.update_plot()

    tracker.total_time = time.time() - start_time
    print(f"\n{tracker.model_name} 训练完成!")
    print(f"总训练时间: {tracker.total_time:.2f}秒")
    print(f"最终测试准确率: {tracker.test_accs[-1]:.2f}%")


def load_data():
    """加载数据"""
    print("正在加载数据...")
    with open('..\\data\\sentiment-analysis-on-movie-reviews\\train.tsv') as f:
        tsvreader = csv.reader(f, delimiter='\t')
        temp = list(tsvreader)
    data = temp[1:]
    data.sort(key=lambda x: len(x[DESC_LOC].split()))
    print(f"数据加载完成，共 {len(data)} 条记录")
    return data


def run_experiment(epochs=10, batch_size=64):
    """运行对比实验"""
    print("\n" + "="*60)
    print("CNN情感分类对比实验")
    print("="*60)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    data = load_data()

    # 准备两种词嵌入方式的数据
    print("\n准备 Random Embedding 数据...")
    random_emb = RandomEmbedding(data, test_rate, max_item)

    print("准备 Pre-trained Embedding 数据...")
    # 注意：这里需要加载预训练词向量文件，如果没有则使用随机初始化
    use_pretrained = False
    glove_paths = [
        '..\\data\\glove.6B\\glove.6B.50d.txt',
    ]

    for path in glove_paths:
        try:
            print(f"  尝试加载预训练词向量: {path}")
            with open(path, 'rb') as f:
                lines = f.readlines()
            pretrained_emb = GlobalEmbedding(data, test_rate, max_item, lines)
            use_pretrained = True
            print(f"  ✓ 成功加载预训练词向量，共 {len(lines)} 个词")
            break
        except FileNotFoundError:
            print(f"  ✗ 文件不存在")
            continue
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            continue

    if not use_pretrained:
        print("\n" + "="*60)
        print("警告: 未找到预训练词向量文件!")
        print("="*60)
        print("将使用随机初始化作为第二个模型（对比意义不大）")
        print("\n如需使用真正的预训练词向量，请下载GloVe:")
        print("1. 访问: https://nlp.stanford.edu/projects/glove/")
        print("2. 下载 glove.6B.zip")
        print("3. 解压并放到以下任一路径:")
        for path in glove_paths:
            print(f"   - {path}")
        print("="*60 + "\n")
        pretrained_emb = RandomEmbedding(data, test_rate, max_item)

    # 创建数据加载器
    random_train_loader = get_batch(random_emb.train_matrix, random_emb.train_y, batch_size)
    random_test_loader = get_batch(random_emb.test_matrix, random_emb.test_y, batch_size)

    pretrained_train_loader = get_batch(pretrained_emb.train_matrix, pretrained_emb.train_y, batch_size)
    pretrained_test_loader = get_batch(pretrained_emb.test_matrix, pretrained_emb.test_y, batch_size)

    # 创建模型
    print("\n创建模型...")
    model_random = CNN1(
        len_feature=feature_len,
        len_words=len(random_emb.wordDict),
        embedding=None  # 随机初始化
    )

    if use_pretrained:
        model_pretrained = CNN1(
            len_feature=feature_len,
            len_words=len(pretrained_emb.wordDict),
            embedding=pretrained_emb.embedding  # 使用预训练embedding
        )
    else:
        model_pretrained = CNN1(
            len_feature=feature_len,
            len_words=len(pretrained_emb.wordDict),
            embedding=None
        )

    # 创建实验跟踪器
    tracker_random = ExperimentTracker("Random Embedding")
    tracker_pretrained = ExperimentTracker("Pre-trained Embedding")

    # 创建实时绘图器
    plotter = RealtimePlotter()
    plotter.add_tracker(tracker_random)
    plotter.add_tracker(tracker_pretrained)
    plotter.show()  # 显示图表窗口

    # 训练两个模型（串行训练以便观察）
    print(f"\n开始训练 (Epochs: {epochs}, Batch Size: {batch_size})")

    # 训练Random Embedding模型
    train_model(model_random, random_train_loader, random_test_loader,
                epochs, tracker_random, device, plotter)

    # 训练Pre-trained Embedding模型
    train_model(model_pretrained, pretrained_train_loader, pretrained_test_loader,
                epochs, tracker_pretrained, device, plotter)

    # 打印最终对比结果
    print("\n" + "="*60)
    print("实验结果对比")
    print("="*60)
    print(f"{'模型':<30} {'训练时间(s)':<15} {'最终测试准确率(%)':<20}")
    print("-"*60)
    print(f"{tracker_random.model_name:<30} {tracker_random.total_time:<15.2f} {tracker_random.test_accs[-1]:<20.2f}")
    print(f"{tracker_pretrained.model_name:<30} {tracker_pretrained.total_time:<15.2f} {tracker_pretrained.test_accs[-1]:<20.2f}")
    print("="*60)

    # 保存最终图表
    plotter.save_final_plot('glo_rand_cnn_comp.png')

    # 保存详细数据到CSV
    save_results_to_csv(tracker_random, tracker_pretrained)

    # 保持图表显示
    plt.show()

    return tracker_random, tracker_pretrained


def save_results_to_csv(tracker1, tracker2):
    """保存实验结果到CSV文件"""
    filename = 'glo_rand_cnn_comp.csv'

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 写入表头
        writer.writerow(['Epoch',
                        f'{tracker1.model_name} - Train Loss',
                        f'{tracker1.model_name} - Train Acc',
                        f'{tracker1.model_name} - Test Acc',
                        f'{tracker2.model_name} - Train Loss',
                        f'{tracker2.model_name} - Train Acc',
                        f'{tracker2.model_name} - Test Acc'])

        # 写入每个epoch的数据
        max_epochs = max(len(tracker1.epochs), len(tracker2.epochs))
        for i in range(max_epochs):
            row = [i + 1]

            if i < len(tracker1.epochs):
                row.extend([tracker1.train_losses[i],
                           tracker1.train_accs[i],
                           tracker1.test_accs[i]])
            else:
                row.extend(['', '', ''])

            if i < len(tracker2.epochs):
                row.extend([tracker2.train_losses[i],
                           tracker2.train_accs[i],
                           tracker2.test_accs[i]])
            else:
                row.extend(['', '', ''])

            writer.writerow(row)

        # 写入总结信息
        writer.writerow([])
        writer.writerow(['总结'])
        writer.writerow(['模型', '总训练时间(秒)', '最终测试准确率(%)'])
        writer.writerow([tracker1.model_name, f'{tracker1.total_time:.2f}', f'{tracker1.test_accs[-1]:.2f}'])
        writer.writerow([tracker2.model_name, f'{tracker2.total_time:.2f}', f'{tracker2.test_accs[-1]:.2f}'])

    print(f"\n实验数据已保存至: {filename}")


if __name__ == '__main__':
    # 运行实验
    # 可以调整epochs和batch_size
    run_experiment(epochs=100, batch_size=64)
