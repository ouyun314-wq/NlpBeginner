
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from constant import SHUFFLE, SEQUENTIAL
from softmax import SoftMax


def comprehensive_comparison_plot(bags_dict, save_dir='result'):
    """
    综合对比实验：比较不同ngram、不同学习率、不同批次策略的效果

    参数:
        bags_dict: {ngram_name: bag} 字典，如 {"1-gram": bag1, "(1,2)-gram": bag2}
        save_dir: 结果保存目录

    实验配置:
        - 随机批次 (SHUFFLE): sgd(1), 16, 32, 64
        - 顺序批次 (SEQUENTIAL): 1, 16, 32, 64
        - 不同batch_size配置不同学习率范围
    """
    os.makedirs(save_dir, exist_ok=True)

    # 不同batch_size对应的学习率（小batch用小lr，大batch用大lr）
    lr_configs = {
        1: [0.01, 0.05, 0.1, 0.2],
        16: [0.05, 0.1, 0.2, 0.5],
        32: [0.1, 0.2, 0.5, 1],
        64: [0.2, 0.5, 1, 5],
    }

    # 批次配置: (名称, 策略, batch_size)
    batch_configs = [
        # 随机批次
        ("Shuffle_SGD", SHUFFLE, 1),
        ("Shuffle_16", SHUFFLE, 16),
        ("Shuffle_32", SHUFFLE, 32),
        ("Shuffle_64", SHUFFLE, 64),
        # 顺序批次
        ("Sequential_1", SEQUENTIAL, 1),
        ("Sequential_16", SEQUENTIAL, 16),
        ("Sequential_32", SEQUENTIAL, 32),
        ("Sequential_64", SEQUENTIAL, 64),
    ]

    # 统一的epoch数（完整遍历数据集的次数）
    total_epochs = 100
    eval_interval = 5

    # 存储所有结果
    all_results = {}
    best_configs = {}

    print("=" * 60)
    print("综合对比实验：N-gram × 批次策略 × 学习率")
    print(f"统一训练 {total_epochs} 个epoch（完整遍历数据集次数）")
    print("=" * 60)

    for ngram_name, bag in bags_dict.items():
        n_samples = bag.train_x.shape[1]
        n_features = len(bag.word_map)

        print(f"\n{'='*60}")
        print(f"[{ngram_name}] 特征数={n_features}, 样本数={n_samples}")
        print("=" * 60)

        all_results[ngram_name] = {}
        best_acc_for_ngram = 0
        best_config_for_ngram = None

        for batch_name, strategy, batch_size in batch_configs:
            print(f"\n  [{batch_name}]")
            all_results[ngram_name][batch_name] = {}

            lr_list = lr_configs[batch_size]
            iters_per_epoch = (n_samples + batch_size - 1) // batch_size

            # 实时绘图
            plt.ion()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'{ngram_name} - {batch_name} (epochs={total_epochs})')

            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Train Accuracy')
            ax1.grid(True, alpha=0.3)

            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Test Accuracy')
            ax2.grid(True, alpha=0.3)

            plt.show(block=False)

            colors = plt.cm.viridis(np.linspace(0, 1, len(lr_list)))
            train_lines = {}
            test_lines = {}

            for idx, lr in enumerate(lr_list):
                train_lines[lr], = ax1.plot([], [], color=colors[idx], label=f'lr={lr}')
                test_lines[lr], = ax2.plot([], [], color=colors[idx], label=f'lr={lr}')

            ax1.legend(loc='lower right', fontsize=8)
            ax2.legend(loc='lower right', fontsize=8)

            for lr in lr_list:
                soft = SoftMax(n_features)
                train_acc_list = []
                test_acc_list = []
                epoch_list = []

                start_time = time.time()

                for epoch in range(total_epochs):
                    for _ in range(iters_per_epoch):
                        soft.regression(bag.train_x, bag.train_y, strategy, lr, batch_size)

                    if epoch % eval_interval == 0 or epoch == total_epochs - 1:
                        train_acc = soft.correct_rate(bag.train_x, bag.train_type)
                        test_acc = soft.correct_rate(bag.test_x, bag.test_type)

                        train_acc_list.append(train_acc)
                        test_acc_list.append(test_acc)
                        epoch_list.append(epoch)

                        # 实时更新图形
                        train_lines[lr].set_data(epoch_list, train_acc_list)
                        test_lines[lr].set_data(epoch_list, test_acc_list)

                        ax1.relim()
                        ax1.autoscale_view()
                        ax2.relim()
                        ax2.autoscale_view()

                        fig.canvas.draw()
                        fig.canvas.flush_events()
                        plt.pause(0.01)

                elapsed = time.time() - start_time
                final_test_acc = test_acc_list[-1]

                all_results[ngram_name][batch_name][lr] = (train_acc_list, test_acc_list, epoch_list)

                print(f"    lr={lr}: train={train_acc_list[-1]:.4f}, test={final_test_acc:.4f}, time={elapsed:.1f}s")

                if final_test_acc > best_acc_for_ngram:
                    best_acc_for_ngram = final_test_acc
                    best_config_for_ngram = (batch_name, lr, final_test_acc)

            # 保存并关闭当前图
            plt.ioff()
            plt.tight_layout()
            plt.savefig(f'{save_dir}/{ngram_name}_{batch_name}.png', dpi=150)
            plt.close(fig)
            print(f"    保存: {save_dir}/{ngram_name}_{batch_name}.png")

        best_configs[ngram_name] = best_config_for_ngram
        print(f"\n  [{ngram_name}] 最佳配置: {best_config_for_ngram[0]}, lr={best_config_for_ngram[1]}, acc={best_config_for_ngram[2]:.4f}")

    # 绘制汇总图
    _plot_summary(all_results, best_configs, lr_configs, batch_configs, save_dir)

    print("\n" + "=" * 60)
    print("实验完成！结果保存在:", save_dir)
    print("=" * 60)

    return all_results, best_configs


def _plot_summary(all_results, best_configs, lr_configs, batch_configs, save_dir):
    """
    绘制汇总对比图（实时显示）
    """
    ngram_names = list(all_results.keys())
    n_ngrams = len(ngram_names)

    shuffle_batches = [c for c in batch_configs if c[1] == SHUFFLE]
    seq_batches = [c for c in batch_configs if c[1] == SEQUENTIAL]

    # 图1: Shuffle vs Sequential 对比
    print("\n绘制汇总图: Shuffle vs Sequential...")
    plt.ion()
    fig1, axes1 = plt.subplots(n_ngrams, 2, figsize=(14, 5 * n_ngrams))
    if n_ngrams == 1:
        axes1 = axes1.reshape(1, -1)
    fig1.suptitle('Shuffle vs Sequential Comparison (Best LR per config)', fontsize=14)
    plt.show(block=False)

    for i, ngram_name in enumerate(ngram_names):
        # 随机批次
        ax_shuffle = axes1[i, 0]
        ax_shuffle.set_title(f'{ngram_name} - Shuffle (Test Accuracy)')
        ax_shuffle.set_xlabel('Epoch')
        ax_shuffle.set_ylabel('Accuracy')
        ax_shuffle.grid(True, alpha=0.3)

        colors = plt.cm.tab10(np.linspace(0, 1, len(shuffle_batches)))
        for j, (batch_name, _, batch_size) in enumerate(shuffle_batches):
            lr_list = lr_configs[batch_size]
            best_lr = max(lr_list, key=lambda lr: all_results[ngram_name][batch_name][lr][1][-1])
            _, test_acc_list, epoch_list = all_results[ngram_name][batch_name][best_lr]
            label = batch_name.replace("Shuffle_", "") + f" (lr={best_lr})"
            ax_shuffle.plot(epoch_list, test_acc_list, color=colors[j], label=label)
        ax_shuffle.legend(loc='lower right', fontsize=8)

        # 顺序批次
        ax_seq = axes1[i, 1]
        ax_seq.set_title(f'{ngram_name} - Sequential (Test Accuracy)')
        ax_seq.set_xlabel('Epoch')
        ax_seq.set_ylabel('Accuracy')
        ax_seq.grid(True, alpha=0.3)

        colors = plt.cm.tab10(np.linspace(0, 1, len(seq_batches)))
        for j, (batch_name, _, batch_size) in enumerate(seq_batches):
            lr_list = lr_configs[batch_size]
            best_lr = max(lr_list, key=lambda lr: all_results[ngram_name][batch_name][lr][1][-1])
            _, test_acc_list, epoch_list = all_results[ngram_name][batch_name][best_lr]
            label = batch_name.replace("Sequential_", "") + f" (lr={best_lr})"
            ax_seq.plot(epoch_list, test_acc_list, color=colors[j], label=label)
        ax_seq.legend(loc='lower right', fontsize=8)

        fig1.canvas.draw()
        fig1.canvas.flush_events()
        plt.pause(0.1)

    plt.ioff()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/summary_shuffle_vs_sequential.png', dpi=150)
    plt.close(fig1)
    print(f"  保存: {save_dir}/summary_shuffle_vs_sequential.png")

    # 图2: N-gram 最佳配置对比
    if n_ngrams > 1:
        print("\n绘制汇总图: N-gram Best...")
        plt.ion()
        fig2, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle('N-gram Comparison (Best Config)', fontsize=14)
        plt.show(block=False)

        ax_train.set_title('Train Accuracy')
        ax_train.set_xlabel('Epoch')
        ax_train.set_ylabel('Accuracy')
        ax_train.grid(True, alpha=0.3)

        ax_test.set_title('Test Accuracy')
        ax_test.set_xlabel('Epoch')
        ax_test.set_ylabel('Accuracy')
        ax_test.grid(True, alpha=0.3)

        colors = plt.cm.tab10(np.linspace(0, 1, n_ngrams))

        for i, ngram_name in enumerate(ngram_names):
            best_batch, best_lr, _ = best_configs[ngram_name]
            train_acc_list, test_acc_list, epoch_list = all_results[ngram_name][best_batch][best_lr]
            label = f'{ngram_name} ({best_batch}, lr={best_lr})'
            ax_train.plot(epoch_list, train_acc_list, color=colors[i], label=label)
            ax_test.plot(epoch_list, test_acc_list, color=colors[i], label=label)

        ax_train.legend(loc='lower right', fontsize=8)
        ax_test.legend(loc='lower right', fontsize=8)

        fig2.canvas.draw()
        fig2.canvas.flush_events()
        plt.pause(0.5)

        plt.ioff()
        plt.tight_layout()
        plt.savefig(f'{save_dir}/summary_ngram_best.png', dpi=150)
        plt.close(fig2)
        print(f"  保存: {save_dir}/summary_ngram_best.png")

    # 图3: 热力图
    print("\n绘制汇总图: Heatmap...")
    plt.ion()
    fig3, ax3 = plt.subplots(figsize=(12, max(4, n_ngrams * 1.5)))
    plt.show(block=False)

    batch_names = [c[0] for c in batch_configs]
    heatmap_data = np.zeros((n_ngrams, len(batch_configs)))

    for i, ngram_name in enumerate(ngram_names):
        for j, (batch_name, _, batch_size) in enumerate(batch_configs):
            lr_list = lr_configs[batch_size]
            best_acc = max(all_results[ngram_name][batch_name][lr][1][-1] for lr in lr_list)
            heatmap_data[i, j] = best_acc

    im = ax3.imshow(heatmap_data, cmap='YlGn', aspect='auto')
    ax3.set_xticks(np.arange(len(batch_names)))
    ax3.set_yticks(np.arange(n_ngrams))
    ax3.set_xticklabels(batch_names, rotation=45, ha='right')
    ax3.set_yticklabels(ngram_names)

    for i in range(n_ngrams):
        for j in range(len(batch_configs)):
            ax3.text(j, i, f'{heatmap_data[i, j]:.3f}',
                    ha='center', va='center', fontsize=9,
                    color='white' if heatmap_data[i, j] > 0.5 else 'black')

    ax3.set_title('Best Test Accuracy Heatmap (N-gram × Batch Config)', fontsize=12)
    plt.colorbar(im, ax=ax3, label='Test Accuracy')

    fig3.canvas.draw()
    fig3.canvas.flush_events()
    plt.pause(0.5)

    plt.ioff()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/summary_heatmap.png', dpi=150)
    plt.close(fig3)
    print(f"  保存: {save_dir}/summary_heatmap.png")
