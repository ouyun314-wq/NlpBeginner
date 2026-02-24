# 实验2：RNN vs CNN2架构对比

## 实验设置

- **数据集**: Movie Reviews情感分类，5分类
- **CNN2**: 双通道（Random + Pre-trained embedding）
- **RNN**: 基础RNN，hidden_size=128
- **训练轮数**: 50 epochs
- **Batch Size**: 64

---

## 实验结果（先看结论）

| 模型 | 总时间(秒) | 平均时间/epoch(秒) | 最终准确率(%) |
|------|----------|------------------|--------------|
| CNN2 | 466.69 | 9.33 | 61.42 |
| RNN | 373.77 | 7.48 | **65.83** |

**意外发现**: RNN不仅更快（快25%），而且准确率更高（高4.41%）！

---

## 模型原理

### RNN架构

#### 基本公式

对于输入序列 $(x_1, x_2, ..., x_n)$，RNN在每个时间步 $t$ 计算：

$$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$$

其中：
- $h_t \in \mathbb{R}^{128}$: 隐藏状态
- $W_{hh} \in \mathbb{R}^{128 \times 128}$: 循环权重
- $W_{xh} \in \mathbb{R}^{128 \times 50}$: 输入权重
- $x_t$: 词嵌入向量

#### 完整前向传播

$$
\begin{aligned}
h_1 &= \tanh(W_{hh}h_0 + W_{xh}x_1 + b) \\
h_2 &= \tanh(W_{hh}h_1 + W_{xh}x_2 + b) \\
&\vdots \\
h_n &= \tanh(W_{hh}h_{n-1} + W_{xh}x_n + b)
\end{aligned}
$$

输出层：

$$\mathbf{y} = W_{hy}h_n + b_y$$

### CNN2架构

#### 双通道设计

$$\mathbf{X}_{input} = [\mathbf{X}_{rand}; \mathbf{X}_{pre}] \in \mathbb{R}^{2 \times n \times 50}$$

- Channel 1: 随机初始化embedding（可训练）
- Channel 2: 预训练embedding（frozen）

#### 卷积操作

对于卷积核大小 $h \in \{2,3,4,5\}$:

$$c_i = \text{ReLU}\left(\sum_{c=1}^{2}\sum_{j=0}^{h-1}\sum_{k=0}^{49} W_{c,j,k} \cdot X_{c,i+j,k} + b\right)$$

Max pooling:

$$\hat{c} = \max_i c_i$$

---

## 梯度推导

### RNN: BPTT（通过时间反向传播）

#### 前向传播

$$z_t = W_{hh}h_{t-1} + W_{xh}x_t + b$$
$$h_t = \tanh(z_t)$$

#### 反向传播

从输出层传回的梯度：

$$\delta_n = W_{hy}^T(\hat{\mathbf{y}} - \mathbf{y})$$

时间步 $t$ 的梯度（向前传播）：

$$\delta_t = W_{hh}^T[\delta_{t+1} \odot (1 - h_{t+1}^2)]$$

其中 $(1-h^2)$ 是tanh的导数：

$$\frac{d\tanh(x)}{dx} = 1 - \tanh^2(x)$$

#### 权重梯度累积

$$\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^{n} [\delta_t \odot (1-h_t^2)] h_{t-1}^T$$

$$\frac{\partial \mathcal{L}}{\partial W_{xh}} = \sum_{t=1}^{n} [\delta_t \odot (1-h_t^2)] x_t^T$$

### 梯度消失问题

从时间步 $n$ 到时间步 $1$ 的梯度：

$$\frac{\partial h_n}{\partial h_1} = \prod_{t=2}^{n} \frac{\partial h_t}{\partial h_{t-1}} = \prod_{t=2}^{n} W_{hh} \cdot \text{diag}(1-h_{t-1}^2)$$

**当 $\|W_{hh}\| < 1$ 时**:

$$\left\|\frac{\partial h_n}{\partial h_1}\right\| \leq \|W_{hh}\|^{n-1} \to 0$$

这就是著名的梯度消失问题！

但**本实验中RNN表现良好**，因为电影评论句子较短（平均长度<30），梯度消失影响有限。

### CNN: 标准反向传播

CNN没有时序依赖，梯度直接通过卷积层传播，**不会梯度消失**。

Max pooling梯度：

$$\frac{\partial \mathcal{L}}{\partial c_i} = \begin{cases}\frac{\partial \mathcal{L}}{\partial \hat{c}} & i = \arg\max_j c_j \\ 0 & \text{otherwise}\end{cases}$$

卷积层梯度：

$$\frac{\partial \mathcal{L}}{\partial W} = \sum_i \frac{\partial \mathcal{L}}{\partial c_i} \cdot \mathbb{1}_{c_i>0} \cdot X_i$$

---

## 计算复杂度对比

### 时间复杂度

#### RNN

每个时间步：

$$O(d_h^2 + d \cdot d_h) = O(128^2 + 50 \times 128) = O(22.8K)$$

$n$ 个时间步：

$$O(n \cdot d_h(d_h + d))$$

**关键**：必须串行计算，**无法并行**

#### CNN

卷积层：

$$O(K \cdot n \cdot h \cdot d \cdot c) = O(64 \times n \times 4 \times 50 \times 2)$$

**关键**：所有卷积核**完全并行**

### 实际速度对比

| 指标 | CNN2 | RNN | 对比 |
|------|------|-----|------|
| 每epoch时间 | 9.33秒 | 7.48秒 | RNN快20% |
| GPU利用率 | 高 | 中 | CNN理论更好 |

**意外**：RNN居然更快！

可能原因：
1. **句子不够长**：平均<30词，RNN串行计算时间可接受
2. **CNN双通道开销**：两个embedding层增加内存和计算
3. **卷积核数量多**：64个卷积核计算量大
4. **PyTorch优化**：RNN的CUDA kernel高度优化

---

## 为什么RNN准确率更高？

### 65.83% vs 61.42% 的差距分析

#### 1. 序列建模能力

RNN捕获了**词序信息**：

"not good" vs "good"

$$h_{\text{not good}} = f(f(h_0, \text{not}), \text{good})$$

RNN能区分否定

CNN只看到"not"和"good"都存在，可能混淆

#### 2. 长距离依赖

虽然RNN有梯度消失，但短文本(<30词)影响不大

$$h_t = f(...f(f(h_0, x_1), x_2)..., x_t)$$

理论上$h_t$包含所有历史信息

CNN只看局部窗口(2-5词)，可能错过长距离关系

#### 3. 全局语义

RNN的最后隐藏状态$h_n$是整个句子的**全局表示**

CNN通过max pooling聚合，可能丢失信息

#### 4. 实验数据支持

看看学习曲线：

**Epoch 10**:
- CNN2: 58.65%
- RNN: **61.84%**

**Epoch 30**:
- CNN2: 61.15%
- RNN: **64.38%**

**RNN全程领先！**

---

## 并行性对比

### RNN的串行瓶颈

$$h_t = f(h_{t-1}, x_t)$$

时间步 $t$ **必须等待** $t-1$ 完成

**GPU利用率**：

在batch维度并行，但时序维度串行：

$$\text{并行度} = \frac{b}{1} = b$$

其中$b$是batch size

### CNN的并行优势

所有卷积操作独立：

$$c_i = f(X_{i:i+h})$$

**GPU利用率**：

batch、卷积核、位置三个维度都可并行：

$$\text{并行度} = b \times K \times (n-h+1)$$

### 为什么实际中RNN更快？

1. **batch size不够大**：64不足以填满GPU
2. **句子较短**：串行瓶颈不明显
3. **CNN内存访问**：双通道增加内存带宽压力
4. **cuDNN优化**：RNN有高度优化的CUDA实现

---

## 参数数量对比

### RNN参数

$$\begin{aligned}
\text{Embedding} &: |V| \times 50 \\
W_{hh} &: 128 \times 128 = 16.4K \\
W_{xh} &: 128 \times 50 = 6.4K \\
W_{hy} &: 5 \times 128 = 640 \\
\text{Total} &\approx |V| \times 50 + 23.5K
\end{aligned}$$

### CNN2参数

$$\begin{aligned}
\text{Embedding (双通道)} &: 2|V| \times 50 \\
\text{Convolutions} &: 4 \times 16 \times (h \times 50 \times 2) \approx 44.8K \\
W_{fc} &: 64 \times 5 = 320 \\
\text{Total} &\approx 2|V| \times 50 + 45K
\end{aligned}$$

**CNN2参数更多**（双倍embedding + 更大卷积参数）

但**RNN的循环权重 $W_{hh}$ 使优化更困难**（梯度问题）

---

## 学习曲线详细分析

### 前10个epoch

| Epoch | CNN2 Test Acc | RNN Test Acc | 差距 |
|-------|--------------|-------------|------|
| 1 | 34.96% | 45.05% | +10.09% |
| 5 | 52.67% | 54.09% | +1.42% |
| 10 | 58.65% | 61.84% | +3.19% |

**RNN一开始就领先**，可能因为预训练embedding更好地初始化了RNN

### 中期 (Epoch 10-30)

两者都在稳步上升，差距保持在2-4%

### 后期 (Epoch 30-50)

| Epoch | CNN2 Test Acc | RNN Test Acc |
|-------|--------------|-------------|
| 40 | 61.58% | 66.37% |
| 50 | 61.42% | 65.83% |

CNN2开始波动下降，RNN保持上升趋势

---

## 意外发现讨论

### 传统观点

**CNN应该**：
- 训练更快（并行计算）
- 对短文本效果好（局部特征）

**RNN应该**：
- 训练更慢（串行计算）
- 对长文本效果好（序列建模）

### 本实验结果

**RNN实际**：
- ✅ 更快（7.48s vs 9.33s）
- ✅ 更准（65.83% vs 61.42%）

**与传统观点矛盾！**

### 可能解释

#### 1. 数据特点

电影评论虽短，但**词序很重要**：

- "not good" vs "good"
- "bad but interesting" vs "bad and boring"

RNN天然处理这种语序依赖

#### 2. 模型设计

CNN2的**双通道**可能引入复杂性：
- 两个embedding如何协同？
- 是否存在冲突？

RNN只用一个预训练embedding，更简单直接

#### 3. 超参数

- CNN可能需要更多卷积核
- RNN的hidden_size=128可能正好合适

#### 4. 任务特性

情感分类可能更依赖**整体语义理解**而非局部特征

---

## 梯度消失的数学证明

### 连乘形式

$$\frac{\partial h_t}{\partial h_1} = \prod_{k=2}^{t} \frac{\partial h_k}{\partial h_{k-1}}$$

每一步：

$$\frac{\partial h_k}{\partial h_{k-1}} = W_{hh} \cdot \text{diag}(1-h_{k-1}^2)$$

因为$|1-h^2| \leq 1$（tanh导数范围），所以：

$$\left\|\frac{\partial h_k}{\partial h_{k-1}}\right\| \leq \|W_{hh}\|$$

连乘$t-1$次：

$$\left\|\frac{\partial h_t}{\partial h_1}\right\| \leq \|W_{hh}\|^{t-1}$$

**当 $\|W_{hh}\| < 1$ 时**：

$$\lim_{t \to \infty}\|W_{hh}\|^{t-1} = 0$$

### 本实验为什么没事？

平均句子长度约25词，梯度衰减：

$$0.95^{25} \approx 0.28$$

仍然可以传播，**不算严重的梯度消失**

如果$\|W_{hh}\| = 0.5$：

$$0.5^{25} \approx 3 \times 10^{-8}$$

就完全消失了！

---

## 实际应用建议

### 选择RNN的场景

✅ 词序重要的任务（情感分析、问答）
✅ 需要捕获依赖关系
✅ 句子长度适中（<50词）
✅ 准确率优先于速度

### 选择CNN的场景

✅ 关键词检测（广告过滤、spam检测）
✅ 主题分类（新闻分类）
✅ 极短文本（标题、标签）
✅ 需要极致速度（实时系统）

### 本实验的启示

**不要盲信教条**：

- "CNN快"不一定对（取决于具体实现和数据）
- "RNN慢"也不绝对
- "CNN适合短文本"不完全准确

**要根据任务实测！**

---

## 改进方向

### RNN可以更好

1. **LSTM/GRU**：解决梯度消失

$$\text{LSTM: } c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

门控机制保护梯度

2. **双向RNN**：同时看前后文

$$h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$$

3. **Attention**：关注重要部分

$$\alpha_t = \text{softmax}(score(h_t))$$
$$context = \sum_t \alpha_t h_t$$

### CNN可以更好

1. **更多卷积核**：16可能不够，试试64、128

2. **膨胀卷积**：增大感受野

$$c_i = f\left(\sum W_j \cdot x_{i+r \cdot j}\right)$$

$r$是膨胀率

3. **残差连接**：加深网络

$$h_{l+1} = f(h_l) + h_l$$

4. **单通道**：简化模型，可能反而更好

---

## 结论

1. **性能**: RNN > CNN2（65.83% vs 61.42%，高4.41%）
2. **速度**: RNN > CNN2（7.48s vs 9.33s，快20%）
3. **原因**: 任务特性（词序重要）+ 数据特点（句子不长）
4. **教训**: 实验胜于理论假设

**最重要的发现**：对于情感分类这种**词序敏感、需要全局理解的任务**，RNN的序列建模能力超过了CNN的并行优势，即使在短文本上也是如此。

**实践指导**：选择模型时不要只看理论分析，要：
1. 理解任务特点（是否需要词序？）
2. 分析数据特征（长度分布？）
3. 实际测试对比（多试几个模型）
4. 根据结果选择（准确率 vs 速度的权衡）
