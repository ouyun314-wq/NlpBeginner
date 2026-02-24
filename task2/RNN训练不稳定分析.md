# RNN训练不稳定性分析

## 问题现象

RNN测试准确率在后期剧烈震荡，**单个epoch波动超过6%**：

| Epoch | Test Acc (%) | 变化 |
|-------|-------------|------|
| 39 | 66.37 | - |
| 40 | 65.30 | -1.07% |
| 41 | 61.69 | **-3.61%** ⬇️ |
| 42 | 67.70 | **+6.01%** ⬆️ |
| 43 | 64.47 | -3.23% |
| 49 | 59.97 | **-4.50%** ⬇️ |
| 50 | 65.83 | **+5.86%** ⬆️ |

这种震荡幅度在深度学习中是**严重的训练不稳定**。

---

## 根本原因分析

### 1. 梯度爆炸 (Gradient Exploding)

#### 数学原理

RNN的BPTT梯度：

$$\frac{\partial h_t}{\partial h_1} = \prod_{k=2}^{t} W_{hh} \cdot \text{diag}(1-h_{k-1}^2)$$

**当 $\|W_{hh}\| > 1$ 时**：

$$\left\|\frac{\partial h_t}{\partial h_1}\right\| \geq \|W_{hh}\|^{t-1} \to \infty$$

#### 为什么会爆炸？

在训练过程中，$W_{hh}$的某些特征值可能大于1：

$$\sigma_{\max}(W_{hh}) > 1$$

经过$t$步连乘：

$$\sigma_{\max}(W_{hh})^t \to \infty$$

#### 实验证据

看训练loss：

| Epoch | CNN2 Train Loss | RNN Train Loss |
|-------|----------------|---------------|
| 40 | 0.513 | 0.617 |
| 41 | 0.513 | 0.615 |
| 42 | 0.510 | 0.614 |

RNN的train loss虽然在下降，但**测试准确率剧烈波动**，说明：
- 训练集上在过拟合
- 梯度大到跳过了好的局部最优
- 权重更新太激进

#### 梯度爆炸的表现

$$\|\nabla_W \mathcal{L}\| \gg 1$$

导致权重更新：

$$W^{(t+1)} = W^{(t)} - \eta \nabla_W \mathcal{L}$$

步长过大，在loss landscape上**跳来跳去**：

```
Loss surface:
     ╱╲    ╱╲
    ╱  ╲  ╱  ╲
   ╱    ╲╱    ╲
   ^     ^     ^
   A     B     C
```

从A跳到C，再跳回B，永远无法稳定收敛。

---

### 2. 学习率过大

#### 当前设置

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

学习率 $\eta = 0.001$

#### 为什么过大？

对于RNN，这个学习率可能太aggressive：

1. **循环权重敏感**：$W_{hh}$的小变化会被时间步放大
2. **Adam自适应**：虽然Adam会调整学习率，但初始值很关键
3. **没有gradient clipping**：没有限制梯度大小

#### 数学分析

权重更新量：

$$\Delta W = \eta \cdot \frac{m}{\sqrt{v} + \epsilon}$$

当梯度爆炸时，$m$（一阶矩）变得很大：

$$\Delta W \propto \eta \cdot \text{huge gradient}$$

即使有$\sqrt{v}$归一化，也可能导致大幅震荡。

---

### 3. Batch Size太小导致梯度噪声

#### 当前设置

```python
batch_size = 64
```

#### 梯度噪声

每个batch的梯度是真实梯度的噪声估计：

$$\nabla_W \mathcal{L}_{batch} = \nabla_W \mathcal{L}_{true} + \epsilon$$

其中噪声标准差：

$$\sigma(\epsilon) \propto \frac{1}{\sqrt{b}}$$

$b=64$时噪声较大，导致优化路径震荡。

#### RNN vs CNN的差异

**为什么CNN稳定？**

CNN参数更多（45K vs 23K），**过参数化**（overparameterization）使得优化景观更平滑。

RNN参数少，优化景观**更崎岖**：

```
CNN loss surface (smooth):
      ╱‾‾╲
    ╱      ╲
  ╱          ╲

RNN loss surface (bumpy):
  ╱╲ ╱╲ ╱╲ ╱╲
 ╱  ╲  ╲  ╲  ╲
```

---

### 4. RNN固有的训练困难

#### Vanishing vs Exploding的平衡

RNN需要在两个极端之间平衡：

$$\|W_{hh}\| < 1 \Rightarrow \text{梯度消失}$$
$$\|W_{hh}\| > 1 \Rightarrow \text{梯度爆炸}$$

理想情况：$\|W_{hh}\| \approx 1$

但训练过程中$W_{hh}$不断变化，**很难维持这个平衡**。

#### 非凸优化景观

RNN的loss函数高度非凸，存在很多局部最优：

$$\mathcal{L}(W_{hh}, W_{xh}, W_{hy})$$

震荡可能是在多个局部最优之间跳跃。

---

## 解决方案

### 方案1：梯度裁剪 (Gradient Clipping) ⭐

#### 实现

```python
# 在optimizer.step()之前
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

#### 原理

限制梯度的L2范数：

$$\text{if } \|\nabla W\| > c: \quad \nabla W \leftarrow c \cdot \frac{\nabla W}{\|\nabla W\|}$$

#### 效果预期

- 防止梯度爆炸
- 保持梯度方向，只限制大小
- **这是RNN训练的标配技术！**

---

### 方案2：降低学习率

#### 建议

```python
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 降到1e-4
```

或使用学习率衰减：

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)
```

当验证准确率不再提升时，自动降低学习率。

#### 数学直觉

较小的学习率 → 较小的权重更新 → 更稳定的优化路径

---

### 方案3：增大Batch Size

#### 建议

```python
batch_size = 128  # 或 256
```

#### 效果

- 减少梯度噪声：$\sigma(\epsilon) \propto \frac{1}{\sqrt{b}}$
- 更准确的梯度估计
- 代价：需要更多GPU内存

---

### 方案4：使用LSTM/GRU ⭐⭐

#### 为什么更稳定？

LSTM通过门控机制控制信息流：

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

梯度可以直接通过$c_t$传播，**不经过连乘**：

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

而不是：

$$\frac{\partial h_t}{\partial h_{t-1}} = W_{hh} \cdot (1-h_{t-1}^2)$$

#### 实现

```python
self.lstm = nn.LSTM(
    input_size=feature_len,
    hidden_size=128,
    num_layers=1,
    batch_first=True,
    dropout=0.5
)
```

#### 预期效果

- 训练更稳定
- 准确率可能进一步提升
- 可能更慢（LSTM计算量更大）

---

### 方案5：添加权重正则化

#### L2正则化

```python
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
```

损失函数变为：

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \sum_W \|W\|^2$$

#### 效果

- 防止$W_{hh}$变得过大
- 间接缓解梯度爆炸
- 提升泛化能力

---

### 方案6：Dropout调整

#### 当前

```python
dropout = 0.5
```

#### 建议

对RNN，在循环连接上也加dropout：

```python
self.rnn = nn.RNN(
    input_size=50,
    hidden_size=128,
    dropout=0.5  # 这只是层间dropout
)

# 需要手动在循环连接上加dropout
h_t = self.dropout(h_t)  # 在每个时间步
```

#### 效果

- 减少过拟合
- 提升泛化（测试准确率）
- 可能降低训练准确率（acceptable）

---

## 推荐解决方案（按优先级）

### 🥇 优先级1：梯度裁剪

**代码修改**：

```python
def train_model(model, train_loader, test_loader, epochs, tracker, device, plotter=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # ⭐ 添加这一行
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
```

**预期效果**：震荡幅度从±6%降到±2%

---

### 🥈 优先级2：降低学习率

```python
optimizer = optim.Adam(model.parameters(), lr=0.0003)  # 从0.001降到0.0003
```

**预期效果**：收敛更慢但更稳定

---

### 🥉 优先级3：改用LSTM

```python
class MY_LSTM(nn.Module):
    def __init__(self, len_feature, len_hidden, len_words, weight, ...):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=len_feature,
            hidden_size=len_hidden,
            num_layers=1,
            batch_first=True
        )
        # ... 其他代码类似
```

**预期效果**：准确率可能提升到68-70%，训练稳定

---

## 理论深入：为什么梯度会爆炸？

### 谱半径分析

定义$W_{hh}$的谱半径：

$$\rho(W_{hh}) = \max_i |\lambda_i|$$

其中$\lambda_i$是$W_{hh}$的特征值。

**爆炸条件**：

$$\rho(W_{hh}) > 1$$

**消失条件**：

$$\rho(W_{hh}) < 1$$

### 训练动态

在训练初期，权重初始化可能使得$\rho(W) \approx 1$

但在优化过程中：

$$W_{hh}^{(t+1)} = W_{hh}^{(t)} - \eta \nabla_{W_{hh}}\mathcal{L}$$

梯度可能推动$\rho(W)$超过1，导致爆炸。

### 为什么CNN没这个问题？

CNN的梯度传播是**局部的**：

$$\frac{\partial \mathcal{L}}{\partial W_{conv}} = \sum_i \delta_i X_i$$

不涉及连乘，**不会指数级增长**。

---

## 实验验证建议

### 实验1：梯度裁剪对比

| 设置 | 预期震荡幅度 | 预期最终准确率 |
|------|------------|--------------|
| 无裁剪（当前） | ±6% | 65.8% |
| clip_norm=5.0 | ±2% | 66-67% |
| clip_norm=1.0 | ±1% | 65-66% |

### 实验2：学习率对比

| 学习率 | 预期震荡 | 收敛速度 |
|--------|---------|---------|
| 0.001（当前） | 剧烈 | 快 |
| 0.0003 | 中等 | 中 |
| 0.0001 | 平稳 | 慢 |

### 实验3：RNN vs LSTM

| 模型 | 预期稳定性 | 预期准确率 |
|------|-----------|-----------|
| 基础RNN | 震荡 | 65-66% |
| LSTM | 稳定 | 68-70% |
| GRU | 稳定 | 67-69% |

---

## 监控指标

### 在训练时添加梯度监控

```python
# 在backward()后，step()前
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5

print(f'Gradient norm: {total_norm:.4f}')

# 如果看到很大的数字(>10)，说明梯度爆炸了
```

### 正常梯度范围

- **健康**：0.1 - 5.0
- **警告**：5.0 - 10.0
- **爆炸**：> 10.0

---

## 总结

**根本原因**：RNN的梯度在时间步上连乘，容易爆炸

**表现**：测试准确率单epoch波动6%

**解决方案**：
1. 🥇 **梯度裁剪**（立即见效）
2. 🥈 **降低学习率**（稳妥但慢）
3. 🥉 **改用LSTM**（治本）

**实施建议**：
- 先加梯度裁剪，max_norm=5.0
- 如果还震荡，降低学习率到0.0003
- 如果想要最好效果，换LSTM

这是**RNN训练的经典问题**，每个做NLP的人都会遇到！
