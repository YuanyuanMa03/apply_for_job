# Transformer架构数学公式推导

## 1. 缩放点积注意力（Scaled Dot-Product Attention）

缩放点积注意力是Transformer的核心组件，其计算过程如下：

给定查询矩阵 $Q \in \mathbb{R}^{n \times d_k}$、键矩阵 $K \in \mathbb{R}^{m \times d_k}$ 和值矩阵 $V \in \mathbb{R}^{m \times d_v}$，其中：
- $n$ 是查询序列长度
- $m$ 是键值序列长度
- $d_k$ 是查询和键的维度
- $d_v$ 是值的维度

### 1.1 注意力权重计算

首先计算查询和键的点积，然后进行缩放：

$$\text{scores} = \frac{QK^T}{\sqrt{d_k}} \in \mathbb{R}^{n \times m}$$

其中，缩放因子 $\sqrt{d_k}$ 用于防止点积过大导致梯度消失。

### 1.2 注意力权重归一化

对注意力分数进行softmax归一化：

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中，softmax函数定义为：

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{m} e^{x_j}}$$

### 1.3 带掩码的注意力

在训练过程中，可能需要添加掩码（如解码器中的自回归掩码）：

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

其中，$M$ 是掩码矩阵，对于被掩码的位置，$M_{ij} = -\infty$，否则 $M_{ij} = 0$。

## 2. 多头自注意力机制（Multi-Head Attention）

多头注意力将输入分割到不同的子空间，并行计算多个注意力头，然后合并结果。

### 2.1 查询、键、值的线性变换

给定输入 $X \in \mathbb{R}^{n \times d_{\text{model}}}$，其中 $d_{\text{model}}$ 是模型维度：

对于第 $i$ 个注意力头，使用不同的权重矩阵进行线性变换：

$$Q_i = XW_i^Q, \quad K_i = XW_i^K, \quad V_i = XW_i^V$$

其中：
- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$

### 2.2 并行计算注意力头

对每个注意力头计算注意力：

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_iK_i^T}{\sqrt{d_k}}\right)V_i$$

### 2.3 多头注意力合并

将所有注意力头的输出拼接并通过线性变换：

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h)W^O$$

其中：
- $h$ 是注意力头的数量
- $W^O \in \mathbb{R}^{h \cdot d_v \times d_{\text{model}}}$ 是输出权重矩阵
- 通常设置 $d_k = d_v = d_{\text{model}}/h$

## 3. 位置编码（Positional Encoding）

由于Transformer不包含递归或卷积结构，需要显式地添加位置信息。

### 3.1 正弦和余弦位置编码

对于位置 $pos$ 和维度 $i$，位置编码定义为：

$$PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

其中：
- $pos$ 是位置索引（从0开始）
- $i$ 是维度索引（从0开始）
- $d_{\text{model}}$ 是模型维度

### 3.2 位置编码的性质

这种位置编码具有以下重要性质：
1. 对于任意固定的偏移量 $k$，$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数
2. 相对位置关系在不同位置间保持一致

### 3.3 位置编码的添加

将位置编码加到输入嵌入中：

$$X' = X + PE$$

其中 $X \in \mathbb{R}^{n \times d_{\text{model}}}$ 是输入嵌入，$PE \in \mathbb{R}^{n \times d_{\text{model}}}$ 是位置编码。

## 4. 编码器-解码器结构（Encoder-Decoder Architecture）

Transformer采用编码器-解码器架构，用于序列到序列的任务。

### 4.1 编码器结构

编码器由 $N$ 个相同的层堆叠而成，每层包含两个子层：

$$\text{EncoderLayer}(X) = \text{LayerNorm}(X + \text{MultiHeadAttention}(X, X, X))$$

$$\text{Output} = \text{LayerNorm}(\text{EncoderLayer}(X) + \text{FeedForward}(\text{EncoderLayer}(X)))$$

### 4.2 解码器结构

解码器也由 $N$ 个相同的层堆叠而成，每层包含三个子层：

1. 掩码多头自注意力：
$$\text{MaskedMultiHeadAttention}(X, X, X)$$

2. 编码器-解码器注意力：
$$\text{MultiHeadAttention}(X, \text{Memory}, \text{Memory})$$

3. 前馈网络：
$$\text{FeedForward}(X)$$

完整解码器层计算：

$$\text{DecoderLayer}(X) = \text{LayerNorm}(X + \text{MaskedMultiHeadAttention}(X, X, X))$$

$$\text{DecoderLayer}_2 = \text{LayerNorm}(\text{DecoderLayer}(X) + \text{MultiHeadAttention}(\text{DecoderLayer}(X), \text{Memory}, \text{Memory}))$$

$$\text{Output} = \text{LayerNorm}(\text{DecoderLayer}_2 + \text{FeedForward}(\text{DecoderLayer}_2))$$

### 4.3 最终输出层

解码器输出通过线性变换和softmax得到概率分布：

$$\text{Output} = \text{Softmax}(\text{Linear}(X))$$

## 5. 前馈神经网络（Feed Forward Network）

每个编码器和解码器层都包含一个前馈神经网络，该网络独立应用于每个位置。

### 5.1 前馈网络结构

前馈网络由两个线性变换和一个非线性激活函数组成：

$$\text{FFN}(x) = \text{max}(0, xW_1 + b_1)W_2 + b_2$$

其中：
- $x \in \mathbb{R}^{d_{\text{model}}}$ 是输入
- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$ 是第一个线性变换的权重矩阵
- $W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$ 是第二个线性变换的权重矩阵
- $b_1 \in \mathbb{R}^{d_{ff}}$ 和 $b_2 \in \mathbb{R}^{d_{\text{model}}}$ 是偏置向量
- $d_{ff}$ 是前馈网络的隐藏层维度，通常设置为 $4d_{\text{model}}$
- $\text{max}(0, \cdot)$ 是ReLU激活函数

### 5.2 前馈网络的作用

前馈网络的作用：
1. 提供非线性变换能力
2. 增强模型表达能力
3. 在不同位置间独立处理信息

## 6. 层归一化（Layer Normalization）

层归一化是Transformer中的重要组件，用于稳定训练过程。

### 6.1 层归一化定义

对于输入 $x \in \mathbb{R}^{d_{\text{model}}}$，层归一化定义为：

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta$$

其中：
- $\mu = \frac{1}{d_{\text{model}}}\sum_{i=1}^{d_{\text{model}}} x_i$ 是均值
- $\sigma = \sqrt{\frac{1}{d_{\text{model}}}\sum_{i=1}^{d_{\text{model}}}(x_i - \mu)^2 + \epsilon}$ 是标准差
- $\gamma \in \mathbb{R}^{d_{\text{model}}}$ 是缩放参数
- $\beta \in \mathbb{R}^{d_{\text{model}}}$ 是偏移参数
- $\epsilon$ 是一个小的常数，用于数值稳定性
- $\odot$ 表示逐元素乘法

### 6.2 残差连接与层归一化

Transformer中采用残差连接和层归一化：

$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

其中，$\text{Sublayer}(x)$ 是子层（如多头注意力或前馈网络）的输出。

### 6.3 层归一化的优势

层归一化的优势：
1. 稳定训练过程
2. 允许更高的学习率
3. 减少对初始化的敏感性
4. 加速收敛

## 7. 完整Transformer模型

### 7.1 编码器-解码器整体流程

给定输入序列 $X = (x_1, x_2, \ldots, x_n)$ 和目标序列 $Y = (y_1, y_2, \ldots, y_m)$：

1. 输入嵌入和位置编码：
$$X_{\text{embed}} = \text{Embedding}(X) + \text{PositionalEncoding}(n)$$
$$Y_{\text{embed}} = \text{Embedding}(Y) + \text{PositionalEncoding}(m)$$

2. 编码器处理：
$$H = \text{Encoder}(X_{\text{embed}})$$

3. 解码器处理：
$$Z = \text{Decoder}(Y_{\text{embed}}, H)$$

4. 输出概率：
$$P(Y|X) = \text{Softmax}(\text{Linear}(Z))$$

### 7.2 训练目标

Transformer使用交叉熵损失函数：

$$\mathcal{L} = -\sum_{t=1}^{m} \log P(y_t | y_{<t}, X)$$

其中，$y_{<t} = (y_1, y_2, \ldots, y_{t-1})$ 是目标序列的前缀。

## 8. 复杂度分析

### 8.1 计算复杂度

对于序列长度为 $n$，模型维度为 $d$ 的Transformer：

- 自注意力机制的计算复杂度：$\mathcal{O}(n^2 \cdot d)$
- 前馈网络的计算复杂度：$\mathcal{O}(n \cdot d^2)$
- 总体计算复杂度：$\mathcal{O}(n^2 \cdot d + n \cdot d^2)$

### 8.2 内存复杂度

- 自注意力机制的内存复杂度：$\mathcal{O}(n^2)$
- 前馈网络的内存复杂度：$\mathcal{O}(n \cdot d)$

## 9. 关键超参数

Transformer的关键超参数及其典型值：

| 超参数 | 符号 | 典型值 | 说明 |
|--------|------|--------|------|
| 模型维度 | $d_{\text{model}}$ | 512 | 输入嵌入和输出的维度 |
| 注意力头数 | $h$ | 8 | 多头注意力的头数 |
| 前馈网络维度 | $d_{ff}$ | 2048 | 前馈网络隐藏层维度 |
| 编码器层数 | $N$ | 6 | 编码器堆叠层数 |
| 解码器层数 | $N$ | 6 | 解码器堆叠层数 |
| Dropout率 | $p_{\text{dropout}}$ | 0.1 | Dropout概率 |

## 10. Transformer的变体与改进

### 10.1 常见变体

1. **Transformer-XL**：引入分段循环机制和相对位置编码
2. **Universal Transformer**：引入自适应计算时间和深度
3. **Longformer**：引入稀疏注意力机制
4. **Reformer**：使用可逆层和局部敏感哈希注意力

### 10.2 效率优化

1. **稀疏注意力**：减少计算复杂度从 $\mathcal{O}(n^2)$ 到 $\mathcal{O}(n \log n)$ 或更低
2. **线性注意力**：使用核函数近似注意力机制
3. **低秩近似**：使用矩阵分解减少参数量

## 结论

Transformer通过自注意力机制、位置编码和前馈网络的组合，实现了强大的序列建模能力。其数学基础清晰，结构设计优雅，已成为自然语言处理和其他序列任务的基础架构。通过理解其数学公式和计算过程，可以更好地应用和改进这一架构。