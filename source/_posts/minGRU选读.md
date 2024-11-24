---
title: minGRU选读
date: 2024-10-04 20:52:42
tags: 自学
categories: 人工智能
mathjax: true
description: 本文详细介绍了minGRU的来源及其实现，包括对LSTM和GRU的回顾、时间反向传播（BPTT）的理解以及Blelloch并行扫描算法的应用。通过移除隐藏状态依赖性，minGRU实现了高效的并行训练。
cover: https://ooo.0x0.ooo/2024/10/04/O46Qf6.jpg
---

## minGRU 来源

minGRU 来源于 10/2/2024 Mila 等人提出的`Were RNNs All We Needed?`一文，下面来浅要阅读和总结一下主要内容也分析一下源码。

### `Were RNNs All We Needed?`

> 这个“Were”有一点点搞。大有一种江山已逝的美感。

1. LSTMs 回顾：
   长短期记忆网络（Long Short-Term Memory, LSTM）是一种特殊的循环神经网络（RNN），能够学习长期依赖关系。LSTM 通过引入门控机制来解决传统 RNN 中的梯度消失和梯度爆炸问题。LSTM 的核心是其细胞状态（Cell State），它通过一系列门控单元来控制信息的流动。LSTM 分为 4 层：

   1. Forget: 遗忘层，使用一个 sigmoid（决定遗忘值）
      遗忘层的作用是决定哪些信息需要从细胞状态中遗忘。通过一个 sigmoid 函数，输出一个 0 到 1 之间的值，表示需要遗忘的信息比例。

      $$
      f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
      $$

      其中，$ f \dot t $ 是遗忘门的输出，$ W_f $ 和 $ b_f $ 分别是权重矩阵和偏置向量，$ h\*{t-1} $ 是前一时刻的隐藏状态，$ x_t $ 是当前输入。

   2. Store: 存储层，一个 sigmoid（决定更新值），一个 tanh（向量生成）
      存储层的作用是决定哪些新的信息需要存储到细胞状态中。首先通过一个 sigmoid 函数决定需要更新的信息比例，然后通过一个 tanh 函数生成新的候选细胞状态。

      $$
      i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
      $$

      $$
      \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
      $$

      其中，$ i_t $ 是输入门的输出，$ \tilde{C}\_t $ 是新的候选细胞状态。

   3. Update: 更新层，从 $C*{t-1}$ -> $C_t$，forget 一个逐元素乘积，store 一个逐元素乘积后与上述加和
      更新层的作用是更新细胞状态。通过遗忘门的输出 $ f_t $ 和前一时刻的细胞状态 $ C\*{t-1} $ 逐元素相乘，表示需要保留的旧信息；通过输入门的输出 $ i_t $ 和新的候选细胞状态 $ \tilde{C}\_t $ 逐元素相乘，表示需要添加的新信息。两者相加得到新的细胞状态 $ C_t $。

      $$
      C_t = f_t * C_{t-1} + i_t * \tilde{C}_t
      $$

   4. Output: 输出层，可以理解为过滤器，运行一个 Sigmoid Layer，决定输出哪一部分的 Cell State。

      输出层的作用是决定当前时刻的隐藏状态 $ h_t $。首先通过一个 sigmoid 函数决定需要输出的信息比例，然后通过一个 tanh 函数将细胞状态规范到 -1 到 1 之间，并与 sigmoid 函数的输出逐元素相乘，得到最终的隐藏状态。

      $$
      o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
      $$

      $$
      h_t = o_t * \tanh(C_t)
      $$

      其中，$ o_t $ 是输出门的输出，$ h_t $ 是当前时刻的隐藏状态。
      ![LSTM Structure](https://ooo.0x0.ooo/2024/10/04/O46XBC.png)

2. GRU 回顾：

   一种引人注目的 LSTMs 变体被称为 GRU，它结合了 Forget Gates 和 Input Gates 到一个单独的 Update Gate，同时，也合并了 Cell State 和 Hidden State，并做出了其他的一些改变。

   另外，GRU 比标准 LSTM 更加简洁。

   GRU 结构如下：

   1. 更新门（Update Gate）：

      $$
      z_t = \sigma(W_z \cdot [h_{t-1}, x_t])
      $$

   2. 重置门（Reset Gate）：

      $$
      r_t = \sigma(W_r \cdot [h_{t-1}, x_t])
      $$

   3. 候选隐状态（Candidate Hidden State）：

      $$
      \tilde{h}_t = \tanh(W \cdot [r_t \ast h_{t-1}, x_t])
      $$

   4. 最终隐状态（Final Hidden State）：
      $$
      h_t = (1 - z_t) \ast h_{t-1} + z_t \ast \tilde{h}_t
      $$

   其中，$\sigma$ 表示 sigmoid 函数，$\tanh$ 表示 tanh 函数，$\ast$ 表示逐元素乘法，$W_z$、$W_r$ 和 $W$ 是权重矩阵，$h_t$ 是当前时间步的隐状态，$h_{t-1}$ 是前一时间步的隐状态，$x_t$ 是当前时间步的输入。
   ![GRU Structure](https://ooo.0x0.ooo/2024/10/04/O46Ici.png)

## 理解时间反向传播（BPTT）

理解 BP 即可，可参考 [知乎专栏](https://zhuanlan.zhihu.com/p/447113449#:~:text=%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD%EF%BC%88forwa)抑或这个[博客网站](https://www.cnblogs.com/BlairGrowing/p/14982115.html#:~:text=1.1%20%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD%E7%9A%84%E8%AE%A1)。

BPTT 即为 BP through time.

> 作为十多年前的传统 RNN，LSTM 和 GRU 只能按顺序计算，并需要在训练过程中通过时间反向传播 (BPTT)。因此，LSTM 和 GRU 的速度太慢，无法扩展到几百个代币以上的规模，因此已被淘汰。

## minGRU 如何巧妙的避免时间维度上的 BP

> Revisiting these models, we show that by removing hidden state dependencies from their input, forget, and update gates, LSTMs and GRUs no longer need to BPTT and can be trained efficiently using the **parallel scan algorithm**.
> 重新审视这些模型后，我们发现，通过从输入、遗忘和更新门中移除隐藏状态依赖性，LSTM 和 GRU 不再需要 BPTT，而且可以使用并行扫描算法进行高效训练。

### Parallel Scan

> Blelloch 并行扫描算法

Blelloch 并行扫描算法是一种高效的并行前缀和算法（parallel prefix sum），它用于在并行计算环境中计算前缀和，同时在处理大规模数据时表现出良好的并行性能。为了理解这算法，先简单回顾一下**前缀和**的概念，然后逐步分析 Blelloch 并行扫描算法的工作原理。

- **前缀和（Prefix Sum）**

  前缀和是给定一个数组，计算每个元素的累积和。给定输入数组 `A = [a_1, a_2, ..., a_n]`，其前缀和数组 `S = [s_1, s_2, ..., s_n]` 定义如下：

  - `s_1 = a_1`
  - `s_2 = a_1 + a_2`
  - `s_3 = a_1 + a_2 + a_3`
  - ...
  - `s_n = a_1 + a_2 + ... + a_n`

  通常，前缀和有两种形式：

  1. **包含自身的前缀和**：`S[i] = A[1] + A[2] + ... + A[i]`
  2. **不包含自身的前缀和**：`S[i] = A[1] + A[2] + ... + A[i-1]`，即 **exclusive scan**。

- **并行计算的挑战**

  在串行计算中，前缀和很容易通过线性扫描的方式实现，时间复杂度为 \(O(n)\)。但是，在并行计算中，如果直接顺序计算前缀和，会引入依赖关系，即计算下一个元素时依赖于之前元素的结果，这阻碍了并行化。因此，需要设计特殊的算法来消除这些依赖关系。

- **Blelloch 并行扫描算法的思想**

  Blelloch 并行扫描算法通过两次阶段性操作（**向上阶段（up-sweep）**和**向下阶段（down-sweep）**）实现了前缀和的并行化。这个算法的核心思想是将整个问题分解成两步，每步都是并行操作。

  1. 向上阶段（up-sweep / reduce phase）
     在向上阶段，算法类似于归并操作，将数组元素从下到上累积，通过分组计算来消除数据之间的依赖性。它的目标是通过二叉树结构的方式逐层进行**部分和的计算**。

  - **操作**：从底层的元素开始，依次合并相邻元素，形成二叉树结构。
  - 每一轮中，相邻元素会被加到一起，并放到更高层的节点位置，形成一个树形结构。最终，这个阶段会在根节点获得整个数组的总和。

  2. 向下阶段（down-sweep phase）
     在获得整个数组的总和后，进入向下阶段。这个阶段的目标是计算前缀和。该过程使用向上阶段的部分和的结果，通过从根节点向下传播和交换信息，最终计算出每个位置的前缀和。
     - **操作**：通过向下遍历二叉树，将部分和分配到下层节点。
     - 具体做法是：将根节点的值重置为零，然后逐层向下传播累积和，在每个子节点处，父节点的值和当前子节点的值一起决定子节点的前缀和。

  通过这两步操作，算法将前缀和分配到了所有的叶节点上，从而得到最终的前缀和数组。

- **算法的并行性**

Blelloch 算法的设计很好地利用了并行化：

1. 在**向上阶段**，每一层的节点计算可以并行进行，因为每个节点只依赖于它的两个子节点的值。
2. 在**向下阶段**，类似地，每层的节点可以同时计算，从而得到前缀和的分配。

由于这些操作是在树的层级结构上进行的，而树的高度为 \(O(\log n)\)，因此每一阶段的复杂度都是 \(O(\log n)\)。在一个完美的并行计算环境中，整个算法的并行时间复杂度为 \(O(\log n)\)，显著提高了计算大规模数据的效率。

- **小结**

- **Blelloch 并行扫描算法**通过两个阶段（向上阶段和向下阶段）实现了并行前缀和的高效计算。
- **向上阶段**累积部分和并构建二叉树，**向下阶段**通过传播和交换信息来计算每个位置的前缀和。
- 这种算法特别适合在 GPU、多核 CPU 等并行计算环境下处理大规模数据，且时间复杂度为 \(O(\log n)\)，比顺序计算的线性复杂度更适合并行化任务。

## 实现

- minGRU：

```python
# https://github.com/lucidrains/minGRU-pytorch/tree/main/minGRU_pytorch
# https://arxiv.org/abs/2410.01201v1

import torch
import torch.nn.functional as F
from torch.nn import Linear, Module

def exists(v):
    return v is not None

# appendix B
# https://github.com/glassroom/heinsen_sequence

def heinsen_associative_scan_log(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(dim = 1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim = 1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()

# appendix B.3

def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())

def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

# log-space version of minGRU - B.3.1
# they enforce the hidden states to be positive

class minGRU(Module):
    def __init__(self, dim):
        super().__init__()
        self.to_hidden_and_gate = Linear(dim, dim * 2, bias = False)

    def forward(self, x, prev_hidden = None):
        seq_len = x.shape[1]
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim = -1)

        # handle sequential

        if seq_len == 1:
            hidden = g(hidden)
            gate = gate.sigmoid()
            return torch.lerp(prev_hidden, hidden, gate) if exists(prev_hidden) else (hidden * gate)

        # parallel

        log_coeffs = -F.softplus(gate)

        log_z = -F.softplus(-gate)
        log_tilde_h = log_g(hidden)
        log_values = log_z + log_tilde_h

        if exists(prev_hidden):
            log_values = torch.cat((log_g(prev_hidden), log_values), dim = 1)
            log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

        out = heinsen_associative_scan_log(log_coeffs, log_values)
        return out[:, -seq_len:]

```

## Reference

1. [minGRU PyTorch Implementation](https://github.com/lucidrains/minGRU-pytorch/tree/main/minGRU_pytorch)
2. [CSDN Blog on RNNs](https://blog.csdn.net/qq_40994260/article/details/104233435)
3. [Zhihu Column on RNNs](https://zhuanlan.zhihu.com/p/61472450#:~:text=%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%20(Re)
