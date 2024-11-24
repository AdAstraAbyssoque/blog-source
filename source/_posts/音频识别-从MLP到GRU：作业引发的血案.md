---
title: 音频识别-从MLP到GRU：作业引发的血案
date: 2024-10-03 17:20:38
tags: AIAA-2205
categories: 人工智能
mathjax: true
description: 本文详细介绍了从MLP到GRU在音频识别任务中的应用，包括支持向量机（SVM）、逻辑回归（LR）、多层感知器（MLP）、长短期记忆网络（LSTM）和门控循环单元（GRU）的实现和比较。
cover: https://ooo.0x0.ooo/2024/10/04/O46DH1.jpg
---

## 起因

> AIAA 2205 把作业发在了 Kaggle 上，并且作业计量需要按 Kaggle 的排名积分，于是大家开始内卷抢排名。

### 题面

给定 mfcc 文件夹，内含约 8000 个 39\*999 的 mfcc 矩阵，以`.mfcc.csv`格式标识,给定`labels`，`test_for_student.label`内命名为数个形如`HW00002897.mp4`的文件名，`trainval.csv`是形如下面的数据文件。

```csv
Id,Category
HW00002897,1
HW00001276,1
HW00000794,1
HW00001003,1
HW00003647,1
HW00001784,1
HW00007717,1
HW00007694,1
HW00001891,1
HW00007365,1
HW00007926,1
HW00001162,1
HW00002171,1
HW00002795,1
HW00005799,1
HW00001615,1
HW00007024,1
HW00003824,1
```

另附有`videos.name.lst`一个，这里我估计是助教处理数据的时候搞出来的坑有数据掉了，所以他输出了一个文件列表来掩饰自己的错误(bushi).

形如：

```csv
HW00000000
HW00000001
HW00000002
HW00000003
HW00000004
HW00000005
HW00000006
HW00000007
HW00000008
HW00000009
HW00000010
HW00000011
HW00000012
HW00000013
```

总之最后我们需要对`test_for_student.label`内的文件进行分类预测，原始 class 有 10 个。

需要的输出是：

```csv
Id,Category
HW00002897,1
HW00001276,1
HW00000794,1
HW00001003,1
HW00003647,1
HW00001784,1
HW00007717,1
HW00007694,1
HW00001891,1
HW00007365,1
HW00007926,1
HW00001162,1
HW00002171,1
```

### 内卷

提交的情况是前面的学生发疯了一样疯狂提交，因为说不定第二天起来就发现自己掉了十几二十名，最前面的 10 个人像是有默契的配合，有人上来把自己超过便会拿出自己压箱底的准备提交上去让排名恢复原状。

据我的大佬舍友 A 的讲述，他准备不要再受这样的内卷纷扰了，准备在最后一天交一次直接砍下第一。（这样不是更吓人吗）

## 解法

其实想听一听其他人的想法，但是可能得成绩出来之后才能同步了。

### SVM，LR

先写一点和神经网络与深度学习无关的方法，这些属于是前置基础，也是课程需要我们理解的范畴。

虽然看上去可能 Acc 不够高，但是有必要拿出来复习。

#### SVM

先复习一下支持向量机的原理：

支持向量机（Support Vector Machine, SVM）是一种监督学习模型，通常用于分类和回归分析。SVM 的目标是找到一个最佳的超平面，以最大化类别之间的间隔。以下是 SVM 的基本原理：

1. **超平面**：在 n 维空间中，超平面是一个 n-1 维的子空间。对于二分类问题，SVM 试图找到一个将数据点分开的超平面。

2. **支持向量**：支持向量是离超平面最近的数据点。这些点对超平面的位置有直接影响。SVM 通过这些支持向量来定义和构建超平面。

3. **间隔最大化**：SVM 的核心思想是找到一个超平面，使得两类数据点之间的间隔（即支持向量到超平面的距离）最大化。这个间隔被称为“margin”。

4. **线性可分与核函数**：对于线性可分的数据，SVM 可以直接找到一个线性超平面。然而，对于非线性可分的数据，SVM 使用核函数（如多项式核、径向基函数等）将数据映射到更高维的空间，使其在该空间中线性可分。

5. **优化问题**：SVM 的训练过程可以归结为一个凸优化问题，通常通过拉格朗日乘子法和二次规划来求解。

以下是 SVM 的数学表达式：

- **目标函数**：最大化间隔
  $$\text{maximize} \quad \frac{2}{\|\mathbf{w}\|}$$
  其中，$\mathbf{w}$是超平面的法向量。

- **约束条件**：确保所有数据点被正确分类
  $$y_i (\mathbf{w} \cdot \mathbf{x}\_i + b) \geq 1, \quad \forall i$$
  其中，$y_i$ 是数据点的标签，$\mathbf{x}\_i$ 是数据点的特征向量，$b$ 是偏置项。

通过求解上述优化问题，SVM 可以找到最佳的超平面，从而实现分类任务。

样例代码(注意 cpu 不能直接跑 `mfcc` ，数据量太大了所以还挺麻烦的)：

```python
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义常量
MFCC_DIR = "/home/shan/workplace/24fall/Homework/AIAA-2205/hkustgz-aiaa-2205-hw-1-fall-2024/mfcc.tgz/mfcc"
TRAINVAL_LABEL = "/home/shan/workplace/24fall/Homework/AIAA-2205/hkustgz-aiaa-2205-hw-1-fall-2024/labels/trainval.csv"
TEST_LABEL = "/home/shan/workplace/24fall/Homework/AIAA-2205/hkustgz-aiaa-2205-hw-1-fall-2024/labels/test_for_student.label"
MODEL_DIR = "/home/shan/workplace/24fall/Homework/AIAA-2205/hkustgz-aiaa-2205-hw-1-fall-2024/models"
MAX_SEQ_LENGTH = 999  # 最大序列长度

# 定义 SVM 模型
class SVMClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SVMClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

# 定义数据集类
class MFCCDataset(Dataset):
    def __init__(self, label_file, mfcc_dir, is_test=False):
        self.mfcc_dir = mfcc_dir
        self.data = []
        self.labels = []

        with open(label_file, 'r') as f:
            if is_test:
                for line in f:
                    video_id = line.strip().split('.')[0]
                    self.data.append(video_id)
            else:
                reader = csv.reader(f)
                next(reader)  # 跳过标题行
                for row in reader:
                    video_id, label = row
                    self.data.append(video_id)
                    self.labels.append(int(label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_id = self.data[idx]
        mfcc_file = os.path.join(self.mfcc_dir, f"{video_id}.mfcc.csv")

        if not os.path.exists(mfcc_file):
            return None

        # 读取 MFCC 特征文件
        with open(mfcc_file, 'r') as f:
            mfcc = np.array([list(map(float, line.strip().split(';'))) for line in f])

        # 预处理: 取平均值作为特征
        mfcc_mean = np.mean(mfcc, axis=0)
        mfcc_tensor = torch.FloatTensor(mfcc_mean)

        if len(self.labels) > 0:
            label = self.labels[idx]
            return mfcc_tensor, label
        else:
            return mfcc_tensor, video_id

# 数据加载函数
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

# 训练函数
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar.set_postfix({
            'Loss': f'{total_loss/(batch_idx+1):.3f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })

    return total_loss / len(train_loader), 100. * correct / total

# 验证函数
def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    average_loss = total_loss / len(val_loader)
    return average_loss, accuracy

# 推理函数
def inference(model, test_loader):
    model.eval()
    predictions = {}

    with torch.no_grad():
        for inputs, video_ids in tqdm(test_loader, desc='Inference'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            for video_id, pred in zip(video_ids, predicted.cpu().numpy()):
                predictions[video_id] = int(pred)

    return predictions

# 绘制学习曲线
def plot_learning_curves(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Learning Curves - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Learning Curves - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'svm_learning_curves.png'))
    plt.close()

# 主函数
def main():
    # 加载数据
    full_dataset = MFCCDataset(TRAINVAL_LABEL, MFCC_DIR)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 创建模型
    input_size = 39  # MFCC特征维度
    num_classes = 10
    model = SVMClassifier(input_size, num_classes).to(device)

    # 定义损失函数和优化器
    criterion = nn.MultiMarginLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # 训练模型
    num_epochs = 100
    best_val_acc = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch {epoch}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'svm_model_best.pth'))
            print(f'Model saved with validation accuracy: {best_val_acc:.2f}%')

    # 绘制学习曲线
    plot_learning_curves(train_losses, val_losses, train_accs, val_accs)

    # 加载测试数据并进行推理
    test_dataset = MFCCDataset(TEST_LABEL, MFCC_DIR, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'svm_model_best.pth')))
    predictions = inference(model, test_loader)

    # 保存预测结果
    output_file = os.path.join(MODEL_DIR, 'test_predictions_SVM.csv')
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Category'])
        for video_id, label in predictions.items():
            writer.writerow([f"{video_id}", label])

    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()
```

准确率图片准备后边再插入，这个还蛮不好跑的……大致准确率(`mfcc` 原始数据)在 25% ~ 30%左右，如果先做成 bof 再跑通过调参能到 40%.

#### LR

先回忆一下 LR 的原理：

逻辑回归是一种用于分类问题的统计模型，尽管名字中带有“回归”，但它主要用于二分类任务。其核心思想是通过一个逻辑函数（Sigmoid 函数）将线性回归的输出映射到一个概率值，从而实现分类。

1. 线性模型
   逻辑回归首先构建一个线性模型：
   $$ z = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b $$
  其中，$ w $ 是权重，$ x $ 是输入特征，$ b $ 是偏置。

2. Sigmoid 函数
   然后将线性模型的输出 $ z $ 通过一个 Sigmoid 函数映射到一个 0 到 1 之间的概率值：
   $$ \sigma(z) = \frac{1}{1 + e^{-z}} $$
   这个函数的输出可以解释为样本属于某一类的概率。

3. 损失函数
   逻辑回归使用对数损失函数（Log Loss）来衡量模型的预测与实际标签之间的差异：
   $$ L(y, \hat{y}) = - \left( y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right) $$
  其中，$ y $ 是实际标签，$ \hat{y} $ 是预测概率。

4. 参数优化
   通过梯度下降法或其他优化算法，最小化损失函数以找到最优的权重和偏置参数。

解决代码：

```python
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 设置常量
MFCC_DIR = "/home/shan/workplace/24fall/Homework/AIAA-2205/hkustgz-aiaa-2205-hw-1-fall-2024/mfcc.tgz/mfcc"
TRAINVAL_LABEL = "/home/shan/workplace/24fall/Homework/AIAA-2205/hkustgz-aiaa-2205-hw-1-fall-2024/labels/trainval.csv"
TEST_LABEL = "/home/shan/workplace/24fall/Homework/AIAA-2205/hkustgz-aiaa-2205-hw-1-fall-2024/labels/test_for_student.label"
MODEL_DIR = "/home/shan/workplace/24fall/Homework/AIAA-2205/hkustgz-aiaa-2205-hw-1-fall-2024/models"

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据集类
class MFCCDataset(Dataset):
    def __init__(self, label_file, mfcc_dir, is_test=False):
        self.mfcc_dir = mfcc_dir
        self.data = []
        self.labels = []
        self.ids = []

        with open(label_file, 'r') as f:
            if is_test:
                for line in f:
                    video_id = line.strip().split('.')[0]
                    self.ids.append(video_id)
            else:
                reader = csv.reader(f)
                next(reader)  # 跳过标题行
                for row in reader:
                    video_id, label = row
                    self.ids.append(video_id)
                    self.labels.append(int(label))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        video_id = self.ids[idx]
        mfcc_file = os.path.join(self.mfcc_dir, f"{video_id}.mfcc.csv")

        if not os.path.exists(mfcc_file):
            print(f"Warning: MFCC file not found for {video_id}")
            return None

        with open(mfcc_file, 'r') as f:
            mfcc = np.array([list(map(float, line.strip().split(';'))) for line in f])

        # 使用平均值和标准差作为特征
        mfcc_mean = np.mean(mfcc, axis=0)
        mfcc_std = np.std(mfcc, axis=0)
        features = np.concatenate((mfcc_mean, mfcc_std))

        if len(self.labels) > 0:
            label = self.labels[idx]
            return torch.FloatTensor(features), label, video_id
        else:
            return torch.FloatTensor(features), video_id

# 定义逻辑回归模型
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels, _ in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, _ in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# 推理函数
def inference(model, test_loader, device):
    model.eval()
    predictions = {}

    with torch.no_grad():
        for inputs, video_ids in tqdm(test_loader, desc="Inferencing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            for video_id, pred in zip(video_ids, predicted.cpu().numpy()):
                predictions[video_id] = int(pred)

    return predictions

# 绘制学习曲线
def plot_learning_curves(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Learning Curves - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Learning Curves - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'lr_learning_curves.png'))
    plt.close()

# 主函数
def main():
    # 加载训练数据
    train_dataset = MFCCDataset(TRAINVAL_LABEL, MFCC_DIR)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 计算输入特征的维度
    input_size = train_dataset[0][0].shape[0]
    num_classes = len(set(train_dataset.labels))

    # 初始化模型
    model = LogisticRegression(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 50
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, train_loader, criterion, device)  # 使用训练集作为验证集

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # 绘制学习曲线
    plot_learning_curves(train_losses, val_losses, train_accs, val_accs)

    # 加载测试数据
    test_dataset = MFCCDataset(TEST_LABEL, MFCC_DIR, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 进行预测
    predictions = inference(model, test_loader, device)

    # 保存预测结果
    output_file = os.path.join(MODEL_DIR, 'test_predictions_LR.csv')
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Category'])
        for video_id, pred in predictions.items():
            writer.writerow([f"{video_id}.mp4", pred])

    print(f"Predictions saved to {output_file}")
    print(f"Final training accuracy: {train_accs[-1]:.2f}%")
    print(f"Final validation accuracy: {val_accs[-1]:.2f}%")

if __name__ == "__main__":
    main()
```

### MLP

待更新

### LSTM

待更新

### GRU

#### GRU 简介

可参考：
[经典必读：门控循环单元（GRU）的基本概念与原理](https://www.jiqizhixin.com/articles/2017-12-24#:~:text=%E7%BB%8F%E5%85%B8%E5%BF%85%E8%AF%BB%EF%BC%9A%E9%97%A8%E6%8E%A7%E5%BE%AA%E7%8E%AF%E5%8D%95)

#### GRU 解法（加 K 折）

```python
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
import multiprocessing
import copy
from datetime import datetime
import matplotlib.pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义常量
MFCC_DIR = "/home/shan/workplace/24fall/Homework/AIAA-2205/hkustgz-aiaa-2205-hw-1-fall-2024/mfcc.tgz/mfcc"
TRAINVAL_LABEL = "/home/shan/workplace/24fall/Homework/AIAA-2205/hkustgz-aiaa-2205-hw-1-fall-2024/labels/trainval.csv"
TEST_LABEL = "/home/shan/workplace/24fall/Homework/AIAA-2205/hkustgz-aiaa-2205-hw-1-fall-2024/labels/test_for_student.label"
MODEL_DIR = "/home/shan/workplace/24fall/Homework/AIAA-2205/hkustgz-aiaa-2205-hw-1-fall-2024/models"
MAX_SEQ_LENGTH = 999  # 最大序列长度
NUM_FOLDS = 10  # K-折交叉验证的折数

# 定义 GRU 模型
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# 定义数据集类
class MFCCDataset(Dataset):
    def __init__(self, label_file, mfcc_dir, is_test=False):
        self.mfcc_dir = mfcc_dir
        self.data = []
        self.labels = []

        with open(label_file, 'r') as f:
            if is_test:
                for line in f:
                    video_id = line.strip().split('.')[0]
                    self.data.append(video_id)
            else:
                reader = csv.reader(f)
                next(reader)  # 跳过标题行
                for row in reader:
                    video_id, label = row
                    self.data.append(video_id)
                    self.labels.append(int(label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_id = self.data[idx]
        mfcc_file = os.path.join(self.mfcc_dir, f"{video_id}.mfcc.csv")

        if not os.path.exists(mfcc_file):
            return None

        # 读取 MFCC 特征文件
        with open(mfcc_file, 'r') as f:
            mfcc = np.array([list(map(float, line.strip().split(';'))) for line in f])

        # 预处理: 填充或截断
        if mfcc.shape[0] < MAX_SEQ_LENGTH:
            pad_length = MAX_SEQ_LENGTH - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_length), (0, 0)), mode='constant')
        elif mfcc.shape[0] > MAX_SEQ_LENGTH:
            mfcc = mfcc[:MAX_SEQ_LENGTH, :]

        mfcc_tensor = torch.FloatTensor(mfcc)

        if len(self.labels) > 0:
            label = self.labels[idx]
            return mfcc_tensor, label
        else:
            return mfcc_tensor, video_id

# 数据加载函数
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

# 训练函数
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar.set_postfix({
            'Loss': f'{total_loss/(batch_idx+1):.3f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })

    return total_loss / len(train_loader), 100. * correct / total

# 验证函数
def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    average_loss = total_loss / len(val_loader)
    return average_loss, accuracy

# 推理函数
def inference(model, test_loader):
    model.eval()
    predictions = {}

    with torch.no_grad():
        for inputs, video_ids in tqdm(test_loader, desc='Inference'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            for video_id, pred in zip(video_ids, predicted.cpu().numpy()):
                predictions[video_id] = int(pred)

    return predictions

# 绘制学习曲线
def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, fold):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Learning Curves - Loss (Fold {fold})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title(f'Learning Curves - Accuracy (Fold {fold})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f'learning_curves_fold{fold}.png'))
    plt.close()

# 训练和验证函数
def train_and_validate(fold, train_ids, val_ids, full_dataset, input_size, hidden_size, num_layers, num_classes):
    print(f'FOLD {fold}')
    print('--------------------------------')

    # 创建数据加载器
    train_subsampler = SubsetRandomSampler(train_ids)
    val_subsampler = SubsetRandomSampler(val_ids)

    train_loader = DataLoader(full_dataset, batch_size=32, sampler=train_subsampler, collate_fn=collate_fn)
    val_loader = DataLoader(full_dataset, batch_size=32, sampler=val_subsampler, collate_fn=collate_fn)

    model = GRUClassifier(input_size, hidden_size, num_layers, num_classes, dropout=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # 添加L2正则化

    # 训练模型
    num_epochs = 60  # 增加epoch数量
    best_val_acc = 0
    best_model = None
    patience = 20
    no_improve = 0

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    # 初始化标志变量
    start_saving = False

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch {epoch}: '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
            f'Fold {fold}')

        # 检查是否开始保存模型
        if train_loss < 0.5:
            start_saving = True

        if start_saving:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(model)
                print('Model updated on fold {fold}, validation accuracy rises to {best_val_acc:.2f}%')
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    # 保存最佳模型
    torch.save(best_model.state_dict(), os.path.join(MODEL_DIR, f'gru_model_fold{fold}.pth'))

    # 绘制学习曲线
    plot_learning_curves(train_losses, val_losses, train_accs, val_accs, fold)

    print(f'Best validation accuracy for fold {fold}: {best_val_acc:.2f}%')
    print('--------------------------------')

    return best_val_acc, train_accs[-1], val_losses[-1], train_losses[-1]

# 主函数
def main():
    # 加载数据
    full_dataset = MFCCDataset(TRAINVAL_LABEL, MFCC_DIR)

    # 设置交叉验证
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    # 创建模型参数
    input_size = 39
    hidden_size = 256
    num_layers = 2
    num_classes = 10

    # 创建进程池
    with multiprocessing.Pool(processes=5) as pool:
        results = []
        for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset), 1):
            result = pool.apply_async(train_and_validate, (fold, train_ids, val_ids, full_dataset, input_size, hidden_size, num_layers, num_classes))
            results.append(result)

        # 等待所有进程完成
        best_models = []
        for result in results:
            val_acc, train_acc, val_loss, train_loss = result.get()
            best_models.append((fold, val_acc, train_acc, val_loss, train_loss))

    # 选择最佳模型：考虑验证准确率、训练和验证准确率的差异（以检测过拟合），以及损失
    best_model = max(best_models, key=lambda x: x[1] - abs(x[1] - x[2]) - x[3])
    print(f"Selected model from fold {best_model[0]} with val_acc {best_model[1]:.2f}%")

    # 加载测试数据并进行推理
    test_dataset = MFCCDataset(TEST_LABEL, MFCC_DIR, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 使用所有折的模型进行集成预测
    ensemble_predictions = {}
    for fold in range(1, NUM_FOLDS + 1):
        model = GRUClassifier(input_size, hidden_size, num_layers, num_classes, dropout=0.5).to(device)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'gru_model_fold{fold}.pth')))
        fold_predictions = inference(model, test_loader)

        for video_id, pred in fold_predictions.items():
            if video_id not in ensemble_predictions:
                ensemble_predictions[video_id] = []
            ensemble_predictions[video_id].append(pred)

    # 取多数投票作为最终预测
    final_predictions = {video_id: max(set(preds), key=preds.count)
                        for video_id, preds in ensemble_predictions.items()}

    # 保存预测结果
    # 获取当前时间并格式化为字符串
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(MODEL_DIR, f'test_predictions_{timestamp}_GRU2.csv')

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Category'])
        for video_id, label in final_predictions.items():
            writer.writerow([f"{video_id}", label])

    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()
```

### 其他尝试

试过 Ada 优化的 GRU,BiGRU，TCN 等，由于任务匹配性问题不能靠近正常的 GRU 较高的 Acc，目前还在用 GRU。

## 总结

待更新
