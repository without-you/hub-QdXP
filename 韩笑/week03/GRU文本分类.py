import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

"""
数据准备部分
- 加载原始文本数据
- 构建标签映射字典
- 构建字符到索引的映射
"""
#初始化数据集
dataset = pd.read_csv("../week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()
# 将字符串标签转换为数字索引
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 构建字符到索引的映射，包含padding标记
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)
max_len = 40  # 设置最大文本长度

"""
自定义数据集类
继承自PyTorch的Dataset基类，用于数据加载
"""
class CharGRUDataset(Dataset):
    """
    字符级GRU数据集类
    功能：将文本数据转换为模型可用的张量格式
    """

    def __init__(self, texts, labels, char_to_index, max_len):
        """
        初始化数据集
        :param texts: 文本列表
        :param labels: 标签列表
        :param char_to_index: 字符到索引的映射
        :param max_len: 最大文本长度
        """
        self.texts = texts  # 存储所有文本
        self.labels = torch.tensor(labels, dtype=torch.long)  # 将标签转换为张量
        self.char_to_index = char_to_index  # 字符索引映射
        self.max_len = max_len  # 最大长度限制

    def __len__(self):
        """返回数据集的样本数量"""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        获取单个样本
        :param idx: 样本索引
        :return: (字符索引张量, 标签张量)
        """
        text = self.texts[idx]  # 获取对应文本
        # 将文本转换为字符索引列表，截断到最大长度
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        # 如果文本长度不足，则用padding索引填充
        indices += [0] * (self.max_len - len(indices))
        # 转换为张量并返回
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


"""
GRU模型定义
使用GRU（门控循环单元）作为循环神经网络层
"""


class GRUClassifier(nn.Module):
    """
    GRU文本分类器
    结构：嵌入层 -> GRU层 -> 全连接层
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        """
        初始化GRU分类器
        :param vocab_size: 词汇表大小
        :param embedding_dim: 词嵌入维度
        :param hidden_dim: GRU隐藏层维度
        :param output_dim: 输出维度（类别数量）
        :param num_layers: GRU层数
        :param dropout: dropout比率
        """
        super(GRUClassifier, self).__init__()

        # 词嵌入层：将字符索引转换为向量表示
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # GRU层：门控循环单元，用于处理序列数据
        # batch_first=True 表示输入张量的第一个维度是batch_size
        self.gru = nn.GRU(
            embedding_dim,  # 输入维度
            hidden_dim,  # 隐藏层维度
            num_layers=num_layers,  # GRU层数
            batch_first=True,  # 批次维度在最前面
            dropout=dropout if num_layers > 1 else 0,  # 多层时添加dropout
            bidirectional=False  # 单向GRU
        )

        # 全连接层：将GRU输出映射到类别空间
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Dropout层：防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 批归一化层：加速训练并提高稳定性
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量 (batch_size, seq_len)
        :return: 分类输出 (batch_size, output_dim)
        """
        # 步骤1: 将字符索引转换为嵌入向量
        # 输入: (batch_size, seq_len) -> 输出: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)

        # 步骤2: 通过GRU层处理序列
        # GRU输出: (batch_size, seq_len, hidden_dim)
        # 隐藏状态: (num_layers, batch_size, hidden_dim)
        gru_out, hidden = self.gru(embedded)

        # 步骤3: 取最后一个时间步的输出作为整个序列的表示
        # 从GRU输出中提取最后一个时间步的隐藏状态
        # (batch_size, seq_len, hidden_dim) -> (batch_size, hidden_dim)
        last_output = gru_out[:, -1, :]

        # 步骤4: 应用批归一化
        last_output = self.batch_norm(last_output)

        # 步骤5: 通过全连接层得到最终分类输出
        # (batch_size, hidden_dim) -> (batch_size, output_dim)
        output = self.fc(self.dropout(last_output))

        return output


"""
数据加载和模型训练
"""
# 创建数据集实例
gru_dataset = CharGRUDataset(texts, numerical_labels, char_to_index, max_len)
# 创建数据加载器，用于批量加载数据
dataloader = DataLoader(gru_dataset, batch_size=min(16, len(texts)), shuffle=True)

# 模型超参数设置
embedding_dim = 128  # 词嵌入维度
hidden_dim = 256  # GRU隐藏层维度
output_dim = len(label_to_index)  # 输出维度（类别数量）
num_layers = 2  # GRU层数

# 创建GRU模型实例
model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers)

# 损失函数：交叉熵损失，适用于分类任务
criterion = nn.CrossEntropyLoss()
# 优化器：Adam，自动调整学习率
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
# 学习率调度器：定期降低学习率
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

# 打印模型信息
print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"词汇表大小: {vocab_size}")
print(f"类别数量: {output_dim}")
print(f"使用模型: GRU")

"""
训练循环
"""
num_epochs = 10  # 训练轮数
for epoch in range(num_epochs):
    print(f"\n开始第 {epoch + 1} 轮训练...")

    # 设置模型为训练模式
    model.train()
    running_loss = 0.0  # 累计损失
    correct_predictions = 0  # 正确预测数量
    total_samples = 0  # 总样本数量

    # 遍历数据加载器
    for idx, (inputs, labels) in enumerate(dataloader):
        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 梯度裁剪：防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        optimizer.step()

        # 累计损失
        running_loss += loss.item()

        # 计算准确率
        _, predicted = torch.max(outputs, 1)  # 获取预测类别
        correct_predictions += (predicted == labels).sum().item()  # 计算正确预测数
        total_samples += labels.size(0)  # 累计总样本数

        # 定期打印训练状态
        if idx % max(1, len(dataloader) // 4) == 0:
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0
            print(f"Epoch {epoch + 1}, Batch {idx}: Loss={loss.item():.4f}, Acc={accuracy:.4f}")

    # 计算平均损失和准确率
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_predictions / total_samples

    # 更新学习率
    scheduler.step()

    # 打印本轮训练结果
    print(
        f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    print("-" * 70)

"""
预测函数
用于对新文本进行分类预测
"""


def classify_text_gru(text, model, char_to_index, max_len, index_to_label):
    """
    对单个文本进行分类预测
    :param text: 输入文本
    :param model: 训练好的模型
    :param char_to_index: 字符到索引的映射
    :param max_len: 最大文本长度
    :param index_to_label: 索引到标签的映射
    :return: (预测类别, 置信度)
    """
    # 将文本转换为字符索引
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    # 填充到最大长度
    indices += [0] * (max_len - len(indices))
    # 转换为张量并添加批次维度
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    # 设置模型为评估模式
    model.eval()
    with torch.no_grad():  # 不计算梯度，节省内存
        # 前向传播得到输出
        output = model(input_tensor)
        # 计算softmax概率
        probabilities = torch.softmax(output, dim=1)
        # 获取最大概率及其索引
        confidence, predicted_index = torch.max(probabilities, 1)
        predicted_index = predicted_index.item()
        predicted_label = index_to_label[predicted_index]
        confidence_score = confidence.item()

    return predicted_label, confidence_score


# 构建索引到标签的映射
index_to_label = {i: label for label, i in label_to_index.items()}

"""
测试预测
"""
test_texts = [
    "帮我导航到北京",
    "查询明天北京的天气",
    "播放音乐",
    "关闭闹钟",
    "设置提醒"
]

print("\n测试结果:")
for test_text in test_texts:
    predicted_class, confidence = classify_text_gru(test_text, model, char_to_index, max_len, index_to_label)
    print(f"输入: '{test_text}' -> 预测: '{predicted_class}', 置信度: {confidence:.4f}")

# GRU vs LSTM 特点说明
print("\n" + "=" * 60)
print("GRU (Gated Recurrent Unit) 特点:")
print("• 参数比LSTM少，计算效率更高")
print("• 包含重置门和更新门，能更好地捕捉长期依赖")
print("• 在许多任务中表现与LSTM相当甚至更好")
print("• 更适合处理较短到中等长度的序列")
print("=" * 60)
