import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#初始化数据集
dataset = pd.read_csv("../week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)
max_len = 40

#自定义数据集
class CharLSTMDataset(Dataset):
    # 初始化
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts # 文本输入
        self.labels = torch.tensor(labels, dtype=torch.long) # 文本对应的标签
        self.char_to_index = char_to_index # 字符到索引的映射关系
        self.max_len = max_len # 文本最大输入长度

    # 返回数据集样本个数
    def __len__(self):
        return len(self.texts)

    # 获取当个样本
    def __getitem__(self, idx):
        text = self.texts[idx]
        # pad and crop
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

#RNN Model
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super(RNNClassifier, self).__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 使用多层RNN以增加模型表达能力
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # 只有多层时才使用dropout
        )
        # 输出层
        self.fc = nn.Linear(hidden_dim, output_dim)
        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.3)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)  # 添加批归一化

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # RNN前向传播
        rnn_out, hidden = self.rnn(embedded)  # rnn_out: (batch_size, seq_len, hidden_dim)

        # 取最后一个时间步的输出
        last_output = rnn_out[:, -1, :]  # (batch_size, hidden_dim)

        # 通过全连接层得到最终输出
        output = self.fc(self.dropout(last_output))  # (batch_size, output_dim)

        return output

#训练验证
rnn_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(rnn_dataset, batch_size=32, shuffle=True)
# 模型参数
embedding_dim = 128  # 增加嵌入维度
hidden_dim = 256     # 增加隐藏层维度
output_dim = len(label_to_index)
num_layers = 2       # 使用多层RNN
# 创建模型
model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()

        # 计算准确率
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        if idx % max(1, len(dataloader) // 4) == 0:
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0
            print(f"Epoch {epoch + 1}, Batch {idx}: Loss={loss.item():.4f}, Acc={accuracy:.4f}")

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_predictions / total_samples
    scheduler.step()  # 更新学习率

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    print("-" * 60)

def classify_text_lstm(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label
index_to_label = {i: label for label, i in label_to_index.items()}

if len(texts) > 0:
    new_text = "帮我导航到北京"
    predicted_class = classify_text_lstm(new_text, model, char_to_index, max_len, index_to_label)
    print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

    new_text_2 = "查询明天北京的天气"
    predicted_class_2 = classify_text_lstm(new_text_2, model, char_to_index, max_len, index_to_label)
    print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")