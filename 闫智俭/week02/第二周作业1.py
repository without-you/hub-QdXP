import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt  # 用于绘制loss曲线

# ... (数据加载和预处理部分保持不变) ...
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
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

class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]

# 创建数据集
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)

# 分割数据集为训练集和验证集（80% 训练，20% 验证）
dataset_size = len(char_dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(char_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

output_dim = len(label_to_index)

# 定义灵活的模型类，支持可变层数和节点数
class FlexibleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):  # hidden_dims为列表，定义每层节点数
        super(FlexibleClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())  # 使用ReLU激活函数防止梯度消失[1,7](@ref)
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))  # 输出层无激活函数（CrossEntropyLoss内含Softmax）
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 定义不同模型配置（层数和节点数）
configs = [
    {"name": "1层128节点", "hidden_dims": [128]},  # 原始配置作为基线
    {"name": "2层64节点", "hidden_dims": [64, 64]},  # 增加层数，减少每层节点数
    {"name": "1层256节点", "hidden_dims": [256]},  # 增加节点数，提升模型容量
    {"name": "3层64节点", "hidden_dims": [64, 64, 64]},  # 更深网络，学习更复杂特征[6](@ref)
    {"name": "1层512节点", "hidden_dims": [512]}  # 更宽网络，增强拟合能力
]

# 训练参数
num_epochs = 10
criterion = nn.CrossEntropyLoss()
results = {}  # 存储每个模型的loss历史

print("开始训练不同配置的模型...")
for config in configs:
    model_name = config["name"]
    hidden_dims = config["hidden_dims"]
    print(f"\n=== 训练模型: {model_name} ===")

    # 初始化模型和优化器
    model = FlexibleClassifier(vocab_size, hidden_dims, output_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # 固定学习率以公平对比

    train_losses = []  # 记录每个epoch的训练loss
    val_losses = []    # 记录每个epoch的验证loss

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], 训练Loss: {avg_train_loss:.4f}, 验证Loss: {avg_val_loss:.4f}")

    # 存储结果
    results[model_name] = {"train_losses": train_losses, "val_losses": val_losses}

# 绘制所有模型的loss变化曲线
plt.figure(figsize=(12, 6))
for model_name, loss_dict in results.items():
    plt.plot(loss_dict['train_losses'], label=f'{model_name}训练', marker='o', markersize=3)
    plt.plot(loss_dict['val_losses'], label=f'{model_name}验证', linestyle='--', marker='s', markersize=3)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('不同模型配置的Loss变化对比[7](@ref)')
plt.legend()
plt.grid(True)
plt.show()

# 使用基线模型进行预测（可选）
print("\n=== 使用基线模型(1层128节点)进行预测 ===")
baseline_model = FlexibleClassifier(vocab_size, [128], output_dim)
# 这里可以重新训练或直接使用已训练的模型，但为简化，仅演示预测流程
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))
    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1
    bow_vector = bow_vector.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(bow_vector)
    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]
    return predicted_label

index_to_label = {i: label for label, i in label_to_index.items()}
new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, baseline_model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")
