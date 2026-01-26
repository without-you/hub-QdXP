import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ----------------------------
# 1. 数据加载与预处理（保持不变）
# ----------------------------
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

vocab_size = len(char_to_index)
max_len = 40
output_dim = len(label_to_index)
index_to_label = {i: label for label, i in label_to_index.items()}
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        input_dim: 输入维度（vocab_size）
        hidden_dims: 隐藏层列表，如 [128, 64] 表示两层，分别128和64个神经元
        output_dim: 输出类别数
        """
        super(MLPClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_model(model, dataloader, num_epochs=10, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    model.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Model [{model.__class__.__name__}] Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return epoch_losses

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

# 创建 dataset 和 dataloader（只需一次）
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# 定义实验配置：(模型名称, 隐藏层结构)
configs = [
    ("1-layer-64", [64]),
    ("1-layer-128", [128]),
    ("2-layer-128-64", [128, 64]),
    ("3-layer-256-128-64", [256, 128, 64]),
    ("2-layer-64-64", [64, 64]),
]

results = {}

for name, hidden_dims in configs:
    print(f"\n{'=' * 50}")
    print(f"Training model: {name}")
    print(f"{'=' * 50}")

    model = MLPClassifier(vocab_size, hidden_dims, output_dim)
    losses = train_model(model, dataloader, num_epochs=15, lr=0.01)
    results[name] = losses

plt.figure(figsize=(12, 8))
for name, losses in results.items():
    plt.plot(losses, label=name, linewidth=2)

plt.title("Training Loss Comparison of Different MLP Architectures")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
