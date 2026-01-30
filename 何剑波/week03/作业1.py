# 作业1、使用GRU代替LSTM
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#1先读取数据集
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

#做映射
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]
#建立词表
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40

class CharGRUDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

#GRU Model Class
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()

        # 词表大小 -> embedding_dim：把每个字符ID映射成可训练的向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 可训练
        # GRU：相比 LSTM 少了 cell_state，返回 hidden_state
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)

        # GRU 返回 (output, h_n)
        # output: (batch, seq_len, hidden_dim)  每个时间步最后一层的输出
        # h_n:    (num_layers, batch, hidden_dim)  每层最后时间步的 hidden
        gru_out, hidden_state = self.gru(embedded)

        #取最后一层最后时间步的 hidden_state
        out = self.fc(hidden_state.squeeze(0))  # (batch, output_dim)
        return out

#Training and Prediction (GRU)
gru_dataset = CharGRUDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(gru_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

def classify_text_gru(text, model, char_to_index, max_len, index_to_label):
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

new_text = "帮我导航到北京"
predicted_class = classify_text_gru(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")



#作业1.2
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

#0. 复现实验
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#1. 数据读取与字符表
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]
index_to_label = {i: label for label, i in label_to_index.items()}

char_to_index = {'<pad>': 0}
for text in texts:
    for ch in text:
        if ch not in char_to_index:
            char_to_index[ch] = len(char_to_index)

vocab_size = len(char_to_index)
max_len = 40

class CharDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        ids = [self.char_to_index.get(ch, 0) for ch in text[:self.max_len]]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), self.labels[idx]

full_ds = CharDataset(texts, numerical_labels, char_to_index, max_len)

# train/test 切分（80/20）
train_size = int(0.8 * len(full_ds))
test_size = len(full_ds) - train_size
train_ds, test_ds = random_split(full_ds, [train_size, test_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

#2. 统一的 RNN/LSTM/GRU 分类器
class RecurrentClassifier(nn.Module):
    """
    cell_type: 'rnn' | 'lstm' | 'gru'
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 cell_type="lstm", num_layers=1, dropout=0.0):
        super().__init__()
        self.cell_type = cell_type.lower()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if self.cell_type == "rnn":
            self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        elif self.cell_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        else:
            raise ValueError("cell_type must be one of: 'rnn', 'lstm', 'gru'")

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: (N, L) -> embedding: (N, L, E)
        """
        emb = self.embedding(x)

        if self.cell_type == "lstm":
            out, (hn, cn) = self.rnn(emb)           # hn: (num_layers, N, H)
            last_h = hn[-1]                         # (N, H)
        else:
            out, hn = self.rnn(emb)                 # hn: (num_layers, N, H)
            last_h = hn[-1]                         # (N, H)

        logits = self.fc(last_h)                    # (N, C)
        return logits

#3. 训练 / 评估
def evaluate_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)

def train_one_model(cell_type, epochs=4, lr=1e-3):
    embedding_dim = 64
    hidden_dim = 128
    output_dim = len(label_to_index)

    model = RecurrentClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        cell_type=cell_type,
        num_layers=1,
        dropout=0.0
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            # 梯度裁剪：对梯度爆炸做一个优化
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

        train_acc = evaluate_accuracy(model, train_loader)
        test_acc  = evaluate_accuracy(model, test_loader)
        print(f"[{cell_type.upper()}] Epoch {ep}/{epochs} | loss={running_loss/len(train_loader):.4f} "
              f"| train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")

    return model, test_acc

#4. 三种循环层对比实验
results = {}
for ct in ["rnn", "lstm", "gru"]:
    set_seed(42)  # 保证可比性：每个模型从同样随机种子开始
    _, acc = train_one_model(ct, epochs=4, lr=1e-3)
    results[ct] = acc

print("\n Final Test Accuracy")
for k, v in results.items():
    print(f"{k.upper():<5}: {v:.4f}")



#实验结果对比
Final Test Accuracy 
RNN  : 0.1074
LSTM : 0.5388
GRU  : 0.8855
  
