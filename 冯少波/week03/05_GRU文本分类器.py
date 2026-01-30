import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader

dataset = pd.read_csv("../Week01/dataset.csv", sep='\t', header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label:i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i:char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)
print(f"index to char : {index_to_char}")

max_len = 40

# 数据集类
class CharGRUDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts;
        self.labels = labels
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indics = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indics += [0] * (self.max_len - len(indics))
        return torch.tensor(indics, dtype=torch.long), self.labels[idx]


# GRU 模型类
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()

        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # GRU层
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        # 全链接
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden_state = self.gru(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out

# 训练和预测

gru_dataset = CharGRUDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(gru_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)

total_params = sum(p.numel() for p in model.parameters())
print(f"mode total params : {total_params}")

# 优化器和损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if idx % 50 == 0:
            print(f"Batch {idx}, cur batch loss : {loss.item():.4f}")

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(dataloader)

print(f"train success!", "+" * 60)

# 预测函数
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

# 测试
index_to_lable = {i:label for label, i in label_to_index.items()}

test_text = "帮我导航到北京"
predicted_label = classify_text_gru(test_text, model, char_to_index, max_len, index_to_lable)
print(f"input:{test_text}, predicted:{predicted_label}")

