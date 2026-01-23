
# m第二周作业
# 1、调整 09_深度学习文本分类.py 代码中模型的层数和节点个数，对比模型的loss变化。

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 加载数据集
dataset = pd.read_csv('../Week01/dataset.csv', sep='\t', header=None)
texts = dataset[0].tolist()
# texts ['还有双鸭山到淮阴的汽车票吗13号的', '从这里怎么回家', '随便播放一首专辑阁楼里的佛里的歌', '给看一下墓王之王嘛', '我想看挑战两把s686打突变团竞的游戏视频']
string_labels = dataset[1].tolist()
# string_labels ['Travel-Query', 'Travel-Query', 'Music-Play', 'FilmTele-Play', 'Video-Play']

label_to_index = {lable: i for i, lable in enumerate(set(string_labels))}  # 制作标签：序号字典
# label_to_index {'Travel-Query': 0, 'Music-Play': 1, 'FilmTele-Play': 2, 'Video-Play': 3}
# 注意：此处的 set 去重，不可单独写出来再赋给string_labels，会造成 labels长度 和 vocab_size 不同而报错

numeric_labels = [label_to_index[one_lable] for one_lable in string_labels]
# [3, 3, 0, 1, 2] # 获取标签对应的纯序号


# 抽取字符集 去重 并加上对应编号 自增
char_to_index = {'<pad>': 0, '<unk>': 1}  # 通常还有一个<unk>:1来表示未知字符

for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
# char_to_index {'<pad>': 0, '还': 1, '有': 2, '双': 3, '鸭': 4, '山': 5, '到': 6, '淮': 7, '阴': 8, '的': 9, '汽': 10, '车': 11, '票': 12, '吗': 13, '1': 14, '3': 15, '号': 16, '从': 17, '这': 18, '里': 19, '怎': 20, '么': 21, '回': 22, '家': 23, '随': 24, '便': 25, '播': 26, '放': 27, '一': 28, '首': 29, '专': 30, '辑': 31, '阁': 32, '楼': 33, '佛': 34, '歌': 35, '给': 36, '看': 37, '下': 38, '墓': 39, '王': 40, '之': 41, '嘛': 42, '我': 43, '想': 44, '挑': 45, '战': 46, '两': 47, '把': 48, 's': 49, '6': 50, '8': 51, '打': 52, '突': 53, '变': 54, '团': 55, '竞': 56, '游': 57, '戏': 58, '视': 59, '频': 60}

# 转换成 序号:字 的形式
index_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(index_char)  # 计算总词表数量（也就是总字出现次数，含pad和unk）


max_len = 60  # 单句填充总长度，40、60 有影响。 40 听音乐的预测结果是 Radio-Listen，60 的预测结果是 Music-Play


class CharBowDataset(Dataset):
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


# 创建简单分类器
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        # 层初始化
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 手动实现每层的计算
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        x = self.fc3(x)
        return x


char_dataset = CharBowDataset(texts, numeric_labels, char_to_index, max_len, vocab_size)  # 读取单个样本
dataLoader = DataLoader(char_dataset, batch_size=32, shuffle=True)   # 读取批量数据集

hidden_dim = 128
output_dim = len(label_to_index)

model = SimpleClassifier(vocab_size, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for idx, (input, labels) in enumerate(dataLoader):
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch个数{idx}, 当前的batch loss：{loss.item()}")

    print(f"epoch[{epoch+1/num_epochs}], Loss:{running_loss / len(dataLoader):.4f}")


def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vetor = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vetor[index] += 1

    bow_vetor = bow_vetor.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow_vetor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


# label_to_index: {'Music-Play': 0, 'Travel-Query': 1, 'FilmTele-Play': 2, 'Video-Play': 3}
index_to_label = {i: label for label, i in label_to_index.items()}
# print(index_to_label)
# {0: 'Travel-Query', 1: 'Video-Play', 2: 'Music-Play', 3: 'FilmTele-Play'}


new_text = "帮我导航到天安门"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入为{new_text}, 预测结果为{predicted_class}")


new_text = "想听音乐"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入为{new_text}, 预测结果为{predicted_class}")

# 损失函数已优化到 0 - 0.5 之间
# epoch[19.05], Loss:0.2984
# 输入为帮我导航到天安门, 预测结果为Travel-Query
# 输入为想听音乐, 预测结果为Music-Play
