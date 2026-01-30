import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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

class CharLSTMDataset(Dataset):
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

# --- NEW LSTM Model Class ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out

# GRU Model
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        #词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        #核心替换，使用GRU替代LSTM
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        #全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        #映射为稠密向量
        embedded = self.embedding(x)
        #传入GRU，GRU不像LSTM那样返回cell_state
        #gru_out包含所有时间步的输出，hidden_state包含最后一个时间步的状态
        gru_out, hidden_state = self.gru(embedded)
        #通常取最后一次迭代产生的隐藏状态做分类
        #hidden_state性状 (1, batch_size, hidden_dim) 需要去除第0维
        out = self.fc(hidden_state.squeeze(0))
        return out

#RNN Model
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        #词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        #核心替换 使用原始vanilla rnn
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        #全连接输出层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        #将输入字符索引转化为向量 (batch, seq_len, emb_dim)
        embedded = self.embedding(x)
        #传入RNN层 rnn_out-包含所有时间步的隐藏状态 (batch, seq_len, hidden_dim)
        #hidden:包含最后一个时间步的隐藏状态 (1, batch, hidden_dim)
        rnn_out, hidden = self.rnn(embedded)
        #取最后一个时间步的状态进行分类 ps：原生rnn只有hidden没有lstm的cell_stat
        out = self.fc(hidden.squeeze(0))
        return out

# --- Training and Prediction ---
lstm_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

# model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
# Batch 个数 0, 当前Batch Loss: 2.4847030639648438
# Batch 个数 50, 当前Batch Loss: 2.374105930328369
# Batch 个数 100, 当前Batch Loss: 2.323007822036743
# Batch 个数 150, 当前Batch Loss: 2.263131618499756
# Batch 个数 200, 当前Batch Loss: 2.2405474185943604
# Batch 个数 250, 当前Batch Loss: 2.248310089111328
# Batch 个数 300, 当前Batch Loss: 2.3570010662078857
# Batch 个数 350, 当前Batch Loss: 2.2714481353759766
# Epoch [1/4], Loss: 2.3616
# Batch 个数 0, 当前Batch Loss: 2.3663063049316406
# Batch 个数 50, 当前Batch Loss: 2.3782849311828613
# Batch 个数 100, 当前Batch Loss: 1.9986695051193237
# Batch 个数 150, 当前Batch Loss: 2.0666348934173584
# Batch 个数 200, 当前Batch Loss: 1.9186094999313354
# Batch 个数 250, 当前Batch Loss: 1.7312954664230347
# Batch 个数 300, 当前Batch Loss: 2.0061283111572266
# Batch 个数 350, 当前Batch Loss: 1.5375937223434448
# Epoch [2/4], Loss: 1.9661
# Batch 个数 0, 当前Batch Loss: 1.4553852081298828
# Batch 个数 50, 当前Batch Loss: 1.3629624843597412
# Batch 个数 100, 当前Batch Loss: 1.6523841619491577
# Batch 个数 150, 当前Batch Loss: 1.496173620223999
# Batch 个数 200, 当前Batch Loss: 1.5364339351654053
# Batch 个数 250, 当前Batch Loss: 0.9501150846481323
# Batch 个数 300, 当前Batch Loss: 0.9925174117088318
# Batch 个数 350, 当前Batch Loss: 1.6662886142730713
# Epoch [3/4], Loss: 1.3008
# Batch 个数 0, 当前Batch Loss: 1.07622492313385
# Batch 个数 50, 当前Batch Loss: 1.0474295616149902
# Batch 个数 100, 当前Batch Loss: 0.7542171478271484
# Batch 个数 150, 当前Batch Loss: 0.9715112447738647
# Batch 个数 200, 当前Batch Loss: 0.8008679151535034
# Batch 个数 250, 当前Batch Loss: 0.6904715299606323
# Batch 个数 300, 当前Batch Loss: 0.9482693672180176
# Batch 个数 350, 当前Batch Loss: 0.3472236096858978
# Epoch [4/4], Loss: 0.8120
# 输入 '帮我导航到北京' 预测为: 'Travel-Query'
# 输入 '查询明天北京的天气' 预测为: 'Weather-Query'

# model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
# 在本实验中，由于我们的文本分类任务（如导航、天气）句子较短，GRU 的收敛速度更明显
# Batch 个数 0, 当前Batch Loss: 2.5486204624176025
# Batch 个数 50, 当前Batch Loss: 2.536747694015503
# Batch 个数 100, 当前Batch Loss: 2.1681365966796875
# Batch 个数 150, 当前Batch Loss: 1.444897174835205
# Batch 个数 200, 当前Batch Loss: 1.314615249633789
# Batch 个数 250, 当前Batch Loss: 0.6308503150939941
# Batch 个数 300, 当前Batch Loss: 0.9597679376602173
# Batch 个数 350, 当前Batch Loss: 0.7027071118354797
# Epoch [1/4], Loss: 1.4039
# Batch 个数 0, 当前Batch Loss: 0.2437213808298111
# Batch 个数 50, 当前Batch Loss: 0.5778043270111084
# Batch 个数 100, 当前Batch Loss: 0.2745496332645416
# Batch 个数 150, 当前Batch Loss: 0.6582432985305786
# Batch 个数 200, 当前Batch Loss: 0.5284008979797363
# Batch 个数 250, 当前Batch Loss: 0.6060676574707031
# Batch 个数 300, 当前Batch Loss: 0.33377963304519653
# Batch 个数 350, 当前Batch Loss: 0.4205878973007202
# Epoch [2/4], Loss: 0.4648
# Batch 个数 0, 当前Batch Loss: 0.3228023946285248
# Batch 个数 50, 当前Batch Loss: 0.36206942796707153
# Batch 个数 100, 当前Batch Loss: 0.34067970514297485
# Batch 个数 150, 当前Batch Loss: 0.12919268012046814
# Batch 个数 200, 当前Batch Loss: 0.4160274267196655
# Batch 个数 250, 当前Batch Loss: 0.622061014175415
# Batch 个数 300, 当前Batch Loss: 0.3601953983306885
# Batch 个数 350, 当前Batch Loss: 0.1193523108959198
# Epoch [3/4], Loss: 0.2993
# Batch 个数 0, 当前Batch Loss: 0.15946882963180542
# Batch 个数 50, 当前Batch Loss: 0.13315868377685547
# Batch 个数 100, 当前Batch Loss: 0.11090995371341705
# Batch 个数 150, 当前Batch Loss: 0.2270546853542328
# Batch 个数 200, 当前Batch Loss: 0.10338885337114334
# Batch 个数 250, 当前Batch Loss: 0.08749720454216003
# Batch 个数 300, 当前Batch Loss: 0.07135896384716034
# Batch 个数 350, 当前Batch Loss: 0.12549465894699097
# Epoch [4/4], Loss: 0.2131
# 输入 '帮我导航到北京' 预测为: 'Travel-Query'
# 输入 '查询明天北京的天气' 预测为: 'Weather-Query'

model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
# RNN效果比较差
# Batch 个数 0, 当前Batch Loss: 2.451730251312256
# Batch 个数 50, 当前Batch Loss: 2.5444321632385254
# Batch 个数 100, 当前Batch Loss: 2.419220209121704
# Batch 个数 150, 当前Batch Loss: 2.362435817718506
# Batch 个数 200, 当前Batch Loss: 2.35455060005188
# Batch 个数 250, 当前Batch Loss: 2.399824857711792
# Batch 个数 300, 当前Batch Loss: 2.288004159927368
# Batch 个数 350, 当前Batch Loss: 2.3344779014587402
# Epoch [1/4], Loss: 2.3731
# Batch 个数 0, 当前Batch Loss: 2.419520378112793
# Batch 个数 50, 当前Batch Loss: 2.3287508487701416
# Batch 个数 100, 当前Batch Loss: 2.304445266723633
# Batch 个数 150, 当前Batch Loss: 2.4146735668182373
# Batch 个数 200, 当前Batch Loss: 2.25966739654541
# Batch 个数 250, 当前Batch Loss: 2.442767858505249
# Batch 个数 300, 当前Batch Loss: 2.3359549045562744
# Batch 个数 350, 当前Batch Loss: 2.359532356262207
# Epoch [2/4], Loss: 2.3616
# Batch 个数 0, 当前Batch Loss: 2.3005869388580322
# Batch 个数 50, 当前Batch Loss: 2.3065290451049805
# Batch 个数 100, 当前Batch Loss: 2.3168182373046875
# Batch 个数 150, 当前Batch Loss: 2.356201171875
# Batch 个数 200, 当前Batch Loss: 2.5013809204101562
# Batch 个数 250, 当前Batch Loss: 2.437488317489624
# Batch 个数 300, 当前Batch Loss: 2.3831396102905273
# Batch 个数 350, 当前Batch Loss: 2.3667526245117188
# Epoch [3/4], Loss: 2.3592
# Batch 个数 0, 当前Batch Loss: 2.2995169162750244
# Batch 个数 50, 当前Batch Loss: 2.4065349102020264
# Batch 个数 100, 当前Batch Loss: 2.3869378566741943
# Batch 个数 150, 当前Batch Loss: 2.4520535469055176
# Batch 个数 200, 当前Batch Loss: 2.410027503967285
# Batch 个数 250, 当前Batch Loss: 2.233109474182129
# Batch 个数 300, 当前Batch Loss: 2.373570203781128
# Batch 个数 350, 当前Batch Loss: 2.223752737045288
# Epoch [4/4], Loss: 2.3582
# 输入 '帮我导航到北京' 预测为: 'FilmTele-Play'
# 输入 '查询明天北京的天气' 预测为: 'FilmTele-Play'

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 4
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for idx, (inputs, labels) in enumerate(dataloader):
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         if idx % 50 == 0:
#             print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")
#
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

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

new_text = "帮我导航到北京"
predicted_class = classify_text_lstm(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_lstm(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")

# 对RNN、LSTM、GRU三者的Loss数据进行绘图对比
rnn_losses = [2.3731,2.3616,2.3592,2.3582]
lstm_losses = [2.3616,1.9661,1.3008,0.8120]
gru_losses = [1.4039,0.4648,0.2993,0.2131]

def plot_loss_comparison(rnn_loss, lstm_loss, gru_loss):
    epochs = range(1, len(rnn_loss) + 1)

    plt.figure(figsize=(10, 6))

    # 绘制三条曲线
    plt.plot(epochs, rnn_loss, 'r-o', label='Vanilla RNN')
    plt.plot(epochs, lstm_loss, 'b-s', label='LSTM')
    plt.plot(epochs, gru_loss, 'g-^', label='GRU')

    # 修饰图形
    plt.title('Training Loss Comparison: RNN vs LSTM vs GRU', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 保存并显示
    plt.savefig('loss_comparison.png')
    plt.show()

plot_loss_comparison(rnn_losses, lstm_losses, gru_losses)
