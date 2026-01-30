import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 作业内容：调整层数，观察loss变化,vocab_size -> 256 -> 128 -> 64 -> output_dim

"""
概念：
张量就是深度学习里的 “万能数据容器” —— 把数字、列表、表格都装进这个容器里，模型才能看懂、才能高效运算、才能自动学习规律

张量就是 “能在 GPU 上高效运算的多维数组”
所有模型的输入、输出、参数（比如线性层的权重 W）都是张量
张量能记录自身的计算路径，反向传播时自动算出梯度
0 维 : tensor(1)
1 维 : tensor([0.,1.,1.,...])
2 维 : tensor([[0.,1.],[0.,0.]])
3 维 : 比如文本的词嵌入（序列长度 × 批次 × 维度）

bow_matrix = torch.tensor([
    [0.,1.,1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.,0.,1.,1.,1.,1.,1.,2.,1.,1.]
], dtype=torch.float32)
print("维度：", bow_matrix.ndim)  # 输出：2
print("形状：", bow_matrix.shape)  # 输出：torch.Size([2, 14])（2个样本，每个14维）
print("第1个样本的特征：", bow_matrix[0])  # 输出：1维张量（第1行）

"""

dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None, nrows=2)
# ["帮我导航到北京", "查询明天北京的天气"]
texts = dataset[0].tolist()
# 示例数字标签（0=Weather-Query，1=Navigation，这里故意对应：导航→1，天气→0）
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# {'<pad>':0, '帮':1, '我':2, '导':3, '航':4, '到':5, '北':6, '京':7, '查':8, '询':9, '明':10, '天':11, '的':12, '气':13}
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
# 字符表总大小（BoW向量维度）
vocab_size = len(char_to_index)

max_len = 40

"""
省略具体BoW向量生成逻辑（和之前一致），直接给出示例结果
        # 第0个文本“帮我导航到北京”的BoW向量：[0,1,1,1,1,1,1,1,0,0,0,0,0,0]
        # 第1个文本“查询明天北京的天气”的BoW向量：[0,0,0,0,0,0,1,1,1,1,1,2,1,1]
        return torch.tensor([
            [0,1,1,1,1,1,1,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,1,1,1,1,2,1,1]
        ], dtype=torch.float32)

        sample_0 = dataset[0]  # 等价于dataset.__getitem__(0)
"""
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

# (tensor([0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.]), tensor(1))
    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


"""
model(x)时，会自动执行 forward 函数,自动构建反向传播的计算图
训练时loss.backward()能自动算出每个参数（W1、b1、W2、b2）的梯度
nn.Linear:加权求和 + 偏置
"""
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        # 必须调用父类初始化
        super(SimpleClassifier, self).__init__()

        # 第一层: vocab_size -> 256
        self.fc1 = nn.Linear(input_dim, 256)
        # 第二层: 256 -> 128
        self.fc2 = nn.Linear(256, 128)
        # 第三层: 128 -> 64
        self.fc3 = nn.Linear(128, 64)
        # 输出层: 64 -> output_dim
        self.fc4 = nn.Linear(64, output_dim)

        # ReLU激活函数 y = max(0, x)
        self.relu = nn.ReLU()
        # Tanh激活函数
        self.tanh = nn.Tanh()

    def forward(self, x):
        # 第一层: [batch, vocab_size] -> [batch, 256]
        out = self.fc1(x)
        out = self.relu(out)

        # 第二层: [batch, 256] -> [batch, 128]
        out = self.fc2(out)
        out = self.relu(out)

        # 第三层: [batch, 128] -> [batch, 64]
        out = self.fc3(out)
        out = self.tanh(out)

        # 输出层: [batch, 64] -> [batch, output_dim]
        # 输出[32, 5] 比如第一个样本的得分是[9.2, 0.5, 0.3, 0.1, 0.2]，代表 "天气类" 得分最高
        out = self.fc4(out)
        return out

# 读取单个样本
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
# 读取批量数据集 -》 batch数据
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# 隐藏层有 128 个神经元，负责提取中间特征
hidden_dim = 128
# 输出 5 个类别的得分（天气、播放、闹钟、出行、其他）
output_dim = len(label_to_index)
model = SimpleClassifier(vocab_size, hidden_dim, output_dim)
# 损失函数 内部自带激活函数，softmax
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# epoch： 将数据集整体迭代训练一次
# batch： 数据集汇总为一批训练一次

num_epochs = 10
for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
    # 告诉模型：“现在要训练了，认真学！”
    model.train()
    # 记录当前Epoch的总误差（损失），用于后续看训练效果
    running_loss = 0.0

    """
    优化器（SGD）按照反向传播算出的梯度，更新模型的所有参数（比如把 “天” 字的权重从 0.1 调到 0.12）
    """
    for idx, (inputs, labels) in enumerate(dataloader):
        # 清空上一批数据的梯度，避免梯度累加导致调参错误
        optimizer.zero_grad()
        # 把批次数据喂给模型，得到预测结果
        outputs = model(inputs)

        # 对比预测结果和真实标签，算误差
        loss = criterion(outputs, labels)
        # 反向传播：计算每个参数（W1、b1、W2、b2）该怎么调
        loss.backward()

        # 执行调参：根据梯度调整模型的参数
        optimizer.step()
        # 比如第 1 批 loss=0.5，第 2 批 loss=0.4 → running_loss=0.9，最终能算出本轮的平均 loss（running_loss / 总批数）
        running_loss += loss.item()

        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


"""
text = "查询明天北京的天气"
char_to_index：字符→索引映射（<pad>:0、查:8、询:9、明:10、天:11、北:6、京:7、的:12、气:13）
index_to_label：索引→标签映射（0:'Weather-Query'、1:'Navigation'等）
model：训练好的SimpleClassifier
"""
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]] # [8,9,10,11,6,7,12,11,13]
    tokenized += [0] * (max_len - len(tokenized)) # [8,9,10,11,6,7,12,11,13,0,0,...,0]（共 40 个元素）

    # 初始化14维全0向量
    bow_vector = torch.zeros(vocab_size)

    # 遍历tokenized后，非 0 索引的计数：8 (查)→1、9 (询)→1、10 (明)→1、11 (天)→2、6 (北)→1、7 (京)→1、12 (的)→1、13 (气)→1 → 最终bow_vector为
    # [0,0,0,0,0,0,1,1,1,1,1,2,1,1]（14 维，索引 0 是填充符，计数为 0）
    for index in tokenized:
        if index != 0: # 跳过填充符0，只统计真实字符
            bow_vector[index] += 1 # 字符出现一次，对应位置计数+1

    bow_vector = bow_vector.unsqueeze(0) # 原形状[14] → 新形状[1,14]

    # 预测模式 —— 关闭训练时的 Dropout、BatchNorm 等逻辑（避免干扰预测结果）
    model.eval()
    # 预测时不需要调参，禁用梯度计算
    with torch.no_grad():
        output = model(bow_vector) # 模型输出类别得分，示例中形状是[1,5]（1 个样本，5 个类别的得分），比如：[[9.5, 0.2, 0.1, 0.3, 0.4]]

    # 在维度 1（类别维度）找最大值，返回两个值：(最大值, 最大值的索引)—— 我们只需要 “索引”（对应类别）
    _, predicted_index = torch.max(output, 1)
    # 把张量转成普通整数
    predicted_index = predicted_index.item()
    # 数字索引转文字标签
    predicted_label = index_to_label[predicted_index]

    return predicted_label


index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "北京今天几级风"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
