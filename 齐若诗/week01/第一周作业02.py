#模型一：使用支持向量机SVC
import jieba # 中文分词用途
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels

# 1. 数据准备
df = pd.read_csv("C:/Users/ruosh/nlp/nlp20/Week01/dataset.csv", sep='\t', names=['text', 'label'])
# 简单的去停用词
def clean_text(text):
    words = jieba.lcut(str(text))
    return " ".join([w for w in words if len(w) > 1])
df['text_cut'] = df['text'].apply(clean_text)

# 3. 特征提取，训练模型
X_train, X_test, y_train, y_test = train_test_split(df['text_cut'], df['label'], test_size=0.2, random_state=42)
tfidf = TfidfVectorizer(max_features=2000)
train_vectors = tfidf.fit_transform(X_train)
test_vectors = tfidf.transform(X_test)
print("模型 1 (sklearn - SVM) 正在训练...")
model_svm = SVC(kernel='linear')
model_svm.fit(train_vectors, y_train)
print("SVM 准确率:", model_svm.score(test_vectors, y_test))

#测试
test_queries = [
    "帮我播放一下郭德纲的小品",
    "今天北京的天气怎么样",
    "帮我定一个明天早上八点的闹钟"
]

print("-" * 30)
print("方案一：sklearn (SVM) 验证测试")
for query in test_queries:
    # 1. 分词
    test_sentence = " ".join(jieba.lcut(query))
    # 2. 向量化
    test_feature = tfidf.transform([test_sentence])
    # 3. 预测
    prediction = model_svm.predict(test_feature)

    print(f"待预测文本: {query}")
    print(f"SVM 预测结果: {prediction[0]}")
    print("-" * 10)

#模型二：使用torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. 将数据转换为 PyTorch 张量 (Tensor)
# 使用上面 sklearn 生成的 TF-IDF 矩阵作为输入
X_train_torch = torch.FloatTensor(train_vectors.toarray())
X_test_torch = torch.FloatTensor(test_vectors.toarray())

# 将标签转换为数字索引
labels = sorted(list(df['label'].unique()))
label_to_idx = {l: i for i, l in enumerate(labels)}
y_train_torch = torch.LongTensor([label_to_idx[l] for l in y_train])
y_test_torch = torch.LongTensor([label_to_idx[l] for l in y_test])

# 2. 定义神经网络结构
class TextClassifierNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TextClassifierNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # 输入层到隐藏层
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(128, num_classes)  # 隐藏层到输出层

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 3. 训练配置
input_dim = X_train_torch.shape[1]
output_dim = len(labels)
model = TextClassifierNet(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，适用于分类
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练循环
print("\n模型 2 (PyTorch - MLP) 正在训练...")
for epoch in range(50):
    outputs = model(X_train_torch)
    loss = criterion(outputs, y_train_torch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/50], Loss: {loss.item():.4f}')

# 5. 评估
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_torch)
    _, predicted = torch.max(test_outputs.data, 1)
    accuracy = (predicted == y_test_torch).sum().item() / y_test_torch.size(0)
    print(f"PyTorch 模型准确率: {accuracy:.4f}")

# 6.验证测试
print("-" * 50)
print("方案二：PyTorch (MLP) 验证测试")
with torch.no_grad():  # 预测时不需要计算梯度
    for query in test_queries:
        # 1. 预处理
        test_sentence = " ".join(jieba.lcut(query))
        test_feature = tfidf.transform([test_sentence]).toarray()
        # 2. 转为 Tensor 并预测
        input_tensor = torch.FloatTensor(test_feature)
        outputs = model(input_tensor)
        # 3. 获取最大概率的索引
        _, predicted_idx = torch.max(outputs, 1)
        # 4. 转换回原始标签名称
        # unique_labels 
        unique_labels = sorted(list(df['label'].unique()))
        predicted_label = unique_labels[predicted_idx.item()]

        print(f"待预测文本: {query}")
        print(f"PyTorch 预测结果: {predicted_label}")
        print("-" * 10)
