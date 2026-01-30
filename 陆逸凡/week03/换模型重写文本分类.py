from tabulate import tabulate  # 用于以表格格式输出数据
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
from torch.utils.data import Dataset, DataLoader  # 数据集和数据加载器
import pandas as pd  # 数据处理库
import torch.optim as optim  # 优化器
import matplotlib.pyplot as plt  # 绘图库
import time  # 时间相关功能

# 第一步：数据预处理
# 读取CSV文件，使用制表符\t分隔列，没有表头（header=None）
dataset = pd.read_csv("D:/AI/AI work/dataset.csv", sep="\t", header=None)

# 将文本和标签分开，分别转换为列表
# 第一列（索引0）是文本数据
texts = dataset[0].tolist()
# 第二列（索引1）是对应的标签数据
labels = dataset[1].tolist()

# 将标签转换为数字索引
# 使用set获取所有不重复的标签，然后为每个标签分配一个唯一索引
# label_to_index是一个字典：原始标签 -> 数字索引
label_to_index = {label: index for index, label in enumerate(set(labels))}

# 创建反向映射：数字索引 -> 原始标签
# 在预测时，我们需要将模型输出的数字索引转换回原始标签
label_to_index_reverse = {index: label for label, index in label_to_index.items()}

# 将所有标签转换为对应的数字索引
# 为数据集中的每个标签找到对应的数字索引，创建数值标签列表
numerical_labels = [label_to_index[label] for label in labels]

# 创建字符到索引的映射，用于将文本转换为数字序列
# 首先添加填充字符<pad>的映射，索引为0
text_to_index = {'<pad>': 0}
# 遍历所有文本，构建字符到索引的映射字典
for text in texts:
    # 遍历每个文本的每个字符
    for char in str(text):
        # 如果字符不在字典中（即之前没有遇到过）
        if char not in text_to_index:
            # 添加新的映射：字符 -> 当前字典长度（即下一个可用索引）
            text_to_index[char] = len(text_to_index)

        # 创建反向映射：索引 -> 字符
# 在需要查看预测结果时，用于将索引转换回字符（虽然代码中未使用此映射）
text_to_index_reverse = {i: char for i, char in enumerate(text_to_index)}


# 自定义数据集类，继承自PyTorch的Dataset类
class CharDataset(Dataset):
    def __init__(self, texts, labels, text_to_index, max_len):
        """
        初始化数据集
        Args:
            texts: 文本列表
            labels: 标签列表（数字形式）
            text_to_index: 字符到索引的映射字典
            max_len: 文本最大长度，用于填充或截断
        """
        self.texts = texts  # 存储文本列表
        # 将标签列表转换为PyTorch张量，数据类型为长整型
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.text_to_index = text_to_index  # 字符索引映射字典
        self.max_len = max_len  # 最大序列长度

    def __len__(self):
        """返回数据集样本数量"""
        return len(self.texts)  # 数据集中文本的数量

    def __getitem__(self, idx):
        """获取单个样本"""
        text = self.texts[idx]  # 获取指定索引的文本
        # 将文本转换为索引序列
        # 对于每个字符，如果字符在映射字典中，则使用对应索引；否则使用0（<pad>）
        # [:self.max_len]确保序列长度不超过max_len
        indices = [self.text_to_index.get(char, 0) for char in text[:self.max_len]]
        # 填充：如果序列长度小于max_len，用0填充到指定长度
        indices += [0] * (self.max_len - len(indices))
        # 返回文本索引序列和对应的标签
        # 将索引列表转换为PyTorch张量，数据类型为长整型
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


# 序列模型类，继承自nn.Module
class sequence_model(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, layer_type, num_layers=2, dropout=0.3):
        """
        初始化序列模型
        Args:
            vocab_size: 词汇表大小（输入维度）
            embedding_size: 词嵌入维度
            hidden_size: 隐藏层维度
            output_size: 输出类别数量
            layer_type: 序列层类型（RNN/LSTM/GRU）
            num_layers: RNN层数
            dropout: Dropout概率
        """
        super(sequence_model, self).__init__()  # 调用父类nn.Module的初始化方法

        # 词嵌入层：将字符索引映射为稠密向量
        # padding_idx=0表示索引0（填充字符）的嵌入向量不参与梯度更新，始终为0
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        # 根据layer_type参数选择不同的序列层
        if layer_type == 'LSTM':
            # LSTM层：处理长期依赖关系
            # batch_first=True表示输入张量的第一个维度是batch_size
            # dropout参数只在层数>1时生效
            self.act = nn.LSTM(embedding_size, hidden_size, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif layer_type == 'GRU':
            # GRU层：LSTM的简化版，计算更快，参数更少
            self.act = nn.GRU(embedding_size, hidden_size, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif layer_type == 'RNN':
            # 标准RNN层：最简单的循环神经网络
            self.act = nn.RNN(embedding_size, hidden_size, batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # BatchNorm层：稳定训练，加速收敛
        # 作用：对每一批数据的激活值进行归一化，使其均值为0，方差为1
        # 好处：减少内部协变量偏移，允许使用更大的学习率，加速收敛
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # Dropout层：防止过拟合
        # 作用：在训练过程中随机丢弃一部分神经元的输出，减少神经元之间的协同适应性
        # dropout参数指定了丢弃概率
        self.dropout = nn.Dropout(dropout)

        # 全连接层：将隐藏状态映射到输出类别
        # 输入维度是hidden_size，输出维度是output_size（类别数量）
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, layer_type):
        """
        前向传播
        Args:
            x: 输入文本索引序列（形状：[batch_size, sequence_length]）
            layer_type: 序列层类型
        Returns:
            out: 模型输出（未归一化的类别分数，形状：[batch_size, output_size]）
        """
        # 将索引转换为词嵌入向量
        # 输入x的形状：[batch_size, seq_len]
        # 输出x的形状：[batch_size, seq_len, embedding_size]
        x = self.embedding(x)

        # 通过序列层（RNN/LSTM/GRU）
        if layer_type == 'LSTM':
            # LSTM返回两个值：输出序列和(hidden_state, cell_state)元组
            # output: 所有时间步的输出，形状：[batch_size, seq_len, hidden_size]
            # h: 最后一个时间步的隐藏状态，形状：[num_layers, batch_size, hidden_size]
            # c: 最后一个时间步的细胞状态，形状：[num_layers, batch_size, hidden_size]
            output, (h, c) = self.act(x)
        else:
            # RNN/GRU返回两个值：输出序列和hidden_state
            # output: 所有时间步的输出，形状：[batch_size, seq_len, hidden_size]
            # h: 最后一个时间步的隐藏状态，形状：[num_layers, batch_size, hidden_size]
            output, h = self.act(x)

        # h的形状是 (num_layers, batch_size, hidden_size)
        # 我们只需要最后一层的隐藏状态进行分类
        # h[-1]取最后一层，形状变为 (batch_size, hidden_size)
        h = h[-1]

        # 应用BatchNorm：对隐藏状态进行归一化
        # BatchNorm1d期望输入形状为 (batch_size, features)
        h = self.batch_norm(h)

        # 应用Dropout：在训练模式下随机丢弃部分神经元
        h = self.dropout(h)

        # 通过全连接层，得到每个类别的分数
        # 输出形状：[batch_size, output_size]
        out = self.fc(h)

        return out


# 定义损失函数：交叉熵损失，用于多分类问题
# CrossEntropyLoss结合了LogSoftmax和NLLLoss
criterion = nn.CrossEntropyLoss()


def train(model, train_loader, optimizer, epoch, model_type):
    """
    训练函数
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        epoch: 当前训练轮次
        model_type: 模型类型
    Returns:
        loss_mean: 本轮训练的平均损失
    """
    model.train()  # 设置为训练模式，启用Dropout和BatchNorm的训练行为
    loss_total = 0.0  # 累计损失，用于计算平均损失

    # 遍历训练数据加载器中的所有批次
    for batch_idx, (data, target) in enumerate(train_loader):
        # 前向传播：计算模型输出
        outputs = model(data, model_type)
        # 计算损失：预测输出与真实标签的差异
        loss = criterion(outputs, target)

        # 反向传播：计算梯度
        optimizer.zero_grad()  # 清空历史梯度，防止梯度累加
        loss.backward()  # 反向传播，计算损失相对于模型参数的梯度

        # 梯度裁剪：防止梯度爆炸（对RNN特别重要）
        # 将所有参数的梯度范数裁剪到最大为1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()  # 更新模型参数（根据优化算法更新参数）

        # 计算本轮batch的损失贡献
        # loss.item()返回标量损失值，除以批次数量得到平均损失
        running_loss = loss.item() / len(train_loader)
        loss_total += running_loss  # 累加损失

    # 计算本轮平均损失
    loss_mean = loss_total / len(train_loader)
    # 打印当前轮次的损失
    print(f"第{epoch}轮的loss: {loss_mean:.4f}")
    return loss_mean


# 测试文本列表，用于评估模型效果
# 这些是固定的测试用例，用于比较不同模型的性能
check_texts = [
    "明天上海下雨吗",  # Weather-Query（天气查询）
    "导航去清华大学",  # Travel-Query（出行查询）
    "收听中央人民广播电台",  # Radio-Listen（收听广播）
    "播放周杰伦的歌",  # Music-Play（播放音乐）
    "把电视声音调大一点",  # HomeAppliance-Control（家电控制）
    "帮我设置闹钟",  # Alarm-Update（闹钟设置）
    "播放香港电台的邓丽君的经典音乐",  # Radio-Listen（收听广播）
    "计算器打开一下",  # HomeAppliance-Control（家电控制）
    "能把家庭剧家有儿女给我播放一下吗"  # FilmTele-Play（影视播放）
]

# 正确的预测结果，用于计算模型准确率
correct_results = ["Weather-Query", "Travel-Query", "Radio-Listen", "Music-Play",
                   "HomeAppliance-Control", "Alarm-Update", "Radio-Listen",
                   "HomeAppliance-Control", "FilmTele-Play"]

# 全局字典：存储每个模型的正确预测数量
model_correct = {}
# 全局字典：存储每个模型的预测耗时
calculated_time = {}


def check_result(model, max_len, model_type):
    """
    评估函数：在测试文本上评估模型效果
    Args:
        model: 训练好的模型
        max_len: 最大序列长度
        model_type: 模型类型
    Returns:
        testing_results: 测试结果列表，第一个元素是模型类型，后面是每个测试文本的预测结果
    """
    model.eval()  # 设置为评估模式，禁用Dropout和BatchNorm（使用训练好的统计量）
    testing_results = [model_type]  # 结果列表，第一个元素是模型类型
    correct_result_count = 0  # 正确预测的计数器

    with torch.no_grad():  # 不计算梯度，节省内存和计算资源
        start_time = time.time()  # 记录开始时间
        for index, check_text in enumerate(check_texts):  # 遍历每个测试文本
            # 将测试文本转换为索引序列
            # 对于每个字符，如果字符在映射字典中，则使用对应索引；否则使用0（<pad>）
            input_test_text = [text_to_index.get(char, 0) for char in check_text[:max_len]]
            # 填充序列到最大长度
            if len(input_test_text) < max_len:
                input_test_text += [0] * (max_len - len(input_test_text))

            # 添加batch维度（batch_size=1），转换为tensor
            # unsqueeze(0)在维度0添加一个维度，形状从[max_len]变为[1, max_len]
            input_tensor = torch.tensor(input_test_text, dtype=torch.long).unsqueeze(0)

            # 模型预测
            result = model(input_tensor, model_type)

            # 找到概率最高的类别
            # torch.max返回两个值：最大值和最大值对应的索引
            # 参数1表示在维度1（类别维度）上取最大值
            # _表示我们不关心最大值本身（概率值），只关心索引
            _, max_possibility = torch.max(result, 1)

            # 将数字索引转换回原始标签
            # max_possibility.item()获取张量中的标量值
            result_label = label_to_index_reverse[max_possibility.item()]
            testing_results.append(result_label)  # 添加到结果列表

            # 检查预测是否正确
            if result_label == correct_results[index]:
                correct_result_count += 1  # 正确计数加1

        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算总耗时

    # 将模型的准确率和耗时存储到全局字典中
    model_correct[model_type] = correct_result_count / len(check_texts)  # 计算准确率
    calculated_time[model_type] = elapsed_time  # 存储耗时

    return testing_results


def main():
    """
    主函数：组织整个训练和评估流程
    """
    # 训练轮数
    epochs = 10
    # 词汇表大小（字符种类数量）
    input_size = len(text_to_index)
    # 词嵌入维度（每个字符用多少维向量表示）
    embedding_size = 64
    # 隐藏层维度（RNN/LSTM/GRU隐藏状态的维度）
    hidden_size = 128
    # 输出类别数量（标签的种类数）
    output_size = len(label_to_index)
    # 最大序列长度（所有文本都会被填充或截断到这个长度）
    max_len = 40

    # 要训练的模型类型列表
    model_types = ['LSTM', 'RNN', 'GRU']

    # 创建数据集实例
    train_loader = CharDataset(texts, numerical_labels, text_to_index, max_len)
    # 创建数据加载器
    # batch_size=64：每批处理64个样本
    # shuffle=True：每个epoch开始时打乱数据顺序
    dataloader = DataLoader(train_loader, batch_size=64, shuffle=True)

    # 存储每个模型的训练历史（损失值）
    all_losses = {}
    # 存储每个模型的测试结果
    all_testing_results = []

    # 遍历每种模型类型进行训练
    for model_type in model_types:
        print(f"现在开始{model_type}模型的训练")

        # 创建模型实例
        model = sequence_model(input_size, embedding_size, hidden_size,
                               output_size, layer_type=model_type)
        loss_history = []  # 当前模型的损失历史列表

        # 使用Adam优化器，学习率0.001
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练循环
        for epoch in range(epochs):
            # 训练一个epoch，获取平均损失
            loss = train(model, dataloader, optimizer, epoch, model_type)
            loss_history.append(loss)  # 记录损失到历史列表

        # 存储当前模型的损失历史
        all_losses[model_type] = loss_history

        # 在测试集上评估模型
        test_result = check_result(model, max_len, model_type)
        all_testing_results.append(test_result)  # 存储测试结果

    # 打印测试结果表格
    # 构建表头：第一列是"模型名称"，后面是各个测试文本
    header = ["模型名称"] + check_texts
    # 使用tabulate库以表格形式打印结果
    print(tabulate(all_testing_results, headers=header, tablefmt="grid"))

    # 查看预测的精度
    # 找到准确率的最大值
    max_value = max(model_correct.values())
    # 找到所有准确率等于最大值的模型
    max_keys = [k for k, v in model_correct.items() if v == max_value]

    # 初始化最小耗时和对应的模型
    min_time = 1000  # 设置一个大数作为初始最小值
    min_time_key = max_keys[0]  # 默认选择第一个模型

    # 如果有多个模型准确率相同
    if len(max_keys) > 1:
        # 将模型名称用顿号连接成字符串
        models_str = "、".join(max_keys)
        print(f"精度最大的模型有: {models_str}")

        # 从准确率相同的模型中找出耗时最短的
        for key in max_keys:
            if min_time > calculated_time[key]:
                min_time = calculated_time[key]
                min_time_key = key
        print(f"耗时最短的预测模型是：{min_time_key}")

    # 打印最高准确率
    print(f"精度最大的模型的准确率是: {model_correct[max_keys[0]]}")

    # 绘制损失曲线图
    # 为不同模型设置不同颜色
    colors_list = ['b', 'g', 'r']  # 蓝色、绿色、红色
    # 设置图形大小：10英寸宽，6英寸高
    plt.figure(figsize=(10, 6))

    # 遍历所有模型类型，绘制损失曲线
    for idx, model_type in enumerate(model_types):
        if model_type in all_losses:
            # 绘制每个模型的损失曲线
            # x轴：epoch编号（从1到epochs）
            # y轴：损失值
            # label：图例标签
            # color：线条颜色
            # marker：数据点标记为圆形
            plt.plot(range(1, epochs + 1), all_losses[model_type],
                     label=f'{model_type}', color=colors_list[idx], marker='o')

    # 设置图表属性
    plt.title('Training Loss Comparison')  # 图表标题
    plt.xlabel('Epochs')  # X轴标签
    plt.ylabel('Loss')  # Y轴标签
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图形


# 程序入口：如果直接运行此脚本，则执行main函数
if __name__ == '__main__':
    main()