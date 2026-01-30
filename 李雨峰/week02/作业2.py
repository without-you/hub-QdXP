import math

import torch
import torch.nn as nn
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt

class SinModel(nn.Module):
    def __init__(self, hs1 = 128, hs2 = 256, hs3 = 128):
        super(SinModel, self).__init__()
        self.network = nn.Sequential(
            # 第1层：从 input_size 到 hidden_size1
            nn.Linear(1, hs1),
            nn.ReLU(),  # 增加模型的复杂度，非线性

            # 第2层：从 hidden_size1 到 hidden_size2
            nn.Linear(hs1, hs2),
            nn.ReLU(),

            # 第3层：从 hidden_size2 到 hidden_size3
            nn.Linear(hs2, hs3),
            nn.ReLU(),

            # 第3层：从 hidden_size2 到 hidden_size3
            nn.Linear(hs3, 1),
        )

    def forward(self, x):
        return self.network(x)


# 1. 生成模拟数据 (与之前相同)
X_numpy = np.random.rand(100, 1) * 10
# 形状为 (100, 1) 的二维数组，其中包含 100 个在 [0, 1) 范围内均匀分布的随机浮点数。

y_numpy = 10 * np.sin(X_numpy) + np.random.randn(100, 1)
X = torch.from_numpy(X_numpy).float() # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

model = SinModel()

# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = nn.MSELoss() # 回归任务

# 优化器现在直接传入我们手动创建的参数 [a, b]。
# PyTorch 会自动根据这些参数的梯度来更新它们。
optimizer = torch.optim.SGD(model.parameters(), lr=0.001) # 优化器，基于 a b 梯度 自动更新

# 4. 训练模型
num_epochs = 2000
for epoch in range(num_epochs):
    # 前向传播：手动计算 y_pred = a * X + b
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
with torch.no_grad():
    y_predicted = model(X)

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.scatter(X_numpy, y_predicted, label='Neural Network Fit', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

