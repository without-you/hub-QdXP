
# 2、调整 06_torch线性回归.py 构建一个sin函数，
# 然后通过多层网络拟合sin函数，并进行可视化。

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# 1. 生成带噪声的sin数据
np.random.seed(42)
X_numpy = np.linspace(0, 2 * np.pi, 1000).reshape(-1, 1)
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(1000, 1)

# 2. 转换为PyTorch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

# 3. 创建数据集和数据加载器
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# 4. 定义神经网络模型
class SinNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


model = SinNet()

# 5. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# print(dataset)
# exit()
# 6. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 7. 可视化结果
with torch.no_grad():
    y_pred = model(X).numpy()

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统
plt.figure(figsize=(12, 6))
plt.scatter(X_numpy, y_numpy, s=5, label='随机分布的 sin 函数值', alpha=0.5)
plt.plot(X_numpy, np.sin(X_numpy), 'g-', linewidth=3, label='真实的 sin 函数')
plt.plot(X_numpy, y_pred, 'r-', linewidth=2, label='神经网络模型预测')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sin 函数拟合')
plt.legend()
plt.grid(True)
plt.show()


