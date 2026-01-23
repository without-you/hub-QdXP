import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成sin函数数据
np.random.seed(42)
X_numpy = np.random.rand(100, 1) * 10  # X范围[0, 10)
y_numpy = np.sin(X_numpy) + np.random.randn(100, 1) * 0.3  # sin函数 + 噪声

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成")
print(f"X形状: {X.shape}, y形状: {y.shape}")


# 2. 定义多层神经网络
class MultiLayerNet(nn.Module):
    def __init__(self):
        super(MultiLayerNet, self).__init__()
        # 3层网络：输入层 -> 隐藏层1 -> 隐藏层2 -> 输出层
        self.layers = nn.Sequential(
            nn.Linear(1, 64),  # 输入层 -> 64个神经元
            nn.ReLU(),  # 非线性激活
            nn.Linear(64, 32),  # 64 -> 32个神经元
            nn.ReLU(),  # 非线性激活
            nn.Linear(32, 1)  # 32 -> 输出层
        )

    def forward(self, x):
        return self.layers(x)


# 3. 创建模型、损失函数和优化器
model = MultiLayerNet()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

print(f"\n模型结构:")
print(model)
print(f"总参数: {sum(p.numel() for p in model.parameters())}")

# 4. 训练模型
num_epochs = 2000
losses = []

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    losses.append(loss.item())

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每500轮打印一次
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f"\n最终损失: {loss.item():.6f}")

# 5. 预测和可视化
model.eval()
with torch.no_grad():
    # 生成更密集的点用于绘制平滑曲线
    X_range = torch.linspace(0, 10, 200).reshape(-1, 1)
    y_pred_range = model(X_range).numpy()

    # 预测训练数据
    y_pred = model(X).numpy()

# 绘制结果
plt.figure(figsize=(8, 4))

# 子图1：拟合效果
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.subplot(1, 1, 1)
plt.scatter(X_numpy, y_numpy, alpha=0.6, label='原始数据')
plt.plot(X_range, y_pred_range, 'r-', label='网络拟合', linewidth=2)
plt.plot(X_range, np.sin(X_range), 'g--', label='真实sin函数', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('多层网络拟合sin函数')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
