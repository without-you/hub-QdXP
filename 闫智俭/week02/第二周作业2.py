import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 1. 生成sin函数数据
print("生成sin函数数据...")
x_numpy = np.linspace(-2*np.pi, 2*np.pi, 1000).reshape(-1, 1)  # 生成从-2π到2π的1000个点[4,7](@ref)
y_numpy = np.sin(x_numpy) + 0.1 * np.random.randn(1000, 1)  # 添加少量噪声模拟真实数据[2,5](@ref)

# 转换为PyTorch张量
X = torch.from_numpy(x_numpy).float()
y = torch.from_numpy(y_numpy).float()

print(f"数据生成完成，X形状: {X.shape}, y形状: {y.shape}")
print("---" * 10)

# 2. 定义多层神经网络模型[1,7](@ref)
class SinFittingNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[20, 20, 20], output_dim=1):
        """
        多层神经网络拟合sin函数
        hidden_dims: 列表，定义每个隐藏层的神经元数量[5](@ref)
        """
        super(SinFittingNet, self).__init__()

        layers = []
        prev_dim = input_dim

        # 动态构建隐藏层[5](@ref)
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())  # 使用ReLU激活函数[7](@ref)
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 3. 初始化模型、损失函数和优化器
model = SinFittingNet(input_dim=1, hidden_dims=[64, 32, 16], output_dim=1)  # 3层隐藏层[1](@ref)
criterion = nn.MSELoss()  # 均方误差损失[6](@ref)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器[7](@ref)

print(f"模型结构: {model}")
print(f"总参数数量: {sum(p.numel() for p in model.parameters())}")
print("---" * 10)

# 4. 训练模型
num_epochs = 5000
losses = []  # 记录损失变化

print("开始训练模型...")
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    # 每500个epoch打印一次损失
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)

# 5. 可视化结果
plt.figure(figsize=(15, 10))

# 子图1: 拟合效果对比
plt.subplot(2, 2, 1)
with torch.no_grad():
    y_pred_final = model(X)

plt.scatter(x_numpy, y_numpy, label='训练数据 (含噪声)', color='blue', alpha=0.3, s=10)
plt.plot(x_numpy, np.sin(x_numpy), label='真实 sin(x)', color='green', linewidth=2)
plt.plot(x_numpy, y_pred_final.numpy(), label='神经网络拟合', color='red', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('sin(x)函数拟合效果')
plt.legend()
plt.grid(True)

# 子图2: 损失下降曲线
plt.subplot(2, 2, 2)
plt.plot(losses)
plt.yscale('log')  # 使用对数坐标更清晰显示损失下降
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.title('训练损失下降曲线')
plt.grid(True)

# 子图3: 拟合误差
plt.subplot(2, 2, 3)
error = y_pred_final.numpy() - np.sin(x_numpy)
plt.plot(x_numpy, error, color='purple')
plt.xlabel('x')
plt.ylabel('误差')
plt.title('拟合误差 (预测值 - 真实值)')
plt.grid(True)

# 子图4: 在更广范围内的泛化能力测试
plt.subplot(2, 2, 4)
x_extended = np.linspace(-3*np.pi, 3*np.pi, 1000).reshape(-1, 1)
X_extended = torch.from_numpy(x_extended).float()

with torch.no_grad():
    y_pred_extended = model(X_extended)

plt.plot(x_extended, np.sin(x_extended), label='真实 sin(x)', color='green', linewidth=2)
plt.plot(x_extended, y_pred_extended.numpy(), label='神经网络预测', color='red', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('模型泛化能力测试 (扩展范围)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 6. 打印最终训练损失
final_loss = losses[-1]
print(f"最终训练损失: {final_loss:.6f}")

# 7. 测试模型在一些特定点的预测
print("\n在特定点的预测结果:")
test_points = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
print("x值\t\t真实sin(x)\t神经网络预测\t误差")

with torch.no_grad():
    for x_val in test_points:
        x_tensor = torch.tensor([[x_val]]).float()
        pred = model(x_tensor).item()
        true_val = np.sin(x_val)
        error = abs(pred - true_val)
        print(f"{x_val:6.3f}\t{true_val:8.4f}\t{pred:12.4f}\t{error:8.4f}")


