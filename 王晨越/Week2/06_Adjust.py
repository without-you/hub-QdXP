import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
X_numpy = np.linspace(1, 5, 2000).reshape(-1, 1)

y_numpy = np.sin(X_numpy) + 0.1*np.random.rand(2000, 1)
X = torch.from_numpy(X_numpy).float().requires_grad_(True) # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float().requires_grad_(True)

print("数据生成完成。")
print("---" * 10)

class sinvalue(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(sinvalue, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.Tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.Tanh = nn.Tanh()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.Tanh(out)
        out = self.fc2(out)
        out = self.Tanh(out)
        out = self.fc3(out)
        return out

input_size = 1
hidden_dim1 = 128
hidden_dim2 = 128
output_dim = 1
model = sinvalue(input_size, hidden_dim1, hidden_dim2, output_dim)
criterion = nn.MSELoss() # 分类损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01) # Adam 优化器  可以结合梯度 动态调整学习， 0.01 -> 0.001 -> 0.00001

for _ in range(1000):
    model.train()
    optimizer.zero_grad()

    outputs = model(X)
    loss = criterion(outputs, y)

    loss.backward()
    optimizer.step()

    print(f"Training complete. Loss: {loss.item():.4f}")



#5. 打印最终学到的参数
print("\n训练完成！")
print("---" * 10)

# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
with torch.no_grad():
    y_predicted = model(X)

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label=f'Model:Multi Network', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
