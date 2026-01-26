import torch
import numpy as np
import matplotlib.pyplot as plt

# 生成模拟数据
x_data = np.linspace(-np.pi, np.pi, 100)
y_data = np.sin(x_data) + 0.1 * np.random.randn(100)

# 转换为Tensor
X = torch.from_numpy(x_data).float()
Y = torch.from_numpy(y_data).float()

a = torch.randn(1, requires_grad=True, dtype=torch.float)

print(f"初始化参数a: {a.item():.4f}")
print("---" * 10)

loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.SGD([a], lr=0.006)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    y_pred = torch.sin(a * X)

    loss = loss_fn(y_pred, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

print("\n训练完成！")
a_learned = a.item()
print(f"拟合的权重a: {a_learned:.4f}")
print("---" * 10)

with torch.no_grad():
    y_predicted = torch.sin(a * X)

plt.figure(figsize=(8, 5))
plt.scatter(X, Y, label='Raw data', color='blue', alpha=0.6)
plt.plot(X, y_predicted, label=f'Model: y = sin({a_learned:.2f}x)', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
