import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# 一、生成sin测试数据
x_num = np.linspace(0,2 * np.pi,200).reshape(-1,1)
y_num = np.sin(x_num) + np.random.rand(200,1) * 0.1
x_train = torch.from_numpy(x_num).float()
y_train = torch.from_numpy(y_num).float()

# 二、定义多层神经网张
class SinNet(nn.Module):
    def __init__(self):
        super(SinNet,self).__init__()
        self.fc1 = nn.Linear(1,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,1)

        self.reLu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self,x):
        # 第一层线性变换+激活函数
        x = self.fc1(x)
        x = self.reLu(x)

        # 第二层线性变换+激活函数
        x = self.fc2(x)
        x = self.reLu(x)

        # 第三层线性变换+激活函数
        x = self.fc3(x)
        x = self.reLu(x)

        # 第四层输出
        x = self.fc4(x)
        return x

model = SinNet()

# 三、定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

# 四、训练网络
num_epochs = 2000
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(x_train)

    # 计算损失
    loss = criterion(y_pred,y_train)

    # 反向传播与优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch [{epoch + 1} / {num_epochs}], loss : {loss.item():.6f}")

# 五、测试与可视化
x_test = np.linspace(0, 2 * np.pi, 500).reshape(-1,1)
x_test_sensor = torch.from_numpy(x_test).float()

with torch.no_grad():
    y_pred = model(x_test_sensor)
    y_pred_np = y_pred.numpy()

y_true = np.sin(x_test)

# 六、绘制图表
plt.figure(figsize=(10, 6))
plt.scatter(x_num, y_num, label='train data', color='lightblue', alpha=0.6)
plt.plot(x_test, y_true, label='read sin(x)', color='red', linewidth=2)
plt.plot(x_test, y_pred_np, label='NN sin(x)', color='blue', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
