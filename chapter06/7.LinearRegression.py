import torch 
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
# 定义超参数
batch_size = 10
lr = 0.01
epochs = 100


# 1.构建数据集
X = torch.randn(100, 1)
k = torch.tensor([[0.5]])
b = torch.tensor([[0.5]])

Y = k * X + b + torch.normal(size=(100, 1), mean=0, std=0.1) 

# 构建 dataset 与 dataloader
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 2.构建模型
net = nn.Linear(1, 1)

# 3.定义损失函数与优化器
loss = nn.MSELoss()
criterion = optim.SGD(net.parameters(), lr=lr)

# 4.训练模型
loss_list = []
for epoch in range(epochs):
    loss_sum = 0
    for x_train, y_train in dataloader:
        
        # 前向传播计算预测值
        y_pred = net(x_train)

        # 计算损失，得到计算图终点
        loss_value = loss(y_pred, y_train)
        loss_sum += loss_value
        # BP
        loss_value.backward()
        # 更新参数
        criterion.step()
        # 清空梯度
        criterion.zero_grad()

    loss_list.append(loss_sum.item())
    print(f'epoch: {epoch + 1}, loss: {loss_sum.item():.2f}')

# 5.可视化
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# 绘制训练损失
ax[0].plot(loss_list) 
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')


# 绘制预测值与真实值
y_test = net(X)

ax[1].plot(X, y_test.detach(), 'r-', label='predict')
ax[1].scatter(X, Y, label='real')

plt.show()

# 打印模型参数
print(net.weight)
print(net.bias)