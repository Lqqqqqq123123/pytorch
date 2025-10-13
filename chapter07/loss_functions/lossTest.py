import torch
from torch import nn, optim

# 1. 定义神经网络模型

class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(5, 3)
        # 初始化参数
        
        self.linear1.weight.data = torch.tensor([
            [1, 2, 3],
            [3, 2, 1],
            [4, 5, 6],
            [6, 5, 4],
            [7, 8, 9]
        ], dtype=torch.float32).mT

        self.linear1.bias.data.fill_(1)


    def forward(self, x):
        return torch.sigmoid(self.linear1(x))




# 2. 数据集
x = torch.randn(10, 5)
target = torch.zeros(10, 3)

# 3. 定义模型
net = Net()

# 4. 定义损失函数与优化器
loss = nn.MSELoss()

criterion = optim.SGD(net.parameters(), lr=0.5)

# 5. 前向传播
y_hat = net(x)

# 6. 计算损失
loss_value = loss(y_hat, target)

# 7. 反向传播
loss_value.backward()

# 8. 更新参数
criterion.step()
criterion.zero_grad()

# 9. 查看模型参数
d = net.state_dict()
for k in d:
    print(k + ":")
    print(d[k])
