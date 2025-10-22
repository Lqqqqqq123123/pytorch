import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader

# 1. 加载数据集
fashion_train = pd.read_csv(r'../datasets/sgg-data/fashion-mnist_train.csv') # (60000, 785)
fashion_test = pd.read_csv(r'../datasets/sgg-data/fashion-mnist_test.csv') # (10000, 785)

# 2. 划分特征与标签
x_train = torch.tensor(fashion_train.iloc[:, 1:].values / 255.0, dtype=torch.float32).view(-1, 1, 28, 28) # (60000, 1, 28, 28)

y_train = torch.tensor(fashion_train.iloc[:, 0].values, dtype=torch.long) # (60000,)

x_test = torch.tensor(fashion_test.iloc[:, 1:].values / 255.0, dtype=torch.float32).view(-1, 1, 28, 28)

y_test = torch.tensor(fashion_test.iloc[:, 0].values, dtype=torch.long)

# 2.x 显示图像与标签 (随机9张)
# fig, ax = plt.subplots(3, 3)

# for i in range(3):
#     for j in range(3):
#         t = np.random.randint(0, 15000)
#         ax[i][j].imshow(x_train[t, 0, :, :], cmap='gray')
#         ax[i][j].axis('off')
#         print(f"i = {i}, j = {j}, label = {y_train[t]}")


# plt.show()


# 3. 创建数据集
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

# 4. 创建模型

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.pool = nn.AvgPool2d(2, 2)
        self.flatten = nn.Flatten()
    
    def forward(self, x):

        # 第一个卷积+池化
        x = self.pool(torch.sigmoid(self.conv1(x)))

        # 第二个卷积+池化
        x = self.pool(torch.sigmoid(self.conv2(x)))

        # 展平
        x = self.flatten(x)

        # affine
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        y = self.fc3(x)

        return y


# 模型的训练

def train(model, train_dataset, test_dataset, epochs=10, batch_size=64, lr=0.1, device='cpu'):
    # 参数初始化函数
    def init_weights(layer):
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.5)

    model.apply(init_weights)
    model.to(device)

    # 创建dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 定义损失函数
    loss = nn.CrossEntropyLoss()

    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # 记录训练结果
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    for epoch in range(epochs):
        model.train()
        loss_sum, acc_sum = 0.0, 0.0 # 损失和预测正确的个数
        for index, (x, t) in enumerate(train_loader):
            x, t = x.to(device), t.to(device)

            # 前向传播
            y_hat = model(x)

            # 计算损失
            loss_val = loss(y_hat, t)
            loss_sum += loss_val.item() * x.shape[0]

            # 计算预测正确的个数
            pred = torch.argmax(y_hat, dim=1)
            acc_sum += pred.eq(t).sum().item()

            # 反向传播
            loss_val.backward()

            # 优化参数
            optim.step()
            optim.zero_grad()
        
        train_loss.append(loss_sum / len(train_dataset))
        train_acc.append(acc_sum / len(train_dataset))

        print(f"epoch = {epoch}, train_loss = {train_loss[-1]}, train_acc = {train_acc[-1]}")
        
        # 测试
        model.eval()
        test_loader =  DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_loss_sum, test_acc_sum = 0.0, 0.0 # 损失和预测正确的个数
        # 关闭梯度计算
        with torch.no_grad():
            for index, (x, t) in enumerate(test_loader):
                x, t = x.to(device), t.to(device)
                # 前向传播
                y_hat = model(x)
                # 计算损失
                test_loss_val = loss(y_hat, t)
                test_loss_sum += test_loss_val.item() * x.shape[0]
                # 预测正确的个数
                pred = torch.argmax(y_hat, dim=1)
                test_acc_sum += pred.eq(t).sum().item()

        test_loss.append(test_loss_sum / len(test_dataset))
        test_acc.append(test_acc_sum / len(test_dataset))
    
    # 画图
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # ax[0]是损失值的曲线
    ax[0].plot(np.arange(1, epochs + 1), train_loss, 'r--', label='train_loss')
    ax[0].plot(np.arange(1, epochs + 1), test_loss, 'b--', label='test_loss')

    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('loss')
    ax[0].set_title('Loss')
    ax[0].legend()


    # ax[1]是准确率的曲线
    ax[1].plot(np.arange(1, epochs + 1), train_acc, 'r--', label='train_acc')
    ax[1].plot(np.arange(1, epochs + 1), test_acc, 'b--', label='test_acc')
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('acc')
    ax[1].set_title('Acc')
    ax[1].legend()

    plt.show()

        


def main():

    model = Model()
    lr = 0.01
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train(model, train_dataset, test_dataset, epochs=10, batch_size=64, lr=lr, device=device)

        
main()
    
        