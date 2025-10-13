import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import optim


# 数学场景：找到 f(x) = 0.05 * x1^2 + x2^2 的最小值

def f(X):
    return 0.05 * X[0]**2 + X[1]**2



# 定义一些超参数
lr = 0.2
momentum = 0.5
iter_num = 100

# 定义初始点
X = torch.tensor([-7.0, 2.0], requires_grad=True)

# 封装优化器
def gradient_descent(X, optimizer, iter_num):
    X_history = []
    for i in range(iter_num):
        X_history.append(X.detach().numpy().copy())
        # 前向传播
        y = f(X)  # 这个已经是优化目标了，也就是损失值

        # 反向传播 
        y.backward()

        # 更新参数
        optimizer.step()
        optimizer.zero_grad()

    return X_history


# 定义优化器


# 为了用两次，所以clone一下，避免影响初始化点
# SGD
X_clone = X.detach().clone().requires_grad_(True)
sgd = optim.SGD([X_clone], lr=lr)
sgd_history = gradient_descent(X_clone, sgd, iter_num)

# Momentum
X_clone = X.detach().clone().requires_grad_(True)
momentum = optim.SGD([X_clone], lr=lr, momentum=momentum)
momentum_history = gradient_descent(X_clone, momentum, iter_num)


sgd_history = np.array(sgd_history)
momentum_history = np.array(momentum_history)

# 画图
# 等高线
x_grid, y_grid = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))

plt.plot(sgd_history[:, 0], sgd_history[:, 1], 'o-', color='r', label='SGD')
plt.plot(momentum_history[:, 0], momentum_history[:, 1], 'o-', color='b', label='Momentum')

Y = 0.05 * x_grid**2 + y_grid**2
plt.contour(x_grid, y_grid, Y)
plt.plot(0, 0, '+')
plt.legend()
plt.show()



