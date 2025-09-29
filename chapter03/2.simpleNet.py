import sys, os
sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np
from common.functions import softmax
from common.gradient import numerical_gradient
from common.loss import cross_entropy

# 定义一个简单的神经网络
class SimpleNet:
    # 初始化
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 初始化权重
    
    # forward传播
    def forward(self, X):
        a = X @ self.W # 使用sigmoid激活函数
        return softmax(a)
    
    # 计算损失
    def loss(self, x, y):
        y_hat = self.forward(x)
        return cross_entropy(y, y_hat)

# 主流程
if __name__ == '__main__':
    # 1.创建数据
    x = np.array([0.6, 0.9])
    y = np.array([0, 0, 1])  # one-hot编码
    # 2.创建网络
    net = SimpleNet()
    # 3.计算梯度

    f = lambda w: net.loss(x, y)
    dW = numerical_gradient(f, net.W)
    print(dW)

