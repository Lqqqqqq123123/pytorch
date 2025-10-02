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
        self.W = np.random.randn(2, 3)
    
    # forward传播
    def forward(self, x):
        return np.dot(x, self.W)
    
    # 计算损失
    def loss(self, x, t):
        y = self.forward(x)
        y = softmax(y)
        return cross_entropy(t, y) 

# 主流程
if __name__ == '__main__':
    # 1.创建数据
    x = np.array([0.6, 0.9])
    y = np.array([0, 0, 1])  # one-hot编码
    # 2.创建网络
    net = SimpleNet()
    # 3.计算梯度

    f = lambda _: net.loss(x, y)
    dW = numerical_gradient(f, net.W)
    print(dW)

