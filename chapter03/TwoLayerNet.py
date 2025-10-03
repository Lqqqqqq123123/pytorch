import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np
from common.functions import softmax, sigmoid
from common.loss import cross_entropy
from common.gradient import numerical_gradient
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
         

    # forward 
    def forward(self, X):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        z1 = sigmoid(np.dot(X, W1) + b1)
        y = softmax(np.dot(z1, W2) + b2)
        return y
    
    # 计算损失
    def loss(self, x, t):
        y_hat = self.forward(x)
        return cross_entropy(t, y_hat)
    
    # 计算准确率
    def accuracy(self, x, t):
        # 前向传播，获取预测类别
        y_hat = self.forward(x).argmax(axis=1)  # shape: (N,)

        # 如果 t 是独热编码，转换为类别索引
        if t.ndim > 1 and t.shape[1] > 1:  # 判断是否是独热编码
            t = t.argmax(axis=1)  # 转换为类别索引

        # 计算准确率
        acc = np.sum(y_hat == t) / float(x.shape[0])
        return acc
    
    # 计算梯度
    # x: 输入数据 t: 真实标签
    def gradient(self, x, t):
        # 包装损失函数
        loss_W = lambda _:self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads




# 主流程

