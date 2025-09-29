import numpy as np
import os
import sys
from typing import Dict
cur = os.getcwd()
sys.path.append(os.path.dirname(cur))

from common.functions import sigmoid, Softmax, Identity

'''
    三层神经网络的实现，尚硅谷10p
'''

# 初始化权重和偏置

def init_network():
    network = {}
    # 第一层参数
    network['W1'] = np.random.rand(2, 3)  # 2行3列的权重矩阵
    network['B1'] = np.random.rand(3)     # 3维的偏置向量

    # 第二层参数
    network['W2'] = np.random.rand(3, 2)
    network['B2'] = np.random.rand(2)

    # 第三层参数
    network['W3'] = np.random.rand(2, 2)
    network['B3'] = np.random.rand(2)
    return network


def forward(network:Dict, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['B1'], network['B2'], network['B3']
    # 逐层计算
    z1 = sigmoid(np.dot(x, w1) + b1)
    z2 = sigmoid(np.dot(z1, w2) + b2)
    y = Identity(np.dot(z2, w3) + b3)
    return y


if __name__ == '__main__':
    x = np.array([1.0, 0.5])
    network = init_network()
    y = forward(network, x)
    print(y)
