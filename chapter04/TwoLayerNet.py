import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
# 重新定义TwolayerNet
from common.gradient import numerical_gradient
from common.layers import * 
from collections import OrderedDict # 有序字典（这里的有序只是保存了插入的顺序）
import numpy as np


class TwoLayerNet:
    # init
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 创建参数字典params
        self.params = {}
        # 初始化权重矩阵W和偏置向量b
        self.params['W1'] = np.random.randn(input_size, hidden_size) * weight_init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * weight_init_std
        self.params['b2'] = np.zeros(output_size)

        # 创建层对象
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        # 单独创建一个损失层对象
        self.last_layer = SoftmaxWithLoss()
    
    # forward
    def forward(self, x):
        # 逐层前向传播
        # 得到没经过softmax的输出
        for k, layer in self.layers.items():
            x = layer.forward(x)
            
        return x
    
        
    def loss(self, x, t):
        # 先拿到输出层之前的输出
        y = self.forward(x)
        # 用SoftmaxWithLoss计算损失
        return self.last_layer.forward(y, t)
    # x:输入数据 t:数据标签
    def accuracy(self, x, t):
       
        # 如果t是独热编码，则进行转换
        # if t.ndim != 1 and t.shape[1] > 1:
        #     t = np.argmax(t, axis=1)
        
        y_hat = self.forward(x)
        y_hat = np.argmax(y_hat, axis=1)
        return np.sum(y_hat == t) / float(x.shape[0])
    
    # 最开始的计算方法----数值微分
    def numerical_gradient(self, x, t):
         # 包装损失函数
        loss_W = lambda _:self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
    
    # 梯度计算方法----反向传播
    def gradient(self, x, t):
        
        # 梯度清0 todo

        # 先进行前向传播
        self.loss(x, t) 

        # 进行反向传播,理论上，1是反向传播的起点
        dout = 1 * self.last_layer.backward()
        # 逐层进行反向传播
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)
        
        # 提取各层参数的梯度
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].grad_w, self.layers['Affine1'].grad_b
        grads['W2'], grads['b2'] = self.layers['Affine2'].grad_w, self.layers['Affine2'].grad_b

        return grads
    
