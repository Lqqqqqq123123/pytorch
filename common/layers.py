import os, sys
import numpy as np
sys.path.append(os.path.dirname(os.getcwd()))
from common.functions import sigmoid, softmax
from common.loss import cross_entropy
# 各种层结构
# Relu
class Relu:
    
    # init
    def __init__(self):
        # 内部属性，记录那些 x <= 0
        self.mask = None

    # forward
    def forward(self, x):
        self.mask = (x <= 0)
        y = x.copy()
        y[self.mask] = 0
        return y
    
    # backward
    def backward(self, dout):
        out = dout.copy()
        
        # 将x<=0的梯度置零
        out[self.mask] = 0
        return out


# Sigmoid
class Sigmoid:

    # init
    def __init__(self):
        self.out = None
    
    # forward
    def forward(self, x):
        self.out = sigmoid(x)
        return self.out
    
    # backward
    def backward(self, dout):
        return dout * self.out * (1.0 - self.out)
    

# Affine
class Affine:
    
    # init
    def __init__(self, W, b):
        # 权重矩阵W和偏置向量b
        self.W = W; self.b = b
        # 对输入数据x进行保存
        self.X = None; self.origin_shape = None
        # 对当前层的W和b的导数保存，方便后续梯度下降，更新参数
        self.grad_w = None; self.grad_b = None

    # forward
    def forward(self, X):
        
        self.origin_shape = X.shape; self.X = X
        X = X.reshape(X.shape[0], -1)

        y = np.dot(X, self.W) + self.b
        X = X.reshape(self.origin_shape)
        return y
        
    # backward
    def backward(self, dout):
        # 当X是一个三维数组时，将它展平为二维数组，方便后续运算
        self.X = self.X.reshape(self.X.shape[0], -1)
        # 分别计算参数
        out = np.dot(dout, self.W.T)
        self.grad_w = np.dot(self.X.T, dout)
        self.grad_b = np.sum(dout, axis=0)
        # 还原输入数据X
        self.X = self.X.reshape(self.origin_shape)
        # 梯度的反向传播
        return out 
    


# SoftmaxWithLoss
class SoftmaxWithLoss:
    
    # init
    def __init__(self):
        self.y = None
        self.y_hat = None
        self.loss = None

    # forward
    def forward(self, x, t):
        # 保存y,y_hat,方便计算梯度
        self.y_hat = softmax(x); self.y = t
        # 计算损失
        self.loss = cross_entropy(self.y, self.y_hat)
        return self.loss
    
    # backward
    # def backward(self):
    #     # 如果y不是独热编码，则转换为独热编码   
    #     if self.y.ndim == 1:
    #         num_classes = self.y_hat.shape[1]
    #         self.y = np.eye(num_classes)[self.y]

    #     return self.y_hat - self.y
    
    def backward(self):
        n = self.y_hat.shape[0]
        if self.y.size == self.y_hat.size:
            return (self.y_hat - self.y) / n
        else:
            self.y_hat[np.arange(n), self.y] -= 1
            return self.y_hat / n
# test()
def test_softmax_with_loss():
    x = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    #t = np.array([[0, 0, 0, 1, 0]])
    t = np.array([3])

    loss = SoftmaxWithLoss()
    loss.forward(x, t)
    print(loss.backward())

if __name__ == '__main__':
    test_softmax_with_loss()
