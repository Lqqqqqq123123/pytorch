# 优化算法的包装
import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, params, grads):

        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]
    

class AdaGrad:

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
    
    def update(self, params, grads):
        
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * (1 / (np.sqrt(self.h[key] + 1e-7))) * grads[key]
    

class RMSProp:

    def __init__(self, lr=0.01, a=0.9):
        self.lr = lr
        self.a = a
        self.h = None
    

    def update(self, params, grads):

        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] = self.a * self.h[key] + (1 - self.a) * grads[key] * grads[key]
            params[key] -= self.lr * (1 / (np.sqrt(self.h[key] + 1e-7)))


class Adam:

    def __init__(self, lr=0.01, a1=0.9, a2=0.999):
        self.lr = lr
        self.a1 = a1
        self.a2 = a2
        self.v = None
        self.h = None
        self.t = 0

    
    def update(self, params, grads):
        
        if self.v is None:
            self.v, self.h = {}, {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                self.h[key] = np.zeros_like(val)
        
        self.t += 1
        self.lr_temp = self.lr * np.sqrt(1 - self.a2 ** self.t) / (1 - self.a1 ** self.t)

        for key in params.keys():
            # self.v[key] = self.a1 * self.v[key] + (1 - self.a1) * grads[key]
            # self.h[key] = self.a2 * self.h[key] + (1 - self.a2) * grads[key] * grads[key]
            self.v[key] += (1 - self.a1) * (grads[key] - self.v[key])
            self.h[key] += (1 - self.a2) * (grads[key] * grads[key] - self.h[key])
            params[key] -= self.lr_temp * (self.v[key] / (np.sqrt(self.h[key]) + 1e-7))