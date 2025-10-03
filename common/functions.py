import numpy as np
from numpy import array

"""
    阶跃函数
    Args:
        x (ndarray): 输入数据
    Returns:
        ndarray: 阶跃函数输出
"""


def step_function(x):
    return np.array(x > 0, dtype=int)


"""
    sigmoid函数
    Args:
        x (ndarray): 输入数据
    Returns:
        ndarray: sigmoid函数输出
"""


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


"""
    Tanh函数
    Args:
        x (ndarray): 输入数据
    Returns:
        ndarray: Tanh函数输出
"""


def tanh(x):
    return np.tanh(x)


def reLu(x):
    return np.maximum(0, x)


def leakReLu(x, a=0.01):
    if x > 0:
        return x
    else:
        return a * x


def softmax(x):
    
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    else:
        return np.exp(x - np.max(x)) / np.sum(np.exp(x))


def identity(x):
    return x


