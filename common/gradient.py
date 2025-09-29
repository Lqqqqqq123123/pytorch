import numpy as np

# 梯度相关代码

# 输入x是一个标量，对应一元函数
def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)

# 输入x是一个向量，对应多元函数
def _numercal_gradient(f, x):
    # 微小变量，防止log(0)
    h = 1e-4
    # 初始化梯度
    grad = np.zeros_like(x)

    # 对x的每一个元素求偏导
    for i in range(x.size):
        tem = x[i]
        x[i] = tem + h
        fxh1 = f(x)
        x[i] = tem - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = tem  # 还原值
    return grad

# 输入x是一个矩阵，对应多个数据
def numerical_gradient(f, X):

    if X.ndim == 1:
        return _numercal_gradient(f, X)
    else:
        grad = np.zeros_like(X)
        for i in range(X.shape[0]):
            grad[i] = _numercal_gradient(f, X[i])
        return grad 

