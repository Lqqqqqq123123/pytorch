import sys, os
sys.path.append(os.path.dirname(os.getcwd()))

from common.optimizer import *
import matplotlib.pyplot as plt
from collections import OrderedDict

# 定义目标函数 f(x, y) = x^2 / 20 + y^2
def f(x, y):
    return x**2 / 20 + y**2

def f_gard(x, y):
    return x / 10, y * 2

# 定义初始点
init_pos = (-7.0, 2.0)
idx = 1
# 创建优化器实例
optimizers = OrderedDict()

optimizers['SGD'] = SGD(lr=0.1)
optimizers['Momentum'] = Momentum(lr=0.1, momentum=0.9)
optimizers['AdaGrad'] = AdaGrad(lr=0.1)
optimizers['Adam'] = Adam(lr=0.1, a1=0.9, a2=0.999)


# 枚举优化器
import sys, os
sys.path.append(os.path.dirname(os.getcwd()))

from common.optimizer import *
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np

# 定义目标函数 f(x, y) = x^2 / 20 + y^2
def f(x, y):
    return x**2 / 20 + y**2

def f_gard(x, y):
    return x / 10, y * 2

# 定义初始点
init_pos = (-7.0, 2.0)
idx = 1
# 创建优化器实例
optimizers = OrderedDict()

optimizers['SGD'] = SGD(lr=0.7)
optimizers['Momentum'] = Momentum(lr=0.09)
optimizers['AdaGrad'] = AdaGrad(lr=1.2)
optimizers['Adam'] = Adam(lr=0.8, a1=0.5)


# 枚举优化器
for key in optimizers.keys():

    optimizer = optimizers[key]

    x_history, y_history = [], []

    grads, params = {}, {}

    params['x'], params['y'] = init_pos[0], init_pos[1]

    for i in range(30):

        x_history.append(params['x'])
        y_history.append(params['y'])

        # 计算梯度
        grads['x'], grads['y'] = f_gard(params['x'], params['y'])
        optimizer.update(params, grads)
        
    # 画图
    plt.subplot(2, 2, idx)
    idx += 1    

    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(x, y)  # 创建网格点
    Z = f(X, Y)  # 计算每个网格点上的函数值
    
    # 绘制等高线图
    plt.contourf(X, Y, Z)
    
    # 标记最优点
    plt.plot(0, 0, '+')

    # 绘制优化路径
    plt.plot(x_history, y_history, 'o-', color='r', label=key, markersize=3)
    plt.xlim(-10, 10)
    plt.ylim(-5, 5)
    plt.legend()

plt.show()
