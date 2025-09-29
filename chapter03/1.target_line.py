import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
import numpy as np
import matplotlib.pyplot as plt
from common.gradient import numerical_diff


# 原函数
def f(x):
    return 0.01 * x**2 + 0.1 * x

# 在函数f的x处的切线方程
def target_line(f, x) :
    y = f(x)
    # 计算f在x处的导数，也就是切线的斜率
    k = numerical_diff(f, x)
    b = y - k * x
    # 新知识，直接返回函数
    print(f'切线方程: y = {k:.2f}*x + {b:.2f}')
    return lambda t: k*t + b


# 画图
def main():
    # 当前曲线
    x = np.arange(0.0, 20.0, 0.1)
    y = f(x)

    # 切线 计算x=5处的切线
    f_line = target_line(f, 5)
    y_line = f_line(x)
    plt.plot(x, y)
    plt.plot(x, y_line)
    plt.show()

if __name__ == '__main__':
    main()
