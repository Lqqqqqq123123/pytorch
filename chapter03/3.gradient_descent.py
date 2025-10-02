# 梯度下降法
# 参数更新方式 x = x - lr * gradient

import sys, os
sys.path.append(os.path.dirname(os.getcwd()))


from common.gradient import numerical_gradient
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []  # 用于记录 x 的变化历史

    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f, x)
        x -= lr * grad
    
    return x, np.array(x_history)

# 测试 gradient_descent 函数

def test_gradient_descent():

    # 定义一个简单的二元函数 f(x) = x1^2 + x2^2
    def f(x: np.ndarray) -> float:
        return x[0]**2 + x[1]**2

    # 初始化参数
    init_x = np.array([-3.0, 4.0])  # 初始点
    lr = 0.05 # 学习率
    step_num = 100  # 梯度下降步数

    # 调用梯度下降函数
    x_min, x_history = gradient_descent(f, init_x, lr, step_num)

    # 打印结果
    print("最终结果:", x_min)
    print("历史路径:\n", x_history)

    # 验证结果是否接近最小值 (0, 0)
    # assert np.allclose(x_min, np.array([0.0, 0.0]), atol=1e-4), "测试失败: 未找到最小值"
    # print("测试通过: 梯度下降找到最小值")

    # 可视化下降路径
    plt.scatter(x_history[:, 0], x_history[:, 1], color='red')
    plt.plot([-5, 5], [0, 0], '--b')
    plt.plot([0, 0], [-5, 5], '--b')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()

# 调用测试函数
if __name__ == "__main__":
    test_gradient_descent()