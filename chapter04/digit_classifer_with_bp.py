import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
import numpy as np
import matplotlib.pyplot as plt
from common.load_data import load_data
from chapter04.TwoLayerNet import TwoLayerNet

def main():
    
    """
    训练手写数字识别模型的主函数
    
    该函数完成以下主要任务:
    1. 加载并预处理训练数据和测试数据
    2. 初始化两层神经网络模型
    3. 设置训练超参数
    4. 使用小批量梯度下降法训练模型
    5. 记录训练过程中的损失和准确率
    6. 绘制训练结果图表
    
    训练过程中会输出每轮训练的准确率，并最终显示训练集和测试集准确率的变化曲线
    """
     
    # 1.加载数据
    x_train, x_test, y_train, y_test = load_data()

    # 2.加载模型
    net = TwoLayerNet(784, 100, 10)

    # 3.设置超参数
    epochs, batch_size, lr = 20, 64, 0.02
    n = x_train.shape[0]
    iter_per_epoch = np.ceil(n / batch_size)
    iters_num = int(iter_per_epoch * epochs)
    train_loss_list = []; train_acc_list = []; test_acc_list = []


    # 4.训练模型
    for i in range(iters_num):

        # 4.1 获取mini-batch
        batch_mask = np.random.choice(n, batch_size)  
        
        # ndarray的花式索引
        x_batch = x_train[batch_mask]
        y_batch = y_train[batch_mask]


        # 4.2 反向传播
        grads = net.gradient(x_batch, y_batch)
        loss = net.last_layer.loss
        train_loss_list.append(loss)
        # 4.3 更新参数
        for key in ('W1', 'b1', 'W2', 'b2'):
            net.params[key] -= lr * grads[key]
        
        

        # 4.4 计算并保存准确率
        # 每轮迭代都计算一次准确率
        if i % iter_per_epoch == 0:
            train_acc = net.accuracy(x_train, y_train)
            test_acc = net.accuracy(x_test, y_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc) 
            print(f'epoch: {int(i // iter_per_epoch + 1)}, loss:{loss:.2f}, train_acc: {train_acc}, test_acc: {test_acc}')

            

    # 画图
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train_acc')
    plt.plot(x, test_acc_list, label='test_acc', linestyle='--') 
    plt.legend(loc='best'); plt.show()

if __name__ == '__main__':

    main()

