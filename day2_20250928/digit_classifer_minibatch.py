import sys, os
import numpy as np, pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 不用pycharm是这样的，还得手动管理包，hhh
cur = os.getcwd()
sys.path.append(os.path.dirname(cur))
from common.functions import softmax, sigmoid


'''
    对最初的手写数字识别代码进行改进
    也就是当数据集太大时,一次直接处理复杂度会高
    所以,改代码实现了将数据集分批次处理,也就是pytorch的minibatch
'''

def get_data():
    # 加载数据
    df = pd.read_csv(r'../datasets/sgg-data/train.csv')
    # print(df.describe())
    # 数据集划分
    x = df.drop(columns=['label'], axis=1)
    y = df['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 特征工程
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_test, y_test


def init_network():
    dict_model = joblib.load(r'../datasets/sgg-data/nn_sample')
    return dict_model


def forward(net, x):
    # 加载模型
    w1, w2, w3 = net['W1'], net['W2'], net['W3']
    b1, b2, b3 = net['b1'], net['b2'], net['b3']

    # 逐层计算
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)
    return y

# 主流程
def main():
    x, y = get_data()
    net = init_network()

 

if __name__ == '__main__':
    x, y = get_data()
    net = init_network()

    # y_pred = forward(net, x).argmax(axis=1)
    batch_size = 64
    acc_cnt, n = 0, x.shape[0]

    # 分批次处理
    for i in range(0, n, batch_size):
        # 取出一个批次的数据
        x_temp = x[i:i+batch_size]
        y_temp = y[i:i+batch_size]

        y_pred = forward(net, x_temp).argmax(axis=1)
        acc_cnt += np.sum(y_pred == y_temp)

    
    # 计算准确率
    print(f'样本数为{y.shape[0]},准确率为{acc_cnt/n:.2%}')

  



