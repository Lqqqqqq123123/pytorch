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
    使用三层神经网络进行手写数字识别,收获如下
    1.pandas与sklearn的使用
    2.pandas的axis与numpy的axis区别(numpy是沿轴的方向去处理,而pandas是按索引去处理)
    3.对神经网络的本质有了更深的理解
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
    y_pred = forward(net, x).argmax(axis=1)
    acc = np.sum(y_pred == y) / y.shape[0]
    print(f'样本数为{y.shape[0]},准确率为{acc:.2%}')



