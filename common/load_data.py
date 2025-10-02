import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
def load_data():

    # 1.加载数据
    df = pd.read_csv(r'../datasets/sgg-data/train.csv')
    x, y = df.iloc[:, 1:], df['label']
   
    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 3.特征工程 自动转为了ndarray
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 4.将y_train, y_test也转为ndarray
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return x_train, x_test, y_train, y_test


# 测试脚本

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)
