import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline  # 管道操作
from sklearn.impute import SimpleImputer  # 缺失值处理
from torch.utils.data import TensorDataset, DataLoader


# 创建数据集
def create_dataset():
    # 1.读取数据
    df = pd.read_csv(r'E:\Code_Space\1.Python\1.deepLearning\datasets\sgg-data\house_prices.csv')

    # 2.去除无关列
    df.drop(['Id'], axis=1, inplace=True)

    # 3. 划分特征和标签
    x = df.drop(['SalePrice'], axis=1)
    y = df['SalePrice']

    # 4. 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 5.特征工程
    # 5.1 划分数值列和类别列
    num_cols = x.select_dtypes(exclude=['object']).columns
    cat_cols = x.select_dtypes(include=['object']).columns 

    # 5.2 创建管道
    # 创建数值管道
    num_pipe = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='NaN')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')  )
        ]
    )

    # 5.3 创建列转换器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipe, num_cols), 
            ('cat', cat_pipe, cat_cols)
        ]
    )

    # 5.4 进行特征转换
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)

    x_train = pd.DataFrame(x_train.toarray(), columns=preprocessor.get_feature_names_out())
    x_test = pd.DataFrame(x_test.toarray(), columns=preprocessor.get_feature_names_out())

    # 6. 将数据集转换为张量
    train_dataset = TensorDataset(torch.tensor(x_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(x_test.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32))

    return train_dataset, test_dataset, x_train.shape[1]  # 返回特征维度






# 定义一些超参数
epochs, batch_size, lr = 32, 64, 0.5

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 获取数据
train_dataset, test_dataset, dims = create_dataset() # dims = 301


# 定义模型

Model = nn.Sequential(
    nn.Linear(dims, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 1)
)


# 用DataLoader加载数据
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
 

# 定义损失函数
def log_rmse(y_pred, t):
    y_pred = torch.clamp(y_pred, 1, float('inf'))
    mse = nn.MSELoss()
    return torch.sqrt(mse(torch.log(y_pred + 1e-8), torch.log(t + 1e-8)))


# 训练

def main():
    
    # 1. 初始化 model
    net = Model.to(device)
    
    # 1.1 初始化参数
    def init_params(layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.1)
    
    net.apply(init_params)

    # 1.2 定义优化器
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # 1.3 画图所记录的数据
    train_loss_list, test_loss_list = [], []    
    # 2. 模型训练
    for epoch in range(epochs):
        # 2.1 置为训练模式，因为有 bn, dropout 这类层
        net.train()
        loss_sum = 0.0

        for batch_idx, (x, t) in enumerate(train_loader):
            # 将数据加载到指定设备上
            x, t = x.to(device), t.to(device)

            # 前向传播
            y_hat = net(x)

            # 计算损失
            loss_value = log_rmse(y_hat.squeeze(), t)

            # 反向传播
            loss_value.backward()

            # 优化参数
            optimizer.step()
            optimizer.zero_grad()

            # 积累损失和准确度
            loss_sum += loss_value.item()  * x.shape[0]
        
        train_loss_list.append(loss_sum / len(train_dataset))


        # 3. 测试
        net.eval()
        loss_sum = 0.0

        with torch.no_grad():
            for batch_idx, (x, t) in enumerate(test_loader):
                # 将数据加载到指定设备上
                x, t = x.to(device), t.to(device)

                # 前向传播
                y_hat = net(x)

                # 测试损失
                loss_value = log_rmse(y_hat.squeeze(), t)

                # 积累损失
                loss_sum += loss_value.item() * x.shape[0]

        test_loss_list.append(loss_sum / len(test_dataset))

        print(f'epoch: {epoch + 1}, train_loss: {train_loss_list[-1]:.4f}, test_loss: {test_loss_list[-1]:.4f}')

    #  4. 画图
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(np.arange(1, epochs + 1), train_loss_list, label='train_loss')  
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
    ax[0].set_title('Train Loss')
    ax[0].legend()


    ax[1].plot(np.arange(1, epochs + 1), test_loss_list, label='test_loss')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('loss')
    ax[1].set_title('Test Loss')
    ax[1].legend()

    plt.show()

if __name__ == '__main__':
    main()





     





