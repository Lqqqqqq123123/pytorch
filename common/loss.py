''' 
   手写一些损失函数
'''
import numpy as np


def mse_loss(y_true, y_pred):
   return np.mean((y_true - y_pred) ** 2)

def mae_loss(y_true, y_pred):
   return np.mean(np.abs(y_true - y_pred))
 


def cross_entropy(y_true, y_pred, reduction='mean'):
    # 如果 y_pred 是一维数组，调整为二维
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1) 
        y_true = y_true.reshape(1, -1)
    
    # 如果 y_true 是 one-hot 编码，转换为索引
    if y_true.ndim > 1 and y_pred.size == y_true.size:
        y_true = y_true.argmax(axis=1)
    
    n = y_pred.shape[0]
    '''
        shape 分析
        y_true: (n, 1), y_pred: (n, num_classes)
        在计算交叉熵损失时，我们不需要按公式那样去算，而是直接把对应类别的预测概率拿出来，求个和就行
    '''
    # 微小值 1e-7 防止 log(0) 的情况 
    if reduction == 'mean':
        return -np.log(y_pred[np.arange(n), y_true] + 1e-7).sum() / n
    elif reduction == 'sum':
        return -np.log(y_pred[np.arange(n), y_true] + 1e-7).sum()
    elif reduction == 'none':
        return -np.log(y_pred[np.arange(n), y_true] + 1e-7)
   
