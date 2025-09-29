''' 
   手写一些损失函数
'''
import numpy as np


def mse_loss(y_true, y_pred):
   return np.mean((y_true - y_pred) ** 2)

def mae_loss(y_true, y_pred):
   return np.mean(np.abs(y_true - y_pred))
 


def cross_entropy(y_true, y_pred, reduction='mean'):
   if y_pred.ndim == 1:
      y_pred.reshape(1, -1) 
      y_true.reshape(1, -1)
   # 如果y_true是one-hot编码,转换成索引
   if y_pred.size == y_true.size:
      y_true = y_true.argmax(axis=1) 
   
   n = y_pred.shape[0]
   '''
      shape分析
      y_true:(n, 1), y_pred:(n, num_classes)
      在计算交叉熵损失时，我们不需要按公式那样去算，而是直接把对于的类别的预测概率拿出来，求个和就行
   '''
   # 微小值1e-7防止log(0)的情况 
   if reduction == 'mean':
      return -np.log(y_pred[np.arange(n, y_true)] + 1e-7).sum() / n
   elif reduction == 'sum':
      return -np.log(y_pred[np.arange(n, y_true)] + 1e-7).sum()
   elif reduction == 'none':
      return -np.log(y_pred[np.arange(n, y_true)] + 1e-7)
   
