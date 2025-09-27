import sys, os
import numpy as np, pandas as pd
import joblib
cur = os.getcwd()
sys.path.append(os.path.dirname(cur))
from common.functions import Softmax, sigmoid

x = np.array([0.1, 0.5])
print(sigmoid(x))

# 加载数据


