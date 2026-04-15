#loc , iloc 차이와 사용하는 거 깜지 한장 써서 제출
# int loatation index lotation 이야
# 1. 첫번째 카럶

#판다스는 왜 iloc로 datasets.iloc[-144, 1] 로 해야하는가?

학생csv = 'jena_배누리.csv'

path1 = 'C:\\ai5\\_data\\kaggle\\jena\\'
path2 = 'C:\\ai5\\_save\\keras55\\'

import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


datasets = pd.read_csv(path1 +'jena-climate_2009_2016.csv')

y_정답 = datasets.iloc[-144, 1]
print(y_정답)
print(y_정답.shape)

학생꺼 = pd.read_csv(path2 + 학생csv, index_col=0)
print(학생꺼)

print(y_정답[:5])
print(학생꺼[:5])

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', RMSE)