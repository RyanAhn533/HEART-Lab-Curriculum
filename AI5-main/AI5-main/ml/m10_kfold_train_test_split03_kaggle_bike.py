from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM
import sklearn as sk
import numpy as np
import time as t
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping
##############
import warnings
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,StratifiedKFold, KFold, cross_val_score, cross_val_predict

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, SVR
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
###############

path = 'C:\\프로그램\\ai5\\_data\\bike-sharing-demand\\'
# \a  \b 이런걸 하나의 문자로 인식함 줄바꿈 이런거
# # 위와같은 애들 \ -> \\로 바꿔줘야함 / // 도 가능
# path = 'C:/프로그램//ai5\_data\\bike-sharing-demand'

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path + "sampleSubmission.csv", index_col=0)

train_csv = train_csv.dropna() #train_csv 데이터에서 결측치 삭제
test_csv = test_csv.fillna(test_csv.mean()) #test_csv에는 결측치 평균으로 넣기

x = train_csv.drop(['count'], axis = 1)
y = train_csv[['count']] #, 'registered


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True,
                                                    random_state=123, train_size=0.8,
                                                   )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

n_splits=3

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=333) #, random_state=333
#kfold 할 준비 끗

#kfold = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=333)
model = SVR()

scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('ACC ', scores, '\n 평균 ACC', round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test)

print(y_predict)
print(y_test)

r2 = r2_score(y_test, y_predict)
print('cross_val_predict ACC : ', r2)

'''
ACC  [0.79155714 0.82300193 0.81383818] 
 평균 ACC 0.8095
   y = column_or_1d(y, warn=True)
[151.51481059  60.8392281   97.94905033 ... 176.52430174 217.82429448
  91.13152992]
                     count
datetime
2012-03-07 21:00:00    188
2012-04-12 03:00:00      5
2011-02-06 10:00:00     89
2011-12-05 23:00:00     62
2011-08-08 14:00:00    150
...                    ...
2012-02-03 15:00:00    216
2011-02-19 16:00:00    120
2012-03-16 09:00:00    312
2011-05-16 19:00:00    308
2011-10-06 04:00:00      5

[2178 rows x 1 columns]
cross_val_predict ACC :  0.4964013026800478
'''