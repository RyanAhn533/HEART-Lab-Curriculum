from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, LSTM
import sklearn as sk
from sklearn.datasets import load_boston
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
##############
import warnings
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
###############
import tensorflow as tf
import random as rn

rn.seed(337)
tf.random.set_seed(337)
np.random.seed(337)
path = "C:/프로그램/ai5/_data/kaggle/otto/"

train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
samplesubmission1_csv = pd.read_csv(path + "samplesubmission.csv", index_col=0)

print(train_csv.select_dtypes(include=['object']).columns)
print(test_csv.select_dtypes(include=['object']).columns)

train_csv.info()
test_csv.info()
print(train_csv['target'].value_counts())
train_csv['target'] = train_csv['target'].replace({'Class_1' : 1, 'Class_1' : 1, 'Class_2' : 2, 'Class_3' : 3, 'Class_4' : 4, 'Class_5' : 5, 'Class_6' : 6, 'Class_7' : 7, 'Class_8' : 8, 'Class_9' : 9, })



x = train_csv.drop(['target'], axis=1)
"""
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
"""
y = train_csv['target']

scaler = StandardScaler()

x = scaler.fit_transform(x)

x = x.reshape(61878,93,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=3, stratify=y)


from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler


model = Sequential()
model.add(Conv1D(filters=10, kernel_size=2, input_shape=(93, 1)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1))
model.summary()
from tensorflow.keras.optimizers import Adam

for i in range(6): 
    learning_rate = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    # learning_rate = 0.0007       # default = 0.001
    learning_rate = learning_rate[i]
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))

    model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=1, verbose=0,
          batch_size=512,
          )

#4. 평가,예측
    print("=================1. 기본출력 ========================")
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('lr : {0}, 로스 :{1}'.format(learning_rate, loss))

    y_predict = model.predict(x_test, verbose=0)
    r2 = r2_score(y_test, y_predict)
    print('lr : {0}, r2 : {1}'.format(learning_rate, r2))


'''
=================1. 기본출력 ========================
lr : 0.1, 로스 :324.6319274902344
lr : 0.1, r2 : -50.49880111646547
=================1. 기본출력 ========================
lr : 0.01, 로스 :6.303949356079102
lr : 0.01, r2 : -4.232356491007394e-05
=================1. 기본출력 ========================
lr : 0.005, 로스 :6.305667877197266
lr : 0.005, r2 : -0.00031400173978712864
=================1. 기본출력 ========================
lr : 0.001, 로스 :6.3038787841796875
lr : 0.001, r2 : -3.135488602157288e-05
=================1. 기본출력 ========================
lr : 0.0005, 로스 :6.30379581451416
lr : 0.0005, r2 : -1.807336534320747e-05
=================1. 기본출력 ========================
lr : 0.0001, 로스 :6.303757667541504
lr : 0.0001, r2 : -1.2494148068142863e-05

'''