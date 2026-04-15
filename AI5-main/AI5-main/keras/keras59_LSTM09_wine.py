from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM
import sklearn as sk
from sklearn.datasets import load_wine
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
###############
#1.데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

scaler = StandardScaler()
x = scaler.fit_transform(x)

print(x.shape)
print(y.shape)

# x = x.reshape(178,13,1,1)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=138, shuffle=True)
from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler

"""
#모델
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(13,1,1),
                 strides=1,
                 padding='same')) 
model.add(Conv2D(filters=64, kernel_size=(2,2),
                 strides=1,
                 padding='same')) 
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2),
                 strides=1,
                 padding='same'))
model.add(Flatten()) 
model.add(Dense(units=32))
model.add(Dropout(0.2)) 
model.add(Dense(units=16, input_shape=(32,))) 
model.add(Dense(1, activation='softmax'))

"""
model = Sequential()
model.add(LSTM(10, input_shape=(13, 1))) # timesteps , features
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1))

#컴파일 훈련
model.compile(
    loss='mse',
    optimizer='adam')
es= EarlyStopping(monitor='val_loss', mode = 'min', patience=20,
                  restore_best_weights=True)

model.fit(x_train, y_train, epochs=200, batch_size=128,
          verbose=1, validation_split=0.2, callbacks=[es])

#평가예측
loss = model.evaluate(x_test, y_test)
print("로스는 ?", loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2스코어는? ", r2)

#로스는 ? 0.5925925970077515
# r2스코어는?  -0.038461538461538325

# 로스는 ? 0.07431574165821075
# r2스코어는?  0.869768817923948