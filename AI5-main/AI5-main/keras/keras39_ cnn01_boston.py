from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
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
from tensorflow.keras.callbacks import EarlyStopping
##############
import warnings
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
###############


import time

dataset = load_boston()


x = dataset.data
y = dataset.target
"""
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
print(x)
"""
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# 데이터를 4D로 리쉐이프
x = x.reshape(506, 13, 1, 1)
y = y.reshape(506,1,1,1)

"""
############ dhekq##############
x = x.reshape(506, 13, 1, 1)

print(x.shape)
print(y.shape)

"""

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=3)
from sklearn.preprocessing import MinMaxScaler, StandardScaler



print(x_train)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))



model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(13, 1, 1), 
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

#3. 컴파일, 훈련
model.compile(optimizer='adam',
              loss='mse', metrics=['acc'])

es= EarlyStopping(monitor='val_loss', mode = 'min', patience=20, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=1, validation_split=0.2, callbacks=[es])

#4 평가 예측

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_test = np.argmax(y_test, axis=1).reshape(-1, 1)
y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)

acc = accuracy_score(y_test, y_predict)

print('로스 : ', loss[0])
print('acc : ', acc)