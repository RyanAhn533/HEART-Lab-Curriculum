from __future__ import absolute_import, division, print_function, unicode_literals
from __future__ import annotations
####################
import numpy as np
import tensorflow
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping
##############
import warnings
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
###############
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(60000,28,28,1)
# x_test = x_test.reshape(10000, 28, 28, 1)

##################스케일링1-1########################
"""
x_train = x_train/255.
y_test = y_test/255
"""
##################스케일링1-2########################
"""
x_train = (x_train - 127.5) / 127.5
x_test = (x_test -127.5) / 127.5
"""
##################스케일링2-1########################
x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28)
#reshape를 다시 또 해야함

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


print(np.max(x_train), np.min(x_train))
print(np.max(x_test), np.min(x_test))

#x_train = x_train.reshape(60000,28*28)
#x_test = x_test.reshape(10000,28*28)

####################원핫인코딩###################
"""
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
"""

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

"""
print(np.max(x_train)) #1.0
print(np.min(x_train)) #0.0


print(x_train.shape, y_train.shape) #(60000, 28, 28) (10000,) / 가로 28 세로 28 짜리가 6만장 / 10000 <- 이게 1이면 흑백
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)
"""

model = Sequential()
model.add(Dense(64, input_shape=(28*28,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))


#컴파일 훈련
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc'])
es= EarlyStopping(monitor='val_loss', mode = 'min', patience=300,
                  restore_best_weights=True)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss', 
    mode='auto',
    verbose=1,
    save_best_only=True, filepath=("./_save/keras38/keras38_DNN.h1"))


model.fit(x_train, y_train, epochs=100, batch_size=128,
          verbose=1, validation_split=0.2, callbacks=[es,mcp])


#평가예측
loss = model.evaluate(x_test, y_test)


print("로스는 ?", loss)
y_predict = model.predict(x_test)
y_pred = np.round(y_predict)
accuracy_score = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_predict)
print("r2스코어는? ", r2)
print("acc_score : '",accuracy_score)