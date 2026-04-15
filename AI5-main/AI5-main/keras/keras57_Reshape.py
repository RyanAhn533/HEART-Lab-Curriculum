#Reshape 랑 flatten 이랑 친구 정도
# -> 연산량 변화 없음
from __future__ import absolute_import, division, print_function, unicode_literals
from __future__ import annotations
####################
import numpy as np
import tensorflow
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping
##############
import warnings
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
###############
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#reshape를 다시 또 해야함


y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

"""
y_train = to_categorical(y_train)
x_train = to_categorical(x_train)
"""

print(np.max(x_train)) #1.0
print(np.min(x_train)) #0.0


print(x_train.shape, y_train.shape) #(60000, 28, 28) (10000,) / 가로 28 세로 28 짜리가 6만장 / 10000 <- 이게 1이면 흑백
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)


#데이터를 4차원으로 바꿔줘야함

#2. 모델
model = Sequential()
model.add(Dense(100, input_shape = (28,28))) #(n, 28, 28)
model.add(Reshape(target_shape=(28,28,100))) # (28,50,1)
model.add(Flatten())  # 23* 23* 32
#PFM
model.add(Dense(units=32))
model.add(Dense(units=16, input_shape=(32,)))
#shape = (batch_size, input_dim)
model.add(Dense(10, activation='softmax'))
model.summary()

# #3. 컴파일, 훈련
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# es= EarlyStopping(monitor='val_loss', mode = 'min', patience=20, restore_best_weights=True)
# model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=1, validation_split=0.2, callbacks=[es])

# #4 평가 예측

# test_loss, test_acc = model.evaluate(x_train, y_train, verbose=1)

# print("테스트 정확도 : %d" %(test_acc*100))