from sklearn.datasets import load_diabetes
from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM
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
import warnings
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

dataset = load_diabetes()
print(dataset)
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target


scaler = StandardScaler()
x = scaler.fit_transform(x)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    shuffle=True, random_state=3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler


#모델
model = Sequential()
model.add(LSTM(10, input_shape=(10, 1))) # timesteps , features
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(optimizer='adam',
              loss='mse', metrics=['acc'])
import time
start = time.time()

es= EarlyStopping(monitor='val_loss', mode = 'min', patience=20, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=1, validation_split=0.2, callbacks=[es])
end = time.time()
#4 평가 예측
#평가예측
loss = model.evaluate(x_test, y_test, verbose=1)
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('로스값은 : ', loss)
print('y값은? ', y_pred)
print('r2값은?', r2)
y_pred = np.round(y_pred)
from sklearn.metrics import r2_score, accuracy_score
accuracy_score = accuracy_score(y_test, y_pred) 
y_pred = np.round(y_pred) 
print('acc_score :', accuracy_score)
print("걸린 시간 :", round(end-start,2),'초')
loss = model.evaluate(x_test, y_test)
print("로스는 ? :", loss)
#전 : 로스 :  [4228.58740234375, 0.0]
#후 : [4272.41943359375, 0.0]
#좋게 변화 x