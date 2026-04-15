from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
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


from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler

scaler = StandardScaler()
x = scaler.fit_transform(x)
print(x.shape)
x = x.reshape(10886,5,2,1)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    shuffle=True, random_state=3)

#모델
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5,2,1), 
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

# 후 r2 0.9999 로스 0.31
#전 ㄱ2 0.9699 로스 0.11
#Standard  로스는 ? 7.754250526428223
#r2스코어는?  0.9997607093510704
