from sklearn.datasets import fetch_california_housing
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

dataset = fetch_california_housing()
print(dataset)
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

print(x.shape)
print(y.shape)

#(20640, 8)
#(20640,)
x = x.reshape(20640, 2, 2, 2)
y = y.reshape(20640,1,1,1)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    shuffle=True, random_state=3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler
#scaler = MaxAbsScaler()
#scaler = StandardScaler()

#scaler = MaxAbsScaler() 0.61 r2 0.49
#scaler = RobustScaler() r2 0.304  loss 0.91951
#모델
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(2,2,2), 
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
model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=1, validation_split=0.2, callbacks=[es])

#4 평가 예측

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_test = np.argmax(y_test, axis=1).reshape(-1, 1)
y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)

acc = accuracy_score(y_test, y_predict)

print('로스 : ', loss[0])
print('acc : ', acc)

#스케일링 전 0.2713
#스케일링 후 0.7406029396628281
"""

exit()
y_submit = model.predict(test_csv)
samplesubmission_csv['target'] = np.round(y_submit)/10
samplesubmission_csv.to_csv(path + "santafe_7.csv")


로스는 ? [0.2712891101837158, 0.0021802326664328575]
r2스코어는?  0.7948835567807272

로스는 ? [0.3928057849407196, 0.0021802326664328575]
r2스코어는?  0.7030072751830784
Minmax win
"""