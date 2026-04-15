#Conv2D로 시작해서 중간에 LSTM을 넣어서 모델 구성
from __future__ import absolute_import, division, print_function, unicode_literals
from __future__ import annotations
####################
import numpy as np
import tensorflow
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, Input, Conv2D, Flatten, Reshape, MaxPooling2D, Bidirectional, LSTM
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


print(np.max(x_train)) #1.0
print(np.min(x_train)) #0.0


print(x_train.shape, y_train.shape) #(60000, 28, 28) (10000,) / 가로 28 세로 28 짜리가 6만장 / 10000 <- 이게 1이면 흑백
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)



#2. 모델 구성 
model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(28, 28, 1), 
                 strides=1,
                 padding='same'))
model.add(Reshape(target_shape=(28*28,64))) 
model.add(LSTM(62))
model.add(Dense(10, activation='softmax')) #y는 60000,10 으로 onehot encoding해야한다

model.summary()




#3 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    verbose = 1,
    restore_best_weights=True
)

model.fit(x_train, y_train, epochs = 100, batch_size = 1024, verbose=1, validation_split=0.2, callbacks=[es],)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, accuracy_score

# y_test = y_test.to_numpy()
# y_predict = y_predict.to_numpy()

y_test = np.argmax(y_test, axis=1).reshape(-1,1)
y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)

acc = accuracy_score(y_test, y_predict)

print('로스 : ', loss)
print('acc_score :', acc)
print(y_predict)
print(y_predict.shape)
print(y_predict[0])