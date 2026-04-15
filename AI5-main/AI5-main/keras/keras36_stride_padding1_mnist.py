#35_2에서 가져옴
# x_train, x_test는 reshape
# y_tset, y_train OneHotEncoding

import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data() #알아서 데이터 나눠줌
# print(x_train)

##### 스케일링 1-1
x_train = x_train/255.
x_test = x_test/255.

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape)

###원핫 
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

##원핫 1-1
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(sparse=False)
# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)
# y_train = ohe.fit_transform(y_train)
# y_test = ohe.transform(y_test)

###원핫 1-2
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

y_test = y_test.to_numpy()

# print(np.max(x_train), np.min(x_train)) #1.0 0.0
# print(y_train.shape)


###### 스케일링 1-2 
# x_train = (x_train - 127.5) / 127.5
# x_test = (x_test - 127.5) / 127.5
# print(np.max(x_test), np.min(x_test)) # 1.0 -1.0

# ##### 스케일링 2. MinMaxScaler(), StandardScaler
# x_train = x_train.reshape(60000, 28*28)
# x_test = x_test.reshape(10000, 28*28)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.max(x_train), np.min(x_train))




# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)
# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)
 

# print(x_train[0])
# print("y_train[0] : ", y_train[0])

# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,) #색상 값이 숨겨져있다. 사실은 60000, 28, 28, 1
# print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,) 


#2 모델구성

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(28, 28, 1), 
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
model.add(Dense(10, activation='softmax')) #y는 60000,10 으로 onehot encoding해야한다




model.summary() #summary가 됨. 


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

#################mcp 세이브 파일명 만들기 시작##################

import datetime 
date = datetime.datetime.now() #데이트라는 변수에 현재 시간을 반환한다.
print(date) #2024-07-26 16:49:51.174797
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M") #시간을 문자열로 바꿔줌
print(date) #0726_1654
print(type(date))


path = './_save/keras36/_padding1_mnist/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' # '1000-0.7777.hdf5'  #fit에서 반환되는 값을 빼오는 것이다. 
filepath = "".join([path, 'k35_04', date, '_', filename])

############mcp세이브 파일명 만들기 끝################

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 1, #가장 좋은 지점을 알려줄 수 있게 출력함
    save_best_only=True,
    filepath = filepath
)
# 생성 예" './_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5'


start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=1, validation_split = 0.25, callbacks=[es, mcp])
end = time.time()

# model.save('./_save/keras29_mcp/keras29_3_save_model.h5')

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

# padding
# 로스 :  [0.2835237383842468, 0.9211999773979187]
# acc_score : 0.9212