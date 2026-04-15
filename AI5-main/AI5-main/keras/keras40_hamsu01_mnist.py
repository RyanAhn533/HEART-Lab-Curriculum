#35_2에서 가져옴
# x_train, x_test는 reshape
# y_tset, y_train OneHotEncoding

import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, MaxPooling2D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data() #알아서 데이터 나눠줌
print(x_train[0])

##### 스케일링 1-1
x_train = x_train/255.
x_test = x_test/255.

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

y_test = y_test.to_numpy()
y_train = y_train.to_numpy()
print(type(x_train))
print(type(y_train))
"""
#2 모델구성

input1 = Input(shape=(28,28,1))
dense1 = Dense(32, activation='relu', name='ys1')(input1)
dropout1 = Dropout(0.2)(dense1)
dense2 = Dense(16, activation='relu', name='ys2')(dropout1)
dropout2 = Dropout(0.2)(dense2)
dense3 = Dense(8, activation='relu', name='ys3')(dropout2)
dense4 = Dense(4, activation='relu', name='ys4')(dense3)
output1 = Dense(1, name='ys5')(dense4)

model = Model(inputs=input1, outputs=output1)
"""
#2. 모델 구성 
input1 = Input(shape=(28,28,1))
dense1 = Conv2D(64, (3,3), padding='same', activation='relu')(input1)
dense2 = Conv2D(64, (3,3), padding='same', activation='relu')(dense1)
dense3 = Conv2D(32, (2,2), padding='same', activation='relu')(dense2)
dense4 = Conv2D(32, (2,2), padding='same', activation='relu')(dense3)
drop1 = Dropout(0.2)(dense4)
dense5 = Conv2D(32, (2,2), padding='same', activation='relu')(drop1)
maxp1 = MaxPooling2D()(dense5)
dense6 = Conv2D(32, (2,2), padding='same', activation='relu')(maxp1)
maxp2 = MaxPooling2D()(dense6)
Flat1 = Flatten()(maxp2)
dense7 = Dense(32, activation='relu')(Flat1)
drop2 = Dropout(0.2)(dense7)
dense8 = Dense(16, activation='relu')(drop2)
output1 = Dense(10, activation='softmax')(dense8)
model = Model(inputs = input1, outputs = output1)


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


path = './_save/keras37/_mnist/'
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

# MaxPooling2D
# 로스 :  [0.08223845809698105, 0.9740999937057495]
# acc_score : 0.9741