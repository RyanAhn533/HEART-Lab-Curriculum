import numpy as np
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, BatchNormalization
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split

#1. Data
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],
             [7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],
             [30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=3)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional


#2. 모델구성
model = Sequential()
#model.add(SimpleRNN(8, input_shape=(3,1))) #3은 time steps, 1은 features
#model.add(LSTM(8, input_shape=(3,1))) #3은 time steps, 1은 features
#[8,9,10]의 결과 [[9.770123]
model.add(LSTM(32, input_shape=(3,1), return_sequences=True)) #3은 time steps, 1은 features
model.add(LSTM(32,)) #LSTM 안쓰고 Flatten 써줘도 된다. -
#차원을 맞추려고 / 시계열 데이터라는 확신이 있을 때만 쓴다. LSTM 을 두번이상 때리면 속도가 느려져서 잘 쓰지는 않음
#왜 두번하냐? -> 시계열 데이터라는 값을 지니고서 고도화를 시키고 싶을 때, 모델로 하이퍼 파라미터 개선을 계속 시킨 후에는 차원이 바뀜
#그래서 LSTM을 두번 넣어줌
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(
    monitor = 'loss',
    mode = 'min',
    verbose=1,
    patience=100,
    restore_best_weights=True
)
import datetime
date = datetime.datetime.now()
print(date) #2024-07-26 16:49:57.565880
print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M")
print(date) #0726_1654
print(type(date)) #<class 'str'>


path = 'C:\\프로그램\\ai5\\_save\\keras52\\'
filename ='{epoch:04d}-{loss:.4f}.hdf5'   #1000-0.7777.hdf5
filepath = "".join([path, 'k52_02', date, '_' , filename])
#생성 예 : ./_save/keras29_mcp/k29_0726_1654_1000-0.7777.hdf5
################## mcp 세이브 파일명 만들기 끝 ################### 

mcp=ModelCheckpoint(
    monitor='loss',
    mode='auto',
    verbose = 1,
    save_best_only=True,
    filepath=filepath)

model.fit(x,y, epochs=1000, batch_size=7, callbacks=[es, mcp])
#4. 평가, 예측
results = model.evaluate(x,y)
print('loss : ', results)

x_pred = np.array([50,60,70]).reshape(1,3,1) #[[[8]]]
#벡터형태 데이터 (3,) -> (1,3,1)
#스칼라는 행렬아님
y_pred = model.predict(x_pred)
print('[50,60,70]의 결과', y_pred)

#loss :  3.451554675848456e-06
#[50,60,70]의 결과 [[77.29907]]
#[50,60,70]의 결과 [[76.31709]]
#loss :  0.0006053115939721465
#[50,60,70]의 결과 [[77.52706]]

#[50,60,70]의 결과 [[78.45667]]
#PS C:\프로그램\ai5> 