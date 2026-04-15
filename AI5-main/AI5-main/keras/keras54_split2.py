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
a = np.array([[1,2,3,4,5,6,7,8,9,10], [9,8,7,6,5,4,3,2,1,0]]).reshape(10,2)

size = 5

"""
[[ 1  2]
 [ 3  4]
 [ 5  6]
 [ 7  8]
 [ 9 10]
 [ 9  8]
 [ 7  6]
 [ 5  4]
 [ 3  2]
 [ 1  0]]
 """

a = a.reshape(20,1)
a1 = a[:10,-1]
b1 = a[10:20, -1]
a = a1.reshape(10,1)
b = b1.reshape(10,1)

x = np.concatenate((a,b),axis=1)

def split_x(dataset, size) :
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
        
    return np.array(aaa)

bbb1 = split_x(x, size)

print(bbb1)
print(bbb1.shape)
x = bbb1[:,:-1]
y = bbb1[:, -1,0]
print(x.shape)
print(y.shape)
print(y)

"""
x1 = x[:,-1]
x2 = x[:,-2]
print(x1)
print(x2)
def split_x(dataset, size) :
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
        
    return np.array(aaa)

bbb1 = split_x(x1, size)

def split_x(dataset, size) :
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
        
    return np.array(aaa)

bbb2 = split_x(x2, size)

   

k = np.concatenate((bbb1,bbb2),axis=0)
print(k)
x = k[:,:-1]
y = k[:, -1]
print(x)
print(x.shape)
x = x.reshape(2,6,4)
y = y.reshape(2,6)
print(x)
print(y)
"""

#2. 모델구성
model = Sequential()
#model.add(SimpleRNN(8, input_shape=(3,1))) #3은 time steps, 1은 features
#model.add(LSTM(8, input_shape=(3,1))) #3은 time steps, 1은 features
#[8,9,10]의 결과 [[9.770123]
model.add(LSTM(32, input_shape=(4,2), return_sequences=True)) #3은 time steps, 1은 features
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

model.fit(x,y, epochs=1000, batch_size=7, callbacks=[es])
#4. 평가, 예측
results = model.evaluate(x,y)
print('loss : ', results)

x_pred = np.array([[7,3],[8,2],[9,1],[10,0]]).reshape(1,4,2) #[[[8]]]
#벡터형태 데이터 (3,) -> (1,3,1)
#스칼라는 행렬아님
y_pred = model.predict(x_pred)
print('[7,8,9,10]의 결과', y_pred)