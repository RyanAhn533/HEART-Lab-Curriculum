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


a = np.array(range(1,101))
#x_predict = np.array(range(96, 106)) #101부터 107 찾아라


def split_x(dataset, size) :
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
        
    return np.array(aaa)
size = 11

bbb1 = split_x(a, size)

print(bbb1)
print(bbb1.shape)

x = bbb1[:,:-1]
y = bbb1[:, -1]

print(x.shape)
print(y.shape)
#(90, 10)
#(90,)

x = x.reshape(90,5,2)

#2. 모델구성
model = Sequential()

model.add(LSTM(64, input_shape=(5,2), return_sequences=True)) #3은 time steps, 1은 features
model.add(LSTM(64,)) 
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Flatten()) 
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(
    monitor = 'loss',
    mode = 'min',
    verbose=1,
    patience=1000,
    restore_best_weights=True
)

model.fit(x,y, epochs=10000, batch_size=8, callbacks=[es])
#4. 평가, 예측
results = model.evaluate(x,y)
print('loss : ', results)

x_pred = np.array([97,98,99,100,101,102,103,104,105,106]).reshape(1,5,2) #[[[8]]]
#벡터형태 데이터 (3,) -> (1,3,1)
#스칼라는 행렬아님
y_pred = model.predict(x_pred)
print(' 결과', y_pred)
#91,92,93,94,95,96,97,98,99,100