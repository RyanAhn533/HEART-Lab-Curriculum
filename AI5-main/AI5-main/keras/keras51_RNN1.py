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

#SimpleRNN = Vanilla RNN

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])
x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              ])

y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)

#x = x.reshape(7,3,1)
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)
#3-D tensor with shape(batch_size, timesteps, features).

#2. 모델구성
model = Sequential()
#model.add(SimpleRNN(8, input_shape=(3,1))) #3은 time steps, 1은 features
#model.add(LSTM(8, input_shape=(3,1))) #3은 time steps, 1은 features
#[8,9,10]의 결과 [[9.770123]
model.add(GRU(8, input_shape=(3,1))) #3은 time steps, 1은 features
#[8,9,10]의 결과 [[9.770123]


#행이 데이터의 개수이므로 나머지가 shape
#행무시 열우선 to input_shape
model.add(Dense(8))
model.add(Dense(8))
model.add(Dense(8))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(2))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    verbose=1,
    patience=200,
    restore_best_weights=True
)

model.fit(x, y, epochs = 1000, batch_size = 1, 
          verbose=1, validation_split=0.2, callbacks=[es],)

#4. 평가, 예측
results = model.evaluate(x,y)
print('loss : ', results)

x_pred = np.array([8,9,10]).reshape(1,3,1) #[[[8]]]
#벡터형태 데이터 (3,) -> (1,3,1)
#스칼라는 행렬아님
y_pred = model.predict(x_pred)
print('[8,9,10]의 결과', y_pred)