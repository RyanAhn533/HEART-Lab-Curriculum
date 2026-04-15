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
datasets = np.array([1,2,3,4,5,6,7,8,9,10]) #input_lngth=3
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
model.add(SimpleRNN(10, input_shape=(3,1))) #3은 time steps, 1은 features
#(unit * unit) + (unit*feature) + (unit*bias)
#파라미터 갯수 = units * (units + bias + feature)
#[8,9,10]의 결과 [[9.770123]
#행이 데이터의 개수이므로 나머지가 shape
#행무시 열우선 to input_shape
model.add(Dense(7))

model.add(Dense(1))
model.summary()

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 10)                120

#  dense (Dense)               (None, 7)                 77

#  dense_1 (Dense)             (None, 1)                 8

# =================================================================