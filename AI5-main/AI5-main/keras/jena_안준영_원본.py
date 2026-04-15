#https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016

#y 는 T(degC) 로 잡아라.
#자르는 거 맘대로

#predict해야할 부분
#31.12.2016 00:10:00 ~ 01.01.2017 00:00:00 까지 맞춰라 y 144개
#None , 144
#y의 shape는 (n,144)

#url : https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data?select=test.csv
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
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

path = ".\\_data\\kaggle\\jena\\"

a3 = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
b4 = a3.tail(144)
b5 = b4['T (degC)']
a = a3.head(420407)


a1 = a.drop(['T (degC)'], axis=1)
a2 = a['T (degC)']

def split_x(dataset, size) :
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
        
    return np.array(aaa)
size = 144
x = split_x(a1, size)
y = split_x(a2, size)

x= np.delete(x, -1 , axis = 0)
# print("x변환 :", x[-1])
print(x.shape) #(420263, 144, 13)

y= np.delete(y, 0 , axis = 0)
# print("y변환 : ",y[0])
print(y.shape) #(420263, 144)

b = a.tail(144)

b1 = b.drop(['T (degC)'], axis=1)
b2 = b['T (degC)']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=3)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape) 



#2. 모델구성
model = Sequential()
model.add(LSTM(units=64, input_shape=(144,13), return_sequences=True)) # timesteps , features
model.add(LSTM(128, return_sequences=True)) # timesteps , features
model.add(LSTM(256))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(144))
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(
    monitor = 'loss',
    mode = 'min',
    verbose=1,
    patience=20,
    restore_best_weights=True
)
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = '.\\_save\\keras55\\'
filename = 'jena_안준영.hdf5'
filepath = ''.join([path1, filename])


mcp = ModelCheckpoint(
    monitor= 'val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only= True,
    filepath = filepath
)
model.fit(x,y, epochs=200, validation_split=0.2, batch_size=512, callbacks=[es,mcp])
#4. 평가, 예측
results = model.evaluate(x,y, batch_size=512)
print('loss : ', results)

x_pred = np.array([b1]).reshape(1,144,13) #[[[8]]]
#벡터형태 데이터 (3,) -> (1,3,1)
#스칼라는 행렬아님

y_pred = model.predict(x_pred, batch_size=512)
y_pred = np.array([y_pred]).reshape(144,1)
print(' 결과', y_pred)
print(y_pred.shape)

from sklearn.metrics import r2_score, mean_squared_error
# r2 = r2_score(y_test, y_pred)
# print("r2스코어 : ", r2)

def RMSE(y_test, y_pred) : 
    return np.sqrt(mean_squared_error(y_test, y_pred))
rmse = RMSE(b5, y_pred)
#y_predict = 매개변수

print("rmse = ", rmse)

print("rmse = ", rmse)
sample_submission_csv = pd.read_csv(path + "sample_submission_jena.csv", index_col=0)
sample_submission_csv['T (degC)'] = y_pred
sample_submission_csv.to_csv(path1 + "jena9.csv")