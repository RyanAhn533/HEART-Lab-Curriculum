# jena를 Dnn으로 구성
# x : (42만, 144, 144) -> 42만, (144*144)
# y " (42만, 144)"

#https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016

from tensorflow.keras.models import Sequential, load_model
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
x = x.reshape(420263,144*13)
b = a.tail(144)

b1 = b.drop(['T (degC)'], axis=1)
b2 = b['T (degC)']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=5656)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 



#2. 모델구성
model = Sequential()
model.add(Dense(64, input_shape=(144*13,))) # timesteps , features
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(144))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(
    monitor = 'loss',
    mode = 'min',
    verbose=1,
    patience=30,
    restore_best_weights=True
)
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = '.\\_save\\keras55\\'
filename = 'jena_안준영.hdf5'
filepath = ''.join([path1, 'save_model', filename])


mcp = ModelCheckpoint(
    monitor= 'val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only= True,
    filepath = filepath
)
model.fit(x,y, epochs=1000, validation_split=0.2, batch_size=1024, callbacks=[es,mcp])


#4. 평가, 예측
results = model.evaluate(x,y, batch_size=1024)
print('loss : ', results)

x_pred = np.array([b1]).reshape(1,144*13) #[[[8]]]
#벡터형태 데이터 (3,) -> (1,3,1)
#스칼라는 행렬아님

y_pred = model.predict(x_pred, batch_size=1024)
y_pred = np.array([y_pred]).reshape(144,1)
print(' 결과', y_pred)
print(y_pred.shape)

from sklearn.metrics import r2_score, mean_squared_error
# r2 = r2_score(y_test, y_pred)
# print("r2스코어 : ", r2)

def RMSE(y_test, y_pred) : 
    return np.sqrt(mean_squared_error(y_test, y_pred))
rmse = RMSE(b5, y_pred)

print("rmse = ", rmse)

#(144, 1)
#rmse =  2.7745453146667893

#rmse = 1.711
#rmse =  1.6803965965776588

#(144, 1)
#rmse =  1.4031121056311942