from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
import sklearn as sk
from sklearn.datasets import fetch_covtype
import numpy as np
import time as t
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping
##############
import warnings
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import tensorflow as tf
import random as rn
from tensorflow.keras.optimizers import Adam    
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
rn.seed(337)
tf.random.set_seed(337)
np.random.seed(337)
###############


datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(y)
print(y.shape)
y = pd.get_dummies(y)


from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

print(x.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=337)


from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler



from sklearn.preprocessing import MinMaxScaler, StandardScaler,MaxAbsScaler, RobustScaler
#scaler = MaxAbsScaler()
#scaler = StandardScaler()

#scaler = MaxAbsScaler() 0.61 r2 0.49
#scaler = RobustScaler() r2 0.304  loss 0.91951
#모델
model = Sequential()
model.add (Dense(16))
model.add (Dense(8, activation='relu'))
model.add (Dense(7, activation='softmax'))

for i in range(6) : 
    lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    lr = lr[i]
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['acc'])
    

    es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10, verbose=0, 
                   restore_best_weights=True)

    ##### mcp 세이브 파일명 만들기 #####
    import datetime
    date = datetime.datetime.now()
    date = date.strftime("%m%d_%H%M")

    path = 'C:\\프로그램\\ai5\\_save\\ml04\\'
    filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
    filepath = "".join([path, 'ml04_', str(i+1), '_', date, '_', filename])  

    mcp = ModelCheckpoint(
            monitor='val_loss',
            mode='auto',
            verbose=0,     
            save_best_only=True,   
            filepath=filepath,)
    import time

    start = time.time()
    hist = model.fit(x_train, y_train, epochs=200, batch_size=2048,
            verbose=0, 
            validation_split=0.2,
            callbacks=[es, mcp],
            )
    end = time.time()

#4. 평가,예측
    print("=================1. 기본출력 ========================")
    loss = model.evaluate(x_test, y_test, verbose=0)
    print('lr : {0}, 로스 :{1}'.format(lr, loss))

    y_predict = model.predict(x_test, verbose=0)
    r2 = r2_score(y_test, y_predict)
    print('lr : {0}, r2 : {1}'.format(lr, r2))



'''
=================1. 기본출력 ========================
lr : 0.1, 로스 :[0.617184042930603, 0.7304995656013489]
lr : 0.1, r2 : 0.34566197652507885
=================1. 기본출력 ========================
lr : 0.01, 로스 :[0.5786868929862976, 0.7467184066772461]
lr : 0.01, r2 : 0.4243229846500399
=================1. 기본출력 ========================
lr : 0.005, 로스 :[0.5760565400123596, 0.7489501237869263]
lr : 0.005, r2 : 0.4316181103182883
=================1. 기본출력 ========================
lr : 0.001, 로스 :[0.5745171904563904, 0.7491164803504944]
lr : 0.001, r2 : 0.435181546515716
=================1. 기본출력 ========================
lr : 0.0005, 로스 :[0.5743551254272461, 0.7491222023963928]
lr : 0.0005, r2 : 0.4351141973454409
=================1. 기본출력 ========================
lr : 0.0001, 로스 :[0.5742121338844299, 0.7495582699775696]
lr : 0.0001, r2 : 0.43573519254665427
'''