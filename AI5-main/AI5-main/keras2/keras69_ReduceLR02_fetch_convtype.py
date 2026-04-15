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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
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

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, 
                   verbose=1, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', mode = 'auto', 
                        patience=25, verbose=1, factor=0.8)#running rate * factor)


from tensorflow.keras.optimizers import Adam
learning_rate = 0.05 #디폴트 0.001 0.005, 0.0001
#learning late default 0.001
#learning late default 0.00c 1
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate))
#Adam = 로스를 쳐줄여줌
model.fit(x_train, y_train,
          validation_split=0.2, epochs=100,
          batch_size=1024, callbacks=[es, rlr])

#4.평가예측
loss = model.evaluate(x_test, y_test, verbose=0)


y_predict = model.predict(x_test, verbose=0)
r2 = r2_score(y_test, y_predict)
print('lr : {0}, 로스 : {1}'.format(learning_rate, loss))
print('lr : {0}, r2 : {1}'.format(learning_rate, r2))

'''
lr : 0.001, 로스 : 0.050418782979249954
lr : 0.001, r2 : 0.3464441271283408
'''