
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar100
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, MaxPooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from tensorflow.keras.utils import to_categorical
import time
#1 데이터

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print(np.unique(y_train, return_counts=True))

x_train = x_train/255.
x_test = x_test/255.

ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#2 모델 구성
input1 = Input(shape=(32,32,3))
dense1 = Conv2D(64, (2,2), padding ='same', activation='relu')(input1)
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
output1 = Dense(100, activation='softmax')(dense8)
model = Model(inputs = input1, outputs = output1)


#3 컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = './_save/keras37/_cifar100/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path1, 'k30_', date, '_', filename])

es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=30,
    verbose=1,
    restore_best_weights=True
)

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    save_best_only = True,
    verbose=1,
    filepath=filepath
)

model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1, validation_split=0.25, callbacks=[es, mcp])

# 평가예측
loss= model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1) 
y_test = np.argmax(y_test, axis=1).reshape(-1, 1)

acc = accuracy_score(y_test, y_predict)

print('로스 값 : ', loss[0])
print('acc 스코어 : ', round(loss[1], 3))
print('y_predict[9999] 값 : ', y_predict[9999])

# 로스 값 :  3.0715372562408447
# acc 스코어 :  0.231
# y_predict[9999] 값 :  [70]