import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, MaxPooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, MaxPool2D

# GPU 메모리 설정 조정
import tensorflow as tf


start = time.time()

np_path = 'C:\\프로그램\\ai5\\_data\\image\\brain\\save_npy\\'

x_train = np.load(np_path + "keras45_01_x_train.npy")
y_train = np.load(np_path + "keras45_01_y_train.npy")
x_test = np.load(np_path + "keras45_01_x_test.npy")
y_test = np.load(np_path + "keras45_01_y_test.npy")
print(x_train)
print(x_train.shape)

model_path = 'c:/프로그램/ai5/_save/keras45/brain/k45_0805_1313_0017-0.6137.hdf5'
model = load_model(model_path)

"""
#2. 모델 구성
input1 = Input(shape=(200,200,1))
dense1 = Conv2D(32, (2,2), padding ='same', activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense4 = Conv2D(32, (2,2), padding='same', activation='relu')(drop1)
maxp1 = MaxPooling2D()(dense4)
dense6 = Conv2D(32, (2,2), padding='same', activation='relu')(maxp1)
Flat1 = Flatten()(dense6)
dense7 = Dense(32, activation='relu')(Flat1)
output1 = Dense(1, activation='sigmoid')(dense7)
model = Model(inputs = input1, outputs = output1)
"""

#3. 컴파일 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    verbose=1,
    patience=20,
    restore_best_weights=True
)

import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

path1 = './_save/keras45/brain/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path1, 'k45_', date, '_', filename])
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=10,
                   restore_best_weights=True)

mcp = ModelCheckpoint(
    monitor= 'val_loss',
    mode = 'auto',
    verbose=1,
    save_best_only= True,
    filepath = filepath
)


model.fit(x_train, y_train, epochs = 100, batch_size = 160, verbose=1, validation_split=0.2, callbacks=[es, mcp],)

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

#y_test = np.argmax(y_test, axis=1)
#y_predict = np.argmax(y_predict, axis=1)

acc = accuracy_score(y_test, np.round(y_predict))

print('로스 : ', loss[0])
print('acc : ', acc)