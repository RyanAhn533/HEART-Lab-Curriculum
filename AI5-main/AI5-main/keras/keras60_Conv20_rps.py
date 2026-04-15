#3차원으로 바꿔서 LSTM으로 바꾸
#배치를 100으로 잡고
#x, y를 추출해서 모델을 맹그러봐
#acc 0.99이상

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten,LSTM, Dropout, Input, MaxPooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

train_datagen = ImageDataGenerator(
    rescale=1/255,

horizontal_flip=True, #수평 뒤집기
    vertical_flip=True, #수직뒤집기
    width_shift_range=0.1, #평행이동
    height_shift_range=0.1, #평행이동 수직
    rotation_range=5, #정해진 각도만큼 이미지 회전
    zoom_range=1.2, #축소 또는 확대
    shear_range=0.7, # 좌표
    fill_mode="nearest", #비율에 맞춰서 채워라
)


np_path = 'C:/프로그램/ai5/_data/image/rps/'

x_train = np.load(np_path + "keras45_03_x_train.npy")
y_train = np.load(np_path + "keras45_03_y_train.npy")
x_test = np.load(np_path + "keras45_03_x_test.npy")
y_test = np.load(np_path + "keras45_03_y_test.npy")
print(x_train.shape)
print(x_test.shape)
#(2520, 100, 100, 3)
#(2520, 100, 100, 3)

x_train = x_train.reshape(2520,100*100,3)
x_test = x_test.reshape(2520,100*100,3)

x_train, x_test,y_train, y_test = train_test_split(x_train, x_test, train_size=0.8, random_state=3) 
#np.unique로 y확인
#(756, 100, 100, 3)
#(1764, 100, 100, 3)
"""
ohe = OneHotEncoder(sparse=True)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)
"""
ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(10, kernel_size=2, input_shape=(100*100, 3))) # timesteps , features
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
"""
#2. 모델 구성
input1 = Input(shape=(100*100,3))
dense1 = Conv1D(128, (2,2), padding ='same', activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Conv1D(64, (2,2), padding='same', activation='relu')(drop1)
maxp1 = MaxPooling2D()(dense2)
dense3 = Conv1D(32, (2,2), padding='same', activation='relu')(maxp1)
dense4 = Conv1D(32, (2,2), padding='same', activation='relu')(dense3)
Flat1 = Flatten()(dense4)
dense5 = Dense(32, activation='relu')(Flat1)
output1 = Dense(3, activation='softmax')(dense5)
model = Model(inputs = input1, outputs = output1)
"""
#3. 컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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

path1 = './_save/keras36/_cifa10/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path1, 'k30_', date, '_', filename])
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


model.fit(x_train, y_train, epochs = 10, batch_size = 16, verbose=1, validation_split=0.2, callbacks=[es, mcp],)

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

#y_test = np.argmax(y_test, axis=1)
#y_predict = np.argmax(y_predict, axis=1)

acc = accuracy_score(y_test, np.round(y_predict))

print('로스 : ', loss[0])
print('acc : ', acc)
#로스 :  0.5409650802612305
#acc :  0.7265
#로스 :  1.0989115238189697/