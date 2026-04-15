#배치를 100으로 잡고
#x, y를 추출해서 모델을 맹그러봐
#acc 0.99이상

"""
batch_size=160
x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]
"""

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input, MaxPooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time

train_datagen = ImageDataGenerator(
    rescale=1/255,)
"""
    horizontal_flip=True, #수평 뒤집기
    vertical_flip=True, #수직뒤집기
    width_shift_range=0.1, #평행이동
    height_shift_range=0.1, #평행이동 수직
    rotation_range=5, #정해진 각도만큼 이미지 회전
    zoom_range=1.2, #축소 또는 확대
    shear_range=0.7, # 좌표
    fill_mode="nearest", #비율에 맞춰서 채워라
    
    )
"""
test_datagen = ImageDataGenerator(
    rescale=1./255,) #테스트 데이터는 절대 변환하지않는다. 데이터 조작.
path_train = 'C:/프로그램/ai5/_data/image/brain/train/'
path_test =   'C:/프로그램/ai5/_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_test, target_size=(200, 200), #(10,200,200,1)->16개가 나옴 (batchsize, targetsize, channel) 
    #found 160 - xy_train -> batchsize 10 -> 16개있음
    batch_size=160, 
#요 폴더에 있는걸 전부 수치화 해라
class_mode='binary',

color_mode='grayscale',
shuffle=True)
xy_test = test_datagen.flow_from_directory(
    path_test, target_size=(200, 200), #(10,200,200,1)->16개가 나옴 (batchsize, targetsize, channel) 
    #found 160 - xy_train -> batchsize 10 -> 16개있음
    batch_size=160, 
#요 폴더에 있는걸 전부 수치화 해라
class_mode='binary',
color_mode='grayscale',
shuffle=True)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

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


model.fit(x_train, y_train, epochs = 100, batch_size = 160, verbose=1, validation_split=0.2, callbacks=[es, mcp],)

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

#y_test = np.argmax(y_test, axis=1)
#y_predict = np.argmax(y_predict, axis=1)

acc = accuracy_score(y_test, np.round(y_predict))

print('로스 : ', loss[0])
print('acc : ', acc)