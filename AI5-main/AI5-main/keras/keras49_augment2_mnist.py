#35_2에서 가져옴
# x_train, x_test는 reshape
# y_tset, y_train OneHotEncoding

import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Dropout, MaxPooling2D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import to_categorical
import time
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255.
x_test = x_test/255.

train_datagen =  ImageDataGenerator(
    #rescale=1./255,              # 이미지를 수치화 할 때 0~1 사이의 값으로 (스케일링 한 데이터로 사용)
    horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    # vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    width_shift_range=0.2,       # 평행이동  <- 데이터 증폭
    # height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    rotation_range=15,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,              # 축소 또는 확대
    # shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)
augment_size = 40000 # 똑같은 타일을 40000개 찍었다.
print(x_train.shape[0])
 
randidx = np.random.randint(x_train.shape[0], size = augment_size) #60000, size=40000
#randint -> 새로 생성할 데이터 4만개 생성댐
print(randidx)
print(np.min(randidx), np.max(randidx)) # 0 59997
x_augmented = x_train[randidx].copy()  #.copy 하면 메모리 안전빵!
y_augmented = y_train[randidx].copy()

x_augmented = x_augmented.reshape(
                                  x_augmented.shape[0], # 40000
                                  x_augmented.shape[1], #28
                                  x_augmented.shape[2], 1)#28
print(x_augmented.shape) # (40000 28 28 1)


print(x_augmented[0].shape) # (28 28 1)



print(x_augmented.shape, y_augmented.shape) #(40000, 28, 28) (40000,)
x_augmented = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
    shuffle=False,
).next()[0]
#왜 셔플하면 안되는가?
print(x_augmented[0].shape) #(28, 28, 1)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
print(x_train.shape, x_test.shape)


x_train = np.concatenate((x_train,x_augmented), axis = 0)
print(x_train.shape)
y_train = np.concatenate((y_train, y_augmented), axis = 0)
print(y_train.shape)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)


#2. 모델 구성 
input1 = Input(shape=(28,28,1))
dense1 = Conv2D(64, (3,3), padding='same', activation='relu')(input1)
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
output1 = Dense(10, activation='softmax')(dense8)
model = Model(inputs = input1, outputs = output1)
model.summary()
exit()
#3 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    verbose = 1,
    restore_best_weights=True
)

#################mcp 세이브 파일명 만들기 시작##################
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


model.fit(x_train, y_train, epochs = 100, batch_size = 64, verbose=1, validation_split=0.2, callbacks=[es, mcp],)

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

#y_test = np.argmax(y_test, axis=1)
#y_predict = np.argmax(y_predict, axis=1)

acc = accuracy_score(y_test, np.round(y_predict))

print('로스 : ', loss[0])
print('acc : ', acc)

#로스 :  0.006215902976691723
#acc :  0.9905