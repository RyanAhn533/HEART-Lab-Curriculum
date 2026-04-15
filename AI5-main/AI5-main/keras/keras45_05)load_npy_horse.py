#배치를 100으로 잡고
#x, y를 추출해서 모델을 맹그러봐
#acc 0.99이상

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


train_datagen = ImageDataGenerator(
    rescale=1./255,)
path_train = 'C:/프로그램/ai5/_data/image/horse_human/'


xy_train = train_datagen.flow_from_directory(
    path_train, target_size=(100, 100), 
    batch_size=30000, 
class_mode='binary',

color_mode='rgb',
shuffle=True)


x_train,x_test, y_train, y_test = train_test_split(xy_train[0][0], xy_train[0][1], train_size=0.7, random_state=3)


np_path = 'C:/프로그램/ai5/_data/image/horse_human/save_npy/'
x_train = np.load(np_path + "keras45_02_x_train1.npy")
y_train = np.load(np_path + "keras45_02_y_train1.npy")
x_test = np.load(np_path + "keras45_02_x_test.npy")
y_test = np.load(np_path + "keras45_02_y_test.npy")

model_path = 'c:/프로그램/ai5/_save/keras45/brain/k45_0805_1317_0018-0.0279.hdf5'
model = load_model(model_path)

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

"""
#2. 모델 구성
input1 = Input(shape=(100,100,3))
dense1 = Conv2D(128, (2,2), padding ='same', activation='relu')(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Conv2D(64, (2,2), padding='same', activation='relu')(drop1)
maxp1 = MaxPooling2D()(dense2)
dense3 = Conv2D(32, (2,2), padding='same', activation='relu')(maxp1)
dense4 = Conv2D(32, (2,2), padding='same', activation='relu')(dense3)
Flat1 = Flatten()(dense4)
dense5 = Dense(32, activation='relu')(Flat1)
output1 = Dense(1, activation='sigmoid')(dense5)
model = Model(inputs = input1, outputs = output1)


#3. 컴파일 훈련


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

path1 = './_save/keras45/horse/'
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
"""

model.fit(x_train, y_train, epochs = 100, batch_size = 64, verbose=1, validation_split=0.2, callbacks=[es, mcp],)

#평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

#y_test = np.argmax(y_test, axis=1)
#y_predict = np.argmax(y_predict, axis=1)

acc = accuracy_score(y_test, np.round(y_predict))

print('로스 : ', loss[0])
print('acc : ', acc)
